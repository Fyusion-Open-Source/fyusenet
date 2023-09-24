//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Miscellaneous Layers Unit Tests
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cmath>
#include <fstream>
#include <memory>
#include <thread>

//-------------------------------------- Project  Headers ------------------------------------------

#include <gtest/gtest.h>
#include <fyusenet/fyusenet.h>
#include "gltesthelpers.h"
#include <fyusenet/base/layerfactory.h>
#include <fyusenet/gpu/argmaxlayerbuilder.h>
#include <fyusenet/gpu/deep/deepargmaxlayer.h>
#include <fyusenet/gpu/batchnormlayer.h>
#include <fyusenet/gpu/deep/deepbatchnormlayer.h>
#include <fyusenet/gpu/deep/deepgemmlayer.h>
#include "layertestbase.h"

//-------------------------------------- Global Variables ------------------------------------------

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

using namespace fyusion::fyusenet;
using namespace fyusion::fyusenet::gpu;

int main(int argc,char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    GLEnvironment::init();
    return RUN_ALL_TESTS();
}

class MiscLayerTest: public ::testing::Test, public TestContextManager, public LayerTestBase {
 protected:
    MiscLayerTest() : TestContextManager() {
    }

    void SetUp() override {
        // TODO (mw) only set up contexts once
        setupGLContext(4);
        fyusion::fyusenet::GfxContextManager::instance()->setupPBOPools(4, 4);
    }

    void TearDown() override {
        LayerTestBase::cleanup();
        tearDownGLContext();
    }
};


struct ArgMaxParam {
    ArgMaxParam(int w, int h, int c, float rmin=-100.f, float rmax=100.0f, float dmax=0.5f) :
        width(w), height(h), channels(c), range{rmin, rmax}, deltamax(dmax) {}

    int width;
    int height;
    int channels;
    float range[2];
    float deltamax;
};



class ArgMaxTest : public MiscLayerTest, public ::testing::WithParamInterface<ArgMaxParam> {
 protected:


    int * referenceArgMax(const float *input, const ArgMaxParam& param) const {
        int * result = new int[param.width * param.height];
        int stride = param.width;
        int chanstride = param.width*param.height;
        for (int y=0; y < param.height; y++) {
            for (int x=0; x < param.width; x++) {
                float mx = input[y*stride+x];
                int mix = 0;
                for (int c=1; c < param.channels; c++) {
                    if (input[c*chanstride+y*stride+x] > mx) {
                        mx = input[c*chanstride+y*stride+x];
                        mix = c;
                    }
                }
                result[y*stride+x] = mix;
            }
        }
        return result;
    }
};

struct BNParam {
    BNParam(int w, int h, int c) : width(w), height(h), channels(c) {
    }
    int width;
    int height;
    int channels;
};


class BatchNormTest : public MiscLayerTest, public ::testing::WithParamInterface<BNParam> {
 protected:


    float * referenceNorm(const float *input, const float *scaleBias, const BNParam& param) const {
        float * result = new float[param.width * param.height * param.channels];
        for (int c=0; c < param.channels; c++) {
            for (int i=0; i < param.width * param.height; i++) {
                result[c*param.width*param.height+i] = input[c*param.width*param.height+i]*scaleBias[c] + scaleBias[c+param.channels];
            }
        }
        return result;
    }

    float * generateScaleAndBias(int channels) const {
        float * result = new float[channels*2];
        for (int i=0; i< channels; i++) {
            result[i] = ((float)(std::rand()%1000)-500.f) / 250.f;
            result[i + channels] = ((float)(std::rand()%1000)-500.f) / 250.f;
        }
        return result;
    }
};


class GEMMTest : public MiscLayerTest {
 protected:
};

//-----------------------------------------------------------------------------
// Test Fixtures
//-----------------------------------------------------------------------------

TEST_P(ArgMaxTest, ArgMaxTestDeep) {
    auto param = GetParam();
    std::unique_ptr<float[]> input(generateRandomData(param.channels, param.width, param.height, param.range[0], param.range[1]));
    std::unique_ptr<int[]> ref(referenceArgMax(input.get(), param));
    gpu::ArgMaxLayerBuilder bld("argmax");
    bld.context(context()).shape(1, param.height, param.width, param.channels).deep();
    gpu::deep::DeepArgMaxLayer layer(bld, 1);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    layer.setup();
    layer.forward(1, nullptr);
    std::unique_ptr<float[]> result(new float[param.channels * param.width * param.height]);
    layer.copyResult(result.get(), false);
    layer.cleanup();
    int deviations = 0;
    int stride = param.width;
    int chanoffset = param.width*param.height;
    const int * refptr = ref.get();
    const float * inptr = input.get();
    const float * resptr = result.get();
    for (int y=0; y < param.height; y++) {
        for (int x=0; x < param.width; x++) {
            float refval = (float)refptr[y*stride+x];
            float gpuval = resptr[y*stride+x];
            if (std::abs(refval-gpuval) > 0.5) {
                int refidx = refptr[y*stride+x]*chanoffset+y*stride+x;
                int gpuidx = ((int)gpuval)*chanoffset+y*stride+x;
                if (std::abs(inptr[refidx]-inptr[gpuidx]) > param.deltamax) {
                    printf("(%d,%d): ref=%d gpu=%.1f  %f  %f  (%f)\n", x,y,refptr[y*stride+x],resptr[y*stride+x], inptr[refidx], inptr[gpuidx], std::abs(inptr[refidx]-inptr[gpuidx]));
                    deviations++;
                }
            }
        }
    }
    ASSERT_EQ(deviations, 0);
}



TEST_P(BatchNormTest, BNTestShallow) {
    class TestProvider : public ParameterProvider {
     public:
        TestProvider(const float *data) : wrapper_(data) {}
        DataBlob get(const std::string &name, int layerNo, int subIndex=0) const override {
            EXPECT_EQ(subIndex, 0);
            return DataBlob((DataWrapper *)&wrapper_);
        }
     private:
        DefaultDataWrapper<float> wrapper_;
    };
    auto param = GetParam();
    std::unique_ptr<float[]> scalebias(generateScaleAndBias(param.channels));
    std::unique_ptr<float[]> input(generateRandomData(param.channels, param.width, param.height, -10.f, 10.f));
    ASSERT_NE(scalebias, nullptr);
    ASSERT_NE(input, nullptr);
    TestProvider provider(scalebias.get());
    std::unique_ptr<float[]> ref(referenceNorm(input.get(), scalebias.get(), param));
    gpu::GPULayerBuilder bld("bnorm");
    bld.type(LayerType::BATCHNORM).context(context()).shape(param.channels, param.height, param.width, param.channels);
    gpu::BatchNormLayer layer(bld, 1);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    layer.loadParameters(&provider);
    layer.setup();
    layer.forward(1, nullptr);
    std::unique_ptr<float[]> result(new float[param.channels * param.width * param.height]);
    layer.copyResult(result.get(), false);
    layer.cleanup();
    const float * refptr = ref.get();
    const float * resptr = result.get();
    for (int i=0; i < param.width * param.height * param.channels; i++) {
        ASSERT_NEAR(refptr[i], resptr[i], 1e-1f);
    }
}


TEST_P(BatchNormTest, BNTestDeep) {
    class TestProvider : public ParameterProvider {
     public:
        TestProvider(const float *data) : wrapper_(data) {}
        DataBlob get(const std::string &name, int layerNo, int subIndex=0) const override {
            EXPECT_EQ(subIndex, 0);
            return DataBlob((DataWrapper *)&wrapper_);
        }
     private:
        DefaultDataWrapper<float> wrapper_;
    };
    auto param = GetParam();
    std::unique_ptr<float[]> scalebias(generateScaleAndBias(param.channels));
    std::unique_ptr<float[]> input(generateRandomData(param.channels, param.width, param.height, -10.f, 10.f));
    ASSERT_NE(scalebias, nullptr);
    ASSERT_NE(input, nullptr);
    TestProvider provider(scalebias.get());
    std::unique_ptr<float[]> ref(referenceNorm(input.get(), scalebias.get(), param));
    gpu::GPULayerBuilder bld("bnorm");
    bld.type(LayerType::BATCHNORM).context(context()).shape(param.channels, param.height, param.width, param.channels).deep();
    gpu::deep::DeepBatchNormLayer layer(bld, 1);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    layer.loadParameters(&provider);
    layer.setup();
    layer.forward(1, nullptr);
    std::unique_ptr<float[]> result(new float[param.channels * param.width * param.height]);
    layer.copyResult(result.get(), false);
    layer.cleanup();
    const float * refptr = ref.get();
    const float * resptr = result.get();
    for (int i=0; i < param.width * param.height * param.channels; i++) {
        ASSERT_NEAR(refptr[i], resptr[i], 1e-1f);
    }
}


TEST_F(GEMMTest, DeepGEMM) {
    int inchannels = 512;
    int outchannels = 256;
    std::unique_ptr<float[]> weights(new float[outchannels + inchannels*outchannels]);
    std::unique_ptr<float[]> input(generateConstantData(1.0f, inchannels, 1, 1, 0));
    memset(weights.get(), 0, (outchannels + inchannels*outchannels)*sizeof(float));
    for (int row=0; row < outchannels; row++) {
        float * rowptr = weights.get() + row * inchannels + outchannels;
        for (int col=0; col < inchannels; col++) {
            rowptr[col] = (col & 1) ? -1.f : 1.f;
        }
    }
    gpu::GPULayerBuilder bld("GEMM");
    bld.type(LayerType::GEMM).context(context()).shape(outchannels, 1, 1, inchannels).deep();
    gpu::deep::DeepGEMMLayer layer(bld, 1);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    SingleWeightProvider wsrc(weights.get() + outchannels, weights.get());
    layer.loadParameters(&wsrc);
    layer.setup();
    layer.forward(1, nullptr);
    std::unique_ptr<float[]> result(new float[outchannels]);
    layer.copyResult(result.get(), false);
    layer.cleanup();
    const float * resptr = result.get();
    for (int i=0; i < outchannels; i++) {
        ASSERT_NEAR(resptr[i], 0.0f, 1e-3f);
    }
}

// TODO (mw) more test patterns, maybe fuzz-testing with randomization

INSTANTIATE_TEST_CASE_P(ArgMax, ArgMaxTest, testing::Values(
                                                   ArgMaxParam(256, 128, 64),
                                                   ArgMaxParam(120, 80, 3),
                                                   ArgMaxParam(80, 40, 52),
                                                   ArgMaxParam(200, 200, 4),
                                                   ArgMaxParam(50, 50, 31),
                                                   ArgMaxParam(40, 40, 128)));

INSTANTIATE_TEST_CASE_P(BatchNorm, BatchNormTest, testing::Values(
                                                   BNParam(4,4,36),
                                                   BNParam(80, 40, 52),
                                                   BNParam(4, 4, 4),
                                                   BNParam(256, 128, 64),
                                                   BNParam(120, 80, 3),
                                                   BNParam(80, 40, 52),
                                                   BNParam(200, 200, 4),
                                                   BNParam(50, 50, 31),
                                                   BNParam(12, 12, 128)));

// vim: set expandtab ts=4 sw=4:
