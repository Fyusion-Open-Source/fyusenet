//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Pooling Layers Unit Tests
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>
#include <cmath>
#include <fstream>
#include <atomic>
#include <memory>
#include <thread>

//-------------------------------------- Project  Headers ------------------------------------------

#include <gtest/gtest.h>
#include <fyusenet/fyusenet.h>
#include "gltesthelpers.h"
#include <fyusenet/base/layerfactory.h>
#include "layertestbase.h"
#include <fyusenet/gpu/avgpoollayer.h>
#include <fyusenet/gpu/maxpoollayer.h>
#include <fyusenet/gpu/deep/deepavgpoollayer.h>
#include <fyusenet/gpu/deep/deepmaxpoollayer.h>
#include <fyusenet/gpu/deep/deepglobalpoollayer.h>


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

class PoolLayerTest: public ::testing::Test, public TestContextManager, public LayerTestBase {
 protected:
    PoolLayerTest() : TestContextManager() {
    }

    void SetUp() {
        // TODO (mw) only set up contexts once
        setupGLContext(4);
        fyusion::fyusenet::GfxContextManager::instance()->setupPBOPools(4, 4);
    }

    void TearDown() {
        LayerTestBase::cleanup();
        tearDownGLContext();
    }

    static float * computeMaxPool(int xpool, int ypool, int xstride, int ystride, const float *input, int width, int height, int channels) {
        EXPECT_EQ(xpool, xstride);    // for now
        EXPECT_EQ(ypool, ystride);    // for now
        int twidth = width/xstride;
        int theight = height/ystride;
        EXPECT_GT(twidth, 0);
        EXPECT_GT(theight, 0);
        float * output = new float[twidth * theight * channels];
        for (int c=0; c < channels; c++) {
            const float * inptr = input + width*height*c;
            float * outptr = output + twidth*theight*c;
            for (int y=0,yo=0; y < height; y += ystride, yo++) {
                for (int x=0,xo=0; x < width; x += xstride, xo++) {
                    float maxi = inptr[y*width+x];
                    for (int yp=0; yp < ypool; yp++) {
                        for (int xp=0; xp < xpool; xp++) {
                            maxi = std::max(maxi, inptr[(y+yp)*width+(x+xp)]);
                        }
                    }
                    outptr[yo*twidth+xo] = maxi;
                }
            }
        }
        return output;
    }

    static float * computeAvgPool(int xpool, int ypool, int xstride, int ystride, const float *input, int width, int height, int channels) {
        EXPECT_EQ(xpool, xstride);    // for now
        EXPECT_EQ(ypool, ystride);    // for now
        int twidth = width/xstride;
        int theight = height/ystride;
        EXPECT_GT(twidth, 0);
        EXPECT_GT(theight, 0);
        float * output = new float[twidth * theight * channels];
        for (int c=0; c < channels; c++) {
            const float * inptr = input + width*height*c;
            float * outptr = output + twidth*theight*c;
            for (int y=0,yo=0; y < height; y += ystride, yo++) {
                for (int x=0,xo=0; x < width; x += xstride, xo++) {
                    float accu = 0.f;
                    for (int yp=0; yp < ypool; yp++) {
                        for (int xp=0; xp < xpool; xp++) {
                            accu += inptr[(y+yp)*width+(x+xp)];
                        }
                    }
                    outptr[yo*twidth+xo] = accu/(float)(xpool*ypool);
                }
            }
        }
        return output;
    }


};

struct PoolParam {
    PoolParam(int p, int s, int w, int h, int c) :
        pool(p), stride(s), width(w), height(h), channels(c) {}
    int pool;
    int stride;
    int width;
    int height;
    int channels;
};

struct GlobPoolParam {
    GlobPoolParam(int w, int h, int c) : width(w), height(h), channels(c) {
    }
    int width;
    int height;
    int channels;
};

class ParamAvgPoolTest : public PoolLayerTest, public ::testing::WithParamInterface<PoolParam> {
 protected:
    float * referencePool(const float *input, const PoolParam& param) {
        return computeAvgPool(param.pool, param.pool, param.stride,param.stride, input, param.width, param.height, param.channels);
    }
};

class ParamMaxPoolTest : public PoolLayerTest, public ::testing::WithParamInterface<PoolParam> {
 protected:
    float * referencePool(const float *input, const PoolParam& param) {
        return computeMaxPool(param.pool, param.pool,param.stride, param.stride, input, param.width, param.height, param.channels);
    }
};


class ParamGlobalAvgPoolTest : public PoolLayerTest, public ::testing::WithParamInterface<GlobPoolParam> {
 protected:
    float * referencePool(const float *input, const GlobPoolParam& param) {
        return computeAvgPool(param.width, param.height,param.width, param.height, input, param.width, param.height, param.channels);
    }
};


class ParamGlobalMaxPoolTest : public PoolLayerTest, public ::testing::WithParamInterface<GlobPoolParam> {
protected:
    float * referencePool(const float *input, const GlobPoolParam& param) {
        return computeMaxPool(param.width, param.height,param.width, param.height, input, param.width, param.height, param.channels);
    }
};


//-----------------------------------------------------------------------------
// Test Fixtures
//-----------------------------------------------------------------------------

TEST_P(ParamAvgPoolTest, AvgTestShallow) {
    auto param = GetParam();
    float * input = generateRandomData(param.channels, param.width, param.height, -100.0f, 100.0f);
    float * ref = referencePool(input, param);
    gpu::PoolLayerBuilder bld(gpu::PoolLayerBuilder::POOL_AVG, "pool");
    bld.context(context()).shape(param.channels, param.height, param.width, param.channels);
    bld.poolSize(param.pool).downsample(param.stride);
    gpu::AvgPoolLayer layer(bld, 1);
    std::vector<const float *> inputs{input};
    generateTextures(&layer, inputs, nullptr);
    layer.setup();
    layer.forward(1);
    float * result = new float[param.channels * param.width * param.height];
    layer.copyResult(result);
    layer.cleanup();
    int twidth = param.width / param.stride;
    int theight = param.height / param.stride;
    for (int i=0; i < param.channels * twidth * theight; i++) {
        ASSERT_NEAR(result[i], ref[i], 0.5f);
    }
    delete [] result;
    delete [] input;
    delete [] ref;
}


TEST_P(ParamMaxPoolTest, MaxTestShallow) {
    auto param = GetParam();
    std::unique_ptr<float[]> input(generateRandomData(param.channels, param.width, param.height, -100.f, 100.0f));
    std::unique_ptr<float[]> ref(referencePool(input.get(), param));
    gpu::PoolLayerBuilder bld(gpu::PoolLayerBuilder::POOL_AVG, "pool");
    bld.context(context()).shape(param.channels, param.height, param.width, param.channels);
    bld.poolSize(param.pool).downsample(param.stride);
    gpu::MaxPoolLayer layer(bld, 1);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    layer.setup();
    layer.forward(1);
    std::unique_ptr<float[]> result(new float[param.channels * param.width * param.height]);
    layer.copyResult(result.get());
    layer.cleanup();
    int twidth = param.width / param.stride;
    int theight = param.height / param.stride;
    const float * resptr = result.get();
    const float * refptr = ref.get();
    for (int i=0; i < param.channels * twidth * theight; i++) {
        ASSERT_NEAR(resptr[i], refptr[i], 0.5f);
    }
}


TEST_P(ParamAvgPoolTest, AvgTestDeep) {
    auto param = GetParam();
    std::unique_ptr<float[]> input(generateRandomData(param.channels, param.width, param.height, -100.0f, 100.0f));
    std::unique_ptr<float[]> ref(referencePool(input.get(), param));
    gpu::PoolLayerBuilder bld(gpu::PoolLayerBuilder::POOL_AVG, "pool");
    bld.context(context()).shape(param.channels, param.height, param.width, param.channels);
    bld.poolSize(param.pool).downsample(param.stride).deep();
    gpu::deep::DeepAvgPoolLayer layer(bld, 1);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    layer.setup();
    layer.forward(1);
    std::unique_ptr<float[]> result(new float[param.channels * param.width * param.height]);
    layer.copyResult(result.get());
    layer.cleanup();
    int twidth = param.width / param.stride;
    int theight = param.height / param.stride;
    const float * resptr = result.get();
    const float * refptr = ref.get();
    for (int i=0; i < param.channels * twidth * theight; i++) {
        ASSERT_NEAR(resptr[i], refptr[i], 0.5f);
    }
}


TEST_P(ParamMaxPoolTest, MaxTestDeep) {
    auto param = GetParam();
    std::unique_ptr<float[]> input(generateRandomData(param.channels, param.width, param.height, -100.f, 100.0f));
    std::unique_ptr<float[]> ref(referencePool(input.get(), param));
    gpu::PoolLayerBuilder bld(gpu::PoolLayerBuilder::POOL_MAX, "pool");
    bld.context(context()).shape(param.channels, param.height, param.width, param.channels);
    bld.poolSize(param.pool).downsample(param.stride).deep();
    gpu::deep::DeepMaxPoolLayer layer(bld, 1);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    layer.setup();
    layer.forward(1);
    std::unique_ptr<float[]> result(new float[param.channels * param.width * param.height]);
    layer.copyResult(result.get());
    layer.cleanup();
    int twidth = param.width / param.stride;
    int theight = param.height / param.stride;
    const float * resptr = result.get();
    const float * refptr = ref.get();
    for (int i=0; i < param.channels * twidth * theight; i++) {
        ASSERT_NEAR(resptr[i], refptr[i], 0.5f);
    }
}


TEST_P(ParamGlobalAvgPoolTest, GlobAvgTestDeep) {
    auto param = GetParam();
    std::unique_ptr<float[]> input(generateRandomData(param.channels, param.width, param.height, -100.f, 100.0f));
    std::unique_ptr<float[]> ref(referencePool(input.get(), param));
    gpu::PoolLayerBuilder bld(gpu::PoolLayerBuilder::POOL_AVG, "pool");
    bld.context(context()).shape(param.channels, param.height, param.width, param.channels);
    bld.global().deep();
    gpu::deep::DeepGlobalPoolLayer layer(bld, 1);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    layer.setup();
    layer.forward(1);
    std::unique_ptr<float[]> result(new float[param.channels]);
    layer.copyResult(result.get());
    layer.cleanup();
    const float * resptr = result.get();
    const float * refptr = ref.get();
    for (int i=0; i < param.channels; i++) {
        ASSERT_NEAR(resptr[i], refptr[i], 1.0f);
    }
}


TEST_P(ParamGlobalMaxPoolTest, GlobMaxTestDeep) {
    auto param = GetParam();
    std::unique_ptr<float[]> input(generateRandomData(param.channels, param.width, param.height, -100.f, 100.0f));
    std::unique_ptr<float[]> ref(referencePool(input.get(), param));
    gpu::PoolLayerBuilder bld(gpu::PoolLayerBuilder::POOL_MAX, "pool");
    bld.context(context()).shape(param.channels, param.height, param.width, param.channels);
    bld.global().deep();
    gpu::deep::DeepGlobalPoolLayer layer(bld, 1);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    layer.setup();
    layer.forward(1);
    std::unique_ptr<float[]> result(new float[param.channels]);
    layer.copyResult(result.get());
    layer.cleanup();
    const float * resptr = result.get();
    const float * refptr = ref.get();
    for (int i=0; i < param.channels; i++) {
        ASSERT_NEAR(resptr[i], refptr[i], 1.0f);
    }
}



// TODO (mw) more test patterns, maybe fuzz-testing with randomization
INSTANTIATE_TEST_CASE_P(Avg, ParamAvgPoolTest, testing::Values(
                                                     PoolParam(2, 2, 8, 8, 4),
                                                     PoolParam(2, 2, 200, 200, 4),
                                                     PoolParam(2, 2, 80, 40, 12),
                                                     PoolParam(2, 2, 50, 50, 23),
                                                     PoolParam(2, 2, 40, 40, 80)));

INSTANTIATE_TEST_CASE_P(Max, ParamMaxPoolTest, testing::Values(
                                                     PoolParam(2, 2, 200, 200, 4),
                                                     PoolParam(2, 2, 80, 40, 12),
                                                     PoolParam(2, 2, 50, 50, 23),
                                                     PoolParam(2, 2, 40, 40, 80)));

INSTANTIATE_TEST_CASE_P(GlobAvg, ParamGlobalAvgPoolTest, testing::Values(
                                                   GlobPoolParam(80, 40, 56),
                                                   GlobPoolParam(100, 80, 12),
                                                   GlobPoolParam(8, 8, 8),
                                                   GlobPoolParam(200, 200, 4),
                                                   GlobPoolParam(50, 50, 23),
                                                   GlobPoolParam(2, 2, 24),
                                                   GlobPoolParam(8, 4, 24),
                                                   GlobPoolParam(40, 40, 80)));

INSTANTIATE_TEST_CASE_P(GlobMax, ParamGlobalMaxPoolTest, testing::Values(
                                                             GlobPoolParam(80, 40, 56),
                                                             GlobPoolParam(100, 80, 12),
                                                             GlobPoolParam(8, 8, 8),
                                                             GlobPoolParam(200, 200, 4),
                                                             GlobPoolParam(50, 50, 23),
                                                             GlobPoolParam(2, 2, 24),
                                                             GlobPoolParam(8, 4, 24),
                                                             GlobPoolParam(40, 40, 80)));


// vim: set expandtab ts=4 sw=4:
