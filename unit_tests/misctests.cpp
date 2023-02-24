//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Miscellaneous Layers Unit Tests
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
#include <fyusenet/gpu/argmaxlayerbuilder.h>
#include <fyusenet/gpu/deep/deepargmaxlayer.h>
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

    void SetUp() {
        // TODO (mw) only set up contexts once
        setupGLContext(4);
        fyusion::fyusenet::GfxContextManager::instance()->setupPBOPools(4, 4);
    }

    void TearDown() {
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

//-----------------------------------------------------------------------------
// Test Fixtures
//-----------------------------------------------------------------------------

TEST_P(ArgMaxTest, ArgMaxTestDeep) {
    auto param = GetParam();
    float * input = generateRandomData(param.channels, param.width, param.height, param.range[0], param.range[1]);
    int * ref = referenceArgMax(input, param);
    gpu::ArgMaxLayerBuilder bld("argmax");
    bld.context(context()).shape(1, param.width, param.height, param.channels).deep();
    gpu::deep::DeepArgMaxLayer layer(bld, 1);
    std::vector<const float *> inputs{input};
    generateTextures(&layer, inputs, nullptr);
    layer.setup();
    layer.forward(1);
    float * result = new float[param.channels * param.width * param.height];
    layer.copyResult(result);
    layer.cleanup();
    int deviations = 0;
    int stride = param.width;
    int chanoffset = param.width*param.height;
    for (int y=0; y < param.height; y++) {
        for (int x=0; x < param.width; x++) {
            float refval = (float)ref[y*stride+x];
            float gpuval = result[y*stride+x];
            if (std::abs(refval-gpuval) > 0.5) {
                int refidx = ref[y*stride+x]*chanoffset+y*stride+x;
                int gpuidx = ((int)gpuval)*chanoffset+y*stride+x;
                if (std::abs(input[refidx]-input[gpuidx]) > param.deltamax) {
                    printf("(%d,%d): ref=%d gpu=%.1f  %f  %f  (%f)\n", x,y,ref[y*stride+x],result[y*stride+x], input[refidx], input[gpuidx], std::abs(input[refidx]-input[gpuidx]));
                    deviations++;
                }
            }
        }
    }
    ASSERT_EQ(deviations, 0);
    delete [] ref;
    delete [] input;
}


// TODO (mw) more test patterns, maybe fuzz-testing with randomization

INSTANTIATE_TEST_CASE_P(ArgMax, ArgMaxTest, testing::Values(
                                                   ArgMaxParam(256, 128, 64),
                                                   ArgMaxParam(120, 80, 3),
                                                   ArgMaxParam(80, 40, 52),
                                                   ArgMaxParam(200, 200, 4),
                                                   ArgMaxParam(50, 50, 31),
                                                   ArgMaxParam(40, 40, 128)));


// vim: set expandtab ts=4 sw=4:
