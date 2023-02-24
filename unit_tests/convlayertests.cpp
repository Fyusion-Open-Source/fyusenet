//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Convolution Layers Unit Tests
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
#include <fyusenet/gpu/vanilla/convlayer1x1_vanilla.h>
#include <fyusenet/gpu/vanilla/convlayerNxN_vanilla.h>
#include <fyusenet/base/layerfactory.h>
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

class ConvLayerTest: public ::testing::Test, public TestContextManager, public LayerTestBase {
 protected:
    ConvLayerTest() : TestContextManager() {
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


struct ConvParam {
    ConvParam(int k, int w, int h, int ic, int oc) : kernel(k), width(w), height(h), inchans(ic), outchans(oc) {}
    int kernel;
    int width;
    int height;
    int inchans;
    int outchans;
};


class ParamConvLayerTest1x1: public ConvLayerTest, public ::testing::WithParamInterface<ConvParam> {
};


class ParamConvLayerTestNxN: public ConvLayerTest, public ::testing::WithParamInterface<ConvParam> {
};


//-----------------------------------------------------------------------------
// Test Fixtures
//-----------------------------------------------------------------------------

TEST_P(ParamConvLayerTest1x1, ShallowConv1x1) {
    auto param = GetParam();
    gpu::ConvLayerBuilder bld(1,"conv");
    bld.context(context()).shape(param.outchans, param.width, param.height, param.inchans).type(LayerType::CONVOLUTION2D);
    gpu::vanilla::ConvLayer1x1 layer(bld, 1);
    float * input = generateConstantData(1.0f, param.inchans, param.width, param.height);
    ASSERT_NE(input, nullptr);
    std::vector<const float *> inputs{input};
    generateTextures(&layer, inputs, nullptr);
    float ckernel[1] = {1.0f};
    float *wandb = stackConvolution(0.f, ckernel, 1, param.inchans, param.outchans);
    layer.loadWeightsAndBiases(wandb, 0);
    layer.setup();
    layer.forward(1);
    float * result = new float[param.outchans * param.width * param.height];
    layer.copyResult(result);
    for (int i=0; i < param.outchans * param.width * param.height; i++) {
        ASSERT_NEAR(result[i], param.inchans, 1e-3f);
    }
    layer.cleanup();
    delete [] input;
    delete [] wandb;
}



TEST_P(ParamConvLayerTestNxN, ShallowConvNxN) {
    auto param = GetParam();
    std::shared_ptr<LayerFactory> factory = LayerFactory::instance(LayerFactory::GPUFactoryType(LayerFactory::GPUFactoryType::VANILLA));
    gpu::ConvLayerBuilder * bld = new gpu::ConvLayerBuilder(param.kernel,"conv");
    bld->context(context()).shape(param.outchans, param.width, param.height, param.inchans).type(LayerType::CONVOLUTION2D).number(1);
    bld->push(factory);
    CompiledLayers layers = factory->compileLayers();
    gpu::ConvLayerBase * layer = (gpu::ConvLayerBase *)layers["conv"];
    ASSERT_NE(layer, nullptr);
    float * input = generateConstantData(1.0f, param.inchans, param.width, param.height);
    ASSERT_NE(input, nullptr);
    std::vector<const float *> inputs{input};
    generateTextures(layer, inputs, nullptr);
    float * ckernel = new float[param.kernel * param.kernel];
    int mid = (param.kernel * param.kernel - 1) / 2;
    for (int i=0; i < param.kernel*param.kernel; i++) {
        if (i < mid) ckernel[i] = -1.f;
        else if (i == mid) ckernel[i] = 0.f;
        else ckernel[i] = 1.f;
    }
    float *wandb = stackConvolution(0.f, ckernel, param.kernel, param.inchans, param.outchans);
    layer->loadWeightsAndBiases(wandb, 0);
    layer->setup();
    layer->forward(1);
    float * result = new float[param.outchans * param.width * param.height];
    layer->copyResult(result);
    for (int i=0; i < param.outchans * param.width * param.height; i++) {
        ASSERT_NEAR(result[i], 0, 1e-3f);
    }
    layer->cleanup();
    delete [] input;
    delete [] wandb;
}


// TODO (mw) more test patterns, maybe fuzz-testing with randomization

INSTANTIATE_TEST_CASE_P(Conv1x1, ParamConvLayerTest1x1, testing::Values(
                                                                ConvParam(1,64,64,4,4),
                                                                ConvParam(1,64,80,4,4),
                                                                ConvParam(1,128,80,4,8),
                                                                ConvParam(1,128,80,16,8),
                                                                ConvParam(1,256,128,12,4)));



INSTANTIATE_TEST_CASE_P(Conv3x3, ParamConvLayerTestNxN, testing::Values(
                                                            ConvParam(3,64,64,4,4),
                                                            ConvParam(3,64,80,4,4),
                                                            ConvParam(3,128,80,4,8),
                                                            ConvParam(3,128,80,16,8),
                                                            ConvParam(3,256,128,12,8)));

INSTANTIATE_TEST_CASE_P(Conv5x5, ParamConvLayerTestNxN, testing::Values(
                                                            ConvParam(5,64,64,4,4),
                                                            ConvParam(5,64,80,4,4),
                                                            ConvParam(5,128,80,4,8),
                                                            ConvParam(5,128,80,16,8),
                                                            ConvParam(5,256,128,12,8)));


INSTANTIATE_TEST_CASE_P(Conv7x7, ParamConvLayerTestNxN, testing::Values(
                                                            ConvParam(7,64,64,4,4),
                                                            ConvParam(7,64,80,4,4),
                                                            ConvParam(7,128,80,4,8),
                                                            ConvParam(7,128,80,16,8),
                                                            ConvParam(7,256,128,12,8)));

// vim: set expandtab ts=4 sw=4:
