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
#include <fyusenet/gpu/deep/deepconvlayer1x1.h>
#include <fyusenet/gpu/deep/deepconvlayerNxN.h>
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


    /**
     * @brief Reference implementation of a convolution with optional downsampling and input padding
     *
     * @param input Pointer to input data
     * @param weightsAndBiases Pointer to weight/bias data
     * @param outchans
     * @param kernX
     * @param kernY
     * @param inchans
     * @param width
     * @param height
     * @param downX
     * @param downY
     *
     * @return
     *
     * @note Output is always unpadded
     */
    float * paddedConvolution(const float *input, const float *weightsAndBiases, int outchans, int kernX, int kernY, int inchans, int width, int height, int downX=1, int downY=1, bool preReLU=false) const {
        // this implementation is slow and ugly, but then it is only used for unit-testing
        EXPECT_TRUE((kernX & 1) == 1);
        EXPECT_TRUE((kernY & 1) == 1);
        int xpad = (kernX-1)/2;
        int ypad = (kernY-1)/2;
        int outwidth = (width - 2*xpad) / downX;
        int outheight = (height - 2*ypad) / downY;
        float * result = new float[outwidth * outheight * outchans];
        memset(result, 0, outwidth * outheight * outchans * sizeof(float));
        int incstride = width * height;
        int instride = width;
        int outcstride = outwidth * outheight;
        int outstride = outwidth;
        const float * weights = weightsAndBiases + outchans;
        for (int oc=0; oc < outchans; oc++) {
            for (int y=ypad, yo=0; y < height-ypad; y+=downY, yo++) {
                for (int x=xpad,xo=0; x < width-xpad; x+=downX, xo++) {
                    float accu = weightsAndBiases[oc];
                    for (int ic=0; ic < inchans; ic++) {
                        for (int ky=-ypad, kyo=0; ky <= ypad; ky++,kyo++) {
                            for (int kx=-xpad,kxo=0; kx <= xpad; kx++,kxo++) {
                                if (preReLU) {
                                    accu += std::max(0.f, input[ic*incstride + (y+ky)*instride + (x+kx)]) * weights[oc*inchans*kernX*kernY + kyo*kernX*inchans + kxo*inchans +ic];
                                } else {
                                    accu += input[ic*incstride + (y+ky)*instride + (x+kx)] * weights[oc*inchans*kernX*kernY + kyo*kernX*inchans + kxo*inchans +ic];
                                }
                            }
                        }
                    }
                    result[oc*outcstride + yo * outstride + xo] = accu;
                }
            }
        }
        return result;
    }

    float * batchnorm(const float *input, const float * scales, const float * bias, int width, int height, int chans) const {
        float * output = new float[width*height*chans];
        int cstride = width*height;
        for (int c=0; c < chans; c++) {
            for (int y=0; y < height; y++) {
                for (int x=0; x < width; x++) {
                    output[x+y*width+c*cstride] = input[x+y*width+c*cstride] * scales[c] + bias[c];
                }
            }
        }
        return output;
    }

};


struct ConvParam {
    ConvParam(int k, int w, int h, int ic, int oc, int ds=1) : kernel(k), width(w), height(h), inchans(ic), outchans(oc), downsample(ds) {}
    int kernel;
    int width;
    int height;
    int inchans;
    int outchans;
    int downsample;
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
    bld.context(context()).shape(param.outchans, param.height, param.width, param.inchans).type(LayerType::CONVOLUTION2D);
    bld.downsample(param.downsample);
    gpu::vanilla::ConvLayer1x1 layer(bld, 1);
    std::unique_ptr<float[]> input(generateConstantData(1.0f, param.inchans, param.width, param.height));
    ASSERT_NE(input, nullptr);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    float ckernel[1] = {1.0f};
    std::unique_ptr<float[]> wandb(stackConvolution(0.f, ckernel, 1, 1, param.inchans, param.outchans));
    layer.loadWeightsAndBiases(wandb.get(), 0);
    layer.setup();
    layer.forward(1);
    std::unique_ptr<float[]> result(new float[param.outchans * param.width * param.height]);
    layer.copyResult(result.get());
    layer.cleanup();
    const float * rptr = result.get();
    int owidth = param.width / param.downsample;
    int oheight = param.height / param.downsample;
    for (int i=0; i < param.outchans * owidth * oheight; i++) {
        ASSERT_NEAR(rptr[i], param.inchans, 1e-3f);
    }
}

TEST_P(ParamConvLayerTest1x1, DeepConv1x1) {
    auto param = GetParam();
    std::shared_ptr<LayerFactory> factory = LayerFactory::instance(LayerFactory::GPUFactoryType(LayerFactory::GPUFactoryType::VANILLA));
    gpu::ConvLayerBuilder * bld = new gpu::ConvLayerBuilder(1,"conv");
    int pad = 0;
    bld->context(context()).shape(param.outchans, param.height, param.width, param.inchans).type(LayerType::CONVOLUTION2D).number(1).deep().inputPadding(pad);
    bld->downsample(param.downsample);
    bld->push(factory);
    CompiledLayers layers = factory->compileLayers();
    gpu::ConvLayerBase * layer = (gpu::ConvLayerBase *)layers["conv"];
    ASSERT_NE(layer, nullptr);
    ASSERT_EQ(layer->getInputPadding(), pad);
    std::unique_ptr<float[]> input(generateRandomData(param.inchans, param.width, param.height, -10.f, 10.f, layer->getInputPadding()));
    ASSERT_NE(input, nullptr);
    std::vector<const float *> inputs{input.get()};
    generateTextures(layer, inputs, nullptr, true);
    std::unique_ptr<float[]> ckernel(new float[param.kernel * param.kernel]);
    int mid = (param.kernel * param.kernel - 1) / 2;
    for (int i=0; i < param.kernel*param.kernel; i++) {
        if (i < mid) ckernel[i] = -1.f;
        else if (i == mid) ckernel[i] = 0.f;
        else ckernel[i] = 1.f;
    }
    std::unique_ptr<float[]> wandb(stackConvolution(0.f, ckernel.get(), param.kernel, param.kernel, param.inchans, param.outchans));
    int pwidth = param.width + layer->getInputPadding() * 2;
    int pheight = param.height + layer->getInputPadding() * 2;
    std::unique_ptr<float[]> ref(paddedConvolution(input.get(), wandb.get(), param.outchans, param.kernel, param.kernel, param.inchans, pwidth, pheight));
    layer->loadWeightsAndBiases(wandb.get(), 0);
    layer->setup();
    layer->forward(1);
    std::unique_ptr<float[]> result(new float[param.outchans * param.width * param.height]);
    layer->copyResult(result.get());
    layer->cleanup();
    const float * resptr = result.get();
    const float * refptr = ref.get();
    int outwidth = param.width / param.downsample;
    int outheight = param.height / param.downsample;
    for (int i=0; i < outwidth * outheight; i++) {
        ASSERT_NEAR(resptr[i], refptr[i], 1e-3f);
    }
}


TEST_P(ParamConvLayerTestNxN, ShallowConvNxN) {
    auto param = GetParam();
    std::shared_ptr<LayerFactory> factory = LayerFactory::instance(LayerFactory::GPUFactoryType(LayerFactory::GPUFactoryType::VANILLA));
    gpu::ConvLayerBuilder * bld = new gpu::ConvLayerBuilder(param.kernel,"conv");
    bld->context(context()).shape(param.outchans, param.height, param.width, param.inchans).type(LayerType::CONVOLUTION2D).number(1);
    bld->downsample(param.downsample);
    bld->push(factory);
    CompiledLayers layers = factory->compileLayers();
    gpu::ConvLayerBase * layer = (gpu::ConvLayerBase *)layers["conv"];
    ASSERT_NE(layer, nullptr);
    std::unique_ptr<float[]> input(generateConstantData(1.0f, param.inchans, param.width, param.height));
    ASSERT_NE(input, nullptr);
    std::vector<const float *> inputs{input.get()};
    generateTextures(layer, inputs, nullptr);
    std::unique_ptr<float[]> ckernel(new float[param.kernel * param.kernel]);
    int mid = (param.kernel * param.kernel - 1) / 2;
    for (int i=0; i < param.kernel*param.kernel; i++) {
        if (i < mid) ckernel[i] = -1.f;
        else if (i == mid) ckernel[i] = 0.f;
        else ckernel[i] = 1.f;
    }
    std::unique_ptr<float[]> wandb(stackConvolution(0.f, ckernel.get(), param.kernel, param.kernel, param.inchans, param.outchans));
    layer->loadWeightsAndBiases(wandb.get(), 0);
    layer->setup();
    layer->forward(1);
    std::unique_ptr<float[]> result(new float[param.outchans * param.width * param.height]);
    layer->copyResult(result.get());
    layer->cleanup();
    const float * rptr = result.get();
    int owidth = param.width / param.downsample;
    int oheight = param.height / param.downsample;
    for (int i=0; i < param.outchans * owidth * oheight; i++) {
        ASSERT_NEAR(rptr[i], 0, 1e-3f);
    }
}


TEST_P(ParamConvLayerTestNxN, DeepConvNxN) {
    auto param = GetParam();
    std::shared_ptr<LayerFactory> factory = LayerFactory::instance(LayerFactory::GPUFactoryType(LayerFactory::GPUFactoryType::VANILLA));
    gpu::ConvLayerBuilder * bld = new gpu::ConvLayerBuilder(param.kernel,"conv");
    int pad = (param.kernel-1)/2;
    bld->context(context()).shape(param.outchans, param.height, param.width, param.inchans).type(LayerType::CONVOLUTION2D).number(1).deep().inputPadding(pad);
    bld->downsample(param.downsample);
    bld->push(factory);
    CompiledLayers layers = factory->compileLayers();
    gpu::ConvLayerBase * layer = (gpu::ConvLayerBase *)layers["conv"];
    ASSERT_NE(layer, nullptr);
    ASSERT_EQ(layer->getInputPadding(), pad);
    std::unique_ptr<float[]> input(generateConstantData(1.0f, param.inchans, param.width, param.height, layer->getInputPadding()));
    ASSERT_NE(input, nullptr);
    std::vector<const float *> inputs{input.get()};
    generateTextures(layer, inputs, nullptr, true);
    std::unique_ptr<float[]> ckernel(new float[param.kernel * param.kernel]);
    int mid = (param.kernel * param.kernel - 1) / 2;
    for (int i=0; i < param.kernel*param.kernel; i++) {
        if (i < mid) ckernel[i] = -1.f;
        else if (i == mid) ckernel[i] = 0.f;
        else ckernel[i] = 1.f;
    }
    std::unique_ptr<float[]> wandb(stackConvolution(0.f, ckernel.get(), param.kernel, param.kernel, param.inchans, param.outchans));
    int pwidth = param.width + layer->getInputPadding() * 2;
    int pheight = param.height + layer->getInputPadding() * 2;
    std::unique_ptr<float[]> ref(paddedConvolution(input.get(), wandb.get(), param.outchans, param.kernel, param.kernel, param.inchans, pwidth, pheight, param.downsample, param.downsample));
    layer->loadWeightsAndBiases(wandb.get(), 0);
    layer->setup();
    layer->forward(1);
    std::unique_ptr<float[]> result(new float[param.outchans * param.width * param.height]);
    layer->copyResult(result.get());
    layer->cleanup();
    const float * resptr = result.get();
    const float * refptr = ref.get();
    int outwidth = param.width / param.downsample;
    int outheight = param.height / param.downsample;
    for (int i=0; i < outwidth * outheight; i++) {
        ASSERT_NEAR(resptr[i], refptr[i], 1e-3f);
    }
}


TEST_F(ConvLayerTest, ShallowConv1x1) {
    const int kernel = 1;
    const int width = 32;
    const int height = 32;
    const int inchans = 4;
    const int outchans = 4;
    const int downsample = 2;
    gpu::ConvLayerBuilder bld(kernel,"conv");
    bld.context(context()).shape(outchans, height, width, inchans).type(LayerType::CONVOLUTION2D);
    bld.downsample(downsample);
    gpu::vanilla::ConvLayer1x1 layer(bld, 1);
    std::unique_ptr<float[]> input(generateConstantData(1.0f, inchans, width, height));
    ASSERT_NE(input, nullptr);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr);
    float ckernel[1] = {1.0f};
    std::unique_ptr<float[]> wandb(stackConvolution(0.f, ckernel, 1, 1, inchans, outchans));
    layer.loadWeightsAndBiases(wandb.get(), 0);
    layer.setup();
    layer.forward(1);
    std::unique_ptr<float[]> result(new float[outchans * width * height]);
    layer.copyResult(result.get());
    layer.cleanup();
    const float * rptr = result.get();
    int owidth = width / downsample;
    int oheight = height / downsample;
    for (int i=0; i < owidth * oheight; i++) {
        ASSERT_NEAR(rptr[i], inchans, 1e-3f);
    }
}



TEST_F(ConvLayerTest, DeepConv5x5) {
    const int kernel = 5;
    const int width = 64;
    const int height = 64;
    const int inchans = 4;
    const int outchans = 4;
    gpu::ConvLayerBuilder bld(kernel, "conv");
    bld.context(context()).shape(outchans, height, width, inchans).type(LayerType::CONVOLUTION2D).number(1).deep().inputPadding((kernel-1)/2);
    gpu::deep::DeepConvLayerNxN layer(bld, 1);
    ASSERT_EQ(layer.getInputPadding(), (kernel-1)/2);
    std::unique_ptr<float[]> input(generateConstantData(1.0f, inchans, width, height, layer.getInputPadding()));
    ASSERT_NE(input, nullptr);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr, true);
    std::unique_ptr<float[]> ckernel(new float[kernel * kernel]);
    int mid = (kernel * kernel - 1) / 2;
    for (int i=0; i < kernel * kernel; i++) {
        if (i < mid) ckernel[i] = -1.f;
        else if (i == mid) ckernel[i] = 0.f;
        else ckernel[i] = 1.f;
    }
    std::unique_ptr<float[]> wandb(stackConvolution(0.f, ckernel.get(), kernel, kernel, inchans, outchans));
    int pwidth = width + layer.getInputPadding()*2;
    int pheight = height + layer.getInputPadding()*2;
    std::unique_ptr<float[]> ref(paddedConvolution(input.get(), wandb.get(), outchans, kernel, kernel, inchans, pwidth, pheight));
    layer.loadWeightsAndBiases(wandb.get(), 0);
    layer.setup();
    layer.forward(1);
    std::unique_ptr<float[]> result(new float[width * height * outchans]);
    layer.copyResult(result.get());
    layer.cleanup();
    const float * resptr = result.get();
    const float * refptr = ref.get();
    for (int i=0; i < width * height * outchans; i++) {
        ASSERT_NEAR(resptr[i], refptr[i], 1e-3f);
    }
}


TEST_F(ConvLayerTest, DeepConv3x3) {
    const int kernel = 3;
    const int width = 256;
    const int height = 128;
    const int inchans = 12;
    const int outchans = 8;
    const int downsample = 2;
    gpu::ConvLayerBuilder bld(kernel, "conv");    
    bld.context(context()).shape(outchans, height, width, inchans).type(LayerType::CONVOLUTION2D).number(1).deep().inputPadding((kernel-1)/2);
    bld.downsample(downsample);
    gpu::deep::DeepConvLayerNxN layer(bld, 1);
    ASSERT_EQ(layer.getInputPadding(), (kernel-1)/2);
    std::unique_ptr<float[]> input(generateConstantData(1.0f, inchans, width, height, layer.getInputPadding()));
    ASSERT_NE(input, nullptr);
    std::vector<const float *> inputs{input.get()};
    generateTextures(&layer, inputs, nullptr, true);
    std::unique_ptr<float[]> ckernel(new float[kernel * kernel]);
    int mid = (kernel * kernel - 1) / 2;
    for (int i=0; i < kernel * kernel; i++) {
        if (i < mid) ckernel[i] = -1.f;
        else if (i == mid) ckernel[i] = 0.f;
        else ckernel[i] = 1.f;
    }
    std::unique_ptr<float[]> wandb(stackConvolution(0.f, ckernel.get(), kernel, kernel, inchans, outchans));
    int pwidth = width + layer.getInputPadding()*2;
    int pheight = height + layer.getInputPadding()*2;
    std::unique_ptr<float[]> ref(paddedConvolution(input.get(), wandb.get(), outchans, kernel, kernel, inchans, pwidth, pheight, downsample, downsample));
    layer.loadWeightsAndBiases(wandb.get(), 0);
    layer.setup();
    layer.forward(1);
    std::unique_ptr<float[]> result(new float[width * height * outchans]);
    layer.copyResult(result.get());
    layer.cleanup();
    const float * resptr = result.get();
    const float * refptr = ref.get();
    int owidth = width / downsample;
    int oheight = height / downsample;
    for (int i=0; i < owidth * oheight * outchans; i++) {
        ASSERT_NEAR(resptr[i], refptr[i], 1e-3f);
    }
}



// TODO (mw) more test patterns, maybe fuzz-testing with randomization

INSTANTIATE_TEST_CASE_P(Conv1x1, ParamConvLayerTest1x1, testing::Values(
                                                                ConvParam(1,64,64,4,4),
                                                                ConvParam(1,64,80,4,4),
                                                                ConvParam(1,128,80,4,8),
                                                                ConvParam(1,56,56,64,64),
                                                                ConvParam(1,128,80,16,8),
                                                                ConvParam(1,256,128,12,4)));


INSTANTIATE_TEST_CASE_P(Conv1x1DS, ParamConvLayerTest1x1, testing::Values(
                                                            ConvParam(1,64,64,4,4,2),
                                                            ConvParam(1,64,80,4,4,2),
                                                            ConvParam(1,128,80,4,8,2),
                                                            ConvParam(1,128,80,16,8,2),
                                                            ConvParam(1,256,128,12,4,2)));


INSTANTIATE_TEST_CASE_P(Conv3x3, ParamConvLayerTestNxN, testing::Values(
                                                            ConvParam(3,64,64,4,4),
                                                            ConvParam(3,64,80,4,4),
                                                            ConvParam(3,128,80,4,8),
                                                            ConvParam(3,128,80,16,8),
                                                            ConvParam(3,256,128,12,8)));

INSTANTIATE_TEST_CASE_P(Conv3x3DS, ParamConvLayerTestNxN, testing::Values(
                                                            ConvParam(3,64,64,4,4,2),
                                                            ConvParam(3,64,80,4,4,2),
                                                            ConvParam(3,128,80,4,8,2),
                                                            ConvParam(3,128,80,16,8,2),
                                                            ConvParam(3,256,128,12,8,2)));

INSTANTIATE_TEST_CASE_P(Conv5x5, ParamConvLayerTestNxN, testing::Values(
                                                            ConvParam(5,64,64,4,4),
                                                            ConvParam(5,64,80,4,4),
                                                            ConvParam(5,128,80,4,8),
                                                            ConvParam(5,128,80,16,8),
                                                            ConvParam(5,256,128,12,8)));

INSTANTIATE_TEST_CASE_P(Conv5x5DS, ParamConvLayerTestNxN, testing::Values(
                                                            ConvParam(5,64,64,4,4,2),
                                                            ConvParam(5,64,80,4,4,2),
                                                            ConvParam(5,128,80,4,8,2),
                                                            ConvParam(5,128,80,16,8,2),
                                                            ConvParam(5,256,128,12,8,2)));


INSTANTIATE_TEST_CASE_P(Conv7x7, ParamConvLayerTestNxN, testing::Values(
                                                            ConvParam(7,64,64,4,4),
                                                            ConvParam(7,64,80,4,4),
                                                            ConvParam(7,128,80,4,8),
                                                            ConvParam(7,128,80,16,8),
                                                            ConvParam(7,256,128,12,8)));

INSTANTIATE_TEST_CASE_P(Conv7x7DS, ParamConvLayerTestNxN, testing::Values(
                                                            ConvParam(7,64,64,4,4,2),
                                                            ConvParam(7,64,80,4,4,2),
                                                            ConvParam(7,128,80,4,8,2),
                                                            ConvParam(7,128,80,16,8,2),
                                                            ConvParam(7,256,128,12,8,2)));

// vim: set expandtab ts=4 sw=4:
