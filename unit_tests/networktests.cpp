//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Simple Network Unit Tests
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

//-------------------------------------- Global Variables ------------------------------------------

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


int main(int argc,char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    GLEnvironment::init();
    return RUN_ALL_TESTS();
}

class NetworkTestBase : public ::testing::Test, public TestContextManager {
 protected:
    NetworkTestBase() : TestContextManager() {
    }

    virtual void SetUp() override {
        // TODO (mw) only set up contexts once
        setupGLContext(4);
        fyusion::fyusenet::GfxContextManager::instance()->setupPBOPools(4, 4);
    }

    virtual void TearDown() override {
        tearDownGLContext();
    }

};


/**
 * @brief Simple test network which performs a single 3x3 convolution
 */
class TestNet01 : public fyusion::fyusenet::NeuralNetwork {
 public:
    TestNet01(bool async=false) : async_(async) {
    }

    ~TestNet01() {
        delete inputBuffer;
    }

    virtual void setup() override {
        using namespace fyusion::fyusenet;
        using namespace fyusion::fyusenet::cpu;
        NeuralNetwork::setup();
        if (engine_) {
            setInputOutput();
        }
    }

    fyusion::fyusenet::cpu::CPUBuffer * inputBuffer = nullptr;
    fyusion::fyusenet::cpu::CPUBuffer * outputBuffer = nullptr;

 protected:

     virtual void initializeWeights(fyusion::fyusenet::CompiledLayers& layers) override {
         float filter[3*3] = {-1,1,-1,1,0,1,-1,1,-1};
         float wb[3*3*4*8+8]={0};         
         using namespace fyusion::fyusenet;
         int wbidx=8;
         for (int o=0; o<8; o++) {
             for (int y=0; y<3; y++) {
                 for (int x=0; x<3; x++) {
                     float v = filter[y*3+x];
                     for (int i=0; i<4; i++) wb[wbidx++] = v;
                     assert(wbidx < (int)sizeof(wb));
                 }
             }
         }
         ConvLayerInterface * layer = dynamic_cast<ConvLayerInterface *>(layers["conv3x3"]);
         ASSERT_NE(layer, nullptr);
         layer->loadWeightsAndBiases(wb, 0);
     }

    void setInputOutput() {
        using namespace fyusion::fyusenet;
        using namespace fyusion::fyusenet::cpu;
        CompiledLayers & layers = engine_->getLayers();

        LayerBase * layer = layers["upload"];
        ASSERT_NE(layer, nullptr);
        auto specs = layer->getRequiredInputBuffers();
        ASSERT_EQ(specs.size(), 1u);
        const BufferSpec & inspec = specs[0];

        inputBuffer = new CPUBuffer(CPUBufferShape(inspec.width_, inspec.height_, inspec.channels_, 0, CPUBufferShape::FLOAT32, CPUBufferShape::order::GPU_SHALLOW));
        float * in = inputBuffer->map<float>();
        ASSERT_NE(nullptr, in);
        for (int i=0; i < inspec.width_*inspec.height_*inspec.channels_; i++) in[i] = 1.0f;
        inputBuffer->unmap();

        ASSERT_NE(dynamic_cast<cpu::CPULayerInterface *>(layer), nullptr);
        (dynamic_cast<cpu::CPULayerInterface *>(layer))->setInputBuffer(inputBuffer, 0);

        layer = layers["download"];
        ASSERT_NE(layer, nullptr);
        specs = layer->getRequiredOutputBuffers();
        ASSERT_EQ(specs.size(), 1ul);
        const BufferSpec & outspec = specs[0];

        outputBuffer = new CPUBuffer(CPUBufferShape(outspec.width_, outspec.height_, outspec.channels_, 0, CPUBufferShape::FLOAT32, CPUBufferShape::order::GPU_SHALLOW));
        float * out = outputBuffer->map<float>();
        ASSERT_NE(nullptr, in);
        for (int i=0; i < outspec.width_*outspec.height_*outspec.channels_; i++) out[i] = 1.0f;
        outputBuffer->unmap();

        ASSERT_NE(dynamic_cast<cpu::CPULayerInterface *>(layer), nullptr);
        (dynamic_cast<cpu::CPULayerInterface *>(layer))->addOutputBuffer(outputBuffer, 0);
    }

    virtual fyusion::fyusenet::CompiledLayers buildLayers() override {
        using namespace fyusion::fyusenet;
        std::shared_ptr<LayerFactory> factory = getLayerFactory();
        gpu::UpDownLayerBuilder * up = new gpu::UpDownLayerBuilder(gpu::UpDownLayerBuilder::UPLOAD, "upload");
        up->shape(4, 32, 32, 4).context(context_).number(1);
#ifdef FYUSENET_MULTITHREADING
        if (async_) up->async();
#endif
        up->push(factory);
        gpu::ConvLayerBuilder * conv = new gpu::ConvLayerBuilder(3, "conv3x3");
        conv->shape(8, 32, 32, 4).type(LayerType::CONVOLUTION2D).context(context_).number(2);
        conv->push(factory);
        gpu::UpDownLayerBuilder * down = new gpu::UpDownLayerBuilder(gpu::UpDownLayerBuilder::DOWNLOAD, "download");
        down->shape(8, 32, 32, 8).context(context_).number(3);
#ifdef FYUSENET_MULTITHREADING
        if (async_) down->async();
#endif
        down->push(factory);
        return factory->compileLayers();
    }

    virtual void connectLayers(fyusion::fyusenet::CompiledLayers& layers, fyusion::fyusenet::BufferManager * buffers) override {
        buffers->connectLayers(layers[1], layers[2], 0);
        buffers->connectLayers(layers[2], layers[3], 0);
    }

    bool async_= false;

};

//-----------------------------------------------------------------------------
// Test Fixtures
//-----------------------------------------------------------------------------

TEST_F(NetworkTestBase, SimpleSyncTest01GC) {
    using namespace fyusion::fyusenet;
    TestNet01 net;
    net.setup();
    NeuralNetwork::execstate st = net.forward();
    ASSERT_EQ(st.status, NeuralNetwork::state::EXEC_DONE);
    const float * res = net.outputBuffer->map<float>();
    ASSERT_NE(res, nullptr);
    for (int i=0; i < (int)(net.outputBuffer->bytes() / sizeof(float)); i++) {
        ASSERT_EQ(res[i], 0.f);
    }
    net.outputBuffer->unmap();
    net.cleanup();
}

#ifdef FYUSENET_MULTITHREADING
TEST_F(NetworkTestBase, SimpleAsyncTest01GC) {
    using namespace fyusion::fyusenet;
    TestNet01 net(true);
    net.asynchronous();
    net.setup();
    NeuralNetwork::execstate st = net.forward();
    st = net.finish();
    ASSERT_EQ(st.status, NeuralNetwork::state::EXEC_DONE);
    const float * res = net.outputBuffer->map<float>();
    ASSERT_NE(res, nullptr);
    for (int i=0; i < (int)(net.outputBuffer->bytes() / sizeof(float)); i++) {
        ASSERT_EQ(res[i], 0.f);
    }
    net.outputBuffer->unmap();
    net.cleanup();
}
#endif


// vim: set expandtab ts=4 sw=4:
