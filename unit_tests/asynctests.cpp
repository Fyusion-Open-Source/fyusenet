//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Sync/Async Inference Unit Tests
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>
#include <cmath>
#include <fstream>
#include <memory>
#include <thread>
#include <cassert>
#include <functional>
#include <algorithm>

//-------------------------------------- Project  Headers ------------------------------------------

#include <gtest/gtest.h>
#include <fyusenet/fyusenet.h>
#include "gltesthelpers.h"
#include "number_render.h"
#include "../samples/helpers/jpegio.h"
#include "../samples/samplenetworks/stylenet3x3.h"

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


/**
 * @brief
 */
class AsyncTest : public ::testing::Test, public TestContextManager {
 public:
    AsyncTest() {
    }

    ~AsyncTest() {
    }

 protected:

    void forward(const float *image = nullptr) {
        using netstate = fyusion::fyusenet::NeuralNetwork::state;
        using namespace std::chrono_literals;
        ASSERT_NE(network_, nullptr);
        if (!image) {
            ASSERT_NE(image_, nullptr);
        }
        network_->setInputBuffer((image) ? image : image_);
        auto state = network_->forward();
        ASSERT_NE(state.status, netstate::EXEC_ERROR);
    }

    void finish() {
        using netstate = fyusion::fyusenet::NeuralNetwork::state;
        ASSERT_NE(network_, nullptr);
        auto state = network_->finish();
        ASSERT_NE(state.status, netstate::EXEC_ERROR);
    }

    void initNetwork(const char * testImage) {
        int imgwidth, imgheight;
        std::unique_ptr<uint8_t[]> img8bit(JPEGIO::loadRGBImage(testImage, imgwidth, imgheight));
        ASSERT_NE(img8bit, nullptr);
        image_ = new float[imgwidth * imgheight * 3];
        const uint8_t * imgptr = img8bit.get();
        for (int i=0; i < imgwidth * imgheight * 3 ; i++) {
            image_[i] = ((float)imgptr[i])/255.f;
        }
        initNetwork(imgwidth, imgheight);
    }


    void initNetwork(int width, int height) {
        network_ = new StyleNet3x3(width, height, true, true, context());
        StyleNet3x3::AsyncAdapter callbacks;
        callbacks.downloadReady(std::bind(&AsyncTest::imageReady, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3))
            .uploadReady(std::bind(&AsyncTest::uploadReady, this, std::placeholders::_1, std::placeholders::_2));
        network_->asynchronous(callbacks);
        FILE * in = fopen("stylenet3x3_112_v3.dat", "rb");
        ASSERT_NE(in, nullptr);
        fseek(in, 0, SEEK_END);
        size_t fsz = ftell(in);
        fseek(in, 0, SEEK_SET);
        ASSERT_EQ(fsz % sizeof(float), 0ul);
        std::unique_ptr<float[]> weights(new float[fsz / sizeof(float)]);
        ASSERT_NE(weights, nullptr);
        fread(weights.get(), 1, fsz, in);
        fclose(in);
        network_->loadWeightsAndBiases(weights.get(), fsz / sizeof(float));
        network_->setup();
    }


    virtual void SetUp() override {
        // TODO (mw) only set up contexts once
        setupGLContext(4);
        fyusion::fyusenet::GfxContextManager::instance()->setupPBOPools(4, 4);
    }

    virtual void TearDown() override {
        if (network_) {
            network_->finish();
            network_->cleanup();
        }
        delete network_;
        network_ = nullptr;
        tearDownGLContext();
        delete [] image_;
        image_ = nullptr;
    }

    void imageReady(const std::string& name, uint64_t seqNo, fyusion::fyusenet::cpu::CPUBuffer * buffer) {
        if (dwnlCallback_) dwnlCallback_(seqNo, buffer);
    }

    void uploadReady(const std::string& name, uint64_t seqNo) {
        lock_.lock();
        blocked_ = false;
        unblock_.notify_one();
        lock_.unlock();
    }

    float * image_ = nullptr;
    StyleNet3x3 * network_ = nullptr;
    std::mutex lock_;
    std::condition_variable unblock_;
    std::function<void(uint64_t, fyusion::fyusenet::cpu::CPUBuffer *)> dwnlCallback_;
    bool blocked_ = false;
};

//-----------------------------------------------------------------------------
// Test Fixtures
//-----------------------------------------------------------------------------

TEST_F(AsyncTest, AsyncTest01GC) {
    using namespace fyusion::fyusenet;
    initNetwork("butterfly_1524x1856.jpg");
    for (int i=0; i < 100; i++) {
        forward();
    }
    finish();
    // TODO (mw) there is no pass/fail criterion, this test just executes inference
}

TEST_F(AsyncTest, AsyncTest02GC) {
    using namespace fyusion::fyusenet;
    NumberRender render(512, 512, 8, 3);
    initNetwork(512, 512);
    std::unique_ptr<uint8_t[]> rgbout(new uint8_t[512*512*3]);
    auto callback = [&](uint64_t seq, fyusion::fyusenet::cpu::CPUBuffer *buf) {
        char fname[256];
        ASSERT_NE(buf, nullptr);
        const float * src = buf->map<float>();
        ASSERT_NE(src, nullptr);
        uint8_t *rgbptr = rgbout.get();
        for (int i=0; i < 512*512; i++) {
            rgbptr[i*3+0] = (uint8_t)std::min(255,std::max(0, (int)(src[i*4+0]*255.f)));
            rgbptr[i*3+1] = (uint8_t)std::min(255,std::max(0, (int)(src[i*4+1]*255.f)));
            rgbptr[i*3+2] = (uint8_t)std::min(255,std::max(0, (int)(src[i*4+2]*255.f)));
        }
        buf->unmap();
        sprintf(fname,"/tmp/async_%03d.jpg", (int)seq);
        JPEGIO::saveRGBImage(rgbout.get(), 512, 512, fname);
    };
    dwnlCallback_ = callback;
    for (int i=0; i < 20; i++) {
        float * img = render.generate(i, 0);
        forward(img);
        delete [] img;
    }
    finish();
    // TODO (mw) there is no pass/fail criterion, this test just executes inference
}



// vim: set expandtab ts=4 sw=4:
