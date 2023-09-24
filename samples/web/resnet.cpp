//--------------------------------------------------------------------------------------------------
// FyuseNet Samples
//--------------------------------------------------------------------------------------------------
// ResNet-50 (ImageNet) Network Example Main (Web)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <emscripten.h>
#include <cassert>
#include <cstdio>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../samplenetworks/resnet50.h"
#include "../helpers/resnet_provider.h"
#include <fyusenet/gl/fragmentshader.h>

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/**
 * @brief Wrapper class around ResNet network
 */
class ResNetWrapper {
  public:

    /**
     * @brief Constructor
     *
     * @param ctx Link to GL context
     * @param canvasWidth Width of the target canvas to render result to
     * @param canvasHeight Height of the target texture to render result o
     */
    ResNetWrapper(const fyusion::fyusenet::GfxContextLink& ctx, int canvasWidth, int canvasHeight) :
          context_(ctx)  {
    }

    /**
     * @brief Destructor
     *
     * Deallocates resources
     */
    ~ResNetWrapper() {
        if (network_) {            
            network_->cleanup();
            delete network_;
            context_.reset();
            auto glmgr = fyusion::fyusenet::GfxContextManager::instance();
            if (glmgr) glmgr->tearDown();
        }
    }


    /**
     * @brief Initialize / create ResNet network
     *
     * This creates the underlying ResNet network
     */
    void init() {
        network_ = new ResNet50(true, true, context_);
    }

    /**
     * @brief Load weight/bias data into network
     *
     * @param dataPtr Pointer to weight and biases
     * @param dataSize umber of bytes that backs the supplied \p dataPtr
     *
     * @retval true if loading was successful
     * @retval false otherwise
     */
    bool loadWeights(void *dataPtr, size_t dataSize) {
        try {
            uint8_t * weights = reinterpret_cast<uint8_t *>(dataPtr);
            auto * params = new ResNet50Provider(weights, dataSize);
            network_->setParameters(params);
            network_->setup();
            delete params;
        } catch (std::exception& ex) {
            EM_ASM({console.log($0);}, ex.what());
            return false;
        }
        return true;
    }

    void runWithImage(const uint8_t * rgb, int width, int height) {
        // slow conversion of image
        float * rgbf = new float[width * height * 3];
        for (int i=0; i < width*height*3; i++) rgbf[i] = (float)rgb[i] / 255.0f;
        network_->setInputBuffer(rgbf);
        delete [] rgbf;
        network_->forward();
    }

 private:

    fyusion::fyusenet::GfxContextLink context_;
    ResNet50 * network_ = nullptr;
};

static ResNetWrapper * wrapper = nullptr;


/**
 * @brief Initialize GL context on the supplied target canvas and create a network wrapper
 *
 * @param canvas ID of the target canvas in the browser DOM document
 * @param canvasWidth Width of the target canvas
 * @param canvasHeight Height of the target canvas
 *
 * @retval true if context creation was successful
 * @retval false otherwise
 */
extern "C" bool EMSCRIPTEN_KEEPALIVE initContext(char *canvas, int canvasWidth, int canvasHeight) {
    try {
        auto glmgr = fyusion::fyusenet::GfxContextManager::instance();
        if (!glmgr) {
            return false;
        }
        auto context = glmgr->createMainContext(canvas, canvasWidth, canvasHeight);
        if (context.isValid()) {
            wrapper = new ResNetWrapper(context, canvasWidth, canvasHeight);
            return true;
        }
        return false;
    } catch (std::exception& ex) {
        return false;
    }
}

/**
 * @brief Create style-transfer net and initialize it with weights
 *
 * @param dataPtr Pointer to weight/bias data
 * @param dataSize Number of bytes in the dataPtr buffer
 *
 * @retval true if initialization was successful
 * @retval false otherwise
 */
extern "C" bool EMSCRIPTEN_KEEPALIVE createNetwork(void * dataPtr, size_t dataSize) {
    if (wrapper) {
        try {
            wrapper->init();
            wrapper->loadWeights(dataPtr, dataSize);
        } catch (std::exception& ex) {
            return false;
        }
        return true;
    }
    return false;
}


extern "C" void EMSCRIPTEN_KEEPALIVE setImage(void *ptr, size_t dataSize) {
    // FIXME (mw) wrong
    wrapper->runWithImage((const uint8_t *)ptr, 224, 224);
}


/**
 * @brief Deallocate resources consumed by the network
 */
extern "C" void EMSCRIPTEN_KEEPALIVE tearDown() {
    if (wrapper) delete wrapper;
    wrapper = nullptr;
}

/**
 * @brief Create texture handle to be used as input for this module
 *
 * @return GL handle
 *
 * This function creates
 */
extern "C" GLuint EMSCRIPTEN_KEEPALIVE createInputTexture() {
    GLuint tex=0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return tex;
}

extern "C" int main(int argc, char **argv) {
    return 0;
}


// vim: set expandtab ts=4 sw=4:

