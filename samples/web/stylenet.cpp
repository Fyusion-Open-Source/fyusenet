//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2022
//--------------------------------------------------------------------------------------------------
// Style-Transfer Network Example Main (Web)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#ifndef FYUSENET_GL_BACKEND
#error This file should not be included in this build configuration
#endif

//--------------------------------------- System Headers -------------------------------------------

#include <emscripten.h>
#include <cassert>
#include <cstdio>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../samplenetworks/stylenet3x3.h"
#include <fyusenet/gl/texturedquad.h>
#include <fyusenet/gl/vao.h>
#include <fyusenet/gl/vertexshader.h>
#include <fyusenet/gl/fragmentshader.h>
#include <fyusenet/gl/shaderprogram.h>

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/**
 * @brief Wrapper class around style-transfer network
 */
class StyleNetWrapper {
  public:

    /**
     * @brief Constructor
     *
     * @param ctx Link to GL context
     * @param canvasWidth Width of the target canvas to render result to
     * @param canvasHeight Height of the target texture to render result o
     */
    StyleNetWrapper(const fyusion::fyusenet::GfxContextLink& ctx, int canvasWidth, int canvasHeight) :
          context_(ctx), outputSize_{canvasWidth, canvasHeight} {
    }

    /**
     * @brief Destructor
     *
     * Deallocates resoures
     */
    ~StyleNetWrapper() {
        if (network_) {            
            network_->cleanup();
            delete network_;
            if (quad_) quad_->cleanup();
            delete quad_;
            program_.reset();
            context_.reset();
            auto glmgr = fyusion::fyusenet::GfxContextManager::instance();
            if (glmgr) glmgr->tearDown();
        }
    }


    /**
     * @brief Initialize / create style-transfer network
     *
     * @param inputWidth Width of the input image data and therefore the network width
     * @param inputHeight Height of the input image data and therefore the network height
     *
     * This creates the underlying style-transfer network for the supplied image resolution.
     */
    void init(int inputWidth, int inputHeight) {
        network_ = new StyleNet3x3(inputWidth, inputHeight, false, false, context_);
        inputSize_[0] = inputWidth;
        inputSize_[1] = inputHeight;
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
            auto * params = new StyleNet3x3Provider(weights, dataSize);
            network_->setParameters(params);
            network_->setup();
            createBlitter();
            delete params;
        } catch (std::exception& ex) {
            EM_ASM({console.log($0);}, ex.what());
            return false;
        }
        return true;
    }

    /**
     * @brief Execute network on supplied texture
     *
     * @param texID GL texture to run the network on
     */
    void forward(GLuint texID) {
        using namespace fyusion::fyusenet::gpu;
        if ((!inputBuffer_) || (currentInput_ != texID)) {
            std::vector<GPUBuffer::slice> textures;
            textures.emplace_back(fyusion::opengl::Texture2DRef(texID, inputSize_[0], inputSize_[1], fyusion::opengl::Texture2D::UINT8, 3));
            delete inputBuffer_;
            inputBuffer_ = GPUBuffer::createShallowBuffer(BufferShape(inputSize_[1], inputSize_[0], 3, 0, GPUBuffer::type::UINT8, GPUBuffer::order::GPU_SHALLOW),
                                                          textures);
            network_->setInputGPUBuffer(inputBuffer_);
        }
        network_->forward();
        blit(network_->getOutputTexture());
    }

 private:

    /**
     * @brief Create shader program for final stage blitter
     *
     * This creates a shader and proxy geometry for blitting the output of the network to the
     * target canvas.
     */
    void createBlitter() {
        const char * vertSrc = "in highp vec4 attributes0;\n"
                               "out highp vec2 texCoord;\n"
                               "void main(void) {\n"
                               "  gl_Position = vec4(attributes0.x, attributes0.y, 0.0, 1.0);\n"
                               "  texCoord = attributes0.zw;\n"
                               "}\n";
        const char * fragSrc = "precision mediump float;\n"
                               "precision mediump sampler2D;\n"
                               "layout(location=0) out vec4 fragColor;\n"
                               "in highp vec2 texCoord;\n"
                               "uniform sampler2D inputTex;\n"
                               "void main() {\n"
                               "  fragColor.rgb = texture(inputTex, texCoord.xy).rgb;\n"
                               "  fragColor.a = 1.0;\n"
                               "}\n";
        vao_ = new fyusion::opengl::VAO(context_);
        quad_ = new fyusion::opengl::TexturedQuad(context_, true);
        quad_->init(vao_);
        fyusion::opengl::shaderptr vs(new fyusion::opengl::VertexShader(vertSrc, context_));
        fyusion::opengl::shaderptr fs(new fyusion::opengl::FragmentShader(fragSrc, context_));
        program_ = fyusion::opengl::ShaderProgram::createInstance();
        program_->addShader(vs);
        program_->addShader(fs);
        program_->compile();
        program_->link();
        program_->bind();
        program_->setUniformValue("inputTex", 0);
        program_->unbind();
    }

    /**
     * @brief Blit output of network to target canvas
     *
     * @param texID Texture ID to blit to canvas
     */
    void blit(GLuint texID) {
        program_->bind();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texID);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, outputSize_[0], outputSize_[1]);
        glDisable(GL_BLEND);
        vao_->bind();
        quad_->draw();
        program_->unbind();
        vao_->unbind();
    }

    fyusion::fyusenet::GfxContextLink context_;
    StyleNet3x3 * network_ = nullptr;
    fyusion::opengl::VAO * vao_ = nullptr;
    fyusion::opengl::TexturedQuad * quad_ = nullptr;
    fyusion::opengl::programptr program_;
    fyusion::fyusenet::gpu::GPUBuffer * inputBuffer_ = nullptr;
    GLuint currentInput_ = 0;
    int inputSize_[2] = {0};
    int outputSize_[2] = {0};
};

static StyleNetWrapper * wrapper = nullptr;


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
            wrapper = new StyleNetWrapper(context, canvasWidth, canvasHeight);
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
 * @param camWidth Camera input width
 * @param camHeight Camera input height
 * @param dataPtr Pointer to weight/bias data
 * @param dataSize Number of bytes in the dataPtr buffer
 *
 * @retval true if initialization was successful
 * @retval false otherwise
 */
extern "C" bool EMSCRIPTEN_KEEPALIVE createNetwork(int camWidth, int camHeight, void * dataPtr, size_t dataSize) {
    if (wrapper) {
        try {
            wrapper->init(camWidth, camHeight);
            wrapper->loadWeights(dataPtr, dataSize);
        } catch (std::exception& ex) {
            return false;
        }
        return true;
    }
    return false;
}


/**
 * @brief Deallocate resources consumed by the network
 */
extern "C" void EMSCRIPTEN_KEEPALIVE tearDown() {
    if (wrapper) delete wrapper;
    wrapper = nullptr;
}


/**
 * @brief Perform network inference on supplied texture
 *
 * @param texID GL texture handle to run network on
 */
extern "C" void EMSCRIPTEN_KEEPALIVE forward(int texID) {
    if (wrapper) wrapper->forward((GLuint)texID);
}


/**
 * @brief Create texxture handle to be used as input for this module
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

