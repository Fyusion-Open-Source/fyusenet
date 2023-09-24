//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Scaling Layer for Shallow Tensors
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/glexception.h"
#include "../gl/glinfo.h"
#include "../common/logging.h"
#include "scalelayer.h"

namespace fyusion::fyusenet::gpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&,int)
 */
ScaleLayer::ScaleLayer(const ScaleLayerBuilder& builder, int layerNumber) :
      FunctionLayer((const GPULayerBuilder &)builder, layerNumber) {
    type_ = builder.scaleType_;
    rotation_ = builder.rotation_;
    float scalex = (float) builder.upsample_[0] / (float) builder.downsample_[0];
    float scaley = (float) builder.upsample_[1] / (float) builder.downsample_[1];
    viewport_[0] = (int) (scalex * (float) width_) + 2 * outputPadding_;
    viewport_[1] = (int) (scaley * (float) height_) + 2 * outputPadding_;
    currentShader_ = nullptr;
    for (int i = 0; i < FBO::MAX_DRAWBUFFERS; i++) shaders_[i] = nullptr;
    rotate(rotation_);
}


/**
 * @brief Constructor (for use as padding add/removal layer)
 *
 * @param builder GPU-specific layer builder that contains parameterization for the layer
 *
 * @param layerNumber Layer number that defines sequence position in execution
 *
 * @throws FynException in case the layer is initialized with invalid/unsupported parameters
 *
 * @pre GL context that this layer is supposed to be operated under must be current
 *
 * Parses basic information from the supplied \p builder and falls back to to a 1:1 scaling. As
 * the scaling layer uses the most simple shader, this layer can be used to add/remove padding
 * from a tensor.
 */
ScaleLayer::ScaleLayer(const GPULayerBuilder& builder, int layerNumber) :
      FunctionLayer((const GPULayerBuilder &)builder, layerNumber) {
    type_ = ScalingType::NEAREST;
    viewport_[0] = width_ + 2 * outputPadding_;
    viewport_[1] = height_ + 2 * outputPadding_;
    currentShader_ = nullptr;
    for (int i = 0; i < FBO::MAX_DRAWBUFFERS; i++) shaders_[i] = nullptr;
    memset(textureMatrix_, 0, 16 * sizeof(float));
    for (int i = 0; i < 16; i += 5) textureMatrix_[i] = 1.0f;
}


/**
 * @brief Set rotation angle that the image should undergo before scaling it
 *
 * @param degrees Angle of rotation (in degrees, ccw) to rotate the input data in the spatial domain
 *                before applying any scaling. Only rotations in multiple of 90 degrees are allowed
 *                for now.
 *
 * @throws FynException in case an invalid rotation angle has been provided
 */
void ScaleLayer::setRotation(int degrees) {
    if ((degrees % 90) != 0) THROW_EXCEPTION_ARGS(FynException,"Invalid rotation %d supplied", degrees);
    rotate(degrees);
    rotation_ = degrees;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Perform pre-rendering initializations
 *
 * This function is invoked by the forward() function prior to performing any rendering and
 * after prepareRender() has been called, which sets the ROP to the correct mode and also
 * adjusts the viewport. The implementation of this function performs inits and adjustments
 * related to the texture interpolation, depending on what kind of scaling was selected for this
 * layer.
 */
void ScaleLayer::beforeRender() {
    currentShader_ = nullptr;
    for (int i=0; i < (int)inputTextures_.size(); i++) {
        glBindTexture(GL_TEXTURE_2D,inputTextures_.at(i));
        if (type_ == ScalingType::LINEAR) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        }
    }
}


/**
 * @brief Perform post-rendering work
 *
 * This function is invoked by the forward() function after all rendering has been done. The
 * implementation of this function performs required cleanups / data resets in order to
 * prepare the instance for the next round of inference and also resets the texture interpolation
 * on he input textures to its default values, which is nearest-neighbor interpolation.
 */
void ScaleLayer::afterRender() {
    if (currentShader_) currentShader_->unbind();
    currentShader_ = nullptr;
    for (int i = 0; i < (int)inputTextures_.size(); i++) {
        glBindTexture(GL_TEXTURE_2D, inputTextures_.at(i));
        if (type_ == ScalingType::LINEAR) {
            // reset interpolation to nearest -> default
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        }
    }
}


/**
 * @copydoc FunctionLayer::renderChannelBatch
 */
void ScaleLayer::renderChannelBatch(int outPass, int numRenderTargets, int texOffset) {
    for (int tex = 0; tex < numRenderTargets; tex++) {
        glActiveTexture(GL_TEXTURE0 + tex);
        glBindTexture(GL_TEXTURE_2D, inputTextures_.at(tex + texOffset));
    }
    if (currentShader_ != shaders_[numRenderTargets - 1].get()) {
        if (currentShader_) currentShader_->unbind(true);
        currentShader_ = shaders_[numRenderTargets - 1].get();
        currentShader_->bind(shaderStates_[numRenderTargets - 1].get());
        currentShader_->setMappedUniformMat4(TEXTRANS, textureMatrix_);
    }
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const GLvoid *) 0);
}


/**
 * @copydoc FunctionLayer::setupShaders
 */
void ScaleLayer::setupShaders() {
    char preproc[1024] = {0};
    for (int i=1; i <= maxRenderTargets_; i++) {
        snprintf(preproc, sizeof(preproc), "#define NUM_LANES %d\n",i);
        preprocessor_.generatePreprocessorPreamble(flags_, preproc, sizeof(preproc)-strlen(preproc)-1);
        shaders_[i-1] = compileShader(preproc);
        shaders_[i-1]->bind();
        shaderStates_[i-1] = initShader(shaders_[i-1],i);
        shaders_[i-1]->unbind();
    }
}


/**
 * @brief Compile single shader given a preprocessor string
 *
 * @param preproc Preprocessor string
 *
 * @return Shared pointer to compiled and linked shader
 *
 * @throws GLException in case of errors
 */
programptr ScaleLayer::compileShader(const char *preproc) {
    programptr shader = compileShaderPair("shaders/default.vert", "shaders/scaling.frag", preproc, typeid(this));
    try {
        shader->bindAttributeLocation("attributes0",0);
        shader->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    return shader;
}


/**
 * @brief Initialize shader uniform variables in shader state
 *
 * @param shader Shader to create a uniform state object for
 * @param renderTargets Number of render targets for the \p shader
 *
 * @return Shared pointer to UniformState object that stores the shader state
 */
unistateptr ScaleLayer::initShader(programptr shader,int renderTargets) {
    unistateptr state = UniformState::makeShared(shader);
    for (int i = 0; i < renderTargets; i++) {
        char var[128];
        snprintf(var, sizeof(var), "inputLayer%d", i);
        state->setUniformValue(var, i);
        shader->mapUniformLocation("tMatrix", TEXTRANS);
    }
    return state;
}


/**
 * @brief Set rotation (internal)
 *
 * @param degrees Angle of rotation (in degrees, ccw) to rotate the input data in the spatial domain
 *                before applying any scaling
 *
 * @post #textureMatrix_ is updated to reflect the supplied rotation parameter
 *
 * @note This is not really meant to be a general rotator. It is more meant to be used to
 *       flip things that are upside down or portrait vs. landscape
 */
void ScaleLayer::rotate(int degrees) {
    float c = cos(M_PI * ((float) degrees / 180.0f));
    float s = sin(M_PI * ((float) degrees / 180.0f));
    float tx = 0.5 + 0.5 * (s - c);
    float ty = 0.5 - 0.5 * (s + c);
    // beware of column-major order in GL
    textureMatrix_[0] = c; textureMatrix_[4] = -s; textureMatrix_[12] = tx;
    textureMatrix_[1] = s; textureMatrix_[5] =  c; textureMatrix_[13] = ty;
    if ((degrees % 180) != 0) {
        std::swap(viewport_[0], viewport_[1]);
    }
}


} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
