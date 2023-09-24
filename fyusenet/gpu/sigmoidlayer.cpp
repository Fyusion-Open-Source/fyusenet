//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Dedicated Sigmoid Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "sigmoidlayer.h"

namespace fyusion::fyusenet::gpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
SigmoidLayer::SigmoidLayer(const GPULayerBuilder & builder, int layerNumber) : FunctionLayer(builder, layerNumber) {
    if (builder.getFlags() & LayerFlags::POST_BATCHNORM) THROW_EXCEPTION_ARGS(FynException,"Batchnorm not supported for this layer");
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void SigmoidLayer::cleanup() {
    // reset shaders here because the GL context is bound here (in case no cache is used)
    for (int i=0; i < FBO::MAX_DRAWBUFFERS; i++) {
        shaders_[i].reset();
        shaderStates_[i].reset();
    }
    currentShader_ = nullptr;
    FunctionLayer::cleanup();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc FunctionLayer::beforeRender
 */
void SigmoidLayer::beforeRender() {
    currentShader_ = nullptr;
}

/**
 * @copydoc FunctionLayer::afterRender
 */
void SigmoidLayer::afterRender() {
    if (currentShader_) currentShader_->unbind();
    currentShader_ = nullptr;
}

/**
 * @copydoc FunctionLayer::renderChannelBatch
 */
void SigmoidLayer::renderChannelBatch(int outPass, int numRenderTargets, int texOffset) {
    for (int tex=0;tex<numRenderTargets;tex++) {
        glActiveTexture(GL_TEXTURE0+tex);
        glBindTexture(GL_TEXTURE_2D,inputTextures_.at(tex+texOffset));
    }
    if (currentShader_ != shaders_[numRenderTargets-1].get()) {
        if (currentShader_) currentShader_->unbind(true);
        currentShader_ = shaders_[numRenderTargets-1].get();
        currentShader_->bind(shaderStates_[numRenderTargets-1].get());
    }
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const GLvoid *)0);
}


/**
 * @copydoc FunctionLayer::setupShaders
 */
void SigmoidLayer::setupShaders() {
    char preproc[1024] = {0};
    for (int i=1; i <= maxRenderTargets_; i++) {
        snprintf(preproc, sizeof(preproc), "#define NUM_LANES %d\n", i);
        preprocessor_.generatePreprocessorPreamble(flags_, preproc, sizeof(preproc) - strlen(preproc)-1);
        shaders_[i-1] = compileShaderPair("shaders/default.vert", "shaders/sigmoid.frag", preproc, typeid(this));
        try {
            shaders_[i-1]->bindAttributeLocation("attributes0", 0);
            shaders_[i-1]->link();
        } catch (GLException & ex) {
            FNLOGE("Cannot link shader for layer %s",getName().c_str());
            throw;
        }
        shaderStates_[i-1] = UniformState::makeShared(shaders_[i-1]);
        for (int j=0; j < i; j++) {
            snprintf(preproc, sizeof(preproc), "inputLayer%d", j);
            shaderStates_[i-1]->setUniformValue(preproc, j);
        }
    }
}

} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
