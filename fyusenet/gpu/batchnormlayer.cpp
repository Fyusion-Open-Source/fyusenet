//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Explicit BatchNorm Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "batchnormlayer.h"
#include "../gl/glexception.h"
#include "../gl/glinfo.h"

namespace fyusion::fyusenet::gpu {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
BatchNormLayer::BatchNormLayer(const GPULayerBuilder & builder, int layerNumber): FunctionLayer(builder, layerNumber) {
    currentShader_ = nullptr;
    vertexArray_ = nullptr;
    vertexBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    for (int i = 0; i < FBO::MAX_DRAWBUFFERS; i++) shaders_[i] = nullptr;
    maxRenderTargets_ = GLInfo::getMaximumRecommendedDrawBuffers();
    hasParameters_ = true;
}

/**
 * @copydoc GPULayerBase::~GPULayerBase
 */
BatchNormLayer::~BatchNormLayer() {
    for (int i = 0; i < (int)blocks_.size(); i++) {
        delete blocks_.at(i);
    }
    blocks_.clear();
}

/**
 * @copydoc GPULayerBase::cleanup
 */
void BatchNormLayer::cleanup() {
    for (int i = 0; i < FBO::MAX_DRAWBUFFERS; i++) {
        shaders_[i].reset();
        shaderStates_[i].reset();
    }
    currentShader_ = nullptr;
    FunctionLayer::cleanup();
}


/**
 * @brief Load batchnorm data from a parameter provider
 *
 * @param source ParameterProvider instance to load the data from
 *
 * This function retrieves the batch-norm data from a supplied ParameterProvider instances using
 * the layer name suffixed with \c ".bn" as the name and the \c subIndex set to 0. The batchnorm
 * data is supposed to be in the following format:
 *  1. all scales (single value per output channel for a total of \c \#output values)
 *  2. all offsets (single value per output channel for a total of \c \#output values)
 *
 *  @see ParameterProvider
 */
void BatchNormLayer::loadParameters(const ParameterProvider * source) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    int rem = outputChannels_;
    int offset = 0;
    auto blob = source->get(getName()+std::string(".bn"), getNumber(), 0);
    const float *scale = std::any_cast<const float *>(blob.get());
    assert(scale);
    const float *bias = scale + outputChannels_;
    assert(bias);
    while (rem > 0) {
        int remunits = std::min(maxRenderTargets_, (rem + (PIXEL_PACKING - 1)) / PIXEL_PACKING);
        auto *block = new BiasScaleBlock(remunits * PIXEL_PACKING);
        int remelems = std::min(rem, remunits * PIXEL_PACKING);
        block->fill(bias + offset, scale + offset, remelems);
        blocks_.push_back(block);
        rem -= remunits * PIXEL_PACKING;
        offset += remunits * PIXEL_PACKING;
    }
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc FunctionLayer::beforeRender
 */
void BatchNormLayer::beforeRender() {
    currentShader_ = nullptr;
}

/**
 * @copydoc FunctionLayer::afterRender
 */
void BatchNormLayer::afterRender() {
    if (currentShader_) currentShader_->unbind();
    currentShader_ = nullptr;
}


/**
 * @copydoc FunctionLayer::renderChannelBatch
 */
void BatchNormLayer::renderChannelBatch(int outPass, int numRenderTargets, int texOffset) {
    for (int tex = 0; tex < numRenderTargets; tex++) {
        glActiveTexture(GL_TEXTURE0 + tex);
        glBindTexture(GL_TEXTURE_2D, inputTextures_.at(tex + texOffset));
    }
    if (currentShader_ != shaders_[numRenderTargets-1].get()) {
        if (currentShader_)
            currentShader_->unbind(true);
        currentShader_ = shaders_[numRenderTargets-1].get();
        currentShader_->bind(shaderStates_[numRenderTargets-1].get());
    }
    BiasScaleBlock *block = blocks_.at(outPass);
    currentShader_->setMappedUniformVec4Array(UNIFORM_BIASSCALE, block->biasScale_, numRenderTargets * 2);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const GLvoid *) 0);
}


/**
 * @copydoc FunctionLayer::setupShaders
 */
void BatchNormLayer::setupShaders() {
    char extra[512];
    for (int i = 1; i <= maxRenderTargets_; i++) {
        snprintf(extra, sizeof(extra), "#define NUM_LANES %d\n", i);
        shaders_[i-1] = compileShaderPair("shaders/default.vert", "shaders/batchnorm.frag", extra, typeid(this));
        try {
            shaders_[i-1]->bindAttributeLocation("attributes0", 0);
            shaders_[i-1]->link();
        } catch (GLException &ex) {
            FNLOGE("Cannot link shader for layer %s", getName().c_str());
            throw;
        }
        shaderStates_[i-1] = UniformState::makeShared(shaders_[i - 1]);
        for (int j=0; j < i; j++) {
            snprintf(extra, sizeof(extra), "inputLayer%d", j);
            shaderStates_[i-1]->setUniformValue(extra, j);
        }
        shaders_[i-1]->mapUniformLocation("biasscale", UNIFORM_BIASSCALE);
    }
}

} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
