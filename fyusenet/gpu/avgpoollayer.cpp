//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Avgpool Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "avgpoollayer.h"
#include "../gl/glexception.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
AvgPoolLayer::AvgPoolLayer(const PoolLayerBuilder & builder, int layerNumber) : PoolingLayer(builder, layerNumber) {
    vertexArray_ = nullptr;
    vertexBuffer_ = nullptr;
    indexBuffer_ = nullptr;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc PoolingLayer::compileShader
 */
programptr AvgPoolLayer::compileShader(const char *preproc) {
    programptr shader = compileShaderPair("shaders/default.vert","shaders/avgpool.frag",preproc,typeid(this));
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
 * @copydoc PoolingLayer::renderChannelBatch
 */
void AvgPoolLayer::renderChannelBatch(int outPass,int numRenderTargets,int texOffset) {
    for (int tex=0;tex<numRenderTargets;tex++) {
        glActiveTexture(GL_TEXTURE0+tex);
        glBindTexture(GL_TEXTURE_2D,inputTextures_.at(tex+texOffset));
    }
    if (currentShader_ != shaders_[numRenderTargets-1].get()) {
        if (currentShader_) currentShader_->unbind(true);
        currentShader_ = shaders_[numRenderTargets-1].get();
        currentShader_->bind(shaderStates_[numRenderTargets-1].get());
    }
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT,(const GLvoid *)0);
}


/**
 * @copydoc PoolingLayer::beforeRender
 */
void AvgPoolLayer::beforeRender() {
    glBlendEquation(GL_MAX);
    glBlendFunc(GL_ONE, GL_ONE);
}


/**
 * @copydoc PoolingLayer::afterRender
 */
void AvgPoolLayer::afterRender() {
    glBlendEquation(GL_FUNC_ADD);
}


/**
 * @brief Create shader state for supplied shader
 *
 * @param shader Shader to create a uniform state object for
 * @param renderTargets Number of render targets for the \p shader
 *
 * @return Shared pointer to UniformState object that maps values to the uniforms of a shader
 */
unistateptr AvgPoolLayer::initShader(programptr shader, int renderTargets) {
    unistateptr state = UniformState::makeShared(shader);
    for (int i=0; i < renderTargets; i++) {
        char var[128];
        snprintf(var,sizeof(var),"inputLayer%d", i);
        state->setUniformValue(var, i);
    }
    return state;
}


} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
