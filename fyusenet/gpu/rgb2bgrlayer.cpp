//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// RGB -> BGR Conversion Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "rgb2bgrlayer.h"
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
 * @copydoc GPULayerBase::GPULayerBase
 */
RGB2BGRLayer::RGB2BGRLayer(const GPULayerBuilder & builder,int layerNumber):FunctionLayer(builder,layerNumber) {
    currentShader_ = nullptr;
    for (int i=0;i<FBO::MAX_DRAWBUFFERS;i++) shaders_[i]=nullptr;
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void RGB2BGRLayer::cleanup() {
    for (int i=0; i < maxRenderTargets_; i++) shaders_[i].reset();
    currentShader_ = nullptr;
    FunctionLayer::cleanup();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc FunctionLayer::beforeRender
 */
void RGB2BGRLayer::beforeRender() {
    currentShader_ = nullptr;
}


/**
 * @copydoc FunctionLayer::afterRender
 */
void RGB2BGRLayer::afterRender() {
    if (currentShader_) currentShader_->unbind();
    currentShader_ = nullptr;
}


/**
 * @copydoc FunctionLayer::renderChannelBatch
 */
void RGB2BGRLayer::renderChannelBatch(int outPass,int numRenderTargets,int texOffset) {
    for (int tex=0; tex < numRenderTargets; tex++) {
        glActiveTexture(GL_TEXTURE0+tex);
        glBindTexture(GL_TEXTURE_2D,inputTextures_.at(tex+texOffset));
    }
    if (currentShader_ != shaders_[numRenderTargets-1].get()) {
        if (currentShader_) currentShader_->unbind(true);
        currentShader_ = shaders_[numRenderTargets-1].get();
        currentShader_->bind(shaderStates_[numRenderTargets-1].get());
    }
    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,(const GLvoid *)0);
}


/**
 * @copydoc FunctionLayer::setupShaders
 */
void RGB2BGRLayer::setupShaders() {
    char preproc[128] = {0};
    for (int i=1; i <= maxRenderTargets_; i++) {
        snprintf(preproc, sizeof(preproc), "#define NUM_LANES %d\n",i);
        shaders_[i-1] = compileShader(preproc);
        shaderStates_[i-1] = initShader(shaders_[i-1],i);
    }
}


/**
 * @brief Compile tensor addition shader using supplied preprocessor definitions
 *
 * @param preproc Preprocessor definitions to use
 *
 * @return Shared pointer to compiled and linked shader program
 */
programptr RGB2BGRLayer::compileShader(const char *preproc) {
    programptr shader = compileShaderPair("shaders/default.vert","shaders/rgb2bgr.frag",preproc,typeid(this));
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
 * @brief Create shader state for supplied shader
 *
 * @param shader Shader to create a uniform state object for
 * @param renderTargets Number of render targets for the \p shader
 *
 * @return Shared pointer to UniformState object that maps values to the uniforms of a shader
 */
unistateptr RGB2BGRLayer::initShader(programptr shader, int renderTargets) {
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
