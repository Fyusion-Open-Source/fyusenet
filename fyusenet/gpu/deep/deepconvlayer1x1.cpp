//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Convolutional Layer w/ 1x1 mask
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/vertexshader.h"
#include "../../gl/fragmentshader.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/glinfo.h"
#include "../../gl/glexception.h"
#include "../floatconversion.h"
#include "../uniformweightarray.h"
#include "../../common/logging.h"
#include "../../common/performance.h"
#include "deepconvlayer1x1.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @copydoc DeepConvLayerBase::DeepConvLayerBase
 */
DeepConvLayer1x1::DeepConvLayer1x1(const ConvLayerBuilder & builder, int layerNumber) : DeepConvLayerBase(builder, layerNumber) {
    assert(builder.kernel_ == 1);
    assert(builder.groupSize_ == 1);
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepConvLayer1x1::cleanup() {
    shaderState_.reset();
    noBiasShaderState_.reset();
    shader_.reset();
    noBiasShader_.reset();
    DeepConvLayerBase::cleanup();
}

/**
 * @copydoc LayerBase::forward
 */
void DeepConvLayer1x1::forward(uint64_t sequence) {
    if (!valid_) THROW_EXCEPTION_ARGS(FynException,"Trying to invoke forward() on invalid layer");
#ifdef DEBUG
    int err = glGetError();
    if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render entry: 0x%x (%s:%d)[%s]",err,__FILE__,__LINE__,getName().c_str());
#endif
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (outputChanged_) updateFBOs();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    if (tiler_->numInputTiles() <= 1) glDisable(GL_BLEND);
    else {
        glEnable(GL_BLEND);
        glBlendEquationSeparate(GL_FUNC_ADD,GL_FUNC_ADD);
        glBlendFuncSeparate(GL_ONE,GL_ONE,GL_ONE,GL_ONE);
    }
    glViewport(0, 0, viewport_[0], viewport_[1]);
    vertexArray_->bind();
    framebuffers_.at(0)->bind();
    framebuffers_.at(0)->setWriteMask();
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    glActiveTexture(GL_TEXTURE0+DISP_TEXTURE);
    glBindTexture(GL_TEXTURE_2D,inputCoordTexture_);
    glActiveTexture(GL_TEXTURE0+WEIGHT_TEXTURE);
    glBindTexture(GL_TEXTURE_2D,weightTexture_);
    glActiveTexture(GL_TEXTURE0+BIAS_TEXTURE);
    glBindTexture(GL_TEXTURE_2D,biasTexture_);
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        if (residualTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"Residual flag configured, but no such texture found.");
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D,residualTextures_.at(0));
    }
    int instances = tiler_->numInputTiles()*kernel_;
    int tris = tiler_->numOutputTiles();
    shader_->bind(shaderState_.get());
    shader_->setUniformValue("numInputTiles",tiler_->numInputTiles());
    glDrawElements(GL_TRIANGLES,tris*6,GL_UNSIGNED_SHORT,(const GLvoid *)0);
    shader_->unbind((instances > 1) ? true : false);
    if (instances > 1) {
        noBiasShader_->bind(noBiasShaderState_.get());
        noBiasShader_->setUniformValue("numInputTiles",tiler_->numInputTiles());
        glDrawElementsInstanced(GL_TRIANGLES,tris*6,GL_UNSIGNED_SHORT,(const GLvoid *)0,instances-1);
        noBiasShader_->unbind();
    }
    framebuffers_.at(0)->unbind();
    vertexArray_->unbind();
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc DeepConvLayerBase::compileConvolutionShaders
 */
void DeepConvLayer1x1::compileConvolutionShaders(const char *preproc) {
    char finalpreproc[1024+80] = {0};
    strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
    // NOTE (mw) only add residual on the first pass (the shader preprocessing masks out the residual flag for the deepconv layers)
    if (flags_ & LayerFlags::RESIDUAL_INPUT) strncat(finalpreproc,"#define USE_RESIDUAL\n", sizeof(finalpreproc) - strlen(finalpreproc) - 1);
    shader_ = compileShaderPair("shaders/deep/deepconv1x1_tiled.vert","shaders/deep/deepconv1x1_tiled.frag", finalpreproc, typeid(this));
    shaderPostprocessing(shader_);
    shaderState_ = initShader(shader_);
    strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
    strncat(finalpreproc,"#define INSTANCE_OFFSET 1\n#define NO_BIAS\n", sizeof(finalpreproc) - strlen(finalpreproc) - 1);
    noBiasShader_ = compileShaderPair("shaders/deep/deepconv1x1_tiled.vert","shaders/deep/deepconv1x1_tiled.frag",finalpreproc,typeid(this));
    shaderPostprocessing(noBiasShader_);
    noBiasShaderState_ = initShader(noBiasShader_);
}


/**
 * @brief Create shader state for supplied shader
 *
 * @param shader Shader to create a uniform state object for
 *
 * @return Shared pointer to UniformState object that maps values to the uniforms of a shder
 */
unistateptr DeepConvLayer1x1::initShader(programptr shader) {
    unistateptr state = UniformState::makeShared(shader);
    if (!GLInfo::hasBinding()) {
        state->setUniformValue("inputLayer0",0);
        state->setUniformValue("residualLayer0",1,true);
        state->setUniformValue("inputDisplacements",DISP_TEXTURE);
        state->setUniformValue("inputCoeffs",WEIGHT_TEXTURE);
        state->setUniformValue("biasTexture",BIAS_TEXTURE,true);
    }
    return state;
}


} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
