//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Convolutional Layer w/ 3x3 mask
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
#include "../uniformweightarray.h"
#include "../../common/logging.h"
#include "../../common/performance.h"
#include "deepconvlayerNxN.h"

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
DeepConvLayerNxN::DeepConvLayerNxN(const ConvLayerBuilder& builder,int layerNumber):DeepConvLayerBase(builder, layerNumber) {
    assert((builder.kernel_ % 2) == 1);
    assert(builder.kernel_ > 1);
    assert(builder.groupSize_ == 1);
}



/**
 * @copydoc LayerBase::forward
 */
void DeepConvLayerNxN::forward(uint64_t sequence) {
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
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD,GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE,GL_ONE,GL_ONE,GL_ONE);
    glViewport(0, 0, viewport_[0], viewport_[1]);
    vertexArray_->bind();
    framebuffers_.at(0)->bind();
    framebuffers_.at(0)->setWriteMask();
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTextures_.at(0));
    glActiveTexture(GL_TEXTURE0+DISP_TEXTURE);
    glBindTexture(GL_TEXTURE_2D, inputCoordTexture_);
    glActiveTexture(GL_TEXTURE0+WEIGHT_TEXTURE);
    glBindTexture(GL_TEXTURE_2D, weightTexture_);
    glActiveTexture(GL_TEXTURE0+BIAS_TEXTURE);
    glBindTexture(GL_TEXTURE_2D, biasTexture_);
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        if (residualTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"Residual flag configured, but no such texture found.");
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, residualTextures_.at(0));
    }
    int instances = tiler_->numInputTiles()*kernel_;
    int tris = tiler_->numOutputTiles();
    shader_->bind(shaderState_.get());
    glDrawElements(GL_TRIANGLES, tris*6, GL_UNSIGNED_SHORT, (const GLvoid *)0);
    shader_->unbind(true);
    if (instances > 1)  {
        noBiasShader_->bind(noBiasShaderState_.get());
        glDrawElementsInstanced(GL_TRIANGLES, tris*6, GL_UNSIGNED_SHORT, (const GLvoid *)0, instances-1);
    }
    framebuffers_.at(0)->unbind();
    noBiasShader_->unbind();
    vertexArray_->unbind();
}




/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @copydoc DeepConvLayerBase::compileConvolutionShaders
 */
void DeepConvLayerNxN::compileConvolutionShaders(const char *preproc) {
    char finalpreproc[1024+80] = {0};
    char vtxshader[80], frgshader[80];
    snprintf(vtxshader, sizeof(vtxshader), "shaders/deep/deepconv%dx%d_tiled.vert", kernel_, kernel_);
    snprintf(frgshader, sizeof(vtxshader), "shaders/deep/deepconv%dx%d_tiled.frag", kernel_, kernel_);
    strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
    // NOTE (mw) only add residual on the first pass (the shader preprocessing masks out the residual flag for the deepconv layers)
    if (flags_ & LayerFlags::RESIDUAL_INPUT) strncat(finalpreproc,"#define USE_RESIDUAL\n", sizeof(finalpreproc) - strlen(finalpreproc) - 1);
    shader_ = compileShaderPair(vtxshader, frgshader, finalpreproc,typeid(this));
    shaderPostprocessing(shader_);
    shaderState_ = initShader(shader_);
    strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
    strncat(finalpreproc,"#define INSTANCE_OFFSET 1\n#define NO_BIAS\n", sizeof(finalpreproc) - strlen(finalpreproc) - 1);
    noBiasShader_ = compileShaderPair(vtxshader, frgshader, finalpreproc,typeid(this));
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
unistateptr DeepConvLayerNxN::initShader(programptr shader) {
    unistateptr state = UniformState::makeShared(shader);
    if (!GLInfo::hasBinding()) {
        state->setUniformValue("inputLayer0",0);
        state->setUniformValue("residualLayer0",1,true);
        state->setUniformValue("inputDisplacements",DISP_TEXTURE);
        state->setUniformValue("inputCoeffs",WEIGHT_TEXTURE);
        state->setUniformValue("biasTexture",BIAS_TEXTURE,true);
    }
    if (dilation_[0] > 7) {
        state->setUniformValue("dilationStep",tiler_->getTextureStepX() * dilation_[0]);
    }
    state->setUniformValue("numInputTiles",tiler_->numInputTiles());
    return state;
}



} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
