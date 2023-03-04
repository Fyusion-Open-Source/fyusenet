//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Transpose-Convolutional Layer w/ 2x2 mask
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

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
#include "deeptransconvlayer2x2.h"

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
 * @copydoc GPULayerBase::GPULayerBase
 */
DeepTransConvLayer2x2::DeepTransConvLayer2x2(const ConvLayerBuilder& builder,int layerNumber) :
    DeepTransConvLayerBase(builder, layerNumber) {
    assert(builder.kernel_ == 2);
    assert(builder.upsample_[0] == 2 && builder.upsample_[1] == 2);
    if ((builder.upsample_[0] != 2) || (builder.upsample_[1] != 2)) {
        THROW_EXCEPTION_ARGS(FynException, "The current implementation requires an upsampling operation by stride-2");
    }
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        THROW_EXCEPTION_ARGS(FynException, "Transpose convolutions do not support residuals as of now");
    }
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepTransConvLayer2x2::cleanup() {
    shaderState_.reset();
    noBiasShaderState_.reset();
    shader_.reset();
    noBiasShader_.reset();
    DeepTransConvLayerBase::cleanup();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @copydoc DeepTransConvLayerBase::renderPass()
 */
void DeepTransConvLayer2x2::renderPass(int pass) {
    int instances = tiler_->numInputTiles();
    int tris = tiler_->numOutputTiles();
    glStencilFuncSeparate(GL_FRONT_AND_BACK, GL_EQUAL, pass+1, 0xFF);
    shader_->bind(shaderState_.get());
    shader_->setMappedUniformValue(PASS,pass);
    glDrawElements(GL_TRIANGLES,tris*6,GL_UNSIGNED_SHORT,(const GLvoid *)0);
    shader_->unbind((instances > 1) ? true : false);
    if (instances > 1) {
        noBiasShader_->bind(noBiasShaderState_.get());
        noBiasShader_->setMappedUniformValue(PASS,pass);
        glDrawElementsInstanced(GL_TRIANGLES,tris*6,GL_UNSIGNED_SHORT,(const GLvoid *)0,instances-1);
        noBiasShader_->unbind();
    }
}


/**
 * @copydoc DeepConvLayerBase::compileConvolutionShaders
 */
void DeepTransConvLayer2x2::compileConvolutionShaders(const char *preproc) {
    char finalpreproc[1024] = {0};
    strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
    shader_ = compileShaderPair("shaders/deep/deeptransconv2x2_stride2.vert","shaders/deep/deeptransconv2x2_stride2.frag",finalpreproc,typeid(this));
    try {
        shader_->bindAttributeLocation("attributes0",0);
        shader_->bindAttributeLocation("attributes1",1);
        shader_->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    shaderState_ = initShader(shader_);
    strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
    strncat(finalpreproc,"#define INSTANCE_OFFSET 1\n#define NO_BIAS\n", sizeof(finalpreproc) - strlen(finalpreproc) - 1);
    noBiasShader_ = compileShaderPair("shaders/deep/deeptransconv2x2_stride2.vert","shaders/deep/deeptransconv2x2_stride2.frag",finalpreproc,typeid(this));
    noBiasShader_->bindAttributeLocation("attributes0",0);
    noBiasShader_->bindAttributeLocation("attributes1",1);
    noBiasShader_->link();
    noBiasShaderState_ = initShader(noBiasShader_);
}


/**
 * @brief Create shader state for supplied shader
 *
 * @param shader Shader to create a uniform state object for
 *
 * @return Shared pointer to UniformState object that maps values to the uniforms of a shader
 */
unistateptr DeepTransConvLayer2x2::initShader(programptr shader) {
    unistateptr state = UniformState::makeShared(shader);
    state->setUniformValue("inputLayer0",0);
    state->setUniformValue("inputDisplacements",DISP_TEXTURE);
    state->setUniformValue("inputCoeffs",WEIGHT_TEXTURE);
    state->setUniformValue("biasTexture",BIAS_TEXTURE,true);
    state->setUniformValue("numInputTiles",tiler_->numInputTiles());
    float hstep = 0.33f / (float)(tiler_->getInputTextureWidth());   // avoid round away from zero in texture lookup for odd fields (horizontal)
    float vstep = 0.33f / (float)(tiler_->getInputTextureHeight());  // avoid round away from zero in texture lookup for odd fields (vertical)
    state->setUniformVec2("texStep",hstep,vstep,true);
    shader->mapUniformLocation("pass",PASS);
    return state;
}


} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
