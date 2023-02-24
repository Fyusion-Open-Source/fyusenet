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


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


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
