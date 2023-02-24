//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Fractional Convolutional Layer w/ 3x3 mask
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
#include "../../gl/glexception.h"
#include "../../common/logging.h"
#include "../../common/performance.h"
#include "fractionalconvlayerNxN_vanilla.h"

//-------------------------------------- Global Variables ------------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace vanilla {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @copydoc vanilla::ConvLayerBase::ConvLayerBase
 */
FractionalConvLayerNxN::FractionalConvLayerNxN(const ConvLayerBuilder & builder, int layerNumber):ConvLayerNxN(builder, layerNumber) {
    if (builder.dilation_[0] > 1 || builder.dilation_[1] > 1) THROW_EXCEPTION_ARGS(FynException,"Dilations not supported for fractional convolution");
    sourceStep_ = builder.sourceStep_;
    int tgtwidth = (int)((float)width_/(sourceStep_ * (float)downsample_[0]));
    int tgtheight = (int)((float)height_/(sourceStep_ * (float)downsample_[1]));
    viewport_[0] = tgtwidth + 2*outputPadding_;
    viewport_[1] = tgtheight + 2*outputPadding_;
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Perform specific convolution shader compilation
 *
 * @param preproc Pointer to preprocessor string which should be used in the shader compilation
 *
 * This compiles and links the convolution shaders that are required for running the 3x3 fractional
 * convolution and also maps and/or sets the uniforms in the shader code to initialize them and
 * update them properly during rendering.
 */
void FractionalConvLayerNxN::compileConvolutionShaders(const char *preproc) {
    char finalpreproc[1024+128] = {0};
    char extra[128];
    char shadername[64];
    snprintf(shadername,sizeof(shadername),"shaders/vanilla/fraconv%dx%d.frag", kernel_, kernel_);
    for (int i=1; i <= maxRenderTargets_; i++) {
        strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
        snprintf(extra,sizeof(extra),"#define NUM_LANES %d\n",i);
        strncat(finalpreproc, extra, sizeof(finalpreproc)-strlen(finalpreproc)-1);
        convolutionShaders_[i-1] = compileShaderPair("shaders/vanilla/convdefault.vert",shadername,finalpreproc,typeid(this));
        try {
            convolutionShaders_[i-1]->bindAttributeLocation("attributes0",0);
            convolutionShaders_[i-1]->link();
        } catch (GLException& ex) {
            FNLOGE("Cannot link shader for layer %s",getName().c_str());
            throw;
        }
        convolutionShaders_[i-1]->bind();
        convolutionShaderStates_[i-1] = UniformState::makeShared(convolutionShaders_[i-1]);
        convolutionShaderStates_[i-1]->setUniformValue("inputLayer",0);
        convolutionShaderStates_[i-1]->setUniformValue("texStep",sourceStep_/(float)(width_+2*inputPadding_));
        convolutionShaders_[i-1]->mapUniformLocation("coeffs",COEFFICIENTS);
        if (outputPadding_>0) {
            convolutionShaders_[i-1]->mapUniformLocation("bias",BIAS);
            convolutionShaders_[i-1]->setMappedUniformVec4Array(BIAS,zeroBias_,i);
        }
        if (flags_ & LayerFlags::POST_BATCHNORM) {
            convolutionShaders_[i-1]->mapUniformLocation("batchnorm",BATCHNORM_DATA);
        }
        convolutionShaders_[i-1]->unbind();
    }
}



} // vanilla namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
