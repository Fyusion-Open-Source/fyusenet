//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Convolutional Layer w/ 3x3 and larger mask
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/vertexshader.h"
#include "../../gl/fragmentshader.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/glinfo.h"
#include "../../gl/glexception.h"
#include "../convweightarrayKxKxNxM.h"
#include "../../common/logging.h"
#include "../../common/performance.h"
#include "convlayerNxN_vanilla.h"

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
ConvLayerNxN::ConvLayerNxN(const ConvLayerBuilder & builder, int layerNumber):ConvLayerBase(builder,layerNumber) {
    assert((builder.kernel_ % 2) == 1);
    assert(builder.kernel_ > 1);
    assert(builder.kernel_ <= 9);
    for (int i=0; i <= maxRenderTargets_; i++) {
        convolutionShaders_.push_back(programptr());
        convolutionShaderStates_.push_back(unistateptr());
    }
}



/**
 * @copydoc GPULayerBase::cleanup
 */
void ConvLayerNxN::cleanup() {
    // reset shaders here because the GL context is bound here (in case no cache is used)
    convolutionShaders_.clear();
    convolutionShaderStates_.clear();
    ConvLayerBase::cleanup();
}


/**
 * @copydoc LayerBase::forward
 */
void ConvLayerNxN::forward(uint64_t sequence) {
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
    ShaderProgram *shader = nullptr;
    if (!vertexArray_->bind()) {
        FNLOGE("Cannot render layer %s",getName().c_str());
        return;
    }
    for (int outfield = 0 ; outfield < weights_->numOutputRenderPasses(); outfield++) {
        int sindex = weights_->numRenderTargets(outfield)-1;
        if (convolutionShaders_.at(sindex).get() != shader) {
            if (shader) shader->unbind(true);
            shader = convolutionShaders_.at(sindex).get();
            shader->bind(convolutionShaderStates_.at(sindex).get());
        }
        if (flags_ & LayerFlags::RESIDUAL_INPUT) {
            for (int i=0;i<weights_->numRenderTargets(outfield);i++) {
                int texindex = i+weights_->outputTextureOffset(outfield);
                glActiveTexture(RESIDUAL_START_UNIT+i);
                glBindTexture(GL_TEXTURE_2D,residualTextures_.at(texindex));
            }
        }
        int nummatrices = kernel_*weights_->numRenderTargets(outfield);
        framebuffers_.at(outfield)->bind();
        setBias(outfield,weights_);
        glActiveTexture(GL_TEXTURE0);
        for (int infield = 0; infield < weights_->numInputRenderPasses(); infield++) {
            glBindTexture(GL_TEXTURE_2D,inputTextures_.at(infield));
            if (flags_ & LayerFlags::POST_BATCHNORM) {
                shader->setMappedUniformVec4Array(BATCHNORM_DATA,weights_->getPackageBNScale(outfield),weights_->numRenderTargets(outfield));
            }
            for (int conv=0; conv < kernel_; conv++) {
                const float *matrices = weights_->getPackageWeights(infield, outfield, 0, conv);
                shader->setMappedUniformMat4Array(COEFFICIENTS,matrices,nummatrices);
                if (((flags_ & LayerFlags::RESIDUAL_INPUT) || (outputPadding_>0)) && (conv == kernel_ - 1) && (infield == 0)) {
                    if (flags_ & LayerFlags::RESIDUAL_INPUT) shader->setMappedUniformValue(RESIDUAL_SWITCH,(GLint)1);
                    if (outputPadding_ > 0) {
                        shader->setMappedUniformVec4Array(BIAS, weights_->getPackageBias(outfield), weights_->numRenderTargets(outfield));
                    }
                    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,(const GLvoid *)(conv*6*sizeof(short)));
                    if (outputPadding_ > 0) {
                        shader->setMappedUniformVec4Array(BIAS,zeroBias_,weights_->numRenderTargets(outfield));
                    }
                    if (flags_ & LayerFlags::RESIDUAL_INPUT) shader->setMappedUniformValue(RESIDUAL_SWITCH,(GLint)0);
                } else {
                    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,(const GLvoid *)(conv*6*sizeof(short)));
                }
            }
        }
        framebuffers_.at(outfield)->unbind();
    }
    if (shader) shader->unbind();
#ifdef DEBUG
    err = glGetError();
    if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render exit: 0x%x (%s:%d)[%s]",err,__FILE__,__LINE__,getName().c_str());
#endif
    vertexArray_->unbind();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @copydoc ConvLayerBase::setupShaders
 */
void ConvLayerNxN::setupShaders() {
    char preproc[1024] = {0};
    shaderPreprocessing(preproc, sizeof(preproc)-1);
    compileConvolutionShaders(preproc);
}


/**
 * @brief Perform specific convolution shader compilation
 *
 * @param preproc Pointer to preprocessor string which should be used in the shader compilation
 *
 * This compiles and links the convolution shaders that are required for running the 3x3 convolution
 * and also maps and/or sets the uniforms in the shader code to initialize them and update them
 * properly during rendering.
 */
void ConvLayerNxN::compileConvolutionShaders(const char *preproc) {
    char finalpreproc[1024+128] = {0};
    char extra[128];
    char shadername[64];
    snprintf(shadername, sizeof(shadername), "shaders/vanilla/conv%dx%d.frag", kernel_, kernel_);
    for (int i=1; i <= maxRenderTargets_; i++) {
        strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
        snprintf(extra, sizeof(extra), "#define NUM_LANES %d\n",i);
        ssize_t mc = sizeof(finalpreproc)-strlen(finalpreproc)-1;
        assert(mc > 0);
        strncat(finalpreproc, extra, mc);
        mc -= strlen(extra);
        assert(mc > 0);
        convolutionShaders_[i-1] = compileShaderPair("shaders/vanilla/convdefault.vert",shadername,finalpreproc,typeid(this));
        try {
            convolutionShaderStates_[i-1] = UniformState::makeShared(convolutionShaders_.at(i-1));
            convolutionShaders_[i-1]->bindAttributeLocation("attributes0",0);
            if (flags_ & LayerFlags::RESIDUAL_INPUT) convolutionShaders_[i-1]->bindAttributeLocation("attributes1",1);
            convolutionShaders_[i-1]->link();
        } catch (GLException& ex) {
            FNLOGE("Cannot link shader for layer %s",getName().c_str());
            throw;
        }
        if (!convolutionShaders_.at(i-1)->isLinked()) THROW_EXCEPTION_ARGS(FynException,"Invalid shader");
        convolutionShaders_.at(i-1)->bind();
        if (outputPadding_ > 0) {
            convolutionShaders_[i-1]->mapUniformLocation("bias",BIAS);
            convolutionShaders_[i-1]->setMappedUniformVec4Array(BIAS,zeroBias_,i);    // requires bound shader
        }
        if (flags_ & LayerFlags::RESIDUAL_INPUT) {
            for (int j=0; j < i; j++) {
                char name[64];
                snprintf(name,sizeof(name),"resLayer%d",j);
                convolutionShaderStates_[i-1]->setUniformValue(name,j+1);
            }
            convolutionShaders_[i-1]->mapUniformLocation("addResidual",RESIDUAL_SWITCH);
            convolutionShaders_[i-1]->setMappedUniformValue(RESIDUAL_SWITCH,(GLint)0);   // requires bound shader
        }
        convolutionShaderStates_[i-1]->setUniformValue("inputLayer",0);
        convolutionShaders_[i-1]->mapUniformLocation("coeffs",COEFFICIENTS);
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
