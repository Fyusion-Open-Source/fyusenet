//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Depthwise Convolutional Layer w/ 3x3 mask
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../common/fynexception.h"
#include "../../gl/glexception.h"
#include "../../gl/vertexshader.h"
#include "../convweightarray_dw_KxKxNxM.h"
#include "convlayer_dw_3x3_vanilla.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::vanilla {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
DepthwiseConvLayer3x3::DepthwiseConvLayer3x3(const ConvLayerBuilder & builder, int layerNumber) : ConvLayerNxN(builder, layerNumber) {
    assert(builder.kernel_ == 3);
    channelMultiplier_ = outputChannels_ / builder.groupSize_;
    if (channelMultiplier_ != 1) THROW_EXCEPTION_ARGS(FynException,"Channel multipliers are currently not supported");
    int maxdrawbuffers = GLInfo::getMaximumDrawBuffers();
    if (channelMultiplier_ == 1) maxdrawbuffers = GLInfo::getMaximumRecommendedDrawBuffers();    // NOTE (mw) if we don't need to spread out, we use capping for Mali GPUs for example
    int maxvecs = GLInfo::getMaxUniformVectors(GLInfo::FRAGMENT);
    int biasvec = (outputPadding_) ? 1 : 0;
    int maxrt = (maxvecs-VEC_OVERHEAD)/(kernel_ * kernel_ + biasvec);
    if (maxrt>maxdrawbuffers) maxRenderTargets_ = maxdrawbuffers;
    else maxRenderTargets_ = maxrt;
    if (maxRenderTargets_ < channelMultiplier_) THROW_EXCEPTION_ARGS(FynException,"Cannot instantiate depthwise convolution layer, channelmult %d is larger than max rt %d",channelMultiplier_,maxRenderTargets_);
    int usedvecs = maxRenderTargets_*4*kernel_ + VEC_OVERHEAD + biasvec;
    int intex = 2;
    int maxtex = GLInfo::getMaximumRecommendedTextureUnits();
    while ((maxvecs >= usedvecs) && (intex <= maxtex)) {
        usedvecs = maxRenderTargets_*4*intex*kernel_ + VEC_OVERHEAD + biasvec;
        intex++;
    }
    maxInputTextures_ = intex - 1;
    if (maxInputTextures_*channelMultiplier_ > maxRenderTargets_) maxInputTextures_=maxRenderTargets_/channelMultiplier_;
    // NOTE (mw) simplifying assumptions for channelmult == 1
    maxInputTextures_ = std::min(maxInputTextures_,maxRenderTargets_);
    maxRenderTargets_ = maxInputTextures_;
}


/**
 * @copydoc ConvLayerBase::loadParameters
 */
void DepthwiseConvLayer3x3::loadParameters(const ParameterProvider *weights) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    weights_ = new DepthwiseConvWeightArrayKxKxNxM(kernel_, inputChannels_, channelMultiplier_, maxRenderTargets_, maxInputTextures_);
    weights->map(getName() + std::string(".bias"), getNumber(), 1).with([&](const std::any & data) {
        weights_->extractBiasData(std::any_cast<const float *>(data));
    });
    weights->map(getName() + std::string(".weights"), getNumber(), 0).with([&](const std::any & data) {
        weights_->extractWeightData(std::any_cast<const float *>(data));
    });
    if (flags_ & LayerFlags::POST_BATCHNORM) {
        weights->map(getName() + std::string(".bn"), getNumber(), 2).with([&](const std::any & data) {
            weights_->extractBatchnormData(std::any_cast<const float *>(data));
        });
    }
}


/**
 * @copydoc LayerBase::forward
 */
void DepthwiseConvLayer3x3::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (!valid_) THROW_EXCEPTION_ARGS(FynException,"Trying to invoke forward() on invalid layer");
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render entry: 0x%x (%s:%d)[%s]",err,__FILE__,__LINE__,getName().c_str());
#endif
    if (outputChanged_) updateFBOs();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendEquationSeparate(GL_FUNC_ADD,GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE,GL_ONE,GL_ONE,GL_ONE);
    glViewport(0,0,viewport_[0],viewport_[1]);
    ShaderProgram *shader = nullptr;
    vertexArray_->bind();
    int textureoffset = 0;
    for (int outfield = 0 ; outfield < weights_->numOutputRenderPasses(); outfield++) {
        int sindex = weights_->numRenderTargets(outfield)-1;
        if (convolutionShaders_.at(sindex).get()!=shader) {
            if (shader) shader->unbind(true);
            shader = convolutionShaders_.at(sindex).get();
            shader->bind(convolutionShaderStates_.at(sindex).get());
        }
        if (flags_ & LayerFlags::RESIDUAL_INPUT) {
            for (int i=0;i<weights_->numRenderTargets(outfield);i++) {
                int texindex = i+weights_->outputTextureOffset(outfield);
                glActiveTexture(GL_TEXTURE8+i);
                glBindTexture(GL_TEXTURE_2D,residualTextures_.at(texindex));
            }
        }
        int numcvecs = kernel_ * kernel_ * weights_->numRenderTargets(outfield);
        framebuffers_.at(outfield)->bind();
        framebuffers_.at(outfield)->setWriteMask();
        setBias(outfield,weights_);
        if (flags_ & LayerFlags::POST_BATCHNORM) {
            shader->setMappedUniformVec4Array(BATCHNORM_DATA,weights_->getPackageBNScale(outfield),weights_->numRenderTargets(outfield));
        }
        for (int infield = 0; infield < weights_->numInputRenderPasses(); infield++) {
            for (int t=0; t < weights_->numRenderTargets(outfield) ; t++) {
                glActiveTexture(GL_TEXTURE0+t);
                glBindTexture(GL_TEXTURE_2D,inputTextures_.at(t+textureoffset));
            }

            const float *cvecs = weights_->getPackageWeights(infield,outfield,0,0);
            shader->setMappedUniformVec4Array(COEFFICIENTS,cvecs,numcvecs);
            if (flags_ & LayerFlags::RESIDUAL_INPUT) shader->setMappedUniformValue(RESIDUAL_SWITCH,(GLint)1);
            if (outputPadding_ > 0) {
                shader->setMappedUniformVec4Array(BIAS,weights_->getPackageBias(outfield),weights_->numRenderTargets(outfield));
            }
            glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,(const GLvoid *)nullptr);
            if (outputPadding_ > 0) {
                shader->setMappedUniformVec4Array(BIAS,zeroBias_,weights_->numRenderTargets(outfield));
            }
            if (flags_ & LayerFlags::RESIDUAL_INPUT) shader->setMappedUniformValue(RESIDUAL_SWITCH,(GLint)0);
        }
        textureoffset += weights_->numRenderTargets(outfield);
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
 * @copydoc ConvLayerBase::setBias
 */
void DepthwiseConvLayer3x3::setBias(int outPass,const UniformWeightArray *bias) {
    if (outputPadding_ > 0) {
        glClearColor(0.0f,0.0f,0.0f,0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    } else {
        const float *data = bias->getPackageBias(outPass);
        for (int i=0; i < bias->numRenderTargets(outPass);i++) {
            glClearBufferfv(GL_COLOR,i,data + i*PIXEL_PACKING);
        }
    }
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
void DepthwiseConvLayer3x3::compileConvolutionShaders(const char *preproc) {
    // NOTE (mw) some simplifying assumptions about the relationship of MRT and MIT#
    assert(inputChannels_ == outputChannels_);
    for (int i=1; i <= maxRenderTargets_; i++) {
        programptr shader = compileSingleShader(i, i, preproc);
        convolutionShaders_[i-1] = shader;
        convolutionShaderStates_[i-1] = UniformState::makeShared(convolutionShaders_.at(i-1));
    }
}


/**
 * @brief Compile single convolution shader
 *
 * @param outputLanes Number of output "lanes", one lane being 4 channels, relating to the number of
 *                    multi-render targets
 * @param inputLanes Number of input "lanes", one lane being 4 channels, relating to the number of
 *                   input textures
 * @param preproc Existing preprocessor string
 *
 * @return Shared pointer to compiled shader program
 */
programptr DepthwiseConvLayer3x3::compileSingleShader(int outputLanes, int inputLanes, const char *preproc) {
#if defined(WIN32) || defined(WIN64)
        using ssize_t = int64_t;
#endif
    char finalpreproc[1024+512]={0}, extra[128];
    strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
    ssize_t mc = (ssize_t)(sizeof(finalpreproc) - strlen(finalpreproc) - 1);
    snprintf(extra, sizeof(extra), "#define NUM_LANES %d\n",outputLanes);
    strncat(finalpreproc, extra, mc);
    mc -= (ssize_t)strlen(extra);
    assert(mc > 0);
    snprintf(extra,sizeof(extra),"#define NUM_INPUT_LANES %d\n",inputLanes);
    strncat(finalpreproc, extra, mc);
    mc -= (ssize_t)strlen(extra);
    assert(mc > 0);
    snprintf(extra,sizeof(extra),"#define CHANNEL_MULTIPLIER %d\n",channelMultiplier_);
    strncat(finalpreproc, extra, mc);
    mc -= (ssize_t)strlen(extra);
    assert(mc > 0);
    programptr shader = compileShaderPair("shaders/vanilla/convdefault.vert","shaders/vanilla/conv_dw_3x3.frag",finalpreproc,typeid(this));
    try {
        shader->bindAttributeLocation("attributes0",0);
        if (flags_ & LayerFlags::RESIDUAL_INPUT) shader->bindAttributeLocation("attributes1",0);
        shader->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    shader->bind();
    if (outputPadding_ > 0) {
        shader->mapUniformLocation("bias",BIAS);
        shader->setMappedUniformVec4Array(BIAS,zeroBias_,outputLanes);
    }
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        for (int j=0;j<outputLanes;j++) {
            char name[64];
            snprintf(name,sizeof(name),"resLayer%d",j);
            shader->setUniformValue(name,j+1);
        }
        shader->mapUniformLocation("addResidual",RESIDUAL_SWITCH);
        shader->setMappedUniformValue(RESIDUAL_SWITCH,(GLint)0);
    }
    for (int i=0; i < inputLanes; i++) {
        char iname[64];
        snprintf(iname,sizeof(iname),"inputLayer%d", i);
        shader->setUniformValue(iname, i);
    }
    shader->mapUniformLocation("coeffs",COEFFICIENTS);
    if (flags_ & LayerFlags::POST_BATCHNORM) shader->mapUniformLocation("batchnorm",BATCHNORM_DATA);
    shader->unbind();
    return shader;
}


/**
 * @copydoc ConvLayerBase::setupNetworkPolygons
 */
void DepthwiseConvLayer3x3::setupNetworkPolygons(VAO *vao, int kernel) {
    ConvLayerBase::setupNetworkPolygons(vao,1);
}


} // fyusion::fyusenet:gpu::vanilla namespace

// vim: set expandtab ts=4 sw=4:
