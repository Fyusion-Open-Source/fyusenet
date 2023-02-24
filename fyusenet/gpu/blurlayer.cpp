//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Blurring Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cmath>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/glexception.h"
#include "../gl/glinfo.h"
#include "../common/logging.h"
#include "blurlayer.h"

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
 *
 * @throws FynException in case the kernel size supplied in the \p builder is not supported
 */
BlurLayer::BlurLayer(const BlurLayerBuilder & builder, int layerNumber) : FunctionLayer((const GPULayerBuilder &)builder, layerNumber) {
    if ((builder.kernel_ & 1) == 0) THROW_EXCEPTION_ARGS(FynException,"This layer only supports odd kernel sizes");
    kernelWeights_ = nullptr;
    blurType_ = builder.blurType_;
    kernelSize_ = builder.kernel_;
}


/**
 * @copydoc GPULayerBase::~GPULayerBase
 */
BlurLayer::~BlurLayer() {
    delete [] kernelWeights_;
    kernelWeights_ = nullptr;
}


/**
 * @copydoc FunctionLayer::cleanup
 */
void BlurLayer::cleanup() {
    for (int i=0; i < FBO::MAX_DRAWBUFFERS; i++) {
        shaders_[i].reset();
        shaderStates_[i].reset();
    }
    currentShader_ = nullptr;
    FunctionLayer::cleanup();
}


/**
 * @copydoc FunctionLayer::setup
 */
void BlurLayer::setup() {
    if (blurType_ == BlurKernelType::AVERAGE) computeAverageWeights();
    else computeGaussianWeights();
    FunctionLayer::setup();
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/



/**
 * @brief Precompute weights for Gaussian blur
 */
void BlurLayer::computeGaussianWeights() {
    float gauss1d[kernelSize_];
    float fac = 1.0/sqrt((2.0*M_PI));
    for (int i=0; i < kernelSize_; i++) {
        float x = (float)(i-(kernelSize_-1)/2);
        gauss1d[i] = fac*expf(-(x*x));
    }
    if (!kernelWeights_) kernelWeights_ = new float[kernelSize_*kernelSize_*PIXEL_PACKING];
    float denom=0.0f;
    for (int y=0; y <kernelSize_; y++) {
        for (int x=0; x < kernelSize_; x++) {
            float val = gauss1d[y]*gauss1d[x];
            for (int i=0; i < PIXEL_PACKING; i++) kernelWeights_[(y*kernelSize_+x)*PIXEL_PACKING+i]=val;
            denom+=val;
        }
    }
    for (int i=0; i< kernelSize_*kernelSize_*PIXEL_PACKING; i++) kernelWeights_[i]/=denom;
}


/**
 * @brief Precompute weights for box-filtering (average blur)
 */
void BlurLayer::computeAverageWeights() {
    if (!kernelWeights_) kernelWeights_ = new float[kernelSize_*kernelSize_*PIXEL_PACKING];
    float coeff = 1.0f/(float)(kernelSize_*kernelSize_);
    for (int i=0; i < kernelSize_*kernelSize_*PIXEL_PACKING; i++) kernelWeights_[i]=coeff;
}


/**
 * @copydoc FunctionLayer::renderChannelBatch
 */
void BlurLayer::renderChannelBatch(int outPass, int numRenderTargets, int texOffset) {
    for (int tex=0; tex < numRenderTargets; tex++) {
        glActiveTexture(GL_TEXTURE0+tex);
        glBindTexture(GL_TEXTURE_2D,inputTextures_.at(tex+texOffset));
    }
    if (currentShader_ != shaders_[numRenderTargets-1].get()) {
        if (currentShader_) currentShader_->unbind(true);
        currentShader_ = shaders_[numRenderTargets-1].get();
        currentShader_->bind(shaderStates_[numRenderTargets-1].get());
        currentShader_->setMappedUniformVec4Array(SHADER_WEIGHTS,kernelWeights_,kernelSize_*kernelSize_);
    }
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const GLvoid *)0);
}


/**
 * @copydoc FunctionLayer::setupShaders
 */
void BlurLayer::setupShaders() {
    char preproc[1024] = {0};
    // TODO (mw) use preproc part and check shader if it supports activation
    for (int i=1; i <= maxRenderTargets_; i++) {
        snprintf(preproc,sizeof(preproc),"#define NUM_LANES %d\n#define KERNEL_SIZE %d\n", i, kernelSize_);
        handlePreprocFlags(flags_, preproc, sizeof(preproc)-strlen(preproc)-1);
        shaders_[i-1] = compileShader(preproc);
        shaders_[i-1]->mapUniformLocation("kernelCoeffs", SHADER_WEIGHTS);
        unistateptr state = UniformState::makeShared(shaders_[i-1]);
        for (int j=0; j < i; j++) {
            char var[80];
            snprintf(var,sizeof(var),"inputLayer%d", j);
            state->setUniformValue(var, j);
        }
        shaderStates_[i-1] = state;
    }
}



/**
 * @brief Compile blur kernel shader using supplied preprocessor definitions
 *
 * @param preproc Preprocessor definitions to use
 *
 * @return Shared pointer to compiled and linked shader program
 */
programptr BlurLayer::compileShader(const char *preproc) {
    programptr shader = compileShaderPair("shaders/default.vert",
                                          "shaders/generickernel.frag",
                                          preproc, typeid(this));
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
 * @copydoc FunctionLayer::beforeRender
 */
void BlurLayer::beforeRender() {
    currentShader_ = nullptr;
}


/**
 * @copydoc FunctionLayer::afterRender
 */
void BlurLayer::afterRender() {
    if (currentShader_) currentShader_->unbind();
    currentShader_ = nullptr;
}


} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
