//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Transpose Convolutional Layer w/ 3x3 mask
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/glinfo.h"
#include "../../gl/vertexshader.h"
#include "../../gl/fragmentshader.h"
#include "../transconvweightarray3x3xNxM.h"
#include "transconvlayer3x3_vanilla.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::vanilla {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&,int)
 */
TransConvLayer3x3::TransConvLayer3x3(const ConvLayerBuilder& builder, int layerNumber):TransConvLayerBase(builder, layerNumber) {
}



/**
 * @copydoc ConvLayerBase::loadParameters
 */
void TransConvLayer3x3::loadParameters(const ParameterProvider *weights) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    weights_ = new TransConvWeightArray3x3xNxM(upsample_, inputChannels_, outputChannels_, maxRenderTargets_);
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


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Setup shaders programs
 *
 * This function compiles a set of shaders that are specific to the number of render targets and
 * to the stratum index on which they are used. The shader setup here is quite specific to a 3x3
 * transposed convolution with an upsampling factor of 2.
 */
void TransConvLayer3x3::setupShaders() {
    char preproc[768] = {0};
    shaderPreprocessing(preproc, sizeof(preproc)-1);
    if (upsample_ == 2) {
        for (int i=1; i <= maxRenderTargets_; i++) {
            char fullpreproc[1024];
            snprintf(fullpreproc,sizeof(fullpreproc),"#define CONVSIZE 1\n#define NUM_LANES %d\n#define STEP 1\n",i);
            strncat(fullpreproc, preproc, sizeof(fullpreproc)-strlen(fullpreproc)-1);
            shaders_[0].push_back(compileShaderPair("shaders/vanilla/convtransNxN.vert","shaders/vanilla/convtrans3x3_stride2.frag",fullpreproc,typeid(this)));
            shaderStates_[0].push_back(configureShader(shaders_[0].at(i-1),0));

            snprintf(fullpreproc,sizeof(fullpreproc),"#define CONVSIZE 2\n#define NUM_LANES %d\n#define STEP 2\n",i);
            strncat(fullpreproc, preproc, sizeof(fullpreproc)-strlen(fullpreproc)-1);
            shaders_[1].push_back(compileShaderPair("shaders/vanilla/convtransNxN.vert","shaders/vanilla/convtrans3x3_stride2.frag",fullpreproc,typeid(this)));
            shaderStates_[1].push_back(configureShader(shaders_[1].at(i-1),1));

            snprintf(fullpreproc,sizeof(fullpreproc),"#define CONVSIZE 2\n#define NUM_LANES %d\n#define STEP 3\n",i);
            strncat(fullpreproc, preproc, sizeof(fullpreproc)-strlen(fullpreproc)-1);
            shaders_[2].push_back(compileShaderPair("shaders/vanilla/convtransNxN.vert","shaders/vanilla/convtrans3x3_stride2.frag",fullpreproc,typeid(this)));
            shaderStates_[2].push_back(configureShader(shaders_[2].at(i-1),2));

            snprintf(fullpreproc,sizeof(fullpreproc),"#define CONVSIZE 4\n#define NUM_LANES %d\n#define STEP 4\n",i);
            strncat(fullpreproc, preproc, sizeof(fullpreproc)-strlen(fullpreproc)-1);
            shaders_[3].push_back(compileShaderPair("shaders/vanilla/convtransNxN.vert","shaders/vanilla/convtrans3x3_stride2.frag",fullpreproc,typeid(this)));
            shaderStates_[3].push_back(configureShader(shaders_[3].at(i-1),3));
        }
    } else {
        THROW_EXCEPTION_ARGS(FynException,"Only stride 2 transposed convs are supported as of now");
    }
}


} // fyusion::fyusenet::gpu::vanilla namespace

// vim: set expandtab ts=4 sw=4:
