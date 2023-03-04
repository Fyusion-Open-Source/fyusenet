//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Uniform Array for Depthwise Convolutional Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstdio>
#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "convweightarray_dw_KxKxNxM.h"
#include "../common/fynexception.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param kernel Spatial kernel size for the convolution, must be identical in both spatial dimensions
 * @param inputChannels Number of input layers for the weights
 * @param channelMultiplier Multiplier to duplicate input channels, defaults to 1
 * @param maxRenderTargets Maximum number of render targets that can be used in one output pass
 * @param maxTextures Maximum number of input textures (currently not used)
 *
 * Constructs empty weight array for depthwise convolution layers. The number of output channels is
 * dependent on the number of input channels and the supplied \p channelMultiplier.
 *
 * @note This array makes assumptions about the relationship between the channel multiplier,
 *       the number of render targets and the maximum number of texture units. Multipliers > 1
 *       are not really tested, so handle with care.
 */
DepthwiseConvWeightArrayKxKxNxM::DepthwiseConvWeightArrayKxKxNxM(int kernel, int inputChannels, int channelMultiplier, int maxRenderTargets, int maxTextures):UniformWeightArray() {
    if (channelMultiplier > 1) {
        if (inputChannels & 3) THROW_EXCEPTION_ARGS(FynException,"Channel multipliers > 1 are only supported on input channels being a multiple of 4");
    }
    kernel_ = kernel;
    channelMultiplier_ = channelMultiplier;
    inputChannels_ = inputChannels;
    outputChannels_ = inputChannels*channelMultiplier;
    maxRenderTargets_ = maxRenderTargets;
    maxTextures_ = maxTextures;
    if (inputChannels_&(PIXEL_PACKING-1)) paddedInputChannels_ = inputChannels_+(PIXEL_PACKING-(inputChannels_&(PIXEL_PACKING-1)));
    else paddedInputChannels_=inputChannels_;
    if (outputChannels_&(PIXEL_PACKING-1)) paddedOutputChannels_ = outputChannels_+(PIXEL_PACKING-(outputChannels_&(PIXEL_PACKING-1)));
    else paddedOutputChannels_=outputChannels_;
    //-----------------------------------------
    // First compute # of output render passes
    // for channel multiplier == 1
    //-----------------------------------------
    outputRenderPasses_=0;
    int mrtoffs=0;
    int rem= inputChannels_;
    int maxrt = maxRenderTargets_;
    do {
        while (rem >= (maxrt*PIXEL_PACKING)) {
            rem -= maxrt*PIXEL_PACKING;
            MRT_.push_back(maxrt);
            MRTChannels_.push_back(0);
            MRTOffsets_.push_back(mrtoffs);
            outputRenderPasses_++;
            mrtoffs += maxrt;
        }
        maxrt--;
    } while (maxrt >= 1);
    if (rem > 0) {
        MRT_.push_back(1);
        MRTOffsets_.push_back(mrtoffs);
    }
    //-----------------------------------------
    // If we have a channel-multiplier > 1,
    // handle that here...
    //-----------------------------------------
    int chan1passes = outputRenderPasses_;
    for (int m=1; m < channelMultiplier; m++) {
        for (int i=0; i < chan1passes; i++) {
            MRT_.push_back(MRT_[i]);
            MRTChannels_.push_back(m);
            MRTOffsets_.push_back(mrtoffs);
            mrtoffs += MRT_[i];
        }
    }
    packOffsets_ = new unsigned int[outputRenderPasses_];
    packSizes_ = new unsigned int[outputRenderPasses_];
}


/**
 * @copydoc UniformWeightArray::~UniformWeightArray
 */
DepthwiseConvWeightArrayKxKxNxM::~DepthwiseConvWeightArrayKxKxNxM() {
    if (packOffsets_) delete [] packOffsets_;
    if (packOffsets_) delete [] packSizes_;
}


/**
 * @copydoc UniformWeightArray::numInputRenderPasses
 */
int DepthwiseConvWeightArrayKxKxNxM::numInputRenderPasses() const {
    return 1;
}



/**
 * @brief Retrieve number of render passes required for computing all output layers
 *
 * @return Number of required render passes
 */
int DepthwiseConvWeightArrayKxKxNxM::numOutputRenderPasses() const {
    return outputRenderPasses_;
}


/**
 * @brief Retrieve number of render-targets for a given output render pass
 *
 * @param outputPass The output render pass that should be processed
 *
 * @return Number of render-targets for the supplied \p outputPass
 */
int DepthwiseConvWeightArrayKxKxNxM::numRenderTargets(int outputPass) const {
    return MRT_[outputPass];
}


/**
 * @copydoc UniformWeightArray::outputTextureOffset
 */
int DepthwiseConvWeightArrayKxKxNxM::outputTextureOffset(int outputPass) const {
    return MRTOffsets_[outputPass];
}


/**
 * @copydoc UniformWeightArray::getPackageWeights
 */
const float * DepthwiseConvWeightArrayKxKxNxM::getPackageWeights(int inputPass, int outputPass, int xIndex, int yIndex) const {
    return weightData_ + packOffsets_[outputPass];
}


/**
 * @copydoc UniformWeightArray::getPackageBias
 */
const float * DepthwiseConvWeightArrayKxKxNxM::getPackageBias(int outputPass) const {
    int offset=(MRTOffsets_[outputPass])*PIXEL_PACKING;
    return biasData_ + offset;
}

/**
 * @copydoc UniformWeightArray::getPackageBNScale
 */
const float * DepthwiseConvWeightArrayKxKxNxM::getPackageBNScale(int outputPass) const {
    int offset=(MRTOffsets_[outputPass])*PIXEL_PACKING;
    return bnScale_ + offset;
}

/**
 * @copydoc UniformWeightArray::extractBiasData
 */
void DepthwiseConvWeightArrayKxKxNxM::extractBiasData(const float *input, size_t offset) {
    if (!biasData_) biasData_ = new float[paddedOutputChannels_];
    memset(biasData_,0,paddedOutputChannels_*sizeof(float));
    for (int i=0;i<outputChannels_;i++) {
        biasData_[i] = input[i+offset];
    }
    if (bnBias_ && bnScale_) {
        for (int i=0;i<outputChannels_;i++) {
            biasData_[i] = biasData_[i] * bnScale_[i] + bnBias_[i];
        }
    }
}


/**
 * @copydoc UniformWeightArray::extractBatchnormData
 */
void DepthwiseConvWeightArrayKxKxNxM::extractBatchnormData(const float *input, size_t offset) {
    if (!bnBias_) bnBias_ = new float[paddedOutputChannels_];
    if (!bnScale_) bnScale_ = new float[paddedOutputChannels_];
    memset(bnBias_,0,paddedOutputChannels_*sizeof(float));
    memset(bnScale_,0,paddedOutputChannels_*sizeof(float));
    for (int i=0;i<outputChannels_;i++) {
        bnScale_[i] = input[i+offset];
        bnBias_[i] = input[i+offset+outputChannels_];
    }
    if (biasData_) {
        for (int i=0;i<outputChannels_;i++) {
            biasData_[i] = biasData_[i] * bnScale_[i] + bnBias_[i];
        }
    }
}


/**
 * @copydoc UniformWeightArray::getPackageSize
 */
int DepthwiseConvWeightArrayKxKxNxM::getPackageSize(int inputPass, int outputPass, int xIndex, int yIndex) const {
    return packSizes_[outputPass];
}


/**
 * @copydoc UniformWeightArray::extractWeightData
 */
void DepthwiseConvWeightArrayKxKxNxM::extractWeightData(const float *input, size_t offset) {
    int fullsize = kernel_*(kernel_*paddedInputChannels_)*channelMultiplier_;
    if (!weightData_) weightData_ = new float[fullsize];
    memset(weightData_,0,fullsize*sizeof(float));
    int dstoffs=0;
    int markeroffset=0;
    for (int opass = 0 ; opass < outputRenderPasses_ ; opass++) {
        if (dstoffs >= fullsize) {
            THROW_EXCEPTION_ARGS(FynException,"Overflow at weight array computation");
        }
        packOffsets_[markeroffset]=dstoffs;
        // this computes the set for one shader pass
        for (int rtarget = 0; rtarget < MRT_[opass] ; rtarget++) {
            int ichan = ((MRTOffsets_[opass]+rtarget)/channelMultiplier_) * PIXEL_PACKING;
            int ilimit = ((ichan+PIXEL_PACKING) > inputChannels_) ? (inputChannels_-ichan) : PIXEL_PACKING;
            int mult = MRTChannels_[opass];
            // this computes one KxK convolution for 4 output channels (=one 4-component pixel)
            for (int yi=0; yi < kernel_; yi++) {
                for (int xi=0; xi < kernel_; xi++) {
                    for (int chan=ichan; chan < ichan+ilimit; chan++) {
                        int srcoffset = offset + chan * kernel_ * kernel_ * channelMultiplier_+(yi*kernel_+xi)*channelMultiplier_+mult;
                        weightData_[dstoffs++] = input[srcoffset];
                    }
                    if (ilimit < PIXEL_PACKING) dstoffs += (PIXEL_PACKING-ilimit);
                }
            }
        }
        packSizes_[markeroffset]=(dstoffs-packOffsets_[markeroffset])*sizeof(float);
        markeroffset++;
    } // output passes
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
