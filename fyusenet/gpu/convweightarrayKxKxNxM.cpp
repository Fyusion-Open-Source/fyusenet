//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Uniform Array for Convolutional Layer KxKxNxM Weights Shader
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "convweightarrayKxKxNxM.h"

namespace fyusion::fyusenet::gpu {

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
 * @param outputChannels Number of output layers for the weights
 * @param maxRenderTargets Maximum number of render targets that can be used in one output pass
 *
 * This parameterizes the weight array with basic shape data as well as information about
 * creating coefficient packages that can be uploaded to the fragment shaders that perform the
 * computation.
 */
ConvWeightArrayKxKxNxM::ConvWeightArrayKxKxNxM(int kernel, int inputChannels, int outputChannels, int maxRenderTargets) : UniformWeightArray() {
    kernel_ = kernel;
    groupSize_ = 1;
    inputChannels_ = inputChannels;
    outputChannels_ = outputChannels;
    maxRenderTargets_ = maxRenderTargets;
    int rem = outputChannels_;
    if (outputChannels_ & (PIXEL_PACKING-1)) paddedOutputChannels_ = outputChannels_ + (PIXEL_PACKING - (outputChannels_ & (PIXEL_PACKING-1)));
    else paddedOutputChannels_ = outputChannels_;
    if (inputChannels_ & (PIXEL_PACKING-1)) paddedInputChannels_ = inputChannels_ + (PIXEL_PACKING - (inputChannels_ & (PIXEL_PACKING-1)));
    else paddedInputChannels_ = inputChannels_;
    outputRenderPasses_ = 0;
    int mrtoffs = 0;
    int maxrt = maxRenderTargets_;
    do {
        while (rem >= (maxrt * PIXEL_PACKING)) {
            rem -= maxrt*PIXEL_PACKING;
            MRT_.push_back(maxrt);
            MRTOffsets_.push_back(mrtoffs);
            outputRenderPasses_++;
            mrtoffs += maxrt;
        }
        maxrt--;
    } while (maxrt >= 1);
    if (rem > 0) {
        MRT_.push_back(1);
        MRTOffsets_.push_back(mrtoffs);
        outputRenderPasses_++;
    }
    inputRenderPasses_ = paddedInputChannels_ / PIXEL_PACKING;
    if (inputRenderPasses_ == 0) THROW_EXCEPTION_ARGS(FynException,"Illegal number of input layers supplied");
    packOffsets_ = new unsigned int[outputRenderPasses_ * kernel_ * (paddedInputChannels_ / PIXEL_PACKING)];
    packSizes_ = new unsigned int[outputRenderPasses_ * kernel_ * (paddedInputChannels_ / PIXEL_PACKING)];
}


/**
 * @copydoc UniformWeightArray::~UniformWeightArray
 */
ConvWeightArrayKxKxNxM::~ConvWeightArrayKxKxNxM() {
    if (packOffsets_) delete [] packOffsets_;
    if (packOffsets_) delete [] packSizes_;
}


/**
 * @copydoc UniformWeightArray::numInputRenderPasses
 */
int ConvWeightArrayKxKxNxM::numInputRenderPasses() const {
    return inputRenderPasses_;
}


/**
 * @copydoc UniformWeightArray::numOutputRenderPasses
 */
int ConvWeightArrayKxKxNxM::numOutputRenderPasses() const {
    return outputRenderPasses_;
}


/**
 * @copydoc UniformWeightArray::numRenderTargets
 */
int ConvWeightArrayKxKxNxM::numRenderTargets(int outputPass) const {
    return MRT_[outputPass];
}


/**
 * @copydoc UniformWeightArray::outputTextureOffset
 */
int ConvWeightArrayKxKxNxM::outputTextureOffset(int outputPass) const {
    return MRTOffsets_[outputPass];
}


/**
 * @copydoc UniformWeightArray::getPackageWeights
 */
const float * ConvWeightArrayKxKxNxM::getPackageWeights(int inputPass, int outputPass, int xIndex, int yIndex) const {
    int offset = (outputPass*(paddedInputChannels_ / PIXEL_PACKING)+inputPass)*kernel_ + yIndex;
    return weightData_ + packOffsets_[offset];
}


/**
 * @copydoc UniformWeightArray::getPackageBias
 */
const float * ConvWeightArrayKxKxNxM::getPackageBias(int outputPass) const {
    int offset = MRTOffsets_[outputPass]*PIXEL_PACKING;
    return biasData_ + offset;
}

/**
 * @copydoc UniformWeightArray::getPackageBNScale
 */
const float * ConvWeightArrayKxKxNxM::getPackageBNScale(int outputPass) const {
    int offset = MRTOffsets_[outputPass] * PIXEL_PACKING;
    return bnScale_ + offset;
}

/**
 * @copydoc UniformWeightArray::extractBiasData
 */
void ConvWeightArrayKxKxNxM::extractBiasData(const float *input) {
    if (!biasData_) biasData_ = new float[paddedOutputChannels_];
    memset(biasData_, 0, paddedOutputChannels_*sizeof(float));
    for (int i=0; i < outputChannels_; i++) {
        biasData_[i] = input[i];
    }
    if (bnBias_ && bnScale_) {
        for (int i=0; i < outputChannels_; i++) {
            biasData_[i] = biasData_[i] * bnScale_[i] + bnBias_[i];
        }
    }
}

/**
 * @copydoc UniformWeightArray::extractBatchnormData
 */
void ConvWeightArrayKxKxNxM::extractBatchnormData(const float *input) {
    if (!bnBias_) bnBias_ = new float[paddedOutputChannels_];
    if (!bnScale_) bnScale_ = new float[paddedOutputChannels_];
    memset(bnBias_, 0, paddedOutputChannels_ * sizeof(float));
    memset(bnScale_, 0, paddedOutputChannels_ * sizeof(float));
    for (int i=0; i < outputChannels_; i++) {
        bnScale_[i] = input[i];
        bnBias_[i] = input[i + outputChannels_];
    }
    if (biasData_) {
        for (int i=0; i < outputChannels_; i++) {
            biasData_[i] = biasData_[i] * bnScale_[i] + bnBias_[i];
        }
    }
}


/**
 * @copydoc UniformWeightArray::getPackageSize
 */
int ConvWeightArrayKxKxNxM::getPackageSize(int inputPass, int outputPass, int xIndex, int yIndex) const {
    int offset = (outputPass*(paddedInputChannels_/PIXEL_PACKING)+inputPass) * kernel_ + yIndex;
    return packSizes_[offset];
}


/**
 * @copydoc UniformWeightArray::extractWeightData
 */
void ConvWeightArrayKxKxNxM::extractWeightData(const float *input) {
    assert(packOffsets_);
    assert(packSizes_);
    int fullsize = kernel_ * (kernel_  *paddedOutputChannels_) * paddedInputChannels_;
    if (!weightData_) weightData_ = new float[fullsize];
    memset(weightData_,0,fullsize * sizeof(float));
    int dstoffs=0;
    int markeroffset=0;
    for (int opass = 0 ; opass < outputRenderPasses_ ; opass++) {
        if (dstoffs >= fullsize) {
            THROW_EXCEPTION_ARGS(FynException,"Overflow at weight array computation");
        }
        for (int ipass=0; ipass < numInputRenderPasses(); ipass++) {
            int ilayer = ipass*PIXEL_PACKING;
            int ilimit = ((inputChannels_-ilayer) >= PIXEL_PACKING) ? PIXEL_PACKING : (inputChannels_-ilayer);
            for (int yi=0; yi < kernel_; yi++) {
                packOffsets_[markeroffset]=dstoffs;
                // this computes the set for one shader pass
                for (int rtarget = 0; rtarget < MRT_[opass] ; rtarget ++) {
                    int olayer = (MRTOffsets_[opass]+rtarget)*PIXEL_PACKING;
                    int olimit = ((olayer + PIXEL_PACKING) > outputChannels_) ? (outputChannels_ - olayer) : PIXEL_PACKING;
                    for (int xi=0; xi < kernel_; xi++) {
                        // this computes one 4x4 matrix
                        for (int l=0; l < ilimit; l++) {
                            for (int o=0; o < olimit; o++) {
                                int srcoffset = (olayer+o)*kernel_*kernel_*inputChannels_+((yi*kernel_+xi)*inputChannels_)+ilayer+l;
                                weightData_[dstoffs++] = input[srcoffset];
                            }
                            if (olimit < PIXEL_PACKING) dstoffs += (PIXEL_PACKING-olimit);
                        }
                        if (ilimit < PIXEL_PACKING) {
                            dstoffs += (PIXEL_PACKING-ilimit) * PIXEL_PACKING;
                        }
                    }
                }
                packSizes_[markeroffset] = (dstoffs - packOffsets_[markeroffset])*sizeof(float);
                markeroffset++;
            } // y-index
        } // input passes
    } // output passes
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


} // fyusion::fyusenet::gpu namespace


// vim: set expandtab ts=4 sw=4:
