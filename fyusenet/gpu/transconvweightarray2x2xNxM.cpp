//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Uniform Array for Transpose-Convolutional Layer 2x2xNxM Weights Shader
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstdio>
#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "transconvweightarray2x2xNxM.h"
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
 * @param stride Spatial stride (isotropic) for the transpose convolution
 * @param inputChannels Number of input layers for the weights
 * @param outputChannels Number of output layers for the weights
 * @param maxRenderTargets Maximum number of render targets that can be used in one output pass
 *
 * This parameterizes the weight array with basic shape data as well as information about
 * creating coefficient packages that can be uploaded to the fragment shaders that perform the
 * computation.
 */
TransConvWeightArray2x2xNxM::TransConvWeightArray2x2xNxM(int stride, int inputChannels, int outputChannels, int maxRenderTargets) : UniformWeightArray() {
    assert(stride == 2);
    kernel_ = 2;
    stride_ = stride;
    markerOffset_ = 0;
    inputChannels_ = inputChannels;
    outputChannels_ = outputChannels;
    paddedOutputChannels_ = PIXEL_PACKING * ((outputChannels + PIXEL_PACKING-1) / PIXEL_PACKING);
    paddedInputChannels_ = PIXEL_PACKING * ((inputChannels + PIXEL_PACKING-1) / PIXEL_PACKING);
    maxRenderTargets_ = maxRenderTargets;
    MRT_.resize(paddedOutputChannels_ / PIXEL_PACKING,  0);
    MRTOffsets_.resize(paddedOutputChannels_ / PIXEL_PACKING, 0);
    outputRenderPasses_ = 0;
    int mrtoffs = 0;
    int maxrt = maxRenderTargets_;
    int rem = outputChannels_;
    do {
        while (rem >= (maxrt*PIXEL_PACKING)) {
            rem -= maxrt*PIXEL_PACKING;
            MRT_[outputRenderPasses_] = maxrt;
            MRTOffsets_[outputRenderPasses_++] = mrtoffs;
            mrtoffs += maxrt;
        }
        maxrt--;
    } while (maxrt >= 1);
    if (rem > 0) {
        MRT_[outputRenderPasses_]=1;
        MRTOffsets_[outputRenderPasses_++]=mrtoffs;
    }
    inputRenderPasses_=paddedInputChannels_/PIXEL_PACKING;
    if (inputRenderPasses_==0) THROW_EXCEPTION_ARGS(FynException,"Illegal number of input layers supplied");
    packOffsets_.resize(outputRenderPasses_ * 3 * 3 * (paddedInputChannels_ / PIXEL_PACKING), 0);
    packSizes_.resize(outputRenderPasses_ * 3 * 3 * (paddedInputChannels_ / PIXEL_PACKING), 0);
}


/**
 * @copydoc UniformWeightArray::~UniformWeightArray
 */
TransConvWeightArray2x2xNxM::~TransConvWeightArray2x2xNxM() {
}


/**
 * @copydoc UniformWeightArray::numOutputRenderPasses
 */
int TransConvWeightArray2x2xNxM::numOutputRenderPasses() const {
    return outputRenderPasses_;
}


/**
 * @copydoc UniformWeightArray::numRenderTargets
 */
int TransConvWeightArray2x2xNxM::numRenderTargets(int outputPass) const {
    return MRT_[outputPass];
}


/**
 * @copydoc UniformWeightArray::numInputRenderPasses
 */
int TransConvWeightArray2x2xNxM::numInputRenderPasses() const {
    return inputRenderPasses_;
}


/**
 * @copydoc UniformWeightArray::outputTextureOffset
 */
int TransConvWeightArray2x2xNxM::outputTextureOffset(int outputPass) const {
    return MRTOffsets_[outputPass];
}


/**
 * @brief Retrieve pointer to weights for a single render target
 * @param inputPass The input batch (offset to the input layer-batch to process)
 * @param outputPass The output batch (offset to the output layer-batch to process)
 * @param xIndex The WxW filter x-index (in 0..W-1)
 * @param yIndex The WxW filter y-index (in 0..W-1)
 *
 * @return Pointer to consecutive float data which can be loaded as uniform float data to a
 *         OpenGL shader
 *
 * @note Assume an input layer batch has a depth of 16. Supplying \c 1 as \p inputPass will therefore
 *       fetch data for input-layers #16..#31. Accordingly, assume an output layer batch has a depth of 4.
 *       Supplying \c 1 as \p outputPass will therefore fetch data for output-layers #4..#7 .
 */
const float * TransConvWeightArray2x2xNxM::getPackageWeights(int inputPass, int outputPass, int xIndex, int yIndex) const {
    int stratskips[4] = {0,1,2,3};
    int stratum = xIndex+2*yIndex;
    int offset = stratskips[stratum] * (paddedOutputChannels_ / PIXEL_PACKING) * (paddedInputChannels_ / PIXEL_PACKING);
    offset += (outputPass * (paddedInputChannels_ / PIXEL_PACKING) + inputPass);
    assert(offset < markerOffset_);
    return weightData_ + packOffsets_[offset];
}


/**
 * @copydoc UniformWeightArray::getPackageBias
 */
const float * TransConvWeightArray2x2xNxM::getPackageBias(int outputPass) const {
    int offset = (MRTOffsets_[outputPass])*PIXEL_PACKING;
    return biasData_+offset;
}


/**
 * @copydoc UniformWeightArray::getPackageBNScale
 */
const float * TransConvWeightArray2x2xNxM::getPackageBNScale(int outputPass) const {
    int offset =  (MRTOffsets_[outputPass])*PIXEL_PACKING;
    return bnScale_+offset;
}


/**
 * @copydoc UniformWeightArray::extractBiasData
 */
void TransConvWeightArray2x2xNxM::extractBiasData(const float *input) {
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
void TransConvWeightArray2x2xNxM::extractBatchnormData(const float *input) {
    if (!bnBias_) bnBias_ = new float[paddedOutputChannels_];
    if (!bnScale_) bnScale_ = new float[paddedOutputChannels_];
    memset(bnBias_,0,paddedOutputChannels_*sizeof(float));
    memset(bnScale_,0,paddedOutputChannels_*sizeof(float));
    for (int i=0; i < outputChannels_;i++) {
        bnBias_[i] = input[i];
        bnScale_[i] = input[i+outputChannels_];
    }
    if (biasData_) {
        for (int i=0; i < outputChannels_;i++) {
            biasData_[i] = biasData_[i] * bnScale_[i] + bnBias_[i];
        }
    }
}


/**
 * @copydoc UniformWeightArray::getPackageSize
 */
int TransConvWeightArray2x2xNxM::getPackageSize(int inputPass, int outputPass, int xIndex, int yIndex) const {
    int stratskips[4]={0,1,2,3};
    int stratum = xIndex+2*yIndex;
    int offset = stratskips[stratum] * (paddedOutputChannels_ / PIXEL_PACKING)*(paddedInputChannels_ / PIXEL_PACKING);
    offset += (outputPass*(paddedInputChannels_ / PIXEL_PACKING)+inputPass);
    assert(offset<markerOffset_);
    return packSizes_[offset];
}


/**
 * @copydoc UniformWeightArray::extractWeightData
 */
void TransConvWeightArray2x2xNxM::extractWeightData(const float *input) {
    int fullsize = kernel_ * (kernel_*paddedOutputChannels_) * paddedInputChannels_;
    if (!weightData_) weightData_ = new float[fullsize];
    memset(weightData_, 0, fullsize*sizeof(float));

    int dstoffs = extractStratum1(input, 0, 0);
    if (dstoffs >= fullsize) {
        THROW_EXCEPTION_ARGS(FynException,"Overflow at weight array computation");
    }

    dstoffs = extractStratum2(input, 0, dstoffs);
    if (dstoffs >= fullsize) {
        THROW_EXCEPTION_ARGS(FynException,"Overflow at weight array computation");
    }

    dstoffs = extractStratum3(input, 0, dstoffs);
    if (dstoffs >= fullsize) {
        THROW_EXCEPTION_ARGS(FynException,"Overflow at weight array computation");
    }

    dstoffs = extractStratum4(input, 0, dstoffs);
    if (dstoffs >= fullsize) {
        THROW_EXCEPTION_ARGS(FynException,"Overflow at weight array computation");
    }
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Extract convolution weights for 1st stratum of transpose convolution
 *
 * @param input Pointer to weight/bias/bn data array
 * @param inputOffset Offset in the \p input array to start reading data from
 * @param dstOffset Offset in the #weightData_ array to start writing the extracted weightz data to
 *
 * @return Updated value for \p dstOffset for next stratum
 *
 * As we use the stencil buffer to perform the broadcast-type of operation that is realized by a
 * transpose convolution and have to split the computation to 4 phases, we smash a label on it and
 * call each of those overlay-type-of rendering passes a stratum. In total there are 4 of these
 * and this function collects the convolution weights for the first stratum.
 *
 * @see TransConvLayerBase
 */
int TransConvWeightArray2x2xNxM::extractStratum1(const float *input, size_t inputOffset, int dstOffset) {
    int ostride = kernel_*kernel_*inputChannels_;
    for (int opass = 0 ; opass < outputRenderPasses_ ; opass++) {
        for (int ipass=0;ipass<numInputRenderPasses();ipass++) {
            packOffsets_[markerOffset_]=dstOffset;
            int ilayer=ipass*PIXEL_PACKING;
            for (int rtarget = 0 ; rtarget < MRT_[opass] ; rtarget++) {
                int olayer = (MRTOffsets_[opass]+rtarget)*PIXEL_PACKING;
                int olimit = ((olayer+PIXEL_PACKING) > outputChannels_) ? (outputChannels_-olayer) : PIXEL_PACKING;
                int ilimit = ((inputChannels_-ilayer) >= PIXEL_PACKING) ? PIXEL_PACKING : (inputChannels_-ilayer);

                for (int l=0; l < ilimit; l++) {
                    for (int o=0; o < olimit; o++) {
                        size_t srcoffset = inputOffset + (olayer+o)*ostride + ilayer + l;
                        weightData_[dstOffset++] = input[srcoffset];
                    }
                    if (olimit < PIXEL_PACKING) dstOffset += (PIXEL_PACKING-olimit);
                }
                if (ilimit < PIXEL_PACKING) {
                    dstOffset += (PIXEL_PACKING-ilimit)*PIXEL_PACKING;
                }
            }
            packSizes_[markerOffset_] = (dstOffset - packOffsets_[markerOffset_]*sizeof(float));
            markerOffset_++;
        }
    }
    return dstOffset;
}


/**
 * @brief Extract convolution weights for 2nd stratum of transpose convolution
 *
 * @param input Pointer to weight/bias/bn data array
 * @param inputOffset Offset in the \p input array to start reading data from
 * @param dstOffset Offset in the #weightData_ array to start writing the extracted weightz data to
 *
 * @return Updated value for \p dstOffset for next stratum
 *
 * As we use the stencil buffer to perform the broadcast-type of operation that is realized by a
 * transpose convolution and have to split the computation to 4 phases, we smash a label on it and
 * call each of those overlay-type-of rendering passes a stratum. In total there are 4 of these
 * and this function collects the convolution weights for the second stratum.
 *
 * @see TransConvLayerBase
 */
int TransConvWeightArray2x2xNxM::extractStratum2(const float *input,size_t inputOffset,int dstOffset) {
    int ostride = kernel_*kernel_*inputChannels_;
    for (int opass = 0 ; opass < outputRenderPasses_ ; opass++) {
        for (int ipass=0; ipass < numInputRenderPasses(); ipass++) {
            packOffsets_[markerOffset_] = dstOffset;
            int ilayer=ipass*PIXEL_PACKING;
            for (int rtarget = 0 ; rtarget < MRT_[opass] ; rtarget++) {
                int olayer = (MRTOffsets_[opass]+rtarget)*PIXEL_PACKING;
                int olimit = ((olayer+PIXEL_PACKING) > outputChannels_) ? (outputChannels_-olayer) : PIXEL_PACKING;
                int ilimit = ((inputChannels_-ilayer) >= PIXEL_PACKING) ? PIXEL_PACKING : (inputChannels_-ilayer);
                for (int l=0; l < ilimit; l++) {
                    for (int o=0; o < olimit; o++) {
                        size_t srcoffset = inputOffset + (olayer+o)*ostride + inputChannels_ + ilayer + l;
                        weightData_[dstOffset++] = input[srcoffset];
                    }
                    if (olimit<PIXEL_PACKING) dstOffset += (PIXEL_PACKING-olimit);
                }
                if (ilimit < PIXEL_PACKING) {
                    dstOffset+=(PIXEL_PACKING-ilimit)*PIXEL_PACKING;
                }
            }
            packSizes_[markerOffset_] = (dstOffset - packOffsets_[markerOffset_]*sizeof(float));
            markerOffset_++;
        }
    }
    return dstOffset;
}


/**
 * @brief Extract convolution weights for 3rd stratum of transpose convolution
 *
 * @param input Pointer to weight/bias/bn data array
 * @param inputOffset Offset in the \p input array to start reading data from
 * @param dstOffset Offset in the #weightData_ array to start writing the extracted weightz data to
 *
 * @return Updated value for \p dstOffset for next stratum
 *
 * As we use the stencil buffer to perform the broadcast-type of operation that is realized by a
 * transpose convolution and have to split the computation to 4 phases, we smash a label on it and
 * call each of those overlay-type-of rendering passes a stratum. In total there are 4 of these
 * and this function collects the convolution weights for the third stratum.
 *
 * @see TransConvLayerBase
 */
int TransConvWeightArray2x2xNxM::extractStratum3(const float *input,size_t inputOffset,int dstOffset) {
    int ostride = kernel_*kernel_*inputChannels_;
    for (int opass = 0 ; opass < outputRenderPasses_ ; opass++) {
        for (int ipass=0; ipass < numInputRenderPasses(); ipass++) {
            packOffsets_[markerOffset_]=dstOffset;
            int ilayer = ipass*PIXEL_PACKING;
            for (int rtarget = 0 ; rtarget < MRT_[opass] ; rtarget++) {
                int olayer = (MRTOffsets_[opass]+rtarget)*PIXEL_PACKING;
                int olimit = ((olayer+PIXEL_PACKING) > outputChannels_) ? (outputChannels_-olayer) : PIXEL_PACKING;
                int ilimit = ((inputChannels_-ilayer) >= PIXEL_PACKING) ? PIXEL_PACKING : (inputChannels_-ilayer);
                for (int l=0; l < ilimit; l++) {
                    for (int o=0; o < olimit; o++) {
                        size_t srcoffset = inputOffset + (olayer+o)*ostride + 2*inputChannels_ + ilayer + l;
                        weightData_[dstOffset++] = input[srcoffset];
                    }
                    if (olimit<PIXEL_PACKING) dstOffset += (PIXEL_PACKING-olimit);
                }
                if (ilimit < PIXEL_PACKING) {
                    dstOffset += (PIXEL_PACKING-ilimit)*PIXEL_PACKING;
                }
            }
            packSizes_[markerOffset_] = (dstOffset-packOffsets_[markerOffset_]*sizeof(float));
            markerOffset_++;
        }
    }
    return dstOffset;
}


/**
 * @brief Extract convolution weights for 4th stratum of transpose convolution
 *
 * @param input Pointer to weight/bias/bn data array
 * @param inputOffset Offset in the \p input array to start reading data from
 * @param dstOffset Offset in the #weightData_ array to start writing the extracted weightz data to
 *
 * @return Updated value for \p dstOffset for next stratum
 *
 * As we use the stencil buffer to perform the broadcast-type of operation that is realized by a
 * transpose convolution and have to split the computation to 4 phases, we smash a label on it and
 * call each of those overlay-type-of rendering passes a stratum. In total there are 4 of these
 * and this function collects the convolution weights for the fourth stratum.
 *
 * @see TransConvLayerBase
 */
int TransConvWeightArray2x2xNxM::extractStratum4(const float *input,size_t inputOffset,int dstOffset) {
    int ostride = kernel_*kernel_*inputChannels_;
    for (int opass = 0 ; opass < outputRenderPasses_ ; opass++) {
        for (int ipass=0;ipass<numInputRenderPasses();ipass++) {
            packOffsets_[markerOffset_]=dstOffset;
            int ilayer=ipass*PIXEL_PACKING;
            for (int rtarget = 0 ; rtarget < MRT_[opass] ; rtarget++) {
                int olayer = (MRTOffsets_[opass]+rtarget)*PIXEL_PACKING;
                int olimit = ((olayer+PIXEL_PACKING) > outputChannels_) ? (outputChannels_-olayer) : PIXEL_PACKING;
                int ilimit = ((inputChannels_-ilayer) >= PIXEL_PACKING) ? PIXEL_PACKING : (inputChannels_-ilayer);
                for (int l=0; l < ilimit; l++) {
                    for (int o=0; o < olimit; o++) {
                        size_t srcoffset = inputOffset + (olayer+o)*ostride + 3*inputChannels_ + ilayer + l;
                        weightData_[dstOffset++] = input[srcoffset];
                    }
                    if (olimit < PIXEL_PACKING) dstOffset+=(PIXEL_PACKING-olimit);
                }
                if (ilimit < PIXEL_PACKING) {
                    dstOffset += (PIXEL_PACKING-ilimit)*PIXEL_PACKING;
                }
            }
            packSizes_[markerOffset_] = (dstOffset-packOffsets_[markerOffset_]*sizeof(float));
            markerOffset_++;
        }
    }
    return dstOffset;
}



} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
