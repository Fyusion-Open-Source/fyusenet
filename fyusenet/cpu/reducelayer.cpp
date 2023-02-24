//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Reduce Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cmath>

//-------------------------------------- Project  Headers ------------------------------------------

#include "reducelayer.h"

namespace fyusion {
namespace fyusenet {
namespace cpu {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc LayerBase::LayerBase
 */
ReduceLayer::ReduceLayer(const ReduceLayerBuilder &builder, int layerNumber):CPULayerBase((const LayerBuilder&)builder,layerNumber) {
    norm_ = builder.norm_;
}


/**
 * @copydoc LayerBase::~LayerBase
 */
ReduceLayer::~ReduceLayer() {
}


/**
 * @copydoc LayerBase::forward
 */
void ReduceLayer::forward(uint64_t sequence) {
    float * output = outputs_.at(0)->map<float>();
    const float * input = inputs_.at(0)->map<float>();
    switch (norm_) {
        case ReduceLayerBuilder::NORM_L1:
            reduceL1AcrossChannels(input, output);
            break;
        case ReduceLayerBuilder::NORM_L2:
            reduceL2AcrossChannels(input, output);
            break;
    }
    outputs_.at(0)->unmap();
    inputs_.at(0)->unmap();
}



/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> ReduceLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> ret;
    ret.push_back(BufferSpec(0, 0, width_ + 2*inputPadding_, height_ + 2*inputPadding_,
                             BufferSpec::SINGLE32F, BufferSpec::SINGLE, BufferSpec::FLOAT,
                             BufferSpec::FUNCTION_SOURCE,
                             inputChannels_).device(BufferSpec::COMP_STOR_CPU).dataOrder(BufferSpec::order::CHANNELWISE));
    return ret;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> ReduceLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> ret;
    ret.push_back(BufferSpec(0,0,width_+2*outputPadding_, height_+2*outputPadding_,
                             BufferSpec::SINGLE32F, BufferSpec::SINGLE, BufferSpec::FLOAT,
                             BufferSpec::FUNCTION_DEST,
                             outputChannels_).device(BufferSpec::COMP_STOR_CPU).dataOrder(BufferSpec::order::CHANNELWISE));
    return ret;
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Compute L1 norm for provided tensor across channel dimension
 *
 * @param input Pointer to input tensor data
 * @param output Pointer to output tensor data (flattened across channel dimension)
 *
 * This computes the L1 norm of the supplied \p input tensor by treating each element in the
 * spatial domain as vector, spanning the channel dimension. The result will be a tensor withe
 * the same spatial dimensions and a depth of one channel.
 */
void ReduceLayer::reduceL1AcrossChannels(const float *input, float *output) {
    // NOTE (mw) unoptimized implementation, used for small tensors only anyway
    int inchanstride = (width_ + 2*inputPadding_) * (height_ + 2*inputPadding_);
    int outstride = width_ + 2*outputPadding_;
    for (int y=inputPadding_, yo=outputPadding_; y < height_+ 2 *inputPadding_; y++, yo++) {
        for (int x=inputPadding_,xo=outputPadding_; x < width_+ 2 *inputPadding_; x++, xo++) {
            float accu = 0.0f;
            const float *in = input + x + (y * (width_+2*inputPadding_));
            for (int l=0; l < inputChannels_; l++) {
                accu += fabsf(*in);
                in += inchanstride;
            }
            output[xo+yo*outstride]=accu;
        }
    }
}


/**
 * @brief Compute L2 norm for provided tensor across channel dimension
 *
 * @param input Pointer to input tensor data
 * @param output Pointer to output tensor data (flattened across channel dimension)
 *
 * This computes the L2 norm of the supplied \p input tensor by treating each element in the
 * spatial domain as vector, spanning the channel dimension. The result will be a tensor withe
 * the same spatial dimensions and a depth of one channel.
 */
void ReduceLayer::reduceL2AcrossChannels(const float *input, float *output) {
    // NOTE (mw) unoptimized implementation, used for small tensors only anyway
    int inchanstride = (width_ + 2*inputPadding_) * (height_ + 2*inputPadding_);
    int outstride = width_ + 2*outputPadding_;
    for (int y=inputPadding_,yo=outputPadding_; y < height_ + 2*inputPadding_; y++,yo++) {
        for (int x=inputPadding_,xo=outputPadding_; x < width_ + 2*inputPadding_; x++,xo++) {
            float accu = 0.0f;
            const float *in = input+x+(y*(width_+2*inputPadding_));
            for (int l=0;l<inputChannels_;l++) {
                accu += (*in)*(*in);
                in += inchanstride;
            }
            output[xo+yo*outstride]=accu;
        }
    }
}


} // cpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
