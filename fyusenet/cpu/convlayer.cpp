//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Convolution Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "convlayer.h"

namespace fyusion::fyusenet::cpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc CPULayerBase::CPULayerBase(const LayerBuilder&, int)
 *
 * @warning If \c PRE_RELU activation is used with this layer, the input data will be overwritten
 */
ConvolutionLayer::ConvolutionLayer(const ConvLayerBuilder &builder, int layerNumber):CPULayerBase((const LayerBuilder&)builder,layerNumber) {
    kernel_ = builder.kernel_;
    dilation_[0] = builder.dilation_[0];
    dilation_[1] = builder.dilation_[1];
    upsample_[0] = builder.upsample_[0];
    upsample_[1] = builder.upsample_[1];
    downsample_[0] = builder.downsample_[0];
    downsample_[1] = builder.downsample_[1];
}


/**
 * @copydoc LayerBase::~LayerBase
 */
ConvolutionLayer::~ConvolutionLayer() {
    delete [] weights_;
    delete [] bias_;
    delete [] bnScale_;
}


/**
 * @copydoc LayerBase::forward
 */
void ConvolutionLayer::forward(uint64_t sequenceNo, StateToken * state) {
    // TODO (mw) seriously painfully unoptimized code as this is only used for very small convs for now
    float * output = outputs_[0]->map<float>();
    int outwidth = (width_ / downsample_[0]) + 2*outputPadding_;
    int outheight = (height_ / downsample_[1]) + 2*outputPadding_;
    int outnetwidth = outwidth - 2*outputPadding_;
    int outnetheight = outheight - 2*outputPadding_;
    // bad code, emulates what we do on the GPU by setting the target buffer to bias values
    for (int ol=0 ; ol < outputChannels_ ; ol++) {
        float *outptr = output+outheight*outwidth*ol;
        for (int y=0; y < outputPadding_; y++) memset(output+y*outwidth,0,outwidth*sizeof(float));
        for (int y=outputPadding_ ; y < outnetheight+outputPadding_ ; y++) {
            for (int x=0 ; x < outputPadding_ ; x++) outptr[y*outwidth+x]=0;
            for (int x=outputPadding_ ; x < outnetwidth+outputPadding_ ; x++) {
                outptr[y*outwidth+x] = bias_[ol];
            }
            for (int x=outnetwidth+outputPadding_ ; x < outwidth; x++) outptr[y*outwidth+x]=0;
        }
        for (int y=outnetheight+outputPadding_ ; y < outheight ; y++) memset(output+y*outwidth,0,outwidth*sizeof(float));
    }
    float *input = inputs_.at(0)->map<float>();
    if (flags_ & LayerFlags::PRE_RELU) preReLU(input);
    if (inputPadding_ > 0) paddedConv(input, output);
    else unpaddedConv(input, output);
    if (flags_ & LayerFlags::POST_RELU) postReLU(output);
    inputs_.at(0)->unmap();
    outputs_[0]->unmap();
}


/**
 * @copydoc LayerBase::loadParameters
 */
void ConvolutionLayer::loadParameters(const ParameterProvider *weights) {
    weights_ = new float[kernel_ * kernel_ * inputChannels_ * outputChannels_];
    weights->map(getName() + std::string(".weights"), getNumber(), 0).with([&](const std::any& data) {
        if (data.has_value()) {
            memcpy(weights_, std::any_cast<const float*>(data), kernel_ * kernel_ * inputChannels_ * outputChannels_ * sizeof(float));
        }
    });
    bias_ = new float[outputChannels_];
    weights->map(getName() + std::string(".bias"), getNumber(), 1).with([&](const std::any& data) {
        if (data.has_value()) {
            memcpy(bias_, std::any_cast<const float*>(data), outputChannels_ * sizeof(float));
        }
    });
    bnScale_ = new float[outputChannels_];
    if (flags_ & LayerFlags::POST_BATCHNORM) {
        weights->map(getName() + std::string(".bn"), getNumber(), 2).with([&](const std::any& data) {
            if (data.has_value()) {
                const float * src = std::any_cast<const float*>(data);
                memcpy(bnScale_, src, outputChannels_ * sizeof(float));
                for (int i=0; i < outputChannels_; i++) bias_[i] = bias_[i] * bnScale_[i] + src[outputChannels_ + i];
            }
        });
    } else {
        for (int i=0; i < outputChannels_;i++) bnScale_[i] = 1.0f;
    }
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> ConvolutionLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> ret;
    ret.push_back(BufferSpec(0, 0, width_ + 2*inputPadding_, height_ + 2*inputPadding_,
                             BufferSpec::sizedformat::SINGLE32F, BufferSpec::genericformat::SINGLE,
                             BufferSpec::dtype::FLOAT, BufferSpec::FUNCTION_SOURCE,
                             inputChannels_).device(BufferSpec::csdevice::COMP_STOR_CPU).dataOrder(BufferSpec::order::CHANNELWISE));
    return ret;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> ConvolutionLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> ret;
    int outwidth = (upsample_[0]*width_) / downsample_[0] + 2 * outputPadding_;
    int outheight = (upsample_[1]*height_) / downsample_[1] + 2 * outputPadding_;
    ret.push_back(BufferSpec(0, 0, outwidth, outheight, BufferSpec::sizedformat::SINGLE32F,
                             BufferSpec::genericformat::SINGLE, BufferSpec::dtype::FLOAT, BufferSpec::FUNCTION_DEST,
                             outputChannels_).device(BufferSpec::csdevice::COMP_STOR_CPU).dataOrder(BufferSpec::order::CHANNELWISE));
    return ret;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Perform simple (pre) ReLU activation (in-situ)
 *
 * @param[inout] data Pointer to data that is to be "ReLUed" in-place
 *
 * @warning Overwrites the supplied \p data
 */
void ConvolutionLayer::preReLU(float *data) {
    // NOTE (mw) painfully unoptimized implementation as this is only used for very small convs for now (improve in future)
    int inwidth = width_ + 2*inputPadding_;
    int inheight = height_ + 2*inputPadding_;
    for (int ol=0; ol < outputChannels_;ol++) {
        float *ptr = data+ol*(inwidth * inheight);
        for (int y=inputPadding_; y < inheight-inputPadding_; y++) {
            for (int x=inputPadding_; x < inwidth-inputPadding_; x++) {
                if (ptr[x+y*inwidth] < 0) ptr[x+y*inwidth] = 0.0f;
            }
        }
    }
}


/**
 * @brief Perform a simple (post) ReLU activation (in-situ)
 *
 * @param[inout] data Pointer to data to be "ReLUed"
 *
 * Computes a simple ReLU activation on the supplied \p data
 */
void ConvolutionLayer::postReLU(float *data) {
    // NOTE (mw) painfully unoptimized implementation as this is only used for very small convs for now (improve in future)
    int outwidth = width_ / downsample_[0] + 2*outputPadding_;
    int outheight = height_ / downsample_[1] + 2*outputPadding_;
    for (int ol=0; ol < outputChannels_;ol++) {
        float *ptr = data+ol*(outwidth*outheight);
        for (int y=outputPadding_; y < outheight-outputPadding_; y++) {
            for (int x=outputPadding_; x < outwidth-outputPadding_; x++) {
                if (ptr[x+y*outwidth] < 0) ptr[x+y*outwidth] = 0.0f;
            }
        }
    }
}



/**
 * @brief Perform 2D spatial convolution on unpadded data
 *
 * @param input Pointer to input tensor
 * @param output Pointer to output tensor
 *
 * Performs kxk 2D convolution of the tensor data in \p input and writes the results to \p output.
 */
void ConvolutionLayer::unpaddedConv(const float *input, float *output) {
    // NOTE (mw) painfully unoptimized implementation as this is only used for very small convs for now (improve in future)
    int fstride = width_ * height_;
    int inwidth = width_ + 2*inputPadding_;
    int inheight = height_ + 2*inputPadding_;
    int fyshift = (kernel_-1) / 2;
    int fxshift = (kernel_-1) / 2;
    int outwidth = width_ / downsample_[0]+2*outputPadding_;
    int outheight = height_ / downsample_[1]+2*outputPadding_;
    for (int ol=0; ol < outputChannels_; ol++) {
        float *outptr = output+(outwidth*outheight)*ol;
        for (int il=0; il < inputChannels_; il++) {
            const float *inptr = input + (inwidth*inheight)*il;
            const float *wptr = weights_ + ol*fstride*inputChannels_+il;
            for (int y=outputPadding_,yi=0; y < outheight-outputPadding_; y++,yi+=downsample_[1]) {
                for (int x=outputPadding_,xi=0; x < outwidth-outputPadding_; x++,xi+=downsample_[0]) {
                    float out=0.0f;
                    for (int fy=0; fy < kernel_; fy++) {
                        int cy = ((fy-fyshift+y) < 0) ? 0 : fy-fyshift+y;
                        cy = (cy < height_) ? cy : height_-1;
                        for (int fx=0; fx < kernel_; fx++) {
                            int cx = ((fx-fxshift+x) < 0) ? 0 : fx-fxshift+x;
                            cx = (cx < width_) ? cx : width_-1;
                            float inpix = inptr[cx+cy*inwidth];
                            float wgt = wptr[fx*inputChannels_+fy*(inputChannels_*kernel_)];
                            out+=inpix*wgt;
                        }
                    }
                    outptr[x+y*outwidth] += out*bnScale_[ol];
                }
            }
        }
    }
}


/**
 * @brief Perform 2D spatial convolution on padded data
 *
 * @param input Pointer to input tensor
 * @param output Pointer to output tensor
 *
 * Performs kxk 2D convolution of the tensor data in \p input and writes the results to \p output.
 */
void ConvolutionLayer::paddedConv(const float *input, float *output) {
    // NOTE (mw) unoptimized implementation as this is only used for very small convs, improve in future
    int fstride = width_ * height_;
    int inwidth = width_ + 2*inputPadding_;
    int inheight = height_ + 2*inputPadding_;
    int fyshift = (kernel_-1) / 2;
    int fxshift = (kernel_-1) / 2;
    int outwidth = width_ / downsample_[0] + 2*outputPadding_;
    int outheight = height_ / downsample_[1] + 2*outputPadding_;
    for (int ol=0; ol < outputChannels_; ol++) {
        float *outptr = output+(outwidth*outheight)*ol;
        for (int il=0; il < inputChannels_; il++) {
            const float *inptr = input + (inwidth*inheight)*il+inputPadding_+inputPadding_*inwidth;
            const float *wptr = weights_ + ol*fstride*inputChannels_+il;
            for (int y=outputPadding_,yi=0; y < outheight-outputPadding_; y++,yi+=downsample_[1]) {
                for (int x=outputPadding_,xi=0; x < outwidth-outputPadding_; x++,xi+=downsample_[0]) {
                    float out = 0.0f;
                    for (int fy=0; fy < kernel_; fy++) {
                        for (int fx=0; fx < kernel_; fx++) {
                            float inpix = inptr[xi+fx-fxshift+(yi+fy-fyshift)*inwidth];
                            float wgt = wptr[fx*inputChannels_+fy*(inputChannels_*kernel_)];
                            out += inpix*wgt;
                        }
                    }
                    outptr[x+y*outwidth] += out*bnScale_[ol];
                }
            }
        }
    }
}

} // fyusion::fyusenet::cpu namespace

// vim: set expandtab ts=4 sw=4:
