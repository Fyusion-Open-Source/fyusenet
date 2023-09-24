//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Base class for misc. layer testing
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "layertestbase.h"
#include <fyusenet/gpu/concatlayer.h>
#include <fyusenet/gpu/deep/deepconcatlayer.h>

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @brief Create array of convolution coefficients and a single bias
 *
 * @param bias Bias value for all channels
 * @param channelData Single KxK convolution kernel, will be replicated for every channel
 * @param kernelX Kernel size (K) of the kernel data in \p channelData (horizontal dimension)
 * @param kernelY Kernel size (K) of the kernel data in \p channelData (vertical dimension)
 * @param inputChannels Number of input channels for the convolution
 * @param outputChannels Number of output channels for the convolution
 *
 * @return Pointer to weight and bias array for the convolution, ownership is passed to the caller
 *
 * This function generates a convolution tensor by stacking the supplied \p channelData for each channel
 * and then adding the supplied bias (also to each channel).
 */
float * LayerTestBase::stackConvolution(float bias, const float * channelData, int kernelX, int kernelY, int inputChannels, int outputChannels) {
    EXPECT_NE(channelData, nullptr);
    float * data = new float[outputChannels + kernelX*kernelY*inputChannels*outputChannels];
    for (int i=0; i < outputChannels; i++) data[i] = bias;
    float * ptr = data + outputChannels;
    for (int out=0; out < outputChannels; out++) {
        for (int y=0; y < kernelY; y++) {
            for (int x=0; x < kernelX; x++) {
                for (int in=0; in < inputChannels; in++) {
                    ptr[out*kernelX*kernelY*inputChannels + y*kernelX*inputChannels + x * inputChannels + in] = channelData[y*kernelX+x];
                }
            }
        }
    }
    return data;
}



/**
 * @brief Generate 3D tensor with constant data
 *
 * @param content Constant value to put into every element of the generated tensor
 * @param channels Number of channels for the tensor
 * @param width Width of the tensor
 * @param height Height of the tensor
 * @param padding Add isotropic padding in the spatial dimension
 *
 * @return Pointer to 3D tensor, ownership transferred to caller
 */
float * LayerTestBase::generateConstantData(float content, int channels, int width, int height, int padding) {
    EXPECT_GT(width, 0);
    EXPECT_GT(height, 0);
    EXPECT_GT(channels, 0);
    float * data = new float[channels*(width+2*padding)*(height+2*padding)];
    memset(data, 0, channels*(width+2*padding)*(height+2*padding)*sizeof(float));
    int stride = width + 2*padding;
    int cstride = (width+2*padding)*(height+2*padding);
    for (int c=0; c < channels; c++) {
        for (int y=padding; y < height+padding; y++) {
            for (int x=padding; x < width+padding; x++) {
                data[c*cstride+y*stride+x] = content;
            }
        }
    }
    return data;
}


/**
 * @brief Generate 3D tensor with random data in given range
 *
 * @param channels Number of channels for the tensor
 * @param width Width of the tensor
 * @param height Height of the tensor
 * @param low Lower bound of data to be generated
 * @param high Upper bound of data to be generated
 * @param padding Add isotropic padding in the spatial dimension
 *
 * @return Pointer to 3D tensor, ownership transferred to caller
 */
float * LayerTestBase::generateRandomData(int channels, int width, int height, float low, float high, int padding) {
    EXPECT_GT(width, 0);
    EXPECT_GT(height, 0);
    EXPECT_GT(channels, 0);
    float * data = new float[channels*(width+2*padding)*(height+2*padding)];
    memset(data, 0, channels*(width+2*padding)*(height+2*padding)*sizeof(float));
    float range = (high-low);
    EXPECT_GT(range, 0.0f);
    int stride = width + 2*padding;
    int cstride = (width+2*padding)*(height+2*padding);
    for (int c=0; c < channels; c++) {
        for (int y=padding; y < height+padding; y++) {
            for (int x=padding; x < width+padding; x++) {
                int item = std::rand();
                data[c*cstride+y*stride+x] = ((range * (float)item)/(float)RAND_MAX)+low;
            }
        }
    }
    return data;
}


/**
 * @brief Generate simple "bilinear" data ramps, x-wise on odd channels and y-wise on even channels
 *
 * @param channels Number of channels for the tensor
 * @param width Width of the tensor
 * @param height Height of the tensor
 * @param padding Add isotropic padding in the spatial dimension
 *
 * @return Pointer to 3D tensor, ownership transferred to caller
 */
float * LayerTestBase::generateBilinearData(int channels, int width, int height, int padding) {
    EXPECT_GT(width, 0);
    EXPECT_GT(height, 0);
    EXPECT_GT(channels, 0);
    float * data = new float[channels*(width+2*padding)*(height+2*padding)];
    memset(data, 0, channels*(width+2*padding)*(height+2*padding)*sizeof(float));
    int stride = width + 2*padding;
    int cstride = (width+2*padding)*(height+2*padding);
    for (int c=0; c < channels; c++) {
        for (int y=padding; y < height+padding; y++) {
            for (int x=padding; x < width+padding; x++) {
                if (c&1) {
                    data[c*cstride+y*stride+x] = x;
                } else {
                    data[c*cstride+y*stride+x] = y;
                }
            }
        }
    }
    return data;
}


/**
 * @brief Generate 3D tensor with random data in given range, rounded to integers
 *
 * @param channels Number of channels for the tensor
 * @param width Width of the tensor
 * @param height Height of the tensor
 * @param low Lower bound of data to be generated
 * @param high Upper bound of data to be generated
 * @param padding Add isotropic padding in the spatial dimension
 *
 * @return Pointer to 3D tensor, ownership transferred to caller
 */
float * LayerTestBase::generateRandomIntegerData(int channels, int width, int height, float low, float high, int padding) {
    EXPECT_GT(width, 0);
    EXPECT_GT(height, 0);
    EXPECT_GT(channels, 0);
    float * data = new float[channels*(width+2*padding)*(height+2*padding)];
    memset(data, 0, channels*(width+2*padding)*(height+2*padding)*sizeof(float));
    float range = (high-low);
    EXPECT_GT(range, 0.0f);
    int stride = width + 2*padding;
    int cstride = (width+2*padding)*(height+2*padding);
    for (int c=0; c < channels; c++) {
        for (int y=padding; y < height+padding; y++) {
            for (int x=padding; x < width+padding; x++) {
                int item = std::rand();
                data[c*cstride+y*stride+x] = roundf(((range * (float)item)/(float)RAND_MAX)-low);
           }
        }
    }
    return data;
}


/**
 * @brief Generate textures from CPU tensor data
 *
 * @param layer Pointer to (GPU) layer to generate textures for
 * @param numTokens Number of tokens to fill the texture with
 * @param inputs Input tensor data to put into textures, one buffer per input port
 * @param residual Optional residual tensor data to put into textures
 *
 * This function will generate textures for the supplied \p layer and register those directly with
 * that. If no \p residual is supplied, no residual textures will be generated or added to the target
 * layer. The \p numTokens parameter controls how much information from the \p inputs and
 * \p residual buffers is copied to the textures. The textures themselves will always be generated
 * with full size and padded with zeros.
 */
void LayerTestBase::generateSequenceTextures(fyusion::fyusenet::gpu::GPULayerBase * layer,
                                             int numTokens,
                                             const std::vector<const float *>& inputs,
                                             const float * residual) {
    using namespace fyusion::fyusenet;
    using namespace fyusion::fyusenet::gpu;
    std::vector<BufferSpec> inbufs = layer->getRequiredInputBuffers();
    std::vector<BufferSpec> outbufs = layer->getRequiredOutputBuffers();
    assert(inbufs[0].dataOrder_ == BufferSpec::order::GPU_SEQUENCE);
    assert(outbufs[0].dataOrder_ == BufferSpec::order::GPU_SEQUENCE);
    int totaltex = (int)(inbufs.size() + outbufs.size());
    if (residual) totaltex += (int)outbufs.size();
    int ttoffset = (int)testTextures_.size();
    glGetError();
    testTextures_.resize(ttoffset + totaltex, 0);
    glGenTextures(totaltex, &testTextures_[ttoffset]);
    ASSERT_EQ(glGetError(), (GLenum)GL_NO_ERROR);
    int inputtextures = 0;
    if (!inputs.empty()) {
        // ---------------------------------------------
        // Handle input for sequence-format tensor layers
        // ---------------------------------------------
        for (int port=0; port < (int)inputs.size(); port++) {
            const float *input = inputs.at(port);
            GLuint tex = testTextures_.at(ttoffset + inputtextures);
            copyToSequenceTexture(input, tex, layer->getWidth(), layer->getHeight(), numTokens);
            layer->addInputTexture(testTextures_.at(ttoffset + inputtextures), inputtextures);
            inputtextures++;
        }
    }
    int residualtextures = 0;
    if (residual) {
        // ---------------------------------------------
        // Handle sequence residual textures...
        // ---------------------------------------------
        GLuint tex = testTextures_.at(ttoffset + inputtextures);
        copyToSequenceTexture(residual, tex, layer->getWidth(), layer->getHeight(), numTokens);
        layer->addResidualTexture(testTextures_.at(ttoffset + inputtextures + residualtextures), residualtextures);
        residualtextures++;
    }
    if (outbufs[0].passThrough_) layer->addOutputTexture(testTextures_.at(ttoffset), 0, 0);
    else {
        configureTexture(testTextures_.at(ttoffset + inputtextures + residualtextures),
                         layer->getViewport()[0], layer->getViewport()[1],
                         (GLint)outbufs[0].internalFormat_, (GLenum)outbufs[0].format_, (GLenum)outbufs[0].type_,
                         nullptr);
        layer->addOutputTexture(testTextures_.at(ttoffset + inputtextures+residualtextures), 0, 0);
    }
}


/**
 * @brief Generate textures from CPU tensor data
 *
 * @param layer Pointer to (GPU) layer to generate textures for
 * @param inputs Input tensor data to put into textures, one buffer per input port
 * @param residual Optional residual tensor data to put into textures
 * @param includesPadding Optional flag that indicates if the supplied raw data already includes the padding
 *                        <b>on the input</b> demanded by the target layer, defaults to \c false
 *
 * This function will generate textures for the supplied \p layer and register those directly with
 * that. If no \p residual is supplied, no residual textures will be generated or added to the target
 * layer.
 */
void LayerTestBase::generateTextures(fyusion::fyusenet::gpu::GPULayerBase * layer,
                                     const std::vector<const float *>& inputs,
                                     const float * residual,
                                     bool includesPadding) {
    using namespace fyusion::fyusenet;
    using namespace fyusion::fyusenet::gpu;
    bool sequencein = false, sequenceout = false;
    std::vector<BufferSpec> inbufs = layer->getRequiredInputBuffers();
    std::vector<BufferSpec> outbufs = layer->getRequiredOutputBuffers();
    int ttoffset = (int)testTextures_.size();
    int totaltex = (int)(inbufs.size() + outbufs.size());
    if (residual) totaltex += (int)outbufs.size();
    testTextures_.resize(ttoffset + totaltex, 0);
    glGetError();
    glGenTextures(totaltex, &testTextures_[ttoffset]);
    ASSERT_EQ(glGetError(), (GLenum)GL_NO_ERROR);
    if ((inbufs[0].dataOrder_ == BufferSpec::order::GPU_DEEP) || (outbufs[0].dataOrder_ == BufferSpec::order::GPU_DEEP)) {
        if (dynamic_cast<gpu::deep::DeepLayerBase *>(layer)) {
            tiler_ = dynamic_cast<gpu::deep::DeepLayerBase *>(layer)->getTiler();
            ASSERT_NE(tiler_, nullptr);
        } else if (dynamic_cast<gpu::deep::DeepConvLayerBase *>(layer)) {
            tiler_ = dynamic_cast<gpu::deep::DeepConvLayerBase *>(layer)->getTiler();
            ASSERT_NE(tiler_, nullptr);
            residualTiler_ = dynamic_cast<gpu::deep::DeepConvLayerBase *>(layer)->getResidualTiler();
        } else {
            ASSERT_TRUE(false);
        }
    }
    if (inbufs[0].dataOrder_ == BufferSpec::order::GPU_SEQUENCE) sequencein = true;
    if (outbufs[0].dataOrder_ == BufferSpec::order::GPU_SEQUENCE) sequenceout = true;
    int inputtextures = 0;
    if (!inputs.empty()) {
        if (tiler_) {
            // -------------------------------------------
            // Handle input for deep-format tensor layers
            // -------------------------------------------
            int padding = layer->getInputPadding();
            int netwidth = layer->getWidth();
            int netheight = layer->getHeight();
            for (int port=0; port < (int)inputs.size(); port++) {
                const float * input = inputs.at(port);
                copyToDeepTexture(input, testTextures_.at(ttoffset + inputtextures), tiler_, netwidth, netheight, padding, layer->numInputChannels(port), includesPadding);
                layer->addInputTexture(testTextures_.at(ttoffset + inputtextures), port);
                inputtextures++;
            } // input port loop
        } else {
            // ---------------------------------------------
            // Handle input for shallow-format tensor layers
            // ---------------------------------------------
            int padding = (includesPadding) ? 0 : layer->getInputPadding();
            int netwidth = (includesPadding) ? layer->getWidth() + layer->getInputPadding()*2 : layer->getWidth();
            int netheight = (includesPadding) ? layer->getHeight() + layer->getInputPadding()*2 : layer->getHeight();
            for (int port=0; port < (int)inputs.size(); port++) {
                const float *input = inputs.at(port);
                int remchans = (sequencein) ? LayerBase::PIXEL_PACKING : layer->numInputChannels(port);
                int chanoffset = 0;
                while (remchans > 0) {
                    GLuint tex = testTextures_.at(ttoffset + inputtextures);
                    remchans = copyToShallowTexture(input, tex, netwidth, netheight, padding, chanoffset, remchans);
                    chanoffset += LayerBase::PIXEL_PACKING;
                    layer->addInputTexture(testTextures_.at(ttoffset + inputtextures), inputtextures);
                    inputtextures++;
                }
            }
        }
    } // input textures
    int residualtextures = 0;
    if (residual) {
        if (residualTiler_) {
            // ---------------------------------------------
            // Handle deep  residual textures...
            // ---------------------------------------------
            int netwidth = residualTiler_->getInputWidth();
            int netheight = residualTiler_->getInputHeight();
            copyToDeepTexture(residual, testTextures_.at(ttoffset + inputtextures), tiler_, netwidth, netheight, layer->getOutputPadding(), layer->numOutputChannels(), false);
            layer->addResidualTexture(testTextures_.at(ttoffset + inputtextures), 0);
            residualtextures++;
        } else {
            // ---------------------------------------------
            // Handle shallow residual textures...
            // ---------------------------------------------
            int padding = layer->getOutputPadding();
            int netwidth = layer->getViewport()[0] - 2 * padding;
            int netheight = layer->getViewport()[1] - 2 * padding;
            int remchans = (sequenceout) ? LayerBase::PIXEL_PACKING : layer->numOutputChannels();
            int chanoffset = 0;
            while (remchans > 0) {
                GLuint tex = testTextures_.at(ttoffset + inputtextures+residualtextures);
                remchans = copyToShallowTexture(residual, tex, netwidth, netheight, padding, chanoffset, remchans);
                chanoffset += LayerBase::PIXEL_PACKING;
                layer->addResidualTexture(testTextures_.at(ttoffset + inputtextures + residualtextures), residualtextures);
                residualtextures++;
            }
        }
    } // residual textures
    // output textures
    if (outbufs[0].dataOrder_ == BufferSpec::order::GPU_DEEP) {
        ASSERT_NE(tiler_, nullptr);
        if (outbufs[0].passThrough_) layer->addOutputTexture(testTextures_.at(ttoffset), 0, 0);
        else {
            configureTexture(testTextures_.at(ttoffset + inputtextures + residualtextures), tiler_->getViewportWidth(), tiler_->getViewportHeight(), nullptr);
            layer->addOutputTexture(testTextures_.at(ttoffset + inputtextures + residualtextures), 0, 0);
        }
    } else {
        for (int slice=0; slice < (int)outbufs.size(); slice++) {
            if (outbufs[slice].passThrough_) layer->addOutputTexture(testTextures_.at(ttoffset + slice), slice, 0);
            else {
                configureTexture(testTextures_.at(ttoffset + inputtextures + residualtextures + slice),
                                 layer->getViewport()[0], layer->getViewport()[1],
                                 (GLint)outbufs[0].internalFormat_, (GLenum)outbufs[0].format_, (GLenum)outbufs[0].type_,
                                 nullptr);
                layer->addOutputTexture(testTextures_.at(ttoffset + inputtextures+residualtextures+slice), slice, 0);
            }
        }
    }
}

SingleWeightProvider::SingleWeightProvider(const float *weights, const float * bias, const float *bn) {
    using namespace fyusion::fyusenet;
    weights_ = new DefaultDataWrapper<float>(weights);
    bias_ = new DefaultDataWrapper<float>(bias);
    postNorm_ = new DefaultDataWrapper<float>(bn);
}

SingleWeightProvider::~SingleWeightProvider() {
    delete weights_;
    delete bias_;
    delete postNorm_;
}

fyusion::fyusenet::DataBlob SingleWeightProvider::get(const std::string &name, int layerNo, int subIndex) const {
    switch (subIndex) {
        case 0:
            return fyusion::fyusenet::DataBlob(weights_);
        case 1:
            return fyusion::fyusenet::DataBlob(bias_);
        case 2:
            return fyusion::fyusenet::DataBlob(postNorm_);
        default:
            THROW_EXCEPTION_ARGS(fyusion::FynException, "This should not happen");
    }
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


// TODO (mw) docs
int LayerTestBase::copyToShallowTexture(const float *input, GLuint handle, int netwidth, int netheight, int padding, int chanOffset, int remchans) {
    using namespace fyusion::fyusenet;
    using namespace fyusion::fyusenet::gpu;
    int iwidth = netwidth + 2 * padding;
    int iheight = netheight + 2 * padding;
    float *tmpimg = new float[iwidth * iheight * LayerBase::PIXEL_PACKING];
    memset(tmpimg, 0, iwidth * iheight * LayerBase::PIXEL_PACKING * sizeof(float));
    int cmax = (remchans >= LayerBase::PIXEL_PACKING) ? LayerBase::PIXEL_PACKING : remchans;
    for (int chan = 0; chan < cmax; chan++) {
        for (int y = 0; y < netheight; y++) {
            for (int x = 0; x < netwidth; x++) {
                tmpimg[(y + padding) * iwidth * LayerBase::PIXEL_PACKING + (x + padding) * LayerBase::PIXEL_PACKING + chan] = input[(chanOffset + chan) * netwidth * netheight + y * netwidth + x];
            }
        }
    }
    configureTexture(handle, iwidth, iheight, tmpimg);
    delete [] tmpimg;
    return remchans - cmax;
}


// TODO (mw) docs
void LayerTestBase::copyToDeepTexture(const float * input, GLuint handle, fyusion::fyusenet::gpu::deep::DeepTiler * tiler, int netwidth, int netheight, int padding, int inChans, bool includesPadding) {
    using namespace fyusion::fyusenet;
    using namespace fyusion::fyusenet::gpu;
    // -------------------------------------------
    // Handle input for deep-format tensor layers
    // -------------------------------------------
    int iwidth = tiler->getInputTextureWidth();
    int iheight = tiler->getInputTextureHeight();
    float *tmpimg = new float[iwidth * iheight * LayerBase::PIXEL_PACKING];
    memset(tmpimg, 0, iheight*iwidth*LayerBase::PIXEL_PACKING*sizeof(float));
    int tilex = tiler->numInputTiles(deep::DeepTiler::HORIZONTAL);
    int tiley = tiler->numInputTiles(deep::DeepTiler::VERTICAL);
    int chan = 0;
    int srcstride = (includesPadding) ? netwidth + 2*padding : netwidth;
    int srcstridec = (includesPadding) ? srcstride * (netheight+2*padding) : srcstride * netheight;
    for (int ty=0; ty < tiley; ty++) {
        for (int tx=0; tx < tilex; tx++) {
            int rem = (inChans - chan) > LayerBase::PIXEL_PACKING ? LayerBase::PIXEL_PACKING : inChans - chan;
            if (rem > 0) {
                float * ptr = tmpimg + LayerBase::PIXEL_PACKING * (((padding + ty*(netheight + padding))*iwidth) + padding + (tx*(netwidth + padding)));
                const float * src = (includesPadding) ? input + padding*srcstride + padding : input;
                for (int y=0; y < netheight; y++) {
                    for (int x=0; x < netwidth; x++) {
                        for (int ichan=0; ichan < rem; ichan++) {
                            float val = src[y*srcstride + x + (chan+ichan) * srcstridec];
                            ptr[(y*iwidth+x)*LayerBase::PIXEL_PACKING + ichan] = val;
                        }
                    }
                }
            }
            chan += LayerBase::PIXEL_PACKING;
        }
    }
    configureTexture(handle, iwidth, iheight, tmpimg);
    delete [] tmpimg;
}

// FIXME (mw) docs
void LayerTestBase::copyToSequenceTexture(const float *input, GLuint handle, int width, int height, int numTokens) {
    using namespace fyusion::fyusenet;
    using namespace fyusion::fyusenet::gpu;
    float *tmpimg = new float[width * height * LayerBase::PIXEL_PACKING];
    memset(tmpimg, 0, width * height * LayerBase::PIXEL_PACKING * sizeof(float));
    for (int chan = 0; chan < LayerBase::PIXEL_PACKING; chan++) {
        for (int y = 0; y < numTokens; y++) {
            for (int x = 0; x < width; x++) {
                tmpimg[y * width * LayerBase::PIXEL_PACKING + x * LayerBase::PIXEL_PACKING + chan] = input[(y * width + x) * LayerBase::PIXEL_PACKING + chan];
            }
        }
    }
    configureTexture(handle, width, height, tmpimg);
    delete [] tmpimg;
}


// FIXME (mw) docs
void LayerTestBase::configureTexture(GLuint tex, int width, int height, const void *data) {
    using namespace fyusion::fyusenet;
    using namespace fyusion::fyusenet::gpu;
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, (GLint)GPULayerBase::TEXTURE_IFORMAT_4, width, height, 0, (GLenum)GPULayerBase::TEXTURE_FORMAT_4, GL_FLOAT, data);
}


// FIXME (mw) docs
void LayerTestBase::configureTexture(GLuint tex, int width, int height, GLint iformat, GLenum format, GLenum dtype, const void *data) {
    using namespace fyusion::fyusenet;
    using namespace fyusion::fyusenet::gpu;
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, iformat, width, height, 0, format, dtype, data);
}


// vim: set expandtab ts=4 sw=4:
