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
 * @param kernel Kernel size (K) of the kernel data in \p channelData
 * @param inputChannels Number of input channels for the convolution
 * @param outputChannels Number of output channels for the convolution
 *
 * @return Pointer to weight and bias array for the convolution, ownership is passed to the caller
 *
 * This function generates a convolution tensor by stacking the supplied \p channelData for each channel
 * and then adding the supplied bias (also to each channel).
 */
float * LayerTestBase::stackConvolution(float bias, const float * channelData, int kernel, int inputChannels, int outputChannels) {
    EXPECT_NE(channelData, nullptr);
    float * data = new float[outputChannels + kernel*kernel*inputChannels*outputChannels];
    for (int i=0; i < outputChannels; i++) data[i] = bias;
    float * ptr = data + outputChannels;
    for (int out=0; out < outputChannels; out++) {
        for (int y=0; y < kernel; y++) {
            for (int x=0; x < kernel; x++) {
                for (int in=0; in < inputChannels; in++) {
                    ptr[out*kernel*kernel*inputChannels + y*kernel*inputChannels + x * inputChannels + in] = channelData[y*kernel+x];
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
 *
 * @return Pointer to 3D tensor, ownership transferred to caller
 */
float * LayerTestBase::generateConstantData(float content, int channels, int width, int height) {
    EXPECT_GT(width, 0);
    EXPECT_GT(height, 0);
    EXPECT_GT(channels, 0);
    float * data = new float[channels*width*height];
    for (int i=0; i < width*height*channels; i++) data[i] = content;
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
 *
 * @return Pointer to 3D tensor, ownership transferred to caller
 */
float * LayerTestBase::generateRandomData(int channels, int width, int height, float low, float high) {
    EXPECT_GT(width, 0);
    EXPECT_GT(height, 0);
    EXPECT_GT(channels, 0);
    float * data = new float[channels * width * height];
    float range = (high-low);
    EXPECT_GT(range, 0.0f);
    for (int i=0; i < width*height*channels; i++) {
        int item = std::rand();
        data[i] = ((range * (float)item)/(float)RAND_MAX)+low;
    }
    return data;
}


/**
 * @brief Generate simple "bilinear" data ramps, x-wise on odd channels and y-wise on even channels
 *
 * @param channels Number of channels for the tensor
 * @param width Width of the tensor
 * @param height Height of the tensor
 *
 * @return Pointer to 3D tensor, ownership transferred to caller
 */
float * LayerTestBase::generateBilinearData(int channels, int width, int height) {
    EXPECT_GT(width, 0);
    EXPECT_GT(height, 0);
    EXPECT_GT(channels, 0);
    float * data = new float[channels * width * height];
    for (int c=0; c < channels; c++) {
        for (int y=0; y < height; y++) {
            for (int x=0; x < width; x++) {
                if (c&1) {
                    data[c*width*height+y*width+x] = x;
                } else {
                    data[c*width*height+y*width+x] = y;
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
 *
 * @return Pointer to 3D tensor, ownership transferred to caller
 */
float * LayerTestBase::generateRandomIntegerData(int channels, int width, int height, float low, float high) {
    EXPECT_GT(width, 0);
    EXPECT_GT(height, 0);
    EXPECT_GT(channels, 0);
    float * data = new float[channels * width * height];
    float range = (high-low);
    EXPECT_GT(range, 0.0f);
    for (int i=0; i < width*height*channels; i++) {
        int item = std::rand();
        float aa = roundf(((range * (float)item)/(float)RAND_MAX)+low);
        //data[i] = roundf(((range * (float)item)/(float)RAND_MAX)-low);
        data[i] = aa;
    }
    return data;
}


/**
 * @brief Generate textures from tensor data
 *
 * @param layer Pointer to (GPU) layer to generate textures for
 * @param inputs Input tensor data to put into textures, one buffer per input port
 * @param residual Optional residual tensor data to put into textures
 *
 * This function will generate textures for the supplied \p layer and register those directly with
 * that. If no \p residual is supplied, no residual textures will be generated or added to the target
 * layer.
 */
void LayerTestBase::generateTextures(fyusion::fyusenet::gpu::GPULayerBase * layer, const std::vector<const float *>& inputs, const float * residual) {
    using namespace fyusion::fyusenet;
    using namespace fyusion::fyusenet::gpu;
    std::vector<BufferSpec> inbufs = layer->getRequiredInputBuffers();
    std::vector<BufferSpec> outbufs = layer->getRequiredOutputBuffers();
    int ttoffset = (int)testTextures_.size();
    int totaltex = (int)(inbufs.size() + outbufs.size());
    if (residual) totaltex += (int)outbufs.size();
    testTextures_.resize(ttoffset + totaltex, 0);
    glGetError();
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
    int inputtextures = 0;
    if (inputs.size() > 0) {
        int padding = layer->getInputPadding();
        int netwidth = layer->getWidth();
        int netheight = layer->getHeight();
        int iwidth = netwidth + 2 * padding;
        int iheight = netheight + 2 * padding;
        if (tiler_) {
            for (int port=0; port < (int)inputs.size(); port++) {
                const float * input = inputs.at(port);
                // -------------------------------------------
                // Handle input for deep-format tensor layers
                // -------------------------------------------
                iwidth = tiler_->getInputTextureWidth();
                iheight = tiler_->getInputTextureHeight();
                float *tmpimg = new float[iwidth * iheight * LayerBase::PIXEL_PACKING];
                memset(tmpimg, 0, iheight*iwidth*LayerBase::PIXEL_PACKING*sizeof(float));
                int tilex = tiler_->numInputTiles(deep::DeepTiler::HORIZONTAL);
                int tiley = tiler_->numInputTiles(deep::DeepTiler::VERTICAL);
                int chan = 0;
                for (int ty=0; ty < tiley; ty++) {
                    for (int tx=0; tx < tilex; tx++) {
                        int rem = (layer->numInputChannels(port) - chan) > LayerBase::PIXEL_PACKING ? LayerBase::PIXEL_PACKING : layer->numInputChannels(port) - chan;
                        if (rem > 0) {
                            float * ptr = tmpimg + LayerBase::PIXEL_PACKING * (((padding + ty*(netheight + padding))*iwidth) + padding + (tx*(netwidth + padding)));
                            for (int y=0; y < netheight; y++) {
                                for (int x=0; x < netwidth; x++) {
                                    for (int ichan=0; ichan < rem; ichan++) {
                                        float val = input[y*netwidth + x + (chan+ichan) * (netwidth*netheight)];
                                        ptr[(y*iwidth+x)*LayerBase::PIXEL_PACKING + ichan] = val;
                                    }
                                }
                            }
                        }
                        chan += LayerBase::PIXEL_PACKING;
                    }
                }
                glBindTexture(GL_TEXTURE_2D, testTextures_.at(inputtextures));
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexImage2D(GL_TEXTURE_2D,0,GPULayerBase::TEXTURE_IFORMAT_4,iwidth,iheight,0,GL_RGBA,GL_FLOAT,tmpimg);
                layer->addInputTexture(testTextures_.at(inputtextures), port);
                inputtextures++;
                delete [] tmpimg;
            } // input port loop
        } else {
            // ---------------------------------------------
            // Handle input for shallow-format tensor layers
            // ---------------------------------------------
            for (int port=0; port < (int)inputs.size(); port++) {
                const float * input = inputs.at(port);
                float *tmpimg = new float[iwidth * iheight * LayerBase::PIXEL_PACKING];
                int texslicemax = std::max(1, (layer->numInputChannels(port) + LayerBase::PIXEL_PACKING-1) / LayerBase::PIXEL_PACKING);
                int remchans = layer->numInputChannels(port);
                for (int texslice=0; texslice < texslicemax; texslice++) {
                    memset(tmpimg, 0, iwidth * iheight * LayerBase::PIXEL_PACKING * sizeof(float));
                    int cmax = (remchans >= LayerBase::PIXEL_PACKING) ? LayerBase::PIXEL_PACKING : remchans;
                    for (int chan=0; chan < cmax; chan++) {
                        for (int y=0; y < netheight ; y++) {
                            for (int x=0; x < netwidth ; x++) {
                                tmpimg[(y+padding)*iwidth*LayerBase::PIXEL_PACKING+(x+padding)*LayerBase::PIXEL_PACKING+chan]=input[(texslice*LayerBase::PIXEL_PACKING+chan)*netwidth*netheight+y*netwidth+x];
                            }
                        }
                    }
                    remchans -= cmax;
                    glBindTexture(GL_TEXTURE_2D, testTextures_.at(inputtextures));
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTexImage2D(GL_TEXTURE_2D, 0, GPULayerBase::TEXTURE_IFORMAT_4, iwidth, iheight, 0, GL_RGBA, GL_FLOAT, tmpimg);
                    layer->addInputTexture(testTextures_.at(inputtextures), inputtextures);
                    inputtextures++;
                }
                delete [] tmpimg;
            }
        }
    } // input textures
    int residualtextures = 0;
    if (residual) {
        if (residualTiler_) {
            // deep
            int rvpwidth = residualTiler_->getViewportWidth();
            int rvpheight = residualTiler_->getViewportHeight();
            float *tmpres = new float[rvpwidth * rvpheight * LayerBase::PIXEL_PACKING];
            memset(tmpres, 0, rvpwidth * rvpheight * LayerBase::PIXEL_PACKING * sizeof(float));
            int tilex = residualTiler_->numInputTiles(deep::DeepTiler::HORIZONTAL);
            int tiley = residualTiler_->numInputTiles(deep::DeepTiler::VERTICAL);
            int rwidth = residualTiler_->getInputWidth();
            int rheight = residualTiler_->getInputHeight();
            int rpadding = layer->getOutputPadding();
            int chan = 0;
            int iheight = residualTiler_->getInputTextureHeight();
            int iwidth = residualTiler_->getInputTextureWidth();
            for (int ty=0; ty < tiley; ty++) {
                for (int tx=0; tx < tilex; tx++) {
                    int rem = (layer->numOutputChannels() - chan) > LayerBase::PIXEL_PACKING ? LayerBase::PIXEL_PACKING : layer->numOutputChannels() - chan;
                    float * ptr = tmpres + LayerBase::PIXEL_PACKING * (((rpadding + ty*(rheight + rpadding)) * iwidth)+rpadding + (tx*(rwidth+rpadding)));
                    for (int y=0; y < rheight; y++) {
                        for (int x=0; x < rwidth; x++) {
                            for (int ichan=0; ichan < rem; ichan++) {
                                float val = residual[y*rwidth+x+(chan+ichan)*(rwidth*rheight)];
                                ptr[(y*iwidth+x) * LayerBase::PIXEL_PACKING + ichan] = val;
                            }
                        }
                    }
                    chan += LayerBase::PIXEL_PACKING;
                }
            }
            glBindTexture(GL_TEXTURE_2D, testTextures_.at(inputtextures));
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexImage2D(GL_TEXTURE_2D,0,GPULayerBase::TEXTURE_IFORMAT_4,iwidth,iheight,0,GL_RGBA,GL_FLOAT,tmpres);
            layer->addResidualTexture(testTextures_.at(inputtextures), 0);
            residualtextures = 1;
            delete [] tmpres;
        } else {
            // shallow
            int padding = layer->getOutputPadding();
            int owidth = layer->getViewport()[0];
            int oheight = layer->getViewport()[1];
            int dwidth = owidth - 2*padding;
            int dheight = oheight - 2*padding;
            float *tmpres = new float[owidth * oheight * LayerBase::PIXEL_PACKING];
            int slicemax = std::max(1, (layer->numOutputChannels() + LayerBase::PIXEL_PACKING-1) / LayerBase::PIXEL_PACKING);
            int remchans = layer->numOutputChannels();
            for (int slice=0; slice < slicemax; slice++) {
                assert(remchans > 0);
                memset(tmpres, 0, owidth*oheight * LayerBase::PIXEL_PACKING * sizeof(float));
                int chanmax = (remchans >= LayerBase::PIXEL_PACKING) ? LayerBase::PIXEL_PACKING : remchans;
                for (int ichan=0; ichan < chanmax; ichan++) {
                    for (int y=0; y < dheight; y++) {
                        for (int x=0; x < dwidth; x++) {
                            tmpres[((y+padding)*owidth + x+padding) * LayerBase::PIXEL_PACKING + ichan] = residual[(slice*LayerBase::PIXEL_PACKING + ichan)*dwidth*dheight+y*dwidth+x];
                        }
                    }
                }
                remchans -= chanmax;
                int texidx = inputtextures + slice;
                glBindTexture(GL_TEXTURE_2D, testTextures_.at(texidx));
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexImage2D(GL_TEXTURE_2D, 0, GPULayerBase::TEXTURE_IFORMAT_4, owidth, oheight, 0, GL_RGBA, GL_FLOAT, tmpres);
                layer->addResidualTexture(testTextures_.at(texidx), slice);
                residualtextures++;
            }
            delete [] tmpres;
        }
    } // residual textures
    // output textures
    if (outbufs[0].dataOrder_ == BufferSpec::order::GPU_DEEP) {
        ASSERT_NE(tiler_, nullptr);
        int owidth = tiler_->getViewportWidth();
        int oheight = tiler_->getViewportHeight();
        glBindTexture(GL_TEXTURE_2D, testTextures_.at(inputtextures+residualtextures));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GPULayerBase::TEXTURE_IFORMAT_4, owidth, oheight, 0, GL_RGBA, GL_FLOAT, nullptr);
        layer->addOutputTexture(testTextures_.at(inputtextures+residualtextures), 0);
    } else {
        for (int slice=0; slice < (int)outbufs.size(); slice++) {
            glBindTexture(GL_TEXTURE_2D, testTextures_[inputtextures+residualtextures+slice]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexImage2D(GL_TEXTURE_2D, 0, GPULayerBase::TEXTURE_IFORMAT_4, layer->getViewport()[0], layer->getViewport()[1], 0, GL_RGBA, GL_FLOAT, nullptr);
            layer->addOutputTexture(testTextures_.at(inputtextures+residualtextures+slice), slice);
        }
    }
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/



// vim: set expandtab ts=4 sw=4:

