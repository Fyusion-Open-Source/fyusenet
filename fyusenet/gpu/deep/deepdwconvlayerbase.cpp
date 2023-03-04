//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Depth-Wise Convolution Layer Base Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/glinfo.h"
#include "../gfxcontextlink.h"
#include "deepdwconvlayerbase.h"
#include "deeplayerbase.h"
#include "../floatconversion.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase
 */
DeepDepthwiseConvLayerBase::DeepDepthwiseConvLayerBase(const ConvLayerBuilder & builder, int layerNumber):DeepConvLayerBase(builder, layerNumber) {
    channelMultiplier_ = outputChannels_/builder.groupSize_;
    if (channelMultiplier_ > 1) {
        if (inputChannels_ & 3) THROW_EXCEPTION_ARGS(FynException,"Channel multipliers > 1 are only supported on input channels being a multiple of 4");
    }
}


/**
 * @brief Read weights and biases from raw data and store them into a texture
 *
 * @param biasAndWeights Pointer to array with bias and weight values (see long description)
 * @param offset Optional offset (in floating-point elements) into \p biasAndWeights where to
 *               start reading from
 *
 * This function parses the weights and biases stored in the \p biasAndWeights parameter for
 * usage with the GPU. It is assumed that the biases and weights are stored biases first,
 * followed by the convolution weights. In case a batchnorm operation is used, the batchnorm
 * parameters are following the weight data in the form of all scales and then all offsets.
 * For example, for \e n output channels, the first \e n entries in \p biasAndWeights are the
 * biases. For \e m input channels and a kernel of size \e k (i.e. a kxk kernel), we expect a 4D
 * array of size nxkxkxm with the following index
 * order:
 *
 * @code
 * [outchannel][kernely][kernelx][inchannel]
 * @endcode
 *
 * As opposed to the shallow tensor handling, it is not efficient to use multiple render passes
 * with changing uniforms for the convolution (at least not in my tests on mobile GPUs). Instead
 * we choose a different path, which packs the convolution coefficients into textures and use a
 * few tricks - when available.
 *
 * The texture format for the convolution coefficients is as follows:
 *   - Pixel format is \c RGBA
 *   - Texture \e height corresponds to the number of output channels multiplied by the convolution
 *     kernel size
 *   - Texture \e width corresponds to the number of input channels multiplied by the convolution
 *     kernel size (there is an additional tweak, see below for details)
 *   - Each pixel in the texture corresponds to 4 (or 8) convolution coefficients that are laid
 *     out as as part of 4x4 matrices
 *   - Four (4) consecutive pixels in a row represent a 4x4 matrix with the input channels as their
 *     column space and the output channels as their row space
 *   - Depending on the convolution kernel size, \e k neighboring 4x4 matrices horizontally
 *     represent the horizontal part of the kernel and \e k neighboring 4x4 matrices vertically
 *     represent the vertical part of the kernel
 *   - I should really put a picture here
 *
 * An additional tweak to the setup described above is the capability to contract the RAM
 * requirements by half. In order to do that, we do not use a floating-point texture, but a
 * 32-bit integer (per channel) texture. We then fit two 16-bit floating-point numbers in a
 * single channel and can reduce the texture width by 50% . This has to be decoded by the shader
 * later.
 *
 */
void DeepDepthwiseConvLayerBase::loadWeightsAndBiases(const float *biasAndWeights, size_t offset) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (!weightTexture_) glGenTextures(1,&weightTexture_);
    glBindTexture(GL_TEXTURE_2D,weightTexture_);
    const float * srcweights = biasAndWeights + offset + outputChannels_;
    createWeightTextureMatrix(srcweights, 0, weightTexture_);
    // TODO (mw) put into own function -> promote
    //------------------------------------------------------
    // If we have the post-BN flag set, store the batchnorm
    // stuff...
    //------------------------------------------------------
    if (flags_ & fyusenet::LayerFlags::POST_BATCHNORM) {
        int padout = 4*((outputChannels_+3)/4);
        const float * srcbn = biasAndWeights + outputChannels_ + offset + kernel_*kernel_*inputChannels_*channelMultiplier_;
        postBNScales_ = new float[padout];
        postBNBias_ = new float[padout];
        memset(postBNScales_,0,padout*sizeof(float));
        memset(postBNBias_,0,padout*sizeof(float));
        memcpy(postBNScales_, srcbn, outputChannels_*sizeof(float));
        memcpy(postBNBias_, srcbn+outputChannels_, outputChannels_*sizeof(float));
    }
    //------------------------------------------------------
    // Now for the bias part (and also batchnorm)...
    //------------------------------------------------------
    int bs = PIXEL_PACKING*(1+(outputChannels_+PIXEL_PACKING-1)/PIXEL_PACKING);
    if (flags_ & LayerFlags::POST_BATCHNORM) bs *= 2;
    float * bias = new float[bs];
    memset(bias,0,bs*sizeof(float));
    memcpy(bias+PIXEL_PACKING,biasAndWeights+offset,outputChannels_*sizeof(float));
    // load batchnorm scale and bias if necessary
    if (flags_ & LayerFlags::POST_BATCHNORM) {
        for (int i=0;i<outputChannels_;i++) {
            bias[PIXEL_PACKING+i] = bias[PIXEL_PACKING+i] * postBNScales_[i] + postBNBias_[i];
            bias[PIXEL_PACKING+(bs/2)+i] = postBNScales_[i];
        }
    }
    if (!biasTexture_) glGenTextures(1,&biasTexture_);
    glBindTexture(GL_TEXTURE_2D,biasTexture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,1+(outputChannels_+PIXEL_PACKING-1)/PIXEL_PACKING,(flags_ & LayerFlags::POST_BATCHNORM) ? 2 : 1,0,GL_RGBA,GL_FLOAT,bias);
    delete [] bias;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Compile shaders that implement the actual layer functionality
 *
 * This function obtains required shaders from the resource system, compiles/caches these shaders
 * and performs base initializations on them.
 */
void DeepDepthwiseConvLayerBase::setupShaders() {
    char preproc[1024] = {0};
    snprintf(preproc, sizeof(preproc), "#define WEIGHT_UNIT %d\n#define BIAS_UNIT %d\n",WEIGHT_TEXTURE,BIAS_TEXTURE);
    shaderPreprocessing(preproc, sizeof(preproc)-strlen(preproc)-1);
    compileConvolutionShaders(preproc);
}


/**
 * @brief Create
 *
 * @param srcWeights
 * @param winOffset
 * @param weightTexture GL texture handle to store weight data to
 *
 * This function parses the weights stored in the \p srcWeights parameter for usage with the GPU.
 * For \e m channels and a kernel of size \e k (i.e. a \f$ k \times  k \f$ kernel), this function
 * expects a 3D array of size \f$ n \times k \times k \times k \f$ with the following index order:
 * @code
 * [channel][kernely][kernelx]
 * @endcode
 *
 * Data is stored in (up to) 4x4 matrices,
 * As opposed to the shallow tensor handling, it is not efficient to use multiple render passes
 * with changing uniforms for the convolution (at least not in my tests on mobile GPUs). Instead
 * a different path is chosen, which packs the convolution coefficients into textures and use a
 * few tricks - when available.
 *
 * The texture format for the convolution coefficients is as follows:
 *   - Pixel format is \c RGBA
 *   - Texture \e height corresponds to the (remaining) kernel size along the y-dimension (capped
 *     by \c PIXEL_PACKING), multiplied with the #channelMultiplier_
 *   - Texture \e width corresponds to the number of input channels multiplied by the convolution
 *     kernel size, padded for even size (there is an additional tweak, see below for details)
 *   - Each pixel in the texture corresponds to 4 (or 8) convolution coefficients that are laid
 *     out as as part of 4x4 matrices
 *   - Four (4) consecutive pixels in a row represent a 4x4 matrix with the input channels as their
 *     column space and the output channels as their row space
 *   - The texture rows are structured, such that consecutive rows correspond to kernel rows (y-axis)
 *     to the extent of the kernel (or the kernel remainder for split kernels). After the y-kernel
 *     size has been met, the set of rows are repeated for #channelMultiplier_ values > 1 (not tested).
 *
 * An additional tweak to the setup described above is the capability to contract the RAM
 * requirements by half. In order to do that, we do not use a floating-point texture, but a
 * 32-bit integer (per channel) texture. We then fit two 16-bit floating-point numbers in a
 * single channel and can reduce the texture width by 50% . This has to be decoded by the shader
 * later.
 *
 * Kernel sizes that extent the value of \c PIXEL_PACKING, have to be split into several weight
 * textures. This is not implemented yet in the shader part, as FyuseNet currently does not have
 * any depth-wise convolution layer with a kernel size larger than 4.
 *
 * @warning Values of #channelMultiplier_ other than 1 have not been tested (yet), also there is no
 *          implementation for kernel sizes > \c PIXEL_PACKING in the derived classes (yet).
 */
void DeepDepthwiseConvLayerBase::createWeightTextureMatrix(const float *srcWeights, int winOffset, GLuint weightTexture) {
    // TODO (mw) check for matrix size here (we should support 4x1, 4x2 and 4x4 for simplicity in the shader)
    // as we are storing 4x4 matrices (with one row padded w/ zeros), we are not dividing by PIXEL_PACKING here
    int winmax = std::min(PIXEL_PACKING,kernel_ - winOffset);
    int winrem = PIXEL_PACKING - winmax;
    int chanblocks = ((inputChannels_ % PIXEL_PACKING)==0) ? inputChannels_ / PIXEL_PACKING : (inputChannels_ + (PIXEL_PACKING - (inputChannels_ % PIXEL_PACKING)))/PIXEL_PACKING;
    int texwidth = chanblocks * (kernel_ + (winrem & 1));
    if (texwidth & 1) texwidth++;
    int texheight = winmax * channelMultiplier_;
#ifndef HIGH_PRECISION
    if (halfSupport_) {
        if (((texwidth / 2) > GLInfo::getMaximumTextureSize()) || (texheight > GLInfo::getMaximumTextureSize())) {
            THROW_EXCEPTION_ARGS(FynException,"Weights do not fit into GL texture");
        }
    } else {
        if ((texwidth > GLInfo::getMaximumTextureSize()) || (texheight > GLInfo::getMaximumTextureSize())) {
            THROW_EXCEPTION_ARGS(FynException,"Weights do not fit into GL texture");
        }
    }
#else
    if ((texwidth > GLInfo::getMaximumTextureSize()) || (texheight > GLInfo::getMaximumTextureSize())) {
        THROW_EXCEPTION_ARGS(FynException,"Weights do not fit into GL texture");
    }
#endif
    float * weights = new float[texwidth*texheight*PIXEL_PACKING];
    memset(weights,0,texwidth*texheight*PIXEL_PACKING*sizeof(float));
    for (int chan=0 ; chan < channelMultiplier_; chan++) {
        for (int fy=winOffset; fy < winmax; fy++) {
            float *wptr = weights+(chan*winmax+fy)*texwidth*PIXEL_PACKING;
            // below defines one row in the target texture
            for (int inlayer=0; inlayer < inputChannels_; inlayer += PIXEL_PACKING) {
                int irem = ((inputChannels_ - inlayer) >= PIXEL_PACKING) ? PIXEL_PACKING : (inputChannels_ - inlayer);
                for (int fx=0; fx  <kernel_; fx++) {
                    for (int il=inlayer; il < inlayer+irem; il++) {
                        int srcoffset = il*kernel_*kernel_*channelMultiplier_ + (fy*kernel_ + fx)*channelMultiplier_ + chan;
                        *wptr = srcWeights[srcoffset];
                        wptr++;
                    }
                    wptr += PIXEL_PACKING-irem;
                }
                wptr += (winrem & 1)*PIXEL_PACKING;
            }
        }
    }
    glBindTexture(GL_TEXTURE_2D,weightTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
#ifndef HIGH_PRECISION
    if (halfSupport_) {
        unsigned int * fp16 = FloatConversion::getInstance()->toFP16UI(weights,texwidth*texheight*PIXEL_PACKING);
#ifdef GL_RGBA32UI
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32UI,texwidth/2,texheight,0,GL_RGBA_INTEGER,GL_UNSIGNED_INT,fp16);
#else
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32UI_EXT,texwidth/2,texheight,0,GL_RGBA_INTEGER_EXT,GL_UNSIGNED_INT,fp16);
#endif
        delete [] fp16;
    } else {
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA16F,texwidth,texheight,0,GL_RGBA,GL_FLOAT,weights);
    }
#else
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,texwidth,texheight,0,GL_RGBA,GL_FLOAT,weights);
#endif
    delete [] weights;
}


/**
 * @copydoc DeepConvLayerBase::setupNetworkPolygons
 */
void DeepDepthwiseConvLayerBase::setupNetworkPolygons(VAO *vao) {
    int offset0=0;
    float * attrs0 = new float[tiler_->numOutputTiles()*4*4];
    std::vector<DeepTiler::Tile> tiles = tiler_->createOutputTiles();
    std::vector<DeepTiler::Tile> intiles = tiler_->createInputTiles(0,0);
    //---------------------------------------------
    // VBO parts, first the default output tiling
    // combined with default input tiling...
    //---------------------------------------------
    assert(tiles.size() == intiles.size() * channelMultiplier_);
    size_t chanoffset = 0;
    for (int mult=0; mult < channelMultiplier_; mult++) {
        for (size_t t=0; t < intiles.size(); t++) {
            tiles.at(t+chanoffset).toFloatVec(attrs0,offset0,4);
            intiles.at(t).toFloatVec(attrs0,offset0+2,4);
            offset0+=4*4;
        }
        chanoffset += intiles.size();
    }
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(attrs0, tiler_->numOutputTiles()*4*4*sizeof(float), GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0,4,GL_FLOAT,GL_FALSE,0,0);
    delete [] attrs0;
    //---------------------------------------------
    // Now indices for the bias texture and the row
    // indices for the convolution coeffs (y-part
    // of the convolution)...
    //---------------------------------------------
    int * attrs1 = new int[tiler_->numOutputTiles()*2*4];
    memset(attrs1, 0, tiler_->numOutputTiles()*2*4*sizeof(float));
    chanoffset = 0;
    for (int i=0; i < tiler_->numOutputTiles(); i++) {
        for (int j=0; j<4; j++) {
            attrs1[(i*4+j)*2+0] = (i % intiles.size());
            attrs1[(i*4+j)*2+1] = i;          // to be used for indexing bias texture
        }
        if ((i > 0) && ((i % intiles.size())==0)) chanoffset++;
    }
    textureOffsets_ = new VBO(context_);
    vao->enableArray(1);
    textureOffsets_->setBufferData(attrs1,tiler_->numOutputTiles()*2*4*sizeof(int),GL_STATIC_DRAW);
    textureOffsets_->bind();
    vao->setVertexAttributeBuffer(1,2,GL_INT,0,0);
    delete [] attrs1;
    //---------------------------------------------
    // VBO for optional residual input (to be added
    // to the output after BN/ReLU)
    //---------------------------------------------
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        assert(residualTiler_->numOutputTiles() == residualTiler_->numInputTiles());
        float * attrs2 = new float[residualTiler_->numInputTiles()*2*4];
        std::vector<DeepTiler::Tile> rtiles = residualTiler_->createInputTiles(0,0,0);
        int offset2 = 0;
        for (DeepTiler::Tile tile : rtiles) {
            tile.toFloatVec(attrs2,offset2,2);
            offset2 += 2*4;
        }
        residualBuffer_ = new VBO(context_);
        vao->enableArray(2);
        residualBuffer_->setBufferData(attrs2,residualTiler_->numInputTiles()*2*4*sizeof(float),GL_STATIC_DRAW);
        residualBuffer_->bind();
        vao->setVertexAttributeBuffer(2,2,GL_FLOAT,GL_FALSE,0,0);
        delete [] attrs2;
    }
    //---------------------------------------------
    // IBO part
    //---------------------------------------------
    GLshort * indices = new GLshort[tiler_->numOutputTiles()*6];
    indexBuffer_ = new IBO(context_);
    for (int i=0;i<tiler_->numOutputTiles();i++) {
        int offset = i*4;
        indices[i*6+0] = offset + 0;
        indices[i*6+1] = offset + 1;
        indices[i*6+2] = offset + 2;
        indices[i*6+3] = offset + 0;
        indices[i*6+4] = offset + 2;
        indices[i*6+5] = offset + 3;
    }
    indexBuffer_->setBufferData(indices,6*tiler_->numOutputTiles()*sizeof(GLshort),GL_STATIC_DRAW);
    indexBuffer_->bind();
    delete [] indices;
}


} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
