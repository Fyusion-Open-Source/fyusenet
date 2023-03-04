//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Layer Tiler
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <algorithm>
#include <cmath>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gpulayerbase.h"
#include "deeptiler.h"

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
 * @brief Boring constructor that does nothing
 */
DeepTiler::DeepTiler() {
}


/**
 * @brief Constructor
 *
 * @param ltype Layer type that this tiler is going to be used for
 * @param width Tensor width to maintain tiling for (input side, perhaps output side, see down/upsampling)
 * @param height Tensor height to maintain tiling for (input side, perhaps output side, see down/upsampling)
 * @param inputChannels Number of input channels
 * @param outputChannels Number of output channels
 * @param hscale Horizontal scaling for scaling-type layers
 * @param vscale Vertical scaling for scaling-type layers
 * @param inputPadding Spatial padding on the input tensor (always symmetric)
 * @param outputPadding Spatial padding on the output tensor (always symmetric)
 * @param horizDown Horizontal down-scaling factor
 * @param vertDown Vertical down-scaling factor
 * @param horizUp Horizontal up-scaling factor
 * @param vertUp Vertical up-scaling factor
 * @param kernel Kernel size (isotropic) for convolution-type layers
 *
 * Initializes a tiler object with the supplied parameters, performs tiling and viewport
 * computations.
 */
DeepTiler::DeepTiler(LayerType ltype, int width, int height, int inputChannels, int outputChannels,
                     float hscale, float vscale, int inputPadding, int outputPadding,
                     int horizDown, int vertDown, int horizUp, int vertUp, int kernel) :
    width_(width), height_(height), inputChannels_(inputChannels),
    outputChannels_(outputChannels), kernel_(kernel), layer_(ltype) {
    outputWidth_ = (int)((float)width_ * hscale);
    outputHeight_ = (int)((float)height_ * vscale);
    assert(ltype != LayerType::ILLEGAL);
    if (ltype == LayerType::TRANSCONVOLUTION2D) {
        outputWidth_ += kernel_ - horizUp;
        outputHeight_ += kernel_ - vertUp;
    }
    downsample_[0] = horizDown;
    downsample_[1] = vertDown;
    upsample_[0] = horizUp;
    upsample_[1] = vertUp;
    inputPadding_ = inputPadding;
    outputPadding_ = outputPadding;
    inputTiles_ = (inputChannels + (LayerBase::PIXEL_PACKING-1)) / LayerBase::PIXEL_PACKING;
    outputTiles_ = (outputChannels + (LayerBase::PIXEL_PACKING-1)) / LayerBase::PIXEL_PACKING;
    std::pair<int,int> intile = CPUBufferShape::computeDeepTiling(inputChannels);
    inputTiling_[0] = intile.first;
    inputTiling_[1] = intile.second;
    inputTiling_[2] = 1;
    std::pair<int,int> outtile = CPUBufferShape::computeDeepTiling(outputChannels);
    outputTiling_[0] = outtile.first;
    outputTiling_[1] = outtile.second;
    outputTiling_[2] = 1;
    viewport_[0] = outputTiling_[0] * (outputWidth_ + outputPadding_) + outputPadding_;
    viewport_[1] = outputTiling_[1] * (outputHeight_ + outputPadding_) + outputPadding_;
    inputSize_[0] = inputTiling_[0] * (width_ + inputPadding_) + inputPadding_;
    inputSize_[1] = inputTiling_[1] * (height_ + inputPadding_) + inputPadding_;
}



/**
 * @brief Compute a set of tiles for the output tensor configuration
 *
 * @return Array (vector) of DeepTiler::Tile objects which represent the output tiles to be
 *         rendered to the output texture
 *
 * This function computes a set of output tiles, consisting of the output geometry of each tile,
 * usually represented by a single quadrilateral (actually a pair of triangles). The resulting
 * tiles feature image and device coordinates that can be used for polygon setup.
 */
std::vector<DeepTiler::Tile> DeepTiler::createOutputTiles() const {
    std::vector<Tile> result;
    float tilewidth = (float)outputWidth_;
    float tileheight = (float)outputHeight_;
    int itilewidth = outputWidth_;
    int itileheight = outputHeight_;
    float xextent = (2.0f*tilewidth) / viewport_[0];
    float yextent = (2.0f*tileheight) / viewport_[1];
    int tilenum=0;
    for (int y=0; y < outputTiling_[1]; y++) {
        float by = ((2.0f*((y * (tileheight+outputPadding_)) + outputPadding_)) / (float)viewport_[1])-1.0f;
        for (int x=0; x < outputTiling_[0]; x++) {
            Tile t;
            float bx = ((2.0f*((x * (tilewidth+outputPadding_)) + outputPadding_)) / (float)viewport_[0])-1.0f;
            t.quad_[0*2+0] = bx;
            t.quad_[0*2+1] = by;
            t.quad_[1*2+0] = bx;
            t.quad_[1*2+1] = by+yextent;
            t.quad_[2*2+0] = bx+xextent;
            t.quad_[2*2+1] = by+yextent;
            t.quad_[3*2+0] = bx+xextent;
            t.quad_[3*2+1] = by;
            t.imageCoords_[0] = x*(itilewidth + outputPadding_) + outputPadding_;
            t.imageCoords_[1] = y*(itileheight + outputPadding_) + outputPadding_;
            t.imageExtents_[0] = itilewidth;
            t.imageExtents_[1] = itileheight;
            t.textureID_ = 0;
            t.channels_ = LayerBase::PIXEL_PACKING;
            result.push_back(t);
            tilenum++;
            if (tilenum >= outputTiles_) break;
        }
        if (tilenum >= outputTiles_) break;
    }
    return result;
}


/**
 * @brief Compute a set of tiles for the output tensor configuration
 *
 * @param xPixelOffset For convolution layers, provides the convolution offset along the x-axis
 * @param yPixelOffset For convolution layers, provides the convolution offset along the y-axis
 * @param texID GL texture ID/handle to write into the resulting tiles
 *
 * @return Array (vector) of DeepTiler::Tile objects which represent the input tiles to be mapped
 *         to the output polygons
 *
 * Creates an array of input tiles, where each tile represent a quadrilateral on the input texture
 * that is to be mapped to an output polygon/tile.
 *
 * @todo Fractional step convolution support
 */
std::vector<DeepTiler::Tile> DeepTiler::createInputTiles(int xPixelOffset, int yPixelOffset, int texID) const {
    std::vector<Tile> result;
    float tilewidth = (float)width_;
    float tileheight = (float)height_;
    float xextent = tilewidth / (float)inputSize_[0];
    float yextent = tileheight / (float)inputSize_[1];
    float dx = (globalPooling_) ? 0.0f : 0.5f * (float)(downsample_[0]-1);
    float dy = (globalPooling_) ? 0.0f : 0.5f * (float)(downsample_[1]-1);
    int tilenum=0;
    int remchannels = inputChannels_;
    for (int y=0; y < inputTiling_[1]; y++) {
        float by = (y * (tileheight + (float)inputPadding_) + (float)(inputPadding_ + yPixelOffset) - dy) / (float) inputSize_[1];
        for (int x=0; x < inputTiling_[0]; x++) {
            Tile t;
            float bx = (x * (tilewidth + (float)inputPadding_) + (float)(inputPadding_ + xPixelOffset) - dx) / (float) inputSize_[0];
            t.textureID_ = texID;
            t.quad_[0*2+0] = bx;
            t.quad_[0*2+1] = by;
            t.quad_[1*2+0] = bx;
            t.quad_[1*2+1] = by+yextent;
            t.quad_[2*2+0] = bx+xextent;
            t.quad_[2*2+1] = by+yextent;
            t.quad_[3*2+0] = bx+xextent;
            t.quad_[3*2+1] = by;
            t.imageCoords_[0] = x*(width_+inputPadding_)+inputPadding_;
            t.imageCoords_[1] = y*(height_+inputPadding_)+inputPadding_;
            t.imageExtents_[0] = width_;
            t.imageExtents_[1] = height_;
            t.lowClamp_[0] = bx;
            t.lowClamp_[1] = by;
            t.hiClamp_[0] = bx+xextent;
            t.hiClamp_[1] = by+yextent;
            t.channels_ = (remchannels >= LayerBase::PIXEL_PACKING) ? LayerBase::PIXEL_PACKING : remchannels;
            result.push_back(t);
            tilenum++;
            remchannels -= LayerBase::PIXEL_PACKING;
            if (tilenum >= inputTiles_) break;
        }
        if (tilenum >= inputTiles_) break;
    }
    return result;
}


/**
 * @brief Retrieve horizontal texture step-size for shader
 *
 * @return Spacing (in normalized texture coordinates) for two horizontal texels
 */
float DeepTiler::getTextureStepX() const {
    return 1.0f / (float)(inputSize_[0]);
}


/**
 * @brief Retrieve vertical texture step-size for shader
 *
 * @return Spacing (in normalized texture coordinates) for two vertical texels
 */
float DeepTiler::getTextureStepY() const {
    return 1.0f / (float)(inputSize_[1]);
}


/**
 * @brief Retrieve total viewport width for output rendering
 *
 * @return Viewport width in pixels
 */
int DeepTiler::getViewportWidth() const {
    return viewport_[0];
}


/**
 * @brief Retrieve total viewport height for output rendering
 *
 * @return Viewport height in pixels
 */
int DeepTiler::getViewportHeight() const {
    return viewport_[1];
}


/**
 * @brief Retrieve total input width of input-tensor texture
 *
 * @return Width of input-tensor texture, including padding
 *
 * @warning This is \b not necessarily the width of the input tensor, it is the actual width of the
 *          texture where the tiling was performed on
 */
int DeepTiler::getInputTextureWidth() const {
    return inputSize_[0];
}


/**
 * @brief Retrieve total input height of input-tensor texture
 *
 * @return Height of input-tensor texture, including padding
 *
 * @warning This is \b not necessarily the height of the input tensor, it is the actual height of the
 *          texture where the tiling was performed on
 */
int DeepTiler::getInputTextureHeight() const {
    return inputSize_[1];
}


/**
 * @brief Get width of single output tile (spatial tensor width) without any padding
 *
 * @return Output tile width (net tensor width, without padding)
 */
int DeepTiler::getOutputWidth() const {
    return outputWidth_;
}


/**
 * @brief Get height of single output tile (spatial tensor height) without any padding
 *
 * @return Output tile height (net tensor height, without padding)
 */
int DeepTiler::getOutputHeight() const {
    return outputHeight_;
}


/**
 * @brief Retrieve number of input channels
 *
 * @return Number of input channels
 */
int DeepTiler::getInputChannels() const {
    return inputChannels_;
}


/**
 * @brief Retrieve number of output channels
 *
 * @return Number of output channels
 */
int DeepTiler::getOutputChannels() const {
    return outputChannels_;
}


/**
 * @brief Obtain width of input texture tile
 *
 * @return Spatial width of a single input tile
 */
int DeepTiler::getInputWidth() const {
    return width_;
}


/**
 * @brief Obtain height of input texture tile
 *
 * @return Spatial height of a single input tile
 */
int DeepTiler::getInputHeight() const {
    return height_;
}


/**
 * @brief Retrieve number of input tiles
 *
 * @param mode Retrieval mode, see long description
 *
 * @return Number of tiles, according to \p mode
 *
 * Returns the number of input tiles according to the supplied \p mode:
 *   - \c HORIZONTAL for horizontal number of tiles (# columns)
 *   - \c VERTICAL for vertical number of tiles (# rows)
 *   - \c ALL for cartesian product of horizontal/vertical tiles
 */
int DeepTiler::numInputTiles(tx mode) const {
    switch (mode) {
        case HORIZONTAL:
            return inputTiling_[0];
        case VERTICAL:
            return inputTiling_[1];
        default:
            return inputTiles_;
    }
}


/**
 * @brief Retrieve number of output tiles
 *
 * @param mode Retrieval mode, see long description
 *
 * @return Number of tiles, according to \p mode
 *
 * Returns the number of output tiles according to the supplied \p mode:
 *   - \c HORIZONTAL for horizontal number of tiles (# columns)
 *   - \c VERTICAL for vertical number of tiles (# rows)
 *   - \c ALL for cartesian product of horizontal/vertical tiles
 */
int DeepTiler::numOutputTiles(tx mode) const {
    switch (mode) {
        case HORIZONTAL:
            return outputTiling_[0];
        case VERTICAL:
            return outputTiling_[1];
        default:
            return outputTiles_;
    }
}


/**
 * @brief Enter internal quadrilateral coordinates directly into target array
 *
 * @param[out] tgt Pointer to target array to store the data in
 * @param offset Offset within \p tgt to start writing at
 * @param stride Stride to add to the writing position after each x/y pair
 * @param transpose Write the quad in "transposed" form (i.e. backwards)
 *
 * This convenience function transfers the coordinates of the quad that it stored internally
 * directly into the supplied \p tgt array. Each coordinate (x/y pair) is written in
 * direct sequence and a \p stride can be specified, that is added to the writing position after
 * each pair. For a directly sequential writing of all coordinates, a stride of 2 has to be
 * specified.
 *
 * The regular order of the quad is given by:
 *   - top/left
 *   - bottom/left
 *   - bottom/right
 *   - top/right
 * in terms of x-coordinates decreasing to the left and y-coordinates decreasing to the top. Note
 * that in GL device coordinates the y-axis decreases to the bottom.
 */
void DeepTiler::Tile::toFloatVec(float *tgt, int offset, int stride, bool transpose) const {
    assert(stride != 0);
    if (transpose) {
        tgt[offset]   = quad_[0];
        tgt[offset+1] = quad_[1];
        offset += stride;
        tgt[offset]   = quad_[6];
        tgt[offset+1] = quad_[7];
        offset += stride;
        tgt[offset]   = quad_[4];
        tgt[offset+1] = quad_[5];
        offset += stride;
        tgt[offset]   = quad_[2];
        tgt[offset+1] = quad_[3];
    } else {
        for (int i=0; i < 4; i++) {
            tgt[offset] = quad_[i*2];
            tgt[offset+1] = quad_[i*2+1];
            offset += stride;
        }
    }
}


/**
 * @brief Retrieve midpoint coordinates for a tile
 *
 * @return Pair of x/y midpoint coordinates
 *
 * This convenience function computes the midpoint of a tile, to be used for point-based rendering
 * on 1x1 data.
 */
std::pair<float,float> DeepTiler::Tile::midPoint() const {
    float midx=0.f, midy=0.f;
    for (int i=0; i < 4; i++) {
        midx += quad_[i*2];
        midy += quad_[i*2+1];
    }
    return std::pair<float,float>(midx/4.0f, midy/4.0f);
}


/**
 * @brief Create an (input) tile with a unit-texture quadrilateral
 *
 * @return Tile instance that contains a unit-texture quadrilateral
 */
DeepTiler::Tile DeepTiler::getUnitTextureExtents() {
    Tile result;
    result.quad_[0*2+0] = 0.0f;
    result.quad_[0*2+1] = 0.0f;
    result.quad_[1*2+0] = 0.0f;
    result.quad_[1*2+1] = 1.0f;
    result.quad_[2*2+0] = 1.0f;
    result.quad_[2*2+1] = 1.0f;
    result.quad_[3*2+0] = 1.0f;
    result.quad_[3*2+1] = 0.0f;
    result.lowClamp_[0] = 0.0f;
    result.lowClamp_[1] = 0.0f;
    result.hiClamp_[0] = 1.0f;      // NOTE (mw) only valid for transpose conv
    result.hiClamp_[1] = 1.0f;
    return result;
}


/**
 * @brief Create an input tile with texture extents according to stored tensor parameters
 *
 * @return Tile object with default texture extents for a single input tile according to the
 *         wrapped input/output tensor combination
 */
DeepTiler::Tile DeepTiler::getDefaultTextureExtents() const {
    Tile result;
    float tilewidth = (float)width_;
    float tileheight = (float)height_;
    float xextent = tilewidth / (float)inputSize_[0];
    float yextent = tileheight / (float)inputSize_[1];
    float dx = (globalPooling_) ? 0.0f : 0.5f * (float)(downsample_[0]-1);
    float dy = (globalPooling_) ? 0.0f : 0.5f * (float)(downsample_[1]-1);
    float bx = ((float)inputPadding_ - dx) / (float) inputSize_[0];
    float by = ((float)inputPadding_ - dy) / (float) inputSize_[1];
    result.quad_[0*2+0] = bx;
    result.quad_[0*2+1] = by;
    result.quad_[1*2+0] = bx;
    result.quad_[1*2+1] = by+yextent;
    result.quad_[2*2+0] = bx+xextent;
    result.quad_[2*2+1] = by+yextent;
    result.quad_[3*2+0] = bx+xextent;
    result.quad_[3*2+1] = by;
    result.lowClamp_[0] = bx;
    result.lowClamp_[1] = by;
    result.hiClamp_[0] = bx+xextent;      // NOTE (mw) only valid for transpose conv
    result.hiClamp_[1] = by+yextent;
    return result;
}


/**
 * @brief Retrieve lower clamping values for texture tile coordinates
 *
 * @param[out] tgt Target array to write data to
 * @param offset Offset within the supplied \p target to write data to
 *
 * This function writes two values, low clamp for left and low clamp for top, to the supplied
 * \p tgt array.
 */
void DeepTiler::Tile::lowClamp(float *tgt, int offset) const {
    tgt[offset++] = lowClamp_[0];
    tgt[offset++] = lowClamp_[1];
}


/**
 * @brief Write coordinates into target arrays as displacements to default texture extents
 *
 * @param defaultExtents Default tile extents that the data in this tile should displace
 * @param[out] tgt Pointer to target array
 * @param offset Offset within target array
 */
void DeepTiler::Tile::toDisplacement(const Tile& defaultExtents,float *tgt, int offset) const {
    tgt[offset++] = quad_[0] - defaultExtents.quad_[0];
    tgt[offset++] = quad_[1] - defaultExtents.quad_[1];
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/



} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
