//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Layer Tiler (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>
#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../base/layerflags.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu::deep {

/**
 * @brief Management class for texture tiles in deep neural networks
 *
 * This class handles the internal tiling of textures for deep-channel tensors. These are usually
 * represented as \e tiles on a larger texture, where each tile contains data for 4 channels.
 * In order to unify tiling layout and handling, this class is to be used <i>at all times</i>.
 *
 * Tiles are laid out based on the number of total channels and the tiler tries to maintain a
 * decent aspect ratio of the resulting texture while fitting the tiles.
 *
 * @todo Support fractional-step convolution
 */
class DeepTiler {
 public:

    /**
     * @brief Tile query mode
     *
     * @see #numInputTiles, #numOutputTiles
     */
    enum tx {
        ALL,                //!< Query total amount of tiles (width * height)
        HORIZONTAL,         //!< Query only horizontal amount of tiles (tile columns)
        VERTICAL            //!< Query only vertical amount of tiles (tile rows)
    };

     /**
     * @brief Representation of a single tile on a texture
     *
     * This object stores the geometric parameters for a single tile and offers a few convenience
     * functions for access.
     * Depending on whether or not this object is used for output or input purposes, the data stored
     * in #quad_ is either device coordinates for output or texture coordinates for input.
     */
    struct Tile {
        Tile() = default;

        void toFloatVec(float *tgt, int offset, int stride=2, bool transpose=false) const;
        void lowClamp(float *tgt, int offset) const;
        void toDisplacement(const Tile& defaultExtents, float *tgt, int offset) const;
        [[nodiscard]] std::pair<float,float> midPoint() const;

        int renderTarget_ = 0;       //!< For later expansion
        int textureID_ = 0;          //!< For texture tagging
        int channels_ = 0;           //!< Number of channels for this (and the other) tile
        float quad_[4*2];            //!< Device coordinates for drawing a quad for this tile (for output tiles) / texture coordinates for reading a tile from the input texture (for input tiles)
        float lowClamp_[2];          //!< Clamping values for the tile (left/top) such that the access does not bleed into a neighboring tile, used on input tiles
        float hiClamp_[2];           //!< Clamping values for the tile (right/bottom) such that access does not bleed into a neighboring tile, use in input files
        int imageCoords_[2];         //!< Top/left corner of tile in pixel coordinates (including padding)
        int imageExtents_[2];        //!< Width/Height of tile in pixel coordinates (excluding padding)
    };

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepTiler() = default;
    DeepTiler(LayerType ltype, int width, int height, int inputChannels, int outputChannels,
              float hscale=1.0f, float vscale=1.0f, int inputPadding=0, int outputPadding=0,
              int horizDown=1, int vertDown=1, int horizUp=1, int vertUp=1, int kernel=1);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] std::vector<Tile> createOutputTiles() const;
    [[nodiscard]] std::vector<Tile> createInputTiles(int xPixelOffset,int yPixelOffset,int texID=0) const;
    [[nodiscard]] Tile getDefaultTextureExtents() const;
    [[nodiscard]] static Tile getUnitTextureExtents();
    [[nodiscard]] int getViewportWidth() const;
    [[nodiscard]] int getViewportHeight() const;
    [[nodiscard]] int getInputTextureWidth() const;
    [[nodiscard]] int getInputTextureHeight() const;
    [[nodiscard]] int getInputChannels() const;
    [[nodiscard]] int getOutputChannels() const;
    [[nodiscard]] int getInputWidth() const;
    [[nodiscard]] int getInputHeight() const;
    [[nodiscard]] int numInputTiles(tx mode = ALL) const;
    [[nodiscard]] int numOutputTiles(tx mode = ALL) const;
    [[nodiscard]] int getOutputWidth() const;
    [[nodiscard]] int getOutputHeight() const;
    [[nodiscard]] float getTextureStepX() const;
    [[nodiscard]] float getTextureStepY() const;

    /**
     * @brief Set global pooling mode on tiler
     */
    void setGlobalPooling() {
        globalPooling_ = true;
    }

    /**
     * @brief Check if the tiler generates data for a (spatial) pooling layer
     *
     * @retval true if the tiler is for a pooling layer
     * @retval false otherwise
     */
    bool isPooling() const {
        return (layer_ == LayerType::MAXPOOL2D) || (layer_ == LayerType::AVGPOOL2D);
    }
 private:

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int width_ = 0;                 //!< Width of one input channel (actual net payload, no padding, no texture tricks)
    int height_ = 0;                //!< Height of one input channel (actual net payload, no padding, no texture tricks)
    int inputPadding_ = 0;          //!< Padding on the spatial extents of each channel/tile for the input
    int outputPadding_ = 0;         //!< Padding on the spatial extents of each channel/tile for the output
    int outputWidth_ = 0;           //!< Width of one output channel (actual net payload, no padding, no texture tricks)
    int outputHeight_ = 0;          //!< Height of one output channel (actual net payload, no padding, no texture tricks)
    int inputChannels_ = 0;         //!< Number of input tensor channels
    int outputChannels_ = 0;        //!< Number of output tensor channels
    int inputTiles_ = 0;            //!< Total number of tiles (input side), each tile represents 4 channels
    int outputTiles_ = 0;           //!< Total number of tiles (output side), each tile represents 4 channels
    int inputTiling_[3];            //!< Number of input tiles in x/y direction + potential multi-texture dimension
    int outputTiling_[3];           //!< Number of output tiles in x/y direction + potential MRT dimension
    int kernel_ = 1;                //!< For convolution-type layers, defines the (isotropic) convolution kernel size
    int viewport_[2];               //!< Viewport size to use for rendering (per render-target)
    int inputSize_[2];              //!< Total input texture size (per render-target)
    int downsample_[2];             //!< Downsampling rate for downsampling layers (spatial x/y direction)
    int upsample_[2];               //!< Upsampling rate for upsampling layers (spatial x/y direction)
    bool globalPooling_ = false;    //!< Indicator if the underlying operation is a global pooling operation
    /**
     * @brief The type of layer that this tiler should be used for
     */
    LayerType layer_ = LayerType::ILLEGAL;
};

} // fyusion::fyusenet::gpu::deep namespace

// vim: set expandtab ts=4 sw=4:
