//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Number to RGB(A) Renderer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------


//------------------------------------------ Constants ---------------------------------------------


//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Helper class that renders numbers to an image
 */
class NumberRender {
 public:
    // ------------------------------------------------------------------------
    // Constructor/Destructor
    // ------------------------------------------------------------------------
    NumberRender(int width, int height, int scale, int numChannels=4);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    float * generate(uint16_t number, uint16_t aux);

 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void renderNumber(float * img, uint16_t number, int x, int y, float pixelValue, int numChannels=4);
    static int numDigits(uint16_t value);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int width_ = 0;             //!< Width of images
    int height_ = 0;            //!< Height of images
    int numChannels_ = 4;       //!< Number of channels in the images
    int scale_ = 0;             //!< Scale to use for the number size
};

// vim: set expandtab ts=4 sw=4:
