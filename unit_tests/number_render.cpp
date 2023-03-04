//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Number to RGB(A) Renderer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cmath>
#include <cassert>
#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "number_render.h"

//-------------------------------------- Global Variables ------------------------------------------

//-------------------------------------- Local Definitions -----------------------------------------

namespace internal {
const char *zero = " XXXXX  "
                   "XX   XX "
                   "XX  XXX "
                   "XX XXXX "
                   "XXXX XX "
                   "XXX  XX "
                   " XXXXX  "
                   "        ";

const char * one =  "  XX    "
                    " XXX    "
                    "  XX    "
                    "  XX    "
                    "  XX    "
                    "  XX    "
                    "XXXXXX  "
                    "        ";

const char * two =  " XXXX   "
                    "XX  XX  "
                    "    XX  "
                    "  XXX   "
                    " XX     "
                    "XX  XX  "
                    "XXXXXX  "
                    "        ";

const char * three = " XXXX   "
                     "XX  XX  "
                     "    XX  "
                     "  XXX   "
                     "    XX  "
                     "XX  XX  "
                     " XXXX   "
                     "        ";

const char * four = "   XXX  "
                    "  XXXX  "
                    " XX XX  "
                    "XX  XX  "
                    "XXXXXXX "
                    "    XX  "
                    "   XXXX "
                    "        ";

const char * five = "XXXXXX  "
                    "XX      "
                    "XXXXX   "
                    "    XX  "
                    "    XX  "
                    "XX  XX  "
                    " XXXX   "
                    "        ";

const char * six =  "  XXX   "
                    " XX     "
                    "XX      "
                    "XXXXX   "
                    "XX  XX  "
                    "XX  XX  "
                    " XXXX   "
                    "        ";

const char * seven = "XXXXXX  "
                     "XX  XX  "
                     "    XX  "
                     "   XX   "
                     "  XX    "
                     "  XX    "
                     "  XX    "
                     "        ";

const char * eight = " XXXX   "
                     "XX  XX  "
                     "XX  XX  "
                     " XXXX   "
                     "XX  XX  "
                     "XX  XX  "
                     " XXXX   "
                     "        ";

const char * nine = " XXXX   "
                    "XX  XX  "
                    "XX  XX  "
                    " XXXXX  "
                    "    XX  "
                    "   XX   "
                    " XXX    "
                    "        ";

const char * numbers[10] = {
    zero,
    one,
    two,
    three,
    four,
    five,
    six,
    seven,
    eight,
    nine
};

} // internal namespace

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

NumberRender::NumberRender(int width, int height, int scale, int numChannels) :
      width_(width), height_(height), numChannels_(numChannels), scale_(scale) {
    assert(height > 8*scale + 12);
}


/**
 * @brief Generate image representation of provided number / aux pair
 *
 * @param number Number to render (centered)
 * @param aux Additional number (upper right corner)
 *
 * @return Pointer to image data, ownership transferred to caller
 */
float * NumberRender::generate(uint16_t number, uint16_t aux) {
    float * img = new float[width_ * height_ * numChannels_];
    memset(img, 0, width_ * height_ * numChannels_ * sizeof(float));
    int nwidth = numDigits(number) * (8*scale_ + 4);
    assert(width_ >= nwidth);
    int awidth = numDigits(aux) * (8*scale_ + 4);
    assert(width_ >= awidth+32);
    int cx = (width_ - nwidth) / 2;
    int cy = (height_ - (8 * scale_)) / 2;
    int ax = (width_ - awidth - 32);
    int ay = 12;
    renderNumber(img, number, cx, cy, 1.f, numChannels_);
    renderNumber(img, aux, ax, ay, 1.f, numChannels_);
    return img;
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Crude way to determine number of digits in a 16-bit (positive) integer value
 *
 * @param value Number to determine digits for
 *
 * @return Number of digits in number
 */
int NumberRender::numDigits(uint16_t value) {
    if (value < 10) return 1;
    if (value < 100) return 2;
    if (value < 1000) return 3;
    if (value < 10000) return 4;
    return 5;
}


/**
 * @brief Render a multi-digit number to a position on the supplied image plane
 *
 * @param plane Pointer to (fp) image plane
 *
 * @param number Number to render
 *
 * @param x X-position of the number
 *
 * @param y Y-position of the number
 *
 * @param pixelValue Value to use for set pixels
 *
 * @param numChannels Number of channels in the supplied plane
 */
void NumberRender::renderNumber(float * plane, uint16_t number, int x, int y, float pixelValue, int numChannels) {
    assert(plane);
    int stride = width_;
    for (int dpos = numDigits(number); dpos > 0; dpos--) {
        int digit = (number / ((int)pow(10,dpos-1))) % 10;
        assert(digit>=0);
        assert(digit<=9);
        const char * tpl = internal::numbers[digit];
        for (int ny=0, yi=y; ny < 8; ny++, yi+=scale_) {
            for (int nx=0, xi=x; nx < 8; nx++, xi+=scale_) {
                if (tpl[ny*8+nx] == 'X') {
                    for (int ys=0; ys < scale_; ys++) {
                        for (int xs=0; xs < scale_; xs++) {
                            for(int c=0; c<numChannels; c++) {
                                plane[((yi+ys)*stride+(xi+xs))*numChannels+c] = pixelValue;
                            }
                        }
                    }
                }
            }
        }
        x += 8*scale_ + 4;
    }
}

// vim: set expandtab ts=4 sw=4:
