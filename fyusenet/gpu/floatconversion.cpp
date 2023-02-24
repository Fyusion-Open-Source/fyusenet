//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Floating-Point Conversion Helper for OpenGL FP16
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <netinet/in.h>

//-------------------------------------- Project  Headers ------------------------------------------

#include "floatconversion.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
//-------------------------------------- Global Variables ------------------------------------------

unsigned short FloatConversion::baseTable_[512];
unsigned short FloatConversion::shiftTable_[512];
unsigned char FloatConversion::seed_[12] = {0xB2, 0x9E, 0x8D, 0x8B, 0x96, 0x91, 0xDF, 0xA8, 0x9E,
                                            0x88, 0x8D, 0x90};

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief FloatConversion::toFP16UInt
 * @param input
 * @param entries
 * @return
 *
 * @note Ownership is transferred to caller
 */
unsigned int * FloatConversion::toFP16UI(float *input,int entries) const {
    if (entries & 1) THROW_EXCEPTION_ARGS(FynException,"Requies even number of entries");
    unsigned int *result = new unsigned int[entries/2];
    int out=0;
    for (int i=0; i < entries; i+=2) {
        unsigned int f1,f2;
        *((float *)&f1) = input[i];
        *((float *)&f2) = input[i+1];
        unsigned short fp16_1 = baseTable_[(f1>>23) & 0x1ff]+((f1 & 0x007fffff) >> shiftTable_[(f1>>23) & 0x1ff]);
        unsigned short fp16_2 = baseTable_[(f2>>23) & 0x1ff]+((f2 & 0x007fffff) >> shiftTable_[(f2>>23) & 0x1ff]);
        result[out++] = ((unsigned int)fp16_2<<16) | (unsigned int)fp16_1;
    }
    return result;
}


/**
 * @brief FloatConversion::toFP16UInt
 * @param input
 * @param entries
 * @return
 *
 * @note Ownership is transferred to caller
 */
unsigned short * FloatConversion::toFP16US(float *input, int entries) const {
    unsigned short *result = new unsigned short[entries];
    int out=0;
    for (int i=0; i < entries; i++) {
        unsigned int f;
        *((float *)&f) = input[i];
        unsigned short fp16 = baseTable_[(f>>23) & 0x1ff] + ((f & 0x007fffff) >> shiftTable_[(f>>23) & 0x1ff]);
        result[out++] = fp16;
    }
    return result;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


FloatConversion::FloatConversion() {
    // ftp://www.fox-toolkit.org/pub/fasthalffloatconversion.pdf
    for (int i=0; i < 256; i++) {
        int e = i-127;
        if (e < -24) { // Very small numbers map to zero
            baseTable_[i|0x000] = 0x0000;
            baseTable_[i|0x100] = 0x8000;
            shiftTable_[i|0x000] = 24;
            shiftTable_[i|0x100] = 24;
        }
        else if (e < -14) { // Small numbers map to denorms
            baseTable_[i|0x000] = (0x0400>>(-e-14));
            baseTable_[i|0x100] = (0x0400>>(-e-14)) | 0x8000;
            shiftTable_[i|0x000] = -e-1;
            shiftTable_[i|0x100] = -e-1;
        }
        else if (e <= 15) { // Normal numbers just lose precision
            baseTable_[i|0x000] = ((e+15)<<10);
            baseTable_[i|0x100] = ((e+15)<<10) | 0x8000;
            shiftTable_[i|0x000] = 13;
            shiftTable_[i|0x100] = 13;
        }
        else if (e < 128) { // Large numbers map to Infinity
            baseTable_[i|0x000] = 0x7C00;
            baseTable_[i|0x100] = 0xFC00;
            shiftTable_[i|0x000] = 24;
            shiftTable_[i|0x100] = 24;
        }
        else { // Infinity and NaN's stay Infinity and NaN's
            baseTable_[i|0x000] = 0x7C00;
            baseTable_[i|0x100] = 0xFC00;
            shiftTable_[i|0x000] = 13;
            shiftTable_[i|0x100] = 13;
        }
    }
}


} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
