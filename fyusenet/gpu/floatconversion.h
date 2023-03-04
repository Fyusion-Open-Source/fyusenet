//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Floating-Point Conversion Helper for OpenGL FP16 (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/gl_sys.h"
#include "../gl/fbo.h"
#include "../gl/shaderprogram.h"
#include "functionlayer.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {

/**
 * @brief Floating-point conversion helper
 *
 * Based on paper "Fast Half Float Conversion" by Jeroen van der Zijp
 * ftp://ftp.fox-toolkit.org/pub/fasthalffloatconversion.pdf
 */
class FloatConversion {
 public:

    unsigned int * toFP16UI(float *input,int entries) const;
    unsigned short * toFP16US(float *input,int entries) const;

    inline unsigned short toFP16(float fp) const {
        unsigned int f;
        *((float *)&f) = fp;
        unsigned short fp16 = baseTable_[(f>>23) & 0x1ff] + ((f & 0x007fffff) >> shiftTable_[(f>>23) & 0x1ff]);
        return fp16;
    }

    inline unsigned int toFP16(float fp1,float fp2) const {
        unsigned int f;
        *((float *)&f) = fp1;
        unsigned short fp16_1 = baseTable_[(f>>23) & 0x1ff]+((f & 0x007fffff) >> shiftTable_[(f>>23) & 0x1ff]);
        *((float *)&f) = fp2;
        unsigned short fp16_2 = baseTable_[(f>>23) & 0x1ff]+((f & 0x007fffff) >> shiftTable_[(f>>23) & 0x1ff]);
        return (fp16_1<<16) | fp16_2;
    }

    static inline FloatConversion * getInstance() {
        static FloatConversion singleton;
        return &singleton;
    }

 private:
    FloatConversion();
    static unsigned short baseTable_[512];
    static unsigned short shiftTable_[512];
    static unsigned char seed_[12];
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
