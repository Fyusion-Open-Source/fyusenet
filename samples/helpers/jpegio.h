//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2022
//--------------------------------------------------------------------------------------------------
// Barebones JPEG I/O (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>

//-------------------------------------- Project  Headers ------------------------------------------


//------------------------------------- Public Declarations ----------------------------------------


/**
 * @brief Simple JPEG reader/writer
 */
class JPEGIO {
 public:
    static bool isJPEG(const std::string& name);
    static void saveRGBImage(const uint8_t *img, int width, int height,const std::string& name,int quality=90);
    static uint8_t * loadRGBImage(const std::string& name, int & width, int & height);
};


// vim: set expandtab ts=4 sw=4:
