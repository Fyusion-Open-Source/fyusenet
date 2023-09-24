//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2022
//--------------------------------------------------------------------------------------------------
// Style-Transfer Network 3x3 (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <unordered_map>

//-------------------------------------- Project  Headers ------------------------------------------

#include <fyusenet/fyusenet.h>
#include <fyusenet/gl/gl_sys.h>
#include "stylenet_base.h"

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Neural network that implements a simplistic image style-transfer operation (3x3 convs)
 *
 * This class implements a basic image style-transfer operation based on 3x3 by convolution layers
 * and contains only the specialized setup for the convolution kernel size. All common/base functions
 * are implemented in the StyleNetBase class.
 *
 * @see StyleNetBase
 */
class StyleNet3x3 : public StyleNetBase {
 public:
    using layer_ids = StyleNet3x3Provider::layer_ids;
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    StyleNet3x3(int width, int height, bool upload, bool download, const fyusion::fyusenet::GfxContextLink& ctx = fyusion::fyusenet::GfxContextLink());
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void connectLayers(fyusion::fyusenet::CompiledLayers& layers, fyusion::fyusenet::BufferManager * buffers) override;
    fyusion::fyusenet::CompiledLayers buildLayers() override;
};

// vim: set expandtab ts=4 sw=4:
