//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2022
//--------------------------------------------------------------------------------------------------
// Style-Transfer Network 9x9 (Header)
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
 * @brief Neural network that implements a simplistic image style-transfer operation (9x9 convs)
 *
 * This class implements a basic image style-transfer operation based on 9x9 by convolution layers
 * and contains only the specialized setup for the convolution kernel size. All common/base functions
 * are implemented in the StyleNetBase class.
 *
 * @see StyleNetBase
 */
class StyleNet9x9 : public StyleNetBase {
 public:
    using layer_ids = StyleNet9x9Provider::layer_ids;
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    StyleNet9x9(int width, int height, bool upload, bool download, const fyusion::fyusenet::GfxContextLink& ctx = fyusion::fyusenet::GfxContextLink());

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    fyusion::fyusenet::CompiledLayers buildLayers() override;
    void connectLayers(fyusion::fyusenet::CompiledLayers& layers, fyusion::fyusenet::BufferManager * buffers) override;
};

