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
 * @brief Neural network that implements a simplicistic image style-transfer operation (3x3 convs)
 *
 * This class implements a basic image style-transfer operation based on 3x3 by convolution layers
 * and contains only the specialized setup for the convolution kernel size. All common/base functions
 * are implemented in the StyleNetBase class.
 *
 * @see StyleNetBase
 */
class StyleNet3x3 : public StyleNetBase {
 public:
    /**
     * Indices for the layer numbers
     */
    enum {
        CONV2 = 2,
        CONV3,
        RES1_1,
        RES1_2,
        RES2_1,
        RES2_2,
        DECONV1,
        DECONV2,
        DECONV3,
        SIGMOID,
        DOWNLOAD
    };

    constexpr static int STYLENET_SIZE = 77235;   // number of floats per network weights/biases
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    StyleNet3x3(int width, int height, bool upload, bool download, const fyusion::fyusenet::GfxContextLink& ctx = fyusion::fyusenet::GfxContextLink());
    ~StyleNet3x3();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void loadWeightsAndBiases(float *weightsAndBiases, size_t size) override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void connectLayers(fyusion::fyusenet::CompiledLayers& layers, fyusion::fyusenet::BufferManager * buffers) override;
    virtual void initializeWeights(fyusion::fyusenet::CompiledLayers& layers) override;
    virtual fyusion::fyusenet::CompiledLayers buildLayers() override;
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    float * wbData_ = nullptr;              //!< Deep-copy of weight data supplied in #loadWeightsAndBiases_
};

// vim: set expandtab ts=4 sw=4:
