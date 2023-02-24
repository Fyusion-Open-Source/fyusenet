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
 * @brief Neural network that implements a simplicistic image style-transfer operation (9x9 convs)
 *
 * This class implements a basic image style-transfer operation based on 9x9 by convolution layers
 * and contains only the specialized setup for the convolution kernel size. All common/base functions
 * are implemented in the StyleNetBase class.
 *
 * @see StyleNetBase
 */
class StyleNet9x9 : public StyleNetBase {
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
        RES3_1,
        RES3_2,
        RES4_1,
        RES4_2,
        RES5_1,
        RES5_2,
        DECONV1,
        DECONV2,
        DECONV3,
        SIGMOID,
        DOWNLOAD
    };

    constexpr static int STYLENET_SIZE = 169059;   // number of floats per network weights/biases

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    StyleNet9x9(int width, int height, bool upload, bool download, const fyusion::fyusenet::GfxContextLink& ctx = fyusion::fyusenet::GfxContextLink());
    ~StyleNet9x9();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void loadWeightsAndBiases(float *weightsAndBiases, size_t size) override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void initializeWeights(fyusion::fyusenet::CompiledLayers& layers) override;
    virtual fyusion::fyusenet::CompiledLayers buildLayers() override;
    virtual void connectLayers(fyusion::fyusenet::CompiledLayers& layers, fyusion::fyusenet::BufferManager * buffers) override;
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    float * wbData_ = nullptr;              //!< Deep-copy of weight data supplied in #loadWeightsAndBiases_
};

