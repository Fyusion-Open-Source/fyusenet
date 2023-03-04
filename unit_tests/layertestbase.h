//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Base class for misc. layer testing (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include <gtest/gtest.h>
#include <fyusenet/fyusenet.h>
#include "gltesthelpers.h"
#include <fyusenet/gpu/convlayerbase.h>
#include <fyusenet/gpu/deep/deepconvlayerbase.h>

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Base class for individual layer unit-tests
 *
 * This class contains a few helper routines that can be used by derived test classes to make
 * life easier.
 */
class LayerTestBase {
  protected:

    virtual void cleanup() {
        if (testTextures_.size() > 0) glDeleteTextures(testTextures_.size(), &testTextures_[0]);
    }

    virtual void generateTextures(fyusion::fyusenet::gpu::GPULayerBase * layer, const std::vector<const float *> & inputs, const float * residual=nullptr, bool includesPadding=false);
    static float * generateConstantData(float content, int channels, int width, int height, int padding=0);
    static float * generateRandomData(int channels, int width, int height, float low, float high, int padding=0);
    static float * generateRandomIntegerData(int channels, int width, int height, float low, float high, int padding=0);
    static float * generateBilinearData(int channels, int width, int height, int padding=0);
    virtual float * stackConvolution(float bias, const float * channelData, int kernelX, int kernelY, int inputChannels, int outputChannels);


    std::vector<GLuint> testTextures_;
    fyusion::fyusenet::gpu::deep::DeepTiler * tiler_ = nullptr;            //!< (no ownership)
    fyusion::fyusenet::gpu::deep::DeepTiler * residualTiler_ = nullptr;    //!< (no ownership)

};





// vim: set expandtab ts=4 sw=4:

