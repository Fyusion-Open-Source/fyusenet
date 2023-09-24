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


class SingleWeightProvider : public fyusion::fyusenet::ParameterProvider {
 public:
    explicit SingleWeightProvider(const float *weights, const float * bias = nullptr, const float *bn = nullptr);
    ~SingleWeightProvider();
    [[nodiscard]] fyusion::fyusenet::DataBlob get(const std::string &name, int layerNo, int subIndex) const override;

 private:
    fyusion::fyusenet::DataWrapper * weights_ = nullptr;
    fyusion::fyusenet::DataWrapper * bias_ = nullptr;
    fyusion::fyusenet::DataWrapper * postNorm_ = nullptr;
};

/**
 * @brief Base class for individual layer unit-tests
 *
 * This class contains a few helper routines that can be used by derived test classes to make
 * life easier.
 */
class LayerTestBase {
  protected:

    virtual void cleanup() {
        if (!testTextures_.empty()) glDeleteTextures((GLsizei)testTextures_.size(), &testTextures_[0]);
    }

    void generateTextures(fyusion::fyusenet::gpu::GPULayerBase * layer, const std::vector<const float *> & inputs, const float * residual=nullptr, bool includesPadding=false);
    void generateSequenceTextures(fyusion::fyusenet::gpu::GPULayerBase * layer, int numTokens, const std::vector<const float *>& inputs, const float * residual);
    static float * generateConstantData(float content, int channels, int width, int height, int padding=0);
    static float * generateRandomData(int channels, int width, int height, float low, float high, int padding=0);
    static float * generateRandomIntegerData(int channels, int width, int height, float low, float high, int padding=0);
    static float * generateBilinearData(int channels, int width, int height, int padding=0);
    float * stackConvolution(float bias, const float * channelData, int kernelX, int kernelY, int inputChannels, int outputChannels);
    int copyToShallowTexture(const float *input, GLuint handle, int netwidth, int netheight, int padding, int chanOffset, int remchans);
    void copyToDeepTexture(const float * input, GLuint handle, fyusion::fyusenet::gpu::deep::DeepTiler *tiler, int netwidth, int netheight, int padding, int inChans, bool includesPadding);
    void copyToSequenceTexture(const float *input, GLuint handle, int width, int height, int numTokens);
    void configureTexture(GLuint tex, int width, int height, const void *data);
    void configureTexture(GLuint tex, int width, int height, GLint iformat, GLenum format, GLenum dtype, const void *data);

    [[nodiscard]] static GLuint getInputTexture(const fyusion::fyusenet::gpu::GPULayerBase * layer, int index) {
        EXPECT_NE(layer, nullptr);
        if (!layer) return 0;
        return layer->getInputTexture(index);
    }

    static void addInputTexture(fyusion::fyusenet::gpu::GPULayerBase * layer, GLuint tex, int index) {
        EXPECT_NE(layer, nullptr);
        if (!layer) return;
        layer->addInputTexture(tex, index);
    }

    static void addOutputTexture(fyusion::fyusenet::gpu::GPULayerBase * layer, GLuint tex, int index) {
        EXPECT_NE(layer, nullptr);
        if (!layer) return;
        layer->addOutputTexture(tex, index, 0);
    }

    [[nodiscard]] fyusion::opengl::FBO * getFBO(fyusion::fyusenet::gpu::GPULayerBase * layer, int index) {
        EXPECT_NE(layer, nullptr);
        if (!layer) return nullptr;
        return layer->getFBO(index);
    }

    std::vector<GLuint> testTextures_;
    fyusion::fyusenet::gpu::deep::DeepTiler * tiler_ = nullptr;            //!< (no ownership)
    fyusion::fyusenet::gpu::deep::DeepTiler * residualTiler_ = nullptr;    //!< (no ownership)

};


// vim: set expandtab ts=4 sw=4:

