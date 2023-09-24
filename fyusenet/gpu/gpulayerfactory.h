//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// GPU Layer Factory (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../base/layerfactory.h"
#include "../gpu/gfxcontextlink.h"
#include "gpulayerbuilder.h"
#include "convlayerbuilder.h"
#include "concatlayerbuilder.h"
#include "scalelayerbuilder.h"
#include "poollayerbuilder.h"
#include "argmaxlayerbuilder.h"
#include "blurlayerbuilder.h"
#include "imgextractlayerbuilder.h"
#include "singleton_arithlayerbuilder.h"
#include "castlayerbuilder.h"
#include "customlayerbuilder.h"
#include "transposelayerbuilder.h"
#include "updownlayerbuilder.h"
#include "embeddinglayerbuilder.h"
#include "attentionlayerbuilder.h"
#include "linearlayerbuilder.h"
#include "tokenscoringlayerbuilder.h"

namespace fyusion::fyusenet::gpu {

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Producer backend for GPU-based network layers
 *
 * This class serves as backend for "vanilla" GPU-based layers. "Vanilla" GPU layers contain
 * quite generic shaders and tricks that are not optimized for a particular type or flavour
 * of GPU.
 */
class GPULayerFactoryBackend : public LayerFactoryBackend {
    friend class LayerFactory;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit GPULayerFactoryBackend(GfxContextLink context);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] std::string getName() const override;
    fyusenet::LayerBase * createLayer(LayerType type, LayerBuilder * builder, int layerNumber) override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] GPULayerBase * createAddSubLayer(GPULayerBuilder *builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createOESLayer(GPULayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createPaddingLayer(GPULayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createConvLayer(ConvLayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createTransConvLayer(ConvLayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createFracConvLayer(ConvLayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createScaleLayer(ScaleLayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createConcatLayer(ConcatLayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createS2DLayer(GPULayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createD2SLayer(GPULayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createMaxPoolLayer(PoolLayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createAvgPoolLayer(PoolLayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createArgMaxLayer(ArgMaxLayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createCustomLayer(CustomLayerBuilder *builder,int layerNumber);
    [[nodiscard]] GPULayerBase * createDownloadLayer(UpDownLayerBuilder * builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createUploadLayer(UpDownLayerBuilder * builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createSigmoidLayer(GPULayerBuilder * builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createTanhLayer(GPULayerBuilder * builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createImgExtractLayer(ImgExtractLayerBuilder * builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createNonMax2DLayer(GPULayerBuilder *builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createBlur2DLayer(BlurLayerBuilder *builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createRGB2BGRLayer(GPULayerBuilder *builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createSingletonArithLayer(SingletonArithLayerBuilder *builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createCastLayer(CastLayerBuilder *builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createTransposeLayer(TransposeLayerBuilder * builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createBatchNormLayer(GPULayerBuilder * builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createGEMMLayer(GPULayerBuilder * builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createSiLULayer(GPULayerBuilder *builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createGeLULayer(GPULayerBuilder *builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createRMSNormLayer(GPULayerBuilder * builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createEmbeddingLayer(EmbeddingLayerBuilder * builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createAttentionLayer(AttentionLayerBuilder * builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createLinearLayer(LinearLayerBuilder * builder, int layerNumber);
    [[nodiscard]] GPULayerBase * createTokenScoringLayer(TokenScoringLayerBuilder * builder, int layerNumber);
 private:
    static void checkRequirements();
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    GfxContextLink context_;
};


} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
