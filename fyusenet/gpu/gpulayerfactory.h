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

namespace fyusion {
namespace fyusenet {
namespace gpu {
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
    GPULayerFactoryBackend(GfxContextLink context);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual std::string getName() const override;
    virtual fyusenet::LayerBase * createLayer(LayerType type, LayerBuilder * builder, int layerNumber) override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    GPULayerBase * createAddSubLayer(GPULayerBuilder *builder, int layerNumber);
    GPULayerBase * createOESLayer(GPULayerBuilder *builder,int layerNumber);
    GPULayerBase * createPaddingLayer(GPULayerBuilder *builder,int layerNumber);
    GPULayerBase * createConvLayer(ConvLayerBuilder *builder,int layerNumber);
    GPULayerBase * createTransConvLayer(ConvLayerBuilder *builder,int layerNumber);
    GPULayerBase * createFracConvLayer(ConvLayerBuilder *builder,int layerNumber);
    GPULayerBase * createScaleLayer(ScaleLayerBuilder *builder,int layerNumber);
    GPULayerBase * createConcatLayer(ConcatLayerBuilder *builder,int layerNumber);
    GPULayerBase * createS2DLayer(GPULayerBuilder *builder,int layerNumber);
    GPULayerBase * createD2SLayer(GPULayerBuilder *builder,int layerNumber);
    GPULayerBase * createMaxPoolLayer(PoolLayerBuilder *builder,int layerNumber);
    GPULayerBase * createAvgPoolLayer(PoolLayerBuilder *builder,int layerNumber);
    GPULayerBase * createArgMaxLayer(ArgMaxLayerBuilder *builder,int layerNumber);
    GPULayerBase * createCustomLayer(CustomLayerBuilder *builder,int layerNumber);
    GPULayerBase * createDownloadLayer(UpDownLayerBuilder * builder, int layerNumber);
    GPULayerBase * createUploadLayer(UpDownLayerBuilder * builder, int layerNumber);
    GPULayerBase * createSigmoidLayer(GPULayerBuilder * builder, int layerNumber);
    GPULayerBase * createTanhLayer(GPULayerBuilder * builder, int layerNumber);
    GPULayerBase * createImgExtractLayer(ImgExtractLayerBuilder * builder, int layerNumber);
    GPULayerBase * createNonMax2DLayer(GPULayerBuilder *builder, int layerNumber);
    GPULayerBase * createBlur2DLayer(BlurLayerBuilder *builder, int layerNumber);
    GPULayerBase * createRGB2BGRLayer(GPULayerBuilder *builder, int layerNumber);
    GPULayerBase * createSingletonArithLayer(SingletonArithLayerBuilder *builder, int layerNumber);
    GPULayerBase * createCastLayer(CastLayerBuilder *builder, int layerNumber);
    GPULayerBase * createTransposeLayer(TransposeLayerBuilder * builder, int layerNumber);
 private:
    static void checkRequirements();
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    GfxContextLink context_;
};


} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
