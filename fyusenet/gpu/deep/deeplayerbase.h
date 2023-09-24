//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Base Class for Deep-Channel Tensor Computations (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/vao.h"
#include "../../base/bufferspec.h"
#include "../gpulayerbase.h"
#include "../gpulayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {

/**
 * @brief Base class for (nearly) all GPU-specific deep-tensor data (high channel count) layers
 *
 * This class serves as base for all GPU-specific layers that deal with tensors that have a high
 * channel count. Using the same approach as for the shallow tensors would be inefficient as the
 * benefits of multi-render targets diminish. For this reason, a different format has been
 * chosen that is based on tiles within a single texture. The basic assumption behind that is that
 * for most networks observed in the wild, the spatial size correlates inversely with the channel
 * count.
 *
 * The tiling itself is done in a straightforward way. Each tile contains data for 4 channels
 * and has the spatial dimension of the tensor itself, plus additional padding. Contrary to the
 * shallow format tensors, when shaders access data outside the padding, the padding is not
 * repeated in most of the cases and should be considered as undefined. For example, when
 * performing 5x5 convolutions, a padding size of 2 should be chosen to avoid spilling into
 * neighboring tiles.
 *
 * The tile handling, like computations of texture coordinates and polygon coordinates us done in
 * the DeepTiler class.
 *
 * @todo Currently this base class is not used as a base class for convolutional deep layers. This
 *       is a design flaw that should be corrected.
 *
 * @see DeepTiler
 */
class DeepLayerBase : public GPULayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit DeepLayerBase(const GPULayerBuilder& builder);
    DeepLayerBase(const GPULayerBuilder& builder,int layerNumber);
    ~DeepLayerBase() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void writeResult(const char *fileName, bool includePadding) override;
    void copyResult(float *memory, bool includePadding) override;

    /**
     * @brief Obtain pointer to data tiler that is used for this object
     *
     * @return Pointer to DeepTiler object that is used to compute the tiling for this layer
     */
    [[nodiscard]] DeepTiler * getTiler() const {
        return tiler_;
    }

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    virtual void setupFBOs() override;
    virtual void updateFBOs() override;
    virtual size_t shaderPreprocessing(char *preproc, size_t maxChars);
    [[nodiscard]] BufferSpec::order getInputOrder(int port) const override;
    [[nodiscard]] BufferSpec::order getOutputOrder(int port) const override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    DeepTiler * tiler_ = nullptr;           //!< Pointer to tiler that executes tiling layout computations
    bool mali_ = false;                     //!< Indicator that code runs on a Mali GPU
    bool preG71_ = false;                   //!< Indicator that code runs on a Mali GPU prior to G-71 (i.e. T-series)
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
