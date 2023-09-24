//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Linear Layer for Sequences (Header)                                         (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gpulayerbase.h"
#include "../linearlayerbuilder.h"
#include "rudiments/matmul_const.h"

class SequenceLayerTest;

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet::gpu::sequence {

/**
 * @brief Perform matrix / matrix or matrix / vector multiplication with a constant matrix
 *
 * This class performs a multiplication of two matrices, where the right matrix is a constant
 * matrix that has been uploaded to the GPU before. The left matrix arises from chained
 * computations by the layers in the network.
 * This particular implementation runs on texture layouts used for processing of \e sequences.
 *
 * The operation that is carried out is given by:
 *
 * \f[ \mathbf{Y} = \mathbf{X}\mathbf{W} \f]
 *
 * where \f$ \mathbf{X} \in \mathbb{R}^{n \times m}\f$ is allowed to degenerate into a vector
 * \f$ \mathbf{x} \in \mathbb{R}^{1 \times m}\f$. In the latter case, this class also supports
 * to add a \e bias to the result of the multiplication to yield the affine transform:
 *
 * \f[ \mathbf{y} = \mathbf{x}\mathbf{W} + \mathbf{b} \f]
 *
 * where \f$ \mathbf{b} \in \mathbb{R}^{1 \times m} \f$.
 *
 * More detailed documentation on how the multiplication is carried out can be found in the
 * MatMulConst class that is used to perform the actual multiplication.
 *
 * @see MatMulConst
 */
class LinearLayer : public gpu::GPULayerBase {
    friend class ::SequenceLayerTest;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit LinearLayer(const LinearLayerBuilder & builder);
    LinearLayer(const LinearLayerBuilder & builder, int layerNumber);
    ~LinearLayer() override = default;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void forward(uint64_t sequenceNo, StateToken * state) override;
    void setup() override;
    void cleanup() override;
    void setupFBOs() override;
    void updateFBOs() override;
    void loadParameters(const ParameterProvider * source) override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    [[nodiscard]] std::vector<BufferSpec> getRequiredInputBuffers() const override;
    [[nodiscard]] GPUBuffer *getGPUOutputBuffer(int port) const override;
    [[nodiscard]] GPUBuffer *getGPUInputBuffer(int port) const override;

    void writeResult(const char *fileName, bool includePadding) override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] BufferSpec::order getInputOrder(int port) const override;
    [[nodiscard]] BufferSpec::order getOutputOrder(int port) const override;

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int sequenceLength_ = 0;             //!< Number of rows of the input tensor (not necessarily the texture height)b
    int quantGroupSize_ = 0;             //!< For quantized layers, the quantization group size
    bool hasBias_ = false;               //!< Indicator if layer computes an affine mapping

    /**
     * Instance of the matrix-multiplication operator that performs the heavy lifting
     */
    rudiments::MatMulConst * matMul_ = nullptr;

    /**
    * Type of quantization to be used in computation
    */
    qt_type quantType_ = qt_type::QT_NONE;

    /**
     * Data type for the weights supplied to this layer
     */
    param_type dataType_ = param_type::WGT_FLOAT;
};

} // fyusion::fyusenet::gpu::sequence namespace


// vim: set expandtab ts=4 sw=4:
