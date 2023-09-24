//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Combination of a Linear Layer on top of Hadamard Product (Header)           (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <mutex>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gpulayerbase.h"
#include "../../customlayerbuilder.h"
#include "../../sequence/rudiments/matmul_const.h"

class SequenceLayerTest;

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::gpu::custom::sequence {

/**
 * @brief Custom layer that performs a linear operation on top of a Hadamard product
 *
 * This layer is a custom combination of several operations performed on two input tensors, namely
 * a linear layer (matrix product) on top of a Hadamard product in the form:
 *
 * \f[ \mathbf{Y} = (\mathbf{X}_1 \circledot \mathbf{X}_2) \mathbf{W} \f]
 *
 * Where \f$ \mathbf{X}_1 \f$ and \f$ \mathbf{X}_2 \f$ are the two input tensors and \f$ \mathbf{W} \f$
 * is the weight matrix. In case \f$ \mathbf{X}_1 \f$ and \f$ \mathbf{X}_2 \f$ are vectors, the
 * linear transform can be upgraded to an affine transform:
 *
 * \f[ \mathbf{y} = (\mathbf{x}_1 \circledot \mathbf{x}_2) \mathbf{W} + \mathbf{b} \f]
 *
 * @warning The current implementation only supports 4-bit quantized weight matrices as of now
 *
 * @see MatMulConst
 */
class LinearHadamardLayer : public gpu::GPULayerBase {
    friend class ::SequenceLayerTest;
 public:

    struct BuilderData {
        qt_type quantType_ = qt_type::QT_NONE;
        param_type dataType_ = param_type::WGT_FLOAT;
        int quantGroupSize_ = 0;
        bool hasBias_ = false;
    };

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit LinearHadamardLayer(const CustomLayerBuilder & builder);

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
    static CustomLayerBuilder * createBuilder(const std::string& name, bool bias=false);
    static CustomLayerBuilder * createBuilder(const std::string& name, qt_type quant=qt_type::QT_NONE, param_type dataType=param_type::WGT_FLOAT, int quantGroupSize=0 , bool bias=false);
    void writeResult(const char *fileName, bool includePadding) override;

    /**
     * @copydoc LayerBase::getPortChannelIndex
     */
    [[nodiscard]] int getPortChannelIndex(int port) const override {
        if (port >= numInputPorts()) THROW_EXCEPTION_ARGS(FynException,"Illegal input port %d specified",port);
        return port;
    }

    /**
     * @copydoc LayerBase::numInputPorts
     */
    [[nodiscard]] int numInputPorts() const override {
        return 2 + ((flags_ & LayerFlags::RESIDUAL_INPUT) ? 1 : 0);
    }

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] BufferSpec::order getInputOrder(int port) const override;
    [[nodiscard]] BufferSpec::order getOutputOrder(int port) const override;
    void postprocShader(opengl::ShaderProgram * shader, gpu::sequence::rudiments::MatMulConst::shtype type);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    GLuint weightData_ = 0;        //!< GL texture with (quantized) weight data
    GLuint scaleData_ = 0;         //!< GL texture with scale data for quantization
    GLuint zeroData_ = 0;          //!< GL texture with zero-biases for quantization
    GLuint biasData_ = 0;          //!< GL texture that contains bias data (optional)
    int quantGroupSize_ = 0;       //!< Quantization group size for quantized data
    int sequenceLength_ = 0;       //!< # of token in last sequence (not necessarily texture height)
    bool hasBias_ = false;         //!< Indicator whether bias data is present

    /**
     *
     */
    gpu::sequence::rudiments::MatMulConst * matMul_ = nullptr;

    /**
    * Type of quantization to be used in computation
    */
    qt_type quantType_ = qt_type::QT_NONE;

    /**
     * Data type for the weights supplied to this layer
     */
    param_type dataType_ = param_type::WGT_FLOAT;
};

} // fyusion::fyusenet::gpu::custom::sequence namespace

// vim: set expandtab ts=4 sw=4:
