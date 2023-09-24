//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Attention Layer Builder (Header)                                            (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <vector>
#include <cmath>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gfxcontextlink.h"
#include "gpulayerbase.h"
#include "gpulayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
using namespace opengl;
namespace fyusenet::gpu {

/**
 * @brief Templatized anchor for the AttentionLayerBuilder
 *
 * @see AttentionLayerBuilder
 */
template<typename D = GPULayerBuilderTempl<>>
struct AttentionLayerBuilderTempl : GPULayerBuilderTempl<D> {
    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    explicit AttentionLayerBuilderTempl(const std::string& name) : GPULayerBuilderTempl<D>(name) {
        this->type(LayerType::ATTENTION);
    }

    /**
     * @brief Copy-constructor
     *
     * @param src Source builder to copy data from
     */
    AttentionLayerBuilderTempl(const AttentionLayerBuilderTempl<D>& src) : GPULayerBuilderTempl<D>(src) {
    }

    /**
     * @brief Configure the layer to be (implicitly) causally-masked
     *
     * @return Reference to builder object
     *
     * @note This is currently the only supported operating mode for attention layers in FyuseNet
     */
    D & causal() {
        causal_ = true;
        return *(D *)this;
    }

    /**
     * @brief Set number of attention heads
     *
     * @param num Number of attention heads
     *
     * @return Reference to builder object
     *
     * Sets the number of output attention heads for each Q,K,V component of the attention layer
     */
    D & heads(int num) {
        numHeads_ = num;
        return *(D *)this;
    }


    /**
     * @brief Set positional encoding type for query and key matrices
     *
     * @param enc Positional encoding to be used on the Q,K matrices
     *
     * @return Reference to builder object
     *
     * A positional encoding may be applied to the query and key matrices prior to computing the dot
     * product that computes the attention weights. This method sets the type of positional
     * encoding to be used.
     */
    D & positionalEncoding(PosEncType enc) {
        posEncoding_ = enc;
        return *(D *)this;
    }

    /**
     * @brief Set the base to compute \c theta for attention layers that use rotary encoding
     *
     * @param base Base value to compute \c theta
     *
     * @return Reference to builder object
     *
     * @see https://arxiv.org/pdf/2104.09864.pdf
     */
    D & rotaryThetaBase(float base) {
        thetaBase_ = base;
        return *(D *)this;
    }

    /**
     * @brief Sets the output dimension of each attention head
     *
     * @param dim Dimensionality of each attention head (in atoms, not pixels)
     *
     * @return Reference to builder object
     */
    D & headDim(int dim) {
        headDim_ = dim;
        return *(D *)this;
    }

    /**
     * @brief Adjust quantization type and weight-datatype in the attention layer
     *
     * @param qType Quantization type to set
     * @param wtype Data-type of the weights
     *
     * @return Reference to builder object
     *
     * @note We only support mixed-precision floating-point quantization right now
     */
     // TODO (mw) expand docs by small explanation of quantization types
    D & quantize(qt_type qType, param_type wtype) {
        if (qType != qt_type::QT_MIXED_FLOAT) {
            THROW_EXCEPTION_ARGS(FynException,"Attention layers only support mixed float quantization");
        }
        quantType_ = qType;
        wgtType_ = wtype;
        return *(D *)this;
    }


    /**
     * @brief Set quantization group size for quantized layers
     *
     * @param groupSize Quantization group size
     *
     * The quantization group size controls a row-wise blocking of the input matrices in terms of
     * quantization. By default, mixed-precision quantization assigns one quantization scale to
     * each column, effectively using a quantization group size equivalent to the height of the
     * matrix. If more than one scaling coefficient shall be used for a column, the quantization
     * group size is set to a value \c q smaller than the height and for each \c q rows, a different
     * scalar (and zero-point) is used.
     *
     * @return Reference to builder object
     */
    D & quantGroupSize(int groupSize) {
        quantGroupSize_ = groupSize;
        return *(D *)this;
    }

    /**
     * @brief Enable auto-residual mode
     *
     * @return Reference to builder object
     *
     * Auto-residual mode instructs the attention layer to add its output to its input.
     */
    D & autoResidual() {
        autoResidual_ = true;
        return *(D *)this;
    }

    /**
     * @brief Enable caching (incremental inference) mode
     *
     * @return Reference to builder object
     */
    D & incremental() {
        incremental_ = true;
        return *(D *)this;
    }

    int numHeads_ = 0;            //!< Number of attention heads
    int headDim_ = 0;             //!< Output dimension of each attention head
    int quantGroupSize_ = 0;      //!< For quantized data with quantization grouping, provides the group size for row-wise blocking
    float thetaBase_ = 1.0f;      //!< Base value to compute theta for rotary encoding
    bool autoResidual_ = false;   //!< If set to \c true, the result of the attention layer will be added to its input automatically
    bool incremental_ = false;    //!< If set to \c true, generates an incremental attention layer which caches previous results and appends new queries
    bool causal_ = false;         //!< If set to \c true, generates a causally-masked attention layer (currently the only type that FyuseNet supports)

    /**
     * Positional encoding to use for the query and key matrices
     */
    PosEncType posEncoding_ = PosEncType::NONE;

    /**
     * Quantization type
     */
    qt_type quantType_ = qt_type::QT_NONE;


    /**
     * Data type for the weight data that is expected for the layer
     */
    param_type wgtType_ = param_type::WGT_FLOAT;
};


/**
 * @brief Builder class for attention layers
 *
 * Builder to be used for setting up (multi-head) self-attention layers. (Self) attention layers
 * are used in sequence learning tasks in order to focus "attention" on different parts of
 * sequences. This is done by deriving query, key and value matrices from the input sequence and
 * then computing a dot-product between the query and key matrices, usually followed by a non-linear
 * operation like a softmax. The dot product represents the "attention" that different parts of the
 * query sequence pay to different parts of the key sequence. The attention weights are then used
 * to compute a weighted sum of the value matrix.
 *
 * As of now, FyuseNet only supports causally masked self-attention. This means that the attention
 * weights are computed only for the current and previous elements of the sequence implicitly.
 */
struct AttentionLayerBuilder : AttentionLayerBuilderTempl<AttentionLayerBuilder> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    explicit AttentionLayerBuilder(const std::string& name) : AttentionLayerBuilderTempl<AttentionLayerBuilder>(name) {}

    using AttentionLayerBuilderTempl<AttentionLayerBuilder>::AttentionLayerBuilderTempl;
};

} // fyusenet::gpu namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
