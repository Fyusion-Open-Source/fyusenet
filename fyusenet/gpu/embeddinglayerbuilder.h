//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Embedding Layer Builder (Header)                                            (c) Martin Wawro 2023
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
 * @brief Templatized anchor for embedding layer builders
 *
 * @see EmbeddingLayerBuilder
 */
template<typename D = GPULayerBuilderTempl<>>
struct EmbeddingLayerBuilderTempl : GPULayerBuilderTempl<D> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    explicit EmbeddingLayerBuilderTempl(const std::string& name) : GPULayerBuilderTempl<D>(name) {
        this->type(LayerType::EMBEDDING);
        this->inputChannels_ = 1;
    }

    /**
     * @brief Copy-constructor
     *
     * @param src Source builder to copy data from
     */
    EmbeddingLayerBuilderTempl(const EmbeddingLayerBuilderTempl<D>& src) : GPULayerBuilderTempl<D>(src) {
    }

    /**
     * @brief Set the number of rows in the embedding table
     *
     * @param numRows Number of rows in embedding table
     *
     * @return Reference to this builder
     */
    D & tableRows(int numRows) {
        tableRows_ = numRows;
        return *(D *)this;
    }

    /**
     * @brief Set precision of the data on compute device (e.g. GPU)
     *
     * @param type Data type / precision to be used for storing the embedding data on the compute device
     *
     * @return Reference to this builder
     */
    D & computePrecision(param_type type) {
        devDType_ = type;
        return *(D *)this;
    }

    param_type srcDType_ = param_type::WGT_FLOAT;      //!< (CPU) datatype of data to expect in the parameters (currently fixed)
    param_type devDType_ = param_type::WGT_DEFAULT;    //!< On device data type
    int tableRows_ = 0;                                //!< Number of rows in embedding table
};


/**
 * @brief Builder class for embedding layers
 *
 * Embeddings are simple lookup tables which take integer indices and replace each index by a
 * vector (row) that is picked from an internal embedding table based on the supplied index.
 */
struct EmbeddingLayerBuilder : EmbeddingLayerBuilderTempl<EmbeddingLayerBuilder> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    explicit EmbeddingLayerBuilder(const std::string& name) : EmbeddingLayerBuilderTempl<EmbeddingLayerBuilder>(name) {}
};

} // gpu::fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
