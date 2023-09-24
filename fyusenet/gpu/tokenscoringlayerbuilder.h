﻿//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// De-Embedding Layer Builder (Header)                                         (c) Martin Wawro 2023
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
 * @brief Templatized anchor for TokenScoringLayer objects
 *
 * @see TokenScoringLayerBuilder
 */
template<typename D = GPULayerBuilderTempl<>>
struct TokenScoringLayerBuilderTempl : GPULayerBuilderTempl<D> {
    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     */
    explicit TokenScoringLayerBuilderTempl(const std::string& name) : GPULayerBuilderTempl<D>(name) {
        this->type(LayerType::TOKENSCORING);
        this->outputChannels_ = 1;
    }

    /**
     * @brief Copy-constructor
     *
     * @param src Source builder to copy data from
     */
    TokenScoringLayerBuilderTempl(const TokenScoringLayerBuilderTempl<D>& src) : GPULayerBuilderTempl<D>(src) {
    }

    /**
     * @brief Set temperature for non-deterministic token selection/sampling
     *
     * @param t Temperature to be used
     *
     * @return Reference to this builder
     *
     * The default temperature is 0. Non-deterministic selection/sampling is not supported yet.
     */
    D & temperature(float t) {
        temperature_ = t;
        return *(D *)this;
    }

    /**
     * @brief Set non-deterministic top-K selection/sampling as strategy for the token selection
     *
     * @param k Rank of the top-K selection/sampling
     *
     * @return Reference to this builder
     *
     * The default value for K is 1. Non-deterministic selection/sampling is not supported yet.
     */
    D & topK(int k) {
        topK_ = k;
        return *(D *)this;
    }

    /**
     * @brief Set non-deterministic top-P selection as strategy for the token selection/sampling
     *
     * @param p Probability threshold for the top-P selection/sampling (in [0,..,1])
     *
     * @return Reference to this builder
     *
     * The default value for P is 0.0. Non-deterministic selection/sampling is not supported yet.
     */
    D & topP(float p) {
        topP_ = p;
        return *(D *)this;
    }


    /**
     * @brief Set the number of rows in the embedding table
     *
     * @param numRows Number of rows in embedding table
     *
     * @return Reference to this builder
     *
     * In order to compute the score for an output token, the inner-product between the output token
     * and each row in the embedding is computed and ranked by the resulting similarity score.
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

    float temperature_ = 0.f;                          //!< Temperature for non-deterministic token selection/sampling
    int topK_ = 1;                                     //!< Rank of the top-K selection/sampling (unsupported)
    float topP_ = 0.0f;                                //!< Probability threshold for the top-P selection/sampling (unsupported)
    int tableRows_ = 0;                                //!< Number of rows in embedding table
    param_type srcDType_ = param_type::WGT_FLOAT;      //!< (CPU) datatype of data to expect in the parameters (currently fixed)
    param_type devDType_ = param_type::WGT_DEFAULT;    //!< On device data type for computation
    ScoringType scoringType_ = ScoringType::GREEDY;    //!< Method for token scoring to be used
};


/**
 * @brief Builder class for token scoring layers
 *
 * This class is to be used when building token scoring layers. Those layers are used to project
 * the embedding(s) generated by a sequence-learning network onto a known vocabulary and therefore
 * reduce each embedding to a single value per vocabulary entry. These values measure the "alignment"
 * of the embedding with the vocabulary entry, and have large positive values for a good alignment.
 *
 * As generative sequence-learning predicts and chooses the most likely token as the next in the
 * sequence, choosing the token that maximizes the alignment is the method of choice.
 * There are various ways to do this, which is reflected by the scoring type that can be set in the
 * builder.
 *
 * Currently FyuseNet only supports greedy (or top-1) selection/sampling.
 */
struct TokenScoringLayerBuilder : TokenScoringLayerBuilderTempl<TokenScoringLayerBuilder> {

    /**
     * @brief Constructor
     *
     * @param name Name to be assigned to the built layer
     *
     * This will create a builder for a token scoring layer, with the scoring mode set to
     * ScoringType::GREEDY by default.
     */
    explicit TokenScoringLayerBuilder(const std::string& name) : TokenScoringLayerBuilderTempl<TokenScoringLayerBuilder>(name) {}
    using TokenScoringLayerBuilderTempl<TokenScoringLayerBuilder>::TokenScoringLayerBuilderTempl;
};

} // fyusenet::gpu namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
