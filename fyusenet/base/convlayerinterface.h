//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Convolution Layer Interface (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------


//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {

/**
 * @brief Basic interface for convolution layers
 *
 * This interface defines the convolution filter data interface for all types of convolution layers.
 */
class ConvLayerInterface {
 public:
    /**
     * @brief Read weights and biases from raw data
     *
     * @param biasAndWeights Pointer to array with bias and weight values (see long description)
     *
     * @param offset Optional offset (in floating-point elements) into \p biasAndWeights where to
     *               start reading from
     *
     * This function parses the weights and biases stored in the \p biasAndWeights parameter for
     * usage with the GPU. It is assumed that the biases and weights are stored biases first,
     * followed by the convolution weights. In case a batchnorm operation is used, the batchnorm
     * parameters are also read from the \p biasAndWeights array and are assumed following the weight
     * data in the form of all scales first and then all offsets.
     * For example, for \e n output channels, the first \e n entries in \p biasAndWeights are the
     * biases. For \e m input channels and a kernel of size \e k (i.e. a \f$ k \times k \f$ kernel),
     * we expect a 4D array of size \f$ n \times k \times k \times m \f$ with the following index
     * order:
     *
     * @code
     * [outchannel][kernely][kernelx][inchannel]
     * @endcode
     *
     * @see ConvWeightArrayKxKxNxM
     *
     * @note It is safe to call this function from a context that is shared with the initial GL
     *       context that was used to create the layer.
     */
    virtual void loadWeightsAndBiases(const float *biasAndWeights, size_t offset=0) = 0;
};

}  // fyusenet namespace
}  // fyusion namespace

// vim: set expandtab ts=4 sw=4:

