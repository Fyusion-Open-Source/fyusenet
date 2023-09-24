//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Parameter Provider Interface                                                (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "parameterprovider.h"

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------

namespace fyusion::fyusenet {

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Map parameter for a given layer / parameter-name into a mapper instance
 *
 * @param name Name to identify the parameter by, could be the layer name or some construction
 *             based on the layer name (see long description)
 * @param layerNo Number of layer to map weights for
 * @param subIndex Sub-index for layers that are aggregates of multiple sub-layers or split the
 *                 parameters internally, set to 0 if not needed
 *
 * @return Instance of DataBlobMapper which can be used to access the weights within a mapping
 *         function
 *
 * This function returns a DataBlobMapper instance which can be used to access parameters
 * from a supplied function pointer to avoid taking care of the object life-cycle. The way that
 * a provider distinguishes between different parameter types for the same layer is layer-specific
 * and should be looked up in the layer documentation for \c loadParameters() itself.
 *
 * @see get()
 *
 * @see ConvLayerBase::loadParameters(), DeepConvLayerBase::loadParameters(),
 *      BatchNormLayer::loadParameters(), DeepBatchNormLayer::loadParameters(),
 *      DeepDepthwiseConvLayer::loadParameters(), TransConvLayer3x3::loadParameters(),
 *      DeepTransConvLayerBase::loadParameters(), RMSNormLayer::loadParameters(),
 *      EmbeddingLayer::loadParameters(), LinearLayer::loadParameters(),
 *      CausalMultiHeadAttentionLayer::loadParameters()
 */
const DataBlobMapper ParameterProvider::map(const std::string& name, int layerNo, int subIndex) const {
    return DataBlobMapper(get(name, layerNo, subIndex));
}


/**
 * @brief Get parameters for a given layer
 *
 * @param name Name to identify the parameter by, could be the layer name or some construction
 *             based on the layer name (see long description)
 * @param layerNo Number of layer to get weights for
 * @param subIndex Sub-index for layers that are aggregates of multiple sub-layers or split the
 *                 parameters internally, set to 0 if not needed
 *
 * @return Instance of DataBlob which can be used to access the parameters, see notes about
 *         object life-cycle
 *
 * This function returns a DataBlob instance which can be used to retrieve a raw pointer to
 * the underlying data. The two parameters are used to identify which weights are to be
 * supplied in the blob, making use of either \p name, \p layerNo or \c subIndex as identifier (or
 * a combination of those). It is up to the actual implementation to determine a pattern that
 * provides unique access to the underlying data and the documentation of the respective layers
 * should be consulted on that.
 *
 * @note The life-cycle of the returned DataBlob instance determines the validity of all
 *       pointers retrieved from it. Once the returned DataBlob object is destroyed, do not
 *       use any pointers previously retrieved from it.
 *
 * @see ConvLayerBase::loadParameters(), DeepConvLayerBase::loadParameters(),
 *      BatchNormLayer::loadParameters(), DeepBatchNormLayer::loadParameters(),
 *      DeepDepthwiseConvLayer::loadParameters(), TransConvLayer3x3::loadParameters(),
 *      DeepTransConvLayerBase::loadParameters(), RMSNormLayer::loadParameters(),
 *      EmbeddingLayer::loadParameters(), LinearLayer::loadParameters()
 *      CausalMultiHeadAttentionLayer::loadParameters()
 */
DataBlob ParameterProvider::get(const std::string& name, int layerNo, int subIndex) const {
    return DataBlob();
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:
