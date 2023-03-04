//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Layer Flags (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>

//-------------------------------------- Project  Headers ------------------------------------------

namespace fyusion {
namespace fyusenet {
//------------------------------------- Public Declarations ----------------------------------------

using layerflags = uint32_t;

/**
 * @brief Namespace for covering layer flags (as their bitwise combinations)
 *
 * @todo The layer flags are mainly comprised of activations. Listing those as bitmask is
 *       not future-proof at all. The flags should be refactored into actual flags and
 *       an activation mode.
 */
// TODO (mw) refactor flags into flags and activation modes
namespace LayerFlags {
    constexpr static layerflags NO_LAYER_FLAGS = 0;        //!< This layer has no flags (thus it is 0)
    constexpr static layerflags RESIDUAL_INPUT = 1;        //!< This layer has residual input (another layer directly added to its results)
// TODO (mw) refactor to ACT_ON_RESIDUAL and add more activation modes to the residual
    constexpr static layerflags RELU_ON_RESIDUAL = 2;      //!< The residual to this layer should be subject to a ReLU operation (currently only simple ReLU is supported)
    constexpr static layerflags BATCHNORM_ON_RESIDUAL = 4; //!< Batchnorm (post) should also be applied on the residual
    constexpr static layerflags POST_BATCHNORM = 8;        //!< This layer is subject to a batchnorm-type rescale/bias operation on \e writing of its output data
    constexpr static layerflags DEEP = 16;                 //!< This layer is a deep layer for GPU exeuction (uses different memory layout on GPUs)
    constexpr static layerflags POST_RELU = 32;            //!< This layer is subject to perform a ReLU operation on \e writing of the output data (not supported by GPU layers)
    constexpr static layerflags PRE_RELU = 64;             //!< This layer is subject to perform a ReLU operation on \e reading of the input data
    constexpr static layerflags PRE_CLIP = 128;            //!< This layer is subject to a clipping operation on \e reading of the input data (data will be clipped to a given value range)
// TODO (mw) implement in shader activation
    constexpr static layerflags PRE_SIGMOID = 256;         //!< This layer is subject to a sigmoid activation on \e reading of the input data (not implemented yet)
    constexpr static layerflags PRE_TANH = 512;            //!< This layer is subject to a tanh activation on \e reading of the input data (not implemented yet)
    constexpr static layerflags ACT_MASK = (PRE_RELU | PRE_CLIP | PRE_SIGMOID | PRE_TANH | POST_RELU);
    constexpr static layerflags PRE_ACT_MASK = (PRE_RELU | PRE_CLIP | PRE_SIGMOID | PRE_TANH);
}


/**
 * @brief Identifiers for supported activation functions
 *
 * @note These are currently used inside the builder only. The initial version of FyuseNet had all the
 *       activations as part of their flags. The inwards facing part (the layer code) still has those
 *       in the flags. Those will be separated out in the near future, so do not use any of those
 *       flags externally.
 */
enum class ActType : uint8_t {
    NONE = 0,               //!< Empty/no activation function
    RELU = 1,               //!< Simple ReLU
    LEAKY_RELU,             //!< ReLU with leak
    CLIP,                   //!< Clipping
    SIGMOID,                //!< Sigmoid (not supported yet)
    TANH                    //!< TanH (not supported yet)
};


/**
 * @brief Identifiers for supported postfix normalizations
 *
 * @note These are currently used inside the builder only. The initial version of FyuseNet had all the
 *       normalizations (well, only one) as part of their flags. The inwards facing part (the layer
 *       code) still has those in the flags. Those will be separated out in the near future, so do
 *       not use any of those flags externally.
 */
enum class NormType : uint8_t {
    NONE = 0,               //!< Empty/no activation function
    BATCHNORM = 1,          //!< Batchnorm with fixed parameters from the training runs
};


/**
 * @brief Enumerator for different scaling types for scaling-type layers
 *
 * @see ScalingLayer
 */
enum class ScalingType : uint8_t {
    NEAREST = 0,            //!< Use nearest neighbor "interpolation" for scaling
    LINEAR                  //!< Use bilinear interpolation for scaling
};

/**
 * @brief Enumerator for blur kernel types on blur layers
 *
 * @see BlurLayer
 */
enum class BlurKernelType : uint8_t {
    AVERAGE = 0,            //!< Simple box-filter kernel
    GAUSSIAN = 1            //!< Gaussian filter kernel
};

/**
 * @brief Enumerator for singleton arithmetic layer types
 *
 * @see SingletonArithmeticLayer, DeepSingletonArithmeticLayer
 */
enum class ArithType : uint8_t {
    ADD = 0,                //!< Addition
    SUB,                    //!< Subtraction
    MUL,                    //!< Multiplication
    DIV                     //!< Division
};

/**
 * @brief Target data-types for type-cast layers
 *
 * @see CastLayer
 */
enum class CastTarget : uint8_t {
    CT_INT32 = 0,           //!< Cast to 32-bit signed integer
    CT_INT16,               //!< Cast to 16-bit signed integer
    CT_INT8,                //!< Cast to 8-bit signed integer
    CT_UINT32,              //!< Cast to 32-bit unsigned integer
    CT_UINT16,              //!< Cast to 16-bit unsigned integer
    CT_UINT8,               //!< Cast to 8-bit unsigned integer
    CT_FLOAT16,             //!< Cast to 16-bit half-float
    CT_FLOAT32              //!< Cast to 32-bit float
};

/**
 * @brief Enumerator for the various layer types implemented by this engine
 */
enum class LayerType : uint16_t {
    ADD = 1,                //!< Simple binary addition
    SUB,                    //!< Simple binary subtraction
    ARGMAX,                 //!< (Softish) ArgMax layer
    CAST,                   //!< Type-cast layer
    CONCAT,                 //!< Concatenation layer
    CONVOLUTION2D,          //!< 2D Convolution layer
    FRACCONVOLUTION2D,      //!< 2D Fractional-step convolution layer
    TRANSCONVOLUTION2D,     //!< 2D Transpose convolution layer
    AVGPOOL2D,              //!< 2D Average-Pooling layer
    MAXPOOL2D,              //!< 2D Max-Pooling layer
    PADDING2D,              //!< 2D padding layer, may be internally implemented by a different layer-type (e.g. scaling)
    SCALE2D,                //!< 2D Scale/Upsample layer
    SINGLETON_ARITH,        //!< Singleton-arithmetic layer
    RELU,                   //!< ReLU layer, may be internally implemented by a different layer-type (e.g. scaling)
    CLIP,                   //!< Clip layer, may be internally implemented by a different layer-type (e.g. scaling)
    TANH,                   //!< tanh function layer
    SIGMOID,                //!< Sigmoid layer
    REDUCE,                 //!< Reduction (dot-product) layer
    TRANSPOSE,              //!< Spatial transposition (image width/height) layer
    IMGEXTRACT,             //!< ImgExtract / Flatten
    BLUR2D,                 //!< 2D Blur layer
    NONMAX2D,               //!< 2D Non-Maximum Suppression
    RGB2BGR,                //!< Simple RGB -> BGR swapping on 2D images
    DEEP2SHALLOW,           //!< Deep -> Shallow conversion layer
    SHALLOW2DEEP,           //!< Shallow -> Deep conversion layer
    DOWNLOAD,               //!< GPU -> CPU download layer
    UPLOAD,                 //!< Upload layer
    RESIDUAL,               //!< Residual pseudo-layer (used internally, don't bother for now)
    OESCONV,                //!< Conversion layer that converts OES textures to "normal" textures (EGL / Android only)
    BATCHNORM,              //!< Explicit batchnorm layer
    GEMM,                   //!< Generalized matrix/matrix multiplication, implemented as MV -> 1x1 conv here since we cannot batch anyway
    CUSTOM,                 //!< Custom layer
    LAST_SUPPORTED,         //!< Last supported layer type (+1)
    ILLEGAL = 1000          //!< Placeholder for illegal layer types
};


/**
 * @brief Specifier list for compute devices
 */
enum class compute_device : uint8_t {
    DEV_GPU = 0,              //!< Executes on GPU (default operation mode)
    DEV_CPU,                  //!< Executes on CPU (only rudimentary support)
    DEV_NPU,                  //!< Executes on NPU device (existing support removed for public release)
    DEV_ILLEGAL               //!< Placeholder for illegal devices
};


namespace gpu {
    /**
     * @brief Specific to GPU devices, defines the channel packing factor for each pixel
     *
     * Defines the number of channels that can be stored in a single pixel for GPU-based
     * execution.
     */
    static constexpr int PIXEL_PACKING = 4;
}

} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
