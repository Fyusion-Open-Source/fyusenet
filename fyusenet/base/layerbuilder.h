//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Generic Layer Builder (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <vector>
#include <cassert>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "layerflags.h"
#include "../common/fynexception.h"
#include "layerfactoryinterface.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
namespace fyusenet {

class LayerFactory;

/**
 * @brief Placeholder template type
 */
struct BuilderLeaf {
};


/**
 * @brief Templatized anchor for layer builders
 *
 * In order to facilitate the creation of network layers and to add an abstraction layer from the
 * underlying device-specific implementation, FyuseNet uses a builder pattern which aggregates
 * all parameters in a convenient and flexible way. Once a builder has been fully parameterized,
 * it can be pushed to a LayerFactory instance, which then is able to compile the supplied layers
 * into a network.
 *
 * This particular class serves as generic template anchor for the LayerBuilder class.
 *
 * @see LayerBuilder
 */
template<typename D = BuilderLeaf>
struct LayerBuilderTempl {
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------

    /**
     * @brief Constructor
     *
     * @param name Name for the layer
     */
    LayerBuilderTempl(const std::string& name): name_(name) {
    }

    /**
     * @brief Destructor
     */
    virtual ~LayerBuilderTempl() {
    }

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------

    /**
     * @brief Push this builder to a LayerFactory for later compilation
     *
     * @param factory Shared pointer where this builder should be pushed/appended to
     *
     * Use this function to register this builder instance with a LayerFactory in preparation for
     * the actual layer compilation
     *
     * @see LayerFactory::pushBuilder()
     */
    void push(std::shared_ptr<LayerFactory> & factory) {
        // NOTE (mw) this is ugly, a dynamic_pointer_cast with a check would be safer
        auto * ifc = reinterpret_cast<LayerFactoryInterface *>(factory.get());
        ifc->pushBuilder(reinterpret_cast<LayerBuilder *>(this));
    }

    /**
     * @brief Set layer type in builder object
     *
     * @param type The LayerType to set
     *
     * @return Reference to builder object
     */
    D & type(LayerType type) {
        type_ = type;
        return *(D *)this;
    }

    /**
     * @brief Set layer number in builder object
     *
     * @param no Layer number
     *
     * @return Reference to builder object
     */
    D & number(int no) {
        assert(no >= 0);
        number_ = no;
        return *(D *)this;
    }


    /**
     * @brief Set spatial size in builder object
     *
     * @param width Width of the tensor that the layer should process
     * @param height Height of the tensor that the layer should process
     *
     * @return Reference to builder object
     */
    D & size(short width, short height) {
        width_=width;
        height_=height;
        return *(D *)this;    
    }

    /**
     * @brief Set (isotropic) downsampling in builder object
     *
     * @param ds Downsampling factor that the layer should apply
     *
     * @return Reference to builder object
     */
    D & downsample(int ds) {
        downsample_[0] = ds;
        downsample_[1] = ds;
        return *(D *)this;
    }

    /**
     * @brief Set (anisotropic) downsampling in builder object
     *
     * @param horizontal Horizontal downsampling factor that the layer should apply
     * @param vertical Vertical downsampling factor that the layer should apply
     *
     * @return Reference to builder object
     */
    D & downsample(int horizontal,int vertical) {
        downsample_[0] = horizontal;
        downsample_[1] = vertical;
        return *(D *)this;
    }


    /**
     * @brief Set (isotropic) upsampling in builder object
     *
     * @param upsample Upsampling factor that the layer should apply
     *
     * @return Reference to builder object
     */
    D & upsample(short upsample) {
        upsample_[0] = upsample;
        upsample_[1] = upsample;
        return *(D *)this;
    }


    /**
     * @brief Set (anisotropic) upsampling in builder object
     *
     * @param horizontal Upsampling factor that the layer should apply
     * @param vertical Upsampling factor that the layer should apply
     *
     * @return Reference to builder object
     */
    D & upsample(short horizontal,short vertical) {
        upsample_[0] = horizontal;
        upsample_[1] = vertical;
        return *(D *)this;
    }


    /**
     * @brief Set (isotropic) input padding in builder object
     *
     * @param padding Input data padding (on all sides) that the layer should expect
     *
     * @return Reference to builder object
     */
    D & inputPadding(short padding) {
        inputPadding_ = padding;
        return *(D *)this;
    }


    /**
     * @brief Set (isotropic) input padding for the residual part in the builder object
     *
     * @param padding Residual-in data padding (on all sides) that the layer should expect
     *
     * @return Reference to builder object
     */
    D & residualPadding(short padding) {
        residualPadding_ = padding;
        return *(D *)this;
    }

    /**
     * @brief Set (isotropic) output padding in the builder object
     *
     * @param padding Output padding (on all sides) that the layer should apply
     *
     * @return Reference to builder object
     */
    D & outputPadding(short padding) {
        outputPadding_ = padding;
        return *(D *)this;
    }


    /**
     * @brief Set prefix activation function for this layer
     *
     * @param act Prefix activation to use
     *
     * @return Reference to builder object after update
     */
    D & prefixAct(ActType act) {
        preAct_ = act;
        return *(D *)this;
    }


    /**
     * @brief Set postfix activation function for this layer
     *
     * @param act Postfix activation to use
     *
     * @return Reference to builder object after update
     *
     * @warning Postfix activation is not supported by GPU layers
     */
    D & postfixAct(ActType act) {
        postAct_ = act;
        return *(D *)this;
    }


    /**
     * @brief Set postfix normalization for this layer
     *
     * @param nrm Type of postfix norm to use on this layer
     *
     * @return Reference to builder after update
     */
    D & postfixNorm(NormType nrm) {
        postNorm_ = nrm;
        return *( D *)this;
    }

    /**
     * @brief Mark the layer to be using deep tensor format
     *
     * @return Reference to builder after update
     */
    D & deep() {
        flags_ |= LayerFlags::DEEP;
        return *( D *)this;
    }


    /**
     * @brief Mark the layer to be using an additional input as (additive) residual
     *
     * @param act Optional activation to apply to the residual data prior to adding it to the results,
     *            defaults to no activation
     *
     * @param In case the underlying target layer has a postfix normalization, setting this flag
     *        will apply the same normalization to the residual data, defaults to \c false
     *
     * @return Reference to builder after update
     */
    D & residual(ActType act = ActType::NONE, bool postfixNorm=false) {
        if ((act != ActType::RELU) && (act != ActType::NONE)) THROW_EXCEPTION_ARGS(FynException, "Activation type %d not supported on residual", (int)act);
        flags_ |= LayerFlags::RESIDUAL_INPUT;
        if (act == ActType::RELU) flags_ |= LayerFlags::RELU_ON_RESIDUAL;
        else flags_ &= ~LayerFlags::RELU_ON_RESIDUAL;
        residualNorm_ = postfixNorm;
        return *( D *)this;
    }


    /**
     * @brief Set layer shape in builder object for convolution operations
     *
     * @param outChannels Number of output channels for the layer
     * @param height Height of the \b input tensor
     * @param width Width of the \b input tensor
     * @param inChannels Number of input channels that the layer should expect
     *
     * @return Reference to builder object
     */
    D & shape(int outChannels, int height, int width, int inChannels) {
        width_ = width;
        height_ = height;
        inputChannels_ = inChannels;
        outputChannels_ = outChannels;
        return *(D *)this;
    }


    /**
     * @brief Set layer shape in builder object
     *
     * @param height Height of the \b input tensor
     * @param width Width of the \b input tensor
     * @param channels Number of channels that the layer should expect
     *
     * @return Reference to builder object
     */
    D & shape(int height, int width, int channels) {
        width_ = width;
        height_ = height;
        inputChannels_ = channels;
        outputChannels_ = channels;
        return *(D *)this;
    }


    /**
     * @brief Set input/output channels in builder object
     *
     * @param channels Channels for input \b and output to be set to the layer
     *
     * @return Reference to builder object
     *
     * @warning We do not support more than 32767 channels
     */
    D & channels(short channels) {
        inputChannels_ = channels;
        outputChannels_ = channels;
        return *(D *)this;
    }

    /**
     * @brief Set input channels in builder object
     *
     * @param channels Channels for input tensor to be set to the layer
     *
     * @return Reference to builder object
     *
     * @warning We do not support more than 32767 channels
     */
    D & inChannels(short channels) {
        inputChannels_ = channels;
        return *(D *)this;
    }

    /**
     * @brief Set output channels in builder object
     *
     * @param channels Channels for output tensor to be set to the layer
     *
     * @return Reference to builder object
     *
     * @warning We do not support more than 32767 channels
     */
    D & outChannels(short channels) {
        outputChannels_ = channels;
        return *(D *)this;
    }

    /**
     * @brief Set leak-value for LeakyReLU in builder object
     *
     * @param leak Leak-value to be used in the layer
     *
     * @return Reference to builder object
     */
    D & leakyReLU(float leak) {
        leakyReLU_ = leak;
        return *(D *)this;
    }

    /**
     * @brief Set clipping values for clip activation in builder bject
     *
     * @param low Minimum value to clip at in the layer
     * @param high Maximum value to clip at in the layer
     *
     * @return Reference to builder object
     */
    D & clip(float low,float high) {
        clipLow_ = low;
        clipHigh_ = high;
        return *(D *)this;
    }

    /**
     * @brief Get width of (input) tensor data
     *
     * @return Width (spatial x-dimension)
     */
    virtual short width() const {
        return width_;
    }

    /**
     * @brief Get height of (input) tensor data
     *
     * @return Height (spatial y-dimension)
     */
    virtual short height() const {
        return height_;
    }


    /**
     * @brief Get number of input channels
     *
     * @return # of input channels
     */
    virtual short in() const {
        return inputChannels_;
    }


    /**
     * @brief Get number of output channels
     *
     * @return # of output channels
     */
    virtual short out() const {
        return outputChannels_;
    }


    layerflags getFlags() const {
        layerflags full = flags_;
        // TODO (mw) remove once we separated the activation from the flags
        switch (preAct_) {
            case ActType::NONE:
                break;
            case ActType::RELU:
            case ActType::LEAKY_RELU:
                full |= LayerFlags::PRE_RELU;
                break;
            case ActType::CLIP:
                full |= LayerFlags::PRE_CLIP;
                break;
            case ActType::SIGMOID:
            case ActType::TANH:
                THROW_EXCEPTION_ARGS(FynException, "Activation type not supported yet");
        }
        // TODO (mw) remove once we separated the activation from the flags
        switch (postAct_) {
            case ActType::NONE:
                break;
            case ActType::RELU:
            case ActType::LEAKY_RELU:
                full |= LayerFlags::POST_RELU;
                break;
            case ActType::CLIP:
            case ActType::SIGMOID:
            case ActType::TANH:
                THROW_EXCEPTION_ARGS(FynException, "Activation type not supported yet");
        }
        switch (postNorm_) {
            case NormType::NONE:
                break;
            case NormType::BATCHNORM:
                full |= LayerFlags::POST_BATCHNORM;
                break;
        }
        switch (resAct_) {
            case ActType::NONE:
                break;
            case ActType::RELU:
                full |= LayerFlags::RELU_ON_RESIDUAL;
                break;
            default:
                break;
        }        
        if (residualNorm_ && LayerFlags::POST_BATCHNORM) full |= LayerFlags::BATCHNORM_ON_RESIDUAL;
        return full;
    }

    /**
     * @brief Check if builder is for a deep-tensor format layer
     *
     * @retval true if builder is for deep-tensor format layer
     * @retval false otherwise
     */
    inline bool isDeep() const {
        return ((flags_ & LayerFlags::DEEP) == LayerFlags::DEEP);
    }

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------

    std::string name_;                      //!< Layer name
    short inputPadding_ = 0;                //!< Padding for the input tensor (all spatial sides)
    short outputPadding_ = 0;               //!< Padding for the output tensor (all spatial sides)
    short residualPadding_ = 0;             //!< Padding for the residual (input) tensor (all spatial sides)
    short downsample_[2] = {1,1};           //!< Downsampling values (x/y dimension)
    short upsample_[2] = {1,1};             //!< Upsampling values (x/y dimension)
    ActType preAct_ = ActType::NONE;        //!< Prefix activation function to use
    ActType postAct_ = ActType::NONE;       //!< Postfix activation function to use
    ActType resAct_ = ActType::NONE;        //!< Activation function to use on residual
    NormType postNorm_ = NormType::NONE;    //!< Postfix normalization to use
    float leakyReLU_ = 0.0f;                //!< Leak value for LeakyReLU activation function
    float clipLow_ = 0.0f;                  //!< Min clip value for clipping activation function
    float clipHigh_ = 0.0f;                 //!< Max clip value for clipping activation function
    int number_ = -1;                       //!< Layer number
    LayerType type_ = LayerType::ILLEGAL;   //!< Layer type
    bool residualNorm_ = false;             //!< Apply postfix norm to residual data

    /**
     *  Device on which to construct / execute the layer on
     */
    compute_device device_ = compute_device::DEV_CPU;

 protected:
    uint16_t width_ = 0;                   //!< Width of the input tensor
    uint16_t height_ = 0;                  //!< Height of the input tensor
    uint16_t inputChannels_ = 0;           //!< Number of channels on the input tensor
    uint16_t outputChannels_ = 0;          //!< Number of channels on the output tensor

    /**
     * @brief Flags to be assigned to the layer
     *
     * @todo Refactor the flags to exclude the activations and take them directly from the pre/post acts
     */
    layerflags flags_ = LayerFlags::NO_LAYER_FLAGS;

};


/**
 * @brief Base class for layer builders
 *
 * In order to facilitate the creation of network layers and to add an abstraction layer from the
 * underlying device-specific implementation, FyuseNet uses a builder pattern which aggregates
 * all parameters in a convenient and flexible way. Once a builder has been fully parameterized,
 * it can be pushed to a LayerFactory instance, which then is able to compile the supplied layers
 * into a network.
 */
struct LayerBuilder : LayerBuilderTempl<LayerBuilder> {
    LayerBuilder(const std::string& name):LayerBuilderTempl<LayerBuilder>(name) {}
};



} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
