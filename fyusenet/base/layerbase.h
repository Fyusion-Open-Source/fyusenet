//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Neural Network Layer Base Class (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <vector>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/fynexception.h"
#include "parameterprovider.h"
#include "layerbuilder.h"
#include "layerflags.h"
#include "bufferspec.h"
#include "statetoken.h"
#include "../cpu/cpubuffer.h"

using fyusion::fyusenet::cpu::CPUBuffer;
using fyusion::fyusenet::BufferShape;

namespace fyusion::fyusenet {

//------------------------------------- Public Declarations ----------------------------------------


/**
 * @brief Generic base class for neural network layers
 *
 * This is the base class for all neural network layers in FyuseNet. It provides the basic interface
 * that has to be implemented in all layers, regardless on which device they run.
 *
 * The base-class here keeps track of basic information, like input tensor dimensions (we only
 * support 2D and 3D input tensors), input and output padding, a set of general flags which
 * regulate the application of activation functions and batch-norms, as well as properties like
 * layer variant (shallow or deep) and in-line residual application.
 *
 * In FyuseNet, layers are numbered and there is a strict sequential order in which layers are
 * executed, following the layer number. This is used as basic working assumption in many parts of
 * the code, for example when assigning / re-using textures as buffers between those layers.
 *
 * A bit peculiar is the way activations are treated, which is currently done by using flags in
 * most of the cases. Though FyuseNet has a few dedicated activation layers, these are usually not
 * necessary and should not be used if it can be avoided.
 * FyuseNet was primarily designed to run on mobile GPUs with GLES 3.0 (started with 2.0 even)
 * support, it takes a rather unorthodox way and integrates the activation of a layer in the
 * \e following layer during the buffer-read operations. The reason for that is that there was no
 * well-performing way to apply more or less arbitrary non-linear mappings to the \e output of a
 * shader stage that uses accumulation as its ROP operation. In benchmarks on mobile GPUs (2017), I
 * found that nearly all of our layer implementations were bandwidth-bound and the extra arithmetic
 * in the read phase had no impact on the run-time.
 *
 * Another non-standard feature which is attributed to the emphasis on GPU runtime, is the
 * RESIDUAL_INPUT flag. This can be used to route-in the output of another layer to perform
 * element-wise addition to the results of the current layer. This comes in handy when implementing
 * residual blocks and offers the benefit of reducing required memory bandwidth.
 *
 * Layers also have explicit information about input and output padding which is required to
 * compute the buffer specifiers and perform convolution operations correctly. Note that no
 * layer performs any input padding (but needs to know about that), it assumes that the data
 * arrives with the padding that was specified. On the other hand, every layer is responsible for
 * applying the correct output padding as was specified by the LayerBuilder instance that is used
 * to create a layer. Also note that FyuseNet does not support anisotropic padding. A padding of
 * 1 means that each channel is extended by 1 along each side, therefore adding 2 units per spatial
 * axis to the spatial extents of the buffer/tensor.
 *
 * Last but not least, on the GPU there are two different buffer / tensor representations which
 * are optimized towards shallow (smaller number of channels) and deep (larger number of channels)
 * tensor layouts. Consult the documentation in gpu::GPULayerBase for more information on that.
 *
 * @see gpu::GPULayerBase
 */
class LayerBase {
 public:

    /**
     * @copydoc gpu::PIXEL_PACKING
     */
    static constexpr int PIXEL_PACKING = gpu::PIXEL_PACKING;

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit LayerBase(const LayerBuilder& builder);
    LayerBase(const LayerBuilder& builder, int layerNumber);
    virtual ~LayerBase();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------

    /**
     * @brief Perform setup of the layer code
     *
     * This function performs initializations of the layer prior to be able to be used for inference.
     * Initializations may include buffer allocation, pre-computation of tables and - in the case of
     * GPUs - setting up the necessary interfaces and computation kernels to execute the inference.
     *
     * @pre In case of GPU layers, the GL context that is to be used for running the inference
     *      must be current to the calling thread
     */
    virtual void setup() = 0;

    /**
     * @brief Cleanup / deallocate resources that were allocated during setup
     *
     * @note The main reason why this is not simply done in the destructor of the class, is due to
     *       potential GPU usage, in particular the usage of OpenGL on GPUs. For deallocation of GL
     *       resources, the \e right context must be bound to the calling thread. In order to
     *       prevent API users from just relying on destructors on deletion of a network, this method
     *       reminds them to make sure that a cleanup is called in the right thread. You could do that
     *       with destructors too, but people (read: me) are just not used to that.
     *
     * @see GPULayerBase::cleanup
     */
    virtual void cleanup() = 0;

    /**
     * @brief Load layer parameters from a provider interface
     *
     * @param weights Pointer to a ParameterProvider instance that provides the parameters
     *
     * This function is responsible from loading all required parameters from the supplied provider
     * object. It is reimplemented in derived classes where needed. Check the documentation
     * in ParameterProvider and derived classes for more information.
     *
     * @see ParameterProvider
     */
    virtual void loadParameters(const ParameterProvider * weights) {
        (void)weights;
        // NOTE (mw) empty on purpose, reimplement in derived classes where needed
    }

    /**
     * @brief Obtain buffer specifiers that are required as input for this layer
     *
     * @return Vector of buffer specifiers that specify the format for each required buffer
     *
     * @see BufferSpec
     */
    [[nodiscard]] virtual std::vector<BufferSpec> getRequiredInputBuffers() const  = 0;


    /**
     * @brief Obtain buffer specifiers that are required as output for this layer
     *
     * @return Vector of buffer specifiers that specify the format for each required buffer
     *
     * @see BufferSpec
     */
    [[nodiscard]] virtual std::vector<BufferSpec> getRequiredOutputBuffers() const = 0;

    /**
     * @brief Execute the layer
     *
     * @param sequenceNo Sequence number (\b must be strictly increasing)
     * @param state Optional pointer to a StateToken that tracks and controls inference state, may
     *              be \c nullptr if state does not need to be tracked
     *
     * This function performs the actual computation that maps the input data to the output data
     * for this layer. The supplied \p sequence number must be strictly increasing per inference run
     * and may be used for debugging purposes, in case errors only manifests themselves after a
     * certain number of computation cycles. It can also be used to keep track of the total number
     * of inference runs. Internally, it is used to make sure that asynchronously transmitted data
     * is up-to-date (on PBO reads for example).
     *
     * The optional state token can be used to convey state information between layers and/or inference
     * runs. An example for state information may be the query length of a tensor that is passed in
     * sequence-learning inference.
     */
    virtual void forward(uint64_t sequenceNo, StateToken *state) = 0;

    /**
     * @brief Store computation results of layer in file for debugging purposes
     *
     * @param fileName Name of the output file to write to
     *
     * @param includePadding If \c true, the padding will be included in the output file, otherwise
     *        the padding will be ignored and only the net contents are written to the output file
     *
     * This function writes the content of the output textures as binary dump into the specified file.
     * All data will be written as little-endian 32-bit IEEE-754 floating-point numbers in a channel-
     * by-channel fashion. The data is arranged row-by-row (x-axis as innermost index) for a single
     * channel (y-axis as middle index) where the channels are stacked (channel axis as outermost
     * index).
     *
     * For WebAssembly/WebGL builds, the logging of intermediate layer data is a bit tricky and I
     * opted for a rather hacky version which is not quite scalable. Currently, this code will
     * trigger a download for every layer which has to be acknowledged by the user. Furthermore,
     * it only works if the browser window has the following code defined:
     *
     * @code
     *  function download(data, size, filename) {
     *      const view = new Uint8Array(Module.HEAPU8.buffer, data, size);
     *      const blob = new Blob([view], {
     *          type: 'octet/stream'
     *       });
     *       const url = window.URL.createObjectURL(blob);
     *       const a = document.createElement("a");
     *       document.body.appendChild(a);
     *       a.href = url;
     *       a.style = "display:none";
     *       a.download = UTF8ToString(filename);
     *       a.target = "_self";
     *       a.click();
     *       window.URL.revokeObjectURL(url);
     *       document.body.removeChild(a);
     *  }
     *
     *  window.download = download;
     * @endcode
     *
     * Other options of storing debug data in web browsers will be considered in the future.
     *
     * @note This function only works in a debug build. In release builds, this will be a no-op.
     */
    virtual void writeResult(const char *fileName, bool includePadding=false) = 0;  // NOLINT

    [[nodiscard]] virtual bool isConnected() const;
    [[nodiscard]] virtual bool isConnected(int connIndex) const;
    virtual void addInputConnection(int port, LayerBase *sender, int senderPort);
    virtual void addOutputConnection(int port, LayerBase *receiver, int receiverPort);

    /**
     * @brief Retrieve number of input ports for this layer.
     *
     * Many layers act as unaries, which corresponds to a single input port. Layers that
     * require more than one input should have one input port per input (e.g. concatenation
     * layers).
     *
     * @return # of input ports
     */
    [[nodiscard]] virtual int numInputPorts() const {
        return 1;
    }

    /**
     * @brief Retrieve (virtual) index for first channel of specified input port
     *
     * @param port The port index to retrieve the channel offset for.
     *
     * Layers that have multiple ports are internally stacking their channels from a buffer-management
     * point-of-view. In order to retrieve the right (virtual) starting channel for a certain port,
     * use this function to determine the index in the virtual channel buffers. Note that an index is
     * not necessarily an actual offset as with enumerating the channels, but may be less due to grouping
     * of channels (e.g. 4 channels per buffer in GPU-based layers).
     *
     * @return First (virtual) channel of the specified port.
     */
    [[nodiscard]] virtual int getPortChannelIndex(int port) const {
        if (port >= numInputPorts()) THROW_EXCEPTION_ARGS(FynException,"Illegal input port %d specified",port);
        return 0;
    }

    /**
     * @brief Obtain input padding value
     *
     * @return Input padding (per channel)
     *
     * Padding is assumed to be isotropic and on each side. For example, a 32x32 channels with
     * a padding of 1 will have total dimensions of 34x34
     */
    [[nodiscard]] inline int getInputPadding() const {
        return inputPadding_;
    }

    /**
     * @brief Obtain output padding value
     *
     * @return output padding (per channel)
     *
     * Padding is assumed to be isotropic and on each side. For example, a 32x32 channels with
     * a padding of 1 will have total dimensions of 34x34
     */
    [[nodiscard]] inline int getOutputPadding() const {
        return outputPadding_;
    }

    /**
     * @brief Obtain padding for residual tensor
     *
     * @return residual padding (per channel)
     *
     * @note The residual padding \b must be equivalent to the output padding
     */
    [[nodiscard]] inline int getResidualPadding() const {
        return residualPadding_;
    }

    /**
     * @brief Get (net) width of input buffer
     *
     * @return Width per channel of the input buffer/tensor, \e excluding any padding
     */
    [[nodiscard]] inline int getWidth() const {
        return width_;
    }

    /**
     * @brief Get (net) height of input buffer
     *
     * @return Height per channel of the input buffer/tensor, \e excluding any padding
     */
    [[nodiscard]] inline int getHeight() const {
        return height_;
    }

    /**
     * @brief Obtain layer flags
     *
     * @return Layer flags that are set in this layer
     */
    [[nodiscard]] inline layerflags getFlags() const {
        return flags_;
    }

    /**
     * @brief Obtain layer number
     *
     * @return Layer number
     */
    [[nodiscard]] inline int getNumber() const {
        return layerNumber_;
    }

    /**
     * @brief Retrieve (total) number of input channels
     *
     * @param port Optional port number to get the input channels for, defaults to 0
     *
     * @return Number of input channels
     *
     * For most layers, the number of input channels is the same for every port (if there are
     * multiple ports at all). For some specific layers, like a concatenation layer, the number
     * of input channels per port may differ
     */
    [[nodiscard]] virtual int numInputChannels(int port=0) const {  // NOLINT
        return inputChannels_;
    }

    /**
     * @brief Obtain number of output channels
     *
     * @return Number of output channels
     */
    [[nodiscard]] inline int numOutputChannels() const {
        return outputChannels_;
    }

    /**
     * @brief Obtain layer name / ID
     *
     * @return String containing layer name / ID
     */
    [[nodiscard]] inline const std::string& getName() const {
        return name_;
    }


    /**
     * @brief Check if a layer is valid (was properly initialized)
     *
     * @retval true %Layer was properly initialized
     * @retval false %Layer was not initialized (yet)
     */
    [[nodiscard]] inline bool isValid() const {
        return valid_;
    }

    /**
     * @brief Check if a layer is applicable running under the current execution environment
     *
     * @retval true layer will be able to operate under current execution environment
     * @retval false layer is not supported under the current execution environment
     *
     * Some layers in FyuseNet might require certain GPU functionality that may not be present
     * on every system. These layers will return \c false when the execution environment lacks
     * required functionality.
     */
    [[nodiscard]] virtual bool isApplicable() const {
        return true;
    }


    /**
     * @brief Get device type which this layer is bound to run on.
     *
     * @retval CPU If this layer only runs on the CPU
     * @retval GPU If this layer only runs on the GPU
     */
    [[nodiscard]] inline compute_device getDevice() const {
        return device_;
    }

    /**
     * @brief Check if a layer has parameters (like weights) that need to be set prior to usage
     *
     * @retval true Layer requires parameters to be loaded
     * @retval false Layer is parameter-free
     *
     * @see loadParameters
     */ 
    [[nodiscard]] bool hasParameters() const {
        return hasParameters_;
    }

 protected:

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::string name_;                               //!< Layer identifier
    layerflags flags_ = LayerFlags::NO_LAYER_FLAGS;  //!< Misc flags for this layer
    int width_ = 0;                                  //!< Width (in elements) of a single feature-map slab
    int height_ = 0;                                 //!< Height (in elements) of a single feature-map slab
    int inputChannels_= 0;                           //!< Number of input channels of the input feature-maps
    int outputChannels_ = 0;                         //!< Number of output channels of the output feature-maps
    int layerNumber_ = -1;                           //!< Layer number (defines execution order)
    int inputPadding_ = 0;                           //!< Padding on the input data
    int outputPadding_ = 0;                          //!< Padding on the output data
    int residualPadding_ = 0;                        //!< Padding on the incoming residual data (currently must be the same as #outputPadding_ )
    int inConnections_ = 0;                          //!< Number of connected input ports
    bool outputConnected_ = false;                   //!< Indicator that output port is connected
    std::vector<int> connectedInputPorts_;           //!< Port numbers of all connected input ports (see BufferSpec)
    bool valid_ = false;                             //!< Indicator that this layer is valid for use (i.e. has been properly initialized)
    bool hasParameters_ = false;                      //!< Indicator whether this layer requires parameters to be loaded prior to usage
    /**
     * Device type this layer runs on
     */
    compute_device device_ = compute_device::DEV_ILLEGAL;
};


} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:
