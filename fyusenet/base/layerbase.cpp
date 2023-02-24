//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Neural Network Layer Base Class 
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "layerbase.h"

namespace fyusion {
namespace fyusenet {
//-------------------------------------- Global Variables ------------------------------------------

constexpr int LayerBase::PIXEL_PACKING;

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param builder
 * @param layerNumber
 *
 * Perform basic initialization of the instance with data obtained from the supplied \p builder
 * object. The provided \p layerNumber if important for the order of execution of the layers,
 * as they are executed sequentially based on that number. It is up to the user to make sure
 * that the layer numbering is correct and that there are no clashes where more than one layer
 * uses the same layer number.
 */
LayerBase::LayerBase(const LayerBuilder& builder, int layerNumber) {
    width_ = builder.width();
    height_ = builder.height();
    inputChannels_ = builder.in();
    outputChannels_ = builder.out();
    layerNumber_ = layerNumber;
    flags_ = builder.getFlags();
    inputPadding_ = builder.inputPadding_;
    outputPadding_ = builder.outputPadding_;
    residualPadding_ = builder.residualPadding_;
    name_ = builder.name_;
    leakyReLU_ = builder.leakyReLU_;
    lowClip_ = builder.clipLow_;
    highClip_ = builder.clipHigh_;
    device_ = builder.device_;
    assert(device_ != compute_device::DEV_ILLEGAL);
}


/**
 * @brief Destructor
 *
 * Invalidates the layer.
 */
LayerBase::~LayerBase() {
    connectedInputPorts_.clear();
    valid_ = false;
    device_ = compute_device::DEV_ILLEGAL;
}


/**
 * @brief Indicate an input connection to this layer
 *
 * @param port The port that was connected
 *
 * @param sender Pointer to layer that is sending the data (may be \c nullptr if there is no
 *               layer serving as data origin)
 *
 * @param senderPort If \p sender exists, this specified the port number of the sender
 *
 * The \e presence of connections is tracked separately from the buffer/texture assignment. As
 * in the case of GPU layers, a single port-to-port connection may consist of several textures
 * being passed around. This function tells the layer that the specified input \p port has been
 * completely connected to another layer, meaning that all buffers/textures are accounted for
 * on this specific \p port.
 *
 * @see BufferManager, isConnected
 */
void LayerBase::addInputConnection(int port, LayerBase *sender, int senderPort) {
    if (!isConnected(port)) {
        connectedInputPorts_.push_back(port);
        inConnections_++;
    }
}


/**
 * @brief Indicate an output connection to this layer
 *
 * @param port Output port that was connected (defaults to 0), currently only 0 is supported
 *
 * @param receiver Pointer to receiving layer, ignored by this default implementation. The
 *                 pointer may be a \c nullptr if the receiver is not another layer but a
 *                 texture/buffer sink
 *
 * @param receiverPort Port number on the receiving layer, ignored by this default implementation
 *
 * The \e presence of connections is tracked separately from the buffer/texture assignment. As
 * in the case of GPU layers, a single port-to-port connection may consist of several textures
 * being passed around. This function checks if the output(s) and \e all input ports have been
 * marked as connected by invoking the addInputConnection() and addOutputConnection() method.
 *
 * @note Currently we have no layer that supports more than one output, therefore this base
 *       implementation just sets a flag internally and the \p port parameter is ignored (but
 *       should be set to zero). Override this default implementation if more fine-granular
 *       behaviour is desired.
 */
void LayerBase::addOutputConnection(int port, LayerBase *receiver, int receiverPort) {
    assert(port == 0);
    outputConnected_ = true;
}


/**
 * @brief Check if layer is properly connected (i.e. all input and output ports are connected)
 *
 * @retval true If all input and output ports are connected
 * @retval false If there are missing connection
 *
 * The \e presence of connections is tracked separately from the buffer/texture assignment. As
 * in the case of GPU layers, a single port-to-port connection may consist of several textures
 * being passed around. This function checks if the output(s) and \e all input ports have been
 * marked as connected by invoking the addInputConnection() and addOutputConnection() method.
 */
bool LayerBase::isConnected() const {
    if (!outputConnected_) return false;
    auto ins = getRequiredInputBuffers();
    for (auto it = ins.begin(); it != ins.end(); ++it) {
        if (!isConnected((*it).port_)) return false;
    }
    if (inConnections_ < getRequiredInputBuffers().size()) return false;
    return true;
}


/**
 * @brief Check if a specific input port of this layer is connected
 *
 * @param port Input port number to test for connectivity
 *
 * @retval true Supplied \p port is connected
 * @retval false Supplied \p port is \b not connected
 *
 * @note This implementation is not optimal since it uses a linear search, however we rarely
 *       encounter layers with more than 8 ports and using a set/hash as data structure adds
 *       code size without a strong benefit.
 */
bool LayerBase::isConnected(int port) const {
    for (int i=0; i < (int)connectedInputPorts_.size(); i++) if (connectedInputPorts_.at(i) == port) return true;
    return false;
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
