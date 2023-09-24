//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Layer Interface (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "cpubuffer.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::cpu {

/**
 * @brief Interface for CPU-based data processing
 *
 * This abstract interface is used for adding a CPU-facing tensor-processing part to various
 * layer types. It serves as a base class for CPU-side layers, as well as an interface for
 * layer types that are interfacing between GPU and CPU.
 *
 * @see CPULayerBase, DownloadLayer
 */
class CPULayerInterface {
    friend class BufferManager;
 public:

    /**
     * @brief Register output buffer with this layer
     *
     * @param buf Pointer to CPUBuffer instance that contains the tensor buffer to write output into
     * @param port Optional output port number, currently only one output port is supported
     *
     * This function appends a buffer to the output buffer list. As opposed to the input, layers
     * currently only have one output port, but may be extended to support multiple output ports
     * later (or never).
     *
     * @note This interface does not take ownership over the supplied buffer, it is up to the caller to
     *       maintain its life-cycle.
     *
     * @see BufferManager::connectLayers, BufferManager::connectCPULayers, getCPUOutputBuffer
     */
    virtual void addCPUOutputBuffer(CPUBuffer * buf, int port= 0) = 0;

    /**
     * @brief Register input buffer with this layer
     *
     * @param buf Pointer to CPUBuffer instance that contains the tensor buffer to read data from
     * @param port Input port of this layer to connect the supplied \p buf to
     *
     * This function sets a buffer to the input buffer list at the provided \p port location. A
     * layer can have several input ports and its semantics are determined individually by each
     * layer implementation.
     *
     * @note This interface does not take ownership over the supplied buffer, it is up to the caller
     *       to maintain its life-cycle.
     *
     * @see BufferManager::connectLayers, BufferManager::connectCPULayers, getCPUInputBuffer
     */
    virtual void setCPUInputBuffer(CPUBuffer * buf, int port) = 0;

    /**
     * @brief Append buffer to the list of residual buffers
     *
     * @param buf Pointer to CPUBuffer instance that contains the tensor buffer to read data from
     *
     * This function appends the specified buffer to the list of residual buffers for this
     * layer. In contrast to the input buffer, the residual buffers are restricted to one "port"
     * as they are simply added to the output of the layer. For this reason, no port needs to be
     * specified, as there is only one residual buffer per layer.
     *
     * @note This interface does not take ownership over the supplied buffer, it is up to the caller to
     *       maintain its life-cycle.
     */
    virtual void setCPUResidualBuffer(CPUBuffer * buf) = 0;

    /**
     * @brief Clear/reset input buffers for this layer
     *
     * @param port Input port to clear the buffers from, or -1 to clear \e all ports
     *
     * @post The layer has no associated input buffers on the specified port(s)
     *
     * @note As this interface does not take ownership over the supplied buffer, it is up to the caller
     *       to release the buffers or reuse them. This call merely removes the reference to the
     *       buffer(s).
     */
    virtual void clearCPUInputBuffers(int port=-1) = 0;


    /**
     * @brief Clear/reset output buffers for this layer
     *
     * @param port Output port to clear the buffers from, or -1 to clear \e all ports
     *
     * @post The layer has no associated output buffers on the specified port(s)
     *
     * @note As this class does not take ownership over the supplied buffer, it is up to the caller
     *       to release the buffers or reuse them. This call merely removes the reference to the
     *       buffer(s).
     */
    virtual void clearCPUOutputBuffers(int port=-1) = 0;


    /**
     * @brief Check if a port has a CPU output buffer assigned
     *
     * @param port Port number to check (defaults to 0)
     *
     * @retval true Specified port has an output (CPU) buffer assigned
     * @retval false otherwise
     */
    virtual bool hasCPUOutputBuffer(int port=0) const = 0;

    /**
     * @brief Retrieve output CPU buffer for a specified port
     *
     * @param port Port to retrieve buffer for
     *
     * @return Pointer to CPUBuffer instance that is assigned to the specified \p port
     */
    virtual CPUBuffer * getCPUOutputBuffer(int port=0) const = 0;


    /**
     * @brief Retrieve input CPU buffer for a specified port
     *
     * @param port Port to retrieve buffer for
     *
     * @return Pointer to CPUBuffer instance that is assigned to the specified \p port
     */
    virtual CPUBuffer * getCPUInputBuffer(int port=0) const = 0;
};


} // fyusion::fyusenet::cpu namespace


// vim: set expandtab ts=4 sw=4:
