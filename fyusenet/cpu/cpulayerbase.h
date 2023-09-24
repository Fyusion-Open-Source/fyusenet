//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// CPU Neural Network Layer Base (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../base/layerbase.h"
#include "cpubuffer.h"
#include "cpulayerinterface.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet::cpu {

/**
 * @brief Base class for CPU-based neural network layers
 *
 * This is the base class for all neural network layers that perform their computation on the CPU.
 * Due to FyuseNet being GPU-centric, we do not expect a lot of classes derived from this class.
 *
 * In contrast to the GPU based layers, CPU layers do not differentiate between shallow and deep
 * tensors for now, as the main use-case for CPU layers are to either perform custom post-processing
 * operations or very simple (unoptimized) convolutions.
 */
class CPULayerBase : public LayerBase, public CPULayerInterface {
 public:  
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    CPULayerBase(const LayerBuilder& builder, int layerNumber);
    ~CPULayerBase() override = default;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    void setup() override;
    void cleanup() override;
    void addCPUOutputBuffer(CPUBuffer * buf, int port= 0) override;
    void setCPUInputBuffer(CPUBuffer * buf, int port) override;
    void setCPUResidualBuffer(CPUBuffer * buf) override;
    void clearCPUInputBuffers(int port=-1) override;
    void clearCPUOutputBuffers(int port=-1) override;
    void writeResult(const char *fileName, bool includePadding) override;

    /**
     * @copydoc CPULayerInterface::hasCPUOutputBuffer
     */
    bool hasCPUOutputBuffer(int port=0) const override {
        assert(port >= 0);
        if ((int)outputs_.size() <= port) return false;
        return (outputs_.at(port) != nullptr);
    }


    /**
     * @copydoc CPULayerInterface::getCPUOutputBuffer
     */
    CPUBuffer * getCPUOutputBuffer(int port=0) const override {
        assert(port >= 0);
        if ((int)outputs_.size() <= port) return nullptr;
        return outputs_[port];
    }


    /**
     * @copydoc CPULayerInterface::getCPUInputBuffer
     */
    CPUBuffer * getCPUInputBuffer(int port=0) const override {
        assert(port >= 0);
        if ((int)inputs_.size() <= port) return nullptr;
        return inputs_[port];
    }

 protected:
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::vector<CPUBuffer *> inputs_;         //!< List of input buffers for this layer
    std::vector<CPUBuffer *> outputs_;        //!< List of output buffers for this layer
    std::vector<CPUBuffer *> residuals_;      //!< List of residual buffers for this layer
};

} // fyusion::fyusenet::cpu namespace


// vim: set expandtab ts=4 sw=4:

