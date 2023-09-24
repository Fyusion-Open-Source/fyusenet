//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Asynchronous Layer Interface (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "layerbase.h"

namespace fyusion::fyusenet {

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Interface for asynchronous layers
 *
 * Note that layers deriving from that interface are not necessarily asynchronous, they just have
 * the \e option to run asynchronously. If the user sets up these layers in a synchronous, fashion
 * the asynchronicity is not used at all.
 *
 * @see gpu::UploadLayer, gpu::DownloadLayer
 */
class AsyncLayer {
 public:

    /**
     * @brief Enumerator for asynchronous upload/download states
     */
    enum state {
         UPLOAD_COMMENCED,          //!< Upload has started (data was copied from original buffer)
         UPLOAD_DONE,               //!< Upload has been fully pushed to the GL pipeline
         DOWNLOAD_COMMENCED,        //!< Download has been started and it is safe to swap the buffer
         DOWNLOAD_DONE,             //!< Download is complete and CPU buffer contains correct data
         ASYNC_ERROR                //!< Something went wrong
     };

    /**
     * @brief Check if layer is supposed to run asynchronously
     *
     * @retval true Layer shall run asynchronously
     * @retval false Layer shall run synchronously
     */
    [[nodiscard]] virtual bool isAsync() const = 0;

    /**
     * @brief Add asynchronous dependency on the output of this layer
     *
     * @param target Pointer to target layer that uses the output of this layer in an
     *                asynchronous fashion
     *
     * @param channelOffset First/lowest channel-index in the receiving layer that is assigned to
     *                      the texture set written by the asynchronous source
     *
     * @see lastAsyncDependency()
     */
    virtual void addAsyncDependency(LayerBase *target, int channelOffset) {
        int layerno = target->getNumber();
        lastAsyncDependency_ = std::max(layerno, lastAsyncDependency_);
        firstAsyncDependency_ = (firstAsyncDependency_ == -1) ? layerno : (std::min(layerno, firstAsyncDependency_));
        // we do not expect to have a lot of dependencies, so a linear search is OK here
        for (auto it = dependencies_.begin(); it != dependencies_.end(); ++it) if ((*it) == target) return;
        dependencies_.push_back(target);
        dependencyOffsets_.push_back(channelOffset);
    }

    /**
     * @brief Retrieve last (highest) layer number that has an asynchronous dependency on this layer
     *
     * @return Layer number for last (highest) asynchronous dependency, or -1 if there is none
     *
     * @see addAsyncDependency();
     */
    [[nodiscard]] virtual int lastAsyncDependency() const {
        return lastAsyncDependency_;
    }

    /**
     * @brief Retrieve first (lowest) layer number that has an asynchronous dependency on this layer
     *
     * @return Layer number for first (lowest) asynchronous dependency, or -1 if there is none
     *
     * @see addAsyncDependency();
     */
    [[nodiscard]] virtual int firstAsyncDependency() const {
        return firstAsyncDependency_;
    }

 protected:

    std::vector<LayerBase *> dependencies_;      //!< List of (asynchronous) dependency layers, only used in async layers
    std::vector<int> dependencyOffsets_;         //!< List of port numbers for asynchronous dependencies, only used in async layers
    int lastAsyncDependency_ = -1;               //!< Highest layer number for subsequent layers that have an asynchronous dependency on this layer's output (-1 if none)
    int firstAsyncDependency_ = -1;              //!< Lowest layer number for subsequent layers that have an asynchronous dependency on this layer's output (-1 if none)
};


} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:

