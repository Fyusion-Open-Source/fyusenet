//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Download Layer Common Interface (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdint>
#include <functional>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../base/buffershape.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet {

class Engine;
struct StateToken;

namespace gpu {


/**
 * @brief Interface class for layers that are able to perform sync/async downloads
 */
class DownloadLayerInterface {
    friend class fyusion::fyusenet::Engine;

public:

    /**
     * @brief Wait for the download thread to finish
     */
    virtual void wait(uint64_t sequenceNo) = 0;

    /**
     * @brief Get buffer shape for output port
     */
    [[nodiscard]] virtual BufferShape getOutputShape(int port) const = 0;

#ifdef FYUSENET_MULTITHREADING
    /**
       * @brief Asynchronous layer execution
       *
       * @param sequenceNo Sequence number of the run that this layer is scheduled in
       * @param token State token for the current run
       * @param callback Callback function into the engine that notifies when a download is
       *                 complete
       *
       * This function runs the download in an asynchronous fashion. In order for this function to
       * work properly, it is important that the GL operation that led to the input tensor for
       * this layer was run in the same thread as the calling thread.
       */
      virtual void asyncForward(uint64_t sequenceNo, StateToken * token, const std::function<void(uint64_t)> & callback) = 0;
#endif
};

} // gpu namespace
} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:

