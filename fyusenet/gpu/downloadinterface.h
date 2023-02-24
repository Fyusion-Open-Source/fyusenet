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


//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {

class Engine;

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

#ifdef FYUSENET_MULTITHREADING
    /**
       * @brief Asynchronous layer execution
       *
       * @param sequenceNo Sequence number of the run that this layer is scheduled in
       *
       * @param callback Callback function into the engine that notifies when a download is
       *                 complete
       *
       * This function runs the download in an asynchronous fashion. In order for this function to
       * work properly, it is important that the GL operation that led to the input tensor for
       * this layer was run in the same thread as the calling thread.
       */
      virtual void asyncForward(uint64_t sequenceNo, const std::function<void(uint64_t)> & callback) = 0;
#endif
};


} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:

