//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Upload/Download GPU Layer Builder (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <vector>
#include <functional>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gfxcontextlink.h"
#include "gpulayerbuilder.h"
#include "../base/bufferspec.h"
#include "../cpu/cpubuffer.h"
#include "../base/asynclayerinterface.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {
using namespace opengl;
namespace fyusenet {
namespace gpu {

/**
 * @brief Templatized anchor for GPU upload/download layer builders
 */
template<typename D = LayerBuilderTempl<>>
struct UpDownLayerBuilderTempl : GPULayerBuilderTempl<D> {

    enum dir {
        UPLOAD = 0,
        DOWNLOAD
    };

    /**
     * @brief Constructor
     *
     * @param direction Data direction, either upload to GPU or download from GPU
     * @param name Name to be assigned to the built layer
     */
    UpDownLayerBuilderTempl(dir direction, const std::string& name) : GPULayerBuilderTempl<D>(name), direction_(direction) {
        switch (direction) {
            case UPLOAD:
                LayerBuilderTempl<D>::type_ = LayerType::UPLOAD;
                break;
            default:
                LayerBuilderTempl<D>::type_ = LayerType::DOWNLOAD;
                break;
        }
    }

#ifdef FYUSENET_MULTITHREADING
    /**
     * @brief Build a layer that runs asynchronously to maximize throughput
     *
     * @return Reference to builder after assignment
     */
    D & async()  {
        async_ = true;
        return *(D *)this;
    }
#endif

    /**
     * @brief Build a layer that uses the specific datatype <i>on the CPU</i> for the operation
     *
     * @param dt Datatype identifier (cave: limited support for some datatype/operations)
     *
     * @return Reference to builder after assignment
     */
    D & dataType(BufferSpec::dtype dt) {
        dataType_ = dt;
        return *(D *)this;
    }

#ifdef FYUSENET_MULTITHREADING
    /**
     * @brief Assign callback for asynchronous uploads and downloads
     *
     * @param cb Callback function which takes the sequence ID as an argument
     *
     * @return Reference to builder after assignment
     *
     * Assigns a user-supplied callback function to the upload or download layer that will be invoked
     * on:
     *   - after data was downloaded from the GPU and has been transferre to the CPU buffer on
     *     download layers
     *   - after the upload to the GPU has been triggered and the CPU upload buffer can be re-used
     *
     * The parameters passed to the callback function are the sequence number and the currently
     * active CPUBuffer object that was read from or written to.
     *
     * The supplied callback may be called from a different thread than the engine thread and
     * therefore users have to make sure to stick to thread-safety.
     * In addition, the callback is \e time-critical , which means that it will hold up the
     * processing queue. So if a huge chunk of work has to be done, do not do it inside the callback.
     *
     * @warning The callback implementation itself is \e time-critical.
     */
    D & callback(std::function<void(uint64_t, cpu::CPUBuffer *, AsyncLayer::state)> cb) {
        callback_ = cb;
        return *(D *)this;
    }
#endif

    dir direction_;                 //!< Data direction (either upload to GPU or download from GPU)
#ifdef FYUSENET_MULTITHREADING
    bool async_ = false;            //!< Whether or not the layer should be working asynchronously (default is synchronous)

    /**
     * Callback function for asynchronous upload and download layers, will be called on various
     * occasions with the state set as:
     *   - \c UPLOAD_COMMENCED when an upload was started and the input buffer may be changed
     *   - \c UPLOAD_DONE when an upload has completed in the background
     *   - \c DOWNLOAD_DONE when a download has completed in the background
     *   - \c ERROR when an error has occured
     *
     * Note that \c UPLOAD_COMMENCED states may be called from within the same thread, be aware
     * of locks.
     */
    std::function<void(uint64_t, cpu::CPUBuffer *, AsyncLayer::state)> callback_;
#endif

    /**
     * Datatype <i>on the CPU</i> to be used for the upload/download operation (defaults to 32-bit float)
     */
    BufferSpec::dtype dataType_ = BufferSpec::FLOAT;
};


/**
 * @brief Builder class for upload and download layers on the GPU
 *
 * In order to get data from the CPU to the GPU and vice versa it needs to be uploaded/downloaded
 * to/from the GPU. In terms of OpenGL, the upload is usually done via a call to
 * <a href="https://registry.khronos.org/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml">glTexImage2D</a>
 * and the download is usually done via a call to
 * <a href="https://registry.khronos.org/OpenGL-Refpages/gl4/html/glReadPixels.xhtml">glReadPixels</a>.
 * Both calls - in particular \c glReadPixels() - introduce significant delay and OpenGL offers
 * a few workarounds to alleviate the associated time penalty. Using asynchronous upload and
 * download in conjunction with
 * <a href="https://registry.khronos.org/OpenGL-Refpages/gl4/html/glFenceSync.xhtml">fences</a>,
 * the throughput in the processing pipeline is maximized. The latency however will not be
 * reduced significantly.
 *
 * For asynchronous download type layers in particular, the user can supply a callback function
 * which will be called \e after a download has been
 *
 */
struct UpDownLayerBuilder : UpDownLayerBuilderTempl<UpDownLayerBuilder> {

    /**
     * @brief Constructor
     *
     * @param direction Data direction, either upload to GPU or download from GPU
     * @param name Name to be assigned to the built layer
     */
    // TODO (mw) supply data type
    UpDownLayerBuilder(dir direction, const std::string& name) : UpDownLayerBuilderTempl<UpDownLayerBuilder>(direction, name) {
    }
    using UpDownLayerBuilderTempl<UpDownLayerBuilder>::UpDownLayerBuilderTempl;
};

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
