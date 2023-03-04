//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2022
//--------------------------------------------------------------------------------------------------
// Style-Transfer Network Base Class (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <unordered_map>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>

//-------------------------------------- Project  Headers ------------------------------------------

#include <fyusenet/fyusenet.h>
#include <fyusenet/gl/gl_sys.h>

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Neural network that implements a simplicistic image style-transfer operation
 *
 * This class implements a basic image style-transfer operation based on 3x3 by convolution layers.
 * Different styles can be used by changing the weight/bias data. To initialize this network, perform
 * the following steps:
 *   1. Instantiate the network object
 *   2. Load weights and biases by calling loadWeightsAndBiases() on the object
 *   3. Call setup() on the object
 *   4. Set either input/output buffers or textures
 *
 * To perform inference on the net, simply use the forward() method. Finally for taking down the
 * network, make sure to call cleanup() within a valid GL context prior to deleting the object.
 *
 * If the network is to be used on input and output buffers instead of directly working with input
 * and output textures and asynchronous operation is desired, make sure to invoke the asynchronous()
 * method prior to calling setup(). During the operation, the network handles the multi-buffering
 * internally but there are rules to follow:
 *   - The only safe section to query the (current) output buffer via getOutputBuffer() is when
 *     inside the asynchronous callback that was supplied to asynchronous()
 *   - The output buffer will be swapped as soon as the callback returns, using the buffer beyond
 *     that point is subject to race conditions
 */
class StyleNetBase : public fyusion::fyusenet::NeuralNetwork {
 public:
    /**
     * Common indices for the layer numbers
     */
    enum {
        UNPACK = 0,
        UPLOAD = 0,
        CONV1
    };

    constexpr static int ASYNC_BUFFERS = 2;

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    StyleNetBase(int width, int height, bool upload, bool download, const fyusion::fyusenet::GfxContextLink& ctx = fyusion::fyusenet::GfxContextLink());
    ~StyleNetBase();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual execstate forward() override;
    void setInputTexture(GLuint texture);
    GLuint getOutputTexture() const;
    GLuint getOutputFBO() const;
    void setInputBuffer(const float *data);
    CPUBuffer * getOutputBuffer();

    /**
     * @brief Enable writing of all intermediate layer results to binary dumps
     *
     * @param outDir Output directory to write the binary files to
     *
     * @pre setup() has been invoked on this instance before
     *
     * This enables writing binary dumps of all intermediate layer results into a set of output files
     * which are named after the respective layer names. These files can be used to compare against
     * reference data and/or to inspect for other errors.
     */
    void enableDebugOutput(const std::string& outDir) {
        if (!engine_) THROW_EXCEPTION_ARGS(fyusion::FynException, "Please run setup() before setting debug output");
        engine_->enableIntermediateOutput(outDir);
    }

    /**
     * @brief Enable input texture being passed in as an external OES texture
     *
     * This instructs the network to treat the incoming texture as an external OES texture. This is
     * usually the case when passing a SurfaceTexture from Android to the network. OES textures
     * require a different sampler and an extension to be enabled. This is handled in a dedicated
     * layer inside the network which is enabled when this function is called.
     */
    void enableOESInput() {
        oesInput_ = true;
    }

    virtual void loadWeightsAndBiases(float *weightsAndBiases, size_t size) = 0;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
#ifdef FYUSENET_MULTITHREADING
    void internalDLCallback(uint64_t seqNo, fyusion::fyusenet::cpu::CPUBuffer *buffer, fyusion::fyusenet::AsyncLayer::state state);
    void internalULCallback(uint64_t seqNo, fyusion::fyusenet::cpu::CPUBuffer *buffer, fyusion::fyusenet::AsyncLayer::state state);
#endif
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    GLuint inputTexture_ = 0;               //!< Handle of input texture if upload is not part of the net
    int width_ = 0;                         //!< Input and output width of the network
    int height_ = 0;                        //!< Input and output height of the network
    bool oesInput_ = false;                 //!< Indicator that there is an OES texture unpack step
    bool upload_ = false;                   //!< Indicator that there is an additional upload layer (i.e. data is not supplie via texture)
    bool download_ = false;                 //!< Indicator that the network should end with a GPU->CPU download layer

    /**
     * Stores multiple download CPU buffers for asynchronous operation.
     */
    CPUBuffer * asyncDLBuffers_[ASYNC_BUFFERS];

    /**
     * Externally supplied callback function that is invoked when an asynchronous download has
     * been performed.
     */
    std::function<void(uint64_t, fyusion::fyusenet::cpu::CPUBuffer *)> downloadCallback_;

    /**
     * Stores a key/offset map that stores the offsets for the weight/bias data for the individual
     * layers. The offset is given in number of single-precision floating point values.
     */
    std::unordered_map<int,size_t> weightOffsets_;

    /**
     * For buffer-driven networks, contains the pointers to the input buffers.
     *
     * @see setInputBuffer
     */
    fyusion::fyusenet::cpu::CPUBuffer * inBuffers_[ASYNC_BUFFERS] = {0};

#ifdef FYUSENET_MULTITHREADING
    std::mutex downloadBufferLock_;                     //!< Lock for use with #usedDownloadBuffers_ and #downloadBufferAvail_
    int usedDownloadBuffers_ = 0;                       //!< Number of currently used download buffers
    std::condition_variable downloadBufferAvail_;       //!< Condition that is notified when a download CPU buffer becomes available

    std::mutex uploadBufferLock_;                       //!< Lock for use with #uploadBusy_ , #usedUploadBuffers_ and #uploadBufferAvail_
    int usedUploadBuffers_ = 0;                         //!< Number of currently used upload buffers, max is 2
    bool uploadBusy_ = false;                           //!< Indicator if upload is currently busy and cannot accept a new input buffer
    std::condition_variable uploadBufferAvail_;         //!< Condition that is notified when either the upload is not busy anymore or the number of available upload buffers changed
#endif
};

