//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2023
//--------------------------------------------------------------------------------------------------
// ResNet (50) Classification Network (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


#pragma once

// --------------------------------------- System Headers ------------------------------------------

#include <cstdint>
#include <unordered_map>
#include <memory>

// -------------------------------------- Project Headers ------------------------------------------

#include <fyusenet/fyusenet.h>
#include <fyusenet/gl/gl_sys.h>
#include "../helpers/resnet_provider.h"

// ------------------------------------ Public Declarations ----------------------------------------

class ResNet50Provider;

/**
 * @brief Sample network that runs a ResNet-50 network
 *
 * @see https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50
 */
class ResNet50 : public fyusion::fyusenet::NeuralNetwork {

    constexpr static int ASYNC_BUFFERS = 2;         // NOTE (mw) the async support for this net is not fully done
    constexpr static int IMAGE_SIZE = 224;

 public:
    using fyusion::fyusenet::NeuralNetwork::forward;

    ResNet50(bool upload, bool download, const fyusion::fyusenet::GfxContextLink& ctx = fyusion::fyusenet::GfxContextLink());
    ~ResNet50() override;

    execstate forward(fyusion::fyusenet::StateToken * token) override;
    void setInputBuffer(const float *data);
    CPUBuffer * getOutputBuffer();

    /**
     * @brief Set raw input texture for the network
     *
     * @param texture Raw OpenGL texture handle to be used as input for the network
     */
    void setInputTexture(GLuint texture) {
        inputTexture_ = texture;
        inputTextureChanged_ = true;
    }

    /**
     * @brief Set weights/biases provider into network
     *
     * @param params Pointer to parameter provider
     *
     * This function merely sets the internal parameter provider (nothing is loaded into the network
     * here) and takes ownership over the supplied data.
     */
    void setParameters(ResNet50Provider * params) {
        delete parameters_;
        parameters_ = params;
    }

    void enableLog(const std::string& dir) {
        logDir_ = dir;
    }

 protected:
    fyusion::fyusenet::CompiledLayers buildLayers() override;
    void initializeWeights(fyusion::fyusenet::CompiledLayers & layers) override;
    void connectLayers(fyusion::fyusenet::CompiledLayers & layers, fyusion::fyusenet::BufferManager * buffers) override;
#ifdef FYUSENET_MULTITHREADING
    void internalDLCallback(uint64_t seqNo, fyusion::fyusenet::cpu::CPUBuffer *buffer, fyusion::fyusenet::AsyncLayer::state state);
    void internalULCallback(uint64_t seqNo, fyusion::fyusenet::cpu::CPUBuffer *buffer, fyusion::fyusenet::AsyncLayer::state state);
#endif
    bool upload_ = false;                           //!< Indicator that there is an additional upload layer (i.e. data is not supplie via texture)
    bool download_ = false;                         //!< Indicator that the network should end with a GPU->CPU download layer
    ResNet50Provider * parameters_ = nullptr;       //!< Pointer to instance that provides the weights/biases
    GLuint inputTexture_ = 0;                       //!< GL handle of input texture
    volatile bool inputTextureChanged_ = false;     //!< Indicator that the input texture has changed and needs to be re-bound
    std::string logDir_;

    /**
     * Stores multiple download CPU buffers for asynchronous operation.
     */
    CPUBuffer * asyncDLBuffers_[ASYNC_BUFFERS] = {nullptr};

    /**
     * Externally supplied callback function that is invoked when an asynchronous download has
     * been performed.
     */
    std::function<void(uint64_t, fyusion::fyusenet::cpu::CPUBuffer *)> downloadCallback_;

    /**
     * For buffer-driven networks, contains the pointers to the input buffers.
     *
     * @see setCPUInputBuffer
     */
    fyusion::fyusenet::cpu::CPUBuffer * inBuffers_[ASYNC_BUFFERS] = {nullptr};

    fyusion::fyusenet::gpu::GPUBuffer * gpuIn_ = nullptr;
    fyusion::fyusenet::gpu::GPUBuffer * gpuOut_ = nullptr;
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

// vim: set expandtab ts=4 sw=4:
