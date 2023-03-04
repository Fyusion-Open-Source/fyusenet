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

// ------------------------------------ Public Declarations ----------------------------------------

/**
 * @brief Sample network that runs a ResNet-50 network
 *
 * @see https://pytorch.org/hub/nvidia_deeplearningexamples_resnet50
 */
class ResNet50 : public fyusion::fyusenet::NeuralNetwork {

    constexpr static int ASYNC_BUFFERS = 2;         // NOTE (mw) the async support for this net is not fully done
    constexpr static int IMAGE_SIZE = 224;

 public:
    ResNet50(const fyusion::fyusenet::GfxContextLink& ctx = fyusion::fyusenet::GfxContextLink());
    ~ResNet50();

    virtual execstate forward() override;

    void setInputBuffer(const float *data);
    CPUBuffer * getOutputBuffer();
    void loadWeightsAndBiases(const float *data, size_t numFloats);
 protected:
    virtual fyusion::fyusenet::CompiledLayers buildLayers() override;
    virtual void initializeWeights(fyusion::fyusenet::CompiledLayers & layers) override;
    virtual void connectLayers(fyusion::fyusenet::CompiledLayers & layers, fyusion::fyusenet::BufferManager * buffers) override;
    void initializeWeightOffsets();

    std::unordered_map<int, uint32_t> weightOffsets_;
    std::unordered_map<int, uint32_t> weightSizes_;
    float * wbData_ = nullptr;
    uint32_t totalWeightBytes_ = 0;

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
     * For buffer-driven networks, contains the pointers to the input buffers.
     *
     * @see setInputBuffer
     */
    fyusion::fyusenet::cpu::CPUBuffer * inBuffers_[ASYNC_BUFFERS] = {0};

    GLuint outputTexture_ = 0;                          //!< GL handle of output texture

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
