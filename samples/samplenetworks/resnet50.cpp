//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2022
//--------------------------------------------------------------------------------------------------
// ResNet (50) Classification Network
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstring>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include <fyusenet/fyusenet.h>
#include "resnet50.h"

//-------------------------------------- Global Variables ------------------------------------------

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Construct ResNet-50 classifier network
 *
 * @param ctx Link to GL context that the network should use
 */
ResNet50::ResNet50(const fyusion::fyusenet::GfxContextLink& ctx) : fyusion::fyusenet::NeuralNetwork(ctx),
    upload_(true), download_(true) {
    initializeWeightOffsets();
    wbData_ = new float[totalWeightBytes_ / sizeof(float)];
    memset(wbData_, 0, totalWeightBytes_);
}


/**
 * @brief Destructor
 *
 * Deallocates resources
 */
ResNet50::~ResNet50() {
    delete [] wbData_;
    wbData_ = nullptr;
}



/**
 * @copydoc fyusion::fyusenet::NeuralNetwork::forward()
 */
fyusion::fyusenet::NeuralNetwork::execstate ResNet50::forward() {
#ifdef FYUSENET_MULTITHREADING
    if (async_) {
        std::unique_lock<std::mutex> lck(downloadBufferLock_);
        downloadBufferAvail_.wait(lck, [this]() { return (usedDownloadBuffers_ < ASYNC_BUFFERS);});
        usedDownloadBuffers_++;
        lck.unlock();
        return fyusion::fyusenet::NeuralNetwork::forward();
    } else {
        return fyusion::fyusenet::NeuralNetwork::forward();
    }
#else
    return fyusion::fyusenet::NeuralNetwork::forward();
#endif
}



/**
 * @brief Load weight and bias data from memory into network
 *
 * @param weightsAndBiases Pointer to weight and bias data that controls the convolution operations
 *
 * @param size Number of <i>floating point elements</i> in the weight and bias buffer
 *
 * This function uploads the supplied weights and bias data to the neural network. It may be used
 * multiple times to change the style of the net.
 *
 * @see ConvLayerInterface::loadWeightsAndBiases
 */
void ResNet50::loadWeightsAndBiases(const float *data, size_t numFloats) {
    assert(numFloats * sizeof(float) == totalWeightBytes_);
    assert(wbData_);
    memcpy(wbData_, data, totalWeightBytes_);
    if (setup_) {
        assertContext();
        initializeWeights(engine_->getLayers());
    }
}


/**
 * @brief Try to set input CPU buffer to network by copying the supplied buffer contents
 *
 * @param data Pointer to 32-bit single-precision FP buffer. Must be the same dimensions as the
 *             network processing size and must be 3-channel RGB floats in [0,1] in shallow GPU
 *             order (i.e. triplets of RGB)
 *
 * @pre Network has been set up already
 *
 * @retval true if input buffer was successfully set
 * @retval false if there was no free slot to set the buffer, only happens in asynchronous operation
 *
 * Set an input RGB image to the network. If the network is busy, this function will wait until an
 * upload slot becomes available.
 *
 * @note No ownership over the supplied data is taken, caller is responsible for deallocation. It
 *       is safe to delete and/or overwrite the supplied \p data after this call, because a
 *       deep-copy is made.
 *
 * @warning This function is not re-entrant and must be used from the same thread as forward().
 *          In asynchronous implementations, a call to forward() must be executed after setting
 *          the input buffer to push the buffer through the pipeline, otherwise deadlocks will
 *          occur.
 */
void ResNet50::setInputBuffer(const float *data) {
    using namespace fyusion::fyusenet;
    assert(setup_);
#ifdef FYUSENET_MULTITHREADING
    int numbuffers = (async_) ? ASYNC_BUFFERS : 1;
#else
    int numbuffers = 1;
#endif
    // -------------------------------------------------------
    // Make sure that we have the necessary amount of buffers
    // allocated...
    // -------------------------------------------------------
    for (int i=0; i < numbuffers; i++) {
        if (!inBuffers_[i]) {
            inBuffers_[i] = new cpu::CPUBuffer(cpu::CPUBufferShape(IMAGE_SIZE, IMAGE_SIZE, 3, 0, cpu::CPUBufferShape::type::FLOAT32, BufferSpec::order::GPU_SHALLOW));
        }
    }
    gpu::UploadLayer * upload = static_cast<gpu::UploadLayer *>(engine_->getLayers()["upload"]);
    assert(upload);
    CPUBuffer * buf = nullptr;
    {
#ifdef FYUSENET_MULTITHREADING
        assert(usedUploadBuffers_ <= ASYNC_BUFFERS);
        // -------------------------------------------------------
        // For async uploads, we employ multiple upload buffers
        // which we just cycle through.
        // -------------------------------------------------------
        std::unique_lock<std::mutex> lck(uploadBufferLock_);
        uploadBufferAvail_.wait(lck, [this]() { return (uploadBusy_ == false) && (usedUploadBuffers_ < ASYNC_BUFFERS); });
        if (upload->getInputBuffer()) {
            buf = (upload->getInputBuffer() == inBuffers_[0]) ? inBuffers_[1] : inBuffers_[0];
        } else buf = inBuffers_[0];
        usedUploadBuffers_++;
        uploadBusy_ = true;
#else
        buf = inBuffers_[0];
#endif
    }
    // one deep-copy operation too many
    float * tgt = buf->map<float>();
    assert(tgt);
    // NOTE (mw) it would be cleaner to supply 4-channel data (RGBA) here
    memcpy(tgt, data, buf->shape().bytes(CPUBufferShape::order::CHANNELWISE));
    buf->unmap();
    upload->setInputBuffer(buf, 0);
}



/**
 * @brief Get pointer to output buffer for download-enabled networks
 *
 * @return Pointer to CPUBuffer that is written to by the download layer
 *
 * In case download was activated for this network, this function will return a pointer to the
 * CPUBuffer instance that is being written by the download layer. In case download is not
 * activated, this function will return a \c nullptr .
 *
 * @warning If asynchronous operation is enabled, the output buffer is subject to change and
 *          must be queried inside the download callback every time it is invoked. Obtaining
 *          the output buffer by any other means is prone to errors.
 */
fyusion::fyusenet::cpu::CPUBuffer * ResNet50::getOutputBuffer() {
    using namespace fyusion::fyusenet;
    if ((!download_) || (!setup_)) return nullptr;
    gpu::deep::DeepDownloadLayer * dwn = static_cast<gpu::deep::DeepDownloadLayer *>(engine_->getLayers()["download"]);
    return dwn->getOutputBuffer(0);
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @copydoc NeuralNetwork::buildLayers
 */
fyusion::fyusenet::CompiledLayers ResNet50::buildLayers() {
    using namespace fyusion::fyusenet;
    using namespace fyusion::fyusenet::gpu;
    std::shared_ptr<LayerFactory> factory = getLayerFactory();;

    gpu::UpDownLayerBuilder * up = new gpu::UpDownLayerBuilder(gpu::UpDownLayerBuilder::UPLOAD, "upload");
    up->shape(3, 224, 224, 3).context(context()).number(0);
    up->push(factory);

    GPULayerBuilder * bn2 = new GPULayerBuilder("BN2");
    bn2->type(LayerType::BATCHNORM).number(2).shape(3,224,224,3).outputPadding(1).context(context_);
    bn2->push(factory);
    ConvLayerBuilder * conv3 = new ConvLayerBuilder(7,"Conv3");
    conv3->type(LayerType::CONVOLUTION2D).number(3).shape(64,224,224,3).downsample(2).deep().inputPadding(1).outputPadding(1).postfixNorm(NormType::BATCHNORM).context(context_);
    conv3->push(factory);
    PoolLayerBuilder * maxpool4 = new PoolLayerBuilder(PoolLayerBuilder::POOL_MAX,"MaxPool4");
    maxpool4->type(LayerType::MAXPOOL2D).number(4).shape(64,112,112,64).poolSize(3,3).downsample(2).deep().inputPadding(1).prefixAct(ActType::RELU).context(context_);
    maxpool4->push(factory);
    GPULayerBuilder * bn5 = new GPULayerBuilder("BN5");
    bn5->type(LayerType::BATCHNORM).number(5).shape(64,56,56,64).deep().context(context_);
    bn5->push(factory);
    ConvLayerBuilder * conv6 = new ConvLayerBuilder(1,"Conv6");
    conv6->type(LayerType::CONVOLUTION2D).number(6).shape(64,56,56,64).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv6->push(factory);
    ConvLayerBuilder * conv7 = new ConvLayerBuilder(1,"Conv7");
    conv7->type(LayerType::CONVOLUTION2D).number(7).shape(256,56,56,64).deep().prefixAct(ActType::RELU).context(context_);
    conv7->push(factory);
    ConvLayerBuilder * conv8 = new ConvLayerBuilder(3,"Conv8");
    conv8->type(LayerType::CONVOLUTION2D).number(8).shape(64,56,56,64).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv8->push(factory);
    ConvLayerBuilder * conv9 = new ConvLayerBuilder(1,"Conv9");
    conv9->type(LayerType::CONVOLUTION2D).number(9).shape(256,56,56,64).deep().prefixAct(ActType::RELU).residual(ActType::NONE,false).context(context_);
    conv9->push(factory);
    GPULayerBuilder * bn10 = new GPULayerBuilder("BN10");
    bn10->type(LayerType::BATCHNORM).number(10).shape(256,56,56,256).deep().context(context_);
    bn10->push(factory);
    ConvLayerBuilder * conv11 = new ConvLayerBuilder(1,"Conv11");
    conv11->type(LayerType::CONVOLUTION2D).number(11).shape(64,56,56,256).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv11->push(factory);
    ConvLayerBuilder * conv12 = new ConvLayerBuilder(3,"Conv12");
    conv12->type(LayerType::CONVOLUTION2D).number(12).shape(64,56,56,64).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv12->push(factory);
    ConvLayerBuilder * conv13 = new ConvLayerBuilder(1,"Conv13");
    conv13->type(LayerType::CONVOLUTION2D).number(13).shape(256,56,56,64).deep().prefixAct(ActType::RELU).residual(ActType::NONE,false).context(context_);
    conv13->push(factory);
    GPULayerBuilder * bn14 = new GPULayerBuilder("BN14");
    bn14->type(LayerType::BATCHNORM).number(14).shape(256,56,56,256).deep().context(context_);
    bn14->push(factory);
    ConvLayerBuilder * conv15 = new ConvLayerBuilder(1,"Conv15");
    conv15->type(LayerType::CONVOLUTION2D).number(15).shape(64,56,56,256).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv15->push(factory);
    ConvLayerBuilder * conv16 = new ConvLayerBuilder(3,"Conv16");
    conv16->type(LayerType::CONVOLUTION2D).number(16).shape(64,56,56,64).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv16->push(factory);
    ConvLayerBuilder * conv17 = new ConvLayerBuilder(1,"Conv17");
    conv17->type(LayerType::CONVOLUTION2D).number(17).shape(256,56,56,64).deep().prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).residual(ActType::NONE,true).context(context_);
    conv17->push(factory);
    ConvLayerBuilder * conv18 = new ConvLayerBuilder(1,"Conv18");
    conv18->type(LayerType::CONVOLUTION2D).number(18).shape(128,56,56,256).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv18->push(factory);
    ConvLayerBuilder * conv19 = new ConvLayerBuilder(1,"Conv19");
    conv19->type(LayerType::CONVOLUTION2D).number(19).shape(512,56,56,256).downsample(2).deep().prefixAct(ActType::RELU).context(context_);
    conv19->push(factory);
    ConvLayerBuilder * conv20 = new ConvLayerBuilder(3,"Conv20");
    conv20->type(LayerType::CONVOLUTION2D).number(20).shape(128,56,56,128).downsample(2).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv20->push(factory);
    ConvLayerBuilder * conv21 = new ConvLayerBuilder(1,"Conv21");
    conv21->type(LayerType::CONVOLUTION2D).number(21).shape(512,28,28,128).deep().prefixAct(ActType::RELU).residual(ActType::NONE,false).context(context_);
    conv21->push(factory);
    GPULayerBuilder * bn22 = new GPULayerBuilder("BN22");
    bn22->type(LayerType::BATCHNORM).number(22).shape(512,28,28,512).deep().context(context_);
    bn22->push(factory);
    ConvLayerBuilder * conv23 = new ConvLayerBuilder(1,"Conv23");
    conv23->type(LayerType::CONVOLUTION2D).number(23).shape(128,28,28,512).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv23->push(factory);
    ConvLayerBuilder * conv24 = new ConvLayerBuilder(3,"Conv24");
    conv24->type(LayerType::CONVOLUTION2D).number(24).shape(128,28,28,128).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv24->push(factory);
    ConvLayerBuilder * conv25 = new ConvLayerBuilder(1,"Conv25");
    conv25->type(LayerType::CONVOLUTION2D).number(25).shape(512,28,28,128).deep().prefixAct(ActType::RELU).residual(ActType::NONE,false).context(context_);
    conv25->push(factory);
    GPULayerBuilder * bn26 = new GPULayerBuilder("BN26");
    bn26->type(LayerType::BATCHNORM).number(26).shape(512,28,28,512).deep().context(context_);
    bn26->push(factory);
    ConvLayerBuilder * conv27 = new ConvLayerBuilder(1,"Conv27");
    conv27->type(LayerType::CONVOLUTION2D).number(27).shape(128,28,28,512).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv27->push(factory);
    ConvLayerBuilder * conv28 = new ConvLayerBuilder(3,"Conv28");
    conv28->type(LayerType::CONVOLUTION2D).number(28).shape(128,28,28,128).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv28->push(factory);
    ConvLayerBuilder * conv29 = new ConvLayerBuilder(1,"Conv29");
    conv29->type(LayerType::CONVOLUTION2D).number(29).shape(512,28,28,128).deep().prefixAct(ActType::RELU).residual(ActType::NONE,false).context(context_);
    conv29->push(factory);
    GPULayerBuilder * bn30 = new GPULayerBuilder("BN30");
    bn30->type(LayerType::BATCHNORM).number(30).shape(512,28,28,512).deep().context(context_);
    bn30->push(factory);
    ConvLayerBuilder * conv31 = new ConvLayerBuilder(1,"Conv31");
    conv31->type(LayerType::CONVOLUTION2D).number(31).shape(128,28,28,512).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv31->push(factory);
    ConvLayerBuilder * conv32 = new ConvLayerBuilder(3,"Conv32");
    conv32->type(LayerType::CONVOLUTION2D).number(32).shape(128,28,28,128).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv32->push(factory);
    ConvLayerBuilder * conv33 = new ConvLayerBuilder(1,"Conv33");
    conv33->type(LayerType::CONVOLUTION2D).number(33).shape(512,28,28,128).deep().prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).residual(ActType::NONE,true).context(context_);
    conv33->push(factory);
    ConvLayerBuilder * conv34 = new ConvLayerBuilder(1,"Conv34");
    conv34->type(LayerType::CONVOLUTION2D).number(34).shape(256,28,28,512).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv34->push(factory);
    ConvLayerBuilder * conv35 = new ConvLayerBuilder(1,"Conv35");
    conv35->type(LayerType::CONVOLUTION2D).number(35).shape(1024,28,28,512).downsample(2).deep().prefixAct(ActType::RELU).context(context_);
    conv35->push(factory);
    ConvLayerBuilder * conv36 = new ConvLayerBuilder(3,"Conv36");
    conv36->type(LayerType::CONVOLUTION2D).number(36).shape(256,28,28,256).downsample(2).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv36->push(factory);
    ConvLayerBuilder * conv37 = new ConvLayerBuilder(1,"Conv37");
    conv37->type(LayerType::CONVOLUTION2D).number(37).shape(1024,14,14,256).deep().prefixAct(ActType::RELU).residual(ActType::NONE,false).context(context_);
    conv37->push(factory);
    GPULayerBuilder * bn38 = new GPULayerBuilder("BN38");
    bn38->type(LayerType::BATCHNORM).number(38).shape(1024,14,14,1024).deep().context(context_);
    bn38->push(factory);
    ConvLayerBuilder * conv39 = new ConvLayerBuilder(1,"Conv39");
    conv39->type(LayerType::CONVOLUTION2D).number(39).shape(256,14,14,1024).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv39->push(factory);
    ConvLayerBuilder * conv40 = new ConvLayerBuilder(3,"Conv40");
    conv40->type(LayerType::CONVOLUTION2D).number(40).shape(256,14,14,256).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv40->push(factory);
    ConvLayerBuilder * conv41 = new ConvLayerBuilder(1,"Conv41");
    conv41->type(LayerType::CONVOLUTION2D).number(41).shape(1024,14,14,256).deep().prefixAct(ActType::RELU).residual(ActType::NONE,false).context(context_);
    conv41->push(factory);
    GPULayerBuilder * bn42 = new GPULayerBuilder("BN42");
    bn42->type(LayerType::BATCHNORM).number(42).shape(1024,14,14,1024).deep().context(context_);
    bn42->push(factory);
    ConvLayerBuilder * conv43 = new ConvLayerBuilder(1,"Conv43");
    conv43->type(LayerType::CONVOLUTION2D).number(43).shape(256,14,14,1024).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv43->push(factory);
    ConvLayerBuilder * conv44 = new ConvLayerBuilder(3,"Conv44");
    conv44->type(LayerType::CONVOLUTION2D).number(44).shape(256,14,14,256).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv44->push(factory);
    ConvLayerBuilder * conv45 = new ConvLayerBuilder(1,"Conv45");
    conv45->type(LayerType::CONVOLUTION2D).number(45).shape(1024,14,14,256).deep().prefixAct(ActType::RELU).residual(ActType::NONE,false).context(context_);
    conv45->push(factory);
    GPULayerBuilder * bn46 = new GPULayerBuilder("BN46");
    bn46->type(LayerType::BATCHNORM).number(46).shape(1024,14,14,1024).deep().context(context_);
    bn46->push(factory);
    ConvLayerBuilder * conv47 = new ConvLayerBuilder(1,"Conv47");
    conv47->type(LayerType::CONVOLUTION2D).number(47).shape(256,14,14,1024).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv47->push(factory);
    ConvLayerBuilder * conv48 = new ConvLayerBuilder(3,"Conv48");
    conv48->type(LayerType::CONVOLUTION2D).number(48).shape(256,14,14,256).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv48->push(factory);
    ConvLayerBuilder * conv49 = new ConvLayerBuilder(1,"Conv49");
    conv49->type(LayerType::CONVOLUTION2D).number(49).shape(1024,14,14,256).deep().prefixAct(ActType::RELU).residual(ActType::NONE,false).context(context_);
    conv49->push(factory);
    GPULayerBuilder * bn50 = new GPULayerBuilder("BN50");
    bn50->type(LayerType::BATCHNORM).number(50).shape(1024,14,14,1024).deep().context(context_);
    bn50->push(factory);
    ConvLayerBuilder * conv51 = new ConvLayerBuilder(1,"Conv51");
    conv51->type(LayerType::CONVOLUTION2D).number(51).shape(256,14,14,1024).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv51->push(factory);
    ConvLayerBuilder * conv52 = new ConvLayerBuilder(3,"Conv52");
    conv52->type(LayerType::CONVOLUTION2D).number(52).shape(256,14,14,256).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv52->push(factory);
    ConvLayerBuilder * conv53 = new ConvLayerBuilder(1,"Conv53");
    conv53->type(LayerType::CONVOLUTION2D).number(53).shape(1024,14,14,256).deep().prefixAct(ActType::RELU).residual(ActType::NONE,false).context(context_);
    conv53->push(factory);
    GPULayerBuilder * bn54 = new GPULayerBuilder("BN54");
    bn54->type(LayerType::BATCHNORM).number(54).shape(1024,14,14,1024).deep().context(context_);
    bn54->push(factory);
    ConvLayerBuilder * conv55 = new ConvLayerBuilder(1,"Conv55");
    conv55->type(LayerType::CONVOLUTION2D).number(55).shape(256,14,14,1024).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv55->push(factory);
    ConvLayerBuilder * conv56 = new ConvLayerBuilder(3,"Conv56");
    conv56->type(LayerType::CONVOLUTION2D).number(56).shape(256,14,14,256).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv56->push(factory);
    ConvLayerBuilder * conv57 = new ConvLayerBuilder(1,"Conv57");
    conv57->type(LayerType::CONVOLUTION2D).number(57).shape(1024,14,14,256).deep().prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).residual(ActType::NONE,true).context(context_);
    conv57->push(factory);
    ConvLayerBuilder * conv58 = new ConvLayerBuilder(1,"Conv58");
    conv58->type(LayerType::CONVOLUTION2D).number(58).shape(512,14,14,1024).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv58->push(factory);
    ConvLayerBuilder * conv59 = new ConvLayerBuilder(1,"Conv59");
    conv59->type(LayerType::CONVOLUTION2D).number(59).shape(2048,14,14,1024).downsample(2).deep().prefixAct(ActType::RELU).context(context_);
    conv59->push(factory);
    ConvLayerBuilder * conv60 = new ConvLayerBuilder(3,"Conv60");
    conv60->type(LayerType::CONVOLUTION2D).number(60).shape(512,14,14,512).downsample(2).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv60->push(factory);
    ConvLayerBuilder * conv61 = new ConvLayerBuilder(1,"Conv61");
    conv61->type(LayerType::CONVOLUTION2D).number(61).shape(2048,7,7,512).deep().prefixAct(ActType::RELU).residual(ActType::NONE,false).context(context_);
    conv61->push(factory);
    GPULayerBuilder * bn62 = new GPULayerBuilder("BN62");
    bn62->type(LayerType::BATCHNORM).number(62).shape(2048,7,7,2048).deep().context(context_);
    bn62->push(factory);
    ConvLayerBuilder * conv63 = new ConvLayerBuilder(1,"Conv63");
    conv63->type(LayerType::CONVOLUTION2D).number(63).shape(512,7,7,2048).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv63->push(factory);
    ConvLayerBuilder * conv64 = new ConvLayerBuilder(3,"Conv64");
    conv64->type(LayerType::CONVOLUTION2D).number(64).shape(512,7,7,512).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv64->push(factory);
    ConvLayerBuilder * conv65 = new ConvLayerBuilder(1,"Conv65");
    conv65->type(LayerType::CONVOLUTION2D).number(65).shape(2048,7,7,512).deep().prefixAct(ActType::RELU).residual(ActType::NONE,false).context(context_);
    conv65->push(factory);
    GPULayerBuilder * bn66 = new GPULayerBuilder("BN66");
    bn66->type(LayerType::BATCHNORM).number(66).shape(2048,7,7,2048).deep().context(context_);
    bn66->push(factory);
    ConvLayerBuilder * conv67 = new ConvLayerBuilder(1,"Conv67");
    conv67->type(LayerType::CONVOLUTION2D).number(67).shape(512,7,7,2048).deep().outputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv67->push(factory);
    ConvLayerBuilder * conv68 = new ConvLayerBuilder(3,"Conv68");
    conv68->type(LayerType::CONVOLUTION2D).number(68).shape(512,7,7,512).deep().inputPadding(1).prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).context(context_);
    conv68->push(factory);
    ConvLayerBuilder * conv69 = new ConvLayerBuilder(1,"Conv69");
    conv69->type(LayerType::CONVOLUTION2D).number(69).shape(2048,7,7,512).deep().prefixAct(ActType::RELU).postfixNorm(NormType::BATCHNORM).residual(ActType::NONE,true).context(context_);
    conv69->push(factory);
    PoolLayerBuilder * globavg70 = new PoolLayerBuilder(PoolLayerBuilder::POOL_AVG,"GlobAvg70");
    globavg70->type(LayerType::AVGPOOL2D).number(70).shape(2048,7,7,2048).global().deep().prefixAct(ActType::RELU).context(context_);
    globavg70->push(factory);
    GPULayerBuilder * gemm72 = new GPULayerBuilder("GEMM72");
    gemm72->type(LayerType::GEMM).number(72).shape(1000,1,1,2048).deep().context(context_);
    gemm72->push(factory);
    gpu::UpDownLayerBuilder * down = new gpu::UpDownLayerBuilder(gpu::UpDownLayerBuilder::DOWNLOAD, "download");
    down->shape(1000, 1, 1, 1000).context(context()).deep().number(73);
    down->push(factory);
    return factory->compileLayers();
}


void ResNet50::connectLayers(fyusion::fyusenet::CompiledLayers & layers, fyusion::fyusenet::BufferManager * bufMgr) {
    using namespace fyusion::fyusenet;
    bufMgr->connectLayers(layers[0], layers[2],0);                       // upload -> BN2
    bufMgr->connectLayers(layers[2], layers[3],0);                       // BN2 -> Conv3
    bufMgr->connectLayers(layers[3], layers[4],0);                       // Conv3 -> MaxPool4
    bufMgr->connectLayers(layers[4], layers[5],0);                       // MaxPool4 -> BN5
    bufMgr->connectLayers(layers[5], layers[6],0);                       // BN5 -> Conv6
    bufMgr->connectLayers(layers[5], layers[7],0);                       // BN5 -> Conv7
    bufMgr->connectLayers(layers[6], layers[8],0);                       // Conv6 -> Conv8
    bufMgr->connectLayers(layers[7], layers[9],1);                       // Conv7 -> Conv9 (residual)
    bufMgr->connectLayers(layers[8], layers[9],0);                       // Conv8 -> Conv9
    bufMgr->connectLayers(layers[9], layers[10],0);                      // Conv9 -> BN10
    bufMgr->connectLayers(layers[9], layers[13],1);                      // Conv9 -> Conv13 (residual)
    bufMgr->connectLayers(layers[10], layers[11],0);                     // BN10 -> Conv11
    bufMgr->connectLayers(layers[11], layers[12],0);                     // Conv11 -> Conv12
    bufMgr->connectLayers(layers[12], layers[13],0);                     // Conv12 -> Conv13
    bufMgr->connectLayers(layers[13], layers[14],0);                     // Conv13 -> BN14
    bufMgr->connectLayers(layers[13], layers[17],1);                     // Conv13 -> Conv17 (residual)
    bufMgr->connectLayers(layers[14], layers[15],0);                     // BN14 -> Conv15
    bufMgr->connectLayers(layers[15], layers[16],0);                     // Conv15 -> Conv16
    bufMgr->connectLayers(layers[16], layers[17],0);                     // Conv16 -> Conv17
    bufMgr->connectLayers(layers[17], layers[18],0);                     // Conv17 -> Conv18
    bufMgr->connectLayers(layers[17], layers[19],0);                     // Conv17 -> Conv19
    bufMgr->connectLayers(layers[18], layers[20],0);                     // Conv18 -> Conv20
    bufMgr->connectLayers(layers[19], layers[21],1);                     // Conv19 -> Conv21 (residual)
    bufMgr->connectLayers(layers[20], layers[21],0);                     // Conv20 -> Conv21
    bufMgr->connectLayers(layers[21], layers[22],0);                     // Conv21 -> BN22
    bufMgr->connectLayers(layers[21], layers[25],1);                     // Conv21 -> Conv25 (residual)
    bufMgr->connectLayers(layers[22], layers[23],0);                     // BN22 -> Conv23
    bufMgr->connectLayers(layers[23], layers[24],0);                     // Conv23 -> Conv24
    bufMgr->connectLayers(layers[24], layers[25],0);                     // Conv24 -> Conv25
    bufMgr->connectLayers(layers[25], layers[26],0);                     // Conv25 -> BN26
    bufMgr->connectLayers(layers[25], layers[29],1);                     // Conv25 -> Conv29 (residual)
    bufMgr->connectLayers(layers[26], layers[27],0);                     // BN26 -> Conv27
    bufMgr->connectLayers(layers[27], layers[28],0);                     // Conv27 -> Conv28
    bufMgr->connectLayers(layers[28], layers[29],0);                     // Conv28 -> Conv29
    bufMgr->connectLayers(layers[29], layers[30],0);                     // Conv29 -> BN30
    bufMgr->connectLayers(layers[29], layers[33],1);                     // Conv29 -> Conv33 (residual)
    bufMgr->connectLayers(layers[30], layers[31],0);                     // BN30 -> Conv31
    bufMgr->connectLayers(layers[31], layers[32],0);                     // Conv31 -> Conv32
    bufMgr->connectLayers(layers[32], layers[33],0);                     // Conv32 -> Conv33
    bufMgr->connectLayers(layers[33], layers[34],0);                     // Conv33 -> Conv34
    bufMgr->connectLayers(layers[33], layers[35],0);                     // Conv33 -> Conv35
    bufMgr->connectLayers(layers[34], layers[36],0);                     // Conv34 -> Conv36
    bufMgr->connectLayers(layers[35], layers[37],1);                     // Conv35 -> Conv37 (residual)
    bufMgr->connectLayers(layers[36], layers[37],0);                     // Conv36 -> Conv37
    bufMgr->connectLayers(layers[37], layers[38],0);                     // Conv37 -> BN38
    bufMgr->connectLayers(layers[37], layers[41],1);                     // Conv37 -> Conv41 (residual)
    bufMgr->connectLayers(layers[38], layers[39],0);                     // BN38 -> Conv39
    bufMgr->connectLayers(layers[39], layers[40],0);                     // Conv39 -> Conv40
    bufMgr->connectLayers(layers[40], layers[41],0);                     // Conv40 -> Conv41
    bufMgr->connectLayers(layers[41], layers[42],0);                     // Conv41 -> BN42
    bufMgr->connectLayers(layers[41], layers[45],1);                     // Conv41 -> Conv45 (residual)
    bufMgr->connectLayers(layers[42], layers[43],0);                     // BN42 -> Conv43
    bufMgr->connectLayers(layers[43], layers[44],0);                     // Conv43 -> Conv44
    bufMgr->connectLayers(layers[44], layers[45],0);                     // Conv44 -> Conv45
    bufMgr->connectLayers(layers[45], layers[46],0);                     // Conv45 -> BN46
    bufMgr->connectLayers(layers[45], layers[49],1);                     // Conv45 -> Conv49 (residual)
    bufMgr->connectLayers(layers[46], layers[47],0);                     // BN46 -> Conv47
    bufMgr->connectLayers(layers[47], layers[48],0);                     // Conv47 -> Conv48
    bufMgr->connectLayers(layers[48], layers[49],0);                     // Conv48 -> Conv49
    bufMgr->connectLayers(layers[49], layers[50],0);                     // Conv49 -> BN50
    bufMgr->connectLayers(layers[49], layers[53],1);                     // Conv49 -> Conv53 (residual)
    bufMgr->connectLayers(layers[50], layers[51],0);                     // BN50 -> Conv51
    bufMgr->connectLayers(layers[51], layers[52],0);                     // Conv51 -> Conv52
    bufMgr->connectLayers(layers[52], layers[53],0);                     // Conv52 -> Conv53
    bufMgr->connectLayers(layers[53], layers[54],0);                     // Conv53 -> BN54
    bufMgr->connectLayers(layers[53], layers[57],1);                     // Conv53 -> Conv57 (residual)
    bufMgr->connectLayers(layers[54], layers[55],0);                     // BN54 -> Conv55
    bufMgr->connectLayers(layers[55], layers[56],0);                     // Conv55 -> Conv56
    bufMgr->connectLayers(layers[56], layers[57],0);                     // Conv56 -> Conv57
    bufMgr->connectLayers(layers[57], layers[58],0);                     // Conv57 -> Conv58
    bufMgr->connectLayers(layers[57], layers[59],0);                     // Conv57 -> Conv59
    bufMgr->connectLayers(layers[58], layers[60],0);                     // Conv58 -> Conv60
    bufMgr->connectLayers(layers[59], layers[61],1);                     // Conv59 -> Conv61 (residual)
    bufMgr->connectLayers(layers[60], layers[61],0);                     // Conv60 -> Conv61
    bufMgr->connectLayers(layers[61], layers[62],0);                     // Conv61 -> BN62
    bufMgr->connectLayers(layers[61], layers[65],1);                     // Conv61 -> Conv65 (residual)
    bufMgr->connectLayers(layers[62], layers[63],0);                     // BN62 -> Conv63
    bufMgr->connectLayers(layers[63], layers[64],0);                     // Conv63 -> Conv64
    bufMgr->connectLayers(layers[64], layers[65],0);                     // Conv64 -> Conv65
    bufMgr->connectLayers(layers[65], layers[66],0);                     // Conv65 -> BN66
    bufMgr->connectLayers(layers[65], layers[69],1);                     // Conv65 -> Conv69 (residual)
    bufMgr->connectLayers(layers[66], layers[67],0);                     // BN66 -> Conv67
    bufMgr->connectLayers(layers[67], layers[68],0);                     // Conv67 -> Conv68
    bufMgr->connectLayers(layers[68], layers[69],0);                     // Conv68 -> Conv69
    bufMgr->connectLayers(layers[69], layers[70],0);                     // Conv69 -> GlobAvg70
    bufMgr->connectLayers(layers[70], layers[72],0);                     // GlobAvg70 -> GEMM72
    bufMgr->connectLayers(layers[72], layers[73],0);                     // GEMM -> download
    bufMgr->createCPUOutput(layers[73], true);
}


/**
 * @copydoc NeuralNetwork::initializeWeights
 */
void ResNet50::initializeWeights(fyusion::fyusenet::CompiledLayers & layers) {
    using namespace fyusion::fyusenet;
    assert(wbData_);
    for (auto it = layers.begin(); it != layers.end(); ++it) {
        ConvLayerInterface * conv = dynamic_cast<ConvLayerInterface *>(it.second);
        if (conv) {
            conv->loadWeightsAndBiases(wbData_, weightOffsets_.at(it.second->getNumber()));
        } else {
            BatchNormInterface * bn = dynamic_cast<BatchNormInterface *>(it.second);
            if (bn) {
                bn->loadScaleAndBias(wbData_, weightOffsets_.at(it.second->getNumber()));
            }
        }
    }
}


void ResNet50::initializeWeightOffsets() {
    weightOffsets_[2] = 0 / sizeof(float);
    weightSizes_[2] = 24 / sizeof(float);
    weightOffsets_[3] = 24 / sizeof(float);
    weightSizes_[3] = 38400 / sizeof(float);
    weightOffsets_[5] = 38424 / sizeof(float);
    weightSizes_[5] = 512 / sizeof(float);
    weightOffsets_[6] = 38936 / sizeof(float);
    weightSizes_[6] = 17152 / sizeof(float);
    weightOffsets_[8] = 56088 / sizeof(float);
    weightSizes_[8] = 148224 / sizeof(float);
    weightOffsets_[9] = 204312 / sizeof(float);
    weightSizes_[9] = 66560 / sizeof(float);
    weightOffsets_[7] = 270872 / sizeof(float);
    weightSizes_[7] = 66560 / sizeof(float);
    weightOffsets_[10] = 337432 / sizeof(float);
    weightSizes_[10] = 2048 / sizeof(float);
    weightOffsets_[11] = 339480 / sizeof(float);
    weightSizes_[11] = 66304 / sizeof(float);
    weightOffsets_[12] = 405784 / sizeof(float);
    weightSizes_[12] = 148224 / sizeof(float);
    weightOffsets_[13] = 554008 / sizeof(float);
    weightSizes_[13] = 66560 / sizeof(float);
    weightOffsets_[14] = 620568 / sizeof(float);
    weightSizes_[14] = 2048 / sizeof(float);
    weightOffsets_[15] = 622616 / sizeof(float);
    weightSizes_[15] = 66304 / sizeof(float);
    weightOffsets_[16] = 688920 / sizeof(float);
    weightSizes_[16] = 148224 / sizeof(float);
    weightOffsets_[17] = 837144 / sizeof(float);
    weightSizes_[17] = 68608 / sizeof(float);
    weightOffsets_[18] = 905752 / sizeof(float);
    weightSizes_[18] = 132608 / sizeof(float);
    weightOffsets_[20] = 1038360 / sizeof(float);
    weightSizes_[20] = 591360 / sizeof(float);
    weightOffsets_[21] = 1629720 / sizeof(float);
    weightSizes_[21] = 264192 / sizeof(float);
    weightOffsets_[19] = 1893912 / sizeof(float);
    weightSizes_[19] = 526336 / sizeof(float);
    weightOffsets_[22] = 2420248 / sizeof(float);
    weightSizes_[22] = 4096 / sizeof(float);
    weightOffsets_[23] = 2424344 / sizeof(float);
    weightSizes_[23] = 263680 / sizeof(float);
    weightOffsets_[24] = 2688024 / sizeof(float);
    weightSizes_[24] = 591360 / sizeof(float);
    weightOffsets_[25] = 3279384 / sizeof(float);
    weightSizes_[25] = 264192 / sizeof(float);
    weightOffsets_[26] = 3543576 / sizeof(float);
    weightSizes_[26] = 4096 / sizeof(float);
    weightOffsets_[27] = 3547672 / sizeof(float);
    weightSizes_[27] = 263680 / sizeof(float);
    weightOffsets_[28] = 3811352 / sizeof(float);
    weightSizes_[28] = 591360 / sizeof(float);
    weightOffsets_[29] = 4402712 / sizeof(float);
    weightSizes_[29] = 264192 / sizeof(float);
    weightOffsets_[30] = 4666904 / sizeof(float);
    weightSizes_[30] = 4096 / sizeof(float);
    weightOffsets_[31] = 4671000 / sizeof(float);
    weightSizes_[31] = 263680 / sizeof(float);
    weightOffsets_[32] = 4934680 / sizeof(float);
    weightSizes_[32] = 591360 / sizeof(float);
    weightOffsets_[33] = 5526040 / sizeof(float);
    weightSizes_[33] = 268288 / sizeof(float);
    weightOffsets_[34] = 5794328 / sizeof(float);
    weightSizes_[34] = 527360 / sizeof(float);
    weightOffsets_[36] = 6321688 / sizeof(float);
    weightSizes_[36] = 2362368 / sizeof(float);
    weightOffsets_[37] = 8684056 / sizeof(float);
    weightSizes_[37] = 1052672 / sizeof(float);
    weightOffsets_[35] = 9736728 / sizeof(float);
    weightSizes_[35] = 2101248 / sizeof(float);
    weightOffsets_[38] = 11837976 / sizeof(float);
    weightSizes_[38] = 8192 / sizeof(float);
    weightOffsets_[39] = 11846168 / sizeof(float);
    weightSizes_[39] = 1051648 / sizeof(float);
    weightOffsets_[40] = 12897816 / sizeof(float);
    weightSizes_[40] = 2362368 / sizeof(float);
    weightOffsets_[41] = 15260184 / sizeof(float);
    weightSizes_[41] = 1052672 / sizeof(float);
    weightOffsets_[42] = 16312856 / sizeof(float);
    weightSizes_[42] = 8192 / sizeof(float);
    weightOffsets_[43] = 16321048 / sizeof(float);
    weightSizes_[43] = 1051648 / sizeof(float);
    weightOffsets_[44] = 17372696 / sizeof(float);
    weightSizes_[44] = 2362368 / sizeof(float);
    weightOffsets_[45] = 19735064 / sizeof(float);
    weightSizes_[45] = 1052672 / sizeof(float);
    weightOffsets_[46] = 20787736 / sizeof(float);
    weightSizes_[46] = 8192 / sizeof(float);
    weightOffsets_[47] = 20795928 / sizeof(float);
    weightSizes_[47] = 1051648 / sizeof(float);
    weightOffsets_[48] = 21847576 / sizeof(float);
    weightSizes_[48] = 2362368 / sizeof(float);
    weightOffsets_[49] = 24209944 / sizeof(float);
    weightSizes_[49] = 1052672 / sizeof(float);
    weightOffsets_[50] = 25262616 / sizeof(float);
    weightSizes_[50] = 8192 / sizeof(float);
    weightOffsets_[51] = 25270808 / sizeof(float);
    weightSizes_[51] = 1051648 / sizeof(float);
    weightOffsets_[52] = 26322456 / sizeof(float);
    weightSizes_[52] = 2362368 / sizeof(float);
    weightOffsets_[53] = 28684824 / sizeof(float);
    weightSizes_[53] = 1052672 / sizeof(float);
    weightOffsets_[54] = 29737496 / sizeof(float);
    weightSizes_[54] = 8192 / sizeof(float);
    weightOffsets_[55] = 29745688 / sizeof(float);
    weightSizes_[55] = 1051648 / sizeof(float);
    weightOffsets_[56] = 30797336 / sizeof(float);
    weightSizes_[56] = 2362368 / sizeof(float);
    weightOffsets_[57] = 33159704 / sizeof(float);
    weightSizes_[57] = 1060864 / sizeof(float);
    weightOffsets_[58] = 34220568 / sizeof(float);
    weightSizes_[58] = 2103296 / sizeof(float);
    weightOffsets_[60] = 36323864 / sizeof(float);
    weightSizes_[60] = 9443328 / sizeof(float);
    weightOffsets_[61] = 45767192 / sizeof(float);
    weightSizes_[61] = 4202496 / sizeof(float);
    weightOffsets_[59] = 49969688 / sizeof(float);
    weightSizes_[59] = 8396800 / sizeof(float);
    weightOffsets_[62] = 58366488 / sizeof(float);
    weightSizes_[62] = 16384 / sizeof(float);
    weightOffsets_[63] = 58382872 / sizeof(float);
    weightSizes_[63] = 4200448 / sizeof(float);
    weightOffsets_[64] = 62583320 / sizeof(float);
    weightSizes_[64] = 9443328 / sizeof(float);
    weightOffsets_[65] = 72026648 / sizeof(float);
    weightSizes_[65] = 4202496 / sizeof(float);
    weightOffsets_[66] = 76229144 / sizeof(float);
    weightSizes_[66] = 16384 / sizeof(float);
    weightOffsets_[67] = 76245528 / sizeof(float);
    weightSizes_[67] = 4200448 / sizeof(float);
    weightOffsets_[68] = 80445976 / sizeof(float);
    weightSizes_[68] = 9443328 / sizeof(float);
    weightOffsets_[69] = 89889304 / sizeof(float);
    weightSizes_[69] = 4218880 / sizeof(float);
    weightOffsets_[72] = 94108184 / sizeof(float);
    weightSizes_[72] = 8196000 / sizeof(float);
    totalWeightBytes_ = 102304184;
}



