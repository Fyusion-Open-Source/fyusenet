//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2022
//--------------------------------------------------------------------------------------------------
// Style-Transfer Network w/ 3x3 Convolutions
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstring>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "stylenet3x3.h"

//-------------------------------------- Global Variables ------------------------------------------

//-------------------------------------- Local Definitions -----------------------------------------



/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Construct style-transfer network
 *
 * @param width Width of image data to process
 * @param height Height of image data to process
 * @param upload Create an upload layer in the network (will be named "upload")
 * @param download Create a download layer in the network (will be named "download")
 * @param ctx Link to GL context that the network should use
 */
StyleNet3x3::StyleNet3x3(int width, int height, bool upload, bool download, const fyusion::fyusenet::GfxContextLink& ctx) :
    StyleNetBase(width, height, upload, download, ctx) {
    weightOffsets_[CONV1] = 0;
    weightOffsets_[CONV2] = 336;
    weightOffsets_[CONV3] = 2516;
    weightOffsets_[RES1_1] = 19475;
    weightOffsets_[RES1_2] = 33915;
    weightOffsets_[RES2_1] = 48355;
    weightOffsets_[RES2_2] = 62795;
    weightOffsets_[DECONV1] = 9756;
    weightOffsets_[DECONV2] = 16976;
    weightOffsets_[DECONV3] = 19148;
    wbData_ = new float[STYLENET_SIZE];
    memset(wbData_, 0, STYLENET_SIZE * sizeof(float));
    for (int i=0; i < ASYNC_BUFFERS; i++) inBuffers_[i] = nullptr;
}


/**
 * @brief Destructor
 *
 * Deallocates resources
 */
StyleNet3x3::~StyleNet3x3() {
    delete [] wbData_;
    wbData_ = nullptr;
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
void StyleNet3x3::loadWeightsAndBiases(float *weightsAndBiases, size_t size) {
    assert(size == STYLENET_SIZE);
    assert(wbData_);
    memcpy(wbData_, weightsAndBiases, STYLENET_SIZE*sizeof(float));
    if (setup_) {
        assertContext();
        initializeWeights(engine_->getLayers());
    }
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc NeuralNetwork::initializeWeights
 */
void StyleNet3x3::initializeWeights(fyusion::fyusenet::CompiledLayers & layers) {
    using namespace fyusion::fyusenet;
    assert(wbData_);
    for (auto it = layers.begin(); it != layers.end(); ++it) {
        ConvLayerInterface * conv = dynamic_cast<ConvLayerInterface *>(it.second);
        if (conv) {
            conv->loadWeightsAndBiases(wbData_, weightOffsets_.at(it.second->getNumber()));
        }
    }
}


/**
 * @copydoc NeuralNetwork::buildLayers
 */
fyusion::fyusenet::CompiledLayers StyleNet3x3::buildLayers() {
    using namespace fyusion::fyusenet;
    std::shared_ptr<LayerFactory> factory = getLayerFactory();
    if (upload_) {
        gpu::UpDownLayerBuilder * up = new gpu::UpDownLayerBuilder(gpu::UpDownLayerBuilder::UPLOAD, "upload");
        up->shape(3, width_, height_, 3).context(context()).number(UPLOAD);
#ifdef FYUSENET_MULTITHREADING
        if (async_) up->async().callback(std::bind(&StyleNet3x3::internalULCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
#endif
        up->push(factory);
    } else if (oesInput_) {
#ifdef FYUSENET_USE_EGL
        gpu::GPULayerBuilder * oes = new gpu::GPULayerBuilder("oes");
        oes->shape(3,width_,height_,3).type(LayerType::OESCONV).context(context()).number(UNPACK);
        oes->push(factory);
#else
        assert(false);
#endif
    }

    gpu::ConvLayerBuilder * conv1 = new gpu::ConvLayerBuilder(3,"conv1");
    conv1->shape(12,width_,height_,3).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).context(context()).number(CONV1);
    conv1->push(factory);

    gpu::ConvLayerBuilder * conv2 = new gpu::ConvLayerBuilder(3,"conv2");
    conv2->shape(20,width_,height_,12).type(LayerType::CONVOLUTION2D).downsample(2).prefixAct(ActType::RELU).context(context()).number(CONV2);
    conv2->push(factory);

    gpu::ConvLayerBuilder * conv3 = new gpu::ConvLayerBuilder(3,"conv3");
    conv3->shape(40,width_/2,height_/2,20).type(LayerType::CONVOLUTION2D).downsample(2).prefixAct(ActType::RELU).context(context()).number(CONV3);
    conv3->push(factory);

    gpu::ConvLayerBuilder * res11 = new gpu::ConvLayerBuilder(3,"res1_1");
    res11->shape(40,width_/4,height_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).context(context()).number(RES1_1);
    res11->push(factory);

    gpu::ConvLayerBuilder * res12 = new gpu::ConvLayerBuilder(3,"res1_2");
    res12->shape(40,width_/4,height_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).residual(ActType::RELU).context(context()).number(RES1_2);
    res12->push(factory);

    gpu::ConvLayerBuilder * res21 = new gpu::ConvLayerBuilder(3,"res2_1");
    res21->shape(40,width_/4,height_/4,40).type(LayerType::CONVOLUTION2D).context(context()).number(RES2_1);
    res21->push(factory);

    gpu::ConvLayerBuilder * res22 = new gpu::ConvLayerBuilder(3,"res2_2");
    res22->shape(40,width_/4,height_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).residual().context(context()).number(RES2_2);
    res22->push(factory);

    gpu::ConvLayerBuilder * deconv1 = new gpu::ConvLayerBuilder(3,"deconv1");
    deconv1->shape(20,width_/4,height_/4,40).type(LayerType::FRACCONVOLUTION2D).downsample(2).sourceStep(0.5f).context(context()).number(DECONV1);
    deconv1->push(factory);

    gpu::ConvLayerBuilder * deconv2 = new gpu::ConvLayerBuilder(3,"deconv2");
    deconv2->shape(12,width_/4,height_/4,20).type(LayerType::FRACCONVOLUTION2D).sourceStep(0.25f).downsample(2).prefixAct(ActType::RELU).context(context()).number(DECONV2);
    deconv2->push(factory);

    gpu::ConvLayerBuilder * deconv3 = new gpu::ConvLayerBuilder(3,"deconv3");
    deconv3->shape(3,width_/2,height_/2,12).type(LayerType::FRACCONVOLUTION2D).sourceStep(0.5f).prefixAct(ActType::RELU).context(context()).number(DECONV3);
    deconv3->push(factory);

    gpu::GPULayerBuilder * sigmoid = new gpu::GPULayerBuilder("sigmoid");
    sigmoid->shape(3,width_,height_,3).type(LayerType::SIGMOID).context(context()).number(SIGMOID);
    sigmoid->push(factory);

    if (download_) {
        gpu::UpDownLayerBuilder * down = new gpu::UpDownLayerBuilder(gpu::UpDownLayerBuilder::DOWNLOAD, "download");
        down->shape(4, width_, height_, 4).context(context()).number(DOWNLOAD);
#ifdef FYUSENET_MULTITHREADING
        if (async_) down->async().callback(std::bind(&StyleNet3x3::internalDLCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
#endif
        down->push(factory);
    }

    return factory->compileLayers();
}


/**
 * @copydoc NeuralNetwork::connectLayers
 */
void StyleNet3x3::connectLayers(fyusion::fyusenet::CompiledLayers& layers, fyusion::fyusenet::BufferManager * buffers) {
    using namespace fyusion::fyusenet;
    if (oesInput_) buffers->connectLayers(layers[UNPACK], layers[CONV1], 0);
    else if (upload_) buffers->connectLayers(layers[UPLOAD], layers[CONV1], 0);
    buffers->connectLayers(layers[CONV1], layers[CONV2], 0);
    buffers->connectLayers(layers[CONV2], layers[CONV3], 0);
    buffers->connectLayers(layers[CONV3], layers[RES1_1], 0);
    buffers->connectLayers(layers[CONV3], layers[RES1_2], 1);
    buffers->connectLayers(layers[RES1_1], layers[RES1_2], 0);
    buffers->connectLayers(layers[RES1_2], layers[RES2_1], 0);
    buffers->connectLayers(layers[RES1_2], layers[RES2_2], 1);
    buffers->connectLayers(layers[RES2_1], layers[RES2_2], 0);
    buffers->connectLayers(layers[RES2_2], layers[DECONV1], 0);
    buffers->connectLayers(layers[DECONV1], layers[DECONV2], 0);
    buffers->connectLayers(layers[DECONV2], layers[DECONV3], 0);
    buffers->connectLayers(layers[DECONV3], layers[SIGMOID], 0);
    if (download_) {
        buffers->connectLayers(layers[SIGMOID], layers[DOWNLOAD], 0);
#ifdef FYUSENET_MULTITHREADING
        if (async_) {
            gpu::DownloadLayer * down = (gpu::DownloadLayer *)layers[DOWNLOAD];
            assert(down);
            std::vector<BufferSpec> specs = down->getRequiredOutputBuffers();
            assert(specs.size() == 1);
            CPUBufferShape shape(specs[0].width_, specs[0].height_, specs[0].channels_, 0,
                                 CPUBufferShape::type::FLOAT32);
            asyncDLBuffers_[0] = shape.createBuffer();
            asyncDLBuffers_[1] = shape.createBuffer();
            down->addOutputBuffer(asyncDLBuffers_[0]);
            down->addOutputConnection(0, nullptr, 0);
        } else {
            buffers->createCPUOutput(layers[DOWNLOAD], true);
        }
#else
        buffers->createCPUOutput(layers[DOWNLOAD], true);
#endif
    }
    else {
        buffers->createGPUOutput((gpu::GPULayerBase *)layers[SIGMOID]);
    }
}

