//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2022
//--------------------------------------------------------------------------------------------------
// Style-Transfer Network w/ 9x9 Convolutions
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <memory>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "stylenet9x9.h"

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
StyleNet9x9::StyleNet9x9(int width, int height, bool upload, bool download, const fyusion::fyusenet::GfxContextLink& ctx) :
    StyleNetBase(width, height, upload, download, ctx) {
    weightOffsets_[CONV1] = 0;
    weightOffsets_[CONV2] = 2928;
    weightOffsets_[CONV3] = 5108;
    weightOffsets_[RES1_1] = 24659;
    weightOffsets_[RES1_2] = 39099;
    weightOffsets_[RES2_1] = 53539;
    weightOffsets_[RES2_2] = 67979;
    weightOffsets_[RES3_1] = 82419;
    weightOffsets_[RES3_2] = 96859;
    weightOffsets_[RES4_1] = 111299;
    weightOffsets_[RES4_2] = 125739;
    weightOffsets_[RES5_1] = 140179;
    weightOffsets_[RES5_2] = 154619;
    weightOffsets_[DECONV1] = 12348;
    weightOffsets_[DECONV2] = 19568;
    weightOffsets_[DECONV3] = 21740;
    wbData_ = new float[STYLENET_SIZE];
    memset(wbData_, 0, STYLENET_SIZE * sizeof(float));
    for (int i=0; i < ASYNC_BUFFERS; i++) inBuffers_[i] = nullptr;
}


/**
 * @brief Destructor
 *
 * Deallocates resources
 */
StyleNet9x9::~StyleNet9x9() {
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
void StyleNet9x9::loadWeightsAndBiases(float *weightsAndBiases, size_t size) {
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
void StyleNet9x9::initializeWeights(fyusion::fyusenet::CompiledLayers & layers) {
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
fyusion::fyusenet::CompiledLayers StyleNet9x9::buildLayers() {
    using namespace fyusion::fyusenet;
    std::shared_ptr<LayerFactory> factory = getLayerFactory();
    if (upload_) {
        gpu::UpDownLayerBuilder * up = new gpu::UpDownLayerBuilder(gpu::UpDownLayerBuilder::UPLOAD, "upload");
        up->shape(3, height_, width_, 3).context(context()).number(UPLOAD);
#ifdef FYUSENET_MULTITHREADING
        if (async_) up->async().callback(std::bind(&StyleNet9x9::internalULCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
#endif
        up->push(factory);
    } else if (oesInput_) {
#ifdef FYUSENET_USE_EGL
        gpu::GPULayerBuilder * oes = new gpu::GPULayerBuilder("oes");
        oes->shape(3, height_, width_, 3).type(LayerType::OESCONV).context(context()).number(UNPACK);
	oes->push(factory);
#else
        assert(false);
#endif
    }

    gpu::ConvLayerBuilder * conv1 = new gpu::ConvLayerBuilder(9,"conv1");
    conv1->shape(12,height_, width_,3).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).context(context()).number(CONV1);
    conv1->push(factory);

    gpu::ConvLayerBuilder * conv2 = new gpu::ConvLayerBuilder(3,"conv2");
    conv2->shape(20,height_, width_,12).type(LayerType::CONVOLUTION2D).downsample(2).prefixAct(ActType::RELU).context(context()).number(CONV2);
    conv2->push(factory);

    gpu::ConvLayerBuilder * conv3 = new gpu::ConvLayerBuilder(3,"conv3");
    conv3->shape(40,height_/2, width_/2,20).type(LayerType::CONVOLUTION2D).downsample(2).prefixAct(ActType::RELU).context(context()).number(CONV3);
    conv3->push(factory);

    gpu::ConvLayerBuilder * res11 = new gpu::ConvLayerBuilder(3,"res1_1");
    res11->shape(40,height_/4, width_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).context(context()).number(RES1_1);
    res11->push(factory);

    gpu::ConvLayerBuilder * res12 = new gpu::ConvLayerBuilder(3,"res1_2");
    res12->shape(40,height_/4, width_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).residual(ActType::RELU).context(context()).number(RES1_2);
    res12->push(factory);

    gpu::ConvLayerBuilder * res21 = new gpu::ConvLayerBuilder(3,"res2_1");
    res21->shape(40,height_/4, width_/4,40).type(LayerType::CONVOLUTION2D).context(context()).number(RES2_1);
    res21->push(factory);

    gpu::ConvLayerBuilder * res22 = new gpu::ConvLayerBuilder(3,"res2_2");
    res22->shape(40,height_/4, width_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).residual().context(context()).number(RES2_2);
    res22->push(factory);

    gpu::ConvLayerBuilder * res31 = new gpu::ConvLayerBuilder(3,"res3_1");
    res31->shape(40,height_/4, width_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).context(context()).number(RES3_1);
    res31->push(factory);

    gpu::ConvLayerBuilder * res32 = new gpu::ConvLayerBuilder(3,"res3_2");
    res32->shape(40,height_/4, width_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).residual().context(context()).number(RES3_2);
    res32->push(factory);

    gpu::ConvLayerBuilder * res41 = new gpu::ConvLayerBuilder(3,"res4_1");
    res41->shape(40,height_/4, width_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).context(context()).number(RES4_1);
    res41->push(factory);

    gpu::ConvLayerBuilder * res42 = new gpu::ConvLayerBuilder(3,"res4_2");
    res42->shape(40,height_/4, width_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).residual().context(context()).number(RES4_2);
    res42->push(factory);

    gpu::ConvLayerBuilder * res51 = new gpu::ConvLayerBuilder(3,"res5_1");
    res51->shape(40,height_/4, width_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).context(context()).number(RES5_1);
    res51->push(factory);

    gpu::ConvLayerBuilder * res52 = new gpu::ConvLayerBuilder(3,"res5_2");
    res52->shape(40,height_/4, width_/4,40).type(LayerType::CONVOLUTION2D).prefixAct(ActType::RELU).residual().context(context()).number(RES5_2);
    res52->push(factory);

    gpu::ConvLayerBuilder * deconv1 = new gpu::ConvLayerBuilder(3,"deconv1");
    deconv1->shape(20,height_/4, width_/4,40).type(LayerType::FRACCONVOLUTION2D).downsample(2).sourceStep(0.5f).context(context()).number(DECONV1);
    deconv1->push(factory);

    gpu::ConvLayerBuilder * deconv2 = new gpu::ConvLayerBuilder(3,"deconv2");
    deconv2->shape(12,height_/4, width_/4,20).type(LayerType::FRACCONVOLUTION2D).sourceStep(0.25f).downsample(2).prefixAct(ActType::RELU).context(context()).number(DECONV2);
    deconv2->push(factory);

    gpu::ConvLayerBuilder * deconv3 = new gpu::ConvLayerBuilder(9,"deconv3");
    deconv3->shape(3,height_/2, width_/2,12).type(LayerType::FRACCONVOLUTION2D).sourceStep(0.5f).prefixAct(ActType::RELU).context(context()).number(DECONV3);
    deconv3->push(factory);

    gpu::GPULayerBuilder * sigmoid = new gpu::GPULayerBuilder("sigmoid");
    sigmoid->shape(3,height_, width_,3).type(LayerType::SIGMOID).context(context()).number(SIGMOID);
    sigmoid->push(factory);

    if (download_) {
        gpu::UpDownLayerBuilder * down = new gpu::UpDownLayerBuilder(gpu::UpDownLayerBuilder::DOWNLOAD, "download");
        down->shape(4, height_, width_, 4).context(context()).number(DOWNLOAD);
#ifdef FYUSENET_MULTITHREADING
        if (async_) down->async().callback(std::bind(&StyleNet9x9::internalDLCallback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
#endif
        down->push(factory);
    }

    return factory->compileLayers();
}


/**
 * @copydoc NeuralNetwork::connectLayers
 */
void StyleNet9x9::connectLayers(fyusion::fyusenet::CompiledLayers& layers, fyusion::fyusenet::BufferManager * buffers) {
    using namespace fyusion::fyusenet;
    if (oesInput_) buffers->connectLayers(layers[UNPACK], layers[CONV1], 0);
    else if (upload_) buffers->connectLayers(layers[UPLOAD], layers[CONV1], 0);
    buffers->connectLayers(layers[CONV1],layers[CONV2],0);
    buffers->connectLayers(layers[CONV2],layers[CONV3],0);
    buffers->connectLayers(layers[CONV3],layers[RES1_1],0);
    buffers->connectLayers(layers[CONV3],layers[RES1_2],1);
    buffers->connectLayers(layers[RES1_1],layers[RES1_2],0);
    buffers->connectLayers(layers[RES1_2],layers[RES2_1],0);
    buffers->connectLayers(layers[RES1_2],layers[RES2_2],1);
    buffers->connectLayers(layers[RES2_1],layers[RES2_2],0);
    buffers->connectLayers(layers[RES2_2],layers[RES3_1],0);
    buffers->connectLayers(layers[RES2_2],layers[RES3_2],1);
    buffers->connectLayers(layers[RES3_1],layers[RES3_2],0);
    buffers->connectLayers(layers[RES3_2],layers[RES4_1],0);
    buffers->connectLayers(layers[RES3_2],layers[RES4_2],1);
    buffers->connectLayers(layers[RES4_1],layers[RES4_2],0);
    buffers->connectLayers(layers[RES4_2],layers[RES5_1],0);
    buffers->connectLayers(layers[RES4_2],layers[RES5_2],1);
    buffers->connectLayers(layers[RES5_1],layers[RES5_2],0);
    buffers->connectLayers(layers[RES5_2],layers[DECONV1],0);
    buffers->connectLayers(layers[DECONV1],layers[DECONV2],0);
    buffers->connectLayers(layers[DECONV2],layers[DECONV3],0);
    buffers->connectLayers(layers[DECONV3],layers[SIGMOID],0);
    if (download_) {
        buffers->connectLayers(layers[SIGMOID], layers[DOWNLOAD], 0);
#ifdef FYUSENET_MULTITHREADING
        if (async_) {
            gpu::DownloadLayer * down = (gpu::DownloadLayer *)layers[DOWNLOAD];
            assert(down);
            std::vector<BufferSpec> specs = down->getRequiredOutputBuffers();
            assert(specs.size() == 1);
            CPUBufferShape shape(specs[0].height_, specs[0].width_, specs[0].channels_, 0,
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
