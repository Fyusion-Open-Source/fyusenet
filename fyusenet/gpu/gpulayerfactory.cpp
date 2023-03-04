//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// GPU Layer Factory
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/logging.h"
#include "gpulayerfactory.h"
#include "convlayerbase.h"
#include "deep/deepextractimgpatches.h"
#include "deep/deepmaxpoollayer.h"
#include "deep/deepavgpoollayer.h"
#include "deep/deepconcatlayer.h"
#include "deep/deepscalelayer.h"
#include "deep/deeptransconvlayer2x2.h"
#include "deep/deeptransconvlayer3x3.h"
#include "deep/deepdownloadlayer.h"
#include "deep/deepargmaxlayer.h"
#include "deep/deepconvlayer1x1.h"
#include "deep/deepgemmlayer.h"
#include "deep/deepconvlayerNxN.h"
#include "deep/deepdwconvlayer3x3.h"
#include "deep/deepsigmoidlayer.h"
#include "deep/deeptanhlayer.h"
#include "deep/deep_singleton_arithlayer.h"
#include "deep/deepcastlayer.h"
#include "deep/deeptransposelayer.h"
#include "deep/deepbatchnormlayer.h"
#ifdef FYUSENET_USE_EGL
#include "oesconverter.h"
#endif
#include "downloadlayer.h"
#include "uploadlayer.h"
#include "rgb2bgrlayer.h"
#include "nonmaxsuppression2d.h"
#include "blurlayer.h"
#include "scalelayer.h"
#include "avgpoollayer.h"
#include "maxpoollayer.h"
#include "rgb2bgrlayer.h"
#include "batchnormlayer.h"
#include "sigmoidlayer.h"
#include "tanhlayer.h"
#include "castlayer.h"
#include "singleton_arithlayer.h"
#include "deep2shallow.h"
#include "shallow2deep.h"
#include "concatlayer.h"
#include "addsublayer.h"
#include "vanilla/convlayer1x1_vanilla.h"
#include "vanilla/convlayerNxN_vanilla.h"
#include "vanilla/convlayer_dw_3x3_vanilla.h"
#include "vanilla/transconvlayer2x2_vanilla.h"
#include "vanilla/transconvlayer3x3_vanilla.h"
#include "vanilla/fractionalconvlayerNxN_vanilla.h"
#include "vanilla/transconvlayer2x2_vanilla.h"
#include "vanilla/transconvlayer3x3_vanilla.h"
#include "../gl/glinfo.h"

//-------------------------------------- Global Variables ------------------------------------------


namespace fyusion {
namespace fyusenet {
namespace gpu {
//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @brief Get name of the factory backend
 *
 * @return String with backend name
 */
std::string GPULayerFactoryBackend::getName() const {
    static std::string name("Vanilla [GPU]");
    return name;
}


/**
 * @brief Generate network layer
 *
 * @param type Layer type to generate
 *
 * @param builder LayerBuilder object that contains the parameters for the layer to be built
 *
 * @param layerNumber Layer number (must be unique within a network)
 *
 * @return Raw pointer to generated layer
 *
 * @throws FynException in case there was a problem with the layer generation or the layer is
 *         unsupported by this backend
 *
 * This function checks if the layer type is supported and then dispatches the layer creation to
 * the appropriate subroutine which will instantiate a new layer. The returned layer will be freshly
 * constructed but not initialized (no setup invoked).
 */
fyusenet::LayerBase * GPULayerFactoryBackend::createLayer(LayerType type,LayerBuilder * builder, int layerNumber) {
    if (!builder) THROW_EXCEPTION_ARGS(FynException,"No builder supplied to layer factory line");
    switch (type) {
        case LayerType::OESCONV:
#ifdef FYUSENET_USE_EGL
            return createOESLayer((GPULayerBuilder *)builder,layerNumber);
#else
            THROW_EXCEPTION_ARGS(FynException,"Only works on OpenGL/ES");
#endif
        case LayerType::ADD:
            // intentional fallthrough
        case LayerType::SUB:
            return (fyusenet::LayerBase *)createAddSubLayer((GPULayerBuilder *)builder,layerNumber);
        case LayerType::PADDING2D:
            return (fyusenet::LayerBase *)createPaddingLayer((GPULayerBuilder *)builder,layerNumber);
        case LayerType::CONVOLUTION2D:
            return (fyusenet::LayerBase *)createConvLayer((ConvLayerBuilder *)builder,layerNumber);
        case LayerType::TRANSCONVOLUTION2D:
            return (fyusenet::LayerBase *)createTransConvLayer((ConvLayerBuilder *)builder,layerNumber);
        case LayerType::FRACCONVOLUTION2D:
            return (fyusenet::LayerBase *)createFracConvLayer((ConvLayerBuilder *)builder,layerNumber);
        case LayerType::RELU:
            // we emulate the ReLU layer with a scaling layer
            return (fyusenet::LayerBase *)createScaleLayer((ScaleLayerBuilder *)builder,layerNumber);
        case LayerType::CLIP:
            // we emulate the clip layer with a scaling layer
            return (fyusenet::LayerBase *)createScaleLayer((ScaleLayerBuilder *)builder,layerNumber);
        case LayerType::SCALE2D:
            return (fyusenet::LayerBase *)createScaleLayer((ScaleLayerBuilder *)builder,layerNumber);
        case LayerType::CONCAT:
            return (fyusenet::LayerBase *)createConcatLayer((ConcatLayerBuilder *)builder,layerNumber);
        case LayerType::SHALLOW2DEEP:
            return (fyusenet::LayerBase *)createS2DLayer((GPULayerBuilder *)builder,layerNumber);
        case LayerType::DEEP2SHALLOW:
            return (fyusenet::LayerBase *)createD2SLayer((GPULayerBuilder *)builder,layerNumber);
        case LayerType::MAXPOOL2D:
            return (fyusenet::LayerBase *)createMaxPoolLayer((PoolLayerBuilder *)builder,layerNumber);
        case LayerType::AVGPOOL2D:
            return (fyusenet::LayerBase *)createAvgPoolLayer((PoolLayerBuilder *)builder,layerNumber);
        case LayerType::ARGMAX:
            return (fyusenet::LayerBase *)createArgMaxLayer((ArgMaxLayerBuilder *)builder,layerNumber);
        case LayerType::CUSTOM:
            return (fyusenet::LayerBase *)createCustomLayer((CustomLayerBuilder *)builder,layerNumber);
        case LayerType::DOWNLOAD:
            return (fyusenet::LayerBase *)createDownloadLayer((UpDownLayerBuilder *)builder, layerNumber);
        case LayerType::UPLOAD:
            return (fyusenet::LayerBase *)createUploadLayer((UpDownLayerBuilder *)builder, layerNumber);
        case LayerType::SIGMOID:
            return (fyusenet::LayerBase *)createSigmoidLayer((GPULayerBuilder *)builder, layerNumber);
        case LayerType::IMGEXTRACT:
            return (fyusenet::LayerBase *)createImgExtractLayer((ImgExtractLayerBuilder *)builder, layerNumber);
        case LayerType::NONMAX2D:
            return (fyusenet::LayerBase *)createNonMax2DLayer((GPULayerBuilder *)builder, layerNumber);
        case LayerType::BLUR2D:
            return (fyusenet::LayerBase *)createBlur2DLayer((BlurLayerBuilder *)builder, layerNumber);
        case LayerType::RGB2BGR:
            return (fyusenet::LayerBase *)createRGB2BGRLayer((GPULayerBuilder *)builder, layerNumber);
        case LayerType::TANH:
            return (fyusenet::LayerBase *)createTanhLayer((GPULayerBuilder *)builder, layerNumber);
        case LayerType::SINGLETON_ARITH:
            return (fyusenet::LayerBase *)createSingletonArithLayer((SingletonArithLayerBuilder *)builder, layerNumber);
        case LayerType::CAST:
            return (fyusenet::LayerBase *)createCastLayer((CastLayerBuilder *)builder, layerNumber);
        case LayerType::TRANSPOSE:
            return (fyusenet::LayerBase *)createTransposeLayer((TransposeLayerBuilder *)builder, layerNumber);
        case LayerType::BATCHNORM:
            return (fyusenet::LayerBase *)createBatchNormLayer((GPULayerBuilder *)builder, layerNumber);
        case LayerType::GEMM:
            return (fyusenet::LayerBase *)createGEMMLayer((GPULayerBuilder *)builder, layerNumber);
        default:
            THROW_EXCEPTION_ARGS(FynException,"Unsupported layer type");
    }
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param context Link to GL context to operate under
 *
 * @throws GLException if OpenGL minimum requirements are not met
 *
 * @see checkGLRequirements
 */
GPULayerFactoryBackend::GPULayerFactoryBackend(GfxContextLink context):LayerFactoryBackend(), context_(context) {
    checkRequirements();
}


/**
 * @brief Check if minimum graphics-backend requirements are satisified to use this layer backend
 *
 * @throws GLException in case the requirements are not satisfied
 */
void GPULayerFactoryBackend::checkRequirements() {
    try {
        GLInfo::getVersion();
    } catch (GLException & firstchance) {
        GLInfo::init();
    }
    if ((GLInfo::getVersion() < GLInfo::GL_4_0) ||
            ((GLInfo::getVersion() >= GLInfo::GLES_2_0) && (GLInfo::getVersion() < GLInfo::GLES_3_0)) ||
        ((GLInfo::getVersion() >= GLInfo::WEBGL_1_0 && GLInfo::getVersion() < GLInfo::WEBGL_2_0))) {
        THROW_EXCEPTION_ARGS(GLException,"Unsupported OpenGL version");
    }
}


/**
 * @brief Create an argmax layer
 *
 * @param builder Instance of ArgMaxLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see deep::DeepArgMaxLayer
 *
 * @warning This layer is currently only implemented for deep-tensor format
 */
GPULayerBase * GPULayerFactoryBackend::createArgMaxLayer(ArgMaxLayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        return new deep::DeepArgMaxLayer(*builder,layerNumber);
    }
    THROW_EXCEPTION_ARGS(FynException,"Not implemented yet");
}



/**
 * @brief Create a download layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see DownloadLayer, deep::DeepDownloadLayer
 */
GPULayerBase * GPULayerFactoryBackend::createDownloadLayer(UpDownLayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        return new deep::DeepDownloadLayer(*builder, layerNumber);
    }
    return new DownloadLayer(*builder, layerNumber);
}



/**
 * @brief Create upload layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 */
GPULayerBase * GPULayerFactoryBackend::createUploadLayer(UpDownLayerBuilder * builder, int layerNumber) {
    if (builder->isDeep()) {
        THROW_EXCEPTION_ARGS(FynException,"Deep upload layers not supported as of now");
    }
    return new UploadLayer(*builder, layerNumber);
}



/**
 * @brief Create an ES-texture conversion layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see OESConverter
 */
GPULayerBase * GPULayerFactoryBackend::createOESLayer(GPULayerBuilder *builder,int layerNumber) {
#ifdef FYUSENET_USE_EGL
    return new OESConverter(*builder, layerNumber);
#else
    THROW_EXCEPTION_ARGS(FynException,"This layer (OES) is only supported on GLES");
#endif
}


/**
 * @brief Create a padding layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @note This uses a scaling layer internally as we have no dedicated padding layer
 *
 * @see ScaleLayer, deep::DeepScaleLayer
 */
GPULayerBase * GPULayerFactoryBackend::createPaddingLayer(GPULayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        return new deep::DeepScaleLayer(*builder,layerNumber);
    }
    return new ScaleLayer(*builder, layerNumber);
}


/**
 * @brief Create an addition pr subtraction layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see AddSubLayer
 */
GPULayerBase * GPULayerFactoryBackend::createAddSubLayer(GPULayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        THROW_EXCEPTION_ARGS(FynException, "Deep add/sub not supported yet");
    }
    return new AddSubLayer(*builder, layerNumber);
}


/**
 * @brief Create a 2D convolution layer
 *
 * @param builder Instance of ConvLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see vanilla::ConvLayer1x1,vanilla::ConvLayer3x3,vanilla::ConvLayer5x5,vanilla::ConvLayer7x7
 * @see vanilla::ConvLayer9x9, vanilla::DepthwiseConvLayer3x3
 * @see deep::DeepConvLayer1x1,deep::DeepConvLayer3x3,deep::DeepConvLayer5x5,deep::DeepConvLayer7x7
 * @see deep::DeepConvLayer9x9, deep::DeepDepthwiseConvLayer3x3
 */
GPULayerBase * GPULayerFactoryBackend::createConvLayer(ConvLayerBuilder *builder,int layerNumber) {
    // NOTE (mw) oh boy, this is super-messy, clean it up in the future
    if (builder->isDeep()) {
        switch (builder->kernel_) {
            case 1:
                if ((builder->groupSize_ != 1) && (builder->groupSize_ == builder->in())) {
                    THROW_EXCEPTION_ARGS(FynException,"No 1x1 depthwise layer supported");
                }
                return new deep::DeepConvLayer1x1(*builder,layerNumber);
            case 3:
                if ((builder->groupSize_ != 1) && (builder->groupSize_ == builder->in())) {
                    return new deep::DeepDepthwiseConvLayer3x3(*builder,layerNumber);
                }
                return new deep::DeepConvLayerNxN(*builder,layerNumber);
            default:
                if ((builder->groupSize_ != 1) && (builder->groupSize_ == builder->in())) {
                    THROW_EXCEPTION_ARGS(FynException,"No %dx%d depthwise layer supported", builder->kernel_, builder->kernel_);
                }
                return new deep::DeepConvLayerNxN(*builder,layerNumber);
        }
    }
    switch (builder->kernel_) {
        case 1:
            if ((builder->groupSize_ != 1) && (builder->groupSize_ == builder->in())) {
                THROW_EXCEPTION_ARGS(FynException,"No 1x1 depthwise layer supported");
            }
            return new vanilla::ConvLayer1x1(*builder, layerNumber);
        case 3:
            if ((builder->groupSize_ != 1) && (builder->groupSize_ == builder->in())) {
                return new vanilla::DepthwiseConvLayer3x3(*builder, layerNumber);
            }
            return new vanilla::ConvLayerNxN(*builder, layerNumber);
        default:
            if ((builder->groupSize_ != 1) && (builder->groupSize_ == builder->in())) {
                THROW_EXCEPTION_ARGS(FynException,"No %dx%d depthwise layer supported", builder->kernel_, builder->kernel_);
            }
            return new vanilla::ConvLayerNxN(*builder, layerNumber);
    }
}


/**
 * @brief Create a transpose-convolution layer
 *
 * @param builder Instance of ConvLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see TransConvLayer2x2, TransConvLayer3x3, DeepTransConvLayer2x2, DeepTransConvLayer3x3
 */
GPULayerBase * GPULayerFactoryBackend::createTransConvLayer(ConvLayerBuilder *builder,int layerNumber) {
    if ((builder->kernel_ != 2 && builder->kernel_ != 3)) THROW_EXCEPTION_ARGS(FynException,"Transpose convolution is currently only implemented for 2x2 and 3x3 kernels");
    if (builder->isDeep()) {
        switch (builder->kernel_) {
            case 2:
                return new deep::DeepTransConvLayer2x2(*builder, layerNumber);
            case 3:
                return new deep::DeepTransConvLayer3x3(*builder, layerNumber);
            default:
                THROW_EXCEPTION_ARGS(FynException,"Kernel size %d is not supported for deep transpose conv (%s)",builder->kernel_, builder->name_.c_str());
        }
    }
    switch (builder->kernel_) {
        case 2:
            return new vanilla::TransConvLayer2x2(*builder, layerNumber);
        case 3:
            return new vanilla::TransConvLayer3x3(*builder, layerNumber);
        default:
            THROW_EXCEPTION_ARGS(FynException,"Kernel size %d is not supported for deep transpose conv (%s)",builder->kernel_, builder->name_.c_str());
    }
}



/**
 * @brief Create a fractional convolution layer
 *
 * @param builder Instance of ConvLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see FractionalConvLayer3x3, FractionalConvLayer5x5, FractionalConvLayer7x7, FractionalConvLayer9x9
 *
 * @warning This is only implemented for shallow-format tensors
 */
GPULayerBase * GPULayerFactoryBackend::createFracConvLayer(ConvLayerBuilder *builder,int layerNumber) {
    if (builder->isDeep()) {
        THROW_EXCEPTION_ARGS(FynException,"Fractional convolution is not supported for deep layers");
    }    
    switch (builder->kernel_) {
        case 1:
            THROW_EXCEPTION_ARGS(FynException,"Kernel size 1 not supported for fractional convolution (%s)",builder->name_.c_str());
        default:
            return new vanilla::FractionalConvLayerNxN(*builder, layerNumber);
    }
}


/**
 * @brief Create a scaling layer
 *
 * @param builder Instance of ScaleLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see ScaleLayer, DeepScaleLayer
 */
GPULayerBase * GPULayerFactoryBackend::createScaleLayer(ScaleLayerBuilder *builder,int layerNumber) {
    if (builder->isDeep()) {
        return new deep::DeepScaleLayer(*builder, layerNumber);
    }
    return new ScaleLayer(*builder, layerNumber);
}


/**
 * @brief Create a concatenation layer
 *
 * @param builder Instance of ConcatLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see ConcatLayer, DeepConcatLayer
 */
GPULayerBase * GPULayerFactoryBackend::createConcatLayer(ConcatLayerBuilder *builder,int layerNumber) {
    if (builder->isDeep()) {
        return new deep::DeepConcatLayer(*builder, layerNumber);
    }
    return new ConcatLayer(*builder, layerNumber);
}


/**
 * @brief Create a shallow-to-deep tensor format conversion layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see Shallow2DeepLayer
 */
GPULayerBase * GPULayerFactoryBackend::createS2DLayer(GPULayerBuilder *builder,int layerNumber) {
    return new Shallow2DeepLayer(*builder, layerNumber);
}


/**
 * @brief Create a deep-to-shallow tensor format conversion layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see Deep2ShallowLayer
 */
GPULayerBase * GPULayerFactoryBackend::createD2SLayer(GPULayerBuilder *builder,int layerNumber) {
    return new Deep2ShallowLayer(*builder, layerNumber);
}


/**
 * @brief Create a max-pooling layer
 *
 * @param builder Instance of PoolLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see MaxPoolLayer, DeepMaxPoolLayer
 */
GPULayerBase * GPULayerFactoryBackend::createMaxPoolLayer(PoolLayerBuilder *builder,int layerNumber) {
    if ((builder->poolsize_[0] == 1) && (builder->poolsize_[1] == 1)) {
        THROW_EXCEPTION_ARGS(FynException,"Pooling layer with a pool size of 1 does not make sense, clean your net (%s)", builder->name_.c_str());
    }
    if (builder->isDeep()) {
        return new deep::DeepMaxPoolLayer(*builder, layerNumber);
    }
    return new MaxPoolLayer(*builder, layerNumber);
}


/**
 * @brief Create an average-pooling layer
 *
 * @param builder Instance of PoolLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see AvgPoolLayer, DeepAvgPoolLayer
 */
GPULayerBase * GPULayerFactoryBackend::createAvgPoolLayer(PoolLayerBuilder *builder,int layerNumber) {
    if ((builder->poolsize_[0] == 1) && (builder->poolsize_[1] == 1)) {
        THROW_EXCEPTION_ARGS(FynException,"Pooling layer with a pool size of 1 does not make sense, clean your net (%s)", builder->name_.c_str());
    }
    if (builder->isDeep()) {
        return new deep::DeepAvgPoolLayer(*builder, layerNumber);
    }
    return new AvgPoolLayer(*builder, layerNumber);
}


/**
 * @brief Create a custom layer
 *
 * @param builder Instance of CustomLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * In contrast to the built-in layer generators, this function calls the user-supplied
 * CustomLayerBuilder::init() function to create the custom layer and returns it.
 */
GPULayerBase * GPULayerFactoryBackend::createCustomLayer(CustomLayerBuilder *builder,int layerNumber) {
    builder->number_ = layerNumber;
    return builder->init();
}



/**
 * @brief Create a sigmoid activation layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see SigmoidLayer, DeepSigmoidLayer
 */
GPULayerBase * GPULayerFactoryBackend::createSigmoidLayer(GPULayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        return new deep::DeepSigmoidLayer(*builder, layerNumber);
    }
    return new SigmoidLayer(*builder, layerNumber);
}


/**
 * @brief Create a tanh activation layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see TanhLayer, DeepTanhLayer
 */
GPULayerBase * GPULayerFactoryBackend::createTanhLayer(GPULayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        return new deep::DeepTanhLayer(*builder, layerNumber);
    }
    return new TanhLayer(*builder, layerNumber);
}


/**
 * @brief Create an image-patch extraction layer
 *
 * @param builder Instance of ImgExtraactLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @warning That layer type has not been used for a long time now
 *
 * @see DeepExtractImagePatches
 *
 * @warning This is only implemented for deep-format tensors and has not been used in quite a while
 *          before the public release
 */
GPULayerBase * GPULayerFactoryBackend::createImgExtractLayer(ImgExtractLayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        THROW_EXCEPTION_ARGS(FynException,"Shallow imgextract currently not supported");
    }
    return new deep::DeepExtractImagePatches(*builder, layerNumber);
}


/**
 * @brief Create a 2D non-maximum-suppression layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see NonMaxSuppression2D
 *
 * @warning This is only implemented for shallow-format tensors
 */
GPULayerBase * GPULayerFactoryBackend::createNonMax2DLayer(GPULayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        THROW_EXCEPTION_ARGS(FynException,"Deep nonmax suppression currently not supported");
    }
    return new NonMaxSuppression2D(*builder, layerNumber);
}


/**
 * @brief Create a 2D blurring layer
 *
 * @param builder Instance of BlurLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see BlurLayer
 *
 * @warning This is only implemented for shallow-format tensors
 */
GPULayerBase * GPULayerFactoryBackend::createBlur2DLayer(BlurLayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        THROW_EXCEPTION_ARGS(FynException,"Deep blurring currently not supported");
    }
    return new BlurLayer(*builder, layerNumber);
}


/**
 * @brief Create a RGB->BGR conversion layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see RGB2BGRLayer
 *
 * @warning This is only implemented for shallow format tensors (well, there are only 3 channels in
 *          an RGB/BGR image anyway)
 */
GPULayerBase * GPULayerFactoryBackend::createRGB2BGRLayer(GPULayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        THROW_EXCEPTION_ARGS(FynException,"Deep rgb->bgr currently not supported");
    }
    return new RGB2BGRLayer(*builder, layerNumber);
}


/**
 * @brief Create a singleton arithmetic layer where a singleton is arithmetically combined with a tensor
 *
 * @param builder Instance of SingletonArithLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see SingletonArithmeticLayer, DeepSingletonArithmeticLayer
 */
GPULayerBase * GPULayerFactoryBackend::createSingletonArithLayer(SingletonArithLayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        return new deep::DeepSingletonArithmeticLayer(*builder, layerNumber);
    }
    return new SingletonArithmeticLayer(*builder, layerNumber);
}


/**
 * @brief Create a type-cast layer
 *
 * @param builder Instance of CastLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @note On the GPU, type-casting is more like a rounding mode as the data will still be represented
 *       as floating-point.
 *
 * @see CastLayer, DeepCastLayer
 */
GPULayerBase * GPULayerFactoryBackend::createCastLayer(CastLayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        return new deep::DeepCastLayer(*builder, layerNumber);
    }
    return new CastLayer(*builder, layerNumber);
}

/**
 * @brief Create a transposition layer
 *
 * @param builder Instance of TranposeLayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see DeepTransposeLayer
 *
 * @warning Currently not implemented for shallow tensors
 */
GPULayerBase * GPULayerFactoryBackend::createTransposeLayer(TransposeLayerBuilder *builder, int layerNumber) {
    if (builder->isDeep()) {
        return new deep::DeepTransposeLayer(*builder, layerNumber);
    }
    THROW_EXCEPTION_ARGS(FynException,"No shallow transpose layer support (yet)");
}


/**
 * @brief Create a batchnorm layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see BatchNormLayer, deep::DeepBatchNormLayer
 */
GPULayerBase * GPULayerFactoryBackend::createBatchNormLayer(GPULayerBuilder * builder, int layerNumber) {
    if (builder->isDeep()) {
        return new gpu::deep::DeepBatchNormLayer(*builder, layerNumber);
    }
    return new BatchNormLayer(*builder, layerNumber);
}


/**
 * @brief Create a GEMM layer
 *
 * @param builder Instance of GPULayerBuilder that contains the parameters for the layer
 *
 * @param layerNumber Layer number to assigned to the created layer, must be unique
 *
 * @return Raw pointer to created layer
 *
 * @see vanilla::ConvLayer1x1, deep::DeepGEMMLayer
 */
GPULayerBase * GPULayerFactoryBackend::createGEMMLayer(GPULayerBuilder * builder, int layerNumber) {
    if (builder->isDeep()) {
        return new gpu::deep::DeepGEMMLayer(*builder, layerNumber);
    }
    return new gpu::vanilla::ConvLayer1x1(*builder, layerNumber);
}


} // gpu namespace
} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
