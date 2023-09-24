//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep-tensor Type-Casting Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/glexception.h"
#include "../../gl/glinfo.h"
#include "../../common/logging.h"
#include "deeptiler.h"
#include "deepcastlayer.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
DeepCastLayer::DeepCastLayer(const CastLayerBuilder & builder, int layerNumber):DeepFunctionLayer((GPULayerBuilder &)builder, layerNumber) {
    target_ = builder.target_;
    if (builder.getFlags() & LayerFlags::POST_BATCHNORM) THROW_EXCEPTION_ARGS(FynException,"Batchnorm not supported for this layer");
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepCastLayer::cleanup() {
    shader_.reset();
    DeepFunctionLayer::cleanup();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @copydoc DeepFunctionLayer::renderChannelBatch
 */
void DeepCastLayer::renderChannelBatch() {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    int quads = tiler_->numOutputTiles();
    glDrawElements(GL_TRIANGLES,quads*6,GL_UNSIGNED_SHORT,(const GLvoid *)0);
}


/**
 * @copydoc DeepFunctionLayer::beforeRender
 */
void DeepCastLayer::beforeRender() {
    shader_->bind(shaderState_.get());
}


/**
 * @copydoc DeepFunctionLayer::afterRender
 */
void DeepCastLayer::afterRender() {
    shader_->unbind();
}


/**
 * @copydoc DeepFunctionLayer::setupShaders
 */
void DeepCastLayer::setupShaders() {
    char preproc[1024] = {0};
    const char * tc = nullptr;
    switch (target_) {
        case CastTarget::CT_INT32:
            tc = "INT32";
            break;
        case CastTarget::CT_INT16:
            tc = "INT16";
            break;
        case CastTarget::CT_INT8:
            tc = "INT8";
            break;
        case CastTarget::CT_UINT32:
            tc = "UINT32";
            break;
        case CastTarget::CT_UINT16:
            tc = "UINT16";
            break;
        case CastTarget::CT_UINT8:
            tc = "UINT8";
            break;
        case CastTarget::CT_FLOAT16:
            tc = "FLOAT16";
            break;
        case CastTarget::CT_FLOAT32:
            tc = "FLOAT32";
            break;
        default:
            THROW_EXCEPTION_ARGS(FynException,"Illegal cast target supplied");
    }
    snprintf(preproc, sizeof(preproc), "define CAST_TO_%s\n",tc);
    preprocessor_.generatePreprocessorPreamble(flags_, preproc, sizeof(preproc)-strlen(preproc)-1);
    shader_ = compileShaderPair("shaders/deep/deepdefault.vert","shaders/deep/deepcast.frag",preproc,typeid(this));
    try {
        shader_->bindAttributeLocation("attributes0",0);
        shader_->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    shaderState_ = UniformState::makeShared(shader_);
    shaderState_->setUniformValue("inputLayer0",0);
}


} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
