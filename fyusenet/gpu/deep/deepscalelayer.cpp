//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Scaling Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "deepscalelayer.h"
#include "../../gl/glexception.h"
#include "../../gl/glinfo.h"
#include "../../common/logging.h"
#include "deeptiler.h"

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
 * @copydoc GPULayerBase::GPULayerBase
 */
DeepScaleLayer::DeepScaleLayer(const ScaleLayerBuilder & builder, int layerNumber) :
      DeepFunctionLayer((const GPULayerBuilder &)builder, layerNumber) {
    type_ = builder.scaleType_;
    if ((builder.width() == 1) || (builder.height() == 1)) type_ = ScalingType::NEAREST;         // NOTE (mw) no sense to use bilinear interpolation on a single pixel or 1x2 / 2x1 combinations for now
    shader_ = nullptr;
}


/**
 * @copydoc GPULayerBase::GPULayerBase
 */
DeepScaleLayer::DeepScaleLayer(const GPULayerBuilder & builder, int layerNumber) : DeepFunctionLayer(builder, layerNumber) {
    type_ = ScalingType::NEAREST;
    shader_ = nullptr;
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepScaleLayer::cleanup() {
    // reset shaders here because the GL context is bound here (in case no cache is used)
    shader_.reset();
    DeepFunctionLayer::cleanup();
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> DeepScaleLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0,0,viewport_[0],viewport_[1],TEXTURE_IFORMAT_4,TEXTURE_FORMAT_4,TEXTURE_TYPE_DEFAULT,BufferSpec::FUNCTION_DEST));
    return result;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @copydoc DeepFunctionLayer::renderChannelBatch
 */
void DeepScaleLayer::renderChannelBatch() {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    if (type_ == ScalingType::LINEAR) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    int quads = tiler_->numOutputTiles();
    glDrawElements(GL_TRIANGLES,quads*6,GL_UNSIGNED_SHORT,(const GLvoid *)0);
    if (type_ == ScalingType::LINEAR) {
        // reset sampling to nearest here for other layers (default mode)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
}


/**
 * @copydoc DeepFunctionLayer::beforeRender
 */
void DeepScaleLayer::beforeRender() {
    shader_->bind(shaderState_.get());
}


/**
 * @copydoc DeepFunctionLayer::afterRender
 */
void DeepScaleLayer::afterRender() {
    shader_->unbind();
}


/**
 * @copydoc DeepFunctionLayer::setupShaders
 */
void DeepScaleLayer::setupShaders() {
    char preproc[1024] = {0};
    handlePreprocFlags(flags_, preproc, sizeof(preproc)-1);
    shader_ = compileShaderPair("shaders/deep/deepdefault.vert","shaders/deep/deepdefault.frag",preproc,typeid(this));
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
