//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Sigmoid Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "deepsigmoidlayer.h"
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
DeepSigmoidLayer::DeepSigmoidLayer(const GPULayerBuilder & builder, int layerNumber) :
      DeepFunctionLayer(builder, layerNumber) {
    if (builder.getFlags() & LayerFlags::POST_BATCHNORM) THROW_EXCEPTION_ARGS(FynException,"Batchnorm not supported for this layer");
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepSigmoidLayer::cleanup() {
    // reset shaders here because the GL context is bound here (in case no cache is used)
    shader_.reset();
    DeepFunctionLayer::cleanup();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc DeepFunctionLayer::renderChannelBatch
 */
void DeepSigmoidLayer::renderChannelBatch() {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTextures_.at(0));
    int quads = tiler_->numOutputTiles();
    glDrawElements(GL_TRIANGLES, quads * 6, GL_UNSIGNED_SHORT, (const GLvoid *) 0);
}

/**
 * @copydoc DeepFunctionLayer::beforeRender
 */
void DeepSigmoidLayer::beforeRender() {
    shader_->bind(shaderState_.get());
}


/**
 * @copydoc DeepFunctionLayer::afterRender
 */
void DeepSigmoidLayer::afterRender() {
    shader_->unbind();
}


/**
 * @copydoc DeepFunctionLayer::setupShaders
 */
void DeepSigmoidLayer::setupShaders() {
    char preproc[1024] = {0};
    handlePreprocFlags(flags_, preproc, sizeof(preproc)-1);
    shader_ = compileShaderPair("shaders/deep/deepdefault.vert", "shaders/deep/deepsigmoid.frag", preproc, typeid(this));
    try {
        shader_->bindAttributeLocation("attributes0", 0);
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
