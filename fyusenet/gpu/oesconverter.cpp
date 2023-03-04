//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OES Texture Conversion Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------
#ifdef FYUSENET_USE_EGL

//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "oesconverter.h"
#include "../gl/gl_sys.h"
#include "../gl/glexception.h"
#include "../common/logging.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copdoc GPULayerBase::GPULayerBase
 */
OESConverter::OESConverter(const GPULayerBuilder & builder,int layerNumber):FunctionLayer(builder, layerNumber) {
    //if (depth>PIXEL_PACKING) THROW_EXCEPTION_ARGS(FynException,"OES conversion supports 4-components at max");
}

/**
 * @copydoc GPULayerBase::cleanup
 */
void OESConverter::cleanup() {
     // NOTE (mw) reset shaders here because the GL context is bound here (in case no cache is used)
     shader_.reset();
     FunctionLayer::cleanup();
}

/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> OESConverter::getRequiredOutputBuffers() const {
     std::vector<BufferSpec> result;
     result.push_back(BufferSpec(0,0, viewport_[0], viewport_[1],
                                 TEXTURE_IFORMAT_OES, TEXTURE_FORMAT_OES, TEXTURE_TYPE_OES,
                                 BufferSpec::OES_DEST));
     return result;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc FunctionLayer::beforeRender
 */
void OESConverter::beforeRender() {
    shader_->bind(shaderState_.get());
}


/**
 * @copydoc FunctionLayer::renderChannelBatch
 */
void OESConverter::renderChannelBatch(int outPass, int numRenderTargets, int texOffset) {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, inputTextures_.at(texOffset));
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const GLvoid *)0);
}


/**
 * @copydoc FunctionLayer::afterRender
 */
void OESConverter::afterRender() {
    shader_->unbind();
}


/**
 * @copydoc FunctionLayer::setupShaders
 */
void OESConverter::setupShaders() {
    shader_=compileShaderPair("shaders/default.vert","shaders/oes.frag",nullptr,typeid(this));
    try {
        shader_->bindAttributeLocation("attributes0",0);
        shader_->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    shaderState_ = UniformState::makeShared(shader_);
    shaderState_->setUniformValue("inputLayer",0);
}


} // gpu namespace
} // fyusenet namespace
} // fyusion namespace
#endif // ANDROID

// vim: set expandtab ts=4 sw=4:
