//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Singleton Arithmetic Layer
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
#include "deep_singleton_arithlayer.h"

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
DeepSingletonArithmeticLayer::DeepSingletonArithmeticLayer(const SingletonArithLayerBuilder & builder, int layerNumber):DeepFunctionLayer((const GPULayerBuilder &)builder, layerNumber) {
    assert(builder.type_ != LayerType::ILLEGAL);
    optype_ = builder.opType_;
    operand_ = builder.operand_;
    if (builder.getFlags() & LayerFlags::POST_BATCHNORM) THROW_EXCEPTION_ARGS(FynException,"Batchnorm not supported for this layer");
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepSingletonArithmeticLayer::cleanup() {
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
void DeepSingletonArithmeticLayer::renderChannelBatch() {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    int quads = tiler_->numOutputTiles();
    glDrawElements(GL_TRIANGLES,quads*6,GL_UNSIGNED_SHORT,(const GLvoid *)0);
}


/**
 * @copydoc DeepFunctionLayer::beforeRender
 */
void DeepSingletonArithmeticLayer::beforeRender() {
    shader_->bind(shaderState_.get());
}


/**
 * @copydoc DeepFunctionLayer::afterRender
 */
void DeepSingletonArithmeticLayer::afterRender() {
    shader_->unbind();
}


/**
 * @copydoc DeepFunctionLayer::setupShaders
 */
void DeepSingletonArithmeticLayer::setupShaders() {
    char preproc[1024] = {0};
    const char *opname = nullptr;
    switch (optype_) {
        case ArithType::ADD:
            opname = "ADD";
            break;
        case ArithType::SUB:
            opname = "SUB";
            break;
        case ArithType::MUL:
            opname = "MUL";
            break;
        case ArithType::DIV:
            opname = "DIV";
            break;
    }
    snprintf(preproc,sizeof(preproc),"#define ARITH_OP_%s\n",opname);
    handlePreprocFlags(flags_, preproc, sizeof(preproc)-strlen(preproc)-1);
    shader_ = compileShaderPair("shaders/deep/deepdefault.vert","shaders/deep/deep_singleton_arith.frag",preproc,typeid(this));
    try {
        shader_->bindAttributeLocation("attributes0",0);
        shader_->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    shaderState_ = UniformState::makeShared(shader_);
    shaderState_->setUniformValue("inputLayer0",0);
    shaderState_->setUniformValue("operand",operand_);
}


} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
