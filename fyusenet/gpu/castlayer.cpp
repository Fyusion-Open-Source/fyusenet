//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Isolated Type-Cast Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "castlayer.h"
#include "../gl/glexception.h"
#include "../gl/glinfo.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
CastLayer::CastLayer(const CastLayerBuilder & builder, int layerNumber):FunctionLayer((GPULayerBuilder &)builder, layerNumber) {
    target_ = builder.target_;
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void CastLayer::cleanup() {
    // reset shaders here because the GL context is bound here (in case no cache is used)
    for (int i=0;i<FBO::MAX_DRAWBUFFERS;i++) {
        shaders_[i].reset();
        shaderStates_[i].reset();
    }
    currentShader_ = nullptr;
    FunctionLayer::cleanup();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @copydoc FunctionLayer::beforeRender
 */
void CastLayer::beforeRender() {
    currentShader_ = nullptr;
}

/**
 * @copydoc FunctionLayer::afterRender
 */
void CastLayer::afterRender() {
    if (currentShader_) currentShader_->unbind();
    currentShader_ = nullptr;
}


/**
 * @copydoc FunctionLayer::renderChannelBatch
 */
void CastLayer::renderChannelBatch(int outPass,int numRenderTargets,int texOffset) {
    for (int tex=0; tex<numRenderTargets; tex++) {
        glActiveTexture(GL_TEXTURE0+tex);
        glBindTexture(GL_TEXTURE_2D,inputTextures_.at(tex+texOffset));
    }
    if (currentShader_ != shaders_[numRenderTargets-1].get()) {
        if (currentShader_) currentShader_->unbind(true);
        currentShader_ = shaders_[numRenderTargets-1].get();
        currentShader_->bind(shaderStates_[numRenderTargets-1].get());
    }
    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,(const GLvoid *)0);
}


/**
 * @copydoc FunctionLayer::setupShaders
 */
void CastLayer::setupShaders() {
    char extra[512];
    for (int i=1; i <= maxRenderTargets_; i++) {
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
        snprintf(extra,sizeof(extra),"#define NUM_LANES %d\n#define CAST_TO_%s\n",i,tc);
        shaders_[i-1] = compileShaderPair("shaders/default.vert","shaders/cast.frag",extra,typeid(this));
        try {
            shaders_[i-1]->bindAttributeLocation("attributes0",0);
            shaders_[i-1]->link();
        } catch (GLException & ex) {
            FNLOGE("Cannot link shader for layer %s",getName().c_str());
            throw;
        }
        shaderStates_[i-1] = UniformState::makeShared(shaders_[i-1]);
        for (int j=0; j < i; j++) {
            snprintf(extra,sizeof(extra),"inputLayer%d",j);
            shaderStates_[i-1]->setUniformValue(extra,j);
        }
    }
}

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
