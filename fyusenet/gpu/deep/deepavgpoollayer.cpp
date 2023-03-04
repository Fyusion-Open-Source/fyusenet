//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Average-Pool Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "deepavgpoollayer.h"
#include "../../gl/glexception.h"

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
DeepAvgPoolLayer::DeepAvgPoolLayer(const PoolLayerBuilder & builder,int layerNumber):DeepPoolingLayer(builder,layerNumber) {
    assert(builder.operation_ == PoolLayerBuilder::POOL_AVG);
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepAvgPoolLayer::cleanup() {
    shader_.reset();
    DeepPoolingLayer::cleanup();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @copydoc DeepPoolingLayer::beforeRender
 */
void DeepAvgPoolLayer::beforeRender() {
    shader_->bind(shaderState_.get());
}


/**
 * @copydoc DeepPoolingLayer::renderChannelBatch
 */
void DeepAvgPoolLayer::renderChannelBatch() {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    int quads = tiler_->numOutputTiles();
    glDrawElements(GL_TRIANGLES,quads*6,GL_UNSIGNED_SHORT,(const GLvoid *)0);
}


/**
 * @copydoc DeepPoolingLayer::afterRender
 */
void DeepAvgPoolLayer::afterRender() {
    shader_->unbind();
}

/**
 * @copydoc DeepPoolingLayer::setupShaders
 */
void DeepAvgPoolLayer::setupShaders() {
    char preproc[1024]={0}, add[80];
    ssize_t mc = (ssize_t)shaderPreprocessing(preproc, sizeof(preproc)-1);
    assert(mc > 0);
    bool useloop = false;
    if (equalAspect_) {
        if (poolSize_[0] > 4) useloop = true;
        else {
            snprintf(add, sizeof(add), "#define POOLSIZE %d\n", poolSize_[0]);
            strncat(preproc, add, mc);
            mc -= strlen(add);
            assert(mc > 0);
        }
    } else useloop = true;
    snprintf(add,sizeof(add),"#define POOLSIZE_X %d\n#define POOLSIZE_Y %d\n",poolSize_[0], poolSize_[1]);
    strncat(preproc, add, mc);
    mc -= strlen(add);
    assert(mc > 0);
    shader_ = compileShaderPair("shaders/deep/deepdefault.vert","shaders/deep/deepavgpool.frag",preproc,typeid(this));
    try {
        shader_->bindAttributeLocation("attributes0",0);
        shader_->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    shaderState_ = UniformState::makeShared(shader_);
    shaderState_->setUniformValue("inputLayer0",0);
    if (useloop) {
        shaderState_->setUniformVec2("texStep",tiler_->getTextureStepX(),tiler_->getTextureStepY());
    }
}



} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
