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

#include "deepbatchnormlayer.h"
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
DeepBatchNormLayer::DeepBatchNormLayer(const GPULayerBuilder & builder, int layerNumber) : DeepFunctionLayer(builder, layerNumber) {
    shader_ = nullptr;
}

DeepBatchNormLayer::~DeepBatchNormLayer() {
    delete [] bnBias_;
    delete [] bnScales_;
    bnBias_ = nullptr;
    bnScales_ = nullptr;
}

/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepBatchNormLayer::cleanup() {
    delete scaleAttribs_;
    delete biasAttribs_;
    scaleAttribs_ = nullptr;
    biasAttribs_ = nullptr;
    // reset shaders here because the GL context is bound here (in case no cache is used)
    shader_.reset();
    DeepFunctionLayer::cleanup();
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> DeepBatchNormLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0,0,viewport_[0],viewport_[1],TEXTURE_IFORMAT_4,TEXTURE_FORMAT_4,TEXTURE_TYPE_DEFAULT,BufferSpec::FUNCTION_DEST));
    return result;
}

/**
 * @copydoc BatchNormInterface::loadScaleAndBias()
 */
void DeepBatchNormLayer::loadScaleAndBias(const float * scaleAndBias, size_t sbOffset) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    int padout = PIXEL_PACKING * ((outputChannels_ + PIXEL_PACKING-1) / PIXEL_PACKING);
    const float * srcbn = scaleAndBias + sbOffset;
    bnScales_ = new float[padout];
    bnBias_ = new float[padout];
    memset(bnScales_,0,padout * sizeof(float));
    memset(bnBias_,0,padout * sizeof(float));
    memcpy(bnScales_, srcbn, outputChannels_*sizeof(float));
    memcpy(bnBias_, srcbn + outputChannels_, outputChannels_*sizeof(float));
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc DeepFunctionLayer::setupNetworkPolygons
 */
void DeepBatchNormLayer::setupNetworkPolygons(VAO *vao) {
    assert(bnScales_);
    assert(bnBias_);
    DeepFunctionLayer::setupNetworkPolygons(vao);
    float * attrs1 = new float[tiler_->numOutputTiles() * 4 * PIXEL_PACKING];       // stores scales
    float * attrs2 = new float[tiler_->numOutputTiles() * 4 * PIXEL_PACKING];       // stores biases
    int tgtoffset = 0;
    for (int i=0, chan=0; i < tiler_->numOutputTiles(); i++, chan += PIXEL_PACKING) {
        for (int rep=0; rep < 4; rep++) {
            for (int inner=0; inner < PIXEL_PACKING; inner++) {
                attrs1[tgtoffset] = bnScales_[chan+inner];
                attrs2[tgtoffset++] = bnBias_[chan+inner];
            }
        }
    }
    vao->enableArray(1);
    scaleAttribs_ = new VBO(context_);
    scaleAttribs_->setBufferData(attrs1, tiler_->numOutputTiles()*4*4*sizeof(float), GL_STATIC_DRAW);
    scaleAttribs_->bind();
    vao->setVertexAttributeBuffer(1,4,GL_FLOAT,GL_FALSE,0,0);
    delete [] attrs1;
    biasAttribs_ = new VBO(context_);
    vao->enableArray(2);
    biasAttribs_->setBufferData(attrs2, tiler_->numOutputTiles()*4*4*sizeof(float), GL_STATIC_DRAW);
    biasAttribs_->bind();
    vao->setVertexAttributeBuffer(2,4,GL_FLOAT,GL_FALSE,0,0);
    delete [] attrs2;
}


/**
 * @copydoc DeepFunctionLayer::renderChannelBatch
 */
void DeepBatchNormLayer::renderChannelBatch() {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    int quads = tiler_->numOutputTiles();
    glDrawElements(GL_TRIANGLES,quads*6,GL_UNSIGNED_SHORT,(const GLvoid *)0);
}


/**
 * @copydoc DeepFunctionLayer::beforeRender
 */
void DeepBatchNormLayer::beforeRender() {
    shader_->bind(shaderState_.get());
}


/**
 * @copydoc DeepFunctionLayer::afterRender
 */
void DeepBatchNormLayer::afterRender() {
    shader_->unbind();
}


/**
 * @copydoc DeepFunctionLayer::setupShaders
 */
void DeepBatchNormLayer::setupShaders() {
    char preproc[1024] = {0};
    handlePreprocFlags(flags_, preproc, sizeof(preproc)-1);
    shader_ = compileShaderPair("shaders/deep/deepbatchnorm.vert","shaders/deep/deepbatchnorm.frag",preproc,typeid(this));
    try {
        shader_->bindAttributeLocation("attributes0",0);
        shader_->bindAttributeLocation("attributes1",1);
        shader_->bindAttributeLocation("attributes2",2);
        shader_->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    shaderState_ = UniformState::makeShared(shader_);
    if (!GLInfo::hasBinding()) {
        shaderState_->setUniformValue("inputLayer0",0);
        shaderState_->setUniformValue("residualLayer0",1, true);
    }
}


} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
