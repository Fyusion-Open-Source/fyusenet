//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Global Pooling Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "deepglobalpoollayer.h"
#include "../../gl/glexception.h"

namespace fyusion::fyusenet::gpu::deep {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
DeepGlobalPoolLayer::DeepGlobalPoolLayer(const PoolLayerBuilder & builder,int layerNumber) :
    DeepPoolingLayer(builder,layerNumber) {
    tiler_->setGlobalPooling();

    switch (builder.operation_) {
        case PoolLayerBuilder::POOL_MAX:
            mode_ = MAXPOOL;
            break;
        case PoolLayerBuilder::POOL_AVG:
            mode_ = AVGPOOL;
            break;
    }    
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepGlobalPoolLayer::cleanup() {
    shader_.reset();
    DeepPoolingLayer::cleanup();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc DeepPoolingLayer::beforeRender
 */
void DeepGlobalPoolLayer::beforeRender() {
    shader_->bind(shaderState_.get());
    glDisable(GL_BLEND);
}


/**
 * @copydoc DeepPoolingLayer::renderChannelBatch
 */
void DeepGlobalPoolLayer::renderChannelBatch() {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    int points = tiler_->numOutputTiles();
    glDrawArrays(GL_POINTS, 0, points);
}


/**
 * @copydoc DeepPoolingLayer::afterRender
 */
void DeepGlobalPoolLayer::afterRender() {
    shader_->unbind();
    glDisable(GL_BLEND);
}


/**
 * @copydoc DeepPoolingLayer::setupShaders
 */
void DeepGlobalPoolLayer::setupShaders() {
    char preproc[1024] = {0};
    preprocessor_.generatePreprocessorPreamble(flags_, preproc, sizeof(preproc)-1);
    if (mode_ == AVGPOOL) {
        shader_ = compileShaderPair("shaders/deep/deepdefault.vert","shaders/deep/deepglobavgpool.frag",preproc,typeid(this));
    } else {
        shader_ = compileShaderPair("shaders/deep/deepdefault.vert","shaders/deep/deepglobmaxpool.frag",preproc,typeid(this));
    }
    try {
        shader_->bindAttributeLocation("attributes0",0);
        shader_->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    shaderState_ = UniformState::makeShared(shader_);
    shaderState_->setUniformValue("inputLayer0",0);
    shaderState_->setUniformVec2("imdim", width_, height_);
    shaderState_->setUniformVec2("texStep", 1.0f/(float)tiler_->getInputTextureWidth(), 1.0f/(float)tiler_->getInputTextureHeight());
}


void DeepGlobalPoolLayer::setupNetworkPolygons(VAO *vao) {
    float * attrs0 = new float[tiler_->numOutputTiles()*4];
    std::vector<DeepTiler::Tile> otiles = tiler_->createOutputTiles();
    std::vector<DeepTiler::Tile> itiles = tiler_->createInputTiles(0,0);
    assert(otiles.size() == itiles.size());
    for (int i=0; i < (int)itiles.size(); i++) {
        DeepTiler::Tile & ot = otiles.at(i);
        attrs0[i*4] = (ot.quad_[0] + ot.quad_[2] + ot.quad_[4] + ot.quad_[6]) / 4.0f;
        attrs0[i*4+1] = (ot.quad_[1] + ot.quad_[3] + ot.quad_[5] + ot.quad_[7]) / 4.0f;
        DeepTiler::Tile & it = itiles.at(i);
        attrs0[i*4+2] = it.quad_[0];
        attrs0[i*4+3] = it.quad_[1];
    }
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(attrs0, (GLsizei)(tiler_->numOutputTiles() * 4 * sizeof(float)), GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    delete [] attrs0;
}



} // fyusion::fyusenet::gpu::deep namespace

// vim: set expandtab ts=4 sw=4:
