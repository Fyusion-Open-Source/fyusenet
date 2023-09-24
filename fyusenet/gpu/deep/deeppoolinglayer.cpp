//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Pooling Layer Base Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "deeppoolinglayer.h"
#include "../../gl/glexception.h"
#include "../../gl/glinfo.h"
#include "../../common/logging.h"

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
DeepPoolingLayer::DeepPoolingLayer(const PoolLayerBuilder& builder,int layerNumber) : DeepLayerBase((const GPULayerBuilder &)builder,layerNumber) {
    if (builder.global_) {
        poolSize_[0] = builder.width();
        poolSize_[1] = builder.height();
        downsample_[0] = builder.width();
        downsample_[1] = builder.height();
    } else {
        poolSize_[0] = builder.poolsize_[0];
        poolSize_[1] = builder.poolsize_[1];
        downsample_[0] = builder.downsample_[0];
        downsample_[1] = builder.downsample_[1];
    }
    equalAspect_ = (poolSize_[0]==poolSize_[1]);
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepPoolingLayer::cleanup() {
    if (vertexBuffer_) delete vertexBuffer_;
    if (indexBuffer_) delete indexBuffer_;
    if (vertexArray_) delete vertexArray_;
    vertexBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    vertexArray_ = nullptr;
    DeepLayerBase::cleanup();
}


/**
 * @brief LayerBase::setup
 */
void DeepPoolingLayer::setup() {
    vertexArray_ = new VAO(context_);
    vertexArray_->bind();
    setupNetworkPolygons(vertexArray_);
    vertexArray_->unbind();
    setupShaders();
    setupFBOs();
    valid_ = true;
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> DeepPoolingLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0, tiler_->getInputTextureWidth(), tiler_->getInputTextureHeight(),
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE).dataOrder(BufferSpec::order::GPU_DEEP));
    return result;
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> DeepPoolingLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0 ,0, viewport_[0], viewport_[1],
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_DEST).dataOrder(BufferSpec::order::GPU_DEEP));
    return result;
}


/**
 * @copydoc LayerBase::forward
 */
void DeepPoolingLayer::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (!valid_) THROW_EXCEPTION_ARGS(FynException,"Trying to invoke forward() on invalid layer");
#ifdef DEBUG
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render entry: 0x%x (%s:%d)[%s]",err,__FILE__,__LINE__,getName().c_str());
#endif
    if (outputChanged_) updateFBOs();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_BLEND);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glViewport(0,0,viewport_[0],viewport_[1]);
    framebuffers_.at(0)->bind();
    framebuffers_.at(0)->setWriteMask();
    glClear(GL_COLOR_BUFFER_BIT);         // this is to instruct the tile-engine that we don't need the old tile-content
    vertexArray_->bind();
    beforeRender();
    renderChannelBatch();
    framebuffers_.at(0)->unbind();
    afterRender();
    vertexArray_->unbind();
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Setup a set of proxy polygons that are used to drive the fragment shaders
 *
 * @param vao Pointer to vertex array object that the resulting VBO and IBO are tied to
 *
 * @pre The supplied \p vao vertex array object to be used with this VBO is already bound
 *
 * As fragment shaders are used to perform the computation, a set of proxy polygons is required
 * to cover the output area of the image set which make up the output tensor.
 */
void DeepPoolingLayer::setupNetworkPolygons(VAO *vao) {
    int offset0 = 0;
    float * attrs0 = new float[tiler_->numOutputTiles()*4*4];
    //---------------------------------------------
    // VBO part
    //---------------------------------------------
    std::vector<DeepTiler::Tile> otiles = tiler_->createOutputTiles();
    std::vector<DeepTiler::Tile> itiles = tiler_->createInputTiles(0,0);
    assert(otiles.size() == itiles.size());
    for (int i=0; i < (int)itiles.size(); i++) {
        DeepTiler::Tile & ot = otiles.at(i);
        DeepTiler::Tile & it = itiles.at(i);
        ot.toFloatVec(attrs0, offset0, 4);
        it.toFloatVec(attrs0, offset0+2, 4);
        offset0 += 4*4;
    }
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(attrs0,tiler_->numOutputTiles()*4*4*sizeof(float),GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    delete [] attrs0;
    //---------------------------------------------
    // IBO part
    //---------------------------------------------
    GLshort * indices = new GLshort[tiler_->numOutputTiles()*6];
    indexBuffer_ = new IBO(context_);
    for (int i=0; i < tiler_->numOutputTiles(); i++) {
        int offset = i*4;
        indices[i*6+0] = offset+0;
        indices[i*6+1] = offset+1;
        indices[i*6+2] = offset+2;
        indices[i*6+3] = offset+0;
        indices[i*6+4] = offset+2;
        indices[i*6+5] = offset+3;
    }
    indexBuffer_->setBufferData(indices, 6 * tiler_->numOutputTiles()*sizeof(GLshort), GL_STATIC_DRAW);
    indexBuffer_->bind();
    delete [] indices;
}


} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
