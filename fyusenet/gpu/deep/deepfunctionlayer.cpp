//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Function Layer Base Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/glexception.h"
#include "../../gl/glinfo.h"
#include "../../gl/fbo.h"
#include "../../gl/vao.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../common/logging.h"
#include "deeptiler.h"
#include "deepfunctionlayer.h"

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
DeepFunctionLayer::DeepFunctionLayer(const GPULayerBuilder & builder, int layerNumber) :
    DeepLayerBase(builder, layerNumber) {
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepFunctionLayer::cleanup() {
    delete vertexBuffer_;
    delete indexBuffer_;
    delete vertexArray_;
    vertexBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    vertexArray_ = nullptr;
    DeepLayerBase::cleanup();
}


/**
 * @copydoc LayerBase::setup
 */
void DeepFunctionLayer::setup() {
    vertexArray_ = new VAO(context_);
    vertexArray_->bind();
    setupNetworkPolygons(vertexArray_);
    vertexArray_->unbind();
    setupShaders();
    setupFBOs();
    valid_=true;
}


/**
 * @copydoc LayerBase::forward
 */
void DeepFunctionLayer::forward(uint64_t sequence) {
    if (!valid_) THROW_EXCEPTION_ARGS(FynException,"Trying to invoke forward() on invalid layer");
#ifdef DEBUG
    int err = glGetError();
    if (err != GL_NO_ERROR) FNLOGD("HINT: glerror on render entry: 0x%x (%s:%d)[%s]",err,__FILE__,__LINE__,getName().c_str());
#endif
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
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
    afterRender();
    framebuffers_.at(0)->unbind();
    vertexArray_->unbind();
}

/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> DeepFunctionLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0,0,tiler_->getInputTextureWidth(),tiler_->getInputTextureHeight(),
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE).dataOrder(BufferSpec::order::GPU_DEEP));
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        result.push_back(BufferSpec(0, 1, residualViewport_[0], residualViewport_[1],
                                    TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                    BufferSpec::RESIDUAL_SOURCE).dataOrder(BufferSpec::order::GPU_DEEP));
    }
    return result;
}

/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> DeepFunctionLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0, viewport_[0], viewport_[1],
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_DEST).dataOrder(BufferSpec::order::GPU_DEEP));
    return result;
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Setup a set of proxy polygons that are used to drive the fragment shaders
 *
 * @param vao Vertex array object to be used with the vertex/index buffers created here
 *
 * @pre The vertex array object to be used with this VBO is already bound
 *
 * As fragment shaders are used to perform the computation, a set of proxy polygons is required
 * to cover the output area of the image set which make up a tensor. In particular for deep-channel
 * tensors, a \e set of output polygons ("tiles") is used to represent the tensor data. These
 * tiles are setup here with the help of the DeepTiler class.
 *
 * @see DeepTiler
 */
void DeepFunctionLayer::setupNetworkPolygons(VAO *vao) {
    int offset0=0;
    float * attrs0 = new float[tiler_->numOutputTiles()*4*4];
    //---------------------------------------------
    // VBO part
    //---------------------------------------------
    std::vector<DeepTiler::Tile> otiles = tiler_->createOutputTiles();
    std::vector<DeepTiler::Tile> itiles = tiler_->createInputTiles(0, 0);
    assert(otiles.size() == itiles.size());
    for (int i=0; i < (int)itiles.size(); i++) {
        DeepTiler::Tile ot = otiles.at(i);
        DeepTiler::Tile it = itiles.at(i);
        ot.toFloatVec(attrs0,offset0,4);
        it.toFloatVec(attrs0,offset0+2,4);
        offset0 += 4*4;
    }
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(attrs0, tiler_->numOutputTiles()*4*4*sizeof(float), GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0,4,GL_FLOAT,GL_FALSE,0,0);
    delete [] attrs0;
    //---------------------------------------------
    // IBO part
    //---------------------------------------------
    GLshort * indices = new GLshort[tiler_->numOutputTiles()*6];
    indexBuffer_ = new IBO(context_);
    for (int i=0; i<tiler_->numOutputTiles(); i++) {
        int offset=i*4;
        indices[i*6+0] = offset+0;
        indices[i*6+1] = offset+1;
        indices[i*6+2] = offset+2;
        indices[i*6+3] = offset+0;
        indices[i*6+4] = offset+2;
        indices[i*6+5] = offset+3;
    }
    indexBuffer_->setBufferData(indices, 6*tiler_->numOutputTiles()*sizeof(GLshort), GL_STATIC_DRAW);
    indexBuffer_->bind();
    delete [] indices;
}


} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
