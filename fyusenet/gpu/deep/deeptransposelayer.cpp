//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Image Transposition Layer
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
#include "deeptransposelayer.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

DeepTransposeLayer::DeepTransposeLayer(const TransposeLayerBuilder & builder, int layerNumber) : DeepLayerBase((const GPULayerBuilder &)builder, layerNumber) {
    shader_ = nullptr;
    outTiler_ = new DeepTiler(LayerType::TRANSPOSE, builder.height(), builder.width(), builder.in(), builder.out(), 1.0f, 1.0f, 0, builder.outputPadding_, 1, 1, 1, 1);
    viewport_[0] = outTiler_->getViewportWidth();
    viewport_[1] = outTiler_->getViewportHeight();
}

DeepTransposeLayer::~DeepTransposeLayer() {
    delete outTiler_;
    outTiler_ = nullptr;
}

/**
 * @copydoc GPULayerBase::setup()
 */
void DeepTransposeLayer::setup() {
    vertexArray_ = new VAO(context_);
    vertexArray_->bind();
    setupNetworkPolygons(vertexArray_);
    vertexArray_->unbind();
    setupShaders();
    setupFBOs();
    valid_ = true;
}

/**
 * @copydoc GPULayerBase::cleanup()
 */
void DeepTransposeLayer::cleanup() {
    // reset shaders here because the GL context is bound here (in case no cache is used)
    shader_.reset();
    delete vertexArray_;
    delete vertexBuffer_;
    delete indexBuffer_;
    vertexArray_ = nullptr;
    vertexBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    DeepLayerBase::cleanup();
}


void DeepTransposeLayer::forward(uint64_t sequence) {
    if (!valid_) THROW_EXCEPTION_ARGS(FynException,"Trying to invoke render() on invalid layer");
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
    glClear(GL_COLOR_BUFFER_BIT);
    vertexArray_->bind();
    shader_->bind(shaderState_.get());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTextures_.at(0));
    int quads = outTiler_->numOutputTiles();
    glDrawElements(GL_TRIANGLES, quads*6, GL_UNSIGNED_SHORT, (const GLvoid *)0);
    shader_->unbind();
    framebuffers_.at(0)->unbind();
    vertexArray_->unbind();
}


/**
 * @brief Obtain array of buffer specifications that are required for the input to this layer
 *
 * @return STL vector with specifications for each input buffer/texture
 */
std::vector<BufferSpec> DeepTransposeLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0, tiler_->getInputTextureWidth(), tiler_->getInputTextureHeight(),
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4,TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE).dataOrder(BufferSpec::order::GPU_DEEP));
    return result;
}


/**
 * @brief Obtain array of buffer specifications that are required for the output of this layer
 *
 * @return STL vector with specifications for each output buffer/texture
 */
std::vector<BufferSpec> DeepTransposeLayer::getRequiredOutputBuffers() const {
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
 * @brief Setup vertex and index buffers for rendering the transposed layer data
 *
 * @param vao Pointer to vertex array object
 */
void DeepTransposeLayer::setupNetworkPolygons(VAO *vao) {
    int offset0=0;
    float * attrs0 = new float[tiler_->numOutputTiles()*4*4];
    //---------------------------------------------
    // VBO part
    //---------------------------------------------
    std::vector<DeepTiler::Tile> otiles = outTiler_->createOutputTiles();
    std::vector<DeepTiler::Tile> itiles = tiler_->createInputTiles(0,0);
    for (int i=0; i < (int)itiles.size(); i++) {
        DeepTiler::Tile ot = otiles.at(i);
        DeepTiler::Tile it = itiles.at(i);
        ot.toFloatVec(attrs0, offset0, 4);
        it.toFloatVec(attrs0, offset0+2, 4, true);
        offset0 += 4*4;
    }
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(attrs0, outTiler_->numOutputTiles()*4*4*sizeof(float), GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
    delete [] attrs0;
    //---------------------------------------------
    // IBO part
    //---------------------------------------------
    GLshort * indices = new GLshort[outTiler_->numOutputTiles()*6];
    indexBuffer_ = new IBO(context_);
    for (int i=0; i<outTiler_->numOutputTiles(); i++) {
        int offset=i*4;
        indices[i*6+0] = offset+0;
        indices[i*6+1] = offset+1;
        indices[i*6+2] = offset+2;
        indices[i*6+3] = offset+0;
        indices[i*6+4] = offset+2;
        indices[i*6+5] = offset+3;
    }
    indexBuffer_->setBufferData(indices, 6*outTiler_->numOutputTiles()*sizeof(GLshort), GL_STATIC_DRAW);
    indexBuffer_->bind();
    delete [] indices;
}


void DeepTransposeLayer::setupShaders() {
    char preproc[1024] = {0};
    handlePreprocFlags(flags_, preproc, sizeof(preproc)-1);
    shader_ = compileShaderPair("shaders/deep/deepdefault.vert", "shaders/deep/deepdefault.frag", preproc, typeid(this));
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
