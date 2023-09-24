//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Image-Patch Extraction Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/glexception.h"
#include "../../common/logging.h"
#include "deeptiler.h"
#include "deepextractimgpatches.h"

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
DeepExtractImagePatches::DeepExtractImagePatches(const gpu::ImgExtractLayerBuilder & builder, int layerNumber) : DeepLayerBase((const GPULayerBuilder &)builder, layerNumber) {
    if (flags_ & LayerFlags::RESIDUAL_INPUT) THROW_EXCEPTION_ARGS(FynException,"This layer does not support residual input");
    vertexBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    vertexArray_ = nullptr;
    shader_ = nullptr;
    window_ = builder.window_;
    if (width_ % window_) FNLOGE("Width %d is not divisable by window size %d",width_,window_);
    if (height_ % window_) FNLOGE("Height %d is not divisable by window size %d",height_,window_);
}



/**
 * @copydoc LayerBase::setup
 */
void DeepExtractImagePatches::setup() {
    vertexArray_ = new VAO(context_);
    vertexArray_->bind();
    setupNetworkPolygons(vertexArray_);
    vertexArray_->unbind();
    setupShaders();
    setupFBOs();
    valid_=true;
}



/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepExtractImagePatches::cleanup() {
    // reset shaders here because the GL context is bound here (in case no cache is used)
    shader_.reset();
    delete vertexArray_;
    delete vertexBuffer_;
    delete indexBuffer_;
    vertexBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    vertexArray_ = nullptr;
    DeepLayerBase::cleanup();
}



/**
 * @copydoc LayerBase::forward
 */
void DeepExtractImagePatches::forward(uint64_t sequenceNo, StateToken * state) {
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
    glViewport(0, 0, viewport_[0], viewport_[1]);
    vertexArray_->bind();
    framebuffers_.at(0)->bind();
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);         // this is to instruct the tile-engine that we don't need the old tile-content
    shader_->bind(shaderState_.get());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    glDrawElements(GL_TRIANGLES,6*tiler_->numOutputTiles(),GL_UNSIGNED_SHORT,(const GLvoid *)0);
    framebuffers_.at(0)->unbind();
    vertexArray_->unbind();
    shader_->unbind();
}


/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> DeepExtractImagePatches::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0, tiler_->getInputTextureWidth(), tiler_->getInputTextureHeight(),
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE).dataOrder(BufferSpec::order::GPU_DEEP));
    return result;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> DeepExtractImagePatches::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0 ,0, viewport_[0], viewport_[1],
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_DEST).dataOrder(BufferSpec::order::GPU_DEEP));
    return result;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Setup/compile shaders that implement the actual layer functionality
 *
 * This function obtains required shaders from the resource system, compiles/caches these shaders
 * and performs base initializations on them.
 */
void DeepExtractImagePatches::setupShaders() {
    char preproc[1024] = {0};
    preprocessor_.generatePreprocessorPreamble(flags_, preproc, sizeof(preproc)-1);
    shader_ = compileShaderPair("shaders/deep/deepimgpatch.vert","shaders/deep/deepimgpatch.frag",preproc,typeid(this));
    try {
        shader_->bindAttributeLocation("attributes0",0);
        shader_->bindAttributeLocation("attributes1",1);
        shader_->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    shaderState_ = UniformState::makeShared(shader_);
    shaderState_->setUniformValue("inputLayer",0,true);
    shaderState_->setUniformValue("window",window_);
}


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
void DeepExtractImagePatches::setupNetworkPolygons(VAO *vao) {
    std::vector<DeepTiler::Tile> itiles = tiler_->createInputTiles(0,0);
    std::vector<DeepTiler::Tile> otiles = tiler_->createOutputTiles();
    std::unique_ptr<GLfloat []> verts(new GLfloat[otiles.size()*4*2]);
    std::unique_ptr<GLint []> texoffsets(new GLint[otiles.size()*4*4]);
    GLint * toptr = texoffsets.get();
    int vbooffset=0,posoffset=0;
    int mod = tiler_->numInputTiles();
    int xpix = 0, ypix = 0;
    for (size_t ot=0 ; ot < otiles.size(); ot++) {
        const DeepTiler::Tile& otile = otiles.at(ot);
        int it = ot % mod;
        if ((it==0)&&(ot>0)) {
            xpix++;
            if (xpix>=window_) {
                xpix=0;
                ypix++;
            }
        }
        const DeepTiler::Tile& itile = itiles.at(it);
        otile.toFloatVec(verts.get(),vbooffset,2);
        for (int i=0;i<4;i++) {
            toptr[posoffset+i*4+0]=otile.imageCoords_[0];
            toptr[posoffset+i*4+1]=otile.imageCoords_[1];
            toptr[posoffset+i*4+2]=itile.imageCoords_[0]+xpix;
            toptr[posoffset+i*4+3]=itile.imageCoords_[1]+ypix;
        }
        vbooffset+=2*4;
        posoffset+=4*4;
    }
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(verts.get(),otiles.size()*2*4*sizeof(GLfloat),GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0,2,GL_FLOAT,GL_FALSE,0,0);
    positionBuffer_ = new VBO(context_);
    vao->enableArray(1);
    positionBuffer_->setBufferData(toptr,otiles.size()*4*4*sizeof(GLint),GL_STATIC_DRAW);
    positionBuffer_->bind();
    vao->setVertexAttributeBuffer(1,4,GL_INT,0,0);
    std::unique_ptr<GLshort []> inds(new GLshort[otiles.size()*6]);
    GLshort *idata = inds.get();
    for (size_t i=0, offset=0 ; i < otiles.size(); i++, offset+=4) {
        idata[i*6+0] = offset+0;
        idata[i*6+1] = offset+1;
        idata[i*6+2] = offset+2;
        idata[i*6+3] = offset+0;
        idata[i*6+4] = offset+2;
        idata[i*6+5] = offset+3;
    }
    indexBuffer_ = new IBO(context_);
    indexBuffer_->setBufferData(idata,6*tiler_->numOutputTiles()*sizeof(GLshort),GL_STATIC_DRAW);
    indexBuffer_->bind();
}


} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
