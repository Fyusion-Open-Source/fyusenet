//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Convolutional Layer w/ 3x3 mask
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/vertexshader.h"
#include "../../gl/fragmentshader.h"
#include "../../gl/shaderprogram.h"
#include "../../gl/glinfo.h"
#include "../../gl/glexception.h"
#include "../uniformweightarray.h"
#include "../../common/logging.h"
#include "../../common/performance.h"
#include "../floatconversion.h"
#include "deepdwconvlayer3x3.h"

//-------------------------------------- Global Variables ------------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase
 */
DeepDepthwiseConvLayer3x3::DeepDepthwiseConvLayer3x3(const ConvLayerBuilder& builder,int layerNumber):DeepDepthwiseConvLayerBase(builder, layerNumber) {
    assert(inputChannels_ == outputChannels_);
    assert(builder.kernel_ == 3);
}



/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepDepthwiseConvLayer3x3::cleanup() {
    shaderState_.reset();
    noBiasShaderState_.reset();
    shader_.reset();
    noBiasShader_.reset();
    DeepDepthwiseConvLayerBase::cleanup();
}


/**
 * @copydoc LayerBase::forward
 */
void DeepDepthwiseConvLayer3x3::forward(uint64_t sequence) {
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
    glViewport(0,0,viewport_[0],viewport_[1]);
    vertexArray_->bind();
    framebuffers_.at(0)->bind();
    framebuffers_.at(0)->setWriteMask();
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    glActiveTexture(GL_TEXTURE0+WEIGHT_TEXTURE);
    glBindTexture(GL_TEXTURE_2D,weightTexture_);
    glActiveTexture(GL_TEXTURE0+BIAS_TEXTURE);
    glBindTexture(GL_TEXTURE_2D,biasTexture_);
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        if (residualTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"Residual flag configured, but no such texture found.");
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D,residualTextures_.at(0));
    }
    int tris = tiler_->numOutputTiles();
    shader_->bind(shaderState_.get());
    glDrawElements(GL_TRIANGLES,tris*6,GL_UNSIGNED_SHORT,(const GLvoid *)0);
    shader_->unbind();
    framebuffers_.at(0)->unbind();
    vertexArray_->unbind();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @copydoc DeepConvLayerBase::setupNetworkPolygons
 */
void DeepDepthwiseConvLayer3x3::setupNetworkPolygons(VAO *vao) {
    int offset0 = 0;
    float * attrs0 = new float[tiler_->numOutputTiles()*4*4];
    std::vector<DeepTiler::Tile> tiles = tiler_->createOutputTiles();
    std::vector<DeepTiler::Tile> intiles = tiler_->createInputTiles(0,0);
    //---------------------------------------------
    // VBO parts, first the default output tiling
    // combined with default input tiling...
    //---------------------------------------------
    assert(tiles.size() == intiles.size() * channelMultiplier_);
    size_t chanoffset = 0;
    for (int mult=0; mult < channelMultiplier_; mult++) {
        for (size_t t=0; t < intiles.size(); t++) {
            tiles.at(t+chanoffset).toFloatVec(attrs0,offset0,4);
            intiles.at(t).toFloatVec(attrs0,offset0+2,4);
            offset0 += 4*4;
        }
        chanoffset += intiles.size();
    }
    vertexBuffer_ = new VBO(context_);
    vao->enableArray(0);
    vertexBuffer_->setBufferData(attrs0,tiler_->numOutputTiles()*4*4*sizeof(float),GL_STATIC_DRAW);
    vertexBuffer_->bind();
    vao->setVertexAttributeBuffer(0,4,GL_FLOAT,GL_FALSE,0,0);
    delete [] attrs0;
    //---------------------------------------------
    // Now indices for the bias texture and the row
    // indices for the convolution coeffs (y-part
    // of the convolution)...
    //---------------------------------------------
    int * attrs1 = new int[tiler_->numOutputTiles()*3*4];
    memset(attrs1, 0, tiler_->numOutputTiles()*3*4*sizeof(int));
    chanoffset = 0;
    for (int i=0; i < tiler_->numOutputTiles(); i++) {
        for (int j=0; j < 4; j++) {
            attrs1[(i*4+j)*3+0] = (i % intiles.size());
            attrs1[(i*4+j)*3+1] = i;          // to be used for indexing bias texture
            attrs1[(i*4+j)*3+2] = chanoffset;
        }
        if ((i > 0) && ((i % intiles.size())==0)) chanoffset++;
    }
    textureOffsets_ = new VBO(context_);
    vao->enableArray(1);
    textureOffsets_->setBufferData(attrs1,tiler_->numOutputTiles()*3*4*sizeof(int),GL_STATIC_DRAW);
    textureOffsets_->bind();
    vao->setVertexAttributeBuffer(1,3,GL_INT,0,0);
    delete [] attrs1;
    //---------------------------------------------
    // VBO for optional residual input (to be added
    // to the output after BN/ReLU)
    //---------------------------------------------
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        assert(residualTiler_->numOutputTiles() == residualTiler_->numInputTiles());
        float * attrs2 = new float[residualTiler_->numInputTiles()*2*4];
        std::vector<DeepTiler::Tile> rtiles = residualTiler_->createInputTiles(0,0,0);
        int offset2=0;
        for (DeepTiler::Tile tile : rtiles) {
            tile.toFloatVec(attrs2,offset2,2);
            offset2 += 2*4;
        }
        residualBuffer_ = new VBO(context_);
        vao->enableArray(2);
        residualBuffer_->setBufferData(attrs2,residualTiler_->numInputTiles()*2*4*sizeof(float),GL_STATIC_DRAW);
        residualBuffer_->bind();
        vao->setVertexAttributeBuffer(2,2,GL_FLOAT,GL_FALSE,0,0);
        delete [] attrs2;
    }
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
    indexBuffer_->setBufferData(indices,6*tiler_->numOutputTiles()*sizeof(GLshort),GL_STATIC_DRAW);
    indexBuffer_->bind();
    delete [] indices;
}




/**
 * @copydoc DeepConvLayerBase::compileConvolutionShaders
 */
void DeepDepthwiseConvLayer3x3::compileConvolutionShaders(const char *preproc) {
    char finalpreproc[1024+80] = {0};
    strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
    // NOTE (mw) only add residual on the first pass (the shader preprocessing masks out the residual flag for the deepconv layers)
    if (flags_ & LayerFlags::RESIDUAL_INPUT) strncat(finalpreproc,"#define USE_RESIDUAL\n", sizeof(finalpreproc) - strlen(finalpreproc) - 1);
    shader_ = compileShaderPair("shaders/deep/deepconv_dw3x3_tiled.vert","shaders/deep/deepconv_dw3x3_tiled.frag",finalpreproc,typeid(this));
    shaderPostprocessing(shader_);
    shaderState_ = initShader(shader_);
}


/**
 * @brief Create shader state for supplied shader
 *
 * @param shader Shader to create a uniform state object for
 *
 * @return Shared pointer to UniformState object that maps values to the uniforms of a shder
 */
unistateptr DeepDepthwiseConvLayer3x3::initShader(programptr shader) {
    unistateptr state = UniformState::makeShared(shader);
    if (!GLInfo::hasBinding()) {
        state->setUniformValue("inputLayer0",0);
        state->setUniformValue("residualLayer0",1,true);
        state->setUniformValue("inputCoeffs",WEIGHT_TEXTURE);
        state->setUniformValue("biasTexture",BIAS_TEXTURE,true);
    }
    return state;
}



} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
