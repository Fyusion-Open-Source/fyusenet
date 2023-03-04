//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep GEMM Layer mask
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
#include "../floatconversion.h"
#include "../uniformweightarray.h"
#include "../../common/logging.h"
#include "../../common/performance.h"
#include "deepgemmlayer.h"

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
 * @copydoc DeepConvLayerBase::DeepConvLayerBase
 */
DeepGEMMLayer::DeepGEMMLayer(const GPULayerBuilder & builder, int layerNumber) : DeepConvLayerBase(builder, layerNumber) {
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepGEMMLayer::cleanup() {
    shaderState_.reset();
    noBiasShaderState_.reset();
    shader_.reset();
    noBiasShader_.reset();
    DeepConvLayerBase::cleanup();
}


/**
 * @copydoc LayerBase::forward
 */
void DeepGEMMLayer::forward(uint64_t sequence) {
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
    if (tiler_->numInputTiles() <= 1) glDisable(GL_BLEND);
    else {
        glEnable(GL_BLEND);
        glBlendEquationSeparate(GL_FUNC_ADD,GL_FUNC_ADD);
        glBlendFuncSeparate(GL_ONE,GL_ONE,GL_ONE,GL_ONE);
    }
    glViewport(0, 0, viewport_[0], viewport_[1]);
    vertexArray_->bind();
    framebuffers_.at(0)->bind();
    framebuffers_.at(0)->setWriteMask();
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,inputTextures_.at(0));
    glActiveTexture(GL_TEXTURE0+DISP_TEXTURE);
    glBindTexture(GL_TEXTURE_2D,inputCoordTexture_);
    glActiveTexture(GL_TEXTURE0+WEIGHT_TEXTURE);
    glBindTexture(GL_TEXTURE_2D,weightTexture_);
    glActiveTexture(GL_TEXTURE0+BIAS_TEXTURE);
    glBindTexture(GL_TEXTURE_2D,biasTexture_);
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        if (residualTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"Residual flag configured, but no such texture found.");
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D,residualTextures_.at(0));
    }
    if (usePoints_) {
        int instances = tiler_->numInputTiles();
        int points = tiler_->numOutputTiles();
        shader_->bind(shaderState_.get());
        shader_->setUniformValue("numInputTiles",tiler_->numInputTiles());
        glDrawArrays(GL_POINTS, 0, points);
        shader_->unbind((instances > 1) ? true : false);
        if (instances > 1) {
            noBiasShader_->bind(noBiasShaderState_.get());
            noBiasShader_->setUniformValue("numInputTiles",tiler_->numInputTiles());
            glDrawArraysInstanced(GL_POINTS, 0, points, instances-1);
            noBiasShader_->unbind();
        }
    } else {
        int instances = tiler_->numInputTiles()*kernel_;
        int tris = tiler_->numOutputTiles();
        shader_->bind(shaderState_.get());
        shader_->setUniformValue("numInputTiles",tiler_->numInputTiles());
        glDrawElements(GL_TRIANGLES,tris*6,GL_UNSIGNED_SHORT,(const GLvoid *)0);
        shader_->unbind((instances > 1) ? true : false);
        if (instances > 1) {
            noBiasShader_->bind(noBiasShaderState_.get());
            noBiasShader_->setUniformValue("numInputTiles",tiler_->numInputTiles());
            glDrawElementsInstanced(GL_TRIANGLES,tris*6,GL_UNSIGNED_SHORT,(const GLvoid *)0,instances-1);
            noBiasShader_->unbind();
        }
    }
    framebuffers_.at(0)->unbind();
    vertexArray_->unbind();
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc DeepConvLayerBase::setupNetworkPolygons()
 */
void DeepGEMMLayer::setupNetworkPolygons(VAO *vao) {
    if ((width_ == 1) && (height_ == 1)) {
        usePoints_ = true;
        int offset0=0;
        float * attrs0 = new float[tiler_->numOutputTiles()*4];
        std::vector<DeepTiler::Tile> tiles = tiler_->createOutputTiles();
        DeepTiler::Tile deftex = tiler_->getDefaultTextureExtents();
        //---------------------------------------------
        // VBO parts, first the default output tiling
        //---------------------------------------------
        for (DeepTiler::Tile & tile : tiles) {
            std::pair<float,float> screen = tile.midPoint();
            std::pair<float,float> tex = deftex.midPoint();
            attrs0[offset0] = screen.first;
            attrs0[offset0 + 1] = screen.second;
            attrs0[offset0 + 2] = tex.first;
            attrs0[offset0 + 3] = tex.second;
            offset0 +=4 ;
        }
        vertexBuffer_ = new VBO(context_);
        vao->enableArray(0);
        vertexBuffer_->setBufferData(attrs0,tiler_->numOutputTiles()*4*sizeof(float),GL_STATIC_DRAW);
        vertexBuffer_->bind();
        vao->setVertexAttributeBuffer(0,4,GL_FLOAT,GL_FALSE,0,0);
        delete [] attrs0;
        //---------------------------------------------
        // Now indices for the bias texture and the row
        // indices for the convolution coeffs (y-part
        // of the convolution)...
        //---------------------------------------------
        int * attrs1 = new int[tiler_->numOutputTiles()*2];
        for (int i=0; i < tiler_->numOutputTiles(); i++) {
            attrs1[i*2+0] = i;
            attrs1[i*2+1] = i;
        }
        textureOffsets_ = new VBO(context_);
        vao->enableArray(1);
        textureOffsets_->setBufferData(attrs1,tiler_->numOutputTiles()*2*sizeof(int),GL_STATIC_DRAW);
        textureOffsets_->bind();
        vao->setVertexAttributeBuffer(1, 2, GL_INT, 0, 0);
        delete [] attrs1;
        //---------------------------------------------
        // VBO for optional residual input (to be added
        // to the output after BN/ReLU)
        //---------------------------------------------
        if (flags_ & LayerFlags::RESIDUAL_INPUT) {
            assert(residualTiler_->numOutputTiles() == residualTiler_->numInputTiles());
            float * attrs2 = new float[residualTiler_->numInputTiles()*2];
            std::vector<DeepTiler::Tile> rtiles = residualTiler_->createInputTiles(0,0,0);
            int offset2 = 0;
            for (DeepTiler::Tile tile : rtiles) {
                std::pair<float,float> tex = tile.midPoint();
                attrs2[offset2++] = tex.first;
                attrs2[offset2++] = tex.second;
            }
            residualBuffer_ = new VBO(context_);
            vao->enableArray(2);
            residualBuffer_->setBufferData(attrs2,residualTiler_->numInputTiles()*2*sizeof(float),GL_STATIC_DRAW);
            residualBuffer_->bind();
            vao->setVertexAttributeBuffer(2,2,GL_FLOAT,GL_FALSE,0,0);
            delete [] attrs2;
        }
        //---------------------------------------------------------------------------
        // Dependent texture to perform input lookup in the vertex shader. Takes care
        // of accumulating all input channels to a set of output channels and also
        // shifts the conv-window along the y direction. For each input tile one column
        // in the texture is generated with height equivalent to the kernel size.
        // Each entry in that texture contains a 2D displacement w.r.t. the input
        // texture coordinate system which takes care of the vertical convolution
        // direction...
        //---------------------------------------------------------------------------
        glGenTextures(1,&inputCoordTexture_);
        glBindTexture(GL_TEXTURE_2D,inputCoordTexture_);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        float * texdata = new float[tiler_->numInputTiles()*4];
        DeepTiler::Tile defex = tiler_->getDefaultTextureExtents();
        std::vector<DeepTiler::Tile> intiles = tiler_->createInputTiles(0,0);
        int offset3 = 0;
        for (DeepTiler::Tile & tile : intiles) {
            tile.toDisplacement(defex, texdata, offset3);
            tile.lowClamp(texdata, offset3 + 2);
            offset3 += 4;
        }
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, tiler_->numInputTiles(), 1, 0, GL_RGBA, GL_FLOAT, texdata);
        delete [] texdata;
    } else DeepConvLayerBase::setupNetworkPolygons(vao);
}


/**
 * @copydoc DeepConvLayerBase::compileConvolutionShaders
 */
void DeepGEMMLayer::compileConvolutionShaders(const char *preproc) {
    char finalpreproc[1024+80] = {0};
    strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
    // NOTE (mw) only add residual on the first pass (the shader preprocessing masks out the residual flag for the deepconv layers)
    if (flags_ & LayerFlags::RESIDUAL_INPUT) strncat(finalpreproc,"#define USE_RESIDUAL\n", sizeof(finalpreproc) - strlen(finalpreproc) - 1);
    shader_ = compileShaderPair("shaders/deep/deepconv1x1_tiled.vert","shaders/deep/deepconv1x1_tiled.frag", finalpreproc, typeid(this));
    shaderPostprocessing(shader_);
    shaderState_ = initShader(shader_);
    strncpy(finalpreproc, preproc, sizeof(finalpreproc)-1);
    strncat(finalpreproc,"#define INSTANCE_OFFSET 1\n#define NO_BIAS\n", sizeof(finalpreproc) - strlen(finalpreproc) - 1);
    noBiasShader_ = compileShaderPair("shaders/deep/deepconv1x1_tiled.vert","shaders/deep/deepconv1x1_tiled.frag",finalpreproc,typeid(this));
    shaderPostprocessing(noBiasShader_);
    noBiasShaderState_ = initShader(noBiasShader_);
}


/**
 * @brief Create shader state for supplied shader
 *
 * @param shader Shader to create a uniform state object for
 *
 * @return Shared pointer to UniformState object that maps values to the uniforms of a shder
 */
unistateptr DeepGEMMLayer::initShader(programptr shader) {
    unistateptr state = UniformState::makeShared(shader);
    if (!GLInfo::hasBinding()) {
        state->setUniformValue("inputLayer0",0);
        state->setUniformValue("residualLayer0",1,true);
        state->setUniformValue("inputDisplacements",DISP_TEXTURE);
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
