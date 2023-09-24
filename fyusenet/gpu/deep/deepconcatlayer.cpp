//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Concatenation Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../common/logging.h"
#include "../../gl/glexception.h"
#include "deeptiler.h"
#include "deepconcatlayer.h"

namespace fyusion::fyusenet::gpu::deep {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------



/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase
 */
DeepConcatLayer::DeepConcatLayer(const ConcatLayerBuilder & builder) : DeepConcatLayer(builder, builder.number_) {
}

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder &, int)
 */
DeepConcatLayer::DeepConcatLayer(const ConcatLayerBuilder & builder, int layerNumber) : DeepLayerBase((const GPULayerBuilder &)builder) {
    // TODO (mw) support individual activation types on different concatenation inputs
    int relucnt = 0;
    for (auto it = builder.inputs_.begin(); it != builder.inputs_.end(); ++it) {
        addInput((*it).channels,(*it).padding);
        if ((*it).flags & LayerFlags::PRE_RELU) relucnt++;
    }
    if (relucnt == (int)builder.inputs_.size()) flags_ |= LayerFlags::PRE_RELU;
    else if (relucnt > 0) FNLOGE("WARNING: reLU/non-reLU concats are not supported yet");
}


/**
 * @brief Deallocate (CPU) resources
 */
DeepConcatLayer::~DeepConcatLayer() {
    for (DeepTiler *t : inputTilers_) {
        delete t;
    }
    inputTilers_.clear();
}


/**
 * @copydoc ConcatLayer::setup
 */
void DeepConcatLayer::setup() {
    vertexArray_ = new VAO(context_);
    vertexArray_->bind();
    setupNetworkPolygons(vertexArray_);
    vertexArray_->unbind();
    setupShaders();
    setupFBOs();
    valid_ = true;
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void DeepConcatLayer::cleanup() {
    delete positionBuffer_;
    delete texCompBuffer_;
    delete texShiftBuffer_;
    delete texCoord0Buffer_;
    delete texCoord1Buffer_;
    delete indexBuffer_;
    delete vertexArray_;
    // reset shaders here because the GL context is bound here (in case no cache is used)
    shader_.reset();
    positionBuffer_ = nullptr;
    texCoord0Buffer_ = nullptr;
    texCoord1Buffer_ = nullptr;
    texCompBuffer_ = nullptr;
    texShiftBuffer_ = nullptr;
    indexBuffer_ = nullptr;
    vertexArray_ = nullptr;
    DeepLayerBase::cleanup();
}


/**
 * @brief Execute layer
 *
 * @param sequenceNo Sequence number (\b must be strictly increasing)
 * @param state Pointer to optional StateToken object that encapsulates per-run state information
 *
 * This function executes the layer and performs the actual concatenation of the input textures to
 * an output texture. In order to save on rendering passes, the implementation uses up to 4 input
 * textures in parallel to perform the concatenation.
 */
void DeepConcatLayer::forward(uint64_t sequenceNo, StateToken * state) {
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
    glClearColor(0.0f,0.0f,0.0f,0.0f);
    glViewport(0,0,viewport_[0],viewport_[1]);
    vertexArray_->bind();
    framebuffers_.at(0)->bind();
    framebuffers_.at(0)->setWriteMask();
    glClear(GL_COLOR_BUFFER_BIT);         // this is to instruct the tile-engine that we don't need the old tile-content
    shader_->bind(shaderState_.get());
    for (RenderPassTexEnv env : passEnvironments_) {
        for (int i=0; i < env.numTextures_; i++) {
            glActiveTexture(GL_TEXTURE0+i);
            glBindTexture(GL_TEXTURE_2D,inputTextures_.at(env.textureIndices_[i]));
            //FNLOGI("Pass %d: texunit%d = %d",pass,i,env.TextureIndices[i]);
        }
        shader_->setMappedUniformValue(UNIFORM_NUMTEX,env.numTextures_);
        //FNLOGI("Setting %d textures to shader and elemoffset is %d",env.NumTextures,env.ElementOffset);
        glDrawElements(GL_TRIANGLES,6*env.numElements_,GL_UNSIGNED_SHORT,(const GLvoid *)(env.elementOffset_*sizeof(short)));
    }
    shader_->unbind();
    framebuffers_.at(0)->unbind();
    vertexArray_->unbind();
}


/**
 * @brief Add input tensor (channel) shape / padding for concatenation
 *
 * @param inputChannels Number of input channels for the input tensor to add
 * @param inputPadding Padding for the input tensor
 */
void DeepConcatLayer::addInput(int inputChannels, int inputPadding) {
    inputTilers_.push_back(new DeepTiler(LayerType::CONCAT,width_,height_,inputChannels,inputChannels,1.0f,1.0f,inputPadding,inputPadding,1,1,1,1));
}




/**
 * @copydoc LayerBase::getRequiredInputBuffers
 */
std::vector<BufferSpec> DeepConcatLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    for (int i=0; i < (int)inputTilers_.size(); i++) {
        result.push_back(BufferSpec(0, i, inputTilers_.at(i)->getViewportWidth(), inputTilers_.at(i)->getViewportHeight(),
                                    TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                    BufferSpec::CONCAT_SOURCE).dataOrder(BufferSpec::order::GPU_DEEP));
    }
    return result;
}


/**
 * @copydoc LayerBase::getRequiredOutputBuffers
 */
std::vector<BufferSpec> DeepConcatLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0, viewport_[0], viewport_[1],
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::CONCAT_DEST).dataOrder(BufferSpec::order::GPU_DEEP));
    return result;
}


/**
 * @copydoc LayerBase::numInputPorts
 */
int DeepConcatLayer::numInputPorts() const {
    return (int)inputTilers_.size();
}


/**
 * @copydoc LayerBase::getPortChannelIndex
 */
int DeepConcatLayer::getPortChannelIndex(int port) const {
    return port;
}

/**
 * @copydoc LayerBase::numInputChannels
 */
int DeepConcatLayer::numInputChannels(int port) const {
    if (port >= numInputPorts()) THROW_EXCEPTION_ARGS(FynException,"Illegal input port %d specified",port);
    return inputTilers_.at(port)->getInputChannels();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Compile shaders that implement the actual layer functionality
 *
 * This function obtains required shaders from the resource system, compiles/caches these shaders
 * and performs base initializations on them.
 */
void DeepConcatLayer::setupShaders() {
    char preproc[1024] = {0};
    preprocessor_.generatePreprocessorPreamble(flags_, preproc, sizeof(preproc)-1);
    shader_ = compileShaderPair("shaders/deep/deepconcat.vert","shaders/deep/deepconcat.frag",preproc,typeid(this));
    try {
        shader_->bindAttributeLocation("posAttributes",0);
        shader_->bindAttributeLocation("texAttrs0",1);
        shader_->bindAttributeLocation("texAttrs1",2);
        shader_->bindAttributeLocation("texCompAttrs",3);
        shader_->bindAttributeLocation("texShiftAttrs",4);
        shader_->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    shaderState_ = UniformState::makeShared(shader_);
    shaderState_->setUniformValue("inputLayer0",0);
    shaderState_->setUniformValue("inputLayer1",1);
    shaderState_->setUniformValue("inputLayer2",2);
    shaderState_->setUniformValue("inputLayer3",3);
    shader_->mapUniformLocation("numTextures",UNIFORM_NUMTEX);
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
void DeepConcatLayer::setupNetworkPolygons(VAO *vao) {
    std::vector<DeepTiler::Tile> outtiles = tiler_->createOutputTiles();
    std::vector<DeepTiler::Tile> intiles = inputTilers_.at(0)->createInputTiles(0,0,0);
    for (int i=1; i < (int)inputTilers_.size(); i++) {
        std::vector<DeepTiler::Tile> add = inputTilers_.at(i)->createInputTiles(0,0,i);
        intiles.insert(intiles.end(),add.begin(),add.end());
    }
    // NOTE (mw) we don't care if we allocate a bit too much memory here
    float *posattr = new float[2*4*intiles.size()];
    int *texshifts = new int[4*4*intiles.size()];
    int *texcomps = new int[4*4*intiles.size()];
    float *texattr0 = new float[4*4*intiles.size()];
    float *texattr1 = new float[4*4*intiles.size()];
    memset(posattr,0,2*4*intiles.size()*sizeof(float));
    memset(texshifts,0,4*4*intiles.size()*sizeof(int));
    memset(texcomps,0,4*4*intiles.size()*sizeof(int));
    memset(texattr0,0,4*4*intiles.size()*sizeof(float));
    memset(texattr1,0,4*4*intiles.size()*sizeof(float));
    int inindex = 0;
    int outindex = 0;
    int posoffset = 0;
    int texoffset = 0;
    int elemoffset = 0;

    auto addoutput = [&](const DeepTiler::Tile& tile,RenderPassTexEnv& env) {
        if (env.numElements_ > env.outputs_) {
            env.outputs_++;
            tile.toFloatVec(posattr,posoffset);
            texoffset += 4*4;
            posoffset += 2*4;
            elemoffset += 6;
        }
        outindex++;
    };

    auto addinput = [&](const DeepTiler::Tile& intile,int ti,int backOffset=0) {
        switch (ti) {
            case 0:
                assert((texoffset - backOffset) >= 0);
                intile.toFloatVec(texattr0, texoffset-backOffset, 4);
                break;
            case 1:
                assert((texoffset+2 - backOffset) >= 0);
                intile.toFloatVec(texattr0, texoffset+2-backOffset, 4);
                break;
            case 2:
                assert((texoffset - backOffset) >= 0);
                intile.toFloatVec(texattr1, texoffset-backOffset, 4);
                break;
            case 3:
                assert((texoffset+2 - backOffset) >= 0);
                intile.toFloatVec(texattr1, texoffset+2-backOffset, 4);
                break;
        }
    };

    RenderPassTexEnv rp;
    while ((outindex < (int)outtiles.size()) && (inindex < (int)intiles.size())) {
        DeepTiler::Tile & outtile = outtiles.at(outindex);
        DeepTiler::Tile & intile = intiles.at(inindex);
        int channels = intile.channels_;
        //----------------------------------------------------
        //
        //----------------------------------------------------
        if (rp.numTextures_ > 0) {
            if (rp.numTextures_ > 4) THROW_EXCEPTION_ARGS(FynException,"Invalid concatenation combination");
            int effchan = channels;
            if ((rp.textureIndices_[rp.numTextures_-1] != intile.textureID_) || (rp.shifts_[rp.numTextures_-1] != 0) || (channels != PIXEL_PACKING)) {
                if ((rp.channels_ % PIXEL_PACKING) != 0) {
                    effchan = std::min(PIXEL_PACKING - rp.channels_, channels);
                    rp.textureIndices_[rp.numTextures_] = intile.textureID_;
                    rp.components_[rp.numTextures_] = effchan;
                    rp.shifts_[rp.numTextures_] = 0;
                    addinput(intile,rp.numTextures_,4*4);
                    rp.numTextures_++;
                    rp.channels_ += effchan;
                } else {
                    passEnvironments_.push_back(rp);
                    rp.init(elemoffset,channels,0,intile.textureID_);    // start new environment
                    addinput(intile,0);
                    addoutput(outtile,rp);
                }
            } else {
                rp.numElements_++;
                rp.channels_ += effchan;
                addinput(intile,0);
                addoutput(outtile,rp);
            }
            channels -= effchan;
            if ((rp.numTextures_ > 1) && (rp.channels_ == PIXEL_PACKING)) {
                passEnvironments_.push_back(rp);
                if (channels > 0) {
                    rp.init(elemoffset,channels,intile.channels_-channels,intile.textureID_);    // start new environment on overflow channels
                    addinput(intile,0);
                    addoutput(outtile,rp);
                }
                else rp.clear();
            }
        } else {
            if (rp.numTextures_ == 0) {
                rp.init(elemoffset,intile.channels_,0,intile.textureID_);
                addinput(intile,0);
                if (intile.channels_ == PIXEL_PACKING) addoutput(outtile,rp);
            }
        }
        inindex++;
        if ((inindex >= (int)intiles.size()) && (rp.numTextures_ > 0)) {
            passEnvironments_.push_back(rp);
            addoutput(outtile,rp);
        }
    }
    int shiftoffset = 0;
    for (auto env : passEnvironments_) {
        for (int elem = 0;elem<env.numElements_;elem++) {
            for (int i=0;i<4;i++) {
                for (int j=0;j<4;j++) {
                    texshifts[shiftoffset+i*4+j] = env.shifts_[j];
                    texcomps[shiftoffset+i*4+j] = env.components_[j];
                }
            }
            shiftoffset += 4*4;
        }
    }
    // vertex positions
    positionBuffer_ = new VBO(context_);
    vao->enableArray(0);
    positionBuffer_->setBufferData(posattr,(GLsizei)(posoffset*sizeof(float)),GL_STATIC_DRAW);
    positionBuffer_->bind();
    vao->setVertexAttributeBuffer(0,2,GL_FLOAT,GL_FALSE,0,0);

    // input textures 0, 1
    texCoord0Buffer_ = new VBO(context_);
    vao->enableArray(1);
    texCoord0Buffer_->setBufferData(texattr0,(GLsizei)(texoffset*sizeof(float)),GL_STATIC_DRAW);
    texCoord0Buffer_->bind();
    vao->setVertexAttributeBuffer(1,4,GL_FLOAT,GL_FALSE,0,0);

    // input textures 2, 3
    texCoord1Buffer_ = new VBO(context_);
    vao->enableArray(2);
    texCoord1Buffer_->setBufferData(texattr1,(GLsizei)(texoffset*sizeof(float)),GL_STATIC_DRAW);
    texCoord1Buffer_->bind();
    vao->setVertexAttributeBuffer(2,4,GL_FLOAT,GL_FALSE,0,0);

    // # texture components
    texCompBuffer_ = new VBO(context_);
    vao->enableArray(3);
    texCompBuffer_->setBufferData(texcomps,(GLsizei)(shiftoffset*sizeof(int)),GL_STATIC_DRAW);
    texCompBuffer_->bind();
    vao->setVertexAttributeBuffer(3,4,GL_INT,0,0);

    // shift values
    texShiftBuffer_ = new VBO(context_);
    vao->enableArray(4);
    texShiftBuffer_->setBufferData(texshifts,(GLsizei)(shiftoffset*sizeof(int)),GL_STATIC_DRAW);
    texShiftBuffer_->bind();
    vao->setVertexAttributeBuffer(4,4,GL_INT,0,0);

    delete [] posattr;
    delete [] texshifts;
    delete [] texcomps;
    delete [] texattr0;
    delete [] texattr1;
    //----------------------------------------------------
    // IBO part
    //----------------------------------------------------
    auto * indices = new GLshort[outtiles.size()*6];
    for (int i=0; i < (int)outtiles.size(); i++) {
        int offset=i*4;
        indices[i*6+0] = (GLshort)(offset + 0);
        indices[i*6+1] = (GLshort)(offset + 1);
        indices[i*6+2] = (GLshort)(offset + 2);
        indices[i*6+3] = (GLshort)(offset + 0);
        indices[i*6+4] = (GLshort)(offset + 2);
        indices[i*6+5] = (GLshort)(offset + 3);
    }
    indexBuffer_ = new IBO(context_);
    indexBuffer_->setBufferData(indices,6*outtiles.size()*sizeof(GLshort),GL_STATIC_DRAW);
    indexBuffer_->bind();
    delete [] indices;
}



} // fyusion::fyusenet::gpu::deep namespace

// vim: set expandtab ts=4 sw=4:
