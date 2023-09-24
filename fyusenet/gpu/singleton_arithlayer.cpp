//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Singleton Arithmetic Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "singleton_arithlayer.h"
#include "../gl/glexception.h"
#include "../gl/glinfo.h"

namespace fyusion {
namespace fyusenet {
namespace gpu {
//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
SingletonArithmeticLayer::SingletonArithmeticLayer(const SingletonArithLayerBuilder & builder, int layerNumber):FunctionLayer((GPULayerBuilder &)builder, layerNumber) {
    assert(builder.type_ != LayerType::ILLEGAL);
    optype_ = builder.opType_;
    operand_ = builder.operand_;
    if (builder.getFlags() & LayerFlags::POST_BATCHNORM) THROW_EXCEPTION_ARGS(FynException,"Batchnorm not supported by this layer");
}


/**
 * @copydoc GPULayerBase::cleanup
 */
void SingletonArithmeticLayer::cleanup() {
    for (int i=0; i < maxRenderTargets_; i++) shaders_[i].reset();
    currentShader_ = nullptr;
    FunctionLayer::cleanup();
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc FunctionLayer::beforeRender
 */
void SingletonArithmeticLayer::beforeRender() {
    currentShader_ = nullptr;
}


/**
 * @copydoc FunctionLayer::afterRender
 */
void SingletonArithmeticLayer::afterRender() {
    if (currentShader_) currentShader_->unbind();
    currentShader_ = nullptr;
}


/**
 * @copydoc FunctionLayer::renderChannelBatch
 */
void SingletonArithmeticLayer::renderChannelBatch(int outPass,int numRenderTargets,int texOffset) {
    assert(inputTextures_.size() == outputTextures_.size());
    for (int tex=0; tex < numRenderTargets; tex++) {
        glActiveTexture(GL_TEXTURE0+tex);
        glBindTexture(GL_TEXTURE_2D,inputTextures_.at(tex+texOffset));
    }
    if (currentShader_ != shaders_[numRenderTargets-1].get()) {
        if (currentShader_) currentShader_->unbind(true);
        currentShader_ = shaders_[numRenderTargets-1].get();
        currentShader_->bind(shaderStates_[numRenderTargets-1].get());
    }
    glDrawElements(GL_TRIANGLES,6,GL_UNSIGNED_SHORT,(const GLvoid *)0);
}


/**
 * @copydoc FunctionLayer::setupShaders
 */
void SingletonArithmeticLayer::setupShaders() {
    char preproc[1024] = {0};
    const char *opname = nullptr;
    switch (optype_) {
        case ArithType::ADD:
            opname = "ADD";
            break;
        case ArithType::SUB:
            opname = "SUB";
            break;
        case ArithType::MUL:
            opname = "MUL";
            break;
        case ArithType::DIV:
            opname = "DIV";
            break;
    }
    for (int i=1; i<=maxRenderTargets_; i++) {
        snprintf(preproc, sizeof(preproc), "#define NUM_LANES %d\n#define ARITH_OP_%s\n", i, opname);
        preprocessor_.generatePreprocessorPreamble(flags_, preproc, sizeof(preproc)-strlen(preproc)-1);
        shaders_[i-1] = compileShader(preproc);
        shaderStates_[i-1] = initShader(shaders_[i-1],i);
    }
}


/**
 * @brief Compile shader using supplied preprocessor macros
 *
 * @param preproc String with preprocessor macros to be used in the shader
 *
 * @return Shared pointer to compiled (and linked) shader program
 */
programptr SingletonArithmeticLayer::compileShader(const char *preproc) {
    programptr shader = compileShaderPair("shaders/default.vert","shaders/singleton_arith.frag",preproc,typeid(this));
    try {
        shader->bindAttributeLocation("attributes0",0);
        shader->link();
    } catch (GLException& ex) {
        FNLOGE("Cannot link shader for layer %s",getName().c_str());
        throw;
    }
    return shader;
}


/**
 * @brief Initialize uniform shader state to supplied shader program
 *
 * @param shader Compiled and linked shader program
 *
 * @param renderTargets Number of render targets (for multi-render targets) that are used by the
 *                      shader
 *
 * @return Shared pointer to UniformState instance that holds the static uniforms for the supplied
 *         shader
 */
unistateptr SingletonArithmeticLayer::initShader(programptr shader,int renderTargets) {
    unistateptr state = UniformState::makeShared(shader);
    for (int i=0;i<renderTargets;i++) {
        char var[128];
        snprintf(var,sizeof(var),"inputLayer%d",i);
        state->setUniformValue(var,i);
    }
    state->setUniformValue("operand",operand_);
    return state;
}



} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
