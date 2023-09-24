//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep tanh Layer
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "deeptanhlayer.h"
#include "../../gl/glexception.h"
#include "../../gl/glinfo.h"
#include "../../common/logging.h"
#include "deeptiler.h"

namespace fyusion::fyusenet::gpu::deep {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
DeepTanhLayer::DeepTanhLayer(const GPULayerBuilder & builder, int layerNumber) :
      DeepSigmoidLayer(builder, layerNumber) {
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc DeepFunctionLayer::setupShaders
 */
void DeepTanhLayer::setupShaders() {
    char preproc[1024] = {0};
    preprocessor_.generatePreprocessorPreamble(flags_, preproc, sizeof(preproc)-1);
    shader_ = compileShaderPair("shaders/deep/deepdefault.vert", "shaders/deep/deeptanh.frag", preproc, typeid(this));
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


} // fyusion::fyusenet::gpu::deep namespace

// vim: set expandtab ts=4 sw=4:
