//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Isolated/Explicit SiLU Layer                                                (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "silulayer.h"
#include "../gl/glexception.h"
#include "../gl/glinfo.h"

namespace fyusion::fyusenet::gpu {

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
SiLULayer::SiLULayer(const GPULayerBuilder & builder, int layerNumber) : SigmoidLayer(builder, layerNumber) {
    if (builder.getFlags() & LayerFlags::POST_BATCHNORM) THROW_EXCEPTION_ARGS(FynException,"Batchnorm not supported fo this layer");
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc FunctionLayer::setupShaders
 */
void SiLULayer::setupShaders() {
    char preproc[1024] = {0};
    for (int i=1; i <= maxRenderTargets_; i++) {
        snprintf(preproc, sizeof(preproc), "#define NUM_LANES %d\n", i);
        preprocessor_.generatePreprocessorPreamble(flags_, preproc, sizeof(preproc) - strlen(preproc)-1);
        // NOTE (mw) we use the activation.inc for the activation and employ the scaling shader as proxy
        shaders_[i-1] = compileShaderPair("shaders/default.vert", "shaders/scaling.frag", preproc, typeid(this));
        try {
            shaders_[i-1]->bindAttributeLocation("attributes0",0);
            shaders_[i-1]->link();
        } catch (GLException & ex) {
            FNLOGE("Cannot link shader for layer %s",getName().c_str());
            throw;
        }
        shaderStates_[i-1] = UniformState::makeShared(shaders_[i-1]);
        for (int j=0; j < i; j++) {
            snprintf(preproc, sizeof(preproc), "inputLayer%d", j);
            shaderStates_[i-1]->setUniformValue(preproc,j);
        }
    }
}

} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:
