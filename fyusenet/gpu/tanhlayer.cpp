//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Isolated/Explicit tanh Layer Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "tanhlayer.h"
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
 * @copydoc GPULayerBase::GPULayerBase
 */
TanhLayer::TanhLayer(const GPULayerBuilder & builder, int layerNumber) : SigmoidLayer(builder, layerNumber) {
    if (builder.getFlags() & LayerFlags::POST_BATCHNORM) THROW_EXCEPTION_ARGS(FynException,"Batchnorm not supported fo this layer");
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @copydoc FunctionLayer::setupShaders
 */
void TanhLayer::setupShaders() {
    char preproc[1024] = {0};
    for (int i=1; i <= maxRenderTargets_; i++) {
        snprintf(preproc, sizeof(preproc), "#define NUM_LANES %d\n", i);
        handlePreprocFlags(flags_, preproc, sizeof(preproc) - strlen(preproc)-1);
        shaders_[i-1] = compileShaderPair("shaders/default.vert", "shaders/tanh.frag", preproc, typeid(this));
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

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
