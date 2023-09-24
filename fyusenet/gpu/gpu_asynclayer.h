//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Asynchronous Layer Interface for GPUs (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../base/asynclayerinterface.h"
#include "gpulayerbase.h"

namespace fyusion::fyusenet::gpu {

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Interface for asynchronous GPU layers
 *
 * Note that layers deriving from that interface are not necessarily asynchronous, they just have
 * the \e option to run asynchronously. If the user sets up these layers in a synchronous, fashion
 * the asynchronicity is not used at all.
 *
 * @see gpu::UploadLayer, gpu::DownloadLayer
 */
class GPUAsyncLayer : public AsyncLayer {

 protected:

    /**
     * @brief Update input textures in dependent (receiving) layers
     *
     * @param textures New texture IDs to set in the dependent layers
     *
     * This function iterates over all receiving (dependent) layers of this upload layer and updates
     * the input texture IDs with the IDs from the current sequence number.
     */
    void updateDependencies(const std::vector<GLuint> &textures) const {
        for (int i=0; i < (int)dependencies_.size(); i++) {
            auto * tgt = dynamic_cast<GPULayerBase *>(dependencies_.at(i));
            assert(tgt);
            int chanidx = dependencyOffsets_.at(i);
            for (int ti=0; ti < (int)textures.size(); ti++) {
                tgt->updateInputTexture(textures.at(ti), chanidx + ti);
            }
        }
    }    
};


} // fyusion::fyusenet::gpu namespace

// vim: set expandtab ts=4 sw=4:

