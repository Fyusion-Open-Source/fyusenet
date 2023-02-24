//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Base Class for Deep-Channel Tensor Computations
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------


#include "../../common/logging.h"
#include "../../gl/glinfo.h"
#include "../../gl/fbo.h"
#include "deeptiler.h"
#include "deeplayerbase.h"

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
 * @copydoc GPULayerBase::GPULayerBase
 */
DeepLayerBase::DeepLayerBase(const GPULayerBuilder& builder, int layerNumber) : GPULayerBase(builder,layerNumber),
    tiler_(new DeepTiler(builder.type_, builder.width(), builder.height(), builder.in(), builder.out(), (float)builder.upsample_[0]/(float)builder.downsample_[0], (float)builder.upsample_[1]/(float)builder.downsample_[1] ,builder.inputPadding_, builder.outputPadding_, builder.downsample_[0], builder.downsample_[1], builder.upsample_[0], builder.upsample_[1])) {
    viewport_[0] = tiler_->getViewportWidth();
    viewport_[1] = tiler_->getViewportHeight();
    if (GLInfo::getGPUType() == GLInfo::ARM_MALI) {
        mali_ = true;
        std::string renderer = GLInfo::getRendererString();
        if (!renderer.empty()) {
            if (strstr(renderer.c_str(),"-T")) preG71_=true;
        }
    }
    if (flags_ & LayerFlags::RESIDUAL_INPUT) {
        DeepTiler restiler(LayerType::RESIDUAL,builder.width(),builder.height(),builder.in(),builder.out(),(float)builder.upsample_[0]/(float)builder.downsample_[0],(float)builder.upsample_[1]/(float)builder.downsample_[1],0,builder.residualPadding_,builder.downsample_[0],builder.downsample_[1],builder.upsample_[0],builder.upsample_[1]);
        residualViewport_[0] = restiler.getViewportWidth();
        residualViewport_[1] = restiler.getViewportHeight();
    }
}


/**
 * @copydoc GPULayerBase::~GPULayerBase
 */
DeepLayerBase::~DeepLayerBase() {
    delete tiler_;
    tiler_ = nullptr;
}



/**
 * @copydoc LayerBase::writeResult
 */
void DeepLayerBase::writeResult(const char *fileName,bool includePadding) {
#ifdef DEBUG
    int owidth = tiler_->getViewportWidth();
    int oheight = tiler_->getViewportHeight();
    float * data = new float[oheight*owidth*PIXEL_PACKING];
    int lwidth = tiler_->getOutputWidth();
    int lheight = tiler_->getOutputHeight();
    if (includePadding) {
        lwidth += 2*inputPadding_;
        lheight += 2*inputPadding_;
    }
#ifndef FYUSENET_USE_WEBGL
    FILE *out = fopen(fileName,"w");
    if (out) {
#else
    uint8_t * download = new uint8_t[lwidth * lheight * outputChannels_];
    uint8_t * downptr = download;
    if (true) {
#endif
        float * layer = new float[lwidth*lheight];
        memset(layer,0,lwidth*lheight*sizeof(float));
        int layernum=0;
        for (int fb = 0 ; fb < numFBOs(); fb++ ) {
            memset(data, 0, owidth*oheight*PIXEL_PACKING*sizeof(float));
            FBO *fbo = getFBO(fb);
            fbo->writeToMemory<float,GL_FLOAT>(data,PIXEL_PACKING,owidth*oheight*PIXEL_PACKING*sizeof(float));
            for (int ty=0; ty< tiler_->numOutputTiles(DeepTiler::VERTICAL); ty++) {
                for (int tx=0; tx < tiler_->numOutputTiles(DeepTiler::HORIZONTAL); tx++) {
                    int rem = ((outputChannels_-layernum)>PIXEL_PACKING) ? PIXEL_PACKING : outputChannels_-layernum;
                    float * in = data + ((outputPadding_+ty*(lheight+outputPadding_))*owidth + outputPadding_+tx*(lwidth+outputPadding_))*PIXEL_PACKING;
                    float * outptr = (includePadding) ? layer + (outputPadding_*lwidth)+outputPadding_ : layer;
                    for (int l=0; l < rem;l++) {
                        for (int y=0; y < lheight;y++) {
                            for (int x=0; x < lwidth;x++) {
                                outptr[x+y*lwidth] = in[(y*owidth+x)*PIXEL_PACKING+l];
                            }
                        }
#ifndef FYUSENET_USE_WEBGL
                        fwrite(layer,1,lwidth*lheight*sizeof(float),out);
#else
                        memcpy(downptr, layer, owidth * oheight * sizeof(float));
                        downptr += lwidth * lheight;
#endif
                    }
                    layernum += PIXEL_PACKING;
                }
            }
        }
        delete [] data;
        delete [] layer;
#ifndef FYUSENET_USE_WEBGL
        fclose(out);
#else
        EM_ASM({window.download($0, $1, $2);}, download, owidth * oheight * outputChannels_ * sizeof(float), fileName);
        delete [] download;
#endif
    } else FNLOGE("Cannot open file %s for writing",fileName);
#endif
}


/**
 * @copydoc GPULayerBase::copyResult
 */
void DeepLayerBase::copyResult(float *memory, bool includePadding) {
#ifdef DEBUG
    if (memory) {
        int owidth = tiler_->getViewportWidth();
        int oheight = tiler_->getViewportHeight();
        float * data = new float[oheight*owidth*PIXEL_PACKING];
        int lwidth = tiler_->getOutputWidth();
        int lheight = tiler_->getOutputHeight();
        if (includePadding) {
            lwidth += 2*inputPadding_;
            lheight += 2*inputPadding_;
        }
        int layernum=0;
        float * layer = memory;
        for (int fb = 0 ; fb < numFBOs(); fb++ ) {
            memset(data, 0, owidth*oheight*PIXEL_PACKING*sizeof(float));
            FBO *fbo = getFBO(fb);
            assert(fbo->numAttachments() == 1);
            fbo->writeToMemory<float,GL_FLOAT>(data,PIXEL_PACKING,owidth*oheight*PIXEL_PACKING*sizeof(float));
            for (int ty=0; ty < tiler_->numOutputTiles(DeepTiler::VERTICAL); ty++) {
                for (int tx=0; tx < tiler_->numOutputTiles(DeepTiler::HORIZONTAL); tx++) {
                    int rem = ((outputChannels_ - layernum) > PIXEL_PACKING) ? PIXEL_PACKING : outputChannels_ - layernum;
                    const float * in = data + ((outputPadding_ + ty*(lheight + outputPadding_))*owidth + outputPadding_ + tx*(lwidth + outputPadding_))*PIXEL_PACKING;
                    float * outptr = (includePadding) ? layer + (outputPadding_ * lwidth) + outputPadding_ : layer;
                    for (int l=0; l < rem; l++) {
                        for (int y=0; y < lheight; y++) {
                            for (int x=0; x < lwidth; x++) {
                                outptr[x+y*lwidth] = in[(y*owidth+x)*PIXEL_PACKING+l];
                            }
                        }
                        layer += lwidth*lheight;
                        outptr += lwidth*lheight;
                    }
                    layernum += PIXEL_PACKING;
                }
            }
        }
        delete [] data;
    }
#endif
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Basic shader preprocessing on source level for deep-tensor layers
 *
 * @param[inout] preproc Pointer to target pre-processor string which will be used as preprocessor
 *                       definitions with GPULayerBase::compileShaderPair
 *
 * @param maxChars Maximum available characters in the \p preproc array
 *
 * @return Remaining capacity of \p preproc buffer
 *
 * This function constructs (parts of) a preprocessor string for use in the vertex and fragment
 * shaders. It currently takes care of GPU specific definitions for ARM Mali GPUs:
 *  - definition of \c MALI preprocessor item in case an ARM Mali GPU has been found
 *  - definition of \c PRE_G71 preprocessor item in case an older Mali GPU has been found (T-series)
 */
size_t DeepLayerBase::shaderPreprocessing(char *preproc, size_t maxChars) {
    ssize_t mc = (ssize_t)handlePreprocFlags(flags_, preproc, maxChars);
    if (mali_) {
        strncat(preproc, "#define MALI\n", mc);
        mc = maxChars-strlen(preproc);  // ouch
    }
    if (preG71_) {
        strncat(preproc, "#define PRE_G71\n", mc);
        mc = maxChars-strlen(preproc);  // ouch
    }
    return (size_t)mc;
}


/**
 * @copydoc GPULayerBase::setupFBOs
 */
void DeepLayerBase::setupFBOs() {
    if (outputTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"No output texture set in layer %s",getName().c_str());
    FBO * fbo = new FBO(context_,viewport_[0],viewport_[1],outputTextures_.at(0));
    fbo->unbind();
    framebuffers_.push_back(fbo);
    outputChanged_=false;
}


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void DeepLayerBase::updateFBOs() {
    if (outputTextures_.empty()) THROW_EXCEPTION_ARGS(FynException,"No output texture set in layer %s",getName().c_str());
    if (framebuffers_.empty()) THROW_EXCEPTION_ARGS(FynException,"No framebuffers to update in layer %s",getName().c_str());
    FBO * fbo = framebuffers_.at(0);
    fbo->bind();
    fbo->updateColorAttachment(GL_COLOR_ATTACHMENT0,outputTextures_.at(0));
    fbo->unbind();
    outputChanged_=false;
}


} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
