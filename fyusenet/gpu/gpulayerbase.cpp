//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Neural Network Layer Base Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstring>
#include <algorithm>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/logging.h"
#include "gpulayerbase.h"
#include "../gl/fbo.h"
#include "../gl/shaderexception.h"
#include "../gl/shaderresource.h"
#include "../gl/glexception.h"
#include "../gl/shadercache.h"
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
 * @brief Constructor
 *
 * @param builder GPU-specific layer builder that contains parameterization for the layer
 *
 * @param layerNumber Layer number that defines sequence position in execution
 *
 * @throws FynException in case the layer is initialized with invalid/unsupported parameters
 *
 * @pre GL context that this layer is supposed to be operated under must be current
 *
 * Parses basic information from the supplied \p builder, including the output viewport and the
 * GL context.
 */
GPULayerBase::GPULayerBase(const GPULayerBuilder & builder,int layerNumber) :
      fyusenet::LayerBase((const LayerBuilder &)builder, layerNumber) {
    device_ = compute_device::DEV_GPU;
    setContext(builder.context_);
    // default viewport assumption
    viewport_[0] = width_ + 2*outputPadding_;
    viewport_[1] = height_ + 2*outputPadding_;
    residualViewport_[0] = width_ + 2*residualPadding_;
    residualViewport_[1] = height_ + 2*residualPadding_;
}


/**
 * @brief Destructor
 *
 * Deallocates CPU resources consumed by the layer
 *
 * @note In order to release the GL/GPU resources of this layer, use the cleanup() method
 *
 * @see cleanup
 */
GPULayerBase::~GPULayerBase() {
    assert(!valid_);
    if (valid_) {
        // TODO (mw) maybe throw an exception here
        FNLOGE("Cleanup was not called on layer %s, this may leak OpenGL memory", name_.c_str());
    }
    inputTextures_.clear();
    outputTextures_.clear();
    residualTextures_.clear();
    if (framebuffers_.size() > 0) {
        FNLOGE("Framebuffers not cleaned, this might leak OpenGL context memory (rem=%d)",(int)framebuffers_.size());
    }
}


/**
 * @brief Deallocates GL resources used by this layer
 *
 * @pre GL context that is used by this layer is bound to the calling thread
 *
 * @note The main reason why this is not simply done in the destructor of the class, is due to
 *       GPU usage, in particular the usage of OpenGL on GPUs. For deallocation of GL resources,
 *       the \e right context must be bound to the calling thread. In order to prevent API users
 *       from just relying on destructors on deletion of a network, this method reminds them
 *       to make sure that a cleanup is called in the right thread. You could do that with
 *       destructors too, but people (read: me) are just not used to that.
 */
void GPULayerBase::cleanup() {
    valid_ = false;
    // do NOT _delete_ input and residual textures here, they are managed by the BufferManager
    inputTextures_.clear();
    residualTextures_.clear();
    for (FBO *fbo : framebuffers_) {
        delete fbo;
    }
    framebuffers_.clear();
}


/**
 * @brief Check if an input texture has been assigned to the specified index
 *
 * @param channelIndex Index within the set of input textures of this layer
 *
 * @retval true if a texture was assigned to the specified \p channelIndex
 * @retval false otherwise
 *
 * This implementation checks if an input texture has been added to the specified \p channelIndex
 * via the #addInputTexture method.
 *
 * As a layer can have several input ports and each port may consist of more than one texture,
 * the \p channelIndex specifies a flattened offset into this list. For example, assume that a layer
 * has 2 input ports, where the first port has 24 channels and the second port has 32 channels.
 * This equals to 6 textures on the first port and 8 textures on the second port (each texture
 * aggregates 4 channels). A channel index of 4 will therefore be channels 16 to 19 (inclusive) of
 * the first port and a channel index of 12 will be channels 24..27 (inclusive) of the second port.
 *
 * @note The \p channelIndex parameter here is not equivalent to a \e port as it is used with a set
 *       of textures, which are not necessarily equivalent to the number of buffers/tensors that
 *       represent the port interface of a layer. A port can have multiple textures associated with
 *       it.
 *
 * @warning This default implementation only copes with a single input port (port 0). For multi-port
 *          inputs, please override this method.
 *
 * @see BufferManager::connectLayers, BufferManager::connectGPULayers, addInputTexture, getInputTexture
 */
bool GPULayerBase::hasInputTexture(int channelIndex) const {
    if ((channelIndex < 0)||(channelIndex >= (int)inputTextures_.size())) return false;
    return true;
}


/**
 * @brief Retrieve raw OpenGL texture handle of input texture at specified index
 *
 * @param channelIndex Index within the set of input textures of this layer
 *
 * @return Raw OpenGL texture handle of the input texture connected to the specified \p channelIndex
 *
 * This implementation retrieves the raw OpenGL handle of the specified input texture, which must
 * have been previously added to the specified \p channelIndex via the #addInputTexture method.
 *
 * As a layer can have several input ports and each port may consist of more than one texture,
 * the \p channelIndex specifies a flattened offset into this list. For example, assume that a layer
 * has 2 input ports, where the first port has 24 channels and the second port has 32 channels.
 * This equals to 6 textures on the first port and 8 textures on the second port (each texture
 * aggregates 4 channels). A channel index of 4 will therefore be channels 16 to 19 (inclusive) of
 * the first port and a channel index of 12 will be channels 24..27 (inclusive) of the second port.
 *
 * @throws FynException in case an illegal \p channelIndex has been provided
 *
 * @note The \p channelIndex parameter here is not equivalent to a \e port as it is used with a set
 *       of textures, which are not necessarily equivalent to the number of buffers/tensors that
 *       represent the port interface of a layer. A port can have multiple textures associated with
 *       it.
 *
 * @see addInputTexture, hasInputTexture
 */
GLuint GPULayerBase::getInputTexture(int channelIndex) const {
    if ((channelIndex < 0)||(channelIndex >= (int)inputTextures_.size())) {
        THROW_EXCEPTION_ARGS(FynException,"Illegal index %d for input texture (size is %d)",channelIndex,inputTextures_.size());
    }
    return inputTextures_.at(channelIndex);
}


/**
 * @brief Add texture to the list of residual textures
 *
 * @param textureID Raw OpenGL texture handle to be \e appended to the list of residual textures
 * @param channelIndex Index within the set of input textures of this layer
 *
 * This function appends the specified texture handle to the list of residual textures for this
 * layer. In contrast to the input texture, the residual textures are restricted to one "port"
 * as they are simply added to the output of the layer. For this reason, no index needs to be
 * specified, but the caller has to be aware that this call is an "append" operation.
 *
 * @note This class does not take ownership over the supplied texture, it is up to the caller to
 *       maintain its life-cycle.
 */
void GPULayerBase::addResidualTexture(GLuint textureID, int channelIndex) {
    while ((int)residualTextures_.size() < channelIndex) residualTextures_.push_back(0);      // we expect incrementing channel indices in the default case
    if (channelIndex == (int)residualTextures_.size()) residualTextures_.push_back(textureID);
    else residualTextures_[channelIndex] = textureID;
}


/**
 * @brief Clear all input textures registered with this layer
 *
 * Resets the layer's input textures to the empty set. Note that this does \e not deallocate
 * the GL resources of the registered input textures.
 */
void GPULayerBase::clearInputTextures() {
    inputTextures_.clear();
}


/**
 * @brief Clear all output textures registered with this layer
 *
 * Resets the layer's output textures to the empty set. Note that this does \e not deallocate
 * the GL resources of the registered output textures.
 */
void GPULayerBase::clearOutputTextures() {
    outputTextures_.clear();
    outputChanged_ = true;
}


/**
 * @brief Register input texture with this layer
 *
 * @param textureID Raw OpenGL texture handle to be added to the list of input textures
 *
 * @param channelIndex Index which is based on input port and channel within that port
 *
 * This function adds a texture to the input texture list at the provided \p channelIndex location.
 *
 * As a layer can have several input ports and each port may consist of more than one texture,
 * the \p channelIndex specifies a flattened offset into this list. The current implementation
 * unfortunately does not differentiate between shallow and deep tensor format, which renders the
 * meaning of the \p channelIndex parameter to be quite ambiguous.
 *
 * For shallow tensor layers, the behaviour of the \p channelIndex is as follows:
 * Assume that a shallow layer has 2 input ports, where the first port has 24 channels and the
 * second port has 32 channels. Because each texture aggregates 4 channels, this equals to 6
 * textures on the first port and 8 textures on the second port. For example. a channel index of 4
 * will refer to channels 16 to 19 (inclusive) of the first port and a channel index of 12 will
 * refer to channels 24..27 (inclusive) of the second port.
 *
 * For deep tensor layers, the behaviour of \p channelIndex is quite different and it will
 * be equivalent to the input port number, as all channels are stored in a single texture.
 *
 * In case this function is called for the same channel index multiple times, this default
 * implementation will \e overwrite the texture ID at that index. Specific layers might choose
 * a different approach. In general it is not recommended to use this function for \e changing
 * a texture, please see the updateInputTexture() function for that.
 *
 * @note This class does not take ownership over the supplied texture, it is up to the caller to
 *       maintain its life-cycle.
 *
 * @see BufferManager::connectLayers, BufferManager::connectGPULayers, getInputTexture, updateInputTexture
 */
void GPULayerBase::addInputTexture(GLuint textureID, int channelIndex) {
    while ((int)inputTextures_.size() < channelIndex) inputTextures_.push_back(0);      // we expect incrementing channel indices in the default case
    if (channelIndex == (int)inputTextures_.size()) inputTextures_.push_back(textureID);
    else inputTextures_[channelIndex] = textureID;
}


/**
 * @brief Update existing (previously added) texture slot with a new texture
 *
 * @param textureID New texture ID to set
 *
 * @param channelIndex Index which is based on input port and channel within that port
 *
 * This function \e updates an existing input texture slot at the specified \p channelIndex with the
 * the supplied \p textureID, overwriting the old texture ID. The new texture must have the same
 * dimensions as the old texture.
 *
 * This function may be overriden for classes that require change/setup code whenever an input
 * texture changes.
 *
 * @see addInputTexture
 */
void GPULayerBase::updateInputTexture(GLuint textureID, int channelIndex) {
    if ((int)inputTextures_.size() <= channelIndex) THROW_EXCEPTION_ARGS(FynException,"Invalid channel index %d supplied", channelIndex);
    inputTextures_[channelIndex] = textureID;
}


/**
 * @brief Register output texture with this layer
 *
 * @param textureID Raw OpenGL texture handle to be added to the list of output textures
 *
 * @param channelIndex Index which is based on output channel
 *
 * @param shadowIndex Optional index that adds a multiple textures to the same port/channelindex
 *                    for multi-buffering. If this functionality is to be used, this method has
 *                    to be overriden, otherwise any value other than 0 will raise an exception.
 *
 * This function adds a texture to the output texture list at the provided \p channelIndex location.
 * Opposed to the input, layers currently only have one output port, but may be extended to support
 * multiple output ports later (or never).
 *
 * Unlike the input, layers currently only have one output port, but may be extended to
 * support multiple output ports later (or never). If a layer has more than one output port
 * in the future, each port may consist of more than one texture. The \p channelIndex specifies
 * a flattened offset into this list. For example, assume that a layer has 2 output ports,
 * where the first port has 24 channels and the second port has 32 channels. This equals to 6
 * textures on the first port and 8 textures on the second port (each texture aggregates 4
 * channels). A channel index of 4 will therefore be channels 16 to 19 (inclusive) of the
 * first port and a channel index of 12 will be channels 24..27 (inclusive) of the second port.
 *
 * @post #outputChanged_ is set to \c true to indicate that some parts may have to be reinitialized
 *
 * @note This class does not take ownership over the supplied texture, it is up to the caller to
 *       maintain its life-cycle.
 *
 * @throws FynException if invalid parameters are supplied
 *
 * @see BufferManager::connectLayers, BufferManager::connectGPULayers, getOutputTexture
 */
void GPULayerBase::addOutputTexture(GLuint textureID, int channelIndex, int shadowIndex) {
    if (textureID == 0) THROW_EXCEPTION_ARGS(FynException,"Illegal texture ID %d supplied to %s", textureID, name_.c_str());
    if (shadowIndex != 0) THROW_EXCEPTION_ARGS(FynException,"Illegal shadow index %d supplied to %s, please override this method", shadowIndex, name_.c_str());
    while ((int)outputTextures_.size() < channelIndex) outputTextures_.push_back(0);         // we expect sequential channelIndex parameters for this method
    if (channelIndex == (int)outputTextures_.size()) outputTextures_.push_back(textureID);
    else outputTextures_[channelIndex] = textureID;
    outputChanged_ = true;
}


/**
 * @brief Check if layer has a valid output texture at specified index
 *
 * @param channelIndex Index which is based on output channel
 *
 * @retval true if an output texture has been assigned to that channel
 * @retval false otherwise
 *
 * This implementation checks if an output texture has been added to the specified \p channelIndex
 * via the #addOutputTexture method.
 *
 * Unlike the input, layers currently only have one output port, but may be extended to
 * support multiple output ports later (or never). If a layer has more than one output port
 * in the future, each port may consist of more than one texture. The \p channelIndex specifies
 * a flattened offset into this list. For example, assume that a layer has 2 output ports,
 * where the first port has 24 channels and the second port has 32 channels. This equals to 6
 * textures on the first port and 8 textures on the second port (each texture aggregates 4
 * channels). A channel index of 4 will therefore be channels 16 to 19 (inclusive) of the
 * first port and a channel index of 12 will be channels 24..27 (inclusive) of the second port.
 *
 * @see BufferManager::connectLayers, BufferManager::connectGPULayers, addOutputTexture, getOutputTexture
 */
bool GPULayerBase::hasOutputTexture(int channelIndex) const {
    if ((channelIndex < 0) || (channelIndex >= (int)outputTextures_.size())) return false;
    return true;
}


/**
 * @brief Retrieve raw OpenGL texture handle of output texture at specified index
 *
 * @param channelIndex Index within the set of output textures of this layer
 *
 * @return Raw OpenGL texture handle of the output texture connected to the specified \p channelIndex
 *
 * This implementation retrieves the raw OpenGL handle of the specified output texture, which must
 * have been previously added to the specified \p channelIndex via the #addOutputTexture method.
 *
 * Unlike the input, layers currently only have one output port, but may be extended to
 * support multiple output ports later (or never). If a layer has more than one output port
 * in the future, each port may consist of more than one texture. The \p channelIndex specifies
 * a flattened offset into this list. For example, assume that a layer has 2 output ports,
 * where the first port has 24 channels and the second port has 32 channels. This equals to 6
 * textures on the first port and 8 textures on the second port (each texture aggregates 4
 * channels). A channel index of 4 will therefore be channels 16 to 19 (inclusive) of the
 * first port and a channel index of 12 will be channels 24..27 (inclusive) of the second port.
 *
 * @throws FynException in case an illegal \p channelIndex has been provided
 *
 *
 * @see addOutputTexture, hasOutputTexture
 */
GLuint GPULayerBase::getOutputTexture(int channelIndex) const {
    if ((channelIndex < 0) || (channelIndex >= (int)outputTextures_.size())) THROW_EXCEPTION_ARGS(FynException,"Illegal channel index %d for output texture (size is %d)",channelIndex,outputTextures_.size());
    return outputTextures_.at(channelIndex);
}


/**
 * @brief Retrieve number of FBOs used by the layer output
 *
 * @return Number of FBOs that are allocated
 *
 * GPU-based NN layers use FBO instances to "render" the result of the layer operation into the
 * output textures. The number of %FBOs used is not necessarily the same as the number of
 * output textures, in case multi render targets are used (for shallow layers).
 *
 * @see getFBO
 */
int GPULayerBase::numFBOs() const {
    return framebuffers_.size();
}


/**
 * @brief Get FBO at specified index
 *
 * @param index Index within the %FBO list
 *
 * @return Pointer to FBO object at specified \p index
 *
 * Use this function to gain direct access to an output %FBO of the layer (for debugging purposes
 * for example). Note that when accessing an %FBO of a layer after other layers have been
 * invoked subsequently, the output may not be as expected, as the BufferManager reuses textures
 * wherever possible.
 */
FBO * GPULayerBase::getFBO(int index) const {
    return framebuffers_.at(index);
}


/**
 * @brief Check if a layer is able to be executed
 *
 * @retval true if layer is able to be executed
 * @retval false if layer cannot run
 *
 * The return value of this function is usually equivalent to isValid(), but there may be
 * implementations that have to handle it otherwise.
 *
 * @todo Retire this method
 */
bool GPULayerBase::canRun() const {
    return isValid();
}



/**
 * @copydoc LayerBase::writeResult
 */
void GPULayerBase::writeResult(const char *fileName, bool includePadding) {
#ifdef DEBUG
    int owidth = viewport_[0];
    int oheight = viewport_[1];
    if (!includePadding) {
        owidth -= 2 * outputPadding_;
        oheight -= 2 * outputPadding_;
    }
#ifndef FYUSENET_USE_WEBGL
    FILE *out = fopen(fileName,"w");
    if (out) {
#else
    uint8_t * download = new uint8_t[owidth * oheight * outputChannels_];
    uint8_t * downptr = download;
    if (true) {
#endif
        int outblocks = (outputChannels_ % PIXEL_PACKING) ? (outputChannels_ / PIXEL_PACKING)+1 : (outputChannels_ / PIXEL_PACKING);
        if (outblocks > FBO::MAX_DRAWBUFFERS) outblocks = FBO::MAX_DRAWBUFFERS;
        if ((owidth <= 0 ) || (oheight <= 0)) THROW_EXCEPTION_ARGS(FynException, "Illegal writing resolution %dx%d encountered",owidth,oheight);
        float * data = new float[viewport_[0] * viewport_[1] * PIXEL_PACKING * outblocks];
        float * layer = new float[owidth * oheight];
        int rem = outputChannels_;
        for (int fb = 0 ; fb < numFBOs(); fb++ ) {
            memset(data, 0, viewport_[0] * viewport_[1] * outblocks * PIXEL_PACKING * sizeof(float));
            FBO *fbo = getFBO(fb);            
            fbo->writeToMemory<float,GL_FLOAT>(data, PIXEL_PACKING, viewport_[0] * viewport_[1] * outblocks * PIXEL_PACKING * sizeof(float));
            int fborem = fbo->numAttachments() * PIXEL_PACKING;
            if (fborem > rem) fborem = rem;
            float *ptr = data;
            int padoffset = (includePadding) ? 0 : outputPadding_;
            while (fborem > 0) {
                int ml = (fborem >= PIXEL_PACKING) ? PIXEL_PACKING : fborem;
                for (int l=0; l <ml; l++) {
                    for (int y=0; y  < oheight; y++) {
                        for (int x=0; x < owidth; x++) {
                            layer[x+y*owidth]=ptr[l+PIXEL_PACKING*((y+padoffset)*viewport_[0]+x+padoffset)];
                        }
                    }
#ifndef FYUSENET_USE_WEBGL
                    fwrite(layer, 1, owidth * oheight * sizeof(float), out);
#else
                    memcpy(downptr, layer, owidth * oheight * sizeof(float));
                    downptr += owidth * oheight;
#endif
                }
                fborem -= ml;
                rem -= ml;
                ptr += viewport_[0] * viewport_[1] * PIXEL_PACKING;
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
    } else {
        FNLOGE("Cannot open %s for output",fileName);
    }
#endif
}


/**
 * @brief Copy computation results of layer to CPU memory for debugging purposes
 *
 * @param[out] memory Pointer to memory (32-bit FP) where data should be written to, it is the
 *                    caller's responsibility to make sure that enough memory is available
 *
 * @param includePadding If \c true, the padding will be included in the output file, otherwise
 *        the padding will be ignored and only the net contents are written to the output file
 *
 * This function copies the content of the output textures as binary dump into the specified memory.
 * region. All data will be written as little-endian 32-bit IEEE-754 floating-point numbers in a
 * channel-by-channel fashion. The data is arranged row-by-row (x-axis as innermost index) for a
 * single channel (y-axis as middle index) where the channels are stacked (channel axis as outermost
 * index).
 *
 * @note This function only works in a debug build. In release builds, this will be a no-op.
 */
void GPULayerBase::copyResult(float *memory, bool includePadding) {
#ifdef DEBUG
    int owidth = viewport_[0];
    int oheight = viewport_[1];
    if (!includePadding) {
        owidth -= 2 * outputPadding_;
        oheight -= 2 * outputPadding_;
    }
    if ((owidth <= 0 ) || (oheight <= 0)) THROW_EXCEPTION_ARGS(FynException,"Illegal resolution %dx%d encountered",owidth,oheight);
    int rem = outputChannels_;
    float * target = memory;
    for (int fb = 0 ; fb < numFBOs(); fb++ ) {
        FBO *fbo = getFBO(fb);
        assert(fbo);
        float * tmp = new float[fbo->numAttachments() * PIXEL_PACKING * viewport_[0] * viewport_[1]];
        fbo->writeToMemory<float,GL_FLOAT>(tmp, PIXEL_PACKING, viewport_[0] * viewport_[1] * fbo->numAttachments() * PIXEL_PACKING*sizeof(float));
        int fborem = fbo->numAttachments() * PIXEL_PACKING;
        if (fborem > rem) fborem = rem;
        int padoffset = (includePadding) ? 0 : outputPadding_;
        const float * ptr  = tmp;
        while (fborem > 0) {
            int ml = (fborem >= PIXEL_PACKING) ? PIXEL_PACKING : fborem;
            for (int l=0; l < ml;l++) {
                for (int y=0; y  < oheight; y++) {
                    for (int x=0; x < owidth; x++) {
                        target[x+y*owidth] = ptr[l+PIXEL_PACKING*((y+padoffset)*viewport_[0]+x+padoffset)];
                    }
                }
                target += owidth*oheight;
            }
            fborem -= ml;
            rem -= ml;
            ptr += viewport_[0] * viewport_[1] * PIXEL_PACKING;
        }
        delete [] tmp;
    }
#else
    THROW_EXCEPTION_ARGS(FynException, "This function is not available in release mode");
#endif
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Preprocess and compile/cache a vertex/fragment shader pair
 *
 * @param vertexName Resource name for the vertex shader to use
 *
 * @param fragmentName Resource name for the fragment shader to use
 *
 * @param preprocDefs Optional string with additional preprocessor definitions
 *
 * @param typeInfo Implementation specific type information from the caller that ensures that
 *                 shaders are cached uniquely with the type of layer they are used for
 *
 * @throws ShaderException in case a shader was not found or could not be compiled
 *
 * @return Shader pointer to compiled shader program (see warning)
 *
 * This function offers a convenient interface to compile a vertex/fragment shader pair with some
 * additional benefits:
 *   1. Provided resource names are loaded from the resource system
 *   2. Resulting shader sources are preprocessed
 *   2. Shader sources are compiled and cached
 *   3. Resulting shader program is cached
 *
 * The preprocessing includes adding the provided \p preprocDefs after the shader preamble and
 * also resolving any \c \#include statements in the shader sources by substituting the right
 * ShaderSnippet for that. If a shader cache is available, the preprocessed shader sources are
 * then checked for presence in the cache, in which case the already compiled shader is taken from
 * the cache. Otherwise the shaders are compiled and put into the shader cache. Following the
 * compilation / cache-lookup, it is checked if the shaders are already available as a linked
 * program for this type of layer (this is where the \p typeInfo parameter comes into play). If
 * that is not the case, the shaders are linked and put into the cache, otherwise the cached
 * instance is used.
 *
 * The main reason for the \p typeInfo parameter is to make sure that shader \e programs are not
 * cached between different types of layer as some static settings on the uniform variables may
 * differ. It is up to the implementation of the actual layers to make sure that uniforms which
 * are not exactly the same based on the layer type (e.g. they are dependent on image resolution),
 * are re-set before running the shader
 *
 * @warning This function does \b not link the resulting shader program and it is up to the
 *          caller to make sure of that. However, when calling this function with a set of shaders
 *          for which a shader program was already cached, the returned shader program <i>might
 *          already be linked</i>. Make sure to query the shader state before conducting operations
 *          that require a certain state.
 *
 * @see ShaderRepository::getShader, ShaderCache::findShader
 */
programptr GPULayerBase::compileShaderPair(const char *vertexName, const char *fragmentName,
                                           const char *preprocDefs, const std::type_info& typeInfo) {
    using namespace fyusion::opengl;
    const char *vert = ShaderRepository::getShader(vertexName);
    const char *frag = ShaderRepository::getShader(fragmentName);
    if (!vert) THROW_EXCEPTION_ARGS(ShaderException, "Cannot load vertex shader %s (not found)", vertexName);
    if (!frag) THROW_EXCEPTION_ARGS(ShaderException, "Cannot load fragmnet shader %s (not found)", fragmentName);
    shaderptr vshader(new VertexShader(context_));
    shaderptr fshader(new FragmentShader(context_));
    vshader->setResourceName(vertexName);
    fshader->setResourceName(fragmentName);
    vshader->setCode(vert);
    fshader->setCode(frag);
    vshader->setPreprocDefs(preprocDefs);
    fshader->setPreprocDefs(preprocDefs);
    ShaderCache *cache = ShaderCache::getInstance(context_);
    try {
        if (cache) {
            size_t modhash = typeInfo.hash_code();
            shaderptr vcache = cache->findShader(vshader);
            shaderptr fcache = cache->findShader(fshader);
            if (vcache && fcache) {
                std::vector<GLuint> handles{vcache->getHandle(), fcache->getHandle()};
                programptr prog = cache->findProgram(modhash, handles);
                if (prog) {
                    return prog;
                }
            }
            programptr prog = ShaderProgram::createInstance(context_);
            prog->addShader( (vcache) ? vcache : vshader );
            prog->addShader( (fcache) ? fcache : fshader );
            prog->compile();
            if ((!vcache) && (cache)) cache->putShader(vshader);
            if ((!fcache) && (cache)) cache->putShader(fshader);
            if (cache) cache->putProgram(prog, modhash);
            return prog;
        } else {
            programptr prog = ShaderProgram::createInstance(context_);
            prog->addShader(vshader);
            prog->addShader(fshader);
            prog->compile();
            return prog;
        }
    } catch (GLException& ex) {
        FNLOGE("Cannot compile shader in layer %s", getName().c_str());
        throw;
    }
}


/**
 * @brief Mix user-supplied preprocessor definitions with flag-induced definitions
 *
 * @param flags Layer flags to be turned into preprocessor definitions
 *
 * @param[inout] preproc User-defined preprocessor definitions
 *
 * @param maxChars Maximum number of characters available in the \p preproc string
 *
 * @return Buffer capacity left in the supplied \p preproc buffer
 *
 * This function blocks the handling of preprocessor definitions in conjunction with layer-flags.
 * Based on the flags that were passed in \p flags, it appends preprocessor definitions to the
 * supplied \p preproc string. The following preprocessor strings are set for the layer flags:
 *  \li \c PRE_RELU adds the \c ACT_RELU preprocessor definition. In case a leaky ReLU is selected,
 *      and additional definition \c LEAKY_RELU with the leak value is added
 *  \li \c PRE_CLIP adds the \c ACT_CLIP preprocessor definition as well as CLIP_LOW and CLIP_HIGH
 *      with the respective values for the clipping activation function
 *  \li \c RESIDUAL_INPUT adds the \c USE_RESIDUAL preprocessor definition
 *  \li \c RELU_ON_RESIDUAL adds the \c RELU_ON_RESIDUAL preprocessor definition
 *  \li \c POST_BATCHNORM adds the \c POST_BATCHNORM preprocessor definition
 *
 *  In case no activation function was specified, \c NO_ACT is added as preprocessor definition.
 *
 *  Independent of the layer flags, a couple of additional preprocessor definitions are set, which
 *  are:
 *   - \c PIXEL_PACKING which defines the number of channels per pixel (4 usually)
 *   - \c PADDING defines the padding value on the input data
 *   - \c NO_HALF if set, indicates that 16-bit floating point data is not available as texture
 *        format
 *   - \c HIGH_PRECISION if set, indicates that high precision (full 32-bit FP) are desired and the
 *        precision qualifiers should be set to "high"
 *
 * The result of the preprocesser handling is appended to the supplied \p preproc data.
 */
size_t GPULayerBase::handlePreprocFlags(layerflags flags, char *preproc, size_t maxChars) {
    char extra[80];
    assert(maxChars > 0);
    ssize_t mc = (ssize_t)handleActivationPreproc(flags, preproc, maxChars);
    if (flags & LayerFlags::RESIDUAL_INPUT) {
        strncat(preproc, "#define USE_RESIDUAL\n", mc);
        mc = maxChars-strlen(preproc);  // ouch
    }
    assert(mc > 0);
    if (flags & LayerFlags::RELU_ON_RESIDUAL) {
        strncat(preproc, "#define RELU_ON_RESIDUAL\n", mc);
        mc = maxChars-strlen(preproc);  // ouch
    }
    assert(mc > 0);
    if (flags & LayerFlags::BATCHNORM_ON_RESIDUAL) {
        strncat(preproc, "#define BATCHNORM_ON_RESIDUAL\n", mc);
        mc = maxChars-strlen(preproc); // ouch
    }
    assert(mc > 0);
    if (flags & LayerFlags::POST_BATCHNORM) {
        strncat(preproc, "#define POST_BATCHNORM\n", mc);
        mc = maxChars-strlen(preproc);  // ouch
    }
    snprintf(extra, sizeof(extra), "#define PIXEL_PACKING %d\n",PIXEL_PACKING);
    strncat(preproc, extra, mc);
    mc -= strlen(extra);
    assert(mc > 0);
#ifdef HIGH_PRECISION
    strncat(preproc, "#define NO_HALF\n", mc);
    mc -= 16;
#else
    if (!GLInfo::supportsHalf()) {
        strncat(preproc, "#define NO_HALF\n", mc);
        mc -= 16;
    }
#endif
    snprintf(extra, sizeof(extra), "#define PADDING %d\n",inputPadding_);
    strncat(preproc, extra, mc);
    mc -= strlen(extra);
    assert(mc >= 0);
#ifdef HIGH_PRECISION
    strncat(preproc, "#define HIGH_PRECISION\n", mc);
    mc = maxChars-strlen(preproc);  // ouch
#endif
    return (size_t)std::max((ssize_t)0,mc);
}


/**
 * @brief Handle preprocessor flags related to activation functions
 *
 * @param flags Layer flags to be turned into preprocessor definitions
 *
 * @param[inout] preproc User-defined preprocessor definitions
 *
 * @param maxChars Maximum number of characters available in the \p preproc string
 *
 * @return Buffer capacity left in the supplied \p preproc buffer
 *
 * This function handles the activation-related preprocessor flags (e.g. ReLU, clipping etc.) and
 * appends the correct preprocessor definitions to the supplied \p preproc. The following
 * substitutions are currently done:
 *
 *  \li \c PRE_RELU adds the \c ACT_RELU preprocessor definition. In case a leaky ReLU is selected,
 *      and additional definition \c LEAKY_RELU with the leak value is added
 *  \li \c PRE_CLIP adds the \c ACT_CLIP preprocessor definition as well as CLIP_LOW and CLIP_HIGH
 *      with the respective values for the clipping activation function
 *
 *  In case no activation function was specified, \c NO_ACT is added as preprocessor definition.
 */
// TODO (mw) refactor this for more flexible activation
size_t GPULayerBase::handleActivationPreproc(layerflags flags, char *preproc, size_t maxChars) {
    char extra[80];
    ssize_t mc = (ssize_t)maxChars;
    if ((flags & (LayerFlags::PRE_RELU | LayerFlags::PRE_CLIP)) == 0) {
        strncat(preproc, "#define NO_ACT\n", mc);
        mc = maxChars-strlen(preproc);
        assert(mc > 0);
    } else {
        if (flags & LayerFlags::PRE_CLIP) {
            snprintf(extra, sizeof(extra), "#define ACT_CLIP\n#define CLIP_LOW %f\n#define CLIP_HIGH %f\n", lowClip_, highClip_);
            strncat(preproc, extra, mc);
            mc -= strlen(extra);
            assert(mc > 0);
        } else {
            // TODO (mw) support: ACT_SIGMOID , ACT_TANH and ACT_CLIP
            strncat(preproc, "#define ACT_RELU\n", mc);
            mc = maxChars-strlen(preproc);
            assert(mc > 0);
            if (leakyReLU_ != 0.0f) {
                snprintf(extra, sizeof(extra),"#define LEAKY_RELU %f\n", leakyReLU_);
                strncat(preproc, extra, mc);
                mc -= strlen(extra);
            }
            assert(mc > 0);
        }
    }
    return (size_t)std::max((ssize_t)0,mc);
}


/**
 * @brief Prepare layer for rendering operation
 *
 * @param blend If set to \c true, enables alpha blending in the output (used for accumulation),
 *              defaults to \c true
 * @param depth If set to \c true, enables depth buffer testing, default is \c false
 */
void GPULayerBase::prepareRender(bool blend, bool depth) {
    if (outputChanged_) updateFBOs();
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    if (blend) glEnable(GL_BLEND);
    else glDisable(GL_BLEND);
    if (depth) glEnable(GL_DEPTH_TEST);
    else glDisable(GL_DEPTH_TEST);
    if (blend) {
        glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
        glBlendFuncSeparate(GL_ONE,GL_ONE, GL_ONE,GL_ONE);
    }
    glClearColor(0, 0, 0, 0);
    glViewport(0, 0, viewport_[0], viewport_[1]);
}

} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
