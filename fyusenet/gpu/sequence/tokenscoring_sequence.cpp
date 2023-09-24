//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Token-Scoring Layer for Sequences (Header)                                  (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <algorithm>
#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/vertexshader.h"
#include "../../gl/scoped_texturepool.h"
#include "../floatconversion.h"
#include "../rudiments/proxygenerator.h"
#include "../../common/miscdefs.h"
#include "embedding_sequence.h"
#include "tokenscoring_sequence.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet::gpu::sequence {

//-------------------------------------- Local Definitions -----------------------------------------


#ifdef DEBUG
/**
 * @brief Little helper function to write FBO contents to a file
 *
 * @param filename Output filename
 * @param fbo Pointer to FBO to write
 * @param allocWidth Allocation width of FBO
 * @param allocHeight Allocation height of FBO
 * @param storeHeight Height of data to store
 * @param channels Number of channels to store
 *
 * @tparam T data type
 * @tparam V GL data type ID
 * @tparam integral true if data type is integral
 *
 */
template<typename T, int V, bool integral>
static void writeFBO(const char *filename, FBO * fbo, int allocWidth, int allocHeight, int storeHeight, int channels) {
    std::vector<uint8_t> data(allocWidth * allocHeight * channels * sizeof(T));
    fbo->writeToMemory<T,V>((T *)&data[0], channels, allocWidth * allocHeight * channels * sizeof(T), integral);
#ifndef FYUSENET_USE_WEBGL
    FILE *out = fopen(filename, "wb");
    if (out) {
        fwrite(&data[0], 1, allocWidth * storeHeight * channels * sizeof(T), out);
        fclose(out);
    }
#else
    EM_ASM({window.download($0, $1, $2);}, &data[0], allocWidth * storeHeight * channels * sizeof(T), filename);
#endif
}
#endif


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @copydoc GPULayerBase::GPULayerBase
 */
TokenScoringLayer::TokenScoringLayer(const TokenScoringLayerBuilder &builder) : TokenScoringLayer(builder, builder.number_) {
}

/**
 * @copydoc GPULayerBase::GPULayerBase(const GPULayerBuilder&, int)
 */
TokenScoringLayer::TokenScoringLayer(const TokenScoringLayerBuilder &builder, int layerNumber)
        : GPULayerBase((GPULayerBuilder &) builder, layerNumber) {
    assert(builder.maxSequenceLen_ > 0);
    embedDim_ = inputChannels_;
    width_ = (embedDim_ + PIXEL_PACKING -1) / PIXEL_PACKING;
    height_ = builder.maxSequenceLen_;
    tableRows_ = builder.tableRows_;
    assert(tableRows_ > 0);
    viewport_[0] = 1;
    viewport_[1] = builder.maxSequenceLen_;
    temperature_ = builder.temperature_;
    topK_ = builder.topK_;
    topP_ = builder.topP_;
    scoring_ = builder.scoringType_;
    vocabAggregateSize_ = MAX_VOCAB_AGGREGATE_SIZE;         // TODO (mw) make this GPU specific ?
    hasParameters_ = true;
}

/**
 * @copydoc GPULayerBase::cleanup
 */
void TokenScoringLayer::cleanup() {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    embeddingTextures_.clear();
    proShader_.reset();
    pass1FlatShader_.reset();
    pass2FlatShader_.reset();
    scatterShader_.reset();
    selectionShader_.reset();
    FNET_DEL_AND_CLEAR(proVerts_)
    FNET_DEL_AND_CLEAR(proIndices_)
    FNET_DEL_AND_CLEAR(proArray_)
    FNET_DEL_AND_CLEAR(pass1FlatVerts_)
    FNET_DEL_AND_CLEAR(pass1FlatArray_)
    FNET_DEL_AND_CLEAR(selectionFBO_)
    FNET_DEL_AND_CLEAR(projectionFBO_)
    FNET_DEL_AND_CLEAR(scatterFBO_)
    FNET_DEL_AND_CLEAR(scatterArray_)
    FNET_DEL_AND_CLEAR(scatterVerts_)
    for (int i = 0; i < 2; ++i) {
        FNET_DEL_AND_CLEAR(flatFBOs_[i]);
    }
    if (scatterDepth_) glDeleteRenderbuffers(1, &scatterDepth_);
    scatterDepth_ = 0;
    GPULayerBase::cleanup();
}


/**
 * @copydoc LayerBase::setup
 */
void TokenScoringLayer::setup() {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    CLEAR_GFXERR_DEBUG
    proxyGeometry();
    compileShaders();
    setupFBOs();
    assert(glGetError() == GL_NO_ERROR);
    valid_ = true;
}



/**
 * @copydoc LayerBase::forward
 */
void TokenScoringLayer::forward(uint64_t sequenceNo, StateToken * state) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    if (!valid_) THROW_EXCEPTION_ARGS(FynException, "Trying to invoke forward() on invalid layer");
    if (!state) THROW_EXCEPTION_ARGS(FynException, "Trying to invoke forward() without token state");
    CLEAR_GFXERR_DEBUG
    glDisable(GL_SCISSOR_TEST);
    prepareRender(true, false, true);
    projectToken(state->seqLength - 1);
    flatten();
    scatter();
    selection();
    for (int i=0; i < (int)embeddingTextures_.size(); i++) {
        glActiveTexture(GL_TEXTURE1+i);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}


/**
 * @brief Obtain buffer specifiers that are required as output for this layer
 *
 * @return Vector of buffer specifiers that specify the format for each required buffer
 *
 * @see BufferSpec
 *
 * @note This layer differs from the standard 2D image layers found in FyuseNet. In particular, the
 *       viewport width is fixed to 1 and the viewport height is equivalent to the maximum sequence
 *       length.
 */
std::vector<BufferSpec> TokenScoringLayer::getRequiredOutputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0,
                                viewport_[0], viewport_[1],
                                BufferSpec::sizedformat::SINGLE32UI, BufferSpec::genericformat::SINGLE_INT, BufferSpec::dtype::UINT32,
                                BufferSpec::FUNCTION_DEST).dataOrder(BufferSpec::order::GPU_SEQUENCE));
    return result;
}


/**
 * @brief Obtain buffer specifiers that are required as input for this layer
 *
 * @return Vector of buffer specifiers that specify the format for each required buffer
 *
 * @see BufferSpec
 *
 * @note This layer differs from the standard 2D image layers found in FyuseNet. In particular, the
 *       width stored in this layer is equivalent to the embedding size (divided by 4) and the height
 *       is equivalent to the maximum sequence length.
 */
std::vector<BufferSpec> TokenScoringLayer::getRequiredInputBuffers() const {
    std::vector<BufferSpec> result;
    result.push_back(BufferSpec(0, 0,
                                width_, height_,
                                TEXTURE_IFORMAT_4, TEXTURE_FORMAT_4, TEXTURE_TYPE_DEFAULT,
                                BufferSpec::FUNCTION_SOURCE).dataOrder(BufferSpec::order::GPU_SEQUENCE));
    return result;
}


/**
 * @brief Load parameters from provider
 *
 * @param source Pointer to provider for parameter data
 *
 * This function retrieves the embedding vectors / vocabulary from the \p source and stores them in
 * a set of textures to be used in the lookup shader. The provider will be called with the following
 * parameters:
 *   - for \c name: \c layername.embed
 *   - for \c subIndex: 0
 *
 * Where \c layername is the name assigned to this layer by the builder
 */
void TokenScoringLayer::loadParameters(const ParameterProvider * source) {
    std::lock_guard<std::recursive_mutex> lck(processingLock_);
    using namespace opengl;
    assert(source);
    if (((embedDim_ + PIXEL_PACKING -1) / PIXEL_PACKING) > GLInfo::getMaximumTextureSize()) {
        THROW_EXCEPTION_ARGS(FynException,"Embedding dimension %d is too large for GPU", embedDim_);
    }
    DataBlob table = source->get(getName()+std::string(".embed"), getNumber(), 0);
    assert(!table.empty());
    // -------------------------------------------------------
    // Select texture format (currently no int quantization
    // supported)...
    // -------------------------------------------------------
#ifdef HIGH_PRECISION
    opengl::Texture::pixtype pixtype = opengl::Texture::FLOAT32;
#else
    opengl::Texture::pixtype pixtype = opengl::Texture::FLOAT16;
#endif
    // -------------------------------------------------------
    // Allocate and fill textures...
    // -------------------------------------------------------
    int tgtsize = std::min((tableRows_  + HARD_TOKEN_TEXTURE_MAX-1) / HARD_TOKEN_TEXTURE_MAX,
                           GLInfo::getMaximumTextureSize());
    if (tgtsize & 1) tgtsize++;
    texWidth_ = (embedDim_ + PIXEL_PACKING -1) / PIXEL_PACKING;
    int numtextures = (tableRows_ + tgtsize - 1) / tgtsize;
    if (numtextures > HARD_TOKEN_TEXTURE_MAX) THROW_EXCEPTION_ARGS(FynException, "Vocabulary size (%d) too large", tableRows_);
    embeddingTextures_.reserve(numtextures+1);
    Texture::pixtype dtype = Texture::FLOAT32;
    int elsize = sizeof(float);
    const uint8_t * ptr = nullptr;
    switch (source->dataType(getName()+std::string(".embed"), getNumber(), 0)) {
        case param_type::WGT_FLOAT32:
            ptr = reinterpret_cast<const uint8_t *>(std::any_cast<const float *>(table.get()));
            break;
        case param_type::WGT_FLOAT16:
            elsize = sizeof(uint16_t);
            ptr = reinterpret_cast<const uint8_t *>(std::any_cast<const uint16_t *>(table.get()));
            dtype = Texture::FLOAT16;
            break;
        case param_type::WGT_INT8:
            // intentional fallthrough
        case param_type::WGT_INT4:
            THROW_EXCEPTION_ARGS(FynException,"Data type not supported");
        default:
            break;
    }
    assert(ptr);
    // FIXME (mw) handle the case when embedDim is not a multiple of 4
    for (int tex=0,rem=tableRows_; tex < numtextures; tex++, rem -= tgtsize) {
        int th = std::min(rem, tgtsize);
        opengl::Texture2D newtex(texWidth_, th, pixtype, 4);
        newtex.upload(ptr, dtype);
        ptr += th * embedDim_ * elsize;
        embeddingTextures_.push_back(newtex);
    }
    if (embeddingTextures_.empty()) THROW_EXCEPTION_ARGS(FynException, "Cannot create textures for embedding table (embed=%d height=%d)", embedDim_, tableRows_);
    setupProjectionTexture();
}


/**
 * @brief Clone (shallow-copy) embedding table from another layer
 *
 * @param src Source layer to copy the vocabulary texture(s) from
 */
void TokenScoringLayer::cloneEmbeddingTable(const EmbeddingLayer& src) {
    embeddingTextures_ = src.embeddingTextures_;
    texWidth_ = src.texWidth_;
    tableRows_ = src.tableRows_;
    setupProjectionTexture();
}


/**
 * @copydoc LayerBase::writeResult
 */
void TokenScoringLayer::writeResult(const char *fileName, bool includePadding) {
#ifdef DEBUG
    FBO * fbo = getFBO(0);
    writeFBO<uint32_t, GL_UNSIGNED_INT, true>(fileName, fbo, fbo->width(), fbo->height(), 1, 1);
    const char * suffix = strrchr(fileName, '.');
    if (suffix) {
        char add[768];
        memcpy(add, fileName, strlen(fileName)+1);
        char * inspoint = strrchr(add, '.');
        int offset = (int)(inspoint-add);
        snprintf(inspoint, sizeof(add)-offset-1, "_scores.%s", suffix + 1);
        writeFBO<float, GL_FLOAT, false>(add, projectionFBO_, projectionFBO_->width(), projectionFBO_->height(),
                          projectionFBO_->height(), 4);
        snprintf(inspoint, sizeof(add)-offset-1, "_scatter.%s", suffix + 1);
    }
#endif  // DEBUG
}

/**
 * @copydoc GPULayerBase::getGPUOutputBuffer
 */
GPUBuffer *TokenScoringLayer::getGPUOutputBuffer(int port) const {
    assert(port == 0);
    if (outputTextures_.empty()) return nullptr;
    auto * out = createGPUBuffer(1, height_, 1, getOutputOrder(port), getOutputType(port), 0);
    pushSliceToBuffer(out, outputTextures_[0], 1, height_, 1, getOutputType(port));
    return out;
}


/**
 * @copydoc GPULayerBase::getGPUInputBuffer
 */
GPUBuffer *TokenScoringLayer::getGPUInputBuffer(int port) const {
    if (inputTextures_.empty()) return nullptr;
    auto * out = createGPUBuffer(width_, height_, PIXEL_PACKING, getInputOrder(port), getInputType(port), 0);
    pushSliceToBuffer(out, inputTextures_[0], width_, height_, PIXEL_PACKING, getInputType(port));
    return out;
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Setup texture to hold all projection (inner product) results
 *
 * The first step of the scoring process is to compute the inner product of the last token's
 * embedding with the vocabulary stores in this object. For example, if the vocabulary consists
 * of 32000, we will need a projection texture with 32000 entries. Due to limitations in texture
 * sizes, we spread out these 32000 entries in a rectangular texture.
 * Together with the projection texture, this function also allocates two additional textures
 * which are used in a reduction step to aggregate basic statistics of the individual scores
 * in order to parameterize the scatter/sorting step later.
 *
 * @see projectToken()
 */
void TokenScoringLayer::setupProjectionTexture() {
    assert(embedDim_);
    assert(tableRows_);
    assert(!embeddingTextures_.empty());
    if (projectionTexture_.empty()) {
        // ----------------------------------------------------------
        // Setup projection result texture, first compute the right
        // size and then allocate the texture / FBO
        // ----------------------------------------------------------
        projectionSegments_.reserve(embeddingTextures_.size());
        int tokens = embeddingTextures_.at(0).height();
        assert((tokens % PIXEL_PACKING) == 0);
        int pixels = tokens / PIXEL_PACKING;
        int linewidth = 0;
        for (int factor = (int)sqrtf((float)pixels); factor >= 2; factor--) {
            if ((pixels % factor) == 0) {
                linewidth = factor;
                break;
            }
        }
        if (linewidth == 0) {
            for (int factor = (int)sqrtf((float)pixels); factor <= pixels/2; factor++) {
                if ((pixels % factor) == 0) {
                    linewidth = factor;
                    break;
                }
            }
        }
        if (linewidth == 0) THROW_EXCEPTION_ARGS(FynException,"Cannot find a suitable projection texture size for %d-height textures", embeddingTextures_.at(0).height());
        int numrows = 0;
        for (auto & tex : embeddingTextures_) {
            int h = (tex.height() + PIXEL_PACKING-1) / PIXEL_PACKING;
            int lines = (h + linewidth-1) / linewidth;
            numrows += lines;
            projectionSegments_.emplace_back(lines);
        }
        projectionSize_[0] = linewidth;
        projectionSize_[1] = numrows;
        uint32_t scope = (context().texturePool()) ? context().texturePool()->scopeID() : 0;
        projectionTexture_ = opengl::Texture2D(linewidth, numrows, opengl::Texture::FLOAT32, 4, context().texturePool(), scope, false);
        projectionFBO_ = new opengl::FBO(context(), projectionTexture_);
        // ----------------------------------------------------------
        // Setup textures for flattening / data consolidation passes
        // ----------------------------------------------------------
        float ar = (float)numrows / (float)linewidth;
        int hsub = (linewidth <= 64) ? 2 : 4;
        int vsub = (int)ceilf(ar * (float)hsub);
        while (hsub * vsub > vocabAggregateSize_) vsub--;
        assert(vsub > 1);
        int flattenwidth = (linewidth + hsub - 1) / hsub;
        int flattenheight = (numrows + vsub - 1) / vsub;
        flatSubsampling_[0] = hsub;
        flatSubsampling_[1] = vsub;
        assert(flattenwidth > 0);
        assert(flattenheight > 0);
        // TODO (mw) use texture pooling here (low priority as textures are small)
        flatFBOs_[0] = new FBO(context(), flattenwidth, flattenheight, 4, Texture::pixtype::FLOAT32);
        flatFBOs_[0]->addTexture(GL_COLOR_ATTACHMENT1, 4, Texture::pixtype::FLOAT32);
        flatFBOs_[0]->unbind();
        flatFBOs_[1] = new FBO(context(), 2, 1, 4, Texture::pixtype::FLOAT32);
        const auto [darr, dverts] = rudiments::ProxyGenerator::texturedDotMatrix(context(), flattenwidth, flattenheight);
        pass1FlatArray_ = darr;
        pass1FlatVerts_ = dverts;
    }
}


/**
 * @brief Perform projection of the supplied token's embedding against all vocabulary entries
 *
 * @param token The token index in the input embeddings to project
 *
 * This function uses the input embeddings and the vocabulary texture and computes all combinations
 * of inner products (so it's technically a matrix/vector multiplication) between them and stores
 * the results in the projection texture. We are interested in entries that have a high score as
 * they indicate that the match between a token and the embedding is high. As this part only
 * computes the projection, a set of subsequent steps is required to actually extract the
 * "best match" from the projections.
 */
void TokenScoringLayer::projectToken(int token) {
    assert(projectionFBO_);
    projectionFBO_->bind();
    proArray_->bind();
    int instances = (width_ + proInstanceWidth_ - 1)/ proInstanceWidth_;
    glViewport(0, 0, projectionSize_[0], projectionSize_[1]);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, inputTextures_.at(0));
    int ywindow = 0;
    proShader_->bind();
    proShader_->setUniformValue("token", token);
    for (size_t segment=0; segment < embeddingTextures_.size(); segment++) {
        proShader_->setUniformVec2("viewport", projectionSize_[0], projectionSegments_[segment]);
        glViewport(0, ywindow, projectionSize_[0], projectionSegments_[segment]);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, embeddingTextures_[segment].getHandle());
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (const GLvoid *) nullptr, instances);
        ywindow += projectionSegments_[segment];
    }
    proShader_->unbind(true);
    projectionFBO_->unbind();
    proArray_->unbind();
}


/**
 * @brief Reduce/aggregate the projection results using basic statistcs measures
 *
 * This function uses the projection results and aggregates those into some basic statistical
 * data in two steps, resulting in a final texture with a total of 8 values:
 *   - min/max of the projection results
 *   - mean/stddev of the projection results
 *   - max over all regional minima from pass 1
 *   - count of values greater or equal than a 95% mix of maximum and mean
 *   - count of values greater or equal than a 90% mix of maximum and mean
 *   - count of values greater or equal than a 75% mix of maximum and mean
 *
 * @see projectToken(), scatter()
 */
void TokenScoringLayer::flatten() {
    assert(pass1FlatArray_);
    CLEAR_GFXERR_DEBUG
    glDisable(GL_BLEND);
    // --------------------------------------------------------
    // Pass 1: perform some pre-aggregation and compute some
    // basic stats about the output distributions
    // --------------------------------------------------------
    flatFBOs_[0]->bind();
    flatFBOs_[0]->setWriteMask();
    glViewport(0, 0, flatFBOs_[0]->width(), flatFBOs_[0]->height());
    glClear(GL_COLOR_BUFFER_BIT);
    pass1FlatArray_->bind();
    pass1FlatShader_->bind();
    pass1FlatShader_->setUniformVec2("textSize", projectionSize_[0], projectionSize_[1]);
    pass1FlatShader_->setUniformVec2("shift", 0.5f/(float)flatFBOs_[0]->width(), 0.5f/(float)flatFBOs_[0]->height());
    pass1FlatShader_->setUniformVec2("contractionRange", flatSubsampling_[0], flatSubsampling_[1]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, projectionTexture_.getHandle());
    glDrawArrays(GL_POINTS, 0, flatFBOs_[0]->width() * flatFBOs_[0]->height());
    pass1FlatShader_->unbind(true);
    flatFBOs_[0]->unbind();
    pass1FlatArray_->unbind();
    // --------------------------------------------------------
    // Pass 2: make use of pre-aggregate stats and narrow it
    // down to some usable range values for the following
    // scatter operation...
    // --------------------------------------------------------
    flatFBOs_[1]->bind();
    pass2FlatShader_->bind();
    scatterArray_->bind();
    glViewport(0, 0, 2, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    pass2FlatShader_->setUniformVec2("contractionRange", flatFBOs_[0]->width(), flatFBOs_[0]->height());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, flatFBOs_[0]->getAttachment(GL_COLOR_ATTACHMENT0));
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, flatFBOs_[0]->getAttachment(GL_COLOR_ATTACHMENT1));
    glDrawArrays(GL_POINTS, 0, 2);
    pass2FlatShader_->unbind(true);
    scatterArray_->unbind();
    flatFBOs_[1]->unbind();
    assert(glGetError() == GL_NO_ERROR);
}


/**
 * @brief Perform scatter pass of token scores to perform sorting
 *
 * This function uses the vertex shader to perform a scatter operation on the token scores by
 * rendering the scores to multiple rows in the output texture. For each row we perform a "lossy"
 * variant of a bucket sort. Data is sorted along the rows into pixels (the buckets) and the
 * z-buffer / depth-testing hardware is used to make sure that on bucket collisions only the maximum
 * value is retained.
 *
 * The first row renders a quite narrow range of token scores, where higher scores are placed at
 * lower x-coordinates (left). This row is ideal to pick the absolute maximum or to do a quite
 * narrow top-k sampling. The second row renders a wider range of token scores, using the same
 * ordering and might be usable for a wider top-k sampling and other approaches.
 *
 * The output of the scatter operation (aside from the z-buffer) consists of two textures with
 * two rows (for now). The first texture stores integer indices into the token table (offset by
 * 1, i.e. token #0 will be stored as 1 in the texture in order to distinguish it from the empty
 * buckets). The second texture stores RGBA values which contain the token score as well as the
 * indices into the source texture.
 *
 * @see selection()
 */
void TokenScoringLayer::scatter() {
    glDisable(GL_BLEND);
    CLEAR_GFXERR_DEBUG
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    assert(glGetError() == GL_NO_ERROR);
    glDepthFunc(GL_LESS);
    assert(glGetError() == GL_NO_ERROR);
    scatterFBO_->bind();
    scatterFBO_->setWriteMask();
    glViewport(0, 0, SCATTER_WIDTH, 2);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    scatterArray_->bind();
    scatterShader_->bind();
    scatterShader_->setUniformVec2("projSize", projectionSize_[0], projectionSize_[1]);
    scatterShader_->setUniformVec2("scatterShift", 0.5f/(float)SCATTER_WIDTH, 0.5f/2.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, projectionFBO_->getAttachment());
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, flatFBOs_[1]->getAttachment());
    glDrawArraysInstanced(GL_POINTS, 0, tableRows_, 2);
    scatterArray_->unbind();
    scatterFBO_->unbind();
    scatterShader_->unbind(true);
    glDisable(GL_DEPTH_TEST);
}


/**
 * @brief Select predicted token based on the scores
 *
 * This function selects the token that constitutes the prediction output of the network. Currently
 * supported modes are:
 *  - greedy/argmax: select the token with the highest score
 *  - top-k: select a random token of the top-k tokens with the highest scores
 *
 * @see scatter()
 */
void TokenScoringLayer::selection() {
    framebuffers_.at(0)->bind();
    glEnable(GL_SCISSOR_TEST);
    glViewport(0, 0, 1, 1);
    glScissor(0, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    scatterArray_->bind();
    selectionShader_->bind();
    selectionShader_->setUniformValue("seed", 0, true);       // FIXME (mw) use something random here
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, scatterFBO_->getAttachment(GL_COLOR_ATTACHMENT0));
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, scatterFBO_->getAttachment(GL_COLOR_ATTACHMENT1));
    glDrawArrays(GL_POINTS, 0, 1);
    framebuffers_.at(0)->unbind();
#if 0   // multi-buffering extension
    // ---------------------------------------------------------------------
    // We could now blit the result to the secondary buffer, but the code is
    // so lightweight, we just render it again. Randomized parts in the
    // shader use a PRNG, so not changing the seed should yield the same
    // results..
    // ---------------------------------------------------------------------
    // bind other stuff
    glDrawArrays(GL_POINTS, 0, 1);
    // unbind other stuff
#endif
    glDisable(GL_SCISSOR_TEST);
    selectionShader_->unbind();
    scatterArray_->unbind();
}


/**
 * @brief Shader compilation
 *
 * This function compiles a set of shaders for the following passes:
 *  - projection pass
 *  - flattening passes
 *  - scatter / sorting pass
 *  - selection pass
 *
 *  @see flatten(), scatter(), selection()
 */
void TokenScoringLayer::compileShaders() {
    using namespace opengl;
    char preproc[512];
    CLEAR_GFXERR_DEBUG
    float fmax = std::numeric_limits<float>::max() - 1.0f;
    snprintf(preproc, sizeof(preproc), "#define INSTANCE_WIDTH %d\n", proInstanceWidth_);
    proShader_ = ShaderRepository::compileShaderPair("shaders/sequence/tokenscoring_projection.vert", "shaders/sequence/tokenscoring_projection.frag", preproc, typeid(this), context());
    proShader_->bindAttributeLocation("attributes0", 0);
    proShader_->link();
    snprintf(preproc, sizeof(preproc) - 1, "#define FLT_MAX %.10e\n#define BUFFER_SIZE %d\n", fmax, vocabAggregateSize_);
    pass1FlatShader_ = ShaderRepository::compileShaderPair("shaders/sequence/tokenscoring_flat_pass1.vert", "shaders/sequence/tokenscoring_flat_pass1.frag", preproc, typeid(this), context());
    pass1FlatShader_->bindAttributeLocation("attributes0", 0);
    pass1FlatShader_->link();
    snprintf(preproc, sizeof(preproc) - 1, "#define FLT_MAX %.10e\n", fmax);
    pass2FlatShader_ = ShaderRepository::compileShaderPair("shaders/sequence/tokenscoring_flat_pass2.vert", "shaders/sequence/tokenscoring_flat_pass2.frag", preproc, typeid(this), context());
    pass2FlatShader_->bindAttributeLocation("attributes0", 0);
    pass2FlatShader_->link();
    snprintf(preproc, sizeof(preproc) - 1, "#define FLT_MAX %.10e\n#define SCATTER_WIDTH %d\n", fmax, SCATTER_WIDTH);
    scatterShader_ = ShaderRepository::compileShaderPair("shaders/sequence/tokenscoring_scatter.vert", "shaders/sequence/tokenscoring_scatter.frag", preproc, typeid(this), context());
    scatterShader_->bindAttributeLocation("attributes0", 0);
    scatterShader_->link();
    switch (scoring_) {
        case ScoringType::GREEDY:
            strcat(preproc,"#define GREEDY\n");
            break;
        case ScoringType::TOP_K:
            strcat(preproc,"#define TOP_K\n");
            break;
        case ScoringType::TOP_P:
            THROW_EXCEPTION_ARGS(FynException,"Not supported yet");
    }
    selectionShader_ = ShaderRepository::compileShaderPair("shaders/sequence/tokenscoring_selection.vert", "shaders/sequence/tokenscoring_selection.frag", preproc, typeid(this), context());
    selectionShader_->bindAttributeLocation("attributes0", 0);
    selectionShader_->link();
    prepShaders();
    assert(glGetError() == GL_NO_ERROR);
}


/**
 * @brief Prepare shader uniforms
 *
 * Sets static uniform values for the (linked) shaders for older GL versions that do not support
 * binding.
 */
void TokenScoringLayer::prepShaders() {
    assert(proShader_->isLinked());
    assert(pass1FlatShader_->isLinked());
    if (!GLInfo::hasBinding()) {
        proShader_->bind();
        proShader_->setUniformValue("inputEmbeddings", 0);
        proShader_->setUniformValue("vocabulary", 1);
        proShader_->unbind();
        pass1FlatShader_->bind();
        pass1FlatShader_->setUniformValue("projection", 0);
        pass1FlatShader_->unbind();
        pass2FlatShader_->bind();
        pass2FlatShader_->setUniformValue("pass1DataA", 0);
        pass2FlatShader_->setUniformValue("pass1DataB", 1);
        pass2FlatShader_->unbind();
        scatterShader_->bind();
        scatterShader_->setUniformValue("projection", 0);
        scatterShader_->setUniformValue("stats", 1);
        scatterShader_->unbind();
        selectionShader_->bind();
        selectionShader_->setUniformValue("tokenData", 0);
        selectionShader_->unbind();
    }
}


/**
 * @brief Generate proxy geometry for the various passes
 *
 * Generates a textured quad for the projection pass as well as a set of placeholder points for
 * the scatter / selection passes.
 */
void TokenScoringLayer::proxyGeometry() {
    using namespace opengl;
    const auto [arr, verts, inds] = rudiments::ProxyGenerator::texturedQuad(context());
    proArray_ = arr;
    proVerts_ = verts;
    proIndices_ = inds;
    // ----------------------------------------------------
    // Scattering...
    // ----------------------------------------------------
    std::unique_ptr<GLuint[]> indices(new GLuint[tableRows_]);
    for (GLuint i=0, *ptr=indices.get(); i < (GLuint)tableRows_; i++) ptr[i] = i;
    scatterArray_ = new VAO(context());
    scatterArray_->bind();
    scatterVerts_  = new VBO(context());
    scatterArray_->enableArray(0);
    scatterVerts_->setBufferData((void *)indices.get(), (GLsizei)(tableRows_ * sizeof(GLuint)), GL_STATIC_DRAW);
    scatterVerts_->bind();
    scatterArray_->setVertexAttributeBuffer(0, 1, GL_UNSIGNED_INT, 0, 0);
    scatterArray_->unbind();
}


/**
 * @copydoc GPULayerBase::updateFBOs
 */
void TokenScoringLayer::updateFBOs() {
    framebuffers_[0]->bind();
    framebuffers_[0]->updateColorAttachment(GL_COLOR_ATTACHMENT0, outputTextures_[0]);
    framebuffers_[0]->unbind();
    outputChanged_ = false;
}


/**
 * @copydoc GPULayerBase::setupFBOs
 */
void TokenScoringLayer::setupFBOs() {
    assert(outputTextures_.size() == 1);
    CLEAR_GFXERR_DEBUG
    scatterMatches_ = opengl::Texture2D(SCATTER_WIDTH, 2, Texture::pixtype::FLOAT32, 4, true);
    scatterFBO_ = new FBO(context(), SCATTER_WIDTH, 2, 1, Texture::pixtype::UINT32_INTEGRAL);
    scatterFBO_->addTexture(GL_COLOR_ATTACHMENT1, scatterMatches_);
    scatterFBO_->unbind();
    glGenRenderbuffers(1, &scatterDepth_);
    glBindRenderbuffer(GL_RENDERBUFFER, scatterDepth_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, SCATTER_WIDTH, 2);
    scatterFBO_->addRenderbuffer(GL_DEPTH_ATTACHMENT, scatterDepth_);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    scatterFBO_->unbind();
    framebuffers_.push_back(new FBO(context(), 1, height_, outputTextures_.at(0)));
    assert(glGetError() == GL_NO_ERROR);
}


/**
 * @copydoc GPULayerBase::getInputOrder
 */
BufferSpec::order TokenScoringLayer::getInputOrder(int port) const {
    return BufferSpec::order::GPU_SEQUENCE;
}


/**
 * @copydoc GPULayerBase::getOutputOrder
 */
BufferSpec::order TokenScoringLayer::getOutputOrder(int port) const {
    return BufferSpec::order::GPU_SEQUENCE;
}


/**
 * @copydoc GPULayerBase::getInputType
 */
BufferSpec::dtype TokenScoringLayer::getInputType(int port) const {
    return GPULayerBase::TEXTURE_TYPE_DEFAULT;
}


/**
 * @copydoc GPULayerBase::getOutputType
 */
BufferSpec::dtype TokenScoringLayer::getOutputType(int port) const {
    return BufferSpec::dtype::UINT32;
}

} // fyusion::fyusenet::gpu::sequence namespace

// vim: set expandtab ts=4 sw=4:
