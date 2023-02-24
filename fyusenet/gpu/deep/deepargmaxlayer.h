//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep ArgMax Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

#ifdef ANDROID
#include <GLES3/gl3.h>
#else
#ifdef __APPLE__
#include <OpenGL/gl3.h>
#include <OpenGL/glext.h>
#else
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#endif
#endif

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/fbo.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/vao.h"
#include "../../gl/uniformstate.h"
#include "../../base/bufferspec.h"
#include "deeplayerbase.h"
#include "../argmaxlayerbuilder.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {


/**
 * @brief Approximate argmax layer for deep tensors
 *
 * This layer performs argmax/max function over all channels of a tensor, resulting in the
 * index (arg) and actual max values for each element in the spatial domain by scanning all
 * channels. The argmax is not exact and accuracy is traded for speed. This layer should therefore
 * not be used to perform an argmax on a single classification (e.g. after an FC layer on a
 * classification-type network).
 *
 * For cases where a per-pixel classification is desired and some additional post-processing to
 * smooth-out errors is implemented, this implementation usually performs well enough.
 *
 * The main trick that is used here is to mix the argument position (its channel number) with the
 * value of the argument itself. We do this by masking out some bits of the floating-point
 * representation of the values and replace those by the channel number using some bit arithmetic.
 *
 * The way it work is quite simple, when looking at a 32-bit single-precision FP following the
 * IEEE-754 standard, it looks like following:
 *
 * @code
 * /1\/------------------- 23 ------------------\/------- 8 ------\
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |S|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|M|E|E|E|E|E|E|E|E|
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * @endcode
 *
 * It has a leading sign bit, 23-bit mantissa and an 8-bit exponent. We can use the least significant
 * bits and swap them out for an integer representation of the channel number that is associated
 * with a value and recover them later on.
 *
 * On the implementation side, we simply use the ROPs max-blending function along with some code
 * in the fragment shaders to comnpute the maximum. As a side-effect, the 2nd channel of the output
 * will be the maximum value that matches the index in the first channel.
 *
 * As is obvious from that approach, this will lead to multiple forms of imprecisions/errors. First,
 * by removing bits from the mantissa representation, we introduce an additional truncation error
 * and second, we introduce a bias on top of that by letting the channel number mimick bits of the
 * original value.
 *
 * TLDR: the results returned by this layer are not 100% accurate and they might lead to false
 * argmax responses. However the false maxima (not their argument) would be very close to the true
 * maximum. If your classes are well separated, this should not pose a problem, however we highly
 * recommend not using this layer for classification-only problems. In the past, we used this
 * layer to run classification on a per-pixel level where we would accept noisy output and it
 * performed really well for us.
 */
class DeepArgMaxLayer : public DeepLayerBase {
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepArgMaxLayer(const ArgMaxLayerBuilder & builder, int layerNumber);

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void setup() override;
    virtual void cleanup() override;
    virtual void forward(uint64_t sequence) override;
    virtual std::vector<BufferSpec> getRequiredInputBuffers() const override;
    virtual std::vector<BufferSpec> getRequiredOutputBuffers() const override;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupNetworkPolygons();
    virtual void setupFBOs() override;
    void setupShaders();
    unistateptr initShader(programptr shader);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    VAO *pass1VAO_ = nullptr;                 //!< Vertex array object for 1st pass render
    VBO *pass1VBOA_ = nullptr;                //!< %VBO for vertex coordinates, 1st pass render
    VBO *pass1VBOB_ = nullptr;                //!< %VBO for channel offsets, 1st pass render
    VBO *pass1VBOC_ = nullptr;                //!< %VBO for channel masking, 1st pass render
    IBO *pass1IBO_ = nullptr;                 //!< Polygon connectivity for 1st pass render
    VAO *pass2VAO_ = nullptr;                 //!< Vertex array object for 2nd pass (postproc) render
    VBO *pass2VBO_ = nullptr;                 //!< Vertex coordinates for 2nd pass
    IBO *pass2IBO_ = nullptr;                 //!< Polygon connectivity for 2nd pass
    FBO *pass1FBO_ = nullptr;                 //!< %FBO to be used as render target for the 1st pass
    unistateptr pass1State_;                  //!< Shader state for 1st pass render
    unistateptr pass2State_;                  //!< Shader state for 2nd pass render
    programptr pass1Shader_;                  //!< Shader for 1st pass render
    programptr pass2Shader_;                  //!< Shader for 2nd pass render
    int channelBits_ = 0;                     //!< Number of bits required to store channel information
    unsigned int pass1Mask_ = 0;
    unsigned int pass2Mask_ = 0;
    constexpr static int MANTISSABITS = 23;   //!< Number of bits for float mantissa (32-bit single FP IEEE-754)
    constexpr static int EXPONENT_MAX = 127;  //!< Maximum exponent value for float (32-bit single FP IEEE-754)
    constexpr static int EXPONENT_MIN = -126; //!< Minimum exponent value for float (32-bit single FP IEEE-754)
    constexpr static int EXPONENT_BITS = 8;   //!< Number of exponent bits in used floating-point representation (32-bit single FP IEEE-754)
    constexpr static int GUARD_BITS = 0;      //!< Number of additional guard bits on the LSB part of the mantissa
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
