//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Deep Concatenation Layer (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../../gl/gl_sys.h"
#include "../../gl/uniformstate.h"
#include "../../gl/fbo.h"
#include "../../gl/vbo.h"
#include "../../gl/ibo.h"
#include "../../gl/vao.h"
#include "../../gl/shaderprogram.h"
#include "../concatlayerbuilder.h"
#include "deeplayerbase.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {
namespace gpu {
namespace deep {


/**
 * @brief Perform concatenation of several deep format tensors into a target (deep) tensor
 *
 * This class implements a concatenation layer, which is able to concatenate an variable amount
 * of input tensors into a single output tensor. Deep-formatted tensors always require a render
 * step to concatenate them into a new tensor, unlike shallow tensors.
 *
 * The interface is exactly the same as for the shallow pendant in vanilla::ConcatLayer.
 *
 * @todo Derive this class and vanilla::ConcatLayer from a common interface for cleanliness and
 *       also add a version that concatenates shallow layers into deep layers to save on conversion
 *       layers
 *
 * @note The concatentation is currently restricted regarding the application of activation
 *       functions to the input. Either \e all inputs have the same activation functions or
 *       \e none of the inputs have an activation. It is currently not possible to mix
 *       these.
 */
class DeepConcatLayer : public DeepLayerBase {
 public:
    enum {
      UNIFORM_NUMTEX=1
    };

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    DeepConcatLayer(const ConcatLayerBuilder & builder,int layerNumber);
    virtual ~DeepConcatLayer();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual void cleanup() override;
    virtual void setup() override;
    virtual void forward(uint64_t sequence) override;
    virtual void addInput(int inputDepth,int inputPadding);
    virtual std::vector<BufferSpec> getRequiredInputBuffers() const override;
    virtual std::vector<BufferSpec> getRequiredOutputBuffers() const override;
    virtual int numInputPorts() const override;
    virtual int getPortChannelIndex(int port) const override;
    virtual int numInputChannels(int port=0) const override;

 protected:

   /**
    * @brief Helper structure for tracking information related to individual render passes
    */
    struct RenderPassTexEnv {
        void clear() {
            numTextures_ = 0;
            elementOffset_ = 0;
            channels_ = 0;
            numElements_ = 0;
            outputs_ = 0;
            textureIndices_[0] = -1;
            textureIndices_[1] = -1;
            textureIndices_[2] = -1;
            textureIndices_[3] = -1;
            for (int i=0; i < 4; i++) {
                shifts_[i] = 0;
                components_[i] = PIXEL_PACKING;
            }
        }
        void init(int elemOffset,int channels,int shift,GLint texID) {
            outputs_ = 0;
            numTextures_ = 1;
            elementOffset_ = elemOffset;
            textureIndices_[0] = texID;
            shifts_[0] = shift;
            channels_ = channels;
            components_[0] = channels;
            numElements_ = 1;
        }
        int outputs_ = 0;
        int channels_ = 0;
        int elementOffset_ = 0;
        int numTextures_ = 0;
        int numElements_ = 0;
        int textureIndices_[4] = {0, 0, 0, 0};
        unsigned char shifts_[4] = {0, 0, 0, 0};
        unsigned char components_[4] = {0, 0, 0, 0};
    };

    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void setupShaders();
    void setupNetworkPolygons(VAO *vao);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::vector<DeepTiler *> inputTilers_;  //!< Tiler instances for each input
    programptr shader_;                     //!< Actual concatenation shader
    unistateptr shaderState_;               //!< Uniform state for #shader_
    VAO *vertexArray_ = nullptr;            //!< Pointer to vertex-array object which maintains the VBO / IBO config
    VBO *positionBuffer_ = nullptr;         //!< %VBO that stores the position information of the vertices
    VBO *texCoord1Buffer_ = nullptr;        //!< %VBO that stores texture coordinates for a single render pass with up to 2 inputs (index 0,1)
    VBO *texCoord0Buffer_ = nullptr;        //!< %VBO that stores texture coordinates for a single render pass with up to 4 inputs (index 2,3)
    VBO *texCompBuffer_ = nullptr;          //!< %VBO that stores number of components in input textures to transfer
    VBO *texShiftBuffer_ = nullptr;         //!< %VBO that stores shift values in case non-multiple-of 4 components are encountered
    IBO *indexBuffer_ = nullptr;            //!< Index buffer object that stores the polygon connectivity
    /**
     * @brief Render-pass environments that store relevant short-hand information for execution
     */
    std::vector<RenderPassTexEnv> passEnvironments_;
};

} // deep namespace
} // gpu namespace
} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
