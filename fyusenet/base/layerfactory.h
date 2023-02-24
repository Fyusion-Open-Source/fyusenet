//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Layer Factory (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <unordered_map>
#include <memory>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gpu/gfxcontextlink.h"
#include "layerfactoryinterface.h"
#include "layerbase.h"
#include "compiledlayers.h"

namespace fyusion {
namespace fyusenet {
//------------------------------------- Public Declarations ----------------------------------------

class LayerFactoryBackend;

/**
 * @brief Base class for neural network layer-factories
 *
 * This class provides the interface for a factory that creates a complete neural network layer by
 * layer by translating a set of builders, which serve as input to the factory, into a set of
 * neural network layers that satisfy the parameters supplied in those builders.
 *
 * Factories are target-hardware specific in several ways. First in a generic way, as in the target
 * to perform the compute in is the CPU or the GPU (or the IPU/NPU if supported). Second in a
 * target-subtype-specific way (handled via factory backends), which may generate layers that are
 * optimized for a certain target GPU/CPU/IPU.
 *
 * The usual way to instantiate a layer factory is shown below:
 * @code
 * LayerFactory factory = LayerFactory::instance(LayerFactory::GPUFactoryType(LayerFactory::GPUFactoryType::VANILLA));
 * @endcode
 *
 * which in that case will generate a layer factory that creates layers which perform the computation
 * on the GPU and those layers are not optimized for a specific GPU subtype. Note that all layer
 * factories support creating layers on the CPU, as they may be needed to perform some of the last
 * bits of processing, even when predominantly using GPU layers.
 *
 * @todo The instantiation pattern is not really nice, improve on that in the future.
 */
class LayerFactory : LayerFactoryInterface {
 public:

    /**
     * @brief Structure that represents a generic factory type (e.g. CPU, GPU, IPU)
     */
    struct FactoryType {

        FactoryType(compute_device t):factoryType(t) {
        }

        virtual LayerFactoryBackend * createBackend() = 0;

        compute_device factoryType;
    };


    /**
     * @brief GPU-specific factory type
     */
    struct GPUFactoryType : FactoryType {
        enum gputype {
            VANILLA = 0,            //!< Vanilla GPU layers, not optimized for any GPU in particular
            SPECIALIZED             //!< GPU-model specific layers, will be optimized for the GPU found on runtime system
        };

        GPUFactoryType(gputype tp, GfxContextLink context = GfxContextLink()) :
            FactoryType(compute_device::DEV_GPU), gpuType(tp), gfxContext(context) {
        }

        virtual LayerFactoryBackend * createBackend() override;

        gputype gpuType;
        GfxContextLink gfxContext;
    };

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    virtual ~LayerFactory();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    std::string getName() const;
    virtual void pushBuilder(LayerBuilder *builder) override;
    virtual CompiledLayers compileLayers();

    /**
     * @brief Get a usable LayerFactory instance
     *
     * @param typ Backend type to use
     *
     * @return Shared pointer to LayerFactory instance
     *
     * @note This is \b not a singleton pattern, in fact \b new instances are generated with
     *       every call to this function and every factory keeps track of the layers it created.
     *       After the layers have been compiled, it is safe to discard a factory object again.
     */
    template<class T>
    static std::shared_ptr<LayerFactory> instance(T typ) {
#ifdef DEBUG
        return std::shared_ptr<LayerFactory>(instanceInternal<T>(typ, true));
#else
        return std::shared_ptr<LayerFactory>(instanceInternal<T>(typ, false));
#endif
    }
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    template<class T>
    static LayerFactory * instanceInternal(T backendType, bool debug);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    LayerFactory(LayerFactoryBackend *backend);
    LayerFactoryBackend *backend_;                      //!< Pointer to target-specific factory backend
    LayerFactoryBackend *cpuBackend_;                   //!< CPU factory backend (present in every factory)
    std::unordered_map<int,LayerBuilder *> builders_;   //!< Map of builders that contain the information about the layers to be built
    CompiledLayers layers_;
};



/**
 * @brief Interface for layer factory backends
 *
 * As certain types of layer factories - for example GPU layer factories - might want to enable
 * specific optimizations for GPUs that are found in the runtime system, the layer factory itself
 * delegates a lot of the factory work to a LayerFactoryBackend instance. The backend
 */
class LayerFactoryBackend {
    friend class LayerFactory;
    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    /**
     * @brief Retrieve the name of the factory backend (for debug/logging purposes)
     *
     * @return String with the name of the backend
     */
    virtual std::string getName() const=0;


    /**
     * @brief Create layer based on the supplied layer type and associated builder object
     *
     * @param type Layer type to build
     *
     * @param builder Pointer to builder that contains layer-specific data
     *
     * @param layerNumber Number of the layer to be built
     *
     * @return Raw pointer to layer that has been created
     *
     * @throws FynException in case there was a problem building the layer
     *
     * Instantiates a new layer of the supplied type, fulfilling all the requirements presented in
     * the supplied \p builder object.
     */
    virtual fyusenet::LayerBase * createLayer(LayerType type, LayerBuilder * builder, int layerNumber)=0;
 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    /**
     * @brief Destructor
     */
    virtual ~LayerFactoryBackend() {
    }
};


/**
 * @brief Get name of layer factory (for debug / logging purposes)
 *
 * @return String with the name of the factory
 */
inline std::string LayerFactory::getName() const {
    return backend_->getName();
}


} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
