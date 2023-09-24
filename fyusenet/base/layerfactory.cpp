//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Layer Factory
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "layerfactory.h"
#include "../gpu/gpulayerfactory.h"
#include "../cpu/cpulayerfactory.h"

//-------------------------------------- Global Variables ------------------------------------------


namespace fyusion::fyusenet {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Destructor
 *
 * Deallocates resources. Does \b not deallocate any layers that were created by this factory.
 */
LayerFactory::~LayerFactory() {
    delete cpuBackend_;
    delete backend_;
    cpuBackend_ = nullptr;
    backend_ = nullptr;
    for (auto it = builders_.begin(); it != builders_.end(); ++it) {
        delete it->second;
    }
    builders_.clear();
}



/**
 * @brief Add a builder instance to the list of layers to be built and transfer ownership
 *
 * @param builder Pointer to LayerBuilder instance
 *
 * This function adds the supplied builder object to the internal list of layer builders. A call
 * to createLayers() will generate a keyed map of layers that can be used for network inference.
 * Note that the ownership of the supplied \p builder is transferred to the factory.
 *
 * @throws FynException in case of errors (unsupported layer types or double-use of layer numbers)
 */
void LayerFactory::pushBuilder(LayerBuilder *builder) {
    assert(builder);
    if (builder->number_ < 0) THROW_EXCEPTION_ARGS(FynException,"Must identify each layer with a valid number (found %d in the builder)", builder->number_);
    if (builder->type_ >= LayerType::LAST_SUPPORTED) {
        THROW_EXCEPTION_ARGS(FynException,"Unsupported layer type %d", builder->type_);
    } else {
        if (builders_.find(builder->number_) != builders_.end()) THROW_EXCEPTION_ARGS(FynException,"Trying to insert a layer on a position that is already taken (%d)",builder->number_);
        builders_[builder->number_] = builder;
    }
}


/**
 * @brief Create the actual layer instances basedd on the builders stored in the factory
 *
 * @return Repository that contains the compiled layers
 *
 * This function creates all layers from the builders that are stored in this factory and stores
 * them in a map which maps the layer number to the raw pointer of the layer. This map can be
 * used to "execute" the neural network by invoking \c forward() on the layers in the map in
 * sequential key order. This invocation is handled by the Engine.
 *
 * @see Engine
 */
CompiledLayers LayerFactory::compileLayers() {
    CompiledLayers layers;
    for (auto it = builders_.begin(); it != builders_.end(); ++it) {
        if (it->second->device_ == compute_device::DEV_CPU) {
            layers.setLayer(cpuBackend_->createLayer(it->second->type_, it->second, it->second->number_));
        } else {
            layers.setLayer(backend_->createLayer(it->second->type_, it->second, it->second->number_));
        }
    }
    return layers;
}



/**
 * @brief Generate an instance of the layer factory with a target-specific backend
 *
 * @param backendType Type of factory to instantiate (CPU, GPU or IPU/NPU)
 * @param debug Indicator whether or not the call comes from a debug build
 *
 * @return Pointer to object that is derived from LayerFactory and wraps a target-specific backend
 *
 * @throws FynException on debug/release build mismatch
 *
 * The factory backend is the target-hardware-specific part of the layer factory. It may differ
 * depending on the type of CPU/GPU that was detected during runtime and will be generated here
 * and wrapped by an object that implements the LayerFactory interface. The \p debug parameter
 * is passed from the instantiation call in the header-file to make sure that when FyuseNet is
 * used from a shared library, the debug/release modes are consistent.
 *
 * @see createBackend
 */
template<class T>
LayerFactory * LayerFactory::instanceInternal(T backendType, bool debug) {
#ifdef DEBUG
    // TODO (mw) should we be that pedantic ?
    if (!debug) THROW_EXCEPTION_ARGS(FynException,"This fyusenet library is from a debug build and is not compatible with the release build");
#else
    // TODO (mw) should we be that pedantic ?
    if (debug) THROW_EXCEPTION_ARGS(FynException,"This fyusenet library is from a release build and is not compatible with the build");
#endif
    LayerFactoryBackend *backend = backendType.createBackend();
    if (!backend) THROW_EXCEPTION_ARGS(FynException, "Cannot create backend");
    return new LayerFactory(backend);
}


/**
 * @brief Create target-specific layer-generator backend
 *
 * @return Pointer to instance that implements the LayerFactoryBackend interface
 *
 * The factory backend is the target-hardware-specific part of the layer factory. It may differ
 * depending on the type of CPU/GPU that was detected during runtime.
 */
LayerFactoryBackend * LayerFactory::GPUFactoryType::createBackend() {
    using namespace gpu;
    switch (gpuType) {
        case SPECIALIZED:
#ifndef ANDROID
            return new GPULayerFactoryBackend(gfxContext);
#else
            switch (opengl::GLInfo::getGPUType()) {
                case opengl::GLInfo::ARM_MALI:
                    // NOTE (mw) currently disabled, Mali specific code is not part of the public release
                    //return MaliLayerFactory(gfxContext);
                    return new GPULayerFactoryBackend(gfxContext);
                default:
                    return new GPULayerFactoryBackend(gfxContext);
            }
#endif
            break;
        case VANILLA:
            return new GPULayerFactoryBackend(gfxContext);
        default:
            return nullptr;
    }
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param backend Backend to wrap which does the heavy-lifting
 */
LayerFactory::LayerFactory(LayerFactoryBackend *backend) : backend_(backend) {
    cpuBackend_ = new cpu::CPULayerFactoryBackend();
}


/*##################################################################################################
#                  E X P L I C I T   T E M P L A T E    I N S T A N T I A T I O N S                #
##################################################################################################*/


template std::shared_ptr<LayerFactory> LayerFactory::instance<LayerFactory::GPUFactoryType>(LayerFactory::GPUFactoryType typ);
template LayerFactory * LayerFactory::instanceInternal<LayerFactory::GPUFactoryType>(LayerFactory::GPUFactoryType backendType, bool debug);


} // fyusion::fyusenet namespace


// vim: set expandtab ts=4 sw=4:
