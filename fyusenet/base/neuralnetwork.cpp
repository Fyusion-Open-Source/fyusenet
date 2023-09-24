//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Neural Network Base Class
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "neuralnetwork.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion::fyusenet {

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor (idle)
 */
NeuralNetwork::NeuralNetwork(const GfxContextLink& ctx) {
    setContext(ctx);
}

/**
 * @brief Destructor
 */
NeuralNetwork::~NeuralNetwork() {
    assert(engine_ == nullptr);
    if (engine_) FNLOGW("Please call cleanup() before deleting network instance");
}


/**
 * @brief Cleanup and deallocate (GPU) resources taken by the network
 *
 * @pre GL context that is associated to this network must be curren to the calling thread. Also,
 *      the finish() method shall have been called prior to the cleanup.
 *
 * This function performs a cleanup of most resources consumed by the neural network, in particular
 * it will deallocate GPU resources taken by the net, such as buffers and textures. Note that
 * the GLSL shaders are kept in a central shader cache which will not be cleaned by this function.
 *
 * @see finish, Engine::cleanup
 */
void NeuralNetwork::cleanup() {
    assert(setup_);
#ifndef FYUSENET_MULTITHREADING
    assertContext();
#endif
    auto broom = [this]() {
        if (bufferMgr_) bufferMgr_->cleanup();
        delete bufferMgr_;
    };
    if (engine_) engine_->cleanup(broom);
    delete engine_;
    engine_ = nullptr;
    setup_ = false;
}


/**
 * @brief Setup neural network / allocate (GPU) resources
 *
 * @pre GL context that is associated to this network must be current to the calling thread in case
 *      of non-multithreaded / synchronous operation.
 *
 * This function sets up the neural network by instantiating and initializing all layers and
 * reserving resources for the intermediate tensor buffers. Depending on whether or not support
 * for multi-threading was compiled in, the setup either instantiates all resources within the
 * GL context of the calling thread, or it spawns an "engine thread" which will create all
 * GL resources. The engine thread will run with a GL context that is shared with the calling
 * context, such that textures can be interchanged.
 *
 * @see Engine::setup, glSetup
 */
void NeuralNetwork::setup() {
    if (setup_) return;
    assert(engine_ == nullptr);    
#ifdef FYUSENET_MULTITHREADING
    engine_ = new Engine(context(), async_);
#else
    assertContext();
    engine_ = new Engine(context(), false);
#endif
    engine_->setup(this);
    setup_ = true;
#ifdef FYUSENET_MULTITHREADING
    if (asyncCallbacks_.newSeq_) engine_->setNewSequenceCallback(asyncCallbacks_.newSeq_);
    if (asyncCallbacks_.seqDone_) engine_->setSequenceCallback(asyncCallbacks_.seqDone_);
#endif
}


/**
 * @brief Flushes pending operations in the network
 *
 * @return Combination of engine execution state and sequence ID that was assigned to this run
 *
 * @pre GL context that is associated to this network must be current to the calling thread
 *      for non-multithreaded / synchronous operation.
 *
 * This function flushes pending operations in the network until all operations have been fully
 * executed. Use this function prior to taking down the neural network to make sure that no async
 * operation is still running in the background.
 *
 * @warning This function is not re-entrant. Please only use it from a single thread.
 */
NeuralNetwork::execstate NeuralNetwork::finish() {
    assert(setup_);
#ifndef FYUSENET_MULTITHREADING
    assertContext();
#endif
    execstate rc;
    if (engine_) {
        engine_->finish();
        rc.status = Engine::EXEC_DONE;
    } else {
        rc.status = Engine::EXEC_ERROR;
    }
    rc.sequenceNo = engine_->lastSequenceNo();
    return rc;
}


/**
 * @brief Execute neural network without any state token
 *
 * @return Combination of engine execution state and sequence ID that was assigned to this run
 *
 * This is merely a convenience function that invokes forward(StateToken *token) with a \c nullptr
 * as token.
 */
NeuralNetwork::execstate NeuralNetwork::forward() {
    return forward(nullptr);
}


/**
 * @brief Execute neural network
 *
 * @param token Optional pointer to a StateToken that tracks and controls inference state
 *
 * @return Combination of engine execution state and sequence ID that was assigned to this run
 *
 * @pre GL context that is associated to this network must be current to the calling thread
 *
 * This function executes the network by iterating over all layers in enumeration order and in
 * turn call LayerBase::forward() on these. For multi-threaded builds and layers that support
 * asynchronous operation (which are the upload and download layers), this function may choose
 * to return \e before all layers have been executed and defer further execution to a engine
 * thread that runs in the background.
 *
 * The sequence ID that is part of the return code is a strictly monotonous identifier which is
 * assigned to each forward run in the engine and which will be returned by this function as part
 * of the execution state. This sequence ID can be tracked in callbacks for example. Please note
 * that due to the nature of asynchronous execution and use of callbacks, a callback \e may be
 * called \e prior to this function returning. In which case the callback receiver would not know
 * about the sequence ID as it is assigned internally. The caller is responsible for establishing a
 * mechanism that is able to cope with "unknown" sequence IDs.
 *
 * As for the "state" part of the returned execution state, the following states are defined.
 *   - \c EXEC_DONE indicates that the network was fully executed, possibly pending async operations
 *   - \c EXEC_DEFERRED indicates that the network was not fully executed and is waiting for an async
 *         operation to finish using a background engine thread
 *   - \c EXEC_STOPPED indicates that the network has been torn down or is in the process of being
 *        torn down and cannot operate anymore
 *   - \c EXEC_ERROR indicates that there was an error during network execution
 *
 * @todo Implement a fixed callback that will be called prior to executing the network which will
 *       inform subscribers about new sequence IDs to prevent unknown sequence IDs.
 *
 * @warning This function is not re-entrant. Please only use it from a single thread.
 *
 * @see finish
 */
NeuralNetwork::execstate NeuralNetwork::forward(StateToken *token) {
    assert(setup_);
#ifndef FYUSENET_MULTITHREADING
    assertContext();
#endif
    execstate state;
    if (engine_) {
        state.status = engine_->forwardLayers(token);
        state.sequenceNo = engine_->lastSequenceNo();
    } else {
        state.status = Engine::EXEC_STOPPED;
        state.sequenceNo = 0;
    }
    return state;
}


#ifdef FYUSENET_MULTITHREADING
/**
 * @brief Enable asynchronous (upload/download) operation prior to setup
 *
 * @param adapter Reference to optional AsyncAdapter that contains the asynchronous callback profile
 *
 * This function enables asynchronous operation of the network and sets optional callback functions
 * that are to be used for notification purposes.
 *
 * @note This function must be invoked before calling setup()
 *
 * @throw FynException if the network was not in the correct state for switching it to asynchronous mode
 *
 * @see AsyncAdapter
 */
void NeuralNetwork::asynchronous(const AsyncAdapter & adapter) {
    if (engine_ || setup_) {
        THROW_EXCEPTION_ARGS(FynException, "Network must be switched to asynchronous before calling setup()");
    }
    async_ = true;
    asyncCallbacks_ = adapter;
}
#endif


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Instantiate layers and initialize GL resources
 *
 * @return Compound object that contains all compiled layers
 *
 * This function sets up the OpenGL specific part of the neural network by calling overriden
 * (abstract) initialization methods, starting with buildLayers(), which should contain an
 * implementation of using the layer factories to instantiate the actual layers. After that, the
 * connectLayers() function will be invoked, which establishes the network connectivity and
 * allocates GPU resources for the intermediate tensors. This is followed by the weight
 * initialization of the network layers and finally LayerBase::setup() is invoked on every layer.
 *
 * This function may either be called directly from the main thread (if multithreading is not
 * compiled in), or from the engine thread. It is important to perform all inference calls to the
 * created network from the same thread, because the intermediate %FBOs that the layers write to
 * are not shared among GL contexts.
 *
 * @see CompiledLayers, Engine::asyncSetup, buildLayers, connectLayers, initializeWeights
 */
CompiledLayers NeuralNetwork::gpuSetup() {
    assert(engine_);
    CompiledLayers layers = buildLayers();
    // TODO (mw) should we allow for an already existing buffer manager ?
    if (!bufferMgr_) bufferMgr_ = new BufferManager(context());
    connectLayers(layers, bufferMgr_);
    initializeWeights(layers);
    for (auto it = layers.begin(); it != layers.end(); ++it) {
        assert(it.second);
        it.second->setup();
    }
    return layers;
}


/**
 * @brief Obtain network layer factory for a specific compute device type
 *
 * @param dev Compute device type to obtain layer factory for
 *
 * @return LayerFactory instance which can be used in conjunction with the LayerBuilder (and derived)
 *                      class(es) to generate layers.
 */
std::shared_ptr<LayerFactory> NeuralNetwork::getLayerFactory(compute_device dev) {
    if (dev != compute_device::DEV_GPU) THROW_EXCEPTION_ARGS(FynException,"We currently only support GPU networks");
    switch (dev) {
        case compute_device::DEV_CPU:
            assert(false);
        case compute_device::DEV_NPU:
            assert(false);
        default:
            return LayerFactory::instance(LayerFactory::GPUFactoryType(LayerFactory::GPUFactoryType::SPECIALIZED));
    };
}

#ifdef FYUSENET_GL_BACKEND
/**
 * @brief Get OpenGL output FBO from specified layer
 *
 * @param layer Pointer to layer to retrieve FBO from
 * @param index FBO index within the output FBOs (defaults to 0)
 *
 * @return Pointer to FBO object at specified index/layer, may be a \c nullptr
 *
 * @warning This function is only available in OpenGL builds
 */
fyusion::opengl::FBO * NeuralNetwork::getFBO(const gpu::GPULayerBase * layer, int index) {
    if (!layer) THROW_EXCEPTION_ARGS(FynException, "Cannot work with null layer");
    return layer->getFBO(index);
}
#endif


} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:

