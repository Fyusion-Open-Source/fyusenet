//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Compiled Layers Compound (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

//-------------------------------------- Project  Headers ------------------------------------------

#include "layerbase.h"

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion::fyusenet {

namespace gpu {
    class GPULayerFactoryBackend;
}

/**
 * @brief Compounding object for a set of neural network layers
 *
 * This class aggregates a set of neural network layers into a single object which allows for indexing
 * these layers by their layer number. The execution order and thus the overall network behaviour
 * itself, is defined by the layer numbers as they shall be executed in strictly ascending order.
 * Throughout the inference engine, this class serves as the central storage point for the layers.
 *
 * To facilitate access to individual layers, this class features an iterator which can be used
 * to iterate over the layers in ascending order, as well as index-based access either by layer
 * name or layer number.
 *
 * Internally, this class stores a shared pointer to the individual layers and passing this object
 * around via copying is a lightweight operation. Once the last instance is destroyed, the
 * underlying layers are also deleted. It is important to note that in case of GPU layers, the
 * cleanup() functions of all layers shall be called before deleting the layers
 * from memory to make sure that all GL resources are freed.
 *
 * @see LayerBase::cleanup
 */
class CompiledLayers {
    friend class CPULayerFactoryBackend;
    friend class gpu::GPULayerFactoryBackend;
    friend class LayerFactory;
 public:
    /**
     * @brief Iterator for the layers stored in the CompiledLayers object
     */
    struct iterator {
        friend class CompiledLayers;

        /**
         * @brief Constructor for an invalid iterator
         */
        iterator() : first(-1), index_(-1), last_(-1) {
        }

        /**
         * @brief Pre-increment operator
         *
         * Advances the iterator to the next layer in the list
         */
        iterator& operator++() {
            int idx = index_ + 1;
            auto strong = data_.lock();
            // NOTE (mw) this is not optimal if there are huge gaps in the layer enumeration
            if (strong.get()) {
                while ((idx <= last_) && (strong.get()->at(idx) == nullptr)) idx++;
                index_ = idx;
                first = idx;
                second = (idx <= last_) ? strong.get()->at(idx) : nullptr;
            } else {
                index_ = last_ + 1;
                second = nullptr;
                first = index_;
            }
            return *this;
        }

        /**
         * @brief Comparator (equality)
         *
         * @param cmp Object to compare to
         *
         * @retval true iterators are the same
         * @retval false iterators are not the same
         */
        bool operator==(const iterator& cmp) const {
            // we assume that the iterators are set on the same layer set
            return (index_ == cmp.index_);
        }

        /**
         * @brief Comparator (inequality)
         *
         * @param cmp Object to compare to
         *
         * @retval true iterators are not the same
         * @retval false iterators are the same
         */
        bool operator!=(const iterator& cmp) const {
            // we assume that the iterators are set on the same layer set
            return (index_ != cmp.index_);
        }

        /**
         * @brief Check if the current iterator is smaller (=before) a comparison iterator
         *
         * @param cmp Iterator to compare against
         *
         * @retval true if current iterator is smaller
         * @retval false otherwise
         */
        bool operator<(const iterator& cmp) const {
            // we assume that the iterators are set on the same layer set
            return (index_ < cmp.index_);
        }


        /**
         * @brief Check if iterator is valid (not associated with being "in range" of the data)
         *
         * @retval true if iterator is valid
         * @retval false otherwise
         */
        bool valid() const {
            return (data_.lock().get());
        }

        /**
         * @brief Retrieve layer number that the iterator points to
         *
         * @return Layer number of the iterator
         */
        int layer() const {
            return index_;
        }

        int first;                          //!< For compatibility with stdc++ map types
        LayerBase * second = nullptr;       //!< For compatibility with stdc++ map types

    private:
        /**
         * @brief Constructor
         *
         * Generates iterator instance for the layers stored in this class
         */
        iterator(std::shared_ptr<std::vector<LayerBase *>> data, int idx, int last) :
            first(idx), data_(data), index_(idx), last_(last) {
            second = (idx <= last) ? data->at(idx) : nullptr;
        }
        std::weak_ptr<std::vector<LayerBase *>> data_;      //!< Weak pointer to actual layers
        int index_;     //!< Current index in the layer data (equivalent to layer number)
        int last_;      //!< Last valid index in the layer data
    };


    /**
     * @brief Constructor
     */
    CompiledLayers() {
        layers_ = std::shared_ptr<std::vector<LayerBase *>>(new std::vector<LayerBase *>());
    }

    /**
     * @brief Destructor
     *
     * Deallocates layer data if this is the last instance holding them.
     *
     * @pre cleanup() must have been called before deleting the last instance
     */
    ~CompiledLayers() {
        // TODO (mw) thread-safety
        if (layers_.unique()) {
            for (auto * layer : *(layers_)) {
                delete layer;
            }
            layers_->clear();
            layersByName_.clear();
        }
    }

    /**
     * @brief Access layer by layer number
     *
     * @param idx Layer number of the layer to fetch
     *
     * @return Pointer to layer
     *
     * @throws FynException in case a layer with the specified number does not exist in the collection
     */
    LayerBase * operator[](int idx) {
#ifdef DEBUG
        if ((*(layers_))[idx] == nullptr) THROW_EXCEPTION_ARGS(FynException,"Layer number %d does not exist in collection", idx);
#endif
        return (*(layers_))[idx];
    }


    /**
     * @brief Access layer by layer name
     *
     * @param name Name of layer to fetch
     *
     * @return Pointer to layer
     *
     * @throws FynException in case a layer with the specified name does not exist in the collection
     *
     * @warning FyuseNet does currently not require for layers to have unique names, as the primary key
     *          is the layer number. Make sure to have \e unique names for all layers if you want to
     *          use this function. If two layers share the same name, this function will always return
     *          the layer that has the highest layer number with the matching name.
     */
    LayerBase * operator[](const std::string& name) {
        auto it = layersByName_.find(name);
#ifdef DEBUG
        if (it == layersByName_.end()) THROW_EXCEPTION_ARGS(FynException,"Layer %s does not exist in collection", name.c_str());
#endif
        return (it != layersByName_.end()) ? it->second : nullptr;
    }


    /**
     * @brief Perform cleanup of all (GPU) resources used by the layers in this object
     *
     * @pre The GL context that was used to create the layers must be the current context
     *
     * @see LayerBase::cleanup
     */
    void cleanup() {
        for (auto it = layers_->begin(); it != layers_->end(); ++it) {
            if (*it) (*it)->cleanup();
        }
    }


    /**
     * @brief Get iterator to first layer in the list
     *
     * @return Iterator to first layer
     */
    iterator begin() {
        return iterator(layers_, minIndex_, maxIndex_);
    }

    /**
     * @brief Get iterator \e past the last layer in the list
     *
     * @return Iterator that points past the last layer in the list
     */
    iterator end() {
        return iterator(layers_, maxIndex_ + 1, maxIndex_);
    }

 private:

    /**
     * @brief Set/add a layer to the list of layers
     *
     * @param layer Layer to set
     */
    void setLayer(LayerBase *layer) {
        assert(layer->getNumber() >= 0);
        int idx = layer->getNumber();
        if ((int)layers_->size() <= idx) layers_->resize(idx+2, nullptr);
#ifdef DEBUG
        if ((*(layers_.get()))[layer->getNumber()] != nullptr) {
            THROW_EXCEPTION_ARGS(FynException,"A layer (%s) already exists at index %d", (*(layers_.get()))[layer->getNumber()]->getName().c_str(), layer->getNumber());
        }
#endif
        (*(layers_.get()))[layer->getNumber()] = layer;
        minIndex_ = std::min(minIndex_, idx);
        maxIndex_ = std::max(maxIndex_, idx);
        layersByName_[layer->getName()] = layer;
    }


    std::shared_ptr<std::vector<LayerBase *>> layers_;              //!< List of layers that constitute the neural network
    std::unordered_map<std::string, LayerBase *> layersByName_;     //!< Index from layer names to layer numbers
    int minIndex_ = INT32_MAX;                                      //!< First index in the layer list
    int maxIndex_ = INT32_MIN;                                      //!< Last index (inclusive) in the layer list
};


} // fyusion::fyusenet namespace


// vim: set expandtab ts=4 sw=4:

