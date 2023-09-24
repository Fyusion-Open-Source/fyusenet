//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Parameter Provider Interface (Header)                                       (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <any>
#include <atomic>
#include <functional>

//-------------------------------------- Project  Headers ------------------------------------------

#include "layerflags.h"
#include "../common/miscdefs.h"

//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion::fyusenet {

/**
 * @brief Wrapper class that keeps track of reference counts for data blobs (base)
 *
 * This class is used to provide an interface to a raw pointer of underlying data with associated
 * reference counting. A wrapper instance never takes ownership over the data it wraps, but rather
 * is used to (optionally) inform the owner of the data when it is no longer needed. This may
 * be helpful in cases where data is dynamically mapped into memory or directly read from a file
 * and buffered in smaller buffers.
 *
 * It is up to the implementations in derived classes how to handle the reference counting / scoping.
 */
class DataWrapper {
    friend class DataBlob;
 public:
    virtual ~DataWrapper() = default;

    /**
     * @brief Retrieve (raw) pointer to underlying data
     *
     * @return Pointer to underlying data, may be invalid, check with std::any::has_value()
     */
    [[nodiscard]] virtual const std::any get() const = 0;

 protected:

    /**
     * @brief Increase reference count
     */
    virtual void inc() const {
        refCount_.fetch_add(1);
    }

    /**
     * @brief Decrease reference count
     */
    virtual int dec() const {
        return refCount_.fetch_sub(1)-1;
    }

    mutable std::atomic<int> refCount_{0};
};


/**
 * @brief Default data-wrapper class
 *
 * @tparam T Data type that is wrapped
 *
 * This class provides the default implementation of a DataWrapper which basically just performs
 * reference counting and stores a raw pointer to the underlying data. This may be used for
 * situations where the data is fully stored in memory and can be randomly accessed.
 */
template<typename T>
class DefaultDataWrapper : public DataWrapper {
 public:
    explicit DefaultDataWrapper(const T * ptr) : ptr_(ptr) {
    }

    DefaultDataWrapper(const DefaultDataWrapper& src) {
        ptr_ = src.ptr_;
    }

    const std::any get() const override {
        return std::any(ptr_);
    }

protected:
    const T * ptr_;
};


/**
 * @brief Access provider to layer parameter data
 *
 * This class is used to provide pointers to weights, biases and other associated data to the
 * network on a layer-by-layer basis. Technically it is wrapping a DataWrapper instance and
 * provides access to its underlying pointer.
 *
 * Due to the reference counting in the DataWrapper object that is the underlying for this class,
 * this object's life-cycle determines the validity of the pointers stored within it. This means
 * that it is not safe to retrieve a pointer using the get() method and use this pointer after
 * the DataBlob object has been destroyed.
 */
class DataBlob {
 public:

    /**
     * @brief Constructor
     *
     * @param wrapper Pointer to DataWrapper instance that should be maintained by this object
     *
     * @note This class does \b not take ownership over the wrapper
     */
    explicit DataBlob(DataWrapper * wrapper = nullptr) : wrapper_(wrapper) {
        if (wrapper_) wrapper_->inc();
    }

    /**
     * @brief Copy constructor
     *
     * @param src Object to copy from
     */
    DataBlob(const DataBlob &src) {
        wrapper_ = src.wrapper_;
        if (wrapper_) wrapper_->inc();
    }

    /**
     * @brief Assignment operator
     *
     * @param src Object to assign data from
     *
     * @return Reference to this object
     */
    DataBlob& operator=(const DataBlob& src) {
        if (this == &src) return *this;
        if (wrapper_) {
            wrapper_->dec();
        }
        wrapper_ = src.wrapper_;
        if (wrapper_) wrapper_->inc();
        return *this;
    }

    /**
     * @brief Destructor
     */
    ~DataBlob() {
        if (wrapper_) wrapper_->dec();
    }

    /**
     * @brief Retrieve (raw) pointer to underlying data
     *
     * @return Pointer to underlying data, may be invalid, check with std::any::has_value()
     */
    [[nodiscard]] std::any get() const {
        if (wrapper_) return wrapper_->get();
        else return {};
    }

    /**
     * @brief Check if this object is empty
     *
     * @retval true object has data wrapped
     * @retval false object has no data wrapped
     */
    [[nodiscard]] bool empty() const {
        return wrapper_ == nullptr;
    }

 private:
    DataWrapper * wrapper_ = nullptr;
};


/**
 * @brief Access provider to layer parameter data
 *
 * This class is used to provide pointers to weights, biases and other associated data to the
 * network on a layer-by-layer basis. This specific variant is used for access to the data
 * via a mapping mechanism by supplying a function that is called with the underlying pointer.
 * The advantage is that temporary objects of this class can be used to access the data without
 * issues regarding the life-cycle of the underlying data.
 */
class DataBlobMapper {
 public:
    explicit DataBlobMapper(const DataBlob& src) : wrap_(src) {
    }
    void with(const std::function<void(const std::any&)> & func) const {
        func(wrap_.get());
    }
 private:
    const DataBlob wrap_;
};


/**
 * @brief Base class for network parameter providers
 *
 * This class is used to provide weights and other data to the network on a layer-by-layer
 * basis. Actual parameter providers shall derive from this class and implement/override the
 * interface as needed.
 *
 * The following example shows the usage of the ParameterProvider interface:
 * @code
 * ParameterProvider * weights = new MyWeights(...);
 * ConvolutionLayer * layer = <constructor>;
 * <some init code>
 * layer->loadParameters(weights);
 * layer->setup();
 * ....
 * @endcode
 *
 * A ParameterProvider instance may wrap memory or may wrap a file or network calls, depending on the
 * specific implementation. The interface is designed to be agnostic of the actual data source,
 * which allows versatile use from working with blob to streaming in the data.
 *
 * Parameter providers have two data-interface approaches: simple getters and mappers. The difference
 * between the two is that the getter returns an object of which the object lifetime determines
 * the accessibility of the underlying data, see the DataBlob documentation for details.
 * The mapper functionality returns a DataBlobMapper instance which accepts a function pointer
 * to run on the underlying data.
 *
 * @see LayerBase::loadParameters()
 */
class ParameterProvider {
 public:
    ParameterProvider() = default;
    virtual ~ParameterProvider() = default;
    // ------------------------------------------------------------------------
    // Mapper / Getter interface
    // ------------------------------------------------------------------------
    [[nodiscard]] const DataBlobMapper map(const std::string& name, int layerNo, int subIndex) const;
    [[nodiscard]] virtual DataBlob get(const std::string& name, int layerNo, int subIndex) const;

    // ------------------------------------------------------------------------
    // Data type interface
    // ------------------------------------------------------------------------
    /**
     * @brief Get weight(s) data type (if any)
     *
     * @param name Name of the parameter to check the type for
     * @param layerNo Layer number to check the type for
     * @param subIndex Sub-index of the parameter to check the type for, if applicable. Otherwise
     *                 just set to 0
     *
     * @return Weight data type
     */
    [[nodiscard]] virtual param_type dataType(const std::string& name, int layerNo, int subIndex) const {
        return param_type::WGT_DEFAULT;
    }
};


} // fyusion::fyusenet namespace

// vim: set expandtab ts=4 sw=4:

