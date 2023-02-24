//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Pool-Managed PBO Wrapper (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "pbo.h"

//------------------------------------------ Constants ---------------------------------------------


namespace fyusion {
namespace opengl {
//------------------------------------- Public Declarations ----------------------------------------

class PBOPool;

/**
 * @brief Wrapper class for PBO objects that are managed by a PBOPool
 *
 * This class wraps a low-level PBO object and augments it with additional meta-data that is
 * required to track the usage of the wrapped PBO, such that the PBOPool where it originates
 * from can perform proper resource tracking. Since it only wraps the actual PBO, all functionality
 * of a PBO is accessible, simply by dereferencing the ManagedPBO object.
 */
class ManagedPBO {
    friend class PBOPool;
 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ManagedPBO();
    ManagedPBO(const ManagedPBO & src);
    ~ManagedPBO();
    // ------------------------------------------------------------------------
    // Overloaded operators
    // ------------------------------------------------------------------------
    ManagedPBO & operator=(const ManagedPBO & src);

    PBO * operator->() {
        return pbo_;
    }
    PBO * operator*() {
        return pbo_;
    }

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------

    bool isValid() const {
        return (pool_ != nullptr);
    }

    int index() const {
        return pboIndex_;
    }

    bool isPending() const {
        assert(pending_);
        return *pending_;
    }

    void clearPending() const {
        assert(pending_);
        *pending_ = false;
    }

    void setPending() {
        assert(pending_);
        *pending_ = true;
    }

 private:
    ManagedPBO(PBO * pbo, PBOPool * pool, std::atomic<uint32_t> * refcount, bool * pending, int index);
    PBO *pbo_ = nullptr;                            //!< Pointer to actual %PBO
    PBOPool * pool_ = nullptr;                      //!< Pointer to %PBO pool
    mutable bool * pending_ = nullptr;              //!< Indicator if there is a pending read/write operation on the %PBO
    std::atomic<uint32_t> * refcount_ = nullptr;    //!< Reference count to the original %PBO
    int pboIndex_ = -1;                             //!< Per-Context %PBO index, maintained by PBOPool
};


} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
