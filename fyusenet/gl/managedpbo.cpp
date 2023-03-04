//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Pool-Managed PBO Wrapper
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "managedpbo.h"
#include "pbopool.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion {
namespace opengl {
//-------------------------------------- Local Definitions -----------------------------------------

// NOTE (mw) enable this definition for more zealous tracking in non-debug mode
//#define ZEALOUS


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Empty constructor
 */
ManagedPBO::ManagedPBO() {
    // special case of an empty PBO manager
}


/**
 * @brief Copy-constructor
 *
 * @param src Object to create a copy of
 *
 * @post Reference count of underlying PBO will be incremented
 *
 * Creates a ManagedPBO object as copy of the \p src object
 */
ManagedPBO::ManagedPBO(const ManagedPBO & src) {
    if (src.pool_) {
        assert(src.refcount_);
        pool_ = src.pool_;
        pbo_ = src.pbo_;
        pboIndex_ = src.pboIndex_;
        refcount_ = src.refcount_;
        pending_ = src.pending_;
        refcount_->fetch_add(1);
    }
}


/**
 * @brief Destructor
 *
 * Decreases the reference counter to the underlying PBO, in case this is the last (external)
 * reference, this function calls PBOPool::releasePBO to release the %PBO back into the pool.
 */
ManagedPBO::~ManagedPBO() {
    if (pool_) {
        assert(refcount_);
        assert(refcount_->load() > 0);
        int cnt = refcount_->fetch_sub(1);
        if ((pbo_) && (cnt == 1)) {
#ifdef ZEALOUS
            if (isPending()) THROW_FATAL_ARGS("PBO %p still pending (cnt=%d)", this, cnt);
#endif
            assert(!isPending());
            pool_->releasePBO(pbo_);
        }
    }
}


/**
 * @brief Assignment operator
 *
 * @param src Object to assign the data from to the current object
 *
 * @return Reference to current object after assignment
 *
 * @post Reference counter of previously wrapped PBO will be decremented, reference counter of
 *       assigned PBO will be incremented.
 *
 * This function copies al data from the supplied \p src to the current object before returning
 * a reference to itself.
 */
ManagedPBO & ManagedPBO::operator=(const ManagedPBO & src) {
    if (this == &src) return *this;
    auto oldref = refcount_;
    pool_ = src.pool_;
    pbo_ = src.pbo_;
    pboIndex_ = src.pboIndex_;
    refcount_ = src.refcount_;
    pending_ = src.pending_;
    if (refcount_) refcount_->fetch_add(1);
    if (oldref) oldref->fetch_sub(1);
    return *this;
}



/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param pbo Pointer to PBO that should be wrapped
 * @param pool Pointer to pool that controls the %PBO and creates this object
 * @param refcount Pointer to reference counter for the %PBO
 * @param pending Pointer to pending indicator for the %PBO
 * @param index Index of the %PBO in the pool
 */
ManagedPBO::ManagedPBO(PBO * pbo, PBOPool * pool, std::atomic<uint32_t> *refcount, bool * pending, int index) :
    pbo_(pbo), pool_(pool), pending_(pending), refcount_(refcount), pboIndex_(index) {
    assert(refcount_);
    refcount_->fetch_add(1);
}


} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
