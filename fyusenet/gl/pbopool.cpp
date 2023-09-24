//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Simple PBO Pool
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <thread>
#include <cinttypes>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../common/performance.h"
#include "pbopool.h"

//-------------------------------------- Global Variables ------------------------------------------

namespace fyusion {
namespace opengl {
//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param maxPBOs Maximum number of PBOs maintained by the pool
 * @param ctx Optional link to GL context to use for this pool (and its objects)
 */
PBOPool::PBOPool(int maxPBOs, const fyusenet::GfxContextLink& ctx) : GfxContextTracker(), maxPBOs_(maxPBOs) {
    setContext(ctx);
}


/**
 * @brief Destructor
 *
 * @pre The GL context stored with the pool is current to the calling thread and no PBOs of this
 *      pool are in circulation anymore
 *
 * Deletes all PBOs maintained by this pool.
 */
PBOPool::~PBOPool() {
    assertContext();
    for (entry & ent : availablePBOs_) {
        assert(!ent.busy);
        assert(!ent.pending);
        delete ent.pbo;
    }
    availablePBOs_.clear();
}


/**
 * @brief Retrieve a ManagedPBO from the pool
 *
 * @param width Width (pixels) of the %PBO to get
 * @param height Height (pixels) of the %PBO to get
 * @param channels Number of channels for the %PBO to get
 * @param bytesPerChannel Number of bytes per channel
 *
 * @return A ManagedPBO instance that wraps a PBO in a management structure for this pool
 *
 * Retrieve a %PBO for use with either reading or writing. The returned objects are not low-level
 * PBO instances, but ManagedPBO instances that offer full access to the underlying PBO but add
 * transparent management structures to the %PBO to make it easier for this pool to track its
 * resources.
 *
 * @note The number of \p channels may exceed the maximum number of channels per pixel (4), because
 *       the %PBO here is just treated as a buffer.
 */
 // TODO (mw) switch from geometry (width, height) to pure size instead
ManagedPBO PBOPool::getAvailablePBO(int width, int height, int channels, int bytesPerChannel) {
    using namespace std::chrono_literals;
    bool immediate = true;
    while (true) {
        lock_.lock();
        if (immediate) requests_++;
        int idx=0;
        for (int pass=0; pass < 2; pass++) {
            for (auto ii = availablePBOs_.begin() ; ii != availablePBOs_.end(); ++ii,idx++) {
                if (!(*ii).busy) {
                    PBO * pbo = (*ii).pbo;
                    if ( pbo->matches(width, height, channels, bytesPerChannel) || (pass > 0)) {
                        (*ii).busy = true;
                        pbo->resize(width, height, channels, bytesPerChannel);
                        if (immediate) immediateHits_++;
                        lock_.unlock();
                        return {pbo, this, &(*ii).refcount, &(*ii).pending, idx};
                    }
                }
            }
        }
        if (currentPBOs_ < maxPBOs_) {
            PBO * pbo = new PBO(width, height, channels, bytesPerChannel, context());
            idx = (int)availablePBOs_.size();
            availablePBOs_.emplace_back(pbo, true);
            entry & last = availablePBOs_.back();
            currentPBOs_++;
            if (immediate) immediateHits_++;
            lock_.unlock();
            return {pbo, this, &(last.refcount), &(last.pending), idx};
        }
        immediate = false;
        waitCycles_++;
        lock_.unlock();
        std::this_thread::sleep_for(2ms);
    }
}


/**
 * @brief Log basic pool statistics (for debugging)
 */
void PBOPool::logStatistics() {
#ifdef DEBUG
    std::lock_guard<std::mutex> lck(lock_);
    FNLOGD("PBO pool %p access statistics:", this);
    FNLOGD("  # requests: %" PRIu64, requests_);
    FNLOGD("  # immhits: %" PRIu64, immediateHits_);
    FNLOGD("  # wait cycles: %" PRIu64, waitCycles_);
    FNLOGD("  wait time: %" PRIu64 " ms", waitCycles_*2);
#endif
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Release a %PBO back into the pool
 *
 * @param pbo Pointer to PBO which shall be released
 *
 * @pre The supplied \p pbo must not be marked as pending
 * @post Corresponding pool entry will have the %PBO marked as not-busy.
 *
 * This function releases a %PBO back to the pool by locating its entry in the list of PBOs
 * and marking the %PBO as not-busy (available).
 */
void PBOPool::releasePBO(PBO * pbo) {
    std::lock_guard<std::mutex> lck(lock_);
    int idx = 0;
    for (auto ii=availablePBOs_.begin(); ii != availablePBOs_.end(); ++ii,idx++) {
        if ((*ii).pbo == pbo) {
            (*ii).busy = false;
            return;
        }
    }
    // this should not happen
    assert(false);
}


} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
