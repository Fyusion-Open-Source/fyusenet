//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Simple PBO Pool (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <mutex>
#include <atomic>
#include <list>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "../gpu/gfxcontexttracker.h"
#include "../gpu/gfxcontextlink.h"
#include "managedpbo.h"
#include "pbo.h"

//------------------------------------------ Constants ---------------------------------------------


namespace fyusion::opengl {

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Simple PBO pool
 *
 * This class serves as a simple (and thread-safe) PBO pool. It stores a dynamic list of PBOs
 * with a maximum capacity and provides managed PBO instances for multi-threaded scenarios.
 * All instances are tracked by the pool, which retains the ownership, and are made available
 * without prioritization.
 *
 * @see ManagedPBO, PBO
 */
class PBOPool : public fyusenet::GfxContextTracker {
    friend class ManagedPBO;

    /**
     * @brief Single PBO pool entry structure
     *
     * Compound structure that aggregates the actual PBO with meta-information like usage
     * state and reference counting.
     */
    struct entry {
        entry(PBO *p, bool b) : pbo(p), busy(b) {}
        entry(entry && src) noexcept {
            pbo = src.pbo;
            busy = src.busy;
            pending = src.pending;
            // NOTE (mw) not atomic
            refcount.store(src.refcount.load());
        }
        entry(const entry & src) = delete;
        entry & operator=(const entry & src) = delete;

        PBO * pbo = nullptr;                    //!< Pointer to underlying PBO
        bool busy = false;                      //!< Indicator if the #pbo is currently busy (i.e. a reference outside of the pool itself is held)
        bool pending = false;                   //!< Indicator if the #pbo is currently in a pending state (an operation was triggered and the result is still pending)
        std::atomic<uint32_t> refcount{0};      //!< Number of references held to the #pbo, includes a reference by the pool itself
    };
 public:
    // ------------------------------------------------------------------------
    // Constructor/Destructor
    // ------------------------------------------------------------------------
    explicit PBOPool(int maxPBOs, const fyusenet::GfxContextLink& ctx = fyusenet::GfxContextLink());
    ~PBOPool() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    ManagedPBO getAvailablePBO(int width, int height, int channels, int bytesPerChannel);
    void logStatistics();

    /**
     * @brief Set the maximum allowed number of PBOs for the pool
     *
     * @param mx Maximum number of PBOs maintained by the pool
     */
    void setMaxPBOs(int mx) {
        assert(mx >= 0);
        maxPBOs_ = mx;
    }
 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void releasePBO(PBO *pbo);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    int maxPBOs_ = 1;                       //!< Maximum number of PBOs in the pool
    int currentPBOs_ = 0;                   //!< Current number of PBOs in the pool
    std::mutex lock_;                       //!< Serialization to pool resources
    std::list<entry> availablePBOs_;        //!< List of pool resources
    uint64_t requests_ = 0;                 //!< For performance measurement, number of times a %PBO was requested from the pool
    uint64_t immediateHits_ = 0;            //!< For performance measurement, number of times a %PBO was available immediately
    uint64_t waitCycles_ = 0;               //!< For performance measurement, the number of busy-waiting rounds until a %PBO became available
};

} // fyusion::opengl namespace


// vim: set expandtab ts=4 sw=4:
