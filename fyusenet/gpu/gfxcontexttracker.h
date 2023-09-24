//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Interface for GL Context-Link Trackers (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#include "../gpu/gfxcontextlink.h"
#include "../gl/glexception.h"

//------------------------------------------ Constants ---------------------------------------------


namespace fyusion {
namespace fyusenet {
//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Base class for tracking a graphics/GL context
 *
 * Provides standardized interface for attaching a class to a GfxContextLink.
 */
class GfxContextTracker {
 public:

    GfxContextTracker() = default;
    virtual ~GfxContextTracker() = default;

    /**
     * @brief Constructor with existing context link
     *
     * @param ctx Reference to context link to set into this object
     */
    explicit GfxContextTracker(const GfxContextLink& ctx) {
        context_ = ctx;
    }

    /**
     * @brief Set context to track
     *
     * @param context GfxContextLink instance that should be tracked by this instance
     */
    virtual void setContext(const GfxContextLink& context) {
        context_ = context;
    }

    /**
     * @brief Get context that is tracked by this instance
     *
     * @return GfxContextLink that is tracked by this instance
     */
    [[nodiscard]] const GfxContextLink & context() const {
        return context_;
    }
 protected:
    /**
     * @brief Assert that the context that is current is the same as the tracked context
     *
     * @throws opengl::GLException if assertion fails
     */
    void assertContext() const {
        if ((context_ == GfxContextLink::EMPTY) || (!context_.isCurrent())) THROW_EXCEPTION_ARGS(opengl::GLException, "Invalid or mismatching GL context");
    }

    GfxContextLink context_ = GfxContextLink::EMPTY;        //!< Context that is tracked
};

} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
