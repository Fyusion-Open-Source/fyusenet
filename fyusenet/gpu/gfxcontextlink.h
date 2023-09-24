//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Context Link (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

#include "../gl/gl_sys.h"
#include "../gl/glcontextinterface.h"
#include "../common/logging.h"

namespace fyusion {

namespace opengl {
    class AsyncPool;
    class ScopedTexturePool;
}

namespace fyusenet {
//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Lightweight non-system-specific reference to a GL context
 *
 * This class may be used to pass around a GL context to various classes without having to
 * care too much about system specifics.
 *
 * It also maintains a reference counter in the actual GL context, so that GL contexts
 * are not eliminated while there are still pending links (or at least you will get a warning
 * about that).
 *
 * In addition to establishing a link to a context for the underlying gfx system (which currently
 * is OpenGL), this class also provides some convenience methods for creating shaders and managing
 * the graphics state.
 *
 * @note The header file of this link resides in the GPU subfolder instead of the backend-
 *       specific GL folder, whereas the implementation is placed in the GL subfolder. In
 *       addition, the class resides within the fyusenet namespace. This is done for extension
 *       reasons (change of backend) and not by accident. Not very clean though.
 */
class GfxContextLink {
    friend class GfxContextManager;
    friend class opengl::AsyncPool;
 public:
    using syncid = GLsync;
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit GfxContextLink(opengl::GLContextInterface * wrap = nullptr);
    GfxContextLink(const GfxContextLink& src);
    ~GfxContextLink();

    // ------------------------------------------------------------------------
    // Overloaded operators
    // ------------------------------------------------------------------------
    GfxContextLink& operator=(const GfxContextLink& src);

    /**
     * @brief Check if two links point to the same context
     *
     * @param other Link to check for equivalence
     *
     * @retval true if the supplied \p other link points to the same context
     * @retval false otherwise
     */
    bool operator==(const GfxContextLink& other) const {
        return (context_ == other.context_);
    }

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] bool isCurrent() const;
    [[nodiscard]] int device() const;
    void reset();
    syncid issueSync() const;
    void waitSync(syncid sync) const;
    bool waitClientSync(syncid sync, GLuint64 timeout) const;    
    void removeSync(syncid sync) const;
    [[nodiscard]] opengl::ScopedTexturePool * texturePool() const;

    /**
     * @brief Check if this link points to a valid context
     *
     * @retval true If link/context is valid
     * @retval false Otherwise
     */
    [[nodiscard]] bool isValid() const {
        return (context_ != nullptr);
    }

    /**
     * @brief Retrieve read-only pointer to the GL context interface this object links to
     *
     * @return Pointer to high-level interface of GL context (abstracted from system)
     */
    [[nodiscard]] const opengl::GLContextInterface * interface() const {
        return context_;
    }

    /**
     * @brief Retrieve pointer to the GL context interface this object links to
     *
     * @return Pointer to high-level interface of GL context (abstracted from system)
     */
    [[nodiscard]] opengl::GLContextInterface * interface() {
        return context_;
    }


    static const GfxContextLink EMPTY;           //!< Symbolic placeholder for an empty (invalid) context link
 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    explicit GfxContextLink(bool empty);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    opengl::GLContextInterface * context_ = nullptr; //!< Pointer to actual GL context
    uint64_t id_;                                    //!< Context link ID for debugging
};

} // fyusenet namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
