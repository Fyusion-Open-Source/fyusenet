//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Unit-Testing Helpers (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <gtest/gtest.h>

//-------------------------------------- Project  Headers ------------------------------------------

#include <fyusenet/gl/gl_sys.h>
#include <fyusenet/gl/glcontext.h>
#include <fyusenet/gl/glinfo.h>
#include <fyusenet/gpu/gfxcontextmanager.h>
#include <fyusenet/gl/shadercache.h>
#ifdef FYUSENET_MULTITHREADING
#include <fyusenet/gl/asyncpool.h>
#endif

//------------------------------------------ Constants ---------------------------------------------


//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Customized google-test environment to be used with OpenGL
 */
class GLEnvironment : public ::testing::Environment {
 public:

    virtual void SetUp() override {
    }

    static void init() {
        // NOTE (mw) not thread-safe
        if (!initialized_) {
            ::testing::AddGlobalTestEnvironment(new GLEnvironment());
            initialized_ = true;
        }
    }

 private:
    static bool initialized_;
};


/**
 * @brief OpenGL context manager for GLEnvironment
 */
class TestContextManager {
 public:

    fyusion::fyusenet::GfxContextLink & context() {
        return context_;
    }

    static void setupGLContext(int derived=1);
    static void tearDownGLContext();
    static void waitMouse();

    static fyusion::fyusenet::GfxContextLink context_;
};


