//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Unit-Testing Helpers
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <chrono>
#include <thread>
#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gltesthelpers.h"
#include <fyusenet/fyusenet.h>
#include <fyusenet/gl/shadercache.h>
#ifdef FYUSENET_MULTITHREADING
#include <fyusenet/gl/asyncpool.h>
#endif

//-------------------------------------- Global Variables ------------------------------------------

bool GLEnvironment::initialized_ = false;
fyusion::fyusenet::GfxContextLink TestContextManager::context_ = fyusion::fyusenet::GfxContextLink::EMPTY;

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Create OpenGL environment for testing
 *
 * @param derived Number of contexts to derive
 *
 * This function sets up a main GL context and also creates a number of derived (shared) contexts
 * that can be used with the AsyncPool on multi-threaded test-cases.
 */
void TestContextManager::setupGLContext(int derived) {
    auto mgr = fyusion::fyusenet::GfxContextManager::instance();
    ASSERT_NE(mgr, nullptr);
    context_ = mgr->createMainContext();
    ASSERT_TRUE(context_.isCurrent());
    // pre-create derived contexts, we need to do this here as the NVIDIA debugger crashes otherwise
    // we also limit the number of GL threads here for testing purposes
#ifdef FYUSENET_MULTITHREADING
    fyusion::opengl::AsyncPool::setMaxGLThreads(derived);
    fyusion::opengl::AsyncPool::createDerivedBatch(context_, derived);
#endif
#ifdef FYUSENET_USE_GLFW
    static bool buttonup = false;
    const fyusion::opengl::GLContext * ctx = dynamic_cast<const fyusion::opengl::GLContext *>(context_.interface());
    auto mousecb = [](GLFWwindow *win, int bt, int action, int mods) {
        if (action == GLFW_PRESS) buttonup = true;
    };
    glfwSetMouseButtonCallback(ctx->context_, mousecb);
    while (!buttonup) {
        glfwWaitEventsTimeout(0.1);
    }
    for (int i=0; i<6; i++) {
        ctx->sync();
    }
#endif
}


/**
 * @brief Deallocate GL resources and close GL context
 */
void TestContextManager::tearDownGLContext() {
#ifdef FYUSENET_USE_GLFW
    static bool buttondown = false;
    const fyusion::opengl::GLContext * ctx = dynamic_cast<const fyusion::opengl::GLContext *>(context_.interface());
    ctx->sync();
    auto mousecb = [](GLFWwindow *win, int bt, int action, int mods) {
        if (action == GLFW_PRESS) {
            buttondown = true;
        }
    };
    glfwSetMouseButtonCallback(ctx->context_, mousecb);
    while (!buttondown) {
        glfwWaitEventsTimeout(0.1);
    }
#endif
    context_.reset();
    fyusion::fyusenet::GfxContextManager::tearDown();
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


