//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// EGL Helper Routines
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#ifndef FYUSENET_USE_EGL
#error THIS FILE SHOULD NOT BE COMPILED
#else

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../glexception.h"
#include "eglhelper.h"
#include "../glcontext.h"
#include "../../common/logging.h"

//-------------------------------------- Global Variables ------------------------------------------

//-------------------------------------- Local Definitions -----------------------------------------

#ifdef ANDROID
#define ES3BIT EGL_OPENGL_ES3_BIT_KHR
#else
#define ES3BIT EGL_OPENGL_ES3_BIT
#endif

static PFNEGLQUERYDEVICEATTRIBEXTPROC eglQueryDeviceAttribEXT = nullptr;
static PFNEGLQUERYDEVICESTRINGEXTPROC eglQueryDeviceStringEXT = nullptr;
static PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = nullptr;
static PFNEGLQUERYDISPLAYATTRIBEXTPROC eglQueryDisplayAttribEXT = nullptr;
static PFNEGLCREATESTREAMKHRPROC eglCreateStreamKHR = nullptr;
static PFNEGLDESTROYSTREAMKHRPROC eglDestroyStreamKHR = nullptr;
static PFNEGLSTREAMATTRIBKHRPROC eglStreamAttribKHR = nullptr;
static PFNEGLQUERYSTREAMKHRPROC eglQueryStreamKHR = nullptr;
static PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC eglCreateStreamProducerSurfaceKHR = nullptr;
static PFNEGLQUERYSTREAMTIMEKHRPROC eglQueryStreamTimeKHR = nullptr;
static PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = nullptr;

namespace fyusion {
namespace opengl {

static const EGLint displayConfig16Bit[] = {
    EGL_RENDERABLE_TYPE, ES3BIT,
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_RED_SIZE,   5,
    EGL_GREEN_SIZE, 6,
    EGL_BLUE_SIZE,  5,
    EGL_NONE, EGL_NONE,
    EGL_DEPTH_SIZE, 0,
    EGL_STENCIL_SIZE, 0,
    EGL_NONE
};


static const EGLint displayConfig24Bit[] = {
    EGL_RENDERABLE_TYPE, ES3BIT,
    EGL_SURFACE_TYPE, EGL_STREAM_BIT_KHR,
    EGL_RED_SIZE,   8,
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE,  8,
    EGL_NONE, EGL_NONE,
    EGL_DEPTH_SIZE, 0,
    EGL_STENCIL_SIZE, 0,
    EGL_NONE
};

static const EGLint displayConfig32Bit[]= {
    EGL_RENDERABLE_TYPE, ES3BIT,
    EGL_SURFACE_TYPE, EGL_STREAM_BIT_KHR,
    EGL_RED_SIZE,   8,
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE,  8,
    EGL_ALPHA_SIZE, 8,
    EGL_NONE, EGL_NONE
};

static const int maxDevices = 32;

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/




/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

void EGLHelper::initEGLExtensions() {
    if (eglQueryDeviceAttribEXT == nullptr) {
        eglQueryDeviceAttribEXT = (PFNEGLQUERYDEVICEATTRIBEXTPROC)eglGetProcAddress("eglQueryDeviceAttribEXT");
        eglQueryDeviceStringEXT = (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress("eglQueryDeviceStringEXT");
        eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
        eglQueryDisplayAttribEXT = (PFNEGLQUERYDISPLAYATTRIBEXTPROC)eglGetProcAddress("eglQueryDisplayAttribEXT");
        eglCreateStreamKHR = (PFNEGLCREATESTREAMKHRPROC)eglGetProcAddress("eglCreateStreamKHR");
        eglDestroyStreamKHR = (PFNEGLDESTROYSTREAMKHRPROC)eglGetProcAddress("eglDestroyStreamKHR");
        eglStreamAttribKHR = (PFNEGLSTREAMATTRIBKHRPROC)eglGetProcAddress("eglStreamAttribKHR");
        eglQueryStreamKHR = (PFNEGLQUERYSTREAMKHRPROC)eglGetProcAddress("eglQueryStreamKHR");
        eglCreateStreamProducerSurfaceKHR = (PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC)eglGetProcAddress("eglCreateStreamProducerSurfaceKHR");
        eglQueryStreamTimeKHR = (PFNEGLQUERYSTREAMTIMEKHRPROC)eglGetProcAddress("eglQueryStreamTimeKHR");
        eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

        assert(eglQueryDeviceAttribEXT);
        assert(eglQueryDeviceStringEXT);
        assert(eglQueryDevicesEXT);
        assert(eglQueryDisplayAttribEXT);
        assert(eglCreateStreamKHR);
        assert(eglDestroyStreamKHR);
        assert(eglStreamAttribKHR);
        assert(eglQueryStreamKHR);
        assert(eglCreateStreamProducerSurfaceKHR);
        assert(eglQueryStreamTimeKHR);
        assert(eglGetPlatformDisplayEXT);
    }
}

void EGLHelper::iterateEGLDisplays(std::function<void(EGLDisplay eglDisplay, bool* stop)> function) {
    initEGLExtensions();

    EGLDeviceEXT eglDevices[maxDevices];
    EGLint nDevices;
    eglQueryDevicesEXT(maxDevices, eglDevices, &nDevices);

    bool stop = false;
    EGLDisplay eglDisplay;
    for (EGLint i = 0; (i < nDevices) && (!stop); ++i) {
        eglDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevices[i], 0);
        const char *vendor = eglQueryDeviceStringEXT(eglDevices[i], EGL_VENDOR);
        if (eglDisplay == EGL_NO_DISPLAY) {
            EGLint eglerr = eglGetError();
            FNLOGW("eglGetPlatformDisplayEXT failed: %d", eglerr);
        } else {
            if (vendor) {
                function(eglDisplay, &stop);
            }
        }
    }
}


} // opengl namespace
} // fyusion namespace

#endif

// vim: set expandtab ts=4 sw=4:
