//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Very rudimentary logging (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//------------------------------------------------------------------------------

//-------------------------------------- System Headers --------------------------------------------

#if defined(ANDROID)&&!defined(STANDALONE)
#include <android/log.h>
#else
#include <cstdio>
#endif

//-------------------------------------- Project  Headers ------------------------------------------


//------------------------------------- Public Definitions -----------------------------------------


//--------------------------------------- Public Functions -----------------------------------------

#if defined(ANDROID)

#if !defined(FNLOGD) && !defined(FNLOGE)
#define LOG_TAG "fyn"
#ifdef DEBUG
#define FNLOGD(fmt,...) __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,fmt,##__VA_ARGS__);
#else
#define FNLOGD(fmt,...)
#endif
#define FNLOGE(fmt,...) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,fmt,##__VA_ARGS__);
#define FNLOGW(fmt,...) __android_log_print(ANDROID_LOG_WARN,LOG_TAG,fmt,##__VA_ARGS__);
#define FNLOGI(fmt,...) __android_log_print(ANDROID_LOG_INFO,LOG_TAG,fmt,##__VA_ARGS__);
#define GLLOGE() { GLenum __err=glGetError(); if (__err != GL_NO_ERROR) __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,"%s:%d glerr=0x%X",__FILE__,__LINE__,__err); }
#endif

#else

#ifdef DEBUG
#define FNLOGD(fmt,...) printf(fmt"\n",##__VA_ARGS__);
#else
#define FNLOGD(fmt,...)
#endif
#define FNLOGI(fmt,...) printf(fmt"\n",##__VA_ARGS__);
#define FNLOGE(fmt,...) printf(fmt"\n",##__VA_ARGS__);
#define FNLOGW(fmt,...) printf(fmt"\n",##__VA_ARGS__);
#define GLLOGE() { GLenum __err=glGetError(); if (__err != GL_NO_ERROR) printf("%s:%d glerr=0x%X",__FILE__,__LINE__,__err); }

#endif


// vim: set expandtab ts=4 sw=4:
