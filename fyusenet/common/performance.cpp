//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Basic Performance Measurement Functions
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//-------------------------------------- System Headers --------------------------------------------

#if defined(WIN32) || defined(WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

//-------------------------------------- Project  Headers ------------------------------------------

#include "performance.h"


//-------------------------------------- Global Variables ------------------------------------------


//------------------------------------- Private Prototypes -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


tstamp fy_get_stamp() {
#if defined(WIN32) || defined(WIN64)
    LARGE_INTEGER stamp;
    QueryPerformanceCounter(&stamp);
    return (tstamp)stamp.QuadPart;
#else
    struct timespec spec;
    clock_gettime(CLOCK_MONOTONIC, &spec);
    return ((unsigned long long int)spec.tv_sec)*1000000000ULL+(unsigned long long int)spec.tv_nsec;
#endif
}

unsigned int fy_elapsed_nanos(tstamp start,tstamp end) {
#if defined(WIN32) || defined(WIN64)
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    // NOTE (mw) this may be imprecise, depending on the counter resolution and the time interval
    return (unsigned int)(1000000000.0 * (double)(end-start) / (double)(freq.QuadPart));
#else
    return (unsigned int)(end-start);
#endif
}

unsigned int fy_elapsed_micros(tstamp start,tstamp end) {
#if defined(WIN32) || defined(WIN64)
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    // NOTE (mw) this may be imprecise, depending on the counter resolution and the time interval
    return (unsigned int)(1000000.0 * (double)(end-start) / (double)(freq.QuadPart));
#else
    return (unsigned int)((end/1000) - (start/1000));
#endif
}

unsigned int fy_elapsed_millis(tstamp start,tstamp end) {
#if defined(WIN32) || defined(WIN64)
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    // NOTE (mw) this may be imprecise, depending on the counter resolution and the time interval
    return (unsigned int)(1000.0 * (double)(end-start) / (double)(freq.QuadPart));
#else
    return (unsigned int)(((end/1000) - (start/1000))/1000);
#endif
}

unsigned int fy_elapsed_seconds(tstamp start,tstamp end) {
#if defined(WIN32) || defined(WIN64)
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    // NOTE (mw) this may be imprecise, depending on the counter resolution and the time interval
    return (unsigned int)((double)(end-start) / (double)(freq.QuadPart));
#else
    return fy_elapsed_millis(start,end)/1000;
#endif
}


// vim: set expandtab ts=4 sw=4:
