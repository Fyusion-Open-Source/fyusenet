//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Basic Performance Measurement Functions (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//------------------------------------------------------------------------------

//-------------------------------------- System Headers --------------------------------------------

#include <time.h>

//-------------------------------------- Project  Headers ------------------------------------------


//------------------------------------- Public Definitions -----------------------------------------

typedef unsigned long long int tstamp;

//--------------------------------------- Public Functions -----------------------------------------

#ifdef __cplusplus
extern "C" {
#endif
    tstamp get_stamp();
    unsigned int elapsed_micros(tstamp start,tstamp end);
    unsigned int elapsed_nanos(tstamp start,tstamp end);
    unsigned int elapsed_millis(tstamp start,tstamp end);
    unsigned int elapsed_seconds(tstamp start,tstamp end);
#ifdef __cplusplus
}
#endif

// vim: set expandtab ts=4 sw=4:
