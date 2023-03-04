//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Basic Performance Measurement Functions
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//-------------------------------------- System Headers --------------------------------------------

#include <time.h>


//-------------------------------------- Project  Headers ------------------------------------------

#include "performance.h"


//-------------------------------------- Global Variables ------------------------------------------


//------------------------------------- Private Prototypes -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


tstamp fy_get_stamp() {
    struct timespec spec;
    clock_gettime(CLOCK_MONOTONIC, &spec);
    return ((unsigned long long int)spec.tv_sec)*1000000000ULL+(unsigned long long int)spec.tv_nsec;
}

unsigned int fy_elapsed_nanos(tstamp start,tstamp end) {
    return (unsigned int)(end-start);
}

unsigned int fy_elapsed_micros(tstamp start,tstamp end) {
    return (unsigned int)((end/1000) - (start/1000));
}

unsigned int fy_elapsed_millis(tstamp start,tstamp end) {
    return (unsigned int)(((end/1000) - (start/1000))/1000);
}

unsigned int fy_elapsed_seconds(tstamp start,tstamp end) {
    return fy_elapsed_millis(start,end)/1000;
}



// vim: set expandtab ts=4 sw=4:
