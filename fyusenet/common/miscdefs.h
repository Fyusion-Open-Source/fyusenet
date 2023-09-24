//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Miscellaneous Small Helper Definitions (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once


//------------------------------------- Public Definitions -----------------------------------------

#define FNET_DEL_AND_CLEAR(_x) delete _x ; _x = nullptr;
#define FNET_DEL_AND_CLEAR_ARRAY(_x) delete [] _x ; _x = nullptr;

#ifdef DEBUG
#define CLEAR_GFXERR_DEBUG glGetError();
#else
#define CLEAR_GFXERR_DEBUG
#endif


// vim: set expandtab ts=4 sw=4:
