//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Main Convenience Header
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------


//-------------------------------------- Project  Headers ------------------------------------------

#ifndef FYUSENET_INTERNAL
#ifndef FYUSENET_MULTITHREADING
#cmakedefine FYUSENET_MULTITHREADING
#endif
#endif

#include "gpu/gfxcontextmanager.h"
#include "gpu/gfxcontextlink.h"
#include "gpu/gfxcontexttracker.h"
#include "base/compiledlayers.h"
#include "base/neuralnetwork.h"
#include "base/engine.h"
#include "base/layerflags.h"
#include "base/layerbuilder.h"
#include "base/layerbase.h"
#include "base/convlayerinterface.h"
#include "base/batchnorminterface.h"
#include "base/asynclayerinterface.h"
#include "gpu/gpulayerbase.h"
#include "gpu/uploadlayer.h"
#include "gpu/downloadlayer.h"
#include "gpu/convlayerbase.h"
#include "gpu/deep/deepconvlayerbase.h"
#include "gpu/deep/deepdownloadlayer.h"
#include "gpu/deep/deepgemmlayer.h"
#include "cpu/cpubuffershape.h"
#include "cpu/cpubuffer.h"
#include "cpu/cpulayerbase.h"
#include "cpu/cpulayerinterface.h"
#include "cpu/convlayer.h"

#include "gpu/argmaxlayerbuilder.h"
#include "gpu/blurlayerbuilder.h"
#include "gpu/castlayerbuilder.h"
#include "gpu/concatlayerbuilder.h"
#include "gpu/convlayerbuilder.h"
#include "gpu/customlayerbuilder.h"
#include "gpu/imgextractlayerbuilder.h"
#include "gpu/scalelayerbuilder.h"
#include "gpu/singleton_arithlayerbuilder.h"
#include "gpu/poollayerbuilder.h"
#include "gpu/updownlayerbuilder.h"
#include "gpu/transposelayerbuilder.h"


// vim: set expandtab ts=4 sw=4:

