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
#ifndef FYUSENET_GL_BACKEND
#cmakedefine FYUSENET_GL_BACKEND
#endif
#endif

#include "gpu/gfxcontextmanager.h"
#include "gpu/gfxcontextlink.h"
#include "gpu/gfxcontexttracker.h"
#include "base/buffershape.h"
#include "cpu/cpubuffer.h"
#include "gpu/gpubuffer.h"
#include "base/compiledlayers.h"
#include "base/neuralnetwork.h"
#include "base/engine.h"
#include "base/layerflags.h"
#include "base/layerbuilder.h"
#include "base/layerbase.h"
#include "base/asynclayerinterface.h"
#include "base/statetoken.h"
#include "base/parameterprovider.h"
#include "base/layerfactory.h"
#include "common/miscdefs.h"

#include "cpu/cpulayerbase.h"
#include "cpu/cpulayerinterface.h"
#include "cpu/convlayer.h"

#include "gpu/gpulayerbase.h"
#include "gpu/uploadlayer.h"
#include "gpu/downloadlayer.h"
#include "gpu/deep/deepdownloadlayer.h"

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
#include "gpu/embeddinglayerbuilder.h"
#include "gpu/tokenscoringlayerbuilder.h"
#include "gpu/attentionlayerbuilder.h"
#include "gpu/linearlayerbuilder.h"

// vim: set expandtab ts=4 sw=4:

