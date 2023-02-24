//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Interface Definitions for in-place Layer Modifiers (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

//-------------------------------------- Project  Headers ------------------------------------------

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
namespace fyusenet {

/**
 * @brief Modifier interface for supplying a rotation angle to a network layer
 */
class RotationModifier {
 public:
    virtual void setRotation(int degrees) = 0;
};

} // fyusenet namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
