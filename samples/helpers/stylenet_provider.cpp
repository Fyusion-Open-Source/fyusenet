//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Parameter Providers for Style-Transfer Network Samples                      (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include "stylenet_provider.h"

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Create provider object around existing memory block
 *
 * @param memory Pointer to memory to retrieve data from, no ownership is taken
 * @param bytes Number of bytes presented by \p memory
 */
StyleNet3x3Provider::StyleNet3x3Provider(const uint8_t *memory, size_t bytes) : StyleNet3x3Provider() {
    if (bytes < STYLENET_SIZE * sizeof(float)) throw std::runtime_error("Insufficient weight data supplied");
    assert(wbData_);
    memcpy(wbData_, memory, STYLENET_SIZE * sizeof(float));
}


StyleNet3x3Provider::StyleNet3x3Provider(const std::string& fileName)  : StyleNet3x3Provider() {
    assert(wbData_);
    loadFile(fileName, STYLENET_SIZE);
}

/**
 * @brief Create provider object around existing memory block
 *
 * @param memory Pointer to memory to retrieve data from, no ownership is taken
 * @param bytes Number of bytes presented by \p memory
 */
StyleNet9x9Provider::StyleNet9x9Provider(const uint8_t *memory, size_t bytes) : StyleNet9x9Provider() {
    if (bytes < STYLENET_SIZE * sizeof(float)) throw std::runtime_error("Insufficient weight data supplied");
    assert(wbData_);
    memcpy(wbData_, memory, STYLENET_SIZE * sizeof(float));
}


StyleNet9x9Provider::StyleNet9x9Provider(const std::string& fileName)  : StyleNet9x9Provider() {
    assert(wbData_);
    loadFile(fileName, STYLENET_SIZE);
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

void StyleNetProvider::loadFile(const std::string& fileName, int numFloats) {
    assert(wbData_);
    FILE * f = fopen(fileName.c_str(), "rb");
    if (f) {
        size_t read = fread(wbData_, sizeof(float), numFloats, f);
        fclose(f);
        if (read != (size_t)numFloats) {
            throw std::runtime_error("Insufficient weight data supplied in file " + fileName);
        }
    } else {
        throw std::runtime_error("Cannot open file " + fileName);
    };

}

StyleNet3x3Provider::StyleNet3x3Provider() : StyleNetProvider() {
    wbData_ = new float[STYLENET_SIZE];
    memset(wbData_, 0, STYLENET_SIZE * sizeof(float));
    weightBlocks_.emplace(layer_ids::CONV1 , wrapper(wbData_ + 0 + 12));
    weightBlocks_.emplace(layer_ids::CONV2 , wrapper(wbData_ + 336 + 20));
    weightBlocks_.emplace(layer_ids::CONV3 , wrapper(wbData_ + 2516 + 40));
    weightBlocks_.emplace(layer_ids::RES1_1 , wrapper(wbData_ + 19475 + 40));
    weightBlocks_.emplace(layer_ids::RES1_2 , wrapper(wbData_ + 33915 + 40));
    weightBlocks_.emplace(layer_ids::RES2_1 , wrapper(wbData_ + 48355 + 40));
    weightBlocks_.emplace(layer_ids::RES2_2 , wrapper(wbData_ + 62795 + 40));
    weightBlocks_.emplace(layer_ids::DECONV1 , wrapper(wbData_ + 9756 + 20));
    weightBlocks_.emplace(layer_ids::DECONV2 , wrapper(wbData_ + 16976 + 12));
    weightBlocks_.emplace(layer_ids::DECONV3 , wrapper(wbData_ + 19148 + 3));
    biasBlocks_.emplace(layer_ids::CONV1 , wrapper(wbData_ + 0));
    biasBlocks_.emplace(layer_ids::CONV2 , wrapper(wbData_ + 336));
    biasBlocks_.emplace(layer_ids::CONV3 , wrapper(wbData_ + 2516));
    biasBlocks_.emplace(layer_ids::RES1_1 , wrapper(wbData_ + 19475));
    biasBlocks_.emplace(layer_ids::RES1_2 , wrapper(wbData_ + 33915));
    biasBlocks_.emplace(layer_ids::RES2_1 , wrapper(wbData_ + 48355));
    biasBlocks_.emplace(layer_ids::RES2_2 , wrapper(wbData_ + 62795));
    biasBlocks_.emplace(layer_ids::DECONV1 , wrapper(wbData_ + 9756));
    biasBlocks_.emplace(layer_ids::DECONV2 , wrapper(wbData_ + 16976));
    biasBlocks_.emplace(layer_ids::DECONV3 , wrapper(wbData_ + 19148));
}


StyleNet9x9Provider::StyleNet9x9Provider() {
    wbData_ = new float[STYLENET_SIZE];
    memset(wbData_, 0, STYLENET_SIZE * sizeof(float));
    weightBlocks_.emplace(layer_ids::CONV1, wrapper(wbData_ + 0 + 12));
    weightBlocks_.emplace(layer_ids::CONV2, wrapper(wbData_ + 2928 + 20));
    weightBlocks_.emplace(layer_ids::CONV3, wrapper(wbData_ + 5108 + 40));
    weightBlocks_.emplace(layer_ids::RES1_1, wrapper(wbData_ + 24659 + 40));
    weightBlocks_.emplace(layer_ids::RES1_2, wrapper(wbData_ + 39099 + 40));
    weightBlocks_.emplace(layer_ids::RES2_1, wrapper(wbData_ + 53539 + 40));
    weightBlocks_.emplace(layer_ids::RES2_2, wrapper(wbData_ + 67979 + 40));
    weightBlocks_.emplace(layer_ids::RES3_1, wrapper(wbData_ + 82419 + 40));
    weightBlocks_.emplace(layer_ids::RES3_2, wrapper(wbData_ + 96859 + 40));
    weightBlocks_.emplace(layer_ids::RES4_1, wrapper(wbData_ + 111299 + 40));
    weightBlocks_.emplace(layer_ids::RES4_2, wrapper(wbData_ + 125739 + 40));
    weightBlocks_.emplace(layer_ids::RES5_1, wrapper(wbData_ + 140179 + 40));
    weightBlocks_.emplace(layer_ids::RES5_2, wrapper(wbData_ + 154619 + 40));
    weightBlocks_.emplace(layer_ids::DECONV1, wrapper(wbData_ + 12348 + 20));
    weightBlocks_.emplace(layer_ids::DECONV2, wrapper(wbData_ + 19568 + 12));
    weightBlocks_.emplace(layer_ids::DECONV3, wrapper(wbData_ + 21740 +3));
    biasBlocks_.emplace(layer_ids::CONV1, wrapper(wbData_ + 0));
    biasBlocks_.emplace(layer_ids::CONV2, wrapper(wbData_ + 2928));
    biasBlocks_.emplace(layer_ids::CONV3, wrapper(wbData_ + 5108));
    biasBlocks_.emplace(layer_ids::RES1_1, wrapper(wbData_ + 24659));
    biasBlocks_.emplace(layer_ids::RES1_2, wrapper(wbData_ + 39099));
    biasBlocks_.emplace(layer_ids::RES2_1, wrapper(wbData_ + 53539));
    biasBlocks_.emplace(layer_ids::RES2_2, wrapper(wbData_ + 67979));
    biasBlocks_.emplace(layer_ids::RES3_1, wrapper(wbData_ + 82419));
    biasBlocks_.emplace(layer_ids::RES3_2, wrapper(wbData_ + 96859));
    biasBlocks_.emplace(layer_ids::RES4_1, wrapper(wbData_ + 111299));
    biasBlocks_.emplace(layer_ids::RES4_2, wrapper(wbData_ + 125739));
    biasBlocks_.emplace(layer_ids::RES5_1, wrapper(wbData_ + 140179));
    biasBlocks_.emplace(layer_ids::RES5_2, wrapper(wbData_ + 154619));
    biasBlocks_.emplace(layer_ids::DECONV1, wrapper(wbData_ + 12348));
    biasBlocks_.emplace(layer_ids::DECONV2, wrapper(wbData_ + 19568));
    biasBlocks_.emplace(layer_ids::DECONV3, wrapper(wbData_ + 21740));
}

// vim: set expandtab ts=4 sw=4:


