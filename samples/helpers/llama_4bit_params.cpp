//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Parameter Provider for Quantized Llama-derived (7B) networks                (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstdint>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../helpers/zipwalker.h"
#include "llama_4bit_params.h"

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor (around file)
 *
 * @param fileName Filename of parameter file to wrap
 */
LlaMa4BitFileParameters::LlaMa4BitFileParameters(const std::string &fileName) {
    zipFile_ = new ZipWalker(fileName);
}


/**
 * @brief Destructor
 */
LlaMa4BitFileParameters::~LlaMa4BitFileParameters() {
    for (auto * wrapper : wrappers_) delete wrapper;
    wrappers_.clear();
    FNET_DEL_AND_CLEAR(zipFile_);
    zipFile_ = nullptr;
}

/**
 * @copydoc ParameterProvider::get
 */
fyusion::fyusenet::DataBlob LlaMa4BitFileParameters::get(const std::string &name, int layerNo, int subIndex) const {
    using namespace fyusion::fyusenet;
    auto zipped = zipFile_->findFileByName(name);
    if (zipped.empty()) THROW_EXCEPTION_ARGS(fyusion::FynException,"Data %s does not exist in parameter file", name.c_str());
    assert(zipped.size > 0);
    auto type = determineDataType(zipped.name);
    DataWrapper * wrapper = nullptr;
    void * buffer = nullptr;
    switch (type) {
        case qtype::WGT_FLOAT32:
            buffer = reinterpret_cast<void *>(new float[(zipped.size + sizeof(float)-1) / sizeof(float)]);
            wrapper = new DataSegmentWrapper<float>(reinterpret_cast<float *>(buffer));
            break;
        case qtype::WGT_FLOAT16:
            buffer = reinterpret_cast<void *>(new uint16_t[(zipped.size + sizeof(uint16_t)-1) / sizeof(uint16_t)]);
            wrapper = new DataSegmentWrapper<uint16_t>(reinterpret_cast<uint16_t *>(buffer));
            break;
        case qtype::WGT_INT4:
            buffer = reinterpret_cast<void *>(new uint8_t[zipped.size ]);
            wrapper = new DataSegmentWrapper<uint8_t>(reinterpret_cast<uint8_t *>(buffer));
            break;
        default:
            assert(false);
    }
    wrappers_.emplace_back(wrapper);
    DataBlob blob(wrapper);
    assert(buffer);
    zipFile_->readFile(zipped, (uint8_t *)buffer);
    return blob;
}


/**
 * @copydoc ParameterProvider::dataType
 */
fyusion::fyusenet::param_type LlaMa4BitFileParameters::dataType(const std::string &name, int layerNo, int subIndex) const {
    auto zipped = zipFile_->findFileByName(name);
    if (zipped.empty()) THROW_EXCEPTION_ARGS(fyusion::FynException,"Data %s does not exist in parameter file", name.c_str());
    return determineDataType(zipped.name);
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Query the data-type of a named parameter subset
 *
 * @param name Name of the parameter set
 *
 * @return Data type
 */
fyusion::fyusenet::param_type LlaMa4BitFileParameters::determineDataType(const std::string &name) const {
    if (name.find("int32") != std::string::npos) return quant_;
    if (name.find("float16") != std::string::npos) return qtype::WGT_FLOAT16;
    if (name.find("float32") != std::string::npos) return qtype::WGT_FLOAT;
    THROW_EXCEPTION_ARGS(fyusion::FynException,"Unknown data type for parameter %s", name.c_str());
}

// vim: set expandtab ts=4 sw=4:
