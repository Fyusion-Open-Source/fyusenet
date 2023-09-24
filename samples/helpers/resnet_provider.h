//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Parameter Provider for ResNet-50 Network  (Header)                          (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <cstdint>
#include <unordered_map>

//-------------------------------------- Project  Headers ------------------------------------------

#include <fyusenet/fyusenet.h>

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Parameter provider for ResNet-50 network
 *
 * Very simple parameter provider that wraps around a block of memory.
 */
class ResNet50Provider : public fyusion::fyusenet::ParameterProvider {
public:
    using DataBlob = fyusion::fyusenet::DataBlob;
    using param_type = fyusion::fyusenet::param_type;
    using wrapper = fyusion::fyusenet::DefaultDataWrapper<float>;

    ResNet50Provider(const uint8_t *memory, size_t bytes);
    explicit ResNet50Provider(const std::string& fileName);

    ~ResNet50Provider() override {
        delete[] wbData_;
        wbData_ = nullptr;
    }

    /**
     * @copydoc fyusion::fyusenet::ParameterProvider::dataType
     */
    [[nodiscard]] param_type dataType(const std::string &name, int layerNo, int subIndex) const override {
        return param_type::WGT_FLOAT32;
    }

    /**
     * @copydoc fyusion::fyusenet::ParameterProvider::get
     */
    [[nodiscard]] DataBlob get(const std::string &name, int layerNo, int subIndex=0) const override {
        switch (subIndex) {
            case 0:
                if (auto it = weightBlocks_.find(layerNo); it != weightBlocks_.end()) {
                    return DataBlob((fyusion::fyusenet::DataWrapper *) &it->second);
                }
                break;
            case 1:
                if (auto it = biasBlocks_.find(layerNo); it != biasBlocks_.end()) {
                    return DataBlob((fyusion::fyusenet::DataWrapper *) &it->second);
                }
                break;
            case 2:
                if (auto it = bnBlocks_.find(layerNo); it != bnBlocks_.end()) {
                    return DataBlob((fyusion::fyusenet::DataWrapper *) &it->second);
                }
                break;
            default:
                break;
        }
        return DataBlob();
    }

 private:
    ResNet50Provider();
    void loadFile(const std::string& fileName, int numFloats);
    std::unordered_map<int, wrapper> weightBlocks_;
    std::unordered_map<int, wrapper> biasBlocks_;
    std::unordered_map<int, wrapper> bnBlocks_;
    float * wbData_ = nullptr;
    size_t totalWeightBytes_ = 0;
};




// vim: set expandtab ts=4 sw=4:

