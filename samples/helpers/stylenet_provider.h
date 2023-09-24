//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Parameter Providers for Style-Transfer Network Samples (Header)             (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <cstdint>
#include <unordered_map>
#include <exception>

//-------------------------------------- Project  Headers ------------------------------------------

#include <fyusenet/fyusenet.h>

//------------------------------------- Public Declarations ----------------------------------------

/**
 * @brief Parameter provider for sample Style-Transfer network(s)
 */
class StyleNetProvider : public fyusion::fyusenet::ParameterProvider {
 public:
    using DataBlob = fyusion::fyusenet::DataBlob;
    using param_type = fyusion::fyusenet::param_type;
    using wrapper = fyusion::fyusenet::DefaultDataWrapper<float>;

    /**
     * Indices for the layer numbers
     */
    enum layer_ids {
        UNPACK = 0,
        UPLOAD = 0,
        CONV1
    };

    StyleNetProvider() : ParameterProvider() {
    }

    ~StyleNetProvider() override {
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
        assert(subIndex >= 0 && subIndex < 2);
        if (subIndex == 0) {
            if (auto it = weightBlocks_.find(layerNo); it != weightBlocks_.end()) {
                return DataBlob((fyusion::fyusenet::DataWrapper *) &it->second);
            }
        } else {
            if (auto it = biasBlocks_.find(layerNo); it != biasBlocks_.end()) {
                return DataBlob((fyusion::fyusenet::DataWrapper *) &it->second);
            }
        }
        return DataBlob();
    }

 protected:
    void loadFile(const std::string& fileName, int numFloats);
    std::unordered_map<int, wrapper> weightBlocks_;
    std::unordered_map<int, wrapper> biasBlocks_;
    float * wbData_ = nullptr;                          //!< Pointer to weight data block
};


/**
 * @brief Parameter provider for sample 3x3-conv-based Style-Transfer network(s)
 */
class StyleNet3x3Provider : public StyleNetProvider {
    constexpr static int STYLENET_SIZE = 77235;   // number of floats per network weights/biases
 public:
    /**
     * Indices for the layer numbers
     */
    enum layer_ids {
        UNPACK = StyleNetProvider::layer_ids ::UNPACK,
        UPLOAD = StyleNetProvider::layer_ids ::UPLOAD,
        CONV1 = StyleNetProvider::layer_ids ::CONV1,
        CONV2,
        CONV3,
        RES1_1,
        RES1_2,
        RES2_1,
        RES2_2,
        DECONV1,
        DECONV2,
        DECONV3,
        SIGMOID,
        DOWNLOAD
    };

    StyleNet3x3Provider(const uint8_t *memory, size_t bytes);
    explicit StyleNet3x3Provider(const std::string& fileName);
 private:
    StyleNet3x3Provider();
};


/**
 * @brief Parameter provider for sample 9x9-conv-based Style-Transfer network(s)
 */
class StyleNet9x9Provider : public StyleNetProvider {
    constexpr static int STYLENET_SIZE = 169059;   // number of floats per network weights/biases
public:
    /**
     * Indices for the layer numbers
     */
    enum layer_ids {
        UNPACK = StyleNetProvider::layer_ids ::UNPACK,
        UPLOAD = StyleNetProvider::layer_ids ::UPLOAD,
        CONV1 = StyleNetProvider::layer_ids ::CONV1,
        CONV2,
        CONV3,
        RES1_1,
        RES1_2,
        RES2_1,
        RES2_2,
        RES3_1,
        RES3_2,
        RES4_1,
        RES4_2,
        RES5_1,
        RES5_2,
        DECONV1,
        DECONV2,
        DECONV3,
        SIGMOID,
        DOWNLOAD
    };

    StyleNet9x9Provider(const uint8_t *memory, size_t bytes);
    explicit StyleNet9x9Provider(const std::string& fileName);
 private:
    StyleNet9x9Provider();
};


// vim: set expandtab ts=4 sw=4:

