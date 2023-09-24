//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// Parameter Provider for Quantized Llama-derived (7B) networks (Header)       (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <vector>

//-------------------------------------- Project  Headers ------------------------------------------

#include <fyusenet/fyusenet.h>
#include "../helpers/zipwalker.h"

//------------------------------------- Public Declarations ----------------------------------------

class ZipWalker;

/**
 * @brief Parameter provider for (quantized) Llama networks
 *
 * This class wraps the parameters for Llama-type networks. The current implementation is aimed
 * at tested at 4-bit GTPQ-quantized networks.
 *
 * @see fyusion::fyusenet::ParameterProvider
 */
class LlaMa4BitFileParameters : public fyusion::fyusenet::ParameterProvider {
    using qtype = fyusion::fyusenet::param_type;
    template<typename T>
    class DataSegmentWrapper : public fyusion::fyusenet::DataWrapper {
     public:
        explicit DataSegmentWrapper(const T * ptr) : ptr_(ptr) {
        }
        ~DataSegmentWrapper() override {
            FNET_DEL_AND_CLEAR_ARRAY(ptr_);
        }

        const std::any get() const override {
            return std::any(ptr_);
        }

        int dec() const override {
            int rem = DataWrapper::dec();
            if (rem == 0) {
                FNET_DEL_AND_CLEAR_ARRAY(ptr_);
            }
            return rem;
        }

    protected:
        mutable const T * ptr_;
    };

 public:
    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    explicit LlaMa4BitFileParameters(const std::string& fileName);
    ~LlaMa4BitFileParameters() override;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    [[nodiscard]] fyusion::fyusenet::DataBlob get(const std::string &name, int layerNo, int subIndex) const override;
    [[nodiscard]] fyusion::fyusenet::param_type dataType(const std::string &name, int layerNo, int subIndex) const override;
 private:
    [[nodiscard]] fyusion::fyusenet::param_type determineDataType(const std::string& name) const;
    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    mutable std::vector<fyusion::fyusenet::DataWrapper *> wrappers_;
    ZipWalker * zipFile_ = nullptr;
    qtype quant_ = qtype::WGT_INT4;
};

// vim: set expandtab ts=4 sw=4:
