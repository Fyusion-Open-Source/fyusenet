//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// ZIP File Adapter (Header)                                                   (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdio>
#include <string>
#include <list>
#include <unordered_map>
#include <cstdint>

//-------------------------------------- Project  Headers ------------------------------------------


//------------------------------------- Public Declarations ----------------------------------------

#ifndef PACK
#ifdef __GNUC__
#define PACK(_declr_) _declr_ __attribute__((__packed__))
#endif

#ifdef _MSC_VER
#define PACK(_decl_) __pragma(pack(push,1)) _decl_ __pragma(pack(pop))
#endif
#endif


/**
 * @brief Adapter class to read from (uncompressed) ZIP files
 */
class ZipWalker {

    PACK(
    struct LocalFileHeader {
        uint8_t  signature[4];
        uint16_t version;
        uint16_t flags;
        uint16_t compression;
        uint16_t modtime;
        uint16_t moddate;
        uint32_t crc32;
        uint32_t compressed_size;
        uint32_t uncompressed_size;
        uint16_t filename_length;
        uint16_t extra_field_length;
    });

    PACK(
    struct EOCDHeader {
        uint8_t  signature[4];
        uint16_t disk_number;
        uint16_t disk_start;
        uint16_t num_entries;
        uint16_t total_entries;
        uint32_t central_dir_size;
        uint32_t central_dir_offset;
        uint16_t comment_length;
    });

    PACK(
    struct CentralDirHeader {
        uint8_t  signature[4];
        uint16_t version;
        uint16_t version_needed;
        uint16_t flags;
        uint16_t compression;
        uint16_t mod_time;
        uint16_t mod_date;
        uint32_t crc32;
        uint32_t compressed_size;
        uint32_t uncompressed_size;
        uint16_t filename_len;
        uint16_t extra_field_length;
        uint16_t comment_len;
        uint16_t disk_num;
        uint16_t internal_attr;
        uint32_t external_attr;
        uint32_t local_header_offset;
    });

    PACK(
    struct Zip64ExtendedInfo {
        uint16_t header_id;
        uint16_t data_size;
        uint8_t extra[28];
    });

 public:

    struct ZippedFile {
        ZippedFile() = default;
        ZippedFile(const std::string& name, size_t offset, size_t size) : name(name), offset(offset), size(size) {}

        [[nodiscard]] bool empty() const {
            return size == 0;
        }
        std::string name;
        size_t offset = 0;
        size_t size = 0;
    };

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    ZipWalker(const std::string& fileName);
    ~ZipWalker();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    const ZippedFile& findFileByPath(const std::string& name) const;
    const ZippedFile& findFileByName(const std::string& name) const;
    bool readFile(const ZippedFile& file, uint8_t *buffer);

    int numFiles() const {
        return contents_.size();
    }

    bool isEmpty() const {
        return contents_.empty();
    }

    bool isValid() const {
        return valid_;
    }
private:

    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void scanContents();
    bool parseEOCD(EOCDHeader * eocd, size_t offset);
    bool readCentralDirectory();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    FILE * file_ = nullptr;
    std::string name_;
    std::list<ZippedFile> contents_;
    std::unordered_map<std::string, ZippedFile *> contentsByName_;
    std::unordered_map<std::string, ZippedFile *> contentsByPath_;
    int numRecords_ = 0;
    size_t cDirOffset_ = 0;
    bool valid_ = false;
};


// vim: set expandtab ts=4 sw=4:

