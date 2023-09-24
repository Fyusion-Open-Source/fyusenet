//--------------------------------------------------------------------------------------------------
// FyuseNet
//--------------------------------------------------------------------------------------------------
// ZIP File Adapter                                                            (c) Martin Wawro 2023
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <filesystem>
#include <algorithm>
#include <cstring>

//-------------------------------------- Project  Headers ------------------------------------------

#include <fyusenet/fyusenet.h>
#include "zipwalker.h"

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------

static ZipWalker::ZippedFile EMPTYFILE;

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

ZipWalker::ZipWalker(const std::string &fileName) : name_(fileName) {
    const int bufsize = 1024;
    const char eocdsig[4] = {0x50, 0x4b, 0x05, 0x06};
    [[maybe_unused]] size_t discard;
    file_ = fopen(fileName.c_str(), "rb");
    if (!file_) THROW_EXCEPTION_ARGS(fyusion::FynException,"Cannot open file %s", fileName.c_str());
    // FIXME (mw) this EOCD verification is very crude and should be replaced
    char buffer[bufsize];
#ifdef WIN32
    _fseeki64(file_, 0, SEEK_END);
    size_t fullsize = _ftelli64(file_);
#else
    fseek(file_, 0, SEEK_END);
    size_t fullsize = ftell(file_);
#endif
    if (fullsize < bufsize) THROW_EXCEPTION_ARGS(fyusion::FynException, "File %s too small (%ld bytes)", fileName.c_str(), fullsize);
#ifdef WIN32
    _fseeki64(file_, -bufsize, SEEK_END);
    size_t offset = _ftelli64(file_);
#else
    fseek(file_, -bufsize, SEEK_END);
    size_t offset = ftell(file_);
#endif
    discard = fread(buffer, 1, bufsize, file_);
    bool eocdok = false;
    for (int i=0; i < 1024 - (int)sizeof(eocdsig); i++) {
        if ((buffer[i] == eocdsig[0]) && (buffer[i+1] == eocdsig[1]) &&(buffer[i+2] == eocdsig[2]) && (buffer[i+3] == eocdsig[3])) {
            eocdok = parseEOCD((EOCDHeader *)(buffer+i), offset+i);
            if (eocdok) {
                if (!readCentralDirectory()) THROW_EXCEPTION_ARGS(fyusion::FynException, "Cannot read central directory");
                break;
            }
        }
    }
    if (!eocdok) THROW_EXCEPTION_ARGS(fyusion::FynException, "End of central directory not found, invalid zip file");
}

ZipWalker::~ZipWalker() {
    if (file_) fclose(file_);
    file_ = nullptr;
}


const ZipWalker::ZippedFile& ZipWalker::findFileByPath(const std::string &name) const {
    if (auto it = contentsByPath_.find(name) ; it != contentsByPath_.end()) return *it->second;
    else return EMPTYFILE;
}

const ZipWalker::ZippedFile& ZipWalker::findFileByName(const std::string &name) const {
    if (auto it = contentsByName_.find(name) ; it != contentsByName_.end()) return *it->second;
    else return EMPTYFILE;
}


bool ZipWalker::readFile(const ZippedFile& file, uint8_t *buffer) {
    LocalFileHeader hdr = {};
    assert(buffer);
    assert(!file.empty());
    const uint8_t sig[4] = {0x50, 0x4b, 0x03, 0x04};
    [[maybe_unused]] size_t discard;
#ifdef WIN32
    _fseeki64(file_, (size_t)file.offset, SEEK_SET);
#else
    fseek(file_, (size_t)file.offset, SEEK_SET);
#endif
    size_t read = fread(&hdr, 1, sizeof(hdr), file_);
    if (read != sizeof(hdr)) THROW_EXCEPTION_ARGS(fyusion::FynException, "Cannot read local file header");
    if (memcmp(&hdr.signature, sig, 4) != 0) THROW_EXCEPTION_ARGS(fyusion::FynException, "Invalid local file header");
    if (hdr.uncompressed_size != 0xFFFFFFFF) {
        if ((hdr.extra_field_length + hdr.filename_length) > 0) {
#ifdef WIN32
            _fseeki64(file_, hdr.filename_length + hdr.extra_field_length, SEEK_CUR);
#else
            fseek(file_, hdr.filename_length + hdr.extra_field_length, SEEK_CUR);
#endif
        }
        discard = fread(buffer, 1, file.size, file_);
    } else {
        assert(false);
        // TODO (mw) ZIP64 support
    }
    return true;
}


/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

// TODO (mw) docs
bool ZipWalker::readCentralDirectory() {
    CentralDirHeader cdr;
    char namebuffer[1024];
    const uint8_t sig[4] = {0x50, 0x4b, 0x01, 0x02};
#ifdef WIN32
    _fseeki64(file_, cDirOffset_, SEEK_SET);
#else
    fseek(file_, cDirOffset_, SEEK_SET);
#endif
    for (int record=0; record < numRecords_; record++) {
        [[maybe_unused]] size_t discard;
        discard = fread(&cdr, 1, sizeof(cdr), file_);
        if (memcmp(&cdr.signature, sig, 4) != 0) return false;
        if (cdr.compression != 0) return false;
        if ((cdr.filename_len > 1023) || (cdr.filename_len == 0)) return false;
        discard = fread(namebuffer, 1, cdr.filename_len, file_);
        namebuffer[cdr.filename_len] = 0;
        if ((cdr.uncompressed_size != 0xFFFFFFFF) && (cdr.local_header_offset != 0xFFFFFFFF)) {
            contents_.emplace_back(namebuffer, (size_t) cdr.local_header_offset, (size_t) cdr.uncompressed_size);
            if ((cdr.extra_field_length + cdr.comment_len) > 0) fseek(file_, cdr.extra_field_length + cdr.comment_len, SEEK_CUR);
        } else {
            Zip64ExtendedInfo extra;
            assert(cdr.extra_field_length > 0);
            size_t read = fread(&extra, 1, std::min(32, (int)cdr.extra_field_length), file_);
            int offset = 0;
            size_t unsize = cdr.uncompressed_size;
            size_t hoffset = cdr.local_header_offset;
            if (cdr.uncompressed_size == 0xFFFFFFFF) {unsize = *(uint64_t *)(&(extra.extra)+offset); offset+=8;}
            if (cdr.compressed_size == 0xFFFFFFFF) offset+=8;
            if (cdr.local_header_offset == 0xFFFFFFFF) {hoffset = *(uint64_t *)(&(extra.extra)+offset); offset+=8;}
            contents_.emplace_back(namebuffer, hoffset, unsize);
            size_t rem = read - cdr.extra_field_length - cdr.comment_len;
            if (rem > 0) {
#ifdef WIN32
                _fseeki64(file_, rem, SEEK_CUR);
#else
                fseek(file_, rem, SEEK_CUR);
#endif
            }
        }
    }
    for (ZippedFile & item : contents_) {
        std::filesystem::path p(item.name);
        contentsByPath_[item.name] = &item;
        contentsByName_[p.filename().string()] = &item;
    }
    return true;
}


// TODO (mw) docs
bool ZipWalker::parseEOCD(EOCDHeader * eocd, size_t offset) {
    // NOTE (mw) we assume we run on a little-endian architecture
    if (eocd->num_entries == 0xFFFF) {
        // we do not (yet) (fully) support ZIP64
        return false;
    } else {
        numRecords_ = eocd->num_entries;
        cDirOffset_ = eocd->central_dir_offset;
        if (cDirOffset_ > offset) return false;
    }
    return true;
}


// vim: set expandtab ts=4 sw=4:

