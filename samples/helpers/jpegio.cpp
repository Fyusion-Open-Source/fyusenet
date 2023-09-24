//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2022
//--------------------------------------------------------------------------------------------------
// Barebones JPEG I/O
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <jpeglib.h>

//-------------------------------------- Project  Headers ------------------------------------------

#include "jpegio.h"

//-------------------------------------- Global Variables ------------------------------------------

//-------------------------------------- Local Definitions -----------------------------------------


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/


/**
 * @brief Store an 8-bit RGB image data array as JPEG
 *
 * @param img Pointer to image data in RGB order
 * @param width Image width
 * @param height Image height
 * @param name Filename
 * @param quality JPEG quality setting (0..100)
 *
 * This function merely saves an RGB (3-channel) image as JPEG using the specified filename.
 */
void JPEGIO::saveRGBImage(const uint8_t * img, int width, int height, const std::string& name, int quality) {
    struct jpeg_error_mgr jerr;
    struct jpeg_compress_struct jcomp;
    FILE *outfile = fopen(name.c_str(),"wb");
    if (outfile) {
        jcomp.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&jcomp);
        jpeg_stdio_dest(&jcomp, outfile);
        jcomp.input_components = 3;
#ifndef JCS_EXT_RGB
        jcomp.in_color_space = JCS_RGB;
        jcomp.jpeg_color_space = JCS_RGB;
#else
        jcomp.in_color_space = JCS_EXT_RGB;
        jcomp.jpeg_color_space = JCS_EXT_RGB;
#endif
        jcomp.image_width = width;
        jcomp.image_height = height;
        jpeg_set_defaults(&jcomp);
        jcomp.data_precision = 8;
        jpeg_set_quality(&jcomp,quality, TRUE);
        jpeg_start_compress(&jcomp, TRUE);
        JSAMPROW rowptr;
        while ((int)jcomp.next_scanline < height) {
            rowptr = (JSAMPROW) img+jcomp.next_scanline*3*width;
            jpeg_write_scanlines(&jcomp,&rowptr,1);
        }
        jpeg_finish_compress(&jcomp);        
        fclose(outfile);
    }
}


/**
 * @brief Read RGB image from JPEG file
 *
 * @param name Input file name.
 * @param[out] width Width of the image (in pixels)
 * @param[out] height Height of the image (in pixels)
 *
 * @return Pointer to image in RGB pixel order or \c nullptr if image could not be read.
 */
uint8_t * JPEGIO::loadRGBImage(const std::string& name, int & width, int & height) {
    FILE * in = fopen(name.c_str(),"rb");
    if (in) {
        fseek(in,0,SEEK_END);
        size_t size = ftell(in);
        fseek(in,0,SEEK_SET);
        uint8_t * buffer = new uint8_t[size];
        fread(buffer,1,size,in);
        struct jpeg_error_mgr jerr = {0};
        struct jpeg_decompress_struct jdcomp = {0};
        jdcomp.err = jpeg_std_error(&jerr);
        jpeg_create_decompress(&jdcomp);
        jpeg_mem_src(&jdcomp,buffer,size);
        if (jpeg_read_header(&jdcomp , true) != JPEG_HEADER_OK) {
            return nullptr;
        }
        jpeg_start_decompress(&jdcomp);
        width = jdcomp.output_width;
        height = jdcomp.output_height;
        int channels = jdcomp.output_components;
        if (channels != 3) {
            // we only support 3 channels for this sample
            return nullptr;
        }
        uint8_t * image = new uint8_t[width * height * channels];
        while (jdcomp.output_scanline < jdcomp.output_height) {
            uint8_t *ptr = image + jdcomp.output_scanline * channels * width;
            jpeg_read_scanlines(&jdcomp, &ptr, 1);
        }
        jpeg_finish_decompress(&jdcomp);
        jpeg_destroy_decompress(&jdcomp);
        fclose(in);
        delete [] buffer;
        return image;
    } else {
        fprintf(stderr,"Cannot open file %s for reading.\n", name.c_str());
        return nullptr;
    }
}


/**
 * @brief Check if a file is a JPEG file
 *
 * @param name Name of file to check for being a JPEG image
 *
 * @retval true if file is a JPEG image file
 * @retval false otherwise
 *
 * @note This check is rather crude but works for most purposes
 */
bool JPEGIO::isJPEG(const std::string& name) {
    uint8_t expected1[4] = {0xFF, 0xD8, 0xFF, 0xE0};
    uint8_t expected2[4] = {0x4A, 0x46, 0x49, 0x46};
    FILE *in = fopen(name.c_str(),"rb");
    if (!in) return false;
    bool rc = false;
    uint8_t buf[16];
    int read = fread(buf, 1, 16, in);
    if (read != 16) return false;
    if ((!memcmp(buf, expected1, 4)) && (!memcmp(buf+6, expected2, 4))) rc = true;
    fclose(in);
    return rc;
}


// vim: set expandtab ts=4 sw=4:
