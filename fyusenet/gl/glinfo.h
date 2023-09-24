//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// OpenGL Information Object (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <algorithm>
#include <cstdint>

//-------------------------------------- Project  Headers ------------------------------------------

#include "gl_sys.h"
#include "glexception.h"

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

//------------------------------------- Public Declarations ----------------------------------------
namespace fyusion {
//! Lightweight / low-level OpenGL abstraction layer namespace
namespace opengl {


/**
 * @brief Singleton OpenGL information object
 *
 * The GLInfo object collects information about the platform that its running on and stores a set
 * of values and flags that are queried by other parts of the GL abstraction layer. Thus, the
 * GLInfo singleton has to be instantiated (in any GL context) before using any shader or buffer
 * object. Tha instantiation is done by invoking GLInfo::init() .
 *
 * Some of the environment it keeps track of:
 *   - Graphics hardware vendor / model (if available)
 *   - Platform type (desktop/embedded/webGL)
 *   - OpenGL version
 *   - GLSL version
 *   - Number of supported multi-render targets
 *   - Support for 16-bit floating-point
 *   - List of GL extensions
 *
 *  @see GLInfo::init()
 */
class GLInfo {
 public:

    /**
     * @brief Discrete enumerator for OpenGL major/minor versions
     */
    enum glver : uint8_t {
        UNSUPPORTED = 0,            //!< GL version we do not support / know about
        GL_3_0,                     //!< OpenGL 3.0 (desktop)
        GL_3_1,                     //!< OpenGL 3.1 (desktop)
        GL_3_2,                     //!< OpenGL 3.2 (desktop)
        GL_4_0,                     //!< OpenGL 4.0 (desktop)
        GL_4_1,                     //!< OpenGL 4.1 (desktop)
        GL_4_2,                     //!< OpenGL 4.2 (desktop)
        GL_4_3,                     //!< OpenGL 4.3 (desktop)
        GL_4_4,                     //!< OpenGL 4.4 (desktop)
        GL_4_5,                     //!< OpenGL 4.5 (desktop)
        GL_4_6,                     //!< OpenGL 4.6 (desktop)
        GLES_2_0,                   //!< OpenGLES 2.0 (embedded sytems)
        GLES_3_0,                   //!< OpenGLES 3.0 (embedded sytems)
        GLES_3_1,                   //!< OpenGLES 3.1 (embedded sytems)
        GLES_3_2,                   //!< OpenGLES 3.2 (embedded sytems)
        GLES_3_3,                   //!< OpenGLES 3.3 (embedded sytems)
        WEBGL_1_0,                  //!< WebGL 1.0 (browsers), basically like GLES 2.0
        WEBGL_2_0                   //!< WebGL 2.0 (browsers), basically like GLES 3.0
    };

    /**
     * @brief Discrete enumerator for GL shading language version
     */
    enum glslver : uint8_t {
        UNSPECIFIED = 0,
        GLSL_100,
        GLSL_120,
        GLSL_130,
        GLSL_140,
        GLSL_150,
        GLSL_300,
        GLSL_310,
        GLSL_320,
        GLSL_330,
        GLSL_400,
        GLSL_410,
        GLSL_420,
        GLSL_430,
        GLSL_440,
        GLSL_450,
        GLSL_460,
        GLSL_100_ES,
        GLSL_110_ES,
        GLSL_200_ES,
        GLSL_300_ES,
        GLSL_310_ES,
        GLSL_320_ES,
        GLSL_330_ES
    };

    /**
     * @brief Enumerator for GPU type / vendor
     */
    enum gputype : uint8_t {
        GENERIC = 0,
        AMD,
        NVIDIA,
        INTEL,
        ARM_MALI,
        QUALCOMM_ADRENO,
        POWERVR,
        WEBGL
    };

    /**
     * @brief Enumerator for supported shader types
     */
    enum shadertype : uint8_t {
        GEOMETRY = 0,             //!< Geometry shader (not supported by this framework yet)
        TESSELATION,              //!< Tesselation shader (not supported by this framework yet)
        VERTEX,                   //!< Vertex shader
        FRAGMENT,                 //!< Fragment shader
        COMPUTE                   //!< Compute shader (mostly not supported by this framework yet)
    };

    // implementation-specific limits
    constexpr static int MAX_MALI_T_SERIES_RENDER_TARGETS = 2;
    constexpr static int MAX_SUPPORTED_TEXTURE_UNITS = 8;

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------

    /**
     * @brief Retrieve GL platform/version enumerator for this system
     *
     * @return GL platform/version enumerator
     */
    static glver getVersion() {
        if (!initialized_) THROW_EXCEPTION_ARGS(GLException, "GLInfo object not initialized, call init() before using it");
        return instance_.version_;
    }


    /**
     * @brief Check if a system is running GLES
     *
     * @retval true if system is running GLES
     * @retval false otherwise
     *
     * @note When running on WebGL, this function also returns \c false
     */
    static bool isGLES() {
        if (!initialized_) THROW_EXCEPTION_ARGS(GLException, "GLInfo object not initialized, call init() before using it");
        return (instance_.version_ >= GLES_2_0) && (instance_.version_ <= GLES_3_3);
    }

    /**
     * @brief Check if system is running WebGL
     *
     * @retval true if system is running WebGL
     * @retval false otherwise
     */
    static bool isWebGL() {
        if (!initialized_) THROW_EXCEPTION_ARGS(GLException, "GLInfo object not initialized, call init() before using it");
        return instance_.version_ >= WEBGL_1_0;
    }

    /**
     * @brief Retrieve GPU type (actually the vendor)
     *
     * @return GPU vendor
     */
    static gputype getGPUType() {
        if (!initialized_) THROW_EXCEPTION_ARGS(GLException, "GLInfo object not initialized, call init() before using it");
        return instance_.type_;
    }


    /**
     * @brief Check if system's GL implementation support half-precision floating-point
     *
     * @retval true if system support half-precision floating-point
     * @retval false otherwise
     *
     * @see https://www.khronos.org/opengl/wiki/Small_Float_Formats
     */
    static bool supportsHalf() {
        return ((getVersion() >= GLES_3_0) || ((getVersion() >= GL_4_2) && (getVersion()<GLES_2_0)));
    }


    /**
     * @brief Check if system support compute shaders
     *
     * @retval true if compute shaders are supported
     * @retval false otherwise
     */
    static bool supportsComputeShader() {
#if defined(GL_VERSION_4_3) || defined(GL_ES_VERSION_3_1) || defined(GL_ES_VERSION_3_2)
      if ((getVersion() >= GLES_3_1)||
          ((getVersion() >= GL_4_3) && (getVersion() < GLES_2_0))) {
          int data=0;
          glGetError();
          glGetIntegerv(GL_MAX_COMPUTE_UNIFORM_BLOCKS, &data);
          if (glGetError() != GL_NO_ERROR) return false;
          return (data > 0);
      }
      return false;
#else
      return false;
#endif
    }

    /**
     * @brief Check if system supports GL shader layout qualifiers
     *
     * @retval true in case layout qualifiers are supported
     * @retval false otherwise
     *
     * @see https://www.khronos.org/opengl/wiki/Layout_Qualifier_(GLSL)
     */
    static bool hasBinding() {
        if (!initialized_) THROW_EXCEPTION_ARGS(GLException,"GLInfo object not initialized, call init() before using it");
        return instance_.bindingSupport_;
    }

    /**
     * @brief Retrieve (latest) GLSL version supported by the system
     *
     * @return GLSL version enumerator
     */
    static glslver getGLSLVersion() {
        if (!initialized_) THROW_EXCEPTION_ARGS(GLException,"GLInfo object not initialized, call init() before using it");
        return instance_.shaderVersion_;
    }

    /**
     * @brief Retrieve renderer string from OpenGL subsystem
     *
     * @return Renderer string that was found in GL
     */
    static const std::string& getRendererString() {
        if (!initialized_) THROW_EXCEPTION_ARGS(GLException,"GLInfo object not initialized, call init() before using it");
        return instance_.renderer_;
    }

    /**
     * @brief Force abstraction layer to use specified GLSL version
     *
     * @param version GLSL version to use (should be <= the version found on the system)
     */
    static void forceGLSLVersion(glslver version) {
        instance_.shaderVersion_ = version;
    }

    /**
     * @brief Get recommended maximum number of texture units that may be used
     *
     * @return Limit of texture units that should be used
     */
    static int getMaximumRecommendedTextureUnits() {
        return std::min((int)MAX_SUPPORTED_TEXTURE_UNITS, (int)getMaximumTextureUnits());
    }

    static unsigned int getMaxUBOSize();
    static int getMaxVertexUBOs();
    static int getMaxFragmentUBOs();
    static int getMaxUniformVectors(shadertype type);
    static int getMaximumDrawBuffers();
    static int getMaximumRecommendedDrawBuffers();
    static int getMaximumTextureSize();
    static int getMaximumTexArrayDepth();
    static int getMaxVaryingVectors();
    static int getMaximumTextureUnits();
    static bool hasExtension(const char *extension);

    static void init(bool chatty = true);
 private:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    GLInfo();
    void queryVersion();
    void queryChipset();
    void queryShader();
    void queryExtensions();
    static const char * scanShaderVersion(const char *string);
    static void printInfo();

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::string renderer_;                  //!< Renderer string found in GL system
    std::string extensions_;                //!< String with supported GL extensions in the system
    glver version_ = UNSUPPORTED;           //!< OpenGL version found on the system
    gputype type_ = GENERIC;                //!< GPU vendor/type found on the system
    glslver shaderVersion_ = UNSPECIFIED;   //!< GLSL shader version supported by the system
    int recommendedDrawBuffers_ = -1;       //!< Recommended number of draw buffers for multiple render targets
    bool bindingSupport_ = false;           //!< Whether or not system supports layout qualifiers
    static bool initialized_;               //!< Indicator if GLInfo singleton has been initialized
    static GLInfo instance_;                //!< GLInfo singleton
};

} // opengl namespace
} // fyusion namespace


// vim: set expandtab ts=4 sw=4:
