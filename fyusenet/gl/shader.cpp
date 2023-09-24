//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// GLSL Shader Wrapper
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <cstring>
#include <errno.h>

//-------------------------------------- Project  Headers ------------------------------------------

#include "shader.h"
#include "shaderexception.h"
#include "shadersnippet.h"
#include "../common/logging.h"

//-------------------------------------- Global Variables ------------------------------------------

//-------------------------------------- Local Definitions -----------------------------------------

namespace fyusion {
namespace opengl {

/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 *
 * @param type Shader type, e.g. \c GL_VERTEX_SHADER
 * @param context OpenGL context to use this shader with
 * @param version Optional parameter that specifies the GLSL version to use for the shader,
 *                defaults to UNSPECIFIED to use the most recent version available on the platform.
 *
 * Constructs an empty Shader object for the specified shader type. The \p version parameter
 * can be used to override the GLSL version. No shader handle is created at this point. Shader handles
 * are created when the shader is compiled.
 *
 * @note It is recommended to use the derived classes VertexShader or FragmentShader
 */
Shader::Shader(GLenum type, const fyusenet::GfxContextLink & context, GLInfo::glslver version) : GfxContextTracker() {
    setContext(context);
    handle_ = 0;
    if (version == 0) {
        version_ = GLInfo::getGLSLVersion();
    } else {
        version_ = version;
    }
    type_ = type;
    hash_ = 0;
}


/**
 * @brief Destructor
 *
 * Deallocates GL resources associated with the shader by deleting the shader and invalidating
 * the handle.
 *
 * @pre GL context under which the shader was created must be current to the calling thread
 */
Shader::~Shader() {
    if (handle_ != 0) {
        assertContext();
        glDeleteShader(handle_);
        handle_ = 0;
    }
}


/**
 * @brief Retrieve shader type of this object
 *
 * @return GL enumerator with the shader type of this object (e.g. \c GL_FRAGMENT_SHADER )
 */
GLenum Shader::getType() const {
    return type_;
}


/**
 * @brief Explcicitly release shader resources by deleting shader
 *
 * @pre GL context under which the shader was created must be current to the calling thread
 * @post Shader handle is invalidated
 */
void Shader::release() {
    if (isCompiled()) {
        assertContext();
        glDeleteShader(handle_);
        handle_ = 0;
    }
}

/**
 * @brief Check if shader is compiled
 *
 * @retval true if shader was compiled already
 * @retval false if shader is not compiled
 */
bool Shader::isCompiled() const {
    return (handle_ != 0);
}


/**
 * @brief Set code to shader object
 *
 * @param data String that contains the shader source code for this shader object
 *
 * This function uses the supplied \p data to perform three things:
 *   1) Determined a  preamble ("#version") based on the version
 *   2) Resolve \c \#include statements in the shader code
 *   3) Store the preamble and resolved code as object-internal strings
 *
 * No compilation is done at this point, see the #compile() function for that.
 */
void Shader::setCode(const std::string& data) {
    // FIXME (mw) this is a mess, clean it up
    if (!strstr(data.c_str(),"#version")) {
        switch (version_) {
#if defined(FYUSENET_USE_EGL) || defined(FYUSENET_USE_WEBGL)
        case GLInfo::GLSL_100_ES:
            preamble_ = std::string("#version 100 es\n#define GLES\n");
            break;
        case GLInfo::GLSL_110_ES:
            preamble_ = std::string("#version 110 es\n#define GLES\n");
            break;
        case GLInfo::GLSL_200_ES:
#ifdef FYUSENET_USE_WEBGL
            preamble_ = std::string("#version 200 es\n#define GLES\n#define WEBGL1\n");
#else
            preamble_ = std::string("#version 200 es\n#define GLES\n");
#endif
            break;
        case GLInfo::GLSL_300_ES:
#ifdef FYUSENET_USE_WEBGL
            preamble_ = std::string("#version 300 es\n#define GLES\n#define WEBGL2\n");
#else
            preamble_ = std::string("#version 300 es\n#define GLES\n");
#endif
            break;
        case GLInfo::GLSL_310_ES:
            preamble_ = std::string("#version 310 es\n#define BINDING_SUPPORT\n#define GLES\n");
            break;
        case GLInfo::GLSL_320_ES:
            preamble_ = std::string("#version 320 es\n#define BINDING_SUPPORT\n#define GLES\n");
            break;
        case GLInfo::GLSL_330_ES:
            preamble_ = std::string("#version 330 es\n#define BINDING_SUPPORT\n#define GLES\n");
            break;
        default:
#ifdef FYUSENET_USE_WEBGL
            preamble_ = std::string("#version 100 es\n#define GLES\n#define WEBGL1\n");
#else
            preamble_ = std::string("#version 100 es\n#define GLES\n");
#endif
#else
        case GLInfo::GLSL_100:
            preamble_ = std::string("#version 100\n");
            break;
        case GLInfo::GLSL_120:
            preamble_ = std::string("#version 120\n");
            break;
        case GLInfo::GLSL_130:
            preamble_ = std::string("#version 130\n");
            break;
        case GLInfo::GLSL_140:
            preamble_ = std::string("#version 140\n");
            break;
        case GLInfo::GLSL_150:
            preamble_ = std::string("#version 150\n");
            break;
        case GLInfo::GLSL_300:
            preamble_ = std::string("#version 300\n");
            break;
        case GLInfo::GLSL_310:
            preamble_ = std::string("#version 310\n");
            break;
        case GLInfo::GLSL_320:
            preamble_ = std::string("#version 320\n");
            break;
        case GLInfo::GLSL_330:
            preamble_ = std::string("#version 330\n");
            break;
        case GLInfo::GLSL_400:
            preamble_ = std::string("#version 400\n");
            break;
        case GLInfo::GLSL_410:
            preamble_ = std::string("#version 410\n");
            break;
        case GLInfo::GLSL_420:
            preamble_ = std::string("#version 420\n");
            break;
        case GLInfo::GLSL_430:
            preamble_ = std::string("#version 430\n#define BINDING_SUPPORT\n");
            break;
        case GLInfo::GLSL_440:
            preamble_ = std::string("#version 440\n#define BINDING_SUPPORT\n");
            break;
        case GLInfo::GLSL_450:
            preamble_ = std::string("#version 450\n#define BINDING_SUPPORT\n");
            break;
        case GLInfo::GLSL_460:
            preamble_ = std::string("#version 460\n#define BINDING_SUPPORT\n");
            break;
        default:
            preamble_ = std::string("#version 100\n");
#endif
        }
    }
    shaderCode_ = includeSnippets(data);
}


/**
 * @brief Set shader code to shader object
 *
 * @param data Pointer to string with shader code
 *
 * This is an overloaded version of setCode(std::string), see the documentation there.
 */
void Shader::setCode(const char *data) {
    if (data) setCode(std::string(data));
}


/**
 * @brief Log shader (including preamble, extra definitions)
 *
 * This logs/prints the shader code to the log channel (stdout when in doubt)
 */
void Shader::log() const {
    std::string comb = preamble_ + preprocDefs_ + shaderCode_;
    logShader(comb.c_str());
}


/**
 * @brief Set a resource name to the shader
 *
 * @param name Resource name to set into the shader
 *
 * For shaders that originate from the shader resource subsystem, this function can be used to
 * set the resource name (origin) of the shader.
 *
 * @see ShaderBuilder::shaderProgramFromResource
 */
void Shader::setResourceName(const std::string &name) {
    resourceName_ = name;
}


/**
 * @brief Set additional preprocessor definitions for the shader
 *
 * @param defs String with preprocessor definitions to set
 *
 * Use this function to add additional pre-processor definitions into the shader (right after
 * the preamble). These definitions are supplied in the form of strings, for example:
 * @code
 * const char * extra = "#define MYDEF 1\n";
 * shader->setPreprocDefs(extra);
 * @endcode
 * Make sure that every preprocessor definition ends with a newline character, otherwise the
 * shader will not compile.
 */
void Shader::setPreprocDefs(const std::string& defs) {
    preprocDefs_ = defs;
}


/**
 * @brief Set additional preprocessor definitions for the shader
 *
 * @param defs String with preprocessor definitions to set
 *
 * This is an overloaded function provided for convenience, check #setPreprocDefs(const std::string&)
 * for documentation.
 */
void Shader::setPreprocDefs(const char *defs) {
    if (defs) {
        setPreprocDefs(std::string(defs));
    } else preprocDefs_ = std::string();
}


#ifndef ANDROID
/**
 * @brief Load shader from file
 *
 * @param fileName Filename to load
 */
void Shader::loadFromFile(const char *fileName) {
    FILE *f = fopen(fileName,"rb");
    if (f) {
        fseek(f,0,SEEK_END);
        size_t filesize = ftell(f);
        fseek(f,0,SEEK_SET);
        if (filesize > 0) {
            char *code = new char[filesize+1];
            code[filesize] = 0;
            fread(code,1,filesize,f);
            fclose(f);
            setCode(code);
            delete [] code;
        } else fclose(f);
    } else {
        THROW_EXCEPTION_ARGS(ShaderException,"Cannot open shader file %s (errno=%d)",fileName, errno);
    }
}
#endif

/**
 * @brief Compile shader source
 *
 * This function compiles the shader source into a shader object. If no GL handle for the shader
 * was present, a new one is created. Successful compilation of the shader renders it able to
 * be used as part of a shader program, which requires linking the shader (see ShaderProgram
 * class).
 *
 * @throws ShaderException in case the compilation was unsuccessful
 */
void Shader::compile() {
    if (shaderCode_.size() == 0) THROW_EXCEPTION_ARGS(ShaderException,"No shader code supplied");
    std::string comb = preamble_ + preprocDefs_ + shaderCode_;
    compile(comb.c_str());
}


/**
 * @brief Get shader string as it is sent to the GL driver
 *
 * @return String with "fully processed" shader code
 */
std::string Shader::getCode() const {
    return preamble_ + preprocDefs_ + shaderCode_;
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/

/**
 * @brief Compile shader source
 *
 * @param data Shader code to compile
 *
 * This function compiles the provided shader source into a shader object. If no GL handle for the
 * shader was present, a new one is created. Successful compilation of the shader renders it able to
 * be used as part of a shader program, which requires linking the shader (see ShaderProgram
 * class).
 *
 * @throws ShaderException in case the compilation was unsuccessful
 */
void Shader::compile(const char *data) {
    GLint status = GL_FALSE;
    if (!data) THROW_EXCEPTION_ARGS(ShaderException,"Null shader code supplied");
    handle_ = glCreateShader(type_);
    if (handle_ == 0) THROW_EXCEPTION_ARGS(ShaderException,"Cannot create shader");
    glShaderSource(handle_, 1, &data, nullptr);
    glCompileShader(handle_);
    glGetShaderiv(handle_, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        logError();
        logShader(data);
        glDeleteShader(handle_);
        handle_ = 0;
        THROW_EXCEPTION_ARGS(ShaderException,"Error compiling shader");
    }
}


/**
 * @brief Log compiler error message
 *
 * In case the compilation was not successful, this function can be used to log the compiler error
 * message to the logging facility.
 */
void Shader::logError() const {
    GLint loglen = 0;
    glGetShaderiv(handle_, GL_INFO_LOG_LENGTH, &loglen);
    if (loglen > 0) {
        char *log = new char[loglen+1];
        glGetShaderInfoLog(handle_, loglen, &loglen, log);
        char *ptr = log;
        while ((ptr) && (ptr[0] != 0)) {
            char *nptr = strpbrk(ptr,"\n\r");
            if (nptr) {
                while ((nptr[0] == 10) || (nptr[0] == 13)) {
                    nptr[0]=0;
                    nptr++;
                }
                FNLOGI("%s",ptr);
                if (nptr == ptr) ptr=nullptr;
                else ptr = nptr;
            } else {
                FNLOGI("%s",ptr);
                ptr=nullptr;
            }
        }
        delete [] log;
    } else {
        FNLOGI("<no compilation log>");
    }
}



/**
 * @brief Log shader code to logging facility
 *
 * @param data Pointer to shader code that should be logged
 *
 * This function logs the supplied shader source code to the logging facility and adds line numbers
 * to every line.
 *
 * @warning This function is \b destructive to the supplied shader data
 */
void Shader::logShader(const char *data) const {
    int line=1;
    if (!data) {
        FNLOGI("(null)");
        return;
    }
    char *ptr = const_cast<char *>(data);
    while ((ptr) && (ptr[0] != 0)) {
        char *nptr = strpbrk(ptr,"\n");
        if (nptr) {
            nptr[0]=0;
            if (nptr[1] == 13) {
                nptr[1] = 0;
                nptr++;
            }
            nptr++;
            FNLOGI("%4d: %s",line++,ptr);
            ptr = nptr;
        } else {
            FNLOGI("%4d: %s",line++,ptr);
            ptr = nullptr;
        }
    }
}


/**
 * @brief Resolve include statements in shader code
 *
 * @param code Shader source code to run the include-statement resolver on
 *
 * @return Shader source code after include-statement resolution
 *
 * @throws GLException in case of errors
 *
 * This function resolves any \c \#include statements in a shader source by looking up the
 * filenames using the ShaderSnippet class, which is basically a resource directory containing
 * all "shader snippets" which are the shader parts that can be included via the \c \#include
 * statement.
 *
 * @see ShaderSnippet::loadFromResource
 */
std::string Shader::includeSnippets(const std::string& code) {
    size_t copystart = 0;
    size_t ipos = code.find("#include");
    if (ipos == std::string::npos) return code;
    std::string output = code.substr(0,ipos-1);
    do {
        size_t lineend;
        for (lineend = ipos ;lineend < code.size(); lineend++) {
            if ((code[lineend] == 10)||(code[lineend] == 13)) {
                if (lineend+1 < code.size()) {
                    if ((code[lineend+1] == 10)||(code[lineend+1] == 13)) lineend++;
                }
                break;
            }
        }
        std::string incline = code.substr(ipos,lineend-ipos);
        char * fstart = strpbrk(&(incline[0]),"\"<");
        char * fend = strpbrk((fstart) ? fstart+1 : &(incline[0]),"\">");
        if ((!fstart) || (!fend)) {
            THROW_EXCEPTION_ARGS(GLException,"Invalid include statement: %s",incline.c_str());
        }
        char * filename = fstart+1;
        *fend=0;
        const ShaderSnippet * snip = ShaderSnippet::loadFromResource(std::string(filename));
        if (snip) output += snip->code();
        else {
            THROW_EXCEPTION_ARGS(GLException,"Shader snippet %s not found",filename);
        }
        if (lineend >= code.size()) break;
        copystart = lineend+1;
        ipos = code.find("#include",copystart);
        if (ipos == std::string::npos) output += code.substr(copystart,code.size()-copystart);
        else output += code.substr(copystart,ipos-copystart);
    } while (ipos != std::string::npos);
    return output;
}


} // opengl namespace
} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
