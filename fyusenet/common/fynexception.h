//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Custom Exception Baseclass (Header)
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

#pragma once

//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <exception>
#include <stdarg.h>

//-------------------------------------- Project  Headers ------------------------------------------


//-------------------------------------- Public Definitions ----------------------------------------

#define THROW_EXCEPTION_ARGS(exclass,...) throw exclass(__PRETTY_FUNCTION__,__FILE__,__LINE__,__VA_ARGS__)
#define THROW_EXCEPTION_ARGS_EXTRA(exclass,extra,...) throw exclass(__PRETTY_FUNCTION__,__FILE__,__LINE__,extra,__VA_ARGS__)

#define CUSTOM_EXCEPTION(name,base)                                                                \
class name:public base {                                                                           \
 public:                                                                                           \
    name():base() {                                                                                \
    }                                                                                              \
    name(const char *func,const char *file,int line):base() {                                      \
        generateWhat(func,file,line,__FUNCTION__);                                                 \
    }                                                                                              \
    name(const char *func, const char *file, int line, const char *fmt, ...)                       \
               __attribute__ ((format (printf, 5, 6))) : base() {                                  \
        char tmp[MAX_MESSAGE_SIZE];                                                                \
        va_list args;                                                                              \
        va_start(args,fmt);                                                                        \
        vsnprintf(tmp,MAX_MESSAGE_SIZE,fmt,args);                                                  \
        va_end(args);                                                                              \
        generateWhat(func,file,line,__FUNCTION__,tmp);                                             \
    }                                                                                              \
    virtual ~name() throw() {                                                                      \
    }                                                                                              \
    name(const name& ex):base(ex) {                                                                \
    }                                                                                              \
}


//------------------------------------- Public Declarations ----------------------------------------

namespace fyusion {

/**
 * @brief Base exception class for FyuseNet
 *
 * This class derives from the std c++ exception class and is used as base exception class
 * within FyuseNet. It adds a bit of functionality to an exception in order to make debugging
 * and troubleshooting a bit easier.
 *
 * To benefit from the enhanced functionality, always make sure to use the preprocessor macros
 * for throwing exceptions, as in the following example:
 *
 * @code
 * int something_is_wrong = 5;
 * if (something_is_wrong) {
 *     THROW_EXCEPTION_ARGS(FynException,"Well, that did not go as planned (%d)", something_is_wrong);
 * }
 * @endcode
 *
 * This will throw an exception of type FynException that features the supplied error message, as
 * well as the function-name, file-name and line within that file where the exception was thrown.
 */
class FynException:public std::exception {
 public:
    enum {
        MAX_INFO_SIZE = 768,
        MAX_MESSAGE_SIZE = 8192
    };

    // ------------------------------------------------------------------------
    // Constructor / Destructor
    // ------------------------------------------------------------------------
    FynException();
    FynException(const char *function, const char *file, int line, const char *format, ...);
    virtual ~FynException() throw();

    // ------------------------------------------------------------------------
    // Public methods
    // ------------------------------------------------------------------------
    virtual const char * what() const throw() override;

 protected:
    // ------------------------------------------------------------------------
    // Non-public methods
    // ------------------------------------------------------------------------
    void generateWhat(const char *function,const char *file,int line,const char *ex);
    void generateWhat(const char *function,const char *file,int line,const char *ex,char *err);

    // ------------------------------------------------------------------------
    // Member variables
    // ------------------------------------------------------------------------
    std::string message_;
};



} // fyusenet namespace


// vim: set expandtab ts=4 sw=4:
