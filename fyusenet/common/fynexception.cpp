//--------------------------------------------------------------------------------------------------
// FyuseNet                                                               (c) Fyusion Inc. 2016-2022
//--------------------------------------------------------------------------------------------------
// Custom Exception Baseclass
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

//--------------------------------------- System Headers -------------------------------------------

#include <string>
#include <exception>

//-------------------------------------- Project  Headers ------------------------------------------

#include "fynexception.h"

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------

namespace fyusion {


/*##################################################################################################
#                                   P U B L I C  F U N C T I O N S                                 #
##################################################################################################*/

/**
 * @brief Constructor
 */
FynException::FynException():std::exception() {
}


/**
 * @brief Constructor
 *
 * @param function Function name that caused the exception
 * @param file File name that caused the exception
 * @param line Line in file that caused the exception
 * @param format Format string that carries additional information / custom message
 */
FynException::FynException(const char *function, const char *file, int line, const char *format, ...) {
    char tmp[MAX_MESSAGE_SIZE];   // NOTE (mw) this is bad, the vsnprintf() alleviates this, otherwise this would be prone to buffer-overflow attacks
    va_list args;
    va_start(args,format);
    vsnprintf(tmp,MAX_MESSAGE_SIZE,format,args);
    va_end(args);
    generateWhat(function, file, line, __FUNCTION__, tmp);
}


/**
 * @brief Destructor
 */
FynException::~FynException() throw() {
}

/**
 * @brief Retrieve exception message
 *
 * @return Pointer to null-terminated string with information about the exception
 */
const char * FynException::what() const throw() {
    if (message_.size()>0) return message_.c_str();
    else return nullptr;
}

/*##################################################################################################
#                               N O N -  P U B L I C  F U N C T I O N S                            #
##################################################################################################*/


/**
 * @brief Generate exception message
 *
 * @param function Function name that caused the exception
 * @param file File name that caused the exception
 * @param line Line in file that caused the exception
 * @param ex Exception name
 */
void FynException::generateWhat(const char *function,const char *file,int line,const char *ex) {
    char tmp[MAX_MESSAGE_SIZE+MAX_INFO_SIZE];   // NOTE (mw) this is bad, the snprintf() alleviates this, otherwise this would be prone to buffer-overflow attacks
    snprintf(tmp,sizeof(tmp),"%s:%d [%s] threw %s\n",file,line,function,ex);
    message_ = std::string(tmp);
}


/**
 * @brief Generate exception message
 *
 * @param function Function name that caused the exception
 * @param file File name that caused the exception
 * @param line Line in file that caused the exception
 * @param ex Exception name
 * @param err Custom exception message
 */
void FynException::generateWhat(const char *function,const char *file,int line,const char *ex,char *err) {
    char tmp[MAX_MESSAGE_SIZE+MAX_INFO_SIZE];   // NOTE (mw) this is bad, the snprintf() alleviates this, otherwise this would be prone to buffer-overflow attacks
    snprintf(tmp,sizeof(tmp),"%s:%d [%s] threw %s\nDetailed error: %s\n",file,line,function,ex,err);
    message_ = std::string(tmp);
}


} // fyusion namespace

// vim: set expandtab ts=4 sw=4:
