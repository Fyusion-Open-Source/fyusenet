file(GLOB COMMON_SOURCES CONFIGURE_DEPENDS *.cpp *.h)

add_library(common ${COMMON_SOURCES})

if (WIN32)
    add_dependencies(common glew)
endif()

#----------------------------------------------------------------------------
# Installation files
#----------------------------------------------------------------------------

file(GLOB FYUSENET_COMMON_HEADERS *.h)
install(FILES ${FYUSENET_COMMON_HEADERS} DESTINATION include/fyusenet/common)

# vim: set expandtab ts=2 sw=2:
