file(GLOB BASE_SOURCES CONFIGURE_DEPENDS *.cpp *.h)

add_library(base ${BASE_SOURCES})

#----------------------------------------------------------------------------
# Installation files
#----------------------------------------------------------------------------

file(GLOB FYUSENET_BASE_HEADERS *.h)
install(FILES ${FYUSENET_BASE_HEADERS} DESTINATION include/fyusenet/base)

# vim: set expandtab ts=2 sw=2:
