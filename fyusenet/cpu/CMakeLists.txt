file(GLOB CPU_SOURCES CONFIGURE_DEPENDS *.cpp *.h)

add_library(cpu ${CPU_SOURCES})

#----------------------------------------------------------------------------
# Installation files
#----------------------------------------------------------------------------

file(GLOB FYUSENET_CPU_HEADERS *.h)
install(FILES ${FYUSENET_CPU_HEADERS} DESTINATION include/fyusenet/cpu)

# vim: set expandtab ts=2 sw=2:
