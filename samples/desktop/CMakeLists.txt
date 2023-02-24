#----------------------------------------------------------------------------------
# Required packages for the desktop samples
#----------------------------------------------------------------------------------

find_package(JPEG REQUIRED)

#----------------------------------------------------------------------------------
# Misc compiler flags and default link libraries (esp. when using ninja as backend)
#----------------------------------------------------------------------------------

set(DEFAULT_LIBS stdc++ pthread m)

set_source_files_properties(${SHADERMETA} PROPERTIES GENERATED 1)

include_directories(${JPEG_INCLUDE_DIR})

#----------------------------------------------------------------------------------
# Add sources..
#----------------------------------------------------------------------------------

set(SOURCES stylenet.cpp ../samplenetworks/stylenet_base.cpp ../samplenetworks/stylenet_base.h
            ../samplenetworks/stylenet3x3.cpp ../samplenetworks/stylenet3x3.h
            ../samplenetworks/stylenet9x9.cpp ../samplenetworks/stylenet9x9.h
            ../helpers/jpegio.cpp ../helpers/jpegio.h)

#----------------------------------------------------------------------------------
# Set FyuseNet libraries...
#----------------------------------------------------------------------------------

set(FYUSENET_LIBS $<TARGET_OBJECTS:cpu> $<TARGET_OBJECTS:gpu>
                  $<TARGET_OBJECTS:opengl> $<TARGET_OBJECTS:base>
                  $<TARGET_OBJECTS:common>)

#----------------------------------------------------------------------------------
# Actual test executables
#----------------------------------------------------------------------------------

add_executable(stylenet ${SOURCES} ${SHADERMETA} ${SHADERRSRC})
add_dependencies(stylenet shader-meta)

if (APPLE)
  # apple stuff
else(APPLE)
  if (USE_GLFW)
    set(GL_SYS_DEPS glfw X11)
  elseif (USE_EGL)
    # EGL stuff
    set(GL_SYS_DEPS EGL GLESv2)
  else()
    # linux desktop GL
    set(GL_SYS_DEPS X11)
  endif()
endif()

target_link_libraries(stylenet ${FYUSENET_LIBS} ${DEFAULT_LIBS} ${OPENGL_LIBRARIES} ${JPEG_LIBRARIES} ${GL_SYS_DEPS})


# vim: set expandtab ts=2 sw=2:
