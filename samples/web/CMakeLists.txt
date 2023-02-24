#----------------------------------------------------------------------------------
# Misc compiler flags and default link libraries (esp. when using ninja as backend)
#----------------------------------------------------------------------------------

set(DEFAULT_LIBS stdc++ pthread m)

set_source_files_properties(${SHADERMETA} PROPERTIES GENERATED 1)


#----------------------------------------------------------------------------------
# Add sources..
#----------------------------------------------------------------------------------

configure_file(stylenet.html ${CMAKE_CURRENT_BINARY_DIR}/stylenet.html COPYONLY)
configure_file(webcam.js ${CMAKE_CURRENT_BINARY_DIR}/webcam.js COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/stylenet3x3_112_v3.dat ${CMAKE_CURRENT_BINARY_DIR}/stylenet.dat COPYONLY)

set(SOURCES stylenet.cpp ../samplenetworks/stylenet_base.cpp ../samplenetworks/stylenet_base.h
            ../samplenetworks/stylenet3x3.cpp ../samplenetworks/stylenet3x3.h)

#----------------------------------------------------------------------------------
# Set FyuseNet libraries...
#----------------------------------------------------------------------------------

set(FYUSENET_LIBS $<TARGET_OBJECTS:cpu> $<TARGET_OBJECTS:gpu>
                  $<TARGET_OBJECTS:opengl> $<TARGET_OBJECTS:base>
                  $<TARGET_OBJECTS:common>)

set(FYUSENET_LIB ${CMAKE_BINARY_DIR}/fyusenet/libfyusenet.a)

#----------------------------------------------------------------------------------
# Sample executable
#----------------------------------------------------------------------------------

add_executable(stylenet ${SOURCES} ${SHADERMETA} ${SHADERRSRC})
add_dependencies(stylenet shader-meta)
target_link_libraries(stylenet ${FYUSENET_LIB})


# vim: set expandtab ts=2 sw=2:
