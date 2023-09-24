#----------------------------------------------------------------------------------
# Misc compiler flags and default link libraries (esp. when using ninja as backend)
#----------------------------------------------------------------------------------

set(DEFAULT_LIBS stdc++ pthread m)

set_source_files_properties(${SHADERMETA} PROPERTIES GENERATED 1)


#----------------------------------------------------------------------------------
# Add sources..
#----------------------------------------------------------------------------------

configure_file(stylenet.html ${CMAKE_CURRENT_BINARY_DIR}/stylenet.html COPYONLY)
configure_file(resnet.html ${CMAKE_CURRENT_BINARY_DIR}/resnet.html COPYONLY)
configure_file(webcam.js ${CMAKE_CURRENT_BINARY_DIR}/webcam.js COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/stylenet3x3_112_v3.dat ${CMAKE_CURRENT_BINARY_DIR}/stylenet.dat COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/resnet_50.dat ${CMAKE_CURRENT_BINARY_DIR}/resnet_50.dat COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/data/kitten.jpg ${CMAKE_CURRENT_BINARY_DIR}/kitten.jpg COPYONLY)

set(STYLENET_SOURCES stylenet.cpp ../samplenetworks/stylenet_base.cpp ../samplenetworks/stylenet_base.h
            ../samplenetworks/stylenet3x3.cpp ../samplenetworks/stylenet3x3.h ../helpers/stylenet_provider.cpp)

set(RESNET_SOURCES resnet.cpp ../samplenetworks/resnet50.cpp ../helpers/resnet_provider.cpp)

#----------------------------------------------------------------------------------
# Sample executables
#----------------------------------------------------------------------------------

add_executable(stylenet ${STYLENET_SOURCES} ${SHADERMETA} ${SHADERRSRC})
add_dependencies(stylenet shader-meta)
add_dependencies(stylenet fyusenet)
target_link_libraries(stylenet ${FYUSENET_LIB})

add_executable(resnet ${RESNET_SOURCES} ${SHADERMETA} ${SHADERRSRC})
add_dependencies(resnet shader-meta)
add_dependencies(resnet fyusenet)
target_link_libraries(resnet ${FYUSENET_LIB})

# vim: set expandtab ts=2 sw=2:
