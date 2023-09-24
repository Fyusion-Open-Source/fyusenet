#----------------------------------------------------------------------------
# Process embedded shaders by our own shader pre-compiler
#----------------------------------------------------------------------------

if (WIN32)
  set(SHADERPP ${Python_EXECUTABLE})
  set(SHADERPP_FLAGS ${CMAKE_SOURCE_DIR}/buildutils/shaderpp "-i fyusenet/gl" "-n fyusion::fyusenet::gpu" "-p ${CMAKE_CURRENT_SOURCE_DIR}/")
else()
  set(SHADERPP ${CMAKE_SOURCE_DIR}/buildutils/shaderpp)
  set(SHADERPP_FLAGS "-i fyusenet/gl" "-n fyusion::fyusenet::gpu" "-p ${CMAKE_CURRENT_SOURCE_DIR}/")
endif()

if (USE_CUSTOM)
  file(GLOB FRAGSHADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} shaders/*.frag shaders/vanilla/*.frag shaders/deep/*.frag shaders/deep/*.frag shaders/custom/*.frag shaders/sequence/*.frag shaders/custom/sequence/*.frag)
  file(GLOB VERTSHADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} shaders/*.vert shaders/vanilla/*.vert shaders/deep/*.vert shaders/deep/*.vert shaders/custom/*.vert shaders/sequence/*.vert shaders/custom/sequence/*.vert)
  file(GLOB SHADERSNIPS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} shaders/*.inc shaders/vanilla/*.inc shaders/deep/*.inc shaders/deep/*.inc shaders/custom/*.inc shaders/sequence/*.inc shaders/custom/sequence/*.inc)
else()
  file(GLOB FRAGSHADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} shaders/*.frag shaders/vanilla/*.frag shaders/deep/*.frag shaders/deep/*.frag shaders/sequence/*.frag)
  file(GLOB VERTSHADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} shaders/*.vert shaders/vanilla/*.vert shaders/deep/*.vert shaders/deep/*.vert shaders/sequence/*.vert)
  file(GLOB SHADERSNIPS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} shaders/*.inc shaders/vanilla/*.inc shaders/deep/*.inc shaders/deep/*.inc shaders/sequence/*.inc)
endif()

foreach(name ${FRAGSHADERS})
  string(REPLACE ".frag" "_frag.cpp" outfile ${name})
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${outfile} COMMAND ${SHADERPP} ARGS ${SHADERPP_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${name} ${CMAKE_CURRENT_BINARY_DIR}/${outfile} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${name})
  list(APPEND FRAGMETA ${CMAKE_CURRENT_BINARY_DIR}/${outfile})
endforeach(name)

foreach(name ${VERTSHADERS})
  string(REPLACE ".vert" "_vert.cpp" outfile ${name})
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${outfile} COMMAND ${SHADERPP} ARGS ${SHADERPP_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${name} ${CMAKE_CURRENT_BINARY_DIR}/${outfile} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${name})
  list(APPEND VERTMETA ${CMAKE_CURRENT_BINARY_DIR}/${outfile})
endforeach(name)

foreach(name ${SHADERSNIPS})
  string(REPLACE ".inc" "_inc.cpp" outfile ${name})
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${outfile} COMMAND ${SHADERPP} ARGS ${SHADERPP_FLAGS} ${CMAKE_CURRENT_SOURCE_DIR}/${name} ${CMAKE_CURRENT_BINARY_DIR}/${outfile} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${name})
  list(APPEND SNIPMETA ${CMAKE_CURRENT_BINARY_DIR}/${outfile})
endforeach(name)

set(SHADERMETA ${FRAGMETA} ${VERTMETA} ${COMPMETA} ${SNIPMETA} PARENT_SCOPE)
set(SHADERMETA ${FRAGMETA} ${VERTMETA} ${COMPMETA} ${SNIPMETA})
set_source_files_properties(${SHADERMETA} PROPERTIES GENERATED 1)

add_custom_target(shader-meta DEPENDS ${SHADERMETA})
add_custom_target(clear-shader-meta COMMAND cd ${CMAKE_BINARY_DIR} ; rm ${SHADERMETA})
add_custom_target(shader-sources ALL SOURCES ${VERTSHADERS} ${FRAGSHADERS} ${SHADERSNIPS})

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/fyusenet/gpu/shaders)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/fyusenet/gpu/shaders/vanilla)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/fyusenet/gpu/shaders/deep)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/fyusenet/gpu/shaders/sequence)
if (USE_CUSTOM)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/fyusenet/gpu/shaders/custom)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/fyusenet/gpu/shaders/custom/sequence)
endif()

#----------------------------------------------------------------------------
# Remaining sources
#----------------------------------------------------------------------------

file(GLOB GPU_BASE_SOURCES CONFIGURE_DEPENDS *.cpp *.h rudiments/*.cpp rudiments/*.h)
file(GLOB GPU_DEEP_SOURCES CONFIGURE_DEPENDS deep/*.cpp deep/*.h)
file(GLOB GPU_VANILLA_SOURCES CONFIGURE_DEPENDS vanilla/*.cpp vanilla/*.h)
file(GLOB GPU_SEQUENCE_SOURCES CONFIGURE_DEPENDS sequence/*.cpp sequence/*.h sequence/rudiments/*.cpp sequence/rudiments/*.h)
if (USE_CUSTOM)
  file(GLOB GPU_CUSTOM_SOURCES CONFIGURE_DEPENDS custom/*.cpp custom/*.h custom/sequence/*.cpp custom/sequence/*.h)
endif()


if (NOT USE_EGL)
    list(REMOVE_ITEM GPU_BASE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/oesconverter.cpp)
    list(REMOVE_ITEM GPU_BASE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/oesconverter.h)
endif()

add_library(gpu STATIC ${GPU_BASE_SOURCES} ${GPU_DEEP_SOURCES} ${GPU_VANILLA_SOURCES}
                       ${GPU_CUSTOM_SOURCES} ${GPU_SEQUENCE_SOURCES})
if (WIN32)
  add_dependencies(gpu glew)
endif()

#----------------------------------------------------------------------------
# Installation files
#----------------------------------------------------------------------------

file(GLOB FYUSENET_GPU_HEADERS *.h)
file(GLOB FYUSENET_GPU_RUDIMENTS *.h)
file(GLOB FYUSENET_GPU_DEEP_HEADERS deep/*.h)
file(GLOB FYUSENET_GPU_VANILLA_HEADERS vanilla/*.h)
file(GLOB FYUSENET_GPU_CUSTOM_HEADERS custom/*.h)
file(GLOB FYUSENET_GPU_SEQUENCE_HEADERS sequence/*.h)
file(GLOB FYUSENET_GPU_SEQUENCE_RUDIMENTS sequence/rudiments/*.h)
install(FILES ${FYUSENET_GPU_HEADERS} DESTINATION include/fyusenet/gpu)
install(FILES ${FYUSENET_GPU_RUDIMENTS} DESTINATION include/fyusenet/gpu/rudiments)
install(FILES ${FYUSENET_GPU_DEEP_HEADERS} DESTINATION include/fyusenet/gpu/deep)
install(FILES ${FYUSENET_GPU_VANILLA_HEADERS} DESTINATION include/fyusenet/gpu/vanilla)
install(FILES ${FYUSENET_GPU_SEQUENCE_HEADERS} DESTINATION include/fyusenet/gpu/sequence)
install(FILES ${FYUSENET_GPU_SEQUENCE_RUDIMENTS} DESTINATION include/fyusenet/gpu/sequence)
if (USE_CUSTOM)
  install(FILES ${FYUSENET_GPU_CUSTOM_HEADERS} DESTINATION include/fyusenet/gpu/custom)
endif()

# vim: set expandtab ts=2 sw=2:
