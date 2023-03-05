cmake_minimum_required(VERSION 3.20)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.11.0")
  cmake_policy(SET CMP0072 NEW) # Prefer GLVND over GL Legacy
endif()

project(fyusenet LANGUAGES CXX)

set(BUILD_TARGET "Desktop")
set(MT_DEFAULT ON)

if (${CMAKE_SYSTEM_NAME} MATCHES "Emscripten")
  set(BUILD_TARGET "Web")
  set(MT_DEFAULT OFF)
endif()

if (APPLE)
  set(MT_DEFAULT OFF)
endif()

option(BUILD_LIBRARY "Build shared libraries for SDK use" OFF)
option(BUILD_TESTS "Build unit-tests" ON)
option(BUILD_SAMPLES "Build sample networks" ON)
option(USE_MULTITHREADING "Enable multi-threading" ${MT_DEFAULT})
option(USE_EGL "Use embedded GL" OFF)
option(BUILD_DOCS "Build doxygen documentation" OFF)
option(HIGH_PRECISION "Experimental 32-bit FP computation" OFF)

if (ANDROID_ABI)
  set(BUILD_TARGET "Android")
endif()

if (BUILD_TARGET STREQUAL "Android")
  set(USE_EGL ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if (BUILD_TARGET STREQUAL "Desktop")
  option(USE_GLFW "Use GLFW (use this for using NVIDIA Insight for example)" OFF)
endif()

if (BUILD_TARGET STREQUAL "Desktop")
  find_package(OpenGL REQUIRED)
endif()

if (APPLE OR BUILD_TARGET STREQUAL "Web")
  # Force multithreading off for Apple platforms and WebGL platforms (for now)
  set(USE_MULTITHREADING OFF)
endif()

if (NOT USE_MULTITHREADING)
  message(STATUS "Disabling multithreading")
endif()

if (BUILD_LIBRARY)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# C++ standard to use
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags, general C++ compiler
set(CMAKE_CXX_FLAGS_DEBUG "-Wall  -O0 -g -DDEBUG -ffast-math -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -fPIC")

# Compiler and linker flags for WebAssembly / WebGL
set(WASM_GEN_LINKER_FLAGS "-sWASM=1 -sUSE_WEBGL2=1 -sMAX_WEBGL_VERSION=2 -sALLOW_MEMORY_GROWTH -sEXPORTED_RUNTIME_METHODS=\"stringToUTF8,lengthBytesUTF8\"")

set(CMAKE_CXX_FLAGS_EMSCRIPTEN_DEBUG "-Wall -gsource-map -flto -ffast-math -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_EMSCRIPTEN_RELEASE "-ffast-math -flto -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_EMSCRIPTEN_SMALL "-ffast-math -flto -Oz -DNDEBUG")
set(CMAKE_CXX_FLAGS_EMSCRIPTEN_PROFILE "-g --profiling -ffast-math -flto -O3 -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS_EMSCRIPTEN_DEBUG "${CMAKE_CXX_FLAGS_EMSCRIPTEN_DEBUG} ${WASM_GEN_LINKER_FLAGS} -s ASSERTIONS=1 -s FILESYSTEM=0 -s NO_DISABLE_EXCEPTION_CATCHING --source-map-base http://localhost:8000/")
set(CMAKE_EXE_LINKER_FLAGS_EMSCRIPTEN_RELEASE "${CMAKE_CXX_FLAGS_EMSCRIPTEN_RELEASE} ${WASM_GEN_LINKER_FLAGS} -s ASSERTIONS=0 -s FILESYSTEM=0")
set(CMAKE_EXE_LINKER_FLAGS_EMSCRIPTEN_SMALL "${CMAKE_CXX_FLAGS_EMSCRIPTEN_SMALL} ${WASM_GEN_LINKER_FLAGS} -s ASSERTIONS=0 -s FILESYSTEM=0")
set(CMAKE_EXE_LINKER_FLAGS_EMSCRIPTEN_PROFILE "${CMAKE_CXX_FLAGS_EMSCRIPTEN_PROFILE} ${WASM_GEN_LINKER_FLAGS} -s ASSERTIONS=0 -s FILESYSTEM=0")


add_compile_definitions(FYUSENET_INTERNAL FYUSENET_CPU_FLOAT_ONLY)

if (BUILD_TESTS)
  if (NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    message(WARNING "Explicitly setting the DEBUG preprocessor definition for unit-testing.\
                     You should not use the results of this build as release binaries.\
                     Disable the unit tests for release builds")
    add_compile_definitions(DEBUG)
  endif()
endif()

if (HIGH_PRECISION)
  add_compile_definitions(HIGH_PRECISION)
endif()

if (APPLE)
  add_compile_definitions(GL_SILENCE_DEPRECATION)
endif()

if (NOT USE_EGL AND NOT APPLE)
  add_compile_definitions(GL_GLEXT_PROTOTYPES)
endif()

if (USE_GLFW)
  add_compile_definitions(FYUSENET_USE_GLFW)
endif()

if (USE_EGL)
  add_compile_definitions(FYUSENET_USE_EGL)
endif()

if (BUILD_TARGET STREQUAL "Web")
  add_compile_definitions(FYUSENET_USE_WEBGL)
endif()

if (USE_MULTITHREADING)
  add_compile_definitions(FYUSENET_MULTITHREADING)
endif()

add_subdirectory(fyusenet)

if (BUILD_TESTS AND NOT (BUILD_TARGET STREQUAL "Web"))
  add_subdirectory(unit_tests)
endif()

if (BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

if (BUILD_DOCS)
  add_subdirectory(doxygen)
endif()

