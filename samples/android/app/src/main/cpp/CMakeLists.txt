
# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Declares and names the project.

project("styletransfer")

# for this embedded sample we do not use an installed build, this is just a demo so we can live
# with this hack
add_compile_definitions(FYUSENET_INTERNAL FYUSENET_USE_EGL FYUSENET_MULTITHREADING)

include_directories(BEFORE ${CMAKE_SOURCE_DIR}/../../../../../../
                    ${CMAKE_SOURCE_DIR}/../../../../../)

set(SAMPLENET_DIR ${CMAKE_SOURCE_DIR}/../../../../../samplenetworks)
set(HELPER_DIR ${CMAKE_SOURCE_DIR}/../../../../../helpers)

# Create shared library to be used from JNI
add_library(styletransfer SHARED styletransfer.cpp ${SAMPLENET_DIR}/stylenet9x9.cpp ${SAMPLENET_DIR}/stylenet3x3.cpp ${SAMPLENET_DIR}/stylenet_base.cpp
	    ${HELPER_DIR}/stylenet_provider.cpp)

find_library(log-lib log)

# Add FyuseNet library as dependency to the styletransfer module
add_library(fyusenet-lib SHARED IMPORTED)
set_target_properties(fyusenet-lib PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libfyusenet.so)
target_link_libraries(styletransfer ${log-lib} fyusenet-lib)
