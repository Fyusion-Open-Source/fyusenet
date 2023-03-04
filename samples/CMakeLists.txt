include_directories(${CMAKE_SOURCE_DIR})

if (BUILD_TARGET STREQUAL "Desktop")
  add_subdirectory(desktop)
endif()

if (BUILD_TARGET STREQUAL "Web")
  add_subdirectory(web)
endif()

# vim: set expandtab ts=2 sw=2:
