find_package(Doxygen REQUIRED dot)

if (NOT DOXYGEN_FOUND)
    message(FATAL_ERROR "Doxygen/dot not installed.")
endif()

set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_MAN NO)

add_custom_target(doxygen_dir COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/../docs)

add_custom_target(doc_doxygen_api ALL
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)

add_dependencies(doc_doxygen_api doxygen_dir)
