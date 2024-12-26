###########################################################################
# CMakeLists.txt generated for component CR3BP
# Product type: STATIC library
###########################################################################
cmake_minimum_required(VERSION 3.12)
project(CR3BP)

# Propagate the CMAKE_EXPORT_COMPILE_COMMANDS variable from the 
# environment if it is defined as an environment variable, but not as a 
# CMake variable. This is to work around a bug in CMake 3.19 when the 
# "NMake Makefiles" generator is selected.
if(DEFINED ENV{CMAKE_EXPORT_COMPILE_COMMANDS} AND NOT DEFINED CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS $ENV{CMAKE_EXPORT_COMPILE_COMMANDS})
endif()


###########################################################################
## Path variables
###########################################################################
# Derive an absolute path to the code generation anchor folder.
get_filename_component(START_DIR ../../.. ABSOLUTE)

# Special directories defined by using CACHE variables can be overridden 
# by setting the variable from the command line, e.g.,
# 
# 	cmake . -DMATLAB_ROOT=/path/to/another/matlab/root
set(MATLAB_ROOT /Applications/MATLAB_R2024a.app CACHE PATH "")

# Additional variables that are defined conditionally.
if("${CMAKE_CURRENT_BINARY_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    set(BINARY_START_DIR "${START_DIR}")
else()
    set(BINARY_START_DIR "${CMAKE_BINARY_DIR}")
endif()

###########################################################################
## System Libraries
###########################################################################
find_library(FOUND_LIBM m NO_SYSTEM_ENVIRONMENT_PATH PATHS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES} ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})

###########################################################################
## Target definition and commands
###########################################################################

# Definition of target "CR3BP".
add_library(CR3BP STATIC ${START_DIR}/codegen/lib/CR3BP/coder_posix_time.c
    ${START_DIR}/codegen/lib/CR3BP/CR3BP_data.c
    ${START_DIR}/codegen/lib/CR3BP/CR3BP_initialize.c
    ${START_DIR}/codegen/lib/CR3BP/CR3BP_terminate.c
    ${START_DIR}/codegen/lib/CR3BP/CR3BP.c
    ${START_DIR}/codegen/lib/CR3BP/tic.c
    ${START_DIR}/codegen/lib/CR3BP/timeKeeper.c
    ${START_DIR}/codegen/lib/CR3BP/toc.c
    ${START_DIR}/codegen/lib/CR3BP/CoderTimeAPI.c)

# Set properties for target "CR3BP".
set_target_properties(CR3BP PROPERTIES PREFIX ""
    POSITION_INDEPENDENT_CODE ON
    RUNTIME_OUTPUT_DIRECTORY "${BINARY_START_DIR}/codegen/lib/CR3BP/$<0:>"
    LIBRARY_OUTPUT_DIRECTORY "${BINARY_START_DIR}/codegen/lib/CR3BP/$<0:>"
    ARCHIVE_OUTPUT_DIRECTORY "${BINARY_START_DIR}/codegen/lib/CR3BP/$<0:>")

# Specify language features required for target "CR3BP".
target_compile_features(CR3BP PUBLIC c_std_99)

# Specify compiler preprocessor definitions for target "CR3BP".
target_compile_definitions(CR3BP PRIVATE -DMODEL=CR3BP)

# Specify include directories for target "CR3BP".
target_include_directories(CR3BP PUBLIC $<BUILD_INTERFACE:${START_DIR}/codegen/lib/CR3BP>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/codegen/lib/CR3BP>
    $<BUILD_INTERFACE:${START_DIR}>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>>
    $<BUILD_INTERFACE:${MATLAB_ROOT}/extern/include>)

# Specify library link dependencies for target "CR3BP". CMake generator 
# expressions are used to create a CMakeLists.txt file that supports 
# multiple platforms with differently named system library dependencies.
target_link_libraries(CR3BP PUBLIC $<$<BOOL:${FOUND_LIBM}>:m>)


###########################################################################
## Build success message
###########################################################################
add_custom_command(TARGET CR3BP POST_BUILD
     COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "\\#\\#\\# Created STATIC library: $<TARGET_FILE:CR3BP>")

###########################################################################
## Call toolchain hook function if defined
###########################################################################
if(COMMAND toolchain_target_hook)
    toolchain_target_hook(CR3BP)
endif()
