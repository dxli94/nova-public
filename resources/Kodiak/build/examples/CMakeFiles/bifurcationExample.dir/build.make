# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dxli/workspace/model-checkers/nova/resources/Kodiak

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dxli/workspace/model-checkers/nova/resources/Kodiak/build

# Include any dependencies generated for this target.
include examples/CMakeFiles/bifurcationExample.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/bifurcationExample.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/bifurcationExample.dir/flags.make

examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o: examples/CMakeFiles/bifurcationExample.dir/flags.make
examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o: ../examples/bifurcation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dxli/workspace/model-checkers/nova/resources/Kodiak/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o"
	cd /home/dxli/workspace/model-checkers/nova/resources/Kodiak/build/examples && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o -c /home/dxli/workspace/model-checkers/nova/resources/Kodiak/examples/bifurcation.cpp

examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bifurcationExample.dir/bifurcation.cpp.i"
	cd /home/dxli/workspace/model-checkers/nova/resources/Kodiak/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dxli/workspace/model-checkers/nova/resources/Kodiak/examples/bifurcation.cpp > CMakeFiles/bifurcationExample.dir/bifurcation.cpp.i

examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bifurcationExample.dir/bifurcation.cpp.s"
	cd /home/dxli/workspace/model-checkers/nova/resources/Kodiak/build/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dxli/workspace/model-checkers/nova/resources/Kodiak/examples/bifurcation.cpp -o CMakeFiles/bifurcationExample.dir/bifurcation.cpp.s

examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o.requires:

.PHONY : examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o.requires

examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o.provides: examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o.requires
	$(MAKE) -f examples/CMakeFiles/bifurcationExample.dir/build.make examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o.provides.build
.PHONY : examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o.provides

examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o.provides.build: examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o


# Object files for target bifurcationExample
bifurcationExample_OBJECTS = \
"CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o"

# External object files for target bifurcationExample
bifurcationExample_EXTERNAL_OBJECTS =

examples/bifurcationExample: examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o
examples/bifurcationExample: examples/CMakeFiles/bifurcationExample.dir/build.make
examples/bifurcationExample: libkodiak.a
examples/bifurcationExample: examples/CMakeFiles/bifurcationExample.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dxli/workspace/model-checkers/nova/resources/Kodiak/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bifurcationExample"
	cd /home/dxli/workspace/model-checkers/nova/resources/Kodiak/build/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bifurcationExample.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/bifurcationExample.dir/build: examples/bifurcationExample

.PHONY : examples/CMakeFiles/bifurcationExample.dir/build

examples/CMakeFiles/bifurcationExample.dir/requires: examples/CMakeFiles/bifurcationExample.dir/bifurcation.cpp.o.requires

.PHONY : examples/CMakeFiles/bifurcationExample.dir/requires

examples/CMakeFiles/bifurcationExample.dir/clean:
	cd /home/dxli/workspace/model-checkers/nova/resources/Kodiak/build/examples && $(CMAKE_COMMAND) -P CMakeFiles/bifurcationExample.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/bifurcationExample.dir/clean

examples/CMakeFiles/bifurcationExample.dir/depend:
	cd /home/dxli/workspace/model-checkers/nova/resources/Kodiak/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dxli/workspace/model-checkers/nova/resources/Kodiak /home/dxli/workspace/model-checkers/nova/resources/Kodiak/examples /home/dxli/workspace/model-checkers/nova/resources/Kodiak/build /home/dxli/workspace/model-checkers/nova/resources/Kodiak/build/examples /home/dxli/workspace/model-checkers/nova/resources/Kodiak/build/examples/CMakeFiles/bifurcationExample.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/bifurcationExample.dir/depend

