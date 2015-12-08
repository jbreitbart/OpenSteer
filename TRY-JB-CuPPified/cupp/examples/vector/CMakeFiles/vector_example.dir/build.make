# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.6

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canoncical targets will work.
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

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified

# Include any dependencies generated for this target.
include cupp/examples/vector/CMakeFiles/vector_example.dir/depend.make

# Include the progress variables for this target.
include cupp/examples/vector/CMakeFiles/vector_example.dir/progress.make

# Include the compile flags for this target's objects.
include cupp/examples/vector/CMakeFiles/vector_example.dir/flags.make

cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o: cupp/examples/vector/CMakeFiles/vector_example.dir/flags.make
cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o: cupp/examples/vector/vector.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o"
	cd /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector && /usr/bin/g++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/vector_example.dir/vector.o -c /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector/vector.cpp

cupp/examples/vector/CMakeFiles/vector_example.dir/vector.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vector_example.dir/vector.i"
	cd /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector/vector.cpp > CMakeFiles/vector_example.dir/vector.i

cupp/examples/vector/CMakeFiles/vector_example.dir/vector.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vector_example.dir/vector.s"
	cd /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector && /usr/bin/g++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector/vector.cpp -o CMakeFiles/vector_example.dir/vector.s

cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o.requires:
.PHONY : cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o.requires

cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o.provides: cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o.requires
	$(MAKE) -f cupp/examples/vector/CMakeFiles/vector_example.dir/build.make cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o.provides.build
.PHONY : cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o.provides

cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o.provides.build: cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o
.PHONY : cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o.provides.build

# Object files for target vector_example
vector_example_OBJECTS = \
"CMakeFiles/vector_example.dir/vector.o"

# External object files for target vector_example
vector_example_EXTERNAL_OBJECTS =

bin/vector_example: cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o
bin/vector_example: lib/libcupp.so
bin/vector_example: lib/libkernel_vector.a
bin/vector_example: /usr/local/cuda/lib/libcudart.so
bin/vector_example: cupp/examples/vector/CMakeFiles/vector_example.dir/build.make
bin/vector_example: cupp/examples/vector/CMakeFiles/vector_example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../../../bin/vector_example"
	cd /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vector_example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
cupp/examples/vector/CMakeFiles/vector_example.dir/build: bin/vector_example
.PHONY : cupp/examples/vector/CMakeFiles/vector_example.dir/build

cupp/examples/vector/CMakeFiles/vector_example.dir/requires: cupp/examples/vector/CMakeFiles/vector_example.dir/vector.o.requires
.PHONY : cupp/examples/vector/CMakeFiles/vector_example.dir/requires

cupp/examples/vector/CMakeFiles/vector_example.dir/clean:
	cd /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector && $(CMAKE_COMMAND) -P CMakeFiles/vector_example.dir/cmake_clean.cmake
.PHONY : cupp/examples/vector/CMakeFiles/vector_example.dir/clean

cupp/examples/vector/CMakeFiles/vector_example.dir/depend:
	cd /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector /home/jbreitbart/projekte/opensteer/svn/branches/TRY-JB-CuPPified/cupp/examples/vector/CMakeFiles/vector_example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : cupp/examples/vector/CMakeFiles/vector_example.dir/depend

