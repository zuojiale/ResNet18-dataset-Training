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
CMAKE_SOURCE_DIR = /home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic/build

# Include any dependencies generated for this target.
include CMakeFiles/openCV_savePic.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/openCV_savePic.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/openCV_savePic.dir/flags.make

CMakeFiles/openCV_savePic.dir/src/main.cpp.o: CMakeFiles/openCV_savePic.dir/flags.make
CMakeFiles/openCV_savePic.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/openCV_savePic.dir/src/main.cpp.o"
	/usr/bin/x86_64-linux-gnu-g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/openCV_savePic.dir/src/main.cpp.o -c /home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic/src/main.cpp

CMakeFiles/openCV_savePic.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/openCV_savePic.dir/src/main.cpp.i"
	/usr/bin/x86_64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic/src/main.cpp > CMakeFiles/openCV_savePic.dir/src/main.cpp.i

CMakeFiles/openCV_savePic.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/openCV_savePic.dir/src/main.cpp.s"
	/usr/bin/x86_64-linux-gnu-g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic/src/main.cpp -o CMakeFiles/openCV_savePic.dir/src/main.cpp.s

CMakeFiles/openCV_savePic.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/openCV_savePic.dir/src/main.cpp.o.requires

CMakeFiles/openCV_savePic.dir/src/main.cpp.o.provides: CMakeFiles/openCV_savePic.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/openCV_savePic.dir/build.make CMakeFiles/openCV_savePic.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/openCV_savePic.dir/src/main.cpp.o.provides

CMakeFiles/openCV_savePic.dir/src/main.cpp.o.provides.build: CMakeFiles/openCV_savePic.dir/src/main.cpp.o


# Object files for target openCV_savePic
openCV_savePic_OBJECTS = \
"CMakeFiles/openCV_savePic.dir/src/main.cpp.o"

# External object files for target openCV_savePic
openCV_savePic_EXTERNAL_OBJECTS =

../bin/openCV_savePic: CMakeFiles/openCV_savePic.dir/src/main.cpp.o
../bin/openCV_savePic: CMakeFiles/openCV_savePic.dir/build.make
../bin/openCV_savePic: ../../../lib/libtofsdk.a
../bin/openCV_savePic: /usr/local/lib/libopencv_ml.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_shape.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_videostab.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_objdetect.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_stitching.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_superres.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_calib3d.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_dnn.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_features2d.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_photo.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_highgui.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_flann.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_video.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_videoio.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_imgcodecs.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_imgproc.so.3.4.1
../bin/openCV_savePic: /usr/local/lib/libopencv_core.so.3.4.1
../bin/openCV_savePic: CMakeFiles/openCV_savePic.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/openCV_savePic"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/openCV_savePic.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/openCV_savePic.dir/build: ../bin/openCV_savePic

.PHONY : CMakeFiles/openCV_savePic.dir/build

CMakeFiles/openCV_savePic.dir/requires: CMakeFiles/openCV_savePic.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/openCV_savePic.dir/requires

CMakeFiles/openCV_savePic.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/openCV_savePic.dir/cmake_clean.cmake
.PHONY : CMakeFiles/openCV_savePic.dir/clean

CMakeFiles/openCV_savePic.dir/depend:
	cd /home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic /home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic /home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic/build /home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic/build /home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic/build/CMakeFiles/openCV_savePic.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/openCV_savePic.dir/depend

