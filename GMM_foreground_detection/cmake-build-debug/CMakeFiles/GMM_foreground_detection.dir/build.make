# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/lansy/CLionProjects/GMM_foreground_detection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/lansy/CLionProjects/GMM_foreground_detection/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/GMM_foreground_detection.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/GMM_foreground_detection.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GMM_foreground_detection.dir/flags.make

CMakeFiles/GMM_foreground_detection.dir/main.cpp.o: CMakeFiles/GMM_foreground_detection.dir/flags.make
CMakeFiles/GMM_foreground_detection.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lansy/CLionProjects/GMM_foreground_detection/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GMM_foreground_detection.dir/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GMM_foreground_detection.dir/main.cpp.o -c /Users/lansy/CLionProjects/GMM_foreground_detection/main.cpp

CMakeFiles/GMM_foreground_detection.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GMM_foreground_detection.dir/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lansy/CLionProjects/GMM_foreground_detection/main.cpp > CMakeFiles/GMM_foreground_detection.dir/main.cpp.i

CMakeFiles/GMM_foreground_detection.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GMM_foreground_detection.dir/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lansy/CLionProjects/GMM_foreground_detection/main.cpp -o CMakeFiles/GMM_foreground_detection.dir/main.cpp.s

CMakeFiles/GMM_foreground_detection.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/GMM_foreground_detection.dir/main.cpp.o.requires

CMakeFiles/GMM_foreground_detection.dir/main.cpp.o.provides: CMakeFiles/GMM_foreground_detection.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/GMM_foreground_detection.dir/build.make CMakeFiles/GMM_foreground_detection.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/GMM_foreground_detection.dir/main.cpp.o.provides

CMakeFiles/GMM_foreground_detection.dir/main.cpp.o.provides.build: CMakeFiles/GMM_foreground_detection.dir/main.cpp.o


CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o: CMakeFiles/GMM_foreground_detection.dir/flags.make
CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o: ../GMM.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lansy/CLionProjects/GMM_foreground_detection/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o -c /Users/lansy/CLionProjects/GMM_foreground_detection/GMM.cpp

CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lansy/CLionProjects/GMM_foreground_detection/GMM.cpp > CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.i

CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lansy/CLionProjects/GMM_foreground_detection/GMM.cpp -o CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.s

CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o.requires:

.PHONY : CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o.requires

CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o.provides: CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o.requires
	$(MAKE) -f CMakeFiles/GMM_foreground_detection.dir/build.make CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o.provides.build
.PHONY : CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o.provides

CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o.provides.build: CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o


# Object files for target GMM_foreground_detection
GMM_foreground_detection_OBJECTS = \
"CMakeFiles/GMM_foreground_detection.dir/main.cpp.o" \
"CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o"

# External object files for target GMM_foreground_detection
GMM_foreground_detection_EXTERNAL_OBJECTS =

GMM_foreground_detection: CMakeFiles/GMM_foreground_detection.dir/main.cpp.o
GMM_foreground_detection: CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o
GMM_foreground_detection: CMakeFiles/GMM_foreground_detection.dir/build.make
GMM_foreground_detection: /usr/local/lib/libopencv_gapi.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_stitching.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_aruco.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_bgsegm.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_bioinspired.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_ccalib.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_dnn_objdetect.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_dpm.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_face.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_fuzzy.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_hfs.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_img_hash.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_line_descriptor.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_reg.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_rgbd.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_saliency.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_stereo.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_structured_light.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_superres.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_surface_matching.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_tracking.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_videostab.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_xfeatures2d.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_xobjdetect.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_xphoto.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_shape.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_phase_unwrapping.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_optflow.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_ximgproc.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_dnn.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_datasets.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_ml.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_plot.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_video.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_objdetect.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_calib3d.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_features2d.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_flann.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_highgui.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_videoio.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_imgcodecs.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_photo.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_imgproc.4.0.1.dylib
GMM_foreground_detection: /usr/local/lib/libopencv_core.4.0.1.dylib
GMM_foreground_detection: CMakeFiles/GMM_foreground_detection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/lansy/CLionProjects/GMM_foreground_detection/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable GMM_foreground_detection"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GMM_foreground_detection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GMM_foreground_detection.dir/build: GMM_foreground_detection

.PHONY : CMakeFiles/GMM_foreground_detection.dir/build

CMakeFiles/GMM_foreground_detection.dir/requires: CMakeFiles/GMM_foreground_detection.dir/main.cpp.o.requires
CMakeFiles/GMM_foreground_detection.dir/requires: CMakeFiles/GMM_foreground_detection.dir/GMM.cpp.o.requires

.PHONY : CMakeFiles/GMM_foreground_detection.dir/requires

CMakeFiles/GMM_foreground_detection.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GMM_foreground_detection.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GMM_foreground_detection.dir/clean

CMakeFiles/GMM_foreground_detection.dir/depend:
	cd /Users/lansy/CLionProjects/GMM_foreground_detection/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/lansy/CLionProjects/GMM_foreground_detection /Users/lansy/CLionProjects/GMM_foreground_detection /Users/lansy/CLionProjects/GMM_foreground_detection/cmake-build-debug /Users/lansy/CLionProjects/GMM_foreground_detection/cmake-build-debug /Users/lansy/CLionProjects/GMM_foreground_detection/cmake-build-debug/CMakeFiles/GMM_foreground_detection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/GMM_foreground_detection.dir/depend
