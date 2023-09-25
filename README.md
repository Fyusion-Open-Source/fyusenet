# FyuseNet Neural Network Inference Library
FyuseNet is an OpenGL(ES) based library that allows to run neural network inference on GPUs that support
[OpenGL](https://khronos.org/opengl) or [OpenGL/ES](https://khronos.org/opengles), which is the case for most
desktop and mobile GPUs on the market.
The library was written in portable C++ and runs on a variety of desktop, edge and mobile platforms. The 
original target of this library were Android-based smartphones.

## What FyuseNet is Not
FyuseNet is not a replacement for [PyTorch](https://pytorch.org) or [Tensorflow](https://tensorflow.org), it is 
limited to perform inference only and cannot be used to actually _train_ neural networks. It can be compared 
to vendor-specific systems like [TensorRT](https://developer.nvidia.com/tensorrt) from NVIDIA. It adds the
benefit that it can actually run on a wider variety of GPUs, as it tries to be vendor-agnostic. This approach
however bears the drawback that it does not have the same set of capabilities and will also perform slower than
a vendor-specific solution.

## History
FyuseNet was initially developed at Fyusion Inc. at the end of 2016 as a proof-of-concept for running neural networks
on Android smartphones. The initial versions were running on [OpenGL/ES 2.0](https://registry.khronos.org/OpenGL-Refpages/es2.0/)
and over time it has migrated to
[OpenGL/ES 3.0](https://registry.khronos.org/OpenGL-Refpages/es3.0/). FyuseNet started out with rather simple 
networks for style-transfer and since then continued
as a small side-project at Fyusion that was used for translating a variety of different networks to Android
phones, the largest of these had around 200 layers. Whenever demand for new functionality came up, the library was 
expanded by the required layer types, which is reflected in the way it looks like as of today and also explains the subset
of different layers that are supported by it. 

In 2017 an early version of FyuseNet made it into the firmware
of a major smartphone manufacturer as part of the stock camera app. It has also been used to generate FX
for a [music video](https://www.youtube.com/watch?v=pLqVDCXiwGY&ab_channel=Jiox). Other
than that, FyuseNet has only been used internally and is part of a set of Android apps that Fyusion maintains. 
The code has not been significantly changed since 2019 and the version in this repository is a bit stripped down from the internal 
version, which contained shader code that was specifically optimized for ARM Mali GPUs prior to the G-series
(T-880 for example) as well as support for a proprietary NPU. For the public release we chose to exclude that
code for various reasons. 

## License
FyuseNet is published under the [MIT](https://en.wikipedia.org/wiki/MIT_License), see the LICENSE file in this repository
for details.

## General Approach
In contrast to most of the popular machine-learning systems, FyuseNet uses a _layer centric_ approach instead
of a _tensor centric_ approach, as this is more fitting for the GL-based shader architecture. Due to the initial
design having to support OpenGL/ES 2.0, FyuseNet does not use compute shaders and performs all operations using
[vertex-](https://www.khronos.org/opengl/wiki/Vertex_Shader) and [fragment](https://www.khronos.org/opengl/wiki/Fragment_Shader)
shaders instead. The layer-centric approach has the drawback that every type of operation
must be coded into layers, which consists of a bit of C++ code and associated [GLSL](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language)
shader code. It is therefore not as flexible as a tensor centric system that executes (elementary) operations on the tensors
in case there is no specific implementation available for the operation at hand and also offer more flexibility on indexing and reshuffling.

In order to deliver the performance required to run (some) networks in real-time while not consuming too much VRAM and memory bandwidth,
a number of tweaks have been integrated into the library. The most important one being the ability to _fuse_ operations in a single 
layer/shader. For example, when executing a convolution on a feature map, followed by an activation, this would normally require
two or more passes: one set of passes for the convolution and another pass to perform the (non-linear) activation.
In order to avoid that, FyuseNet moves the activation step of one layer to the data-fetch step in the next layer,
resulting in a fused activation/operation step in the next layer. Considering that the arithmetic
intensity in most NN operations is rather low compared to the data-fetch and usually does not exhaust the arithmetic
capacity of the GPU, the added overhead of performing the
activation multiple times on the input is far less than the overhead of having this done in a split operation at
the expense of memory bandwidth. 

A second trick that FyuseNet employs - in particular for convolution operations - is to make use of the
[raster operation processors](https://fgiesen.wordpress.com/2011/07/12/a-trip-through-the-graphics-pipeline-2011-part-9/)
of the GPU. Keep in mind that convolution operations include an accumulation step that
spans over all channels of a feature-map, which can be a lot. As it is hard/impossible to perform the accumulation
in a single rendering step using fragment shaders using the chosen data layout, we use the built-in alpha-blending capability of the 
raster processors to perform the accumulation for us. This has the added benefit of getting some arithmetic
operations essentially for free, as it does not change execution time within the shader.

The trained observer will notice that FyuseNet does not use the usual [im2col](https://towardsdatascience.com/how-are-convolutions-actually-performed-under-the-hood-226523ce7fbf)
approach for convolutions, which we opted against for several reasons. The most important reason was that many of our early
networks had quite wide convolutions and the additional memory overhead posed a problem on the smartphone hardware back in
2016. A drawback of that particular approach is, that the batch size is currently always limited to 1. 
However, as the main use-case for FyuseNet was to use it in real-time scenarios on camera streams from smartphones, this is an 
acceptable compromise.
Last but not least, to further conserve VRAM, FyuseNet re-uses textures whenever possible for intermediary/temporary buffers
along the computational chain.

FyuseNet is a comparably lightweight library. The runtime
has no notable external dependency aside from OpenGL. Technically, the same library binary can be used
to run a variety of networks, only the network-specific frontend parts along with the weight data are changed on
different nets. It can also run on a variety of target architectures, including edge computing devices that use embedded
GPUs (ARM, Qualcomm, etc).

# Building

## Folder Structure

```
fyusenet
   |-- buildutils           (Folder with helper scripts for the build process)
   |-- data                 (Folder with sample network weights and sample images)
   |-- fyusenet             (Folder that contains the main library source code, including shaders)
   |-- samples              (Folder with sample code for various platforms)
   |-- doxygen              (Doxygen documentation)
   |-- unit_tests           (Folder with unit tests)
   |-- templates            (Templates for cpp and h files)
   |-- CMakeLists.txt       (Root build file)
   |-- LICENSE              (Software license, MIT in our case)
   |-- CONTRIBUTING.md      (Hints and rules for contributing to FyuseNet, we encourage to do so)
   |-- CODE_OF_CONDUCT.md   (Ground rules)
   '-- README.md            (This file)
```

## Building from Source
FyuseNet can be build for different target systems, currently supported are Linux, MacOS and Android. We are working on MS Windows
builds, which we will hopefully add soon to the list of supported systems. Parts of the builds can be adjusted via _build configurations_, 
which can enable or disable certain parts of the build or set specific OpenGL targets.

### Build Configuration
FyuseNet supports a set of build flags for customization purposes. Aside from the usual flags like compiling in _release_ or _debug_
mode, it allows for compiling different subprojects and enabling/disabling different target environments.
The following table lists those build flags, along with their default configuration and notable external dependencies if enabled.

| Build Flag               | Default | Description | Notable External Dependency |
|--------------------------|---------|-------------|-----------------------------|
| USE_EGL                  | OFF     | Use [EGL](https://www.khronos.org/egl) instead of [GL](https://www.khronos.org/gl) | [EGL](https://www.khronos.org/egl) | 
| USE_GLFW                 | OFF     | Use [GLFW](https://glfw.org) instead of [GL](https://www.khronos.org/gl), this is useful when using GL debuggers like [NVIDIA nSight](https://developer.nvidia.com/nsight-graphics) on desktop machines | [GLFW](https://glfw.org) | 
| USE_MULTITHREADING       | ON      | Depending on the build platform multi-threading may be on or off by default. For Linux and Android builds it is on | |
| BUILD_SAMPLES            | ON      | Build sample networks | |
| BUILD_TESTS              | ON     | Build unit tests (not for WebGL) | |
| BUILD_DOCS               | OFF     | Build doxygen documentation | [doxygen](https://www.doxygen.nl/index.html) |

Build flags can be set on the command line as parameters to the `cmake` executable. For example:

```
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SAMPLES <path to top-level CMakeLists.txt>
```

### Compiling for Linux Desktop and MacOS
In order to compile FyuseNet for use on Linux-based desktop systems (also including Linux-based SBCs and MacOS), the following 
prerequisites must be installed:

| Package                                           | Min version     | Comment                                       |
|---------------------------------------------------|-----------------|-----------------------------------------------|
| [cmake](https://cmake.org)                        | 3.21.0          | Lower version may work, but mileage will vary |
| [python](https://python.org)                      | 3.0             | Used for shader resource management |
| g++/clang                                         | -               | Any version that supports C++ 14 |
| OpenGL/ES (dev)                                   | GL 4.x / ES 3.x | Header files and runtime libraries of either |
| [doxygen](https://www.doxygen.nl/)                | 1.8.17          | Only if the documentation shall be built |

To compile, change the working directory to the root folder of this repository and create a `build` folder. 
Change the working directory to that `build` folder and determine if desktop GL or
embedded GL should be used and whether or not samples or tests should be built. For example, if you want to build
for desktop GL in debug mode and also build the samples, the following command (issued from within the `build` folder) will
do the work, assuming the default generator for cmake is Unix Makefiles:
```
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_SAMPLES=ON .. && make
```

As another example, using embedded GL in release mode and also building the unit-tests, use this command:
```
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_EGL=ON -DBUILD_SAMPLES=ON -DBUILD_TESTS=ON .. && make
```

This will build a set of static libraries which can be found in their respective folders (each folder generates
a static library) and the main library as a shared object file which can be found in the `fyusenet` subdirectory
after a successful build. The build process will not install the library or header files to a target.

#### Installing
To install the shared library and header files to the appropriate system folders, use `make install`
to run the build and the installation of the appropriate files
to the destination folders. The default installation prefix, which usually is `/usr/local` on Linux can
be changed using the `--prefix` parameter supplied to the `cmake` command.

#### Desktop Samples
Currently this repository only ships with two sample applications for desktop:
 - A simple style-transfer tool
 - A ResNet-50 ImageNet classifier

After building the sample, the applications can be found under
```
<build_directory>/samples/desktop/stylenet
<build_directory>/samples/desktop/resnet
```
To run a 9x9 kernel style-transfer network on an input image, use:
```
stylenet -k 9 -w <weightfile> <input.jpg> <output.jpg>
```

To run the ResNet classifier network on an input image, use:
```
resnet -w <weightfile> -c <classlabels> <input.jpg>
```

Weight files and a few example pictures can be found in the data directory.


### Compiling for Android
Compiling the library for Android should be as straightforward as for desktop Linux. The most important prerequisite for
compiling on Android is the presence of the [Android NDK](https://developer.android.com/ndk/) on the system.
FyuseNet should be able to compile with NDK versions as low as 19 and hopefully still compile with the NDK version that is
current at the time of reading these instructions. 

A more complete list of prerequisites is:
| Package                                           | Min version     | Comment                                       |
|---------------------------------------------------|-----------------|-----------------------------------------------|
| [cmake](https://cmake.org)                        | 3.21.0          | Lower version may work, but mileage will vary |
| [python](https://python.org)                      | 3.0             | Used for shader resource management           |
| [Android NDK](https://developer.android.com/ndk/) | r19             | Any version that supports C++ 14              |
| [doxygen](https://www.doxygen.nl/)                | 1.8.17          | Only if the documentation shall be built      |


The first step is to identify your NDK installation directory. If you installed the NDK from an NDK release and not part of
the Android SDK, then you already know your NDK installation directory: it is simply the top-level directory of the NDK
(for example android-ndk-r21e for the 21e release). If you use the NDK that is embedded in the SDK via the Android
SDK manager, then the installation directory of the NDK can be found by looking into the root directory of the SDK
and spot the `ndk` or `ndk-bundle` subfolder. 

In order to tell `cmake` which toolchain (consisting of compilers and linkers) to use, the following `cmake` variables
must be set:

| Variable              | Description 
|-----------------------|----------------------------------------------------|
| ANDROID_ABI           | Defines the [ABI](https://en.wikipedia.org/wiki/Application_binary_interface) for the target CPU, e.g. `arm64-v8a` for most modern Android devices |
| ANDROID_PLATFORM      | Target Android [API level](https://en.wikipedia.org/wiki/Android_version_history), for example `android-28` for Android 9 and above |
| ANDROID_NDK           | Base directory of the Android NDK installation (see description above) |
| CMAKE_TOOLCHAIN_FILE  | Toolchain definition file for `cmake`, which resides in the NDK installation |

In particular the `CMAKE_TOOLCHAIN_FILE` is usually found at `<ndk-base>/build/cmake/android-toolchain.cmake`.

An easy way to setup the build would be to create a `build-android` directory inside the FyuseNet root directory
and then - from within that directory - execute:

```
cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21 -DANDROID_NDK=<ndkdir> -DCMAKE_TOOLCHAIN_FILE=<ndkdir>/build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SAMPLES=ON <path to top-level CMakeLists.txt> && make
```

#### Android Sample
This repository includes a small sample that demonstrates the usage of FyuseNet in an Android app using Kotlin/JNI. In order for the app to build successfully,
_first_ follow the instructions above to compile FyuseNet with an Android NDK and make sure to use `-DBUILD_SAMPLES=ON` on the `cmake` command line and that multithreading is not turned off for the sample build above. Note 
that in order to build the sample app, you will require the [Android SDK](https://developer.android.com) to be installed.

The Android app can be found in the following folder:

```
<fyusenet_root>/samples/android
```

and consists of an [Android Studio](https://developer.android.com/studio) project. If you do not want to use Android Studio for building, you can simply use
the supplied [gradle](https://gradle.org) build scripts in the Android sample folder by issueing:
```
./gradlew build
```
This will build the app and store the resulting Android package(s) under the `app/build/outputs/apk` folder. The `apk` file can be installed to the
phone by using the `adb` command.

The sample app itself will simply open the front-facing camera and apply style-transfer to the camera feed and display it on the screen. Please
excuse the lack of UI design in the app, it is after all just a sample.

### Building for WebGL

FyuseNet can also be compiled to [WebAssembly](https://webassembly.org) using [emscripten](https://emscripten.org). In this case it uses 
[WebGL](https://khronos.org/webgl) as OpenGL-compatible backend. Due to the usage of GLSL 3 shaders, FyuseNet currently requires WebGL2 
to run, which is supported by the majority of modern browsers.

In order to build for WebGL, the following prerequisites must be present:

| Package                                           | Min version     | Comment                                       |
|---------------------------------------------------|-----------------|-----------------------------------------------|
| [cmake](https://cmake.org)                        | 3.21.0          | Lower version may work, but mileage will vary |
| [python](https://python.org)                      | 3.0             | Used for shader resource management           |
| [emscripten](https://emscripten.org)              | 3.1.x           | Any version that supports C++ 14              |
| [doxygen](https://www.doxygen.nl/)                | 1.8.17          | Only if the documentation shall be built      |

The following `CMAKE_BUILD_TYPES` are supported for usage with emscripten:

| Build Type            | Description 
|-----------------------|----------------------------------------------------|
| EMSCRIPTEN_DEBUG      | Non-optimized debug configuration for development  |
| EMSCRIPTEN_RELEASE    | Optimized release configuration                    |
| EMSCRIPTEN_SMALL      | Size-optimized release configuration               |
| EMSCRIPTEN_PROFILE    | Profiling version for in-depth code profiling      |

The WebAssembly/WebGL build follows the same scheme as the other builds, here is a suggestion for the build procedure:
  1. Create a `build-web` folder in the repository root and change the current directory to that folder
  2. Invoke `emcmake cmake -DCMAKE_BUILD_TYPE=EMSCRIPTEN_RELEASE -DBUILD_SAMPLES=ON ..`
  3. Invoke `make`

This should build a static library of FyuseNet as well as a sample application which will be placed in the `<build>/samples/web` 
folder. To run the sample application simply copy the files in that directory to a web server or start a small web server inside
that directory, for example: `python -m http.server <port>` and point the browser to the `stylenet.html` file.


### Building Documentation
The documentation build is fairly easy and only requires [doxygen](https://www.doxygen.nl/) to be installed. In any of the build
configurations above, simply supplying `-DBUILD_DOCS=ON` to the `cmake` command also flags the documentation to be build.
The HTML output of the documentation will be stored in a folder named `docs` in the top-level source directory.

For convenience purposes, the [documentation](https://fyusion-open-source.github.io/fyusenet) is also supplied as GitHub page and
is updated whenever the main branch is updated.

## Future Improvements
The code in FyuseNet, particularly the CPU part, is not fully optimized yet. For the GPU part, there are still too many calls
that perform redundant operations on the GL engine state and cutting a few of those may result in improved runtime.
By far the most pressing part for future improvements is to support more layer types, as our currently supported subset is
rather small. There is also a bit of optimization potential in some of the shaders that we may exploit in the future.

# Fyusion is Hiring
If you're as excited as we are about making AI/ML products that are blazing fast and accessible, you might be a great fit at Fyusion!
We're a diverse team from all around the globe, who are changing how people see and interact with the world in their everyday lives.

Want to learn more? Check out our [job openings](https://fyusion.com/jobs) and apply today!

