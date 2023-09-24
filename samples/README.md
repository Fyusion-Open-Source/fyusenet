# FyuseNet Neural Network Inference Library Samples

This folder contains basic samples that demonstrate how to use FyuseNet for different types of 
target devices and is structured as follows:

```
samples
   |-- desktop         (Application wrapper code for desktop / edge-computing devices)
   |-- android         (Application wrapper code for Android devices)
   |-- web             (Application wrapper code for web-browser targets)
   |-- samplenetworks  (Contains target-independent neural network code)
   |-- helpers         (Contains a couple of helper classes for resource loading and tokenization)
   |-- python          (Contains a conversion script for Llama-type network weights)
   '-- CMakeLists.txt  (Build file)
```
## Desktop Build (Linux, Windows, MacOS)
Currently this repository ships with three sample applications for desktop:
 - A simple style-transfer network
 - A ResNet-50 ImageNet classifier
 - A [Llama](https://ai.meta.com/blog/large-language-model-llama-meta-ai/)-style LLM network

Please follow the [instructions](../README.md#building) to compile FyuseNet and make sure to supply `-DBUILD_SAMPLES=ON` 

After building the sample, the applications can be found under
```
<build_directory>/samples/desktop/stylenet
<build_directory>/samples/desktop/resnet
<build_directory>/samples/desktop/llama
```
To run a 9x9 kernel style-transfer network on an input image, use:
```
stylenet -k 9 -w <weightfile> <input.jpg> <output.jpg>
```

To run the ResNet classifier network on an input image, use:
```
resnet -w <weightfile> -c <classlabels> <input.jpg>
```

Weight files for these networks and a few example pictures can be found in the data directory.


### LLM (Experimental)
The support for LLM nets is currently restricted to 4-bit quantized networks that were quantized using
[GPTQ](https://arxiv.org/abs/2210.17323). In particular, the sample chat provided here
was made for the [Vicuna 7B-GPTQ](https://huggingface.co/TheBloke/vicuna-7B-1.1-GPTQ)
fine-tuned version of [Llama](https://ai.meta.com/blog/large-language-model-llama-meta-ai/). 

We cannot directly load the exported files from PyTorch, thus we included a small [Python script](python/llama_weight_convert.py)
in the `python` folder to convert those files into an uncompressed zip file which we can easily parse.
To run it on a `.pt` file, use:

```
llama_weight_convert.py <input pt file> 32 <output file>
```
The 32 provides the number of decoder layers, which is 32 for the 7B parameter version of 
Llama, which again is what the sample app is currently fixed to.
For the impatient: you can download an already converted version of the network above from
[this location](https://tactilis.de/vicuna-7B-1.1-GPTQ.zip) (3.8GB) as well as the corresponding [tokenizer model](https://tactilis.de/tokenizer.model).

To run the Llama example, open a console/shell and run:
```
llama -w <weightfilee> -t <tokenizer model>
```

Note that around 5GB of available VRAM are required to run the model. Some graphics drivers (for example NVIDIA/Linux) allow it to run with far less,
however the performance will be severely degraded due to swapping between VRAM and CPU memory.

> :warning: Due to the lack of support for at least somewhat contemporary OpenGL on Apple silicon,
> the performance of the CoreGL implementation used by FyuseNet on Apple hardware is quite underwhelming. It is not
> faster than a decent CPU-based implementation. There is a good chance it will run faster in a web-browser 
> instead, due to the use of an emulation layer on top of Metal (see below). Also, the 16-bit CoreGL inference on Apple silicon is currently broken,
> make sure to use the HIGH_PRECISION build flag for testing with CoreGL on Apple hardware. It might be worth exploring 
> [ANGLE](https://github.com/google/angle) to serve as GL layer on Apple hardware instead of using CoreGL. 

## Android
This repository includes a small sample that demonstrates the usage of FyuseNet in an Android app using Kotlin/JNI. In order for the app to build successfully,
please follow the [instructions](../README.md#building) to compile FyuseNet with an Android NDK and make sure to use `-DBUILD_SAMPLES=ON` on the `cmake` command line and that multithreading is not turned off for the sample build above. Note 
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


## WebGL / WebAssembly
Please follow the [instructions](../README.md#building) to compile FyuseNet for WebAssembly and make sure to use `-DBUILD_SAMPLES=ON` as a parameter to `cmake`.

This should build a static library of FyuseNet as well as a sample application which will be placed in the `<build>/samples/web` 
folder. To run the sample application simply copy the files in that directory to a web server or start a small web server inside
that directory, for example: `python -m http.server <port>` and point the browser to the `stylenet.html` file.

