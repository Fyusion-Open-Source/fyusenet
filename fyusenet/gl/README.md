# FyuseNet Neural Network Inference Library (GL Wrapper)

This folder contains a lightweight and not very abstracting wrapper around [OpenGL](https://khronos.org/opengl).
Its main purpose is not to perform general graphics/rendering of any kind, but to provide some _syntactic sugar_ around
OpenGL which is taylored to be used with the rest of the inference engine. It can easily be used in conjunction with 
low-level GL commands (in fact it is used like that in FyuseNet) at the expense of not really tracking the GL state inside 
the wrapper, which leads to more commands being issued on average.

Other than the syntactic sugar, this wrapper adds a few things that come in handy:
  1. A small shader resource system
  2. A shader cache
  3. Addition of #include statements in shader codes

## Shader Resource System
In order to bake the shader code into the binary, shaders are preprocessed by a small Python script that is
located at `<root>/buildutils/shaderpp`. This script transforms a shader file into a C array which can be 
included at compile time. In addition, the script adds "registration code" to that file which is executed
when the binary is loaded by the runtime linker. This registration makes it necessary to link the shader
files directly into the binaries when using static libraries, as the code will not be executed otherwise.

Inside FyuseNet the shader resources can be accessed by virtual filenames, for example:

```
auto fshader = FragmentShader::fromResource("shaders/vanilla/conv1x1.frag");
```

will create a fragment shader from a compiled in shaders identified by `shaders/vanilla/conv1x1.frag`.
Currently. no compression or obfuscation is applied to baked shader resources, the latter being useless
anyway.

## Shader Cache
As can be seen in the shader codes and the way shaders are handled in FyuseNet, there is a _lot_ of conditional
compilation going on, using preprocessor macros. Also, every layer compiles its own shaders, which results in
a rather large number of shaders. Though these shaders are mostly rather simple, compilation time becomes
an issue at some point. Also, we do not like to throw around more than a thousand shaders. In order to
reduce the load, shader source codes are hashed and compared against a database of already compiled shaders.
If an already existing shader was found, no compilation is done and the cached shader is used.

For full shader programs the procedure works by first querying existing shader handles for the participating
shaders, then use these handles and a _module ID_ as query key for the cache. The module ID is dependent
on the layer type.

## Including Snippets
Some parts of shaders are shared among a large subset of them, in particular activation functions. For this
reason an `#include` statement as in C/C++ would be a great asset to avoid redundancy. FyuseNet offers such
a mechanism, albeit only for the top-level file, which means that `#include` statements cannot be nested.
The parts to be included are called _snippets_ and the filename to be supplied to the `#include` statement
must be its resource name. For example:

```
#include "shaders/activation.inc"

``` 

will insert the referenced resource at the position of the `#include` statement.


