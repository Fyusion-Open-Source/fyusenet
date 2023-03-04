# FyuseNet Neural Network Inference Library
This folder contains the main library source code for FyuseNet. Please refer to the top-level README file
for general information about this library.

## Folder Structure

```
fyusenet
   |-- common         (Common functionality not related to neural networks, exceptions etc)
   |-- base           (Base classes for network layers and associated tensors/buffers)
   |-- gl             (Lightweight wrapper around OpenGL)
   |-- cpu            (CPU code for neural network layers, rudimentary)
   |-- gpu            (GPU code for neural network layers)
   |    '- vanilla    (GPU code for neural network layers not aimed at specific GPU)
   |    '- deep       (GPU code for neural network layers that use deep-tensor data)
   |    '- shaders    (GLSL shaders for all neural network layers)
   |-- fyusenet.h     (Convenience header file)
   '-- CMakeLists.txt (Library build file)
```

