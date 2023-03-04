//--------------------------------------------------------------------------------------------------
// FyuseNet Samples                                                            (c) Fyusion Inc. 2023
//--------------------------------------------------------------------------------------------------
// ResNet (50) Network Example Main
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------

// Model taken from here:
// https://microsoft.github.io/onnxjs-demo/#/resnet50

//--------------------------------------- System Headers -------------------------------------------

#include <cassert>
#include <cstdio>
#include <iostream>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../samplenetworks/resnet50.h"
#include "../helpers/jpegio.h"
#include "cxxopts.hpp"

#ifdef FYUSENET_USE_GLFW
#include <fyusenet/gl/glcontext.h>
#endif

#include <fyusenet/common/performance.h>


//-------------------------------------- Global Variables ------------------------------------------

constexpr int IMAGENET_CLASS_COUNT = 1000;

//-------------------------------------- Local Definitions -----------------------------------------


static float * readImage(const std::string& imageFile, int & width, int & height) {
    if (!JPEGIO::isJPEG(imageFile)) {
        std::cerr<<"File "<<imageFile<<" is not a JPEG file\n";
        return nullptr;
    }
    uint8_t * rgb = JPEGIO::loadRGBImage(imageFile, width, height);
    if (!rgb) {
        std::cerr<<"Cannot read "<<imageFile<<" make sure it is an RGB image\n";
        return nullptr;
    }
    float * image = new float[width * height * 3];
    for (int i=0; i < width*height*3; i++) {
        image[i] = ((float)rgb[i]) / 255.f;
    }
    delete [] rgb;
    return image;
}


static float * loadWeights(const std::string& fileName, size_t & numFloats) {
    FILE * in = fopen(fileName.c_str(),"rb");
    if (!in) {
        std::cerr<<"Cannot open weight file "<<fileName<<" for reading\n";
        return nullptr;
    }
    fseek(in, 0, SEEK_END);
    size_t filesize = ftell(in);
    assert((filesize % sizeof(float)) == 0);
    fseek(in, 0, SEEK_SET);
    numFloats = filesize/sizeof(float);
    float * weights = new float[filesize/sizeof(float)];
    fread(weights, 1, filesize, in);
    fclose(in);
    return weights;
}

int main(int argc, char **argv) {
    cxxopts::Options options(argv[0],"Sample style-transfer network");
    options.add_options()("h,help","Get program help")
                         ("k,kernel", "Kernel size for the convolution layers, either 3 for 3x3 or 9 for 9x9", cxxopts::value<int>()->default_value("3"))
                         ("c,classes", "File name to textfile with the class label names, one label per line (optional)", cxxopts::value<std::string>())
                         ("w,weights", "Use supplied filename as weight file (mandatory)", cxxopts::value<std::string>())
#ifdef DEBUG
                         ("l,log", "Log layer outputs to supplied directory", cxxopts::value<std::string>())
#endif
                         ("input", "Input JPEG file", cxxopts::value<std::string>())
                         ("output", "Output JPEG file", cxxopts::value<std::string>());
    options.parse_positional({"input"});
    auto opts = options.parse(argc, argv);

    if ((opts.count("help") > 0) || (opts.count("input") == 0) || (opts.count("weights") == 0)) {
        std::cout<<options.help()<<std::endl;
        return 0;
    }

    // -------------------------------------------------------
    // Read JPEG image that is to be processed
    // -------------------------------------------------------
    int width, height;
    float * rgb = readImage(opts["input"].as<std::string>(), width, height);
    if (!rgb) return 1;
    if ((width != 224) || (height != 224)) {
        std::cerr<<"Input image must be 224x224 pixels\n";
        return 1;
    }
    // -------------------------------------------------------
    // Setup GL context and thread/PBO pool. If we use GLFW,
    // set mouse-button callbacks and wait for an initial
    // MB press, followed by a couple of empty render calls
    // -------------------------------------------------------
    auto glmgr = fyusion::fyusenet::GfxContextManager::instance();
    if (!glmgr) {
        std::cerr<<"Cannot setup GL context\n";
        delete [] rgb;
        return 1;
    }
    fyusion::fyusenet::GfxContextLink ctx = glmgr->createMainContext();
#ifdef FYUSENET_MULTITHREADING
    fyusion::opengl::AsyncPool::setMaxGLThreads(4);
#endif
    glmgr->setupPBOPools(2, 2);

    // NOTE (mw) this is ugly
#ifdef FYUSENET_USE_GLFW
    static bool buttonup = false;
    const fyusion::opengl::GLContext * glctx = dynamic_cast<const fyusion::opengl::GLContext *>(ctx.interface());
    auto mousecb = [](GLFWwindow *win, int bt, int action, int mods) {
        if (action == GLFW_PRESS) buttonup = true;
    };
    glfwSetMouseButtonCallback(glctx->window(), mousecb);
    while (!buttonup) {
        glfwWaitEventsTimeout(0.1);
    }
    for (int i=0; i<6; i++) {
        glctx->sync();
    }
#endif
    // -------------------------------------------------------
    // Instantiate network
    // -------------------------------------------------------
    ResNet50 * net = new ResNet50();
    // -------------------------------------------------------
    // Load weights, setup and run network...
    // -------------------------------------------------------
    size_t weightfloats;
    float * weights = loadWeights(opts["weights"].as<std::string>(), weightfloats);
    if (!weights) {
        delete [] rgb;
        return 1;
    }
    net->loadWeightsAndBiases(weights, weightfloats);
    net->setup();
    net->setInputBuffer(rgb);
    tstamp start = fy_get_stamp();
    auto state = net->forward();
    tstamp stop = fy_get_stamp();
    assert(state.status == fyusion::fyusenet::NeuralNetwork::state::EXEC_DONE);
    // -------------------------------------------------------
    // Determine most likely class
    // -------------------------------------------------------
    auto * downbuf = net->getOutputBuffer();
    CPUBuffer * chanbuf = (downbuf) ? downbuf->toChannelWise() : nullptr;
    int matchedclass = -1;
    if (chanbuf) {
        const float * ptr = chanbuf->map<float>(true);
        int argmax = 0;
        float valmax = ptr[0];
        for (int i=1; i < IMAGENET_CLASS_COUNT; i++) {
            if (ptr[i] > valmax) {
                argmax = i;
                valmax = ptr[i];
            }
        }
        matchedclass = argmax;
    }
    if (matchedclass >= 0) {
        if (opts.count("classes") > 0) {
            FILE * labfile = fopen(opts["classes"].as<std::string>().c_str(), "r");
            if (!labfile) {
                std::cerr<<"Cannot open class label file "<<opts["classes"].as<std::string>()<<"\n";
                std::cout<<"\nLabel#: "<<matchedclass<<"\n";
            } else {
                fseek(labfile, 0, SEEK_END);
                size_t sz = ftell(labfile);
                fseek(labfile, 0, SEEK_SET);
                std::unique_ptr<char[]> labels(new char[sz+1]);
                fread(labels.get(), 1, sz, labfile);
                fclose(labfile);
                char * ptr = labels.get();
                for (int c=0; c < matchedclass; c++) {
                    char * lineend = strpbrk(ptr,"\n\r");
                    while (*lineend == 10 || *lineend == 13) lineend++;
                    ptr = lineend;
                }
                char * lineend = strpbrk(ptr,"\n\r");
                while (*lineend == 10 || *lineend == 13) lineend++;
                lineend[-1] = 0;
                std::cout<<"\n"<<ptr<<"\n";
            }
        } else {
            std::cout<<"\nLabel#: "<<matchedclass<<"\n";
        }
    } else {
        std::cout<<"Could not match any class to the input\n";
    }
    std::cout<<"Inference took "<<fy_elapsed_millis(start, stop)<<"ms (including texture upload and download)\n";
    // -------------------------------------------------------
    // If we use GLFW, wait for another MB click before
    // terminating
    // -------------------------------------------------------
#ifdef FYUSENET_USE_GLFW
    static bool buttondown = false;
    glctx->sync();
    auto mousecbout = [](GLFWwindow *win, int bt, int action, int mods) {
        if (action == GLFW_PRESS) {
            buttondown = true;
        }
    };
    glfwSetMouseButtonCallback(glctx->window(), mousecbout);
    while (!buttondown) {
        glfwWaitEventsTimeout(0.1);
    }
#endif

    // -------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------
    net->cleanup();
    delete net;
    delete [] rgb;
    ctx.reset();
    glmgr->tearDown();
    return 0;
}


// vim: set expandtab ts=4 sw=4:

