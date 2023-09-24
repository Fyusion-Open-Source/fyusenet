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

#include <cstdio>
#include <iostream>
#include <thread>

//-------------------------------------- Project  Headers ------------------------------------------

#include "../samplenetworks/resnet50.h"
#include "../helpers/jpegio.h"
#include "cxxopts.hpp"

#ifdef FYUSENET_USE_GLFW
#include <fyusenet/gl/glcontext.h>
#endif

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


int main(int argc, char **argv) {
    using namespace std::chrono_literals;
    cxxopts::Options options(argv[0],"Sample ResNet-50 network");
    options.add_options()("h,help","Get program help")
                         ("c,classes", "File name to textfile with the class label names, one label per line (optional)", cxxopts::value<std::string>())
                         ("w,weights", "Use supplied filename as weight file (mandatory)", cxxopts::value<std::string>())
                         ("r,runs", "Perform multiple runs on the same dataset", cxxopts::value<int>())
                         ("m,memory", "Slow down to make it possible to get some memory benchmarks", cxxopts::value<bool>())
#ifdef DEBUG
                         ("l,log", "Log layer outputs to supplied directory", cxxopts::value<std::string>())
#endif
                         ("input", "Input JPEG file", cxxopts::value<std::string>());
    options.parse_positional({"input"});
    auto opts = options.parse(argc, argv);

    if ((opts.count("help") > 0) || (opts.count("input") == 0) || (opts.count("weights") == 0)) {
        std::cout<<options.help()<<std::endl;
        return 0;
    }

    // -------------------------------------------------------
    // Setup GL context and thread/PBO pool. If we use GLFW,
    // set mouse-button callbacks and wait for an initial
    // MB press, followed by a couple of empty render calls
    // -------------------------------------------------------
    auto glmgr = fyusion::fyusenet::GfxContextManager::instance();
    if (!glmgr) {
        std::cerr<<"Cannot setup GL context\n";
        return 1;
    }
    fyusion::fyusenet::GfxContextLink ctx = glmgr->createMainContext();
#ifdef FYUSENET_MULTITHREADING
    fyusion::opengl::AsyncPool::setMaxGLThreads(4);
#endif
    glmgr->setupPBOPools(2, 2);
    // -------------------------------------------------------
    // Read JPEG image that is to be processed
    // -------------------------------------------------------
    int width, height;
    /*
    GLuint tex = readImageToTexture(opts["input"].as<std::string>(), width, height);
    if (!tex) return 1;
    */
    float * rgb = readImage(opts["input"].as<std::string>(), width, height);
    if (!rgb) return 1;
    if ((width != 224) || (height != 224)) {
        std::cerr<<"Input image must be 224x224 pixels\n";
        return 1;
    }

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
    for (int i=0; i<4; i++) {
        glctx->sync();
    }
#endif
    // -------------------------------------------------------
    // Instantiate network
    // -------------------------------------------------------
    auto * net = new ResNet50(true, true);
    auto * params = new ResNet50Provider(opts["weights"].as<std::string>());
    // -------------------------------------------------------
    // Load weights, setup and run network...
    // -------------------------------------------------------
    net->setParameters(params);
    net->setup();
    net->setInputBuffer(rgb);
    if (opts.count("log") > 0) {
        net->enableLog(opts["log"].as<std::string>());
    }
    net->forward();
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
    // -------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------
    net->cleanup();
    delete net;
    ctx.reset();
    glmgr->tearDown();
    return 0;
}


// vim: set expandtab ts=4 sw=4:

