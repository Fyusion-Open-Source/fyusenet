//--------------------------------------------------------------------------------------------------
// FyuseNet Sample                                                             (c) Fyusion Inc. 2022
//--------------------------------------------------------------------------------------------------
// Style-Transfer Network Example for Android
// Creator: Martin Wawro
// SPDX-License-Identifier: MIT
//--------------------------------------------------------------------------------------------------


//--------------------------------------- System Headers -------------------------------------------

#include <jni.h>
#include <android/log.h>
#include <cstdint>
#include <atomic>
#include <unordered_map>

//-------------------------------------- Project  Headers ------------------------------------------

#include <samplenetworks/stylenet3x3.h>
#include <samplenetworks/stylenet9x9.h>
#include <fyusenet/gpu/gfxcontextmanager.h>

//-------------------------------------- Global Variables ------------------------------------------


//-------------------------------------- Local Definitions -----------------------------------------

static std::unordered_map<jlong, fyusion::fyusenet::GfxContextLink> contexts_;



/**
 * @brief Initialize neural network
 *
 * @param env JNI environment
 * @param thiz Java object we were called from
 * @param proc_width Width for the network
 * @param proc_height Height for the network
 * @param kernel_size Kernel size for the first/last convolution, either 3 or 9
 * @param buffer Native bytebuffer that contains the weights/biases for the network
 *
 * @return Network handle that identifies the created neural network
 *
 * @pre GLES context is current to the calling thread (created from within Java/Kotlin)
 *
 * Creates and initializes the neural network for operation.
 */
extern "C"
JNIEXPORT jlong JNICALL
Java_com_fyusion_fyusenetsample_CameraRender_initNetwork(JNIEnv *env, jobject thiz, jint proc_width,
                                                         jint proc_height, jint kernel_size, jobject buffer) {
    auto mgr = fyusion::fyusenet::GfxContextManager::instance();
    const uint8_t *bufptr = (const uint8_t *)env->GetDirectBufferAddress(buffer);
    jlong bufsize = env->GetDirectBufferCapacity(buffer);
    if ((!bufptr) || (bufsize == 0)) return 0;
    auto context = mgr->createMainContextFromCurrent();
    StyleNetBase * net = nullptr;
    switch (kernel_size) {
      case 9:
          net = new StyleNet9x9(proc_width, proc_height, false, false, context);
          break;
      default:
          net = new StyleNet3x3(proc_width, proc_height, false, false, context);
          break;
    }
    net->enableOESInput();
    net->setup();
    net->loadWeightsAndBiases((float *)bufptr, bufsize/4);

    jlong rc = (jlong)net;
    __android_log_print(ANDROID_LOG_DEBUG, "JNI", "allocated %p", net);
    contexts_[rc] = context;
    return rc;
}



/**
 * @brief Retrieve texture ID for network output texture
 *
 * @param env JNI environment
 * @param thiz Java object we were called from
 * @param network Network handle
 *
 * @return Raw GL texture ID of output texture that the network writes to
 */
extern "C"
JNIEXPORT jint JNICALL
Java_com_fyusion_fyusenetsample_CameraRender_getOutputTexture(JNIEnv *env, jobject thiz, jlong network) {
    StyleNetBase * net = reinterpret_cast<StyleNetBase *>(network);
    __android_log_print(ANDROID_LOG_DEBUG, "JNI", "passed network: %p", net);
    return (net) ? net->getOutputTexture() : 0;
}



/**
 * @brief Process single image from an OES texture by the neural network
 *
 * @param env JNI environment
 * @param thiz Java object we were called from
 * @param texture Raw GL texture ID to process
 * @param network Network handle
 */
extern "C"
JNIEXPORT void JNICALL
Java_com_fyusion_fyusenetsample_CameraRender_processOESTexture(JNIEnv *env, jobject thiz,
                                                               jint texture, jlong network) {    
    StyleNetBase * net = reinterpret_cast<StyleNetBase *>(network);
    __android_log_print(ANDROID_LOG_DEBUG, "JNI", "passed network: %p", net);
    if (net) {
        net->setInputTexture(texture);
        net->forward();
    }
}



/**
 * @brief Tear down network
 *
 * @param env JNI environment
 * @param thiz Java object we were called from
*  @param network Network handle
 */
extern "C"
JNIEXPORT void JNICALL
Java_com_fyusion_fyusenetsample_CameraRender_tearDownNetwork(JNIEnv *env, jobject thiz, jlong network) {
    StyleNetBase * net = reinterpret_cast<StyleNetBase *>(network);
    __android_log_print(ANDROID_LOG_DEBUG, "JNI", "passed network: %p", net);
    if (net) {
        jlong rc = (jlong)net;
        __android_log_print(ANDROID_LOG_DEBUG, "JNI", "allocated %p", net);
        // we are not taking down the context here as all this is single-shot. not nice but this
        // is just a sample
        net->cleanup();
        delete net;
    }
}
