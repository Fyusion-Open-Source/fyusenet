/* FyuseNet Android Sample
 * Copyright 2022 Fyusion Inc.
 *
 * SPDX-License-Identifier: MIT
 * Creator: Martin Wawro
 */
package com.fyusion.fyusenetsample

import android.content.Context
import android.graphics.SurfaceTexture
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLES30
import android.opengl.GLSurfaceView
import android.os.Handler
import android.util.Size
import android.view.Surface
import java.lang.Integer.max
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.IntBuffer
import java.util.concurrent.atomic.AtomicBoolean
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.round
import kotlin.math.roundToInt

/**
 * Style-transfer camera renderer
 *
 * This class implements a renderer for a GLSurfaceView instance that takes input from a camera
 * source via the [CameraWrapper] class, routes it through a style-transfer network and blits the
 * output to the surface view.
 */
class CameraRender(ctx: Context, handler: Handler, parent: GLSurfaceView) : GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener {

    private var cameraTexture_ : SurfaceTexture? = null
    private var camera_ : CameraWrapper? = null
    private val context_ = ctx
    private val handler_ = handler
    private val update_ : AtomicBoolean = AtomicBoolean(false)
    private val parent_ = parent
    private var blitter_ : Int = 0
    private var vao_ : Int = 0
    private var vbo_ : Int = 0
    private var mvp_ : Int = 0
    private var cameraTextureID_ : Int = 0
    private var net_ : Long = 0
    private var surfaceSize_ : Size? = null
    private lateinit var weights_ : ByteBuffer
    private var mvpData_ = floatArrayOf(1.0f, 0.0f, 0.0f, 0.0f,
                                        0.0f, 1.0f, 0.0f, 0.0f,
                                        0.0f, 0.0f, 1.0f, 0.0f,
                                        0.0f, 0.0f, 0.0f, 1.0f)

    external fun initNetwork(procWidth: Int, procHeight: Int, kernelSize: Int, weights: ByteBuffer) : Long
    external fun getOutputTexture(network : Long) : Int
    external fun processOESTexture(texture: Int, network: Long)
    external fun tearDownNetwork(network: Long)


    init {
        System.loadLibrary("styletransfer")
        loadWeights(ctx)
    }

    override fun onSurfaceCreated(unused: GL10, config: EGLConfig) {
        if (!compileShaders()) throw Exception("Unable to compile blitter")
        setupProxyGeometry()
    }

    override fun onSurfaceChanged(unused: GL10, width: Int, height: Int) {
        // create proxy SurfaceTexture for the camera preview
        surfaceSize_ = Size(width, height)
        if (cameraTexture_ == null) {
            var texture = IntBuffer.allocate(1)
            GLES20.glGenTextures(1, texture)
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, texture[0])
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,  GLES20.GL_TEXTURE_MIN_FILTER,  GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_MIRRORED_REPEAT)
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_MIRRORED_REPEAT)
            cameraTexture_ = SurfaceTexture(texture[0])
            cameraTexture_?.setDefaultBufferSize(width, height)
            cameraTexture_?.setOnFrameAvailableListener(this)
            cameraTextureID_ = texture[0]
        }
        if (camera_ == null) {
            if (max(width, height) > 720) {
                val scale = max(width, height).toFloat() / 720.0f
                val swidth = ((width / scale).roundToInt()) and 2.inv()
                val sheight  = ((height / scale).roundToInt()) and 2.inv()
                net_ = initNetwork(sheight, swidth, 3, weights_)
            } else {
                // we assume portrait mode all the time
                net_ = initNetwork(height, width, 3, weights_)
            }
            camera_ = CameraWrapper(context_, width, height, Surface(cameraTexture_), handler_, ::cameraReady)
        }
    }

    override fun onDrawFrame(unused: GL10) {
        if (update_.compareAndSet(true, false)) {
            getOutputTexture(net_)
            cameraTexture_?.updateTexImage()
            renderFrame()
        }
    }

    override fun onFrameAvailable(source: SurfaceTexture?) {
        update_.set(true)
        parent_.requestRender()
    }


    private fun cameraReady(size: Size, orientation: Int) {
        if (orientation % 180 == 90) {
            mvpData_[0] = 0.0f
            mvpData_[5] = 0.0f
            mvpData_[1] = if (orientation == 90) -1.0f else 1.0f
            mvpData_[4] = -1.0f
        }
        camera_?.start()
    }

    private fun renderFrame() {
        processOESTexture(cameraTextureID_, net_)
        surfaceSize_?.let { GLES20.glViewport(0, 0, it.width, it.height) }
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDisable(GLES20.GL_CULL_FACE)
        GLES20.glDisable(GLES20.GL_BLEND)
        GLES20.glClearColor(0.0f, 0.0f, 1.0f, 0.0f)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glBindVertexArray(vao_)
        GLES20.glUseProgram(blitter_)
        GLES20.glUniformMatrix4fv(mvp_, 1, false, mvpData_, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, getOutputTexture(net_))
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_FAN, 0, 4)
        GLES20.glUseProgram(0)
        GLES30.glBindVertexArray(0)
    }

    private fun compileShaders() : Boolean {
        // TODO (mw) error checking
        val fragsource = "#version 300 es\n" +
                "uniform sampler2D image;\n" +
                "in mediump vec2 texture_coordinate;\n" +
                "out mediump vec4 frag_color;\n" +
                "void main() {\n" +
                "  frag_color.rgb = texture(image, texture_coordinate).rgb;\n" +
                "  frag_color.a = 1.0;\n" +
                "}";
        val vertsource = "#version 300 es\n" +
                "in vec4 vertex;\n" +
                "out vec2 texture_coordinate;\n" +
                "uniform mat4 MVP;\n" +
                "void main() {\n" +
                "  gl_Position = MVP * vec4(vertex.xy, 0, 1);\n" +
                "  texture_coordinate = vertex.zw;\n" +
                "}";
        var verthdl = GLES20.glCreateShader(GLES20.GL_VERTEX_SHADER)
        var fraghdl = GLES20.glCreateShader(GLES20.GL_FRAGMENT_SHADER)
        GLES20.glShaderSource(verthdl, vertsource)
        GLES20.glShaderSource(fraghdl, fragsource)
        GLES20.glCompileShader(verthdl)
        GLES20.glCompileShader(fraghdl)
        blitter_ = GLES20.glCreateProgram()
        GLES20.glAttachShader(blitter_, verthdl)
        GLES20.glAttachShader(blitter_, fraghdl)
        GLES20.glLinkProgram(blitter_)
        GLES20.glDeleteShader(verthdl)
        GLES20.glDeleteShader(fraghdl)
        mvp_ = GLES20.glGetUniformLocation(blitter_, "MVP")
        return true
    }

    private fun setupProxyGeometry() {
        // TODO error handling
        val vertices = floatArrayOf(-1.0f, -1.0f,  0.0f, 0.0f,
                                     1.0f, -1.0f,  1.0f, 0.0f,
                                     1.0f,  1.0f,  1.0f, 1.0f,
                                    -1.0f,  1.0f,  0.0f, 1.0f)
        val bytebuf = ByteBuffer.allocateDirect(16 * 4)
        bytebuf.order(ByteOrder.nativeOrder())
        val vertbuf = bytebuf.asFloatBuffer()
        vertbuf.put(vertices)
        vertbuf.position(0)
        var vao = IntBuffer.allocate(1)
        GLES30.glGenVertexArrays(1, vao)
        vao_ = vao[0]
        var buffers = IntBuffer.allocate(1)
        GLES20.glGenBuffers(1, buffers)
        vbo_ = buffers[0]
        GLES30.glBindVertexArray(vao_)
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, vbo_)
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, 4*4*4, vertbuf, GLES20.GL_STATIC_DRAW)
        GLES30.glEnableVertexAttribArray(0)
        GLES30.glVertexAttribPointer(0, 4, GLES20.GL_FLOAT, false, 0, 0)
        GLES30.glBindVertexArray(0)
    }

    private fun loadWeights(ctx: Context) {
        val assetManager = ctx.assets
        val inputStream = assetManager.open("style.bin")
        val wbdata = inputStream.readBytes()
        if (wbdata.size == 0) throw Exception("Cannot read weight-data")
        weights_ = ByteBuffer.allocateDirect(wbdata.size)
        weights_.put(wbdata)
    }

}