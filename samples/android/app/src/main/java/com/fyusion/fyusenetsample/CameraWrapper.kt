package com.fyusion.fyusenetsample

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Handler
import android.util.Size
import android.view.Surface
import java.lang.Math.abs

class CameraWrapper(ctx : Context, width: Int, height: Int, surface : Surface, handler : Handler, callback: (sz: Size, orientation: Int) -> Unit) {

    private lateinit var cameraID_ : String
    private var cameraDev_ : CameraDevice? = null
    private var session_ : CameraCaptureSession? = null
    private val callback_ = callback
    private val surface_ = surface
    private val handler_ = handler
    private val surfaceSize_ = Size(width, height)

    init {
        val mgr = ctx.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        if (mgr.cameraIdList.isEmpty()) throw Exception("No camera found")
        var camfound = false
        for (cam in mgr.cameraIdList) {
            val crt = mgr.getCameraCharacteristics(cam)
            if (crt.get<Int>(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT) {
                openCamera(cam, mgr, surface)
                camfound = true
                break
            }
        }
        if (!camfound) throw Exception("No suitable camera found")
    }

    @Suppress("DEPRECATION")
    fun start() {
        cameraDev_?.createCaptureSession(mutableListOf(surface_),
            object: CameraCaptureSession.StateCallback() {

            override fun onConfigureFailed(session: CameraCaptureSession) {
            }

            override fun onConfigured(session: CameraCaptureSession) {
                val request = cameraDev_!!.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addTarget(surface_)
                }
                session.setRepeatingRequest(request.build(), null, Handler { true })
                session_ = session
            }
        }, handler_)
    }

    fun close() {
        // there is a race condition here if close is called before the session was set up
        // as this is just an example, we don't care
        if (cameraDev_ != null && session_ != null) {
            session_!!.close()
            session_ = null
        }
    }

    @SuppressLint("MissingPermission")
    private fun openCamera(camID: String, mgr : CameraManager, surface: Surface) {
        mgr.openCamera(camID, object: CameraDevice.StateCallback() {
            override fun onDisconnected(dev: CameraDevice) {
                close()
            }

            override fun onError(dev: CameraDevice, code: Int) {
                throw Exception("Cannot configure camera stream")
            }

            override fun onOpened(dev: CameraDevice) {
                cameraID_ = camID
                cameraDev_ = dev
                configureStream(dev, mgr.getCameraCharacteristics(camID), surface)
            }
        }, handler_)
    }

    private fun configureStream(dev: CameraDevice, crt: CameraCharacteristics, surface: Surface) {
        var streamconfig = crt[CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP]
        var orientation = crt[CameraCharacteristics.SENSOR_ORIENTATION] ?: 0
        var resolutions = streamconfig?.getOutputSizes(ImageFormat.YUV_420_888)
        val tgtarea = surfaceSize_.width * surfaceSize_.height
        var bestdiff = tgtarea
        var bestsize : Size? = null
        resolutions?.let {
            for (res in it) {
                val area = res.width * res.height
                val diff = abs(tgtarea-area)
                if (diff < bestdiff) {
                    bestdiff = diff
                    bestsize = res
                }
            }
        }
        bestsize?.let {
            callback_(it, orientation)
        } ?: throw Exception("Cannot configure camera stream")
    }
}