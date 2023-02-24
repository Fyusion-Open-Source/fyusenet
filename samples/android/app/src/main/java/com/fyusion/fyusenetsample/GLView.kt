package com.fyusion.fyusenetsample
import android.content.Context
import android.opengl.GLSurfaceView
import android.os.Handler
import android.view.SurfaceHolder

class GLView(ctx: Context, handler: Handler) : GLSurfaceView(ctx) {

    private lateinit var render_ : CameraRender

    init {
        setEGLContextClientVersion(3)
        render_ = CameraRender(ctx, handler, this)
        setRenderer(render_)
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        super.surfaceCreated(holder)
        renderMode = RENDERMODE_WHEN_DIRTY
    }
}