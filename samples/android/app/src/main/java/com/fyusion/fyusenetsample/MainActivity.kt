package com.fyusion.fyusenetsample

import android.Manifest
import android.content.pm.PackageManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity() {

    override fun onStart() {
        super.onStart()
        val camaccess = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (!camaccess) {
            val perms = arrayOf(android.Manifest.permission.CAMERA)
            ActivityCompat.requestPermissions(this, perms, 0)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(GLView(this, Handler { true }))
    }

}