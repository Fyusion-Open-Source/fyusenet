function startWebcam(glContext, initCallback, frameCallback, texID) {
    let ctx = glContext;
    let camera = {
        start: async function() {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }});
            this._video = document.createElement("video");
            this._video.setAttribute("playsinline","");
            if ("srcObject" in this._video) {
                this._video.srcObject = stream;
            } else {
                this._video.src = stream;
            }
            this._video.addEventListener("loadedmetadata", this.metaReady);
            this._video.play();
            setInterval(this.processFrame,50);
        },
        processFrame : function(cam) {
            if (camera._texup) {
                let gl = camera._context;
                gl.bindTexture(gl.TEXTURE_2D, camera._texture);
                gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, camera._video.videoWidth, camera._video.videoHeight, gl.RGBA, gl.UNSIGNED_BYTE, camera._video);
                if (camera._callback) camera._callback(camera._texture, gl);
            }
        },
        metaReady : function() {
            if (camera._play === false) {
                camera._video.play();
                camera._play = true;
            }
            if (camera._video.videoWidth > 0) {
                let gl = camera._context;
                if (!camera._texture) camera._texture = GL.textures[camera._textureID];
                gl.bindTexture(gl.TEXTURE_2D, camera._texture);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, camera._video.videoWidth, camera._video.videoHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, camera._video);
                camera._texup = true;
                if (camera._initCallback) camera._initCallback(camera);
            }
        },
        width: function() {
            return camera._video.videoWidth;
        },
        height: function() {
            return camera._video.videoHeight;
        },
        _context : glContext,
        _texup: false,
        _texture: null,
        _textureID: texID,
        _video: null,
        _callback: frameCallback,
        _initCallback: initCallback,
        _play: false
    }
    camera.start();
    return camera;
}

