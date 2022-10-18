export const debugVideoCapabilities = async () => {
  navigator.mediaDevices
    .getUserMedia({ audio: true, video: true })
    .then((stream) => {
      const track = stream.getVideoTracks()[0];
      console.log("Device: " + track.label);
      let capabilities = track.getCapabilities();
      console.dir(capabilities);
    });
};

export const dummyDiv = (color) => {
  return <div style={{ height: 50, width: 50, backgroundColor: color }}></div>;
};

export const checkSupportedDevices = () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    console.log("enumerateDevices() not supported.");
    return;
  }

  // List cameras and microphones.

  navigator.mediaDevices
    .enumerateDevices()
    .then(function (devices) {
      devices.forEach(function (device) {
        console.log(
          device.kind + ": " + device.label + " id = " + device.deviceId
        );
      });
    })
    .catch(function (err) {
      console.log(err.name + ": " + err.message);
    });
};

export const debugTF = () => {
  // console.log(tfgl.version_webgl);
  // console.log(tf.getBackend());
  // tfgl.webgl.forceHalfFloat();
  // var maxSize = tfgl.webgl_util.getWebGLMaxTextureSize(tfgl.version_webgl);
  // console.log(maxSize);
};
