import { useState, useEffect } from "react";

function getWindowDimensions() {
  //   const { innerWidth: width, innerHeight: height } = window;
  const { width, height } = window.visualViewport;
  return {
    width,
    height,
  };
}

export const calcExpectedWebCamDim = (
  windowHeight,
  windowWidth,
  verticalPanelSplitPercentage,
  horizontalPanelSplitPercentage,
  webCamRatios
) => {
  const webCamPanelHeight = (windowHeight * verticalPanelSplitPercentage) / 100;
  const webCamPanelWidth = (windowWidth * horizontalPanelSplitPercentage) / 100;
  const targetWebCamHeight = webCamPanelWidth / webCamRatios[0];
  const targetWebCamWidth = webCamPanelHeight * webCamRatios[0];
  let expectedWebCamHeight = webCamPanelHeight;
  let expectedWebCamWidth = webCamPanelWidth;
  if (targetWebCamHeight > webCamPanelHeight) {
    expectedWebCamWidth = targetWebCamWidth;
  } else {
    expectedWebCamHeight = targetWebCamHeight;
  }

  return {
    webCamPanelWidth: webCamPanelWidth,
    webCamPanelHeight: webCamPanelHeight,
    expectedWebCamWidth: expectedWebCamWidth,
    expectedWebCamHeight: expectedWebCamHeight,
  };
};

export const calcExpectedImgDim = (
  oriImgWidth,
  oriImgHeight,
  webCamPanelWidth,
  webCamPanelHeight
) => {
  const oriImgRatio = oriImgWidth / oriImgHeight;
  const targetImgHeight = webCamPanelWidth / oriImgRatio;
  const targetImgWidth = webCamPanelHeight * oriImgRatio;
  let expectedImgHeight = webCamPanelHeight;
  let expectedImgWidth = webCamPanelWidth;
  if (targetImgHeight > webCamPanelHeight) {
    expectedImgWidth = targetImgWidth;
  } else {
    expectedImgHeight = targetImgHeight;
  }
  return {
    expectedImgWidth: expectedImgWidth,
    expectedImgHeight: expectedImgHeight,
  };
};

export const buildDetectedObjects = (
  scores,
  threshold,
  boxes,
  drawHeight,
  drawWidth,
  captureHeight,
  captureWidth
) => {
  const detectionObjects = [];

  let i = 0;
  const len = scores.length;
  while (i < len) {
    if (scores[i] < threshold) {
      break;
    }
    const bbox = [];
    const minY = boxes[0][i][0] * drawHeight;
    const minX = boxes[0][i][1] * drawWidth;
    const maxY = boxes[0][i][2] * drawHeight;
    const maxX = boxes[0][i][3] * drawWidth;
    bbox[0] = minX;
    bbox[1] = minY;
    bbox[2] = maxX - minX;
    bbox[3] = maxY - minY;
    // index for sliding
    const oriMinY = parseInt(boxes[0][i][0] * captureHeight);
    const oriMinX = parseInt(boxes[0][i][1] * captureWidth);
    const oriMaxY = parseInt(boxes[0][i][2] * captureHeight);
    const oriMaxX = parseInt(boxes[0][i][3] * captureWidth);
    const slideBox = [
      oriMinY,
      oriMinX,
      oriMaxY - oriMinY + 1,
      oriMaxX - oriMinX + 1,
    ];
    detectionObjects.push({
      label: "face",
      score: scores[i],
      bbox: bbox,
      slideBox: slideBox,
      cropBox: boxes[0][i],
      boxInd: i,
    });
    i++;
  }
  return detectionObjects;
};

const readImageFile = (file) => {
  return new Promise((resolve) => {
    const reader = new FileReader();

    reader.onload = () => resolve(reader.result);

    reader.readAsDataURL(file);
  });
};

const createHTMLImageElement = (imageSrc) => {
  return new Promise((resolve) => {
    const img = new Image();

    img.onload = () => resolve(img);

    img.src = imageSrc;
  });
};

export const getWebCamInfos = async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    console.log("enumerateDevices() not supported.");
    return [
      {
        deviceName: null,
        deviceId: null,
      },
    ];
  }
  return await navigator.mediaDevices
    .enumerateDevices()
    .then(function (devices) {
      let webCamInfos = [];
      let i = 0;
      const len = devices.length;
      while (i < len) {
        if (devices[i].kind === "videoinput") {
          webCamInfos.push({
            deviceName: devices[i].label,
            deviceId: devices[i].deviceId,
          });
        }
        i++;
      }
      return webCamInfos;
    })
    .catch(function (err) {
      console.log(err.name + ": " + err.message);
      return [
        {
          deviceName: null,
          deviceId: null,
        },
      ];
    });
};

/**
 * Hooks for getting dimension of the browser window
 */
export function useWindowDimensions() {
  const [windowDimensions, setWindowDimensions] = useState(
    getWindowDimensions()
  );

  useEffect(() => {
    function handleResize() {
      setWindowDimensions(getWindowDimensions());
    }

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return windowDimensions;
}

// const onCapture = async (detectionModel, classificationModel) => {
//   if (
//     !isUsingWebcam ||
//     detectionModel == null ||
//     classificationModel == null
//   ) {
//     return;
//   }
//   if (
//     typeof webcamRef.current !== "undefined" &&
//     webcamRef.current !== null &&
//     webcamRef.current.video.readyState === 4
//   ) {
//     // tf.engine().startScope();
//     /** @type {HTMLVideoElement} */
//     const video = webcamRef.current.video;

//     // Set video width
//     webcamRef.current.video.width = expectedWebCamWidth;
//     webcamRef.current.video.height = expectedWebCamHeight;

//     const cnvs = document.getElementById("myCanvas");
//     cnvs.width = expectedWebCamWidth;
//     cnvs.height = expectedWebCamHeight;

//     const ctx = cnvs.getContext("2d");

//     // Font options.
//     const font = "16px sans-serif";
//     ctx.font = font;
//     ctx.textBaseline = "top";

//     /** @type {tf.Tensor3D} */
//     const rawImgTensor = tf.browser.fromPixels(video);
//     const inputTensor = tf.tidy(() => {
//       return rawImgTensor.transpose([0, 1, 2]).expandDims();
//     });
//     // let startTime = performance.now();
//     detectionModel
//       .executeAsync(inputTensor)
//       .then((res) => {
//         // const a0 = res[0].arraySync(); // num_detection
//         // const a1 = res[1].arraySync(); // raw_detection_boxes
//         // const a2 = res[2].arraySync(); // detection_anchor_indices
//         // const a3 = res[3].arraySync(); // raw_detection_scores
//         // const a4 = res[4].arraySync(); // detection_boxes
//         // const a5 = res[5].arraySync(); // detection_classes
//         // const a6 = res[6].arraySync(); // detection_scores
//         // const a7 = res[7].arraySync(); // detection_multiclass_scores
//         const detection_boxes = res[4].arraySync();
//         const detection_classes = res[5].arraySync();
//         const detection_scores = res[6].dataSync();

//         const detections = utils.buildDetectedObjects(
//           detection_scores,
//           0.5,
//           detection_boxes,
//           detection_classes,
//           expectedWebCamHeight,
//           expectedWebCamWidth,
//           720,
//           1280
//         );

//         // prepare variables
//         let i;
//         let len;
//         let limitLen;
//         const _faceDetectionScores = Array(maxNumOfDisplayFace).fill(null);
//         const _maskClassificationScores =
//           Array(maxNumOfDisplayFace).fill(null);

//         if (detections.length > 0) {
//           // loop for rendering bounding boxes
//           i = 0;
//           len = detections.length;
//           limitLen = maxNumOfDisplayFace;

//           // clear bounding box canvas
//           ctx.clearRect(0, 0, expectedWebCamHeight, expectedWebCamWidth);
//           while (i < len) {
//             const x = detections[i]["bbox"][0];
//             const y = detections[i]["bbox"][1];
//             const width = detections[i]["bbox"][2];
//             const height = detections[i]["bbox"][3];

//             // Draw the bounding box.
//             ctx.strokeStyle = "#00FFFF";
//             ctx.lineWidth = 4;
//             ctx.strokeRect(x, y, width, height);

//             // Draw the label background.
//             ctx.fillStyle = "#00FFFF";
//             const textWidth = ctx.measureText(
//               detections[i]["label"] +
//                 " " +
//                 (100 * detections[i]["score"]).toFixed(2) +
//                 "%"
//             ).width;
//             const textHeight = parseInt(font, 10); // base 10
//             ctx.fillRect(x, y, textWidth + 4, textHeight + 4);

//             if (i < limitLen) {
//               // store confidence score
//               _faceDetectionScores[i] = detections[i].score;
//             }
//             i++;
//           }

//           detections.forEach((item) => {
//             const x = item["bbox"][0];
//             const y = item["bbox"][1];

//             // Draw the text last to ensure it's on top.
//             ctx.fillStyle = "#000000";
//             ctx.fillText(
//               item["label"] + " " + (100 * item["score"]).toFixed(2) + "%",
//               x,
//               y
//             );
//           });

//           // crop resize
//           const boxes = detections.map((value, index) => value.cropBox);
//           const boxInd = detections.map((value, index) => value.boxInd);
//           const stackImgTensor = tf.tidy(() => {
//             return rawImgTensor
//               .tile([boxInd.length, 1, 1])
//               .reshape([-1, rawImgTensor.shape[0], rawImgTensor.shape[1], 3]);
//           });

//           const resizeImgsTensor = tf.tidy(() => {
//             return tf.image.cropAndResize(
//               stackImgTensor,
//               boxes,
//               boxInd,
//               [224, 224],
//               "bilinear"
//             );
//           });
//           const _maskConfidenceScoresTensor =
//             classificationModel.execute(resizeImgsTensor);
//           const _allMaskClassificationScores =
//             _maskConfidenceScoresTensor.arraySync();

//           i = 0;
//           len = multiFaceCanvasRef.current.length;
//           limitLen = detections.length;
//           const returnedImgs = resizeImgsTensor.arraySync();
//           while (i < len) {
//             if (i < limitLen) {
//               let faceCanvas = multiFaceCanvasRef.current[i];
//               tf.browser.toPixels(returnedImgs[i], faceCanvas);

//               _maskClassificationScores[i] =
//                 _allMaskClassificationScores[i][0];
//             } else {
//               const context = multiFaceCanvasRef.current[i].getContext("2d");
//               context.clearRect(
//                 0,
//                 0,
//                 multiFaceCanvasRef.current[i].width,
//                 multiFaceCanvasRef.current[i].height
//               );
//             }
//             i++;
//           }
//           // dispose all tensor variables
//           stackImgTensor.dispose();
//           resizeImgsTensor.dispose();
//           _maskConfidenceScoresTensor.dispose();
//         } else {
//           // clear bounding box canvas
//           ctx.clearRect(0, 0, expectedWebCamHeight, expectedWebCamWidth);

//           // clear face render canvas
//           i = 0;
//           len = multiFaceCanvasRef.current.length;
//           while (i < len) {
//             const context = multiFaceCanvasRef.current[i].getContext("2d");
//             context.clearRect(
//               0,
//               0,
//               multiFaceCanvasRef.current[i].width,
//               multiFaceCanvasRef.current[i].height
//             );
//             i++;
//           }
//         }

//         setFaceDetectionScores(_faceDetectionScores);
//         setMaskClassificationScores(_maskClassificationScores);
//         // let endTime = performance.now();
//         // console.log(`Took ${endTime - startTime} milliseconds`);
//         return res;
//       })
//       .then((res) => {
//         let i = 0;
//         const len = res.length;
//         while (i < len) {
//           tf.dispose(res[i]);
//           i++;
//         }
//       })
//       .finally(() => {
//         tf.dispose(rawImgTensor);
//         tf.dispose(inputTensor);
//       });
//     console.log(`numTensors: ${tf.memory().numTensors}`);
//     // tf.engine().endScope();
//   }
// };
