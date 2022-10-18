import FormControl from "@mui/material/FormControl";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormGroup from "@mui/material/FormGroup";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Switch from "@mui/material/Switch";
import * as tf from "@tensorflow/tfjs";
import React, { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import "./App.css";
import { styles } from "./styles";
import * as utils from "./utils";

const webCamRatios = [16 / 9, 4 / 3];
const verticalPanelSplitPercentage = 65;
const horizontalPanelSplitPercentage = 70;
const maxNumOfDisplayFace = 5;

const face_detection_model_url = `${process.env.PUBLIC_URL}/tfjs/face_detection/model.json`;
const face_mask_classification_model_url =
  `${process.env.PUBLIC_URL}/tfjs/face_mask_classification/model.json`;
const test_img_url = `${process.env.PUBLIC_URL}/test_imgs/mask_01.jpg`;

function App() {
  const webcamRef = useRef(null);
  const multiFaceCanvasRef = useRef(Array(maxNumOfDisplayFace).fill(0));
  const [faceDetectorProgress, setFaceDetectorProgress] = useState(0);
  const [maskClassifierProgress, setMaskClassifierProgress] = useState(0);
  const [faceDetectionScores, setFaceDetectionScores] = useState([]);
  const [maskClassificationScores, setMaskClassificationScores] = useState([]);
  const [detectionModel, setDetectionModel] = useState(null);
  const [classificationModel, setClassificationModel] = useState(null);
  const [testImgDim, setTestImgDim] = useState({});
  const [isUsingWebcam, setIsUsingWebcam] = useState(false);
  const [selectedWebCamInfo, setSelectedWebCamInfo] = useState({
    deviceName: "",
    deviceId: "",
  });
  const [availableWebCamInfos, setAvailableWebCamInfos] = useState([]);
  const { height: windowHeight, width: windowWidth } =
    utils.useWindowDimensions();

  const {
    webCamPanelWidth,
    webCamPanelHeight,
    expectedWebCamWidth,
    expectedWebCamHeight,
  } = utils.calcExpectedWebCamDim(
    windowHeight,
    windowWidth,
    verticalPanelSplitPercentage,
    horizontalPanelSplitPercentage,
    webCamRatios
  );
  const oriImgWidth =
    testImgDim.oriImgWidth !== undefined ? testImgDim.oriImgWidth : 0;
  const oriImgHeight =
    testImgDim.oriImgHeight !== undefined ? testImgDim.oriImgHeight : 1;
  const { expectedImgWidth, expectedImgHeight } = utils.calcExpectedImgDim(
    oriImgWidth,
    oriImgHeight,
    webCamPanelWidth,
    webCamPanelHeight
  );
  const faceRenderCanvasHeight =
    ((windowHeight * (100 - verticalPanelSplitPercentage)) / 100) * 0.5;

  /**
   * generate dynamic videoConstraintsValue
   */
  const videoConstraints = {
    height: 720,
    width: 1280,
    // facingMode: "environment",
    deviceId: selectedWebCamInfo.deviceId,
  };

  const loadModels = async () => {
    /** @type {tf.GraphModel} */
    if (detectionModel !== null) {
      detectionModel.dispose();
    }
    if (classificationModel !== null) {
      classificationModel.dispose();
    }
    const loadedDetectionModel = await tf.loadGraphModel(
      face_detection_model_url,
      {
        onProgress: (fractions) => {
          setFaceDetectorProgress(fractions);
          // console.log(fractions);
        },
      }
    );
    const loadedClassificationModel = await tf.loadGraphModel(
      face_mask_classification_model_url,
      {
        onProgress: (fractions) => {
          setMaskClassifierProgress(fractions);
          // console.log(fractions);
        },
      }
    );
    setDetectionModel(loadedDetectionModel);
    setClassificationModel(loadedClassificationModel);
    console.log("Model loaded.");
    return [loadedDetectionModel, loadedClassificationModel];
  };

  const onInferencing = async (detectionModel, classificationModel) => {
    if (detectionModel == null || classificationModel == null) {
      return;
    }
    let cnvs;
    let inputData;
    let drawHeight;
    let drawWidth;
    let captureHeight;
    let captureWidth;

    // either using webcam or test image
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      /** @type {HTMLVideoElement} */
      inputData = webcamRef.current.video;
      cnvs = document.getElementById("myCanvas");

      // Set video width
      webcamRef.current.video.width = expectedWebCamWidth;
      webcamRef.current.video.height = expectedWebCamHeight;

      cnvs.width = expectedWebCamWidth;
      cnvs.height = expectedWebCamHeight;

      // dimension variables
      drawHeight = expectedWebCamHeight;
      drawWidth = expectedWebCamWidth;
      captureHeight = 720;
      captureWidth = 1280;
    } else if (!isUsingWebcam) {
      /** @type {HTMLImageElement} */
      inputData = document.getElementById("test_img");
      cnvs = document.getElementById("test_canvas");

      // force resize bounding box canvas
      cnvs.width = expectedImgWidth;
      cnvs.height = expectedImgHeight;

      // dimension variables
      drawHeight = expectedImgHeight;
      drawWidth = expectedImgWidth;
      captureHeight = expectedImgHeight;
      captureWidth = expectedImgWidth;
    } else {
      return;
    }

    // tf.engine().startScope();
    const ctx = cnvs.getContext("2d");

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    /** @type {tf.Tensor3D} */
    const rawImgTensor = tf.browser.fromPixels(inputData);
    const inputTensor = tf.tidy(() => {
      return rawImgTensor.transpose([0, 1, 2]).expandDims();
    });
    // let startTime = performance.now();
    detectionModel
      .executeAsync(inputTensor)
      .then((res) => {
        // const a0 = res[0].arraySync(); // num_detection
        // const a1 = res[1].arraySync(); // raw_detection_boxes
        // const a2 = res[2].arraySync(); // detection_anchor_indices
        // const a3 = res[3].arraySync(); // raw_detection_scores
        // const a4 = res[4].arraySync(); // detection_boxes
        // const a5 = res[5].arraySync(); // detection_classes
        // const a6 = res[6].arraySync(); // detection_scores
        // const a7 = res[7].arraySync(); // detection_multiclass_scores
        const detection_boxes = res[4].arraySync();
        const detection_scores = res[6].dataSync();

        const detections = utils.buildDetectedObjects(
          detection_scores,
          0.5,
          detection_boxes,
          drawHeight,
          drawWidth,
          captureHeight,
          captureWidth
        );

        // prepare variables
        let i;
        let len;
        let limitLen;
        const _faceDetectionScores = Array(maxNumOfDisplayFace).fill(null);
        const _maskClassificationScores = Array(maxNumOfDisplayFace).fill(null);

        if (detections.length > 0) {
          // loop for rendering bounding boxes
          i = 0;
          len = detections.length;
          limitLen = maxNumOfDisplayFace;

          // clear bounding box canvas
          ctx.clearRect(0, 0, expectedWebCamHeight, expectedWebCamWidth);
          while (i < len) {
            const x = detections[i]["bbox"][0];
            const y = detections[i]["bbox"][1];
            const width = detections[i]["bbox"][2];
            const height = detections[i]["bbox"][3];

            // Draw the bounding box.
            ctx.strokeStyle = "#00FFFF";
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, width, height);

            if (i < limitLen) {
              // store confidence score
              _faceDetectionScores[i] = detections[i].score;
            }
            i++;
          }

          // crop resize
          const boxes = detections.map((value, index) => value.cropBox);
          const boxInd = detections.map((value, index) => value.boxInd);

          const resizeImgsTensor = tf.tidy(() => {
            const stackImgTensor = rawImgTensor
              .tile([boxInd.length, 1, 1])
              .reshape([-1, rawImgTensor.shape[0], rawImgTensor.shape[1], 3]);
            return tf.image.cropAndResize(
              stackImgTensor,
              boxes,
              boxInd,
              [224, 224],
              "bilinear"
            );
          });

          const _allMaskClassificationScores = tf.tidy(() => {
            return classificationModel.execute(resizeImgsTensor).arraySync();
          });

          // render faces
          i = 0;
          len = multiFaceCanvasRef.current.length;
          limitLen = detections.length;
          const returnedImgs = resizeImgsTensor.arraySync();
          while (i < len) {
            if (i < limitLen) {
              let faceCanvas = multiFaceCanvasRef.current[i];
              tf.browser.toPixels(returnedImgs[i], faceCanvas);

              _maskClassificationScores[i] = _allMaskClassificationScores[i][0];
            } else {
              const context = multiFaceCanvasRef.current[i].getContext("2d");
              context.clearRect(
                0,
                0,
                multiFaceCanvasRef.current[i].width,
                multiFaceCanvasRef.current[i].height
              );
            }
            i++;
          }
          // dispose all tensor variables
          resizeImgsTensor.dispose();
        } else {
          // clear bounding box canvas
          ctx.clearRect(0, 0, expectedWebCamHeight, expectedWebCamWidth);

          // clear face render canvas
          i = 0;
          len = multiFaceCanvasRef.current.length;
          while (i < len) {
            const context = multiFaceCanvasRef.current[i].getContext("2d");
            context.clearRect(
              0,
              0,
              multiFaceCanvasRef.current[i].width,
              multiFaceCanvasRef.current[i].height
            );
            i++;
          }
        }

        setFaceDetectionScores(_faceDetectionScores);
        setMaskClassificationScores(_maskClassificationScores);
        // let endTime = performance.now();
        // console.log(`Took ${endTime - startTime} milliseconds`);
        return res;
      })
      .then((res) => {
        let i = 0;
        const len = res.length;
        while (i < len) {
          tf.dispose(res[i]);
          i++;
        }
      })
      .finally(() => {
        tf.dispose(rawImgTensor);
        tf.dispose(inputTensor);
      });
    // console.log(`numTensors: ${tf.memory().numTensors}`);
    // tf.engine().endScope();
  };

  /* 
    Run only once
     */
  useEffect(() => {
    tf.ready()
      .then((_) => {
        tf.enableProdMode();
        console.log("tfjs is ready");
      })
      .then(loadModels);
  }, []);

  useEffect(() => {
    if (detectionModel !== null && classificationModel !== null) {
      if (isUsingWebcam) {
        const captureInterval = setInterval(
          onInferencing,
          100,
          detectionModel,
          classificationModel
        );
        return () => clearInterval(captureInterval);
      } else {
        console.log("Run on test img");
        onInferencing(detectionModel, classificationModel);
      }
    }
  }, [
    classificationModel,
    detectionModel,
    windowHeight,
    windowWidth,
    isUsingWebcam,
    testImgDim,
  ]);

  useEffect(() => {
    if (!isUsingWebcam) {
      const img = new Image();
      img.src = test_img_url;
      img.onload = () => {
        setTestImgDim({
          oriImgHeight: img.naturalHeight,
          oriImgWidth: img.naturalWidth,
        });
      };
    }
  }, [isUsingWebcam]);

  useEffect(() => {
    setFaceDetectionScores(Array(maxNumOfDisplayFace).fill(null));
    setMaskClassificationScores(Array(maxNumOfDisplayFace).fill(null));
  }, [maxNumOfDisplayFace]);

  useEffect(() => {
    utils.getWebCamInfos().then((webCamInfos) => {
      setAvailableWebCamInfos(webCamInfos);
      if (webCamInfos.length > 0) {
        setSelectedWebCamInfo(webCamInfos[0]);
      }
    });
  }, [isUsingWebcam]);

  const webCamToggle = () => {
    return (
      <Switch
        checked={isUsingWebcam}
        onChange={(event) => {
          setIsUsingWebcam(event.target.checked);
        }}
        inputProps={{ "aria-label": "controlled" }}
      />
    );
  };

  const videoInputSelector = () => {
    return (
      <Select
        labelId="demo-simple-select-autowidth-label"
        id="demo-simple-select-autowidth"
        value={selectedWebCamInfo.deviceId}
        onChange={(event) => {
          setSelectedWebCamInfo({
            deviceName: event.target.name,
            deviceId: event.target.value,
          });
        }}
        autoWidth
        label="Video Input"
      >
        {availableWebCamInfos.length > 0 ? (
          availableWebCamInfos.map((value, index) => {
            return (
              <MenuItem key={index} value={value.deviceId}>
                {value.deviceName}
              </MenuItem>
            );
          })
        ) : (
          <MenuItem value="">{""}</MenuItem>
        )}
      </Select>
    );
  };

  const faceMaskPanel = () => {
    if (detectionModel == null || classificationModel == null) {
      const faceDetectionModelProgress = `Face Detection Model - ${
        faceDetectorProgress * 100
      }%`;
      const maskClassifierModelProgress = `Mask Classifier Model - ${
        maskClassifierProgress * 100
      }%`;
      return (
        <div>
          <div>{"Downloading models...."}</div>
          <div>{faceDetectionModelProgress}</div>
          <div>{maskClassifierModelProgress}</div>
        </div>
      );
    } else if (multiFaceCanvasRef.current.length < 1) {
      return <div>{"Warming up...."}</div>;
    }
    return multiFaceCanvasRef.current.map((item, i) => {
      let labelName = "";
      let faceScoreLabel = "";
      let maskScoreLabel = "";
      if (
        faceDetectionScores[i] !== null &&
        faceDetectionScores[i] !== undefined &&
        maskClassificationScores[i] !== null &&
        maskClassificationScores[i] !== undefined
      ) {
        labelName = maskClassificationScores[i] > 0 ? "No Mask" : "Mask";
        faceScoreLabel = `Face: ${faceDetectionScores[i].toFixed(2)}`;
        maskScoreLabel = `Mask: ${maskClassificationScores[i].toFixed(2)}`;
      }
      return (
        <div key={i} style={styles.faceMaskContainer}>
          <div>{labelName}</div>
          <canvas
            ref={(el) => (multiFaceCanvasRef.current[i] = el)}
            height={faceRenderCanvasHeight}
            width={faceRenderCanvasHeight}
            style={{
              height: faceRenderCanvasHeight,
              width: faceRenderCanvasHeight,
            }}
          />
          <div>{faceScoreLabel}</div>
          <div>{maskScoreLabel}</div>
        </div>
      );
    });
  };

  return (
    <div className="App" style={styles.root}>
      <div
        style={{
          ...styles.webCamParentPanel,
          flexBasis: `${verticalPanelSplitPercentage}%`,
        }}
      >
        <div
          style={{
            ...styles.webCamPanel,
            flexBasis: `${horizontalPanelSplitPercentage}%`,
          }}
        >
          {isUsingWebcam && (
            <div style={{}}>
              <Webcam
                audio={false}
                style={{
                  ...styles.webCamStackItem,
                  height: expectedWebCamHeight,
                  width: expectedWebCamWidth,
                }}
                id="img"
                ref={webcamRef}
                height={expectedWebCamHeight}
                width={expectedWebCamWidth}
                screenshotQuality={1}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
              />
              <canvas
                id="myCanvas"
                height={expectedWebCamHeight}
                width={expectedWebCamWidth}
                style={{
                  ...styles.webCamStackItem,
                  zIndex: 9999,
                  backgroundColor: "transparent",
                }}
              />
            </div>
          )}
          {!isUsingWebcam && (
            <div style={{}}>
              <img
                id="test_img"
                src={test_img_url}
                style={{
                  ...styles.webCamStackItem,
                  height: expectedImgHeight,
                  width: expectedImgWidth,
                }}
                height={expectedImgHeight}
                width={expectedImgWidth}
                alt=""
              />
              <canvas
                id="test_canvas"
                height={expectedImgHeight}
                width={expectedImgWidth}
                style={{
                  ...styles.webCamStackItem,
                  zIndex: 9999,
                  backgroundColor: "transparent",
                }}
              />
            </div>
          )}
        </div>
        <div
          style={{
            ...styles.settingPanel,
            flexBasis: `${100 - horizontalPanelSplitPercentage}%`,
          }}
        >
          <FormGroup>
            <FormControl sx={{ m: 1, minWidth: 220 }} size="small">
              <InputLabel id="demo-simple-select-autowidth-label">
                Video Input
              </InputLabel>
              {videoInputSelector()}
            </FormControl>
            <FormControlLabel
              control={webCamToggle()}
              label="Webcam"
              labelPlacement="start"
            />
          </FormGroup>
        </div>
      </div>
      <div
        style={{
          ...styles.faceMaskPanel,
          flexBasis: `${100 - verticalPanelSplitPercentage}%`,
        }}
      >
        {faceMaskPanel()}
      </div>
    </div>
  );
}

export default App;
