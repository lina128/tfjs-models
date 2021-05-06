/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';

import {Context} from './camera';
import {setupDatGui} from './option_panel';
import {STATE} from './params';
import {setBackendAndEnvFlags} from './util';

let detector, camera;

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: {width: 500, height: 500},
        multiplier: 0.75
      });
    case posedetection.SupportedModels.MediapipeBlazepose:
      return posedetection.createDetector(
          STATE.model, {quantBytes: 4, lite: false});
    case posedetection.SupportedModels.MoveNet:
      const modelType = STATE.modelConfig.type == 'lightning' ?
          posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING :
          posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      return posedetection.createDetector(STATE.model, {modelType});
  }
}

async function checkGuiUpdate() {
  if (STATE.isModelChanged) {
    detector.dispose();
    detector = await createDetector(STATE.model);
    STATE.isModelChanged = false;
  }

  if (STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;
    detector.dispose();
    await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    detector = await createDetector(STATE.model);
    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

async function renderResult(imgTensor) {
  const poses = await detector.estimatePoses(imgTensor, {
    maxPoses: STATE.modelConfig.maxPoses,
    flipHorizontal: false,
    enableSmoothing: false
  });

  camera.drawCtx();

  console.log(poses);

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old
  // model, which shouldn't be rendered.
  if (poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);
  }
}

async function checkUpdate() {
  await checkGuiUpdate();

  requestAnimationFrame(checkUpdate);
};

async function updateVideo(event) {
  const file = event.target.files[0];

  const image = new Image();
  image.src = URL.createObjectURL(file);
  const parent = document.getElementById('canvas-wrapper');
  parent.appendChild(image);
  camera.setImage(image)
}

async function run() {
  const canvas = document.createElement('canvas');
  const parent = document.getElementById('canvas-wrapper');
  parent.appendChild(canvas);
  camera.setCanvas(canvas);
  canvas.height = camera.image.height;
  canvas.width = camera.image.width;

  const imgTensor = tf.browser.fromPixels(camera.image);

  for (let i = 0; i < 1; i++) {
    await renderResult(imgTensor);
  }
}

async function app() {
  await tf.setBackend(STATE.backend);

  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  await setupDatGui(urlParams);

  detector = await createDetector();
  camera = new Context();

  const runButton = document.getElementById('submit');
  runButton.onclick = run;

  const uploadButton = document.getElementById('videofile');
  uploadButton.onchange = updateVideo;

  checkUpdate();
};

app();
