// Implementing Transfer learning from scratch

import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';

const recordButtons = document.getElementsByClassName('record-button');
const buttonsContainer = document.getElementById('add-samples-container');
const trainButton = document.getElementById('train');
const predictButton = document.getElementById('predict');
const stopButton = document.getElementById('stop');
const statusElement = document.getElementById('status');
const webcamElement = document.getElementById('webcam');
const controllerElement = document.getElementById('controller');
const predictionElement = document.getElementById('prediction');

let webcam, initianModel, mouseDown, newModel;

const totals = [0, 0];
const labels = ['left', 'right'];
const learningRate = 0.0001;
const batchSizeFraction = 0.4;
const epochs = 30;
const denseUnits = 100;

let isTrainning = false;
let isPredicting = false;

const loadModel = async () => {
    const mobilenet = await tf.loadLayersModel(
        "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"
    );

    const layer = mobilenet.getLayer('conv_pw_13_relu');

    return tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output
    });
}

const init = async () => {
   webcam = await tfd.webcam(webcamElement);
   initianModel = await loadModel();
   statusElement.style.display = 'none';
   controllerElement.style.display = 'block';
}

init();

buttonsContainer.onmousedown = (e) => {
    if (e.target === recordButtons[0]) {
        handleAddExample(0);
        return;
    }
    handleAddExample(1);
}

buttonsContainer.onmouseup = (e) => {
    mouseDown = false;
}

const handleAddExample = async (labelIndex) => {
    mouseDown = true;
    const total = document.getElementById(`${labels[labelIndex]}-total`);
    while (mouseDown) {
        addExample(labelIndex);
        total.innerText = ++totals[labelIndex];

        await tf.nextFrame();
    }
}

let xs; // example data
let xy; // labels attached with example data
const addExample = async (labelIndex) => {
    const img = await getImage();
    let example = initianModel.predict(img);

    const y = tf.tidy(() => {
        return tf.oneHot(tf.tensor1d([labelIndex]).toInt(), labels.length);
    });

    if (xs == null) {
        xs = tf.keep(example);
        xy = tf.keep(y);
        return;
    }
    const previousXs = xs;
    xs = tf.keep(previousXs.concat(example, 0));
    const previousYs = xy;
    xy = tf.keep(previousYs.concat(y, 0));
    previousXs.dispose();
    previousYs.dispose();
    y.dispose();
    img.dispose();
}

const getImage = async () => {
    const img = await webcam.capture();
    const processedImg = tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
    img.dispose();
    return processedImg;
}

trainButton.onclick = async () => {
    train();
    statusElement.style.display = 'block';
    statusElement.innerText = 'Training...';
}

const train = () => {
    isTrainning = true;
    if(!xs) {
        throw new Error('No data to train');
    }

    // create a tensorflow js model
    newModel = tf.sequential({
        layers: [
            tf.layers.flatten({ inputShape: initianModel.outputs[0].shape.slice(1) }),
            tf.layers.dense({
                units: denseUnits,
                activation: 'relu',
                kernelInitializer: 'varianceScaling',
                useBias: true,
            }),
            tf.layers.dense({
                units: labels.length,
                kernelInitializer: 'varianceScaling',
                useBias: true,
                activation: 'softmax'
            })
        ]
    });

    const optimizer = tf.train.adam(learningRate);

    newModel.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const batchSize = Math.floor(xs.shape[0] * batchSizeFraction);
    newModel.fit(xs, xy, {
        batchSize,
        epochs,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                statusElement.innerText = `Loss: ${logs.loss.toFixed(5)}`;
            },
            onTrainEnd: () => {
                isTrainning = false;
                statusElement.innerText = 'Trained!';
                predictButton.removeAttribute('disabled');
                stopButton.removeAttribute('disabled');
            }
        }
    });
}

predictButton.onclick = async () => {
    isPredicting = true;
    while (isPredicting) {
        const img = await getImage();
        const initialModelPredictions = initianModel.predict(img);
        const predictions = newModel.predict(initialModelPredictions);
        const predictedLabel = predictions.argMax(1).dataSync()[0];
        showPrediction(labels[predictedLabel])
        img.dispose();
        await tf.nextFrame();
    }
}

stopButton.onclick = async () => {
    isPredicting = false;
    predictionElement.innerText = '⛔';
}

const showPrediction = (label) => {
    predictionElement.innerText = label === 'right' ? '👉' : '👈';
}