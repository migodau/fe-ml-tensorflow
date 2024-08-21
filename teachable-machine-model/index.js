// Trainning model with Teachable machine model

const path = './model/';
const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const webcamContainer = document.getElementById('webcam-container');
const labelEl = document.getElementById('label-container');

let item = document.getElementById('item');

startButton.onclick = () => init();
stopButton.onclick = () => stopLoop();

let model, webcam, stop;

const init = async () => {
    stop = false;
    console.log("Loading model...");
    const modelPath = path + 'model.json';
    const metadataPath = path + 'metadata.json';
    model = await tmImage.load(modelPath, metadataPath);

    // const maxPredictions = model.getTotalClasses();

    webcam = new tmImage.Webcam(200, 200, true);

    console.log("Loading webcam...");
    await webcam.setup();
    await webcam.play();
    window.requestAnimationFrame(loop);

    console.log("Webcam ready!");
    webcamContainer.appendChild(webcam.canvas);
    item.innerText = '🐥'
}

const stopLoop = () => {
    stop = true;
}

const loop = async () => {
    webcam.update();
    await predict();
    if (stop) {
        clearUp();
        return;
    }
    window.requestAnimationFrame(loop);
}

const predict = async () => {
    const predictions = await model.predict(webcam.canvas);
    
    predictions.forEach(element => {
        if (element.probability.toFixed(2) > 0.8) {
            element.className === 'right' ? moveRight() : moveLeft();
            labelEl.innerText = element.className === 'right' ? '👉' : '👈';
            return;
        }
        if (element.probability.toFixed(2) > 0.5) {
            labelEl.innerText = '✋';
        }
    })
}



let moveRight = () => {
    item.style.position = 'absolute'
    const currentpos = item.style.left.match(/\d+/g);
    item.style.left = currentpos ? +currentpos[0] + 1 + 'px' : '1px'
 }

 let moveLeft = () => {
    const currentpos = item.style.left.match(/\d+/g);
    item.style.left = currentpos ? +currentpos[0] - 1 + 'px' : '1px'
 }

const clearUp = () => {
    labelEl.innerText = '';
    webcamContainer.innerHTML = ''
    item.innerText = '';
    webcam.stop();
}
