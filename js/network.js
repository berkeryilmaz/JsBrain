export class NetworkBuilder {
    constructor() {
        this.model = null;
        this.layers = []; 
        this.isTraining = false;
        this.stopRequested = false;
        this.taskType = 'regression'; // 'regression' or 'classification'
    }

    addLayer(config) {
        this.layers.push(config);
    }

    removeLayer(index) {
        if (index >= 0 && index < this.layers.length) {
            this.layers.splice(index, 1);
        }
    }
    
    getLayers() {
        return this.layers;
    }

    buildModel(inputShape, outputShape, lr, isClassification = false) {
        if (this.model) {
            this.model.dispose(); 
        }

        this.taskType = isClassification ? 'classification' : 'regression';
        const outputActivation = isClassification ? 'softmax' : 'linear';

        this.model = tf.sequential();
        
        if (this.layers.length === 0) {
            this.model.add(tf.layers.dense({
                units: outputShape,
                inputShape: [inputShape],
                activation: outputActivation
            }));
        } else {
            let isFirst = true;

            for (let l of this.layers) {
                const layerType = l.type || 'dense'; // Default backwards compatibility
                
                if (layerType === 'dense') {
                    this.model.add(tf.layers.dense({
                        units: l.units,
                        activation: l.activation,
                        ...(isFirst ? { inputShape: [inputShape] } : {})
                    }));
                } else if (layerType === 'dropout') {
                    this.model.add(tf.layers.dropout({
                        rate: l.rate,
                        ...(isFirst ? { inputShape: [inputShape] } : {})
                    }));
                }
                
                isFirst = false;
            }
            
            this.model.add(tf.layers.dense({
                units: outputShape,
                activation: outputActivation
            }));
        }

        const optimizer = tf.train.adam(lr);
        const loss = isClassification ? 'categoricalCrossentropy' : 'meanSquaredError';
        const metrics = isClassification ? ['accuracy'] : [];

        this.model.compile({
            optimizer: optimizer,
            loss: loss,
            metrics: metrics
        });

        return this.model;
    }

    async train(tensors, epochs, batchSize, callbacks) {
        this.isTraining = true;
        this.stopRequested = false;

        if (!this.model) throw new Error("Model not built yet");

        // CRITICAL PERFORMANCE: Single model.fit() call with ALL epochs.
        // Previously we called model.fit(epochs:1) in a JS for-loop N times.
        // Each `await model.fit()` yields to the browser event loop (~1-2ms per yield).
        // With 10K epochs → 10-20 seconds of PURE overhead from context switches.
        //
        // By using a single model.fit(epochs:N) + yieldEvery:'never',
        // TF.js runs the entire training loop in C++/WebGL with zero JS overhead.
        // We yield manually in onEpochEnd every ~100ms for UI responsiveness.

        let lastYieldTime = performance.now();
        const YIELD_INTERVAL = 150; // ms between browser yields

        const config = {
            epochs: epochs,
            shuffle: true,
            yieldEvery: 'never',  // MAXIMUM SPEED — no auto-yielding
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    if (this.stopRequested) {
                        this.model.stopTraining = true;
                    }

                    if (callbacks.onEpoch) {
                        callbacks.onEpoch(epoch, logs);
                    }

                    // Yield to browser ONLY when needed for UI responsiveness
                    const now = performance.now();
                    if (now - lastYieldTime > YIELD_INTERVAL) {
                        await tf.nextFrame();
                        lastYieldTime = performance.now();
                    }
                    // Otherwise: return undefined = no yield = C++ loop continues at full speed
                },
                onTrainEnd: () => {
                    this.isTraining = false;
                    if (callbacks.onEnd) callbacks.onEnd();
                }
            }
        };

        if (batchSize !== "null" && batchSize && !isNaN(parseInt(batchSize))) {
            config.batchSize = parseInt(batchSize);
        }

        if (tensors.xVal && tensors.yVal && tensors.xVal.shape[0] > 0) {
            config.validationData = [tensors.xVal, tensors.yVal];
        }

        try {
            await this.model.fit(tensors.xTrain, tensors.yTrain, config);
        } catch (e) {
            this.isTraining = false;
            console.error(e);
            throw e;
        }
    }

    stopTraining() {
        if (this.isTraining) {
            this.stopRequested = true;
        }
    }

    /**
     * Evaluate model performance on given data.
     * Returns different metrics depending on taskType.
     * For classification: accuracy, per-class precision/recall, confusion matrix
     * For regression: R², MAE, RMSE, per-target stats
     */
    evaluateModel(xTensor, yTensor) {
        if (!this.model) throw new Error("No model available.");

        return tf.tidy(() => {
            const predictions = this.model.predict(xTensor);
            const yData = yTensor.arraySync();
            const predData = predictions.arraySync();
            const numSamples = yData.length;

            if (this.taskType === 'classification') {
                const numClasses = yData[0].length;

                // Build confusion matrix 
                const confMatrix = Array.from({ length: numClasses }, () => new Array(numClasses).fill(0));
                let correct = 0;

                for (let i = 0; i < numSamples; i++) {
                    const trueIdx = yData[i].indexOf(Math.max(...yData[i]));
                    let predIdx = 0;
                    let predMax = predData[i][0];
                    for (let c = 1; c < numClasses; c++) {
                        if (predData[i][c] > predMax) {
                            predMax = predData[i][c];
                            predIdx = c;
                        }
                    }
                    confMatrix[trueIdx][predIdx]++;
                    if (trueIdx === predIdx) correct++;
                }

                const accuracy = correct / numSamples;

                // Per-class precision & recall
                const perClass = [];
                for (let c = 0; c < numClasses; c++) {
                    const tp = confMatrix[c][c];
                    let predTotal = 0, trueTotal = 0;
                    for (let j = 0; j < numClasses; j++) {
                        predTotal += confMatrix[j][c]; // column sum
                        trueTotal += confMatrix[c][j]; // row sum
                    }
                    perClass.push({
                        precision: predTotal > 0 ? tp / predTotal : 0,
                        recall: trueTotal > 0 ? tp / trueTotal : 0,
                        support: trueTotal
                    });
                }

                return {
                    type: 'classification',
                    accuracy,
                    numSamples,
                    numClasses,
                    confusionMatrix: confMatrix,
                    perClass
                };
            } else {
                // Regression metrics per target
                const numTargets = yData[0].length;
                const targetMetrics = [];

                for (let t = 0; t < numTargets; t++) {
                    let sumErr = 0, sumAbsErr = 0, sumSqErr = 0;
                    let sumY = 0;

                    for (let i = 0; i < numSamples; i++) {
                        const err = predData[i][t] - yData[i][t];
                        sumErr += err;
                        sumAbsErr += Math.abs(err);
                        sumSqErr += err * err;
                        sumY += yData[i][t];
                    }

                    const mae = sumAbsErr / numSamples;
                    const mse = sumSqErr / numSamples;
                    const rmse = Math.sqrt(mse);
                    const meanY = sumY / numSamples;

                    let ssTot = 0;
                    for (let i = 0; i < numSamples; i++) {
                        ssTot += (yData[i][t] - meanY) ** 2;
                    }
                    const r2 = ssTot > 0 ? 1 - (sumSqErr / ssTot) : 0;

                    targetMetrics.push({ mae, rmse, r2, mse });
                }

                // Averaged across targets
                const avgMae = targetMetrics.reduce((s, m) => s + m.mae, 0) / numTargets;
                const avgRmse = targetMetrics.reduce((s, m) => s + m.rmse, 0) / numTargets;
                const avgR2 = targetMetrics.reduce((s, m) => s + m.r2, 0) / numTargets;

                return {
                    type: 'regression',
                    numSamples,
                    numTargets,
                    mae: avgMae,
                    rmse: avgRmse,
                    r2: avgR2,
                    perTarget: targetMetrics
                };
            }
        });
    }

    /**
     * Single-row prediction. inputArray is a flat array of normalized input values.
     * Returns an array of output values.
     */
    predict(inputArray) {
        if (!this.model) throw new Error("No model available. Train or load a model first.");
        const inputTensor = tf.tensor2d([inputArray], [1, inputArray.length]);
        const outputTensor = this.model.predict(inputTensor);
        const result = Array.from(outputTensor.dataSync());
        inputTensor.dispose();
        outputTensor.dispose();
        return result;
    }

    /**
     * Batch prediction. inputMatrix is array of arrays (rows x features).
     * Returns array of arrays (rows x targets).
     */
    predictBatch(inputMatrix) {
        if (!this.model) throw new Error("No model available. Train or load a model first.");
        if (inputMatrix.length === 0) return [];
        const inputTensor = tf.tensor2d(inputMatrix, [inputMatrix.length, inputMatrix[0].length]);
        const outputTensor = this.model.predict(inputTensor);
        const outputData = outputTensor.arraySync();
        inputTensor.dispose();
        outputTensor.dispose();
        return outputData;
    }

    /**
     * Extract weights for visualization (SelfDrivingCar pattern).
     * Returns array of { data: Float32Array, shape: [in, out] } per layer.
     */
    extractWeightsForVis() {
        if (!this.model) return [];
        return tf.tidy(() => {
            const weights = [];
            for (let i = 0; i < this.model.layers.length; i++) {
                const layerWeights = this.model.layers[i].getWeights();
                if (layerWeights.length > 0) {
                    const kernelData = layerWeights[0].dataSync();
                    const shape = layerWeights[0].shape;
                    weights.push({
                        data: Array.from(kernelData),
                        shape: shape
                    });
                }
            }
            return weights;
        });
    }

    async saveModel() {
        if (!this.model) throw new Error("No model to save.");
        
        await this.model.save(tf.io.withSaveHandler(async (artifacts) => {
            const modelTopologyAndWeightManifest = {
                modelTopology: artifacts.modelTopology,
                format: artifacts.format,
                generatedBy: artifacts.generatedBy,
                convertedBy: artifacts.convertedBy,
                weightsManifest: [{
                    paths: ['./jsbrain-model.weights.bin'],
                    weights: artifacts.weightSpecs
                }]
            };

            const zip = new JSZip();
            zip.file("jsbrain-model.json", JSON.stringify(modelTopologyAndWeightManifest));
            
            if (artifacts.weightData) {
                zip.file("jsbrain-model.weights.bin", artifacts.weightData);
            }

            const blob = await zip.generateAsync({type: "blob"});
            
            // FileSaver.js handles Mac/Safari blob UUID loss bug natively
            if (window.saveAs) {
                window.saveAs(blob, "jsbrain-model.zip");
            } else {
                // Fallback if FileSaver fails to load
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'jsbrain-model.zip';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                setTimeout(() => URL.revokeObjectURL(url), 1000);
            }
            
            return {
                modelArtifactsInfo: {
                    dateSaved: new Date(),
                    modelTopologyType: 'JSON'
                }
            };
        }));
    }

    async loadModel(files) {
        if (!files || files.length < 2) {
            throw new Error("You MUST select BOTH the .json AND the .bin files together!");
        }
        
        if (this.model) {
            this.model.dispose();
        }

        // browserFiles expects an array or FileList.
        this.model = await tf.loadLayersModel(tf.io.browserFiles(files));
        
        // Detect task type from output activation
        const lastLayer = this.model.layers[this.model.layers.length - 1];
        const lastConfig = lastLayer.getConfig ? lastLayer.getConfig() : {};
        const outputAct = lastConfig.activation || 'linear';
        this.taskType = (outputAct === 'softmax') ? 'classification' : 'regression';

        // Re-compile so it can be resumed
        const lr = 0.01; 
        const optimizer = tf.train.adam(lr);
        const loss = this.taskType === 'classification' ? 'categoricalCrossentropy' : 'meanSquaredError';
        const metrics = this.taskType === 'classification' ? ['accuracy'] : [];
        this.model.compile({
            optimizer: optimizer,
            loss: loss,
            metrics: metrics
        });
        
        return this.model;
    }
}
