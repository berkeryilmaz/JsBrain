export class NetworkBuilder {
    constructor() {
        this.model = null;
        this.layers = []; 
        this.isTraining = false;
        this.stopRequested = false;
    }

    addLayer(units, activation) {
        this.layers.push({ units, activation });
    }

    removeLayer(index) {
        if (index >= 0 && index < this.layers.length) {
            this.layers.splice(index, 1);
        }
    }
    
    getLayers() {
        return this.layers;
    }

    buildModel(inputShape, outputShape, lr) {
        if (this.model) {
            this.model.dispose(); 
        }

        this.model = tf.sequential();
        
        if (this.layers.length === 0) {
            this.model.add(tf.layers.dense({
                units: outputShape,
                inputShape: [inputShape],
                activation: 'linear'
            }));
        } else {
            this.model.add(tf.layers.dense({
                units: this.layers[0].units,
                inputShape: [inputShape],
                activation: this.layers[0].activation
            }));
            
            for (let i = 1; i < this.layers.length; i++) {
                this.model.add(tf.layers.dense({
                    units: this.layers[i].units,
                    activation: this.layers[i].activation
                }));
            }
            
            this.model.add(tf.layers.dense({
                units: outputShape,
                activation: 'linear' 
            }));
        }

        const optimizer = tf.train.adam(lr);
        this.model.compile({
            optimizer: optimizer,
            loss: 'meanSquaredError'
            // NO metrics — loss IS mse, adding metrics:['mse'] computes it TWICE per batch
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
        
        // Re-compile so it can be resumed
        const lr = 0.01; 
        const optimizer = tf.train.adam(lr);
        this.model.compile({
            optimizer: optimizer,
            loss: 'meanSquaredError'
        });
        
        return this.model;
    }
}
