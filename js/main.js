import { DataManager } from './data.js';
import { NetworkBuilder } from './network.js';
import { Chart, Visualizer, showToast } from './ui.js';
import { WindowManager } from './windowManager.js';

document.addEventListener('DOMContentLoaded', async () => {
    // Wait for TF JS
    await tf.ready();
    console.log("TensorFlow.js backend:", tf.getBackend());

    const dataManager = new DataManager();
    const network = new NetworkBuilder();
    const chart = new Chart('chart-canvas');
    const visualizer = new Visualizer('nn-canvas');
    const wm = new WindowManager();

    // Register panels for Window Manager
    wm.register('control-panel', { minW: 280, minH: 300 });
    wm.register('sim-panel', { minW: 350, minH: 250 });
    wm.register('nn-panel', { minW: 300, minH: 250 });
    wm.register('prediction-panel', { minW: 400, minH: 200 });
    wm.register('layer-panel', { minW: 400, minH: 120 });

    // UI Elements
    const uploadInput = document.getElementById('csv-upload');
    const featureList = document.getElementById('feature-config-list');
    const overlay = document.getElementById('data-overlay');
    
    const layerBuilder = document.getElementById('layer-builder');
    const inputDimLabel = document.getElementById('input-dim');
    const outputDimLabel = document.getElementById('output-dim');
    const layerSummary = document.getElementById('layer-summary');

    const btnTrain = document.getElementById('btn-train');
    const btnStop = document.getElementById('btn-stop');
    const btnReset = document.getElementById('btn-reset');
    const btnSave = document.getElementById('btn-save-model');
    const btnLoad = document.getElementById('btn-load-model');

    const pills = {
        epoch: document.getElementById('metric-epoch'),
        loss: document.getElementById('metric-loss'),
        valLoss: document.getElementById('metric-val-loss'),
        acc: document.getElementById('metric-acc')
    };
    
    let isDatasetLoaded = false;
    let isModelTrained = false;
    let inputNodes = 0;
    let outputNodes = 0;
    let lastEvalResults = null; // Store model evaluation results

    // ----- Data Handling -----

    const dropZone = document.getElementById('csv-drop-zone');

    function handleFile(file) {
        if (!file) return;
        dataManager.loadCSV(file, (headers, rowCount) => {
            isDatasetLoaded = true;
            overlay.style.display = 'none';
            document.getElementById('dataset-actions-row').style.display = 'flex';
            showToast(`Loaded ${rowCount} rows`, 'success');
            renderFeatureConfig(headers);
            updateArchitecture();
            
            // Automatically open preview
            openDataPreviewModal();
        }, (err) => {
            showToast(err, 'error');
        });
    }

    uploadInput.addEventListener('change', (e) => {
        handleFile(e.target.files[0]);
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function renderFeatureConfig(headers) {
        featureList.innerHTML = '';
        featureList.style.display = 'flex';
        
        headers.forEach(h => {
            const div = document.createElement('div');
            div.className = 'feature-item';
            
            const name = document.createElement('span');
            name.className = 'feature-name';
            name.textContent = h;
            name.title = h;
            
            const select = document.createElement('select');
            select.className = 'custom-select';
            select.innerHTML = `
                <option value="input" ${dataManager.config[h] === 'input' ? 'selected' : ''}>Input (Numeric)</option>
                <option value="input-cat" ${dataManager.config[h] === 'input-cat' ? 'selected' : ''}>Input (Category)</option>
                <option value="target" ${dataManager.config[h] === 'target' ? 'selected' : ''}>Target (Numeric)</option>
                <option value="target-cat" ${dataManager.config[h] === 'target-cat' ? 'selected' : ''}>Target (Category)</option>
                <option value="ignore" ${dataManager.config[h] === 'ignore' ? 'selected' : ''}>Ignore</option>
            `;
            
            select.addEventListener('change', (e) => {
                dataManager.updateConfig(h, e.target.value);
                updateArchitecture();
            });
            
            div.appendChild(name);
            div.appendChild(select);
            featureList.appendChild(div);
        });
    }

    // ----- Data Preview Modal -----
    
    const previewModal = document.getElementById('data-preview-modal');
    
    document.getElementById('btn-preview-data').addEventListener('click', () => {
        if (!isDatasetLoaded) return;
        openDataPreviewModal();
    });

    document.getElementById('data-preview-close').addEventListener('click', closeDataPreviewModal);
    document.getElementById('data-preview-apply').addEventListener('click', () => {
        closeDataPreviewModal();
        renderFeatureConfig(dataManager.rawHeaders); // sync sidebar
        updateArchitecture();
    });

    function openDataPreviewModal() {
        previewModal.style.display = 'grid';
        renderDataPreviewTable();
    }

    function closeDataPreviewModal() {
        previewModal.style.display = 'none';
    }

    function renderDataPreviewTable() {
        const headers = dataManager.rawHeaders;
        const previewRows = dataManager.rawData.slice(0, 15);
        
        const thead = document.getElementById('data-preview-head');
        const tbody = document.getElementById('data-preview-body');
        const info = document.getElementById('data-preview-info');
        
        info.textContent = `Showing ${previewRows.length} of ${dataManager.rawData.length} total rows.`;
        
        thead.innerHTML = '';
        tbody.innerHTML = '';
        
        // Header Row (Col names + dropdowns)
        const trHead = document.createElement('tr');
        headers.forEach(h => {
            const th = document.createElement('th');
            
            const colName = document.createElement('span');
            colName.className = 'col-name';
            colName.textContent = h;
            colName.title = h;
            
            const select = document.createElement('select');
            select.className = 'custom-select';
            select.innerHTML = `
                <option value="input" ${dataManager.config[h] === 'input' ? 'selected' : ''}>Input (Num)</option>
                <option value="input-cat" ${dataManager.config[h] === 'input-cat' ? 'selected' : ''}>Input (Cat)</option>
                <option value="target" ${dataManager.config[h] === 'target' ? 'selected' : ''}>Target (Num)</option>
                <option value="target-cat" ${dataManager.config[h] === 'target-cat' ? 'selected' : ''}>Target (Cat)</option>
                <option value="ignore" ${dataManager.config[h] === 'ignore' ? 'selected' : ''}>Ignore</option>
            `;
            
            select.addEventListener('change', (e) => {
                dataManager.updateConfig(h, e.target.value);
            });
            
            th.appendChild(colName);
            th.appendChild(select);
            trHead.appendChild(th);
        });
        thead.appendChild(trHead);
        
        // Data Rows
        previewRows.forEach(row => {
            const tr = document.createElement('tr');
            headers.forEach(h => {
                const td = document.createElement('td');
                const val = row[h] !== undefined && row[h] !== null ? String(row[h]) : '';
                td.textContent = val;
                td.title = val; // tooltip for truncated text
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    document.getElementById('select-norm').addEventListener('change', (e) => {
        dataManager.setNormalization(e.target.value);
    });

    // ----- Network Builder -----

    function renderLayerStack() {
        const layers = network.getLayers();
        
        // Remove old dynamic layers
        document.querySelectorAll('.layer-pill.dynamic').forEach(e => e.remove());
        
        const addButton = layerBuilder.querySelector('.add-layer-btn');
        
        layers.forEach((l, index) => {
            const div = document.createElement('div');
            div.className = 'layer-pill dynamic';
            
            const layerType = l.type || 'dense';
            if(layerType === 'dense') {
                div.innerHTML = `
                    Dense<br><span class="layer-dim">${l.units} ${l.activation}</span>
                    <button class="layer-action-remove" title="Remove Layer">×</button>
                `;
            } else if(layerType === 'dropout') {
                div.innerHTML = `
                    Dropout<br><span class="layer-dim">Rate: ${l.rate}</span>
                    <button class="layer-action-remove" title="Remove Layer">×</button>
                `;
            }
            
            div.querySelector('.layer-action-remove').addEventListener('click', () => {
                network.removeLayer(index);
                renderLayerStack();
                updateArchitecture();
            });
            
            layerBuilder.insertBefore(div, addButton);
        });
    }

    function updateArchitecture() {
        if (!isDatasetLoaded) return;
        const counts = dataManager.getFeatureCounts();
        inputNodes = counts.inputs;
        outputNodes = counts.targets;
        
        inputDimLabel.textContent = `${inputNodes} Features`;
        outputDimLabel.textContent = `${outputNodes} Targets`;
        
        visualizer.update(counts.inputs, network.getLayers(), counts.targets);
        
        // Build real node labels from DataManager feature info
        try {
            const featureInfo = dataManager.getFeatureInfo();
            const inLabels = [];
            const outLabels = [];

            for (const f of featureInfo.inputs) {
                if (f.type === 'categorical') {
                    // Each category becomes a one-hot node: is_CategoryValue
                    for (const cat of f.categories) {
                        inLabels.push(`is_${cat}`);
                    }
                } else {
                    inLabels.push(f.name);
                }
            }

            for (const f of featureInfo.targets) {
                if (f.type === 'categorical') {
                    for (const cat of f.categories) {
                        outLabels.push(`is_${cat}`);
                    }
                } else {
                    outLabels.push(f.name);
                }
            }

            visualizer.setLabels(inLabels, outLabels);
        } catch(e) { /* ignore if data not ready */ }

        const layers = network.getLayers();
        let params = 0;
        let prev = counts.inputs;
        
        for (let l of layers) {
            const layerType = l.type || 'dense';
            if (layerType === 'dense') {
                const u = l.units || 64;
                params += (prev * u) + parseInt(u);
                prev = u;
            } else if (layerType === 'dropout') {
                params += 0;
            }
        }
        params += (prev * counts.targets) + parseInt(counts.targets);
        layerSummary.textContent = `${params.toLocaleString()} Params`;
    }

    // Modal logic is attached to window as a quick inline handler in index.html,
    // let's bind it here instead perfectly.
    const modal = document.getElementById('layer-modal');
    window.UI = {
        openLayerModal: () => {
            modal.style.display = 'grid';
        }
    };
    
    document.getElementById('layer-modal-close').onclick = () => modal.style.display = 'none';
    document.getElementById('layer-modal-cancel').onclick = () => modal.style.display = 'none';
    document.getElementById('layer-modal-apply').onclick = () => {
        const type = document.getElementById('layer-type').value;
        if (type === 'dense') {
            const units = parseInt(document.getElementById('layer-units').value);
            const act = document.getElementById('layer-activation').value;
            network.addLayer({ type: 'dense', units, activation: act });
        } else if (type === 'dropout') {
            const rate = parseFloat(document.getElementById('layer-rate').value);
            network.addLayer({ type: 'dropout', rate });
        }
        
        renderLayerStack();
        updateArchitecture();
        modal.style.display = 'none';
    };


    // ----- Training Pipeline -----

    let currentTensors = null;

    btnTrain.addEventListener('click', async () => {
        if (!isDatasetLoaded) {
            showToast('Please upload a CSV dataset first', 'error');
            return;
        }
        
        const lr = parseFloat(document.getElementById('slider-lr').value);
        const epochs = parseInt(document.getElementById('input-epochs').value);
        const batchSize = document.getElementById('select-batch').value;
        const splitRatio = parseFloat(document.getElementById('slider-split').value);

        try {
            // Memory Leak Prevention: Dispose old tensors in WebGL backend
            if (currentTensors) {
                if (currentTensors.xTrain) currentTensors.xTrain.dispose();
                if (currentTensors.yTrain) currentTensors.yTrain.dispose();
                if (currentTensors.xVal) currentTensors.xVal.dispose();
                if (currentTensors.yVal) currentTensors.yVal.dispose();
            }
            currentTensors = dataManager.getTensors(splitRatio);
        } catch(e) {
            showToast(e.message, 'error');
            return;
        }

        if (inputNodes === 0 || outputNodes === 0) {
            showToast("Failed to build model: Input or Target columns not assigned correctly.", "error");
            return;
        }

        try {
            const isClassification = dataManager.isClassification();
            network.buildModel(inputNodes, outputNodes, lr, isClassification);
        } catch(e) {
            console.error(e);
            showToast("Failed to build model geometry. Check layer configurations.", "error");
            return;
        }

        btnTrain.disabled = true;
        btnStop.disabled = false;
        chart.reset();
        
        const statusText = document.getElementById('status-text');
        const statusInd = document.getElementById('status-indicator');
        statusText.textContent = "Training...";
        statusInd.classList.add('training');

        // NO animation during training — it competes with TF.js for GPU
        let lastDrawTime = 0;
        
        try {
            await network.train(currentTensors, epochs, batchSize, {
                onEpoch: (epoch, logs) => {
                    chart.addPoint(logs.loss, logs.val_loss);

                    const now = performance.now();
                    if (now - lastDrawTime > 80 || epoch === epochs - 1) {
                        pills.epoch.textContent = `${epoch + 1} / ${epochs}`;
                        pills.loss.textContent = logs.loss.toFixed(4);
                        if (logs.val_loss) pills.valLoss.textContent = logs.val_loss.toFixed(4);

                        // Update accuracy for classification tasks
                        if (logs.acc !== undefined) {
                            pills.acc.textContent = `${(logs.acc * 100).toFixed(1)}%`;
                        } else if (logs.val_acc !== undefined) {
                            pills.acc.textContent = `${(logs.val_acc * 100).toFixed(1)}%`;
                        }

                        if (!chart.isDrawingPending) {
                            chart.isDrawingPending = true;
                            requestAnimationFrame(() => {
                                chart.draw();
                                chart.isDrawingPending = false;
                            });
                        }
                        
                        lastDrawTime = now;
                    }
                },
                onEnd: () => {
                    btnTrain.disabled = false;
                    btnStop.disabled = true;
                    statusText.textContent = "Idle";
                    statusInd.classList.remove('training');
                    
                    // Extract weights ONLY at end — zero overhead during training
                    try {
                        const wts = network.extractWeightsForVis();
                        visualizer.setWeights(wts);
                        visualizer.draw();
                    } catch(e) {}

                    // Evaluate model performance
                    try {
                        const evalX = currentTensors.xVal || currentTensors.xTrain;
                        const evalY = currentTensors.yVal || currentTensors.yTrain;
                        lastEvalResults = network.evaluateModel(evalX, evalY);

                        // Update header accuracy pill
                        if (lastEvalResults.type === 'classification') {
                            pills.acc.textContent = `${(lastEvalResults.accuracy * 100).toFixed(1)}%`;
                            document.getElementById('pill-acc').querySelector('.pill-label').textContent = 'Accuracy';
                        } else {
                            pills.acc.textContent = `${(lastEvalResults.r2 * 100).toFixed(1)}%`;
                            document.getElementById('pill-acc').querySelector('.pill-label').textContent = 'R² Score';
                        }
                    } catch(e) {
                        console.warn('Evaluation failed:', e);
                    }

                    showToast("Training Complete!", "success");

                    // Enable prediction panel
                    isModelTrained = true;
                    showPredictionPanel();
                }
            });
        } catch (e) {
            btnTrain.disabled = false;
            btnStop.disabled = true;
            statusText.textContent = "Idle";
            statusInd.classList.remove('training');
        }
    });

    btnStop.addEventListener('click', () => {
        network.stopTraining();
    });

    btnReset.addEventListener('click', () => {
        chart.reset();
        pills.epoch.textContent = "0 / 0";
        pills.loss.textContent = "0.000";
        pills.valLoss.textContent = "0.000";
    });

    btnSave.addEventListener('click', async () => {
        try {
            await network.saveModel();
            showToast("Model downloaded successfully", "success");
        } catch (e) {
            showToast("Cannot save. Build and train model first.", "error");
        }
    });

    const loadInput = document.getElementById('model-upload-input');
    btnLoad.addEventListener('click', () => {
        loadInput.click();
    });

    loadInput.addEventListener('change', async (e) => {
        const files = e.target.files;
        if (files.length === 0) return;
        
        try {
            await network.loadModel(files);
            showToast("Model loaded successfully!", "success");
            
            try {
                // Auto mapping layers UI
                // tfjs loadLayersModel preserves topology, we can infer layers
                network.layers = [];
                const topology = network.model.layers;
                // Skips input but includes the explicit dense layers
                for(let i = 0; i < topology.length; i++) {
                    let layer = topology[i];
                    if (layer.name.startsWith('input') || i === topology.length - 1) continue;
                    
                    let act = "relu";
                    if (layer.activation && layer.getConfig && layer.getConfig()) {
                        act = layer.getConfig().activation || layer.activation.constructor.className.toLowerCase() || 'relu';
                    }
                    network.addLayer(layer.units, act);
                }
                renderLayerStack();
                updateArchitecture();
            } catch (uiErr) {
                console.warn("Model loaded but UI layer reconstruction failed:", uiErr);
            }

            // Enable prediction if we have data
            isModelTrained = true;
            
            // Extract weights for visualization
            try {
                const wts = network.extractWeightsForVis();
                visualizer.setWeights(wts);
                visualizer.draw();
            } catch(e) {}

            if (isDatasetLoaded) {
                showPredictionPanel();
            }
        } catch (err) {
            showToast(err.message || "Failed to load model", "error");
        }
        // clear input so same file can be triggered again
        loadInput.value = '';
    });

    // ----- Prediction Panel -----

    const predPanel = document.getElementById('prediction-panel');
    const predInputsGrid = document.getElementById('pred-inputs-grid');
    const predSingleResults = document.getElementById('pred-single-results');
    const btnPredictSingle = document.getElementById('btn-predict-single');
    const predCsvUpload = document.getElementById('pred-csv-upload');
    const predTableHead = document.getElementById('pred-table-head');
    const predTableBody = document.getElementById('pred-table-body');
    const predBatchInfo = document.getElementById('pred-batch-info');
    const predResultsTableWrap = document.getElementById('pred-results-table-wrap');
    const btnDownloadResults = document.getElementById('btn-download-results');

    // Tab switching
    document.querySelectorAll('.pred-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.pred-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.pred-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`pred-${tab.dataset.tab}-content`).classList.add('active');
        });
    });

    let lastBatchResults = null; // Store for CSV download

    function showPredictionPanel() {
        predPanel.style.display = 'flex';
        buildPredictionInputs();
        btnPredictSingle.disabled = false;

        // Show the performance tab if we have eval results
        if (lastEvalResults) {
            renderPerformanceResults(lastEvalResults);
        }

        // Trigger resize so canvases adjust
        setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
    }

    function buildPredictionInputs() {
        predInputsGrid.innerHTML = '';
        if (!isDatasetLoaded) return;

        const featureInfo = dataManager.getFeatureInfo();

        featureInfo.inputs.forEach(f => {
            const group = document.createElement('div');
            group.className = 'pred-input-group';

            const label = document.createElement('label');
            label.className = 'pred-input-label';
            label.textContent = f.name;

            if (f.type === 'categorical') {
                const select = document.createElement('select');
                select.className = 'custom-select pred-input-field';
                select.dataset.feature = f.name;
                select.dataset.type = 'categorical';
                f.categories.forEach(cat => {
                    const opt = document.createElement('option');
                    opt.value = cat;
                    opt.textContent = cat;
                    select.appendChild(opt);
                });
                group.appendChild(label);
                group.appendChild(select);
            } else {
                const input = document.createElement('input');
                input.type = 'number';
                input.className = 'number-input pred-input-field';
                input.dataset.feature = f.name;
                input.dataset.type = 'numeric';
                input.placeholder = '0';
                input.step = 'any';
                group.appendChild(label);
                group.appendChild(input);
            }

            predInputsGrid.appendChild(group);
        });
    }

    // Single prediction
    btnPredictSingle.addEventListener('click', () => {
        if (!isModelTrained || !network.model) {
            showToast("Train or load a model first", "error");
            return;
        }

        try {
            // Collect input values
            const rawValues = {};
            predInputsGrid.querySelectorAll('.pred-input-field').forEach(el => {
                const feature = el.dataset.feature;
                if (el.dataset.type === 'categorical') {
                    rawValues[feature] = el.value;
                } else {
                    rawValues[feature] = parseFloat(el.value) || 0;
                }
            });

            // Normalize
            const normalized = dataManager.normalizeInputRow(rawValues);

            // Predict
            const rawOutput = network.predict(normalized);

            // Denormalize
            const result = dataManager.denormalizeOutput(rawOutput);

            // Render results
            renderSingleResult(result);
        } catch (e) {
            showToast(e.message, "error");
            console.error(e);
        }
    });

    function renderSingleResult(result) {
        predSingleResults.style.display = 'flex';
        predSingleResults.innerHTML = '';

        for (const [key, value] of Object.entries(result)) {
            if (key.endsWith('_confidence')) continue;

            const card = document.createElement('div');
            card.className = 'pred-result-card';

            const label = document.createElement('span');
            label.className = 'pred-result-label';
            label.textContent = key;

            const val = document.createElement('span');
            val.className = 'pred-result-value';

            if (typeof value === 'number') {
                val.textContent = value.toFixed(4);
            } else {
                val.textContent = value;
                // Add confidence if available
                const conf = result[key + '_confidence'];
                if (conf !== undefined) {
                    const confBadge = document.createElement('span');
                    confBadge.className = 'pred-confidence';
                    confBadge.textContent = `${(conf * 100).toFixed(1)}%`;
                    card.appendChild(label);
                    card.appendChild(val);
                    card.appendChild(confBadge);
                    predSingleResults.appendChild(card);
                    continue;
                }
            }

            card.appendChild(label);
            card.appendChild(val);
            predSingleResults.appendChild(card);
        }
    }

    // Batch CSV prediction
    predCsvUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        if (!isModelTrained || !network.model) {
            showToast("Train or load a model first", "error");
            return;
        }

        try {
            const { normalizedRows, rawRows, rowCount } = await dataManager.parsePredictionCSV(file);

            predBatchInfo.style.display = 'block';
            predBatchInfo.textContent = `Processing ${rowCount} rows...`;

            // Batch predict
            const rawOutputs = network.predictBatch(normalizedRows);

            // Denormalize all outputs
            const results = rawOutputs.map(out => dataManager.denormalizeOutput(out));

            // Build table
            const featureInfo = dataManager.getFeatureInfo();
            const inputHeaders = featureInfo.inputs.map(f => f.name);
            const outputHeaders = Object.keys(results[0] || {}).filter(k => !k.endsWith('_confidence'));

            // Store for download
            lastBatchResults = { inputHeaders, outputHeaders, rawRows, results };

            // Render table header
            predTableHead.innerHTML = '';
            const headRow = document.createElement('tr');
            headRow.innerHTML = `<th>#</th>`;
            inputHeaders.forEach(h => { headRow.innerHTML += `<th class="input-col">${h}</th>`; });
            outputHeaders.forEach(h => { headRow.innerHTML += `<th class="output-col">${h}</th>`; });
            predTableHead.appendChild(headRow);

            // Render table body (limit to 500 rows for performance)
            predTableBody.innerHTML = '';
            const maxRows = Math.min(results.length, 500);
            for (let i = 0; i < maxRows; i++) {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td class="row-num">${i + 1}</td>`;
                inputHeaders.forEach(h => {
                    const val = rawRows[i][h];
                    tr.innerHTML += `<td class="input-col">${val !== undefined ? val : ''}</td>`;
                });
                outputHeaders.forEach(h => {
                    const val = results[i][h];
                    const display = typeof val === 'number' ? val.toFixed(4) : val;
                    tr.innerHTML += `<td class="output-col">${display}</td>`;
                });
                predTableBody.appendChild(tr);
            }

            predBatchInfo.textContent = `Showing ${maxRows} of ${rowCount} results`;
            predResultsTableWrap.style.display = 'block';
            btnDownloadResults.style.display = 'flex';

            showToast(`Predicted ${rowCount} rows`, 'success');
        } catch (e) {
            showToast(e.message, 'error');
            console.error(e);
        }

        predCsvUpload.value = '';
    });

    // Download batch results as CSV
    btnDownloadResults.addEventListener('click', () => {
        if (!lastBatchResults) return;

        const { inputHeaders, outputHeaders, rawRows, results } = lastBatchResults;
        const allHeaders = [...inputHeaders, ...outputHeaders];

        let csvContent = allHeaders.join(',') + '\n';
        for (let i = 0; i < results.length; i++) {
            const row = [];
            inputHeaders.forEach(h => row.push(rawRows[i][h] !== undefined ? rawRows[i][h] : ''));
            outputHeaders.forEach(h => {
                const val = results[i][h];
                row.push(typeof val === 'number' ? val.toFixed(6) : val);
            });
            csvContent += row.join(',') + '\n';
        }

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        if (window.saveAs) {
            window.saveAs(blob, 'jsbrain-predictions.csv');
        } else {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'jsbrain-predictions.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
        showToast("Predictions CSV downloaded", "success");
    });

    // ─── Performance Rendering ───

    function renderPerformanceResults(evalResult) {
        const container = document.getElementById('pred-perf-content');
        if (!container) return;
        container.innerHTML = '';

        if (evalResult.type === 'classification') {
            renderClassificationPerformance(container, evalResult);
        } else {
            renderRegressionPerformance(container, evalResult);
        }
    }

    function renderClassificationPerformance(container, result) {
        const featureInfo = dataManager.getFeatureInfo();
        const targetCats = featureInfo.targets.length > 0 && featureInfo.targets[0].categories
            ? featureInfo.targets[0].categories
            : Array.from({ length: result.numClasses }, (_, i) => `Class ${i}`);

        // Summary cards row
        const summaryRow = document.createElement('div');
        summaryRow.className = 'perf-summary-row';
        summaryRow.innerHTML = `
            <div class="perf-metric-card perf-accent">
                <span class="perf-metric-label">Accuracy</span>
                <span class="perf-metric-value">${(result.accuracy * 100).toFixed(1)}%</span>
            </div>
            <div class="perf-metric-card">
                <span class="perf-metric-label">Samples</span>
                <span class="perf-metric-value">${result.numSamples}</span>
            </div>
            <div class="perf-metric-card">
                <span class="perf-metric-label">Classes</span>
                <span class="perf-metric-value">${result.numClasses}</span>
            </div>
        `;
        container.appendChild(summaryRow);

        // Confusion matrix
        const cmSection = document.createElement('div');
        cmSection.className = 'perf-section';
        cmSection.innerHTML = '<h4 class="perf-section-title">Confusion Matrix</h4>';

        const cmWrap = document.createElement('div');
        cmWrap.className = 'confusion-matrix-wrap';

        const table = document.createElement('table');
        table.className = 'confusion-matrix';

        // Header
        const thead = document.createElement('thead');
        let headerHtml = '<tr><th class="cm-corner">Actual \\ Pred</th>';
        targetCats.forEach(c => { headerHtml += `<th class="cm-pred-header">${c}</th>`; });
        headerHtml += '</tr>';
        thead.innerHTML = headerHtml;
        table.appendChild(thead);

        // Body
        const maxVal = Math.max(...result.confusionMatrix.flat(), 1);
        const tbody = document.createElement('tbody');
        for (let r = 0; r < result.numClasses; r++) {
            let rowHtml = `<tr><td class="cm-actual-header">${targetCats[r]}</td>`;
            for (let c = 0; c < result.numClasses; c++) {
                const val = result.confusionMatrix[r][c];
                const intensity = val / maxVal;
                const isDiag = r === c;
                const bgColor = isDiag
                    ? `rgba(0, 206, 201, ${0.1 + intensity * 0.5})`
                    : (val > 0 ? `rgba(255, 107, 107, ${0.1 + intensity * 0.4})` : 'transparent');
                rowHtml += `<td class="cm-cell ${isDiag ? 'cm-diag' : ''}" style="background:${bgColor}">${val}</td>`;
            }
            rowHtml += '</tr>';
            tbody.innerHTML += rowHtml;
        }
        table.appendChild(tbody);
        cmWrap.appendChild(table);
        cmSection.appendChild(cmWrap);
        container.appendChild(cmSection);

        // Per-class metrics
        const classSection = document.createElement('div');
        classSection.className = 'perf-section';
        classSection.innerHTML = '<h4 class="perf-section-title">Per-Class Metrics</h4>';

        const classTable = document.createElement('table');
        classTable.className = 'perf-class-table';
        classTable.innerHTML = `
            <thead><tr>
                <th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>
            </tr></thead>
        `;
        const classTbody = document.createElement('tbody');
        for (let c = 0; c < result.numClasses; c++) {
            const p = result.perClass[c].precision;
            const r2 = result.perClass[c].recall;
            const f1 = (p + r2) > 0 ? 2 * p * r2 / (p + r2) : 0;
            classTbody.innerHTML += `
                <tr>
                    <td class="class-name">${targetCats[c]}</td>
                    <td>${(p * 100).toFixed(1)}%</td>
                    <td>${(r2 * 100).toFixed(1)}%</td>
                    <td>${(f1 * 100).toFixed(1)}%</td>
                    <td>${result.perClass[c].support}</td>
                </tr>
            `;
        }
        classTable.appendChild(classTbody);
        classSection.appendChild(classTable);
        container.appendChild(classSection);
    }

    function renderRegressionPerformance(container, result) {
        const featureInfo = dataManager.getFeatureInfo();
        const targetNames = featureInfo.targets.map(t => t.name);

        // Summary cards
        const summaryRow = document.createElement('div');
        summaryRow.className = 'perf-summary-row';
        summaryRow.innerHTML = `
            <div class="perf-metric-card perf-accent">
                <span class="perf-metric-label">R² Score</span>
                <span class="perf-metric-value">${(result.r2 * 100).toFixed(1)}%</span>
                <div class="perf-metric-bar">
                    <div class="perf-metric-bar-fill" style="width:${Math.max(0, result.r2 * 100)}%"></div>
                </div>
            </div>
            <div class="perf-metric-card">
                <span class="perf-metric-label">MAE</span>
                <span class="perf-metric-value">${result.mae.toFixed(4)}</span>
            </div>
            <div class="perf-metric-card">
                <span class="perf-metric-label">RMSE</span>
                <span class="perf-metric-value">${result.rmse.toFixed(4)}</span>
            </div>
            <div class="perf-metric-card">
                <span class="perf-metric-label">Samples</span>
                <span class="perf-metric-value">${result.numSamples}</span>
            </div>
        `;
        container.appendChild(summaryRow);

        // Per-target breakdown (if multiple targets)
        if (result.numTargets > 1) {
            const targetSection = document.createElement('div');
            targetSection.className = 'perf-section';
            targetSection.innerHTML = '<h4 class="perf-section-title">Per-Target Metrics</h4>';

            const table = document.createElement('table');
            table.className = 'perf-class-table';
            table.innerHTML = `
                <thead><tr><th>Target</th><th>R²</th><th>MAE</th><th>RMSE</th></tr></thead>
            `;
            const tbody = document.createElement('tbody');
            for (let t = 0; t < result.numTargets; t++) {
                const m = result.perTarget[t];
                const name = targetNames[t] || `Target ${t + 1}`;
                tbody.innerHTML += `
                    <tr>
                        <td class="class-name">${name}</td>
                        <td>${(m.r2 * 100).toFixed(1)}%</td>
                        <td>${m.mae.toFixed(4)}</td>
                        <td>${m.rmse.toFixed(4)}</td>
                    </tr>
                `;
            }
            table.appendChild(tbody);
            targetSection.appendChild(table);
            container.appendChild(targetSection);
        }
    }

    // Theme Toggle implementation
    const btnTheme = document.getElementById('btn-theme-toggle');
    btnTheme.addEventListener('click', () => {
        const body = document.body;
        if (body.getAttribute('data-theme') === 'light') {
            body.removeAttribute('data-theme');
        } else {
            body.setAttribute('data-theme', 'light');
        }
        // Resize charts to fix any color overrides
        setTimeout(() => {
            chart.draw();
            visualizer.draw();
        }, 350);
    });

    // Sliders update value labels
    const tSplit = document.getElementById('slider-split');
    const valSplit = document.getElementById('val-split');
    tSplit.addEventListener('input', (e) => {
        let frac = parseFloat(e.target.value);
        valSplit.textContent = `${Math.round(frac*100)}% / ${Math.round((1-frac)*100)}%`;
    });
    
    const tLr = document.getElementById('slider-lr');
    const valLr = document.getElementById('val-lr');
    tLr.addEventListener('input', (e) => {
        valLr.textContent = parseFloat(e.target.value).toFixed(4);
    });

    // Hardware Backend Toggle
    const selectBackend = document.getElementById('select-backend');
    selectBackend.addEventListener('change', async (e) => {
        const backend = e.target.value;
        try {
            await tf.setBackend(backend);
            await tf.ready();
            showToast(`Hardware switched to: ${backend.toUpperCase()}`, "success");
        } catch (err) {
            showToast(`Failed to switch backend: ${err.message}`, "error");
            selectBackend.value = tf.getBackend(); // Revert UI
        }
    });
});
