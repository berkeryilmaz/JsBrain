export class DataManager {
    constructor() {
        this.rawHeaders = [];
        this.rawData = [];
        this.config = {}; 
        this.normalizationMethod = "minmax"; 
        this.stats = {}; 
    }

    loadCSV(file, onComplete, onError) {
        if (!window.Papa) {
            if (onError) onError("PapaParse library is missing.");
            return;
        }
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true,
            complete: (results) => {
                this.rawHeaders = results.meta.fields.map((h, i) => {
                    return (h === "" || h === null || h === undefined) ? ((i === 0) ? "Index" : `Unnamed_${i}`) : h;
                });

                this.rawData = results.data.map(row => {
                    if ("" in row) {
                        row["Index"] = row[""];
                        delete row[""];
                    }
                    return row;
                }).filter(row => {
                    // Minimal validation: check if the row has data
                    return Object.keys(row).length > 0 && row[this.rawHeaders[0]] !== undefined && row[this.rawHeaders[0]] !== null;
                });
                
                if (this.rawData.length === 0) {
                    if (onError) onError("Dataset is empty or contains only strings/invalid data.");
                    return;
                }

                // Auto-detect if column is categorical (contains any text cell)
                const isCategorical = (header) => {
                    return this.rawData.some(row => {
                        let val = row[header];
                        if (val === null || val === undefined || val === "") return false;
                        if (typeof val === 'number') return false;
                        if (typeof val === 'string') {
                            let n = Number(val.replace(',', '.'));
                            if (isNaN(n)) return true; // It's a text Category
                        }
                        return false;
                    });
                };

                this.config = {};
                this.rawHeaders.forEach((h, i) => {
                    const catSuffix = isCategorical(h) ? "-cat" : "";
                    
                    if (h === "Index" && i === 0) {
                        this.config[h] = "ignore";
                    } else if (this.rawHeaders.length > 1 && i === this.rawHeaders.length - 1) {
                        this.config[h] = "target" + catSuffix;
                    } else {
                        this.config[h] = "input" + catSuffix;
                    }
                });
                if(onComplete) onComplete(this.rawHeaders, this.rawData.length);
            },
            error: (err) => {
                if (onError) onError(err.message);
                console.error("PapaParse error:", err);
            }
        });
    }

    updateConfig(header, role) {
        this.config[header] = role;
    }

    setNormalization(method) {
        this.normalizationMethod = method;
    }

    /**
     * Returns true if any target column is categorical (classification task).
     */
    isClassification() {
        for (let h of this.rawHeaders) {
            let role = this.config[h];
            if (role && role.startsWith('target') && role.endsWith('-cat')) return true;
        }
        return false;
    }
    
    getFeatureCounts() {
        let inputs = 0, targets = 0;
        
        let mappings = this._computeMappings();

        for (let h of this.rawHeaders) {
            let role = this.config[h];
            if (role === 'ignore') continue;
            
            let count = mappings[h].type === 'categorical' ? mappings[h].unique.length : 1;

            if (role.startsWith('input')) inputs += count;
            if (role.startsWith('target')) targets += count;
        }
        return { inputs, targets };
    }

    _computeMappings() {
        let mappings = {};
        for (let h of this.rawHeaders) {
            let role = this.config[h];
            if (role === 'ignore') continue;
            
            if (role.endsWith('-cat')) {
                let uniq = new Set();
                this.rawData.forEach(row => {
                    let val = row[h];
                    if (val !== null && val !== undefined && val !== "") uniq.add(String(val).trim());
                });
                mappings[h] = { type: 'categorical', unique: Array.from(uniq).sort() };
            } else {
                mappings[h] = { type: 'numeric' };
            }
        }
        return mappings;
    }

    calculateStats(matrix) {
        if (!matrix || matrix.length === 0) return [];
        const stats = [];
        const cols = matrix[0].length;
        for (let c = 0; c < cols; c++) {
            let colData = matrix.map(r => r[c]);
            let min = colData[0];
            let max = colData[0];
            let sum = 0;
            for (let i = 0; i < colData.length; i++) {
                if (colData[i] < min) min = colData[i];
                if (colData[i] > max) max = colData[i];
                sum += colData[i];
            }
            let mean = sum / colData.length;
            let variance = colData.reduce((a,b) => a + Math.pow(b - mean, 2), 0) / colData.length;
            let std = Math.sqrt(variance) || 1e-7;
            stats.push({ min, max, mean, std });
        }
        return stats;
    }

    normalize(matrix, stats, numericMask) {
        if (this.normalizationMethod === "none") return matrix;
        return matrix.map(row => 
            row.map((val, c) => {
                if (!numericMask[c]) return val; // Skip categorical / one-hot columns
                if (this.normalizationMethod === "minmax") {
                    let range = stats[c].max - stats[c].min;
                    return range === 0 ? 0 : (val - stats[c].min) / range;
                } else if (this.normalizationMethod === "zscore") {
                    return (val - stats[c].mean) / stats[c].std;
                }
                return val;
            })
        );
    }

    getTensors(splitRatio = 0.8) {
        let mappings = this._computeMappings();

        let inputKeys = this.rawHeaders.filter(h => this.config[h] && this.config[h].startsWith("input"));
        let targetKeys = this.rawHeaders.filter(h => this.config[h] && this.config[h].startsWith("target"));

        if (inputKeys.length === 0 || targetKeys.length === 0) {
            throw new Error("You must have at least 1 input and 1 target.");
        }

        let X = [];
        let y = [];

        let shuffled = [...this.rawData].sort(() => 0.5 - Math.random());

        const toNum = (val) => {
            if (val === null || val === undefined || val === "") return NaN;
            if (typeof val === 'string') {
                return Number(val.replace(',', '.'));
            }
            return Number(val);
        };

        for (let row of shuffled) {
            let rowX = [];
            let rowY = [];
            let skipRow = false;

            for (let h of inputKeys) {
                let map = mappings[h];
                if (map.type === 'categorical') {
                    let val = String(row[h]).trim();
                    let oneHot = new Array(map.unique.length).fill(0);
                    let idx = map.unique.indexOf(val);
                    if (idx !== -1) oneHot[idx] = 1;
                    rowX.push(...oneHot);
                } else {
                    let val = toNum(row[h]);
                    if (isNaN(val)) { skipRow = true; break; }
                    rowX.push(val);
                }
            }

            if (skipRow) continue;

            for (let h of targetKeys) {
                let map = mappings[h];
                if (map.type === 'categorical') {
                    let val = String(row[h]).trim();
                    let oneHot = new Array(map.unique.length).fill(0);
                    let idx = map.unique.indexOf(val);
                    if (idx !== -1) oneHot[idx] = 1;
                    rowY.push(...oneHot);
                } else {
                    let val = toNum(row[h]);
                    if (isNaN(val)) { skipRow = true; break; }
                    rowY.push(val);
                }
            }

            if (skipRow) continue;

            X.push(rowX);
            y.push(rowY);
        }

        if (X.length === 0) {
            throw new Error("No valid data found after processing selections. Check data formats.");
        }

        // Build numeric masks
        let xNumerics = [];
        inputKeys.forEach(h => {
             let map = mappings[h];
             if (map.type === 'categorical') map.unique.forEach(() => xNumerics.push(false));
             else xNumerics.push(true);
        });
        
        let yNumerics = [];
        targetKeys.forEach(h => {
             let map = mappings[h];
             if (map.type === 'categorical') map.unique.forEach(() => yNumerics.push(false));
             else yNumerics.push(true);
        });

        this.stats.X = this.calculateStats(X);
        this.stats.y = this.calculateStats(y); 

        let normX = this.normalize(X, this.stats.X, xNumerics);
        let normY = this.normalize(y, this.stats.y, yNumerics); 

        let splitIdx = Math.floor(normX.length * splitRatio);

        let trainX = normX.slice(0, splitIdx);
        let trainY = normY.slice(0, splitIdx);
        let valX = normX.slice(splitIdx);
        let valY = normY.slice(splitIdx);

        return {
            xTrain: tf.tensor2d(trainX, [trainX.length, xNumerics.length]),
            yTrain: tf.tensor2d(trainY, [trainY.length, yNumerics.length]),
            xVal: valX.length > 0 ? tf.tensor2d(valX, [valX.length, xNumerics.length]) : null,
            yVal: valY.length > 0 ? tf.tensor2d(valY, [valY.length, yNumerics.length]) : null,
        };
    }

    /**
     * Get the list of input and target header names along with their mappings,
     * useful for building the prediction UI dynamically.
     */
    getFeatureInfo() {
        let mappings = this._computeMappings();
        let inputKeys = this.rawHeaders.filter(h => this.config[h] && this.config[h].startsWith("input"));
        let targetKeys = this.rawHeaders.filter(h => this.config[h] && this.config[h].startsWith("target"));

        let inputs = [];
        for (let h of inputKeys) {
            let map = mappings[h];
            if (map.type === 'categorical') {
                inputs.push({ name: h, type: 'categorical', categories: map.unique });
            } else {
                inputs.push({ name: h, type: 'numeric' });
            }
        }

        let targets = [];
        for (let h of targetKeys) {
            let map = mappings[h];
            if (map.type === 'categorical') {
                targets.push({ name: h, type: 'categorical', categories: map.unique });
            } else {
                targets.push({ name: h, type: 'numeric' });
            }
        }

        return { inputs, targets };
    }

    /**
     * Normalize a single raw input row for prediction (uses training stats).
     * rawValues: object like { 'feature1': 3.5, 'feature2': 'CategoryA' }
     * Returns a flat array ready for model.predict.
     */
    normalizeInputRow(rawValues) {
        let mappings = this._computeMappings();
        let inputKeys = this.rawHeaders.filter(h => this.config[h] && this.config[h].startsWith("input"));

        let row = [];
        let colIdx = 0;

        const toNum = (val) => {
            if (val === null || val === undefined || val === "") return NaN;
            if (typeof val === 'string') return Number(val.replace(',', '.'));
            return Number(val);
        };

        for (let h of inputKeys) {
            let map = mappings[h];
            if (map.type === 'categorical') {
                let val = String(rawValues[h] || '').trim();
                let oneHot = new Array(map.unique.length).fill(0);
                let idx = map.unique.indexOf(val);
                if (idx !== -1) oneHot[idx] = 1;
                // Categorical columns are not normalized (already 0/1)
                row.push(...oneHot);
                colIdx += map.unique.length;
            } else {
                let val = toNum(rawValues[h]);
                if (isNaN(val)) val = 0;

                // Apply same normalization as training
                if (this.normalizationMethod === 'minmax' && this.stats.X && this.stats.X[colIdx]) {
                    let s = this.stats.X[colIdx];
                    let range = s.max - s.min;
                    val = range === 0 ? 0 : (val - s.min) / range;
                } else if (this.normalizationMethod === 'zscore' && this.stats.X && this.stats.X[colIdx]) {
                    let s = this.stats.X[colIdx];
                    val = (val - s.mean) / s.std;
                }

                row.push(val);
                colIdx++;
            }
        }

        return row;
    }

    /**
     * Denormalize model output values back to original scale.
     * normalizedValues: flat array of output values from the model.
     * Returns an object like { 'target1': 42.5 } or { 'target1': 'CategoryA' }
     */
    denormalizeOutput(normalizedValues) {
        let mappings = this._computeMappings();
        let targetKeys = this.rawHeaders.filter(h => this.config[h] && this.config[h].startsWith("target"));

        let result = {};
        let colIdx = 0;

        for (let h of targetKeys) {
            let map = mappings[h];
            if (map.type === 'categorical') {
                // Find argmax for this target's one-hot
                let catVals = normalizedValues.slice(colIdx, colIdx + map.unique.length);
                let maxIdx = 0;
                for (let k = 1; k < catVals.length; k++) {
                    if (catVals[k] > catVals[maxIdx]) maxIdx = k;
                }
                result[h] = map.unique[maxIdx];
                result[h + '_confidence'] = catVals[maxIdx];
                colIdx += map.unique.length;
            } else {
                let val = normalizedValues[colIdx];
                // Reverse normalization
                if (this.normalizationMethod === 'minmax' && this.stats.y && this.stats.y[colIdx]) {
                    let s = this.stats.y[colIdx];
                    let range = s.max - s.min;
                    val = val * range + s.min;
                } else if (this.normalizationMethod === 'zscore' && this.stats.y && this.stats.y[colIdx]) {
                    let s = this.stats.y[colIdx];
                    val = val * s.std + s.mean;
                }
                result[h] = val;
                colIdx++;
            }
        }

        return result;
    }

    /**
     * Parse a CSV file for prediction (batch test). 
     * Like loadCSV but returns a promise with normalized input matrix.
     */
    parsePredictionCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    try {
                        const rows = results.data;
                        if (rows.length === 0) {
                            reject(new Error("CSV is empty"));
                            return;
                        }

                        let normalizedRows = [];
                        let rawRows = [];

                        for (let row of rows) {
                            // Map raw CSV headers to our configured input headers
                            let rawValues = {};
                            let inputKeys = this.rawHeaders.filter(h => this.config[h] && this.config[h].startsWith("input"));
                            for (let h of inputKeys) {
                                rawValues[h] = row[h] !== undefined ? row[h] : '';
                            }

                            rawRows.push(rawValues);
                            normalizedRows.push(this.normalizeInputRow(rawValues));
                        }

                        resolve({ normalizedRows, rawRows, rowCount: rows.length });
                    } catch (e) {
                        reject(e);
                    }
                },
                error: (err) => reject(err)
            });
        });
    }
}
