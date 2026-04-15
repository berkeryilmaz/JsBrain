export class Chart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.data = { loss: [], valLoss: [] };
        this.maxLoss = 1;
        // Running max tracking — O(1) per addPoint instead of O(n)
        this._runningMaxLoss = 0;
        this._runningMaxValLoss = 0;
        this._decayCounter = 0;
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width * devicePixelRatio;
        this.canvas.height = rect.height * devicePixelRatio;
        this.ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
        this._w = rect.width;
        this._h = rect.height;
        this.draw();
    }

    addPoint(loss, valLoss) {
        this.data.loss.push(loss);
        if (valLoss !== undefined && valLoss !== null && !isNaN(valLoss)) {
            this.data.valLoss.push(valLoss);
            if (valLoss > this._runningMaxValLoss) this._runningMaxValLoss = valLoss;
        }

        if (loss > this._runningMaxLoss) this._runningMaxLoss = loss;

        let cMax = Math.max(this._runningMaxLoss, this._runningMaxValLoss);
        if (cMax > this.maxLoss) {
            this.maxLoss = cMax * 1.1;
        }

        // Aggressively recalculate scale every 20 epochs for responsive chart
        this._decayCounter++;
        if (this._decayCounter >= 20) {
            this._decayCounter = 0;
            this._recalcMax();
        }
    }

    _recalcMax() {
        const n = this.data.loss.length;
        // Look at last 50 points (or all if fewer)
        const window = Math.min(50, n);
        const recent = this.data.loss.slice(-window);
        const recentVal = this.data.valLoss.slice(-window);
        let m1 = 0, m2 = 0;
        for (let i = 0; i < recent.length; i++) { if (recent[i] > m1) m1 = recent[i]; }
        for (let i = 0; i < recentVal.length; i++) { if (recentVal[i] > m2) m2 = recentVal[i]; }
        let recentMax = Math.max(m1, m2);
        // Scale down aggressively — if recent max is less than half of current scale
        if (recentMax < this.maxLoss * 0.5 && this.maxLoss > 0.001) {
            this.maxLoss = Math.max(recentMax * 1.3, 0.001);
            this._runningMaxLoss = m1;
            this._runningMaxValLoss = m2;
        }
    }

    reset() {
        this.data = { loss: [], valLoss: [] };
        this.maxLoss = 1;
        this._runningMaxLoss = 0;
        this._runningMaxValLoss = 0;
        this._decayCounter = 0;
        this.draw();
    }

    draw() {
        const width = this._w || this.canvas.width;
        const height = this._h || this.canvas.height;
        const ctx = this.ctx;
        ctx.clearRect(0, 0, width, height);

        const padding = { top: 30, right: 24, bottom: 36, left: 58 };
        const innerW = width - padding.left - padding.right;
        const innerH = height - padding.top - padding.bottom;

        // Grid lines
        ctx.strokeStyle = 'rgba(136, 136, 168, 0.08)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (innerH / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();
        }

        // Axes
        ctx.strokeStyle = 'rgba(136, 136, 168, 0.25)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, height - padding.bottom);
        ctx.lineTo(width - padding.right, height - padding.bottom);
        ctx.stroke();

        if (this.data.loss.length === 0) return;

        const pts = this.data.loss.length;
        const step = Math.max(1, Math.floor(pts / 500));

        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        // Draw Val Loss with gradient
        if (this.data.valLoss.length > 0) {
            const grad = ctx.createLinearGradient(padding.left, 0, width - padding.right, 0);
            grad.addColorStop(0, '#feca57');
            grad.addColorStop(1, '#ff9f43');

            ctx.beginPath();
            ctx.strokeStyle = grad;
            ctx.lineWidth = 2;
            for (let i = 0; i < this.data.valLoss.length; i += step) {
                const x = padding.left + (i / Math.max(1, pts - 1)) * innerW;
                const y = padding.top + innerH - (this.data.valLoss[i] / this.maxLoss) * innerH;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            if ((pts - 1) % step !== 0 && this.data.valLoss.length > 0) {
                const lx = padding.left + innerW;
                const ly = padding.top + innerH - (this.data.valLoss[this.data.valLoss.length - 1] / this.maxLoss) * innerH;
                ctx.lineTo(lx, ly);
            }
            ctx.stroke();

            // Soft glow (no filter:blur — that is extremely GPU heavy)
            ctx.save();
            ctx.globalAlpha = 0.12;
            ctx.lineWidth = 6;
            ctx.stroke();
            ctx.restore();
        }

        // Draw Loss with gradient
        const lossGrad = ctx.createLinearGradient(padding.left, 0, width - padding.right, 0);
        lossGrad.addColorStop(0, '#00cec9');
        lossGrad.addColorStop(1, '#55efc4');

        ctx.beginPath();
        ctx.strokeStyle = lossGrad;
        ctx.lineWidth = 2.5;
        for (let i = 0; i < pts; i += step) {
            const x = padding.left + (i / Math.max(1, pts - 1)) * innerW;
            const y = padding.top + innerH - (this.data.loss[i] / this.maxLoss) * innerH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        if ((pts - 1) % step !== 0) {
            const lx = padding.left + innerW;
            const ly = padding.top + innerH - (this.data.loss[pts - 1] / this.maxLoss) * innerH;
            ctx.lineTo(lx, ly);
        }
        ctx.stroke();

        // Soft glow (alpha only, no filter:blur)
        ctx.save();
        ctx.globalAlpha = 0.12;
        ctx.lineWidth = 7;
        ctx.stroke();
        ctx.restore();

        // Labels
        ctx.fillStyle = '#8888a8';
        ctx.font = '10px "JetBrains Mono"';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';

        for (let i = 0; i <= 4; i++) {
            const val = this.maxLoss * (1 - i / 4);
            const y = padding.top + (innerH / 4) * i;
            ctx.fillText(val.toFixed(4), padding.left - 8, y);
        }

        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText("0", padding.left, height - padding.bottom + 8);
        ctx.fillText(pts.toString(), width - padding.right, height - padding.bottom + 8);
        if (pts > 2) {
            ctx.fillText(Math.round(pts / 2).toString(), padding.left + innerW / 2, height - padding.bottom + 8);
        }

        // Legend (styled pills)
        const legX = width - padding.right;
        const legY = 8;
        ctx.font = '600 10px Inter';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'top';

        // Train legend dot
        ctx.fillStyle = '#00cec9';
        ctx.beginPath();
        ctx.arc(legX - 38, legY + 5, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText("Train", legX - 18, legY);

        if (this.data.valLoss.length > 0) {
            ctx.fillStyle = '#feca57';
            ctx.beginPath();
            ctx.arc(legX - 92, legY + 5, 3, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillText("Val", legX - 72, legY);
        }
    }
}

export class Visualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.layers = [];
        this.inputNodes = 0;
        this.outputNodes = 0;
        this.layerSizes = [];
        this.weights = null;
        this.activations = null;
        this.inputLabels = [];   // actual feature names
        this.outputLabels = [];  // actual target names
        this._isAnimating = false;
        this._animFrame = null;
        this.animOffset = 0;
        this.resize();
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width * devicePixelRatio;
        this.canvas.height = rect.height * devicePixelRatio;
        this.ctx.scale(devicePixelRatio, devicePixelRatio);
        this._w = rect.width;
        this._h = rect.height;
        this.draw();
    }

    update(inputCount, hiddenLayers, outputCount) {
        this.inputNodes = inputCount || 0;
        this.layers = hiddenLayers || [];
        this.outputNodes = outputCount || 0;

        // Build layerSizes array like SelfDrivingCar
        this.layerSizes = [this.inputNodes];
        for (const l of this.layers) {
            this.layerSizes.push(parseInt(l.units));
        }
        this.layerSizes.push(this.outputNodes);

        this.draw();
    }

    /**
     * Set weight/activation data for rich visualization.
     * Called from main.js after training with network.extractWeightsForVis()
     */
    setWeights(weights) {
        this.weights = weights;
    }

    setActivations(activations) {
        this.activations = activations;
    }

    /**
     * Set node labels for input/output layers.
     * inputLabels: ['sepal_length', 'is_red', ...]
     * outputLabels: ['species', ...]
     */
    setLabels(inputLabels, outputLabels) {
        this.inputLabels = inputLabels || [];
        this.outputLabels = outputLabels || [];
    }

    startAnimation() {
        if (this._isAnimating) return;
        this._isAnimating = true;
        let lastTick = 0;
        const tick = (now) => {
            if (!this._isAnimating) return;
            // Throttle to ~5 FPS — just enough for visual feedback, minimal GPU cost
            if (now - lastTick > 200) {
                this.animOffset += 3;
                this.draw();
                lastTick = now;
            }
            this._animFrame = requestAnimationFrame(tick);
        };
        this._animFrame = requestAnimationFrame(tick);
    }

    stopAnimation() {
        this._isAnimating = false;
        if (this._animFrame) {
            cancelAnimationFrame(this._animFrame);
            this._animFrame = null;
        }
        this.draw();
    }

    draw() {
        const ctx = this.ctx;
        const w = this._w || this.canvas.width;
        const h = this._h || this.canvas.height;

        // Read theme colors from CSS
        const bodyStyle = getComputedStyle(document.body);
        const labelColor = bodyStyle.getPropertyValue('--nn-node-label').trim() || '#8888a8';
        const mutedColor = bodyStyle.getPropertyValue('--text-muted').trim() || '#555570';
        const nnBg = bodyStyle.getPropertyValue('--nn-bg').trim();

        if (nnBg && nnBg !== 'transparent') {
            ctx.fillStyle = nnBg;
            ctx.fillRect(0, 0, w, h);
        } else {
            ctx.clearRect(0, 0, w, h);
        }

        if (this.inputNodes === 0 && this.outputNodes === 0) {
            ctx.fillStyle = mutedColor;
            ctx.font = '13px Inter';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Waiting for data configuration...', w / 2, h / 2);
            return;
        }

        const numLayers = this.layerSizes.length;
        const padding = 60;
        const layerSpacing = (w - padding * 2) / Math.max(1, numLayers - 1);

        const maxNodesDisplay = 16;
        const nodeRadius = 8;

        // --- Compute node positions ---
        const nodePositions = [];
        for (let l = 0; l < numLayers; l++) {
            const numNodes = Math.min(this.layerSizes[l], maxNodesDisplay);
            const truncated = this.layerSizes[l] > maxNodesDisplay;
            const nodeSpacing = Math.min(24, (h - padding * 2) / (numNodes + 1));
            const startY = h / 2 - (numNodes - 1) * nodeSpacing / 2;
            const x = padding + l * layerSpacing;
            const positions = [];

            for (let n = 0; n < numNodes; n++) {
                positions.push({ x, y: startY + n * nodeSpacing });
            }
            nodePositions.push(positions);

            // Layer header labels
            ctx.fillStyle = labelColor;
            ctx.font = '10px Inter';
            ctx.textAlign = 'center';
            if (l === 0) {
                ctx.fillText('Input', x, padding - 30);
            } else if (l === numLayers - 1) {
                ctx.fillText('Output', x, padding - 30);
            } else {
                ctx.fillText(`Hidden ${l}`, x, padding - 30);
            }
            // Node count
            ctx.fillStyle = mutedColor;
            ctx.fillText(`(${this.layerSizes[l]})`, x, padding - 18);

            // Activation label
            if (l > 0 && l < numLayers - 1 && this.layers[l - 1]) {
                ctx.fillStyle = mutedColor;
                ctx.font = '9px "JetBrains Mono"';
                ctx.fillText(this.layers[l - 1].activation, x, padding - 6);
            }

            if (truncated) {
                ctx.fillStyle = mutedColor;
                ctx.font = '9px Inter';
                ctx.fillText(`...${this.layerSizes[l] - maxNodesDisplay} more`, x, startY + numNodes * nodeSpacing + 14);
            }
        }

        // --- Draw connections (weight-colored like SelfDrivingCar) ---
        const weights = this.weights;
        for (let l = 0; l < nodePositions.length - 1; l++) {
            const from = nodePositions[l];
            const to = nodePositions[l + 1];
            const wData = weights ? weights[l] : null;

            for (let i = 0; i < from.length; i++) {
                for (let j = 0; j < to.length; j++) {
                    let wVal = 0;
                    if (wData && wData.data) {
                        const idx = i * wData.shape[1] + j;
                        if (idx < wData.data.length) wVal = wData.data[idx];
                    }

                    const absW = Math.min(Math.abs(wVal), 2) / 2;
                    const alpha = 0.05 + absW * 0.4;
                    const color = wVal >= 0
                        ? `rgba(85, 239, 196, ${alpha})`   // green for positive
                        : `rgba(255, 107, 107, ${alpha})`;  // red for negative

                    ctx.beginPath();
                    ctx.moveTo(from[i].x, from[i].y);
                    ctx.lineTo(to[j].x, to[j].y);
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 0.5 + absW * 1.5;
                    ctx.stroke();
                }
            }

            // Animated dash overlay during training
            if (this._isAnimating) {
                ctx.save();
                ctx.beginPath();
                for (let i = 0; i < from.length; i++) {
                    for (let j = 0; j < to.length; j++) {
                        ctx.moveTo(from[i].x, from[i].y);
                        ctx.lineTo(to[j].x, to[j].y);
                    }
                }
                ctx.setLineDash([3, 9]);
                ctx.lineDashOffset = -this.animOffset;
                ctx.strokeStyle = 'rgba(162, 155, 254, 0.15)';
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.restore();
            }
        }

        // --- Draw nodes (activation-colored like SelfDrivingCar) ---
        const activations = this.activations;
        for (let l = 0; l < nodePositions.length; l++) {
            const positions = nodePositions[l];
            const acts = activations && activations[l] ? activations[l] : null;

            for (let n = 0; n < positions.length; n++) {
                const { x, y } = positions[n];
                let activation = 0;
                if (acts && n < acts.length) activation = acts[n];

                const absAct = Math.min(Math.abs(activation), 1);

                // Soft glow ring
                ctx.beginPath();
                ctx.arc(x, y, nodeRadius + 2, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(108, 92, 231, ${0.1 + absAct * 0.3})`;
                ctx.fill();

                // Node fill with activation color
                ctx.beginPath();
                ctx.arc(x, y, nodeRadius, 0, Math.PI * 2);
                const nodeGrad = ctx.createRadialGradient(x, y, 0, x, y, nodeRadius);
                if (activation >= 0) {
                    nodeGrad.addColorStop(0, `rgba(85, 239, 196, ${0.3 + absAct * 0.7})`);
                    nodeGrad.addColorStop(1, `rgba(0, 206, 201, ${0.2 + absAct * 0.5})`);
                } else {
                    nodeGrad.addColorStop(0, `rgba(255, 107, 107, ${0.3 + absAct * 0.7})`);
                    nodeGrad.addColorStop(1, `rgba(238, 90, 36, ${0.2 + absAct * 0.5})`);
                }
                ctx.fillStyle = nodeGrad;
                ctx.fill();

                // Border ring
                ctx.strokeStyle = `rgba(162, 155, 254, ${0.3 + absAct * 0.5})`;
                ctx.lineWidth = 1;
                ctx.stroke();

                // Input labels — use actual feature names
                if (l === 0 && n < this.inputLabels.length) {
                    ctx.fillStyle = labelColor;
                    ctx.font = '9px Inter';
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'middle';
                    const lbl = this.inputLabels[n];
                    // Truncate long names
                    const maxLen = 12;
                    const display = lbl.length > maxLen ? lbl.slice(0, maxLen) + '…' : lbl;
                    ctx.fillText(display, x - nodeRadius - 6, y);
                } else if (l === 0 && n >= this.inputLabels.length) {
                    ctx.fillStyle = labelColor;
                    ctx.font = '9px Inter';
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(`in_${n + 1}`, x - nodeRadius - 6, y);
                }

                // Output labels — use actual target names
                if (l === nodePositions.length - 1 && n < this.outputLabels.length) {
                    ctx.fillStyle = labelColor;
                    ctx.font = '9px Inter';
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    const lbl = this.outputLabels[n];
                    const maxLen = 12;
                    const display = lbl.length > maxLen ? lbl.slice(0, maxLen) + '…' : lbl;
                    ctx.fillText(display, x + nodeRadius + 6, y);
                } else if (l === nodePositions.length - 1 && n >= this.outputLabels.length) {
                    ctx.fillStyle = labelColor;
                    ctx.font = '9px Inter';
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(`out_${n + 1}`, x + nodeRadius + 6, y);
                }
            }
        }
    }
}


export function showToast(message, type = "info") {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slide-in-toast 0.3s ease reverse forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
