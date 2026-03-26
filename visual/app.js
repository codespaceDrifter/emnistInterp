// loaded from /api/config at startup
let CONFIG = null;

const GRID = 28;
const CELL = 15; // pixels per cell, canvas = 28 * 15 = 420
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// 28x28 float grid, 0.0 = background, 1.0 = stroke
const grid = Array.from({ length: GRID }, () => new Float32Array(GRID));

let drawing = false;

// --- drawing ---

function brush(gx, gy) {
    // center = full intensity
    setPixel(gx, gy, 1.0);
    // direct neighbors (cardinal) = medium falloff
    for (const [dx, dy] of [[0,1],[0,-1],[1,0],[-1,0]]) {
        setPixel(gx + dx, gy + dy, 0.4);
    }
    // diagonal neighbors = light falloff
    for (const [dx, dy] of [[1,1],[1,-1],[-1,1],[-1,-1]]) {
        setPixel(gx + dx, gy + dy, 0.2);
    }
}

function setPixel(x, y, val) {
    if (x >= 0 && x < GRID && y >= 0 && y < GRID) {
        grid[y][x] = Math.max(grid[y][x], val);
    }
}

function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let y = 0; y < GRID; y++) {
        for (let x = 0; x < GRID; x++) {
            const v = Math.round(grid[y][x] * 255);
            ctx.fillStyle = `rgb(${v},${v},${v})`;
            ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
        }
    }
}

function getGridCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    return [Math.floor(mx / CELL), Math.floor(my / CELL)];
}

canvas.addEventListener("mousedown", (e) => {
    drawing = true;
    const [gx, gy] = getGridCoords(e);
    brush(gx, gy);
    render();
});

canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    const [gx, gy] = getGridCoords(e);
    brush(gx, gy);
    render();
});

canvas.addEventListener("mouseup", () => { drawing = false; });
canvas.addEventListener("mouseleave", () => { drawing = false; });

document.getElementById("clear-btn").addEventListener("click", () => {
    for (let y = 0; y < GRID; y++) grid[y].fill(0);
    render();
    document.getElementById("results").innerHTML = "";
});

// --- prediction ---

document.getElementById("predict-btn").addEventListener("click", async () => {
    const image = [];
    for (let y = 0; y < GRID; y++) {
        image.push(Array.from(grid[y]));
    }

    const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config_name: CONFIG.model, image }),
    });
    const data = await resp.json();
    renderResults(data.predictions);
});

function renderResults(predictions) {
    const container = document.getElementById("results");
    container.innerHTML = "";
    for (const pred of predictions) {
        const pct = (pred.prob * 100).toFixed(1);
        const row = document.createElement("div");
        row.className = "pred-row";
        row.innerHTML = `
            <span class="pred-label">${pred.label}</span>
            <div class="pred-bar-bg">
                <div class="pred-bar" style="width: ${pct}%"></div>
            </div>
            <span class="pred-pct">${pct}%</span>
        `;
        container.appendChild(row);
    }
}

// --- load dataset sample ---

document.getElementById("load-sample-btn").addEventListener("click", async () => {
    const label = document.getElementById("sample-class").value;
    const index = parseInt(document.getElementById("sample-index").value) || 0;
    if (!label) return;

    const resp = await fetch(`/api/sample?label=${encodeURIComponent(label)}&index=${index}`);
    if (!resp.ok) {
        const err = await resp.text();
        console.error("Sample load failed:", err);
        alert(err);
        return;
    }
    const data = await resp.json();

    // load the 28x28 image onto the grid
    for (let y = 0; y < GRID; y++) {
        for (let x = 0; x < GRID; x++) {
            grid[y][x] = data.image[y][x];
        }
    }
    render();
});

// --- attribution tab ---

const attribCanvas = document.getElementById("attrib-canvas");
const attribCtx = attribCanvas.getContext("2d");
// separate 28x28 grid for the attribution tab
const attribGrid = Array.from({ length: GRID }, () => new Float32Array(GRID));
let attribDrawing = false;

function renderAttrib() {
    attribCtx.clearRect(0, 0, attribCanvas.width, attribCanvas.height);
    for (let y = 0; y < GRID; y++) {
        for (let x = 0; x < GRID; x++) {
            const v = Math.round(attribGrid[y][x] * 255);
            attribCtx.fillStyle = `rgb(${v},${v},${v})`;
            attribCtx.fillRect(x * CELL, y * CELL, CELL, CELL);
        }
    }
}

function attribBrush(gx, gy) {
    if (gx >= 0 && gx < GRID && gy >= 0 && gy < GRID)
        attribGrid[gy][gx] = Math.max(attribGrid[gy][gx], 1.0);
    for (const [dx, dy] of [[0,1],[0,-1],[1,0],[-1,0]])
        if (gx+dx >= 0 && gx+dx < GRID && gy+dy >= 0 && gy+dy < GRID)
            attribGrid[gy+dy][gx+dx] = Math.max(attribGrid[gy+dy][gx+dx], 0.4);
    for (const [dx, dy] of [[1,1],[1,-1],[-1,1],[-1,-1]])
        if (gx+dx >= 0 && gx+dx < GRID && gy+dy >= 0 && gy+dy < GRID)
            attribGrid[gy+dy][gx+dx] = Math.max(attribGrid[gy+dy][gx+dx], 0.2);
}

attribCanvas.addEventListener("mousedown", (e) => {
    attribDrawing = true;
    const rect = attribCanvas.getBoundingClientRect();
    attribBrush(Math.floor((e.clientX - rect.left) / CELL), Math.floor((e.clientY - rect.top) / CELL));
    renderAttrib();
});
attribCanvas.addEventListener("mousemove", (e) => {
    if (!attribDrawing) return;
    const rect = attribCanvas.getBoundingClientRect();
    attribBrush(Math.floor((e.clientX - rect.left) / CELL), Math.floor((e.clientY - rect.top) / CELL));
    renderAttrib();
});
attribCanvas.addEventListener("mouseup", () => { attribDrawing = false; });
attribCanvas.addEventListener("mouseleave", () => { attribDrawing = false; });

document.getElementById("attrib-clear-btn").addEventListener("click", () => {
    for (let y = 0; y < GRID; y++) attribGrid[y].fill(0);
    renderAttrib();
    document.getElementById("attrib-results").innerHTML = "";
});

// cached profile data for attribution cards
let neuronProfilesCache = null;
let saeProfilesCache = null;

// shared card renderer: template + class bars + real images
// profile may be null if not found in pre-computed profiles
function renderProfileCard(container, headerText, template, profile, globalMaxAct) {
    const card = document.createElement("div");
    card.className = "profile-card";

    const header = document.createElement("div");
    header.className = "profile-header";
    header.textContent = headerText;
    card.appendChild(header);

    const body = document.createElement("div");
    body.className = "profile-body";

    const tmpl = drawWeightTemplate(template, 84);
    tmpl.className = "profile-template";
    body.appendChild(tmpl);

    if (profile) {
        // class selectivity bars
        const bars = document.createElement("div");
        bars.className = "profile-bars";
        for (const cls of profile.top_classes) {
            const pct = globalMaxAct > 0 ? (cls.mean_act / globalMaxAct * 100) : 0;
            const row = document.createElement("div");
            row.className = "pred-row";
            row.innerHTML = `
                <span class="pred-label" style="font-size:16px">${cls.label}</span>
                <div class="pred-bar-bg">
                    <div class="pred-bar" style="width: ${pct}%"></div>
                </div>
                <span class="pred-pct">${cls.mean_act.toFixed(3)}</span>
            `;
            bars.appendChild(row);
        }
        body.appendChild(bars);

        // classifier weight bars (layer 1 neurons only)
        if (profile.top_classifier_weights) {
            const clsBars = document.createElement("div");
            clsBars.className = "profile-bars";
            const clsHeader = document.createElement("div");
            clsHeader.style.cssText = "color:#b97;font-size:11px;margin-bottom:2px";
            clsHeader.textContent = "pushes toward ↓";
            clsBars.appendChild(clsHeader);
            const maxW = Math.max(...profile.top_classifier_weights.map(w => Math.abs(w.weight)));
            for (const cls of profile.top_classifier_weights) {
                const pct = maxW > 0 ? (Math.abs(cls.weight) / maxW * 100) : 0;
                const row = document.createElement("div");
                row.className = "pred-row";
                row.innerHTML = `
                    <span class="pred-label" style="font-size:16px">${cls.label}</span>
                    <div class="pred-bar-bg">
                        <div class="pred-bar" style="width: ${pct}%; background: #d4956a"></div>
                    </div>
                    <span class="pred-pct">${cls.weight.toFixed(3)}</span>
                `;
                clsBars.appendChild(row);
            }
            body.appendChild(clsBars);
        }

        // top activating real images
        const imgsDiv = document.createElement("div");
        imgsDiv.className = "profile-images";
        for (const imgData of profile.top_images) {
            const c = document.createElement("canvas");
            c.width = 56;
            c.height = 56;
            const cx = c.getContext("2d");
            const cs = 56 / GRID;
            for (let y = 0; y < GRID; y++) {
                for (let x = 0; x < GRID; x++) {
                    const v = Math.round(imgData[y][x] * 255);
                    cx.fillStyle = `rgb(${v},${v},${v})`;
                    cx.fillRect(x * cs, y * cs, cs + 1, cs + 1);
                }
            }
            imgsDiv.appendChild(c);
        }
        body.appendChild(imgsDiv);
    }

    card.appendChild(body);
    container.appendChild(card);
}

// shared prediction header + bars
function renderPredictionHeader(container, data) {
    const header = document.createElement("div");
    header.style.cssText = "text-align:center;margin-bottom:16px;font-size:16px;color:#4a9;font-weight:bold";
    header.textContent = `Predicted: "${data.label}" (logit: ${data.logit})`;
    container.appendChild(header);

    for (const pred of data.predictions) {
        const pct = (pred.prob * 100).toFixed(1);
        const row = document.createElement("div");
        row.className = "pred-row";
        row.innerHTML = `
            <span class="pred-label">${pred.label}</span>
            <div class="pred-bar-bg">
                <div class="pred-bar" style="width: ${pct}%"></div>
            </div>
            <span class="pred-pct">${pct}%</span>
        `;
        container.appendChild(row);
    }
}

document.getElementById("attrib-btn").addEventListener("click", async () => {
    const k = parseInt(document.getElementById("attrib-k").value);

    const image = [];
    for (let y = 0; y < GRID; y++) image.push(Array.from(attribGrid[y]));

    // fetch neuron profiles if not cached
    if (!neuronProfilesCache) {
        const profileResp = await fetch("/interp_model/layer1_profiles.json");
        if (profileResp.ok) neuronProfilesCache = await profileResp.json();
    }

    const resp = await fetch("/api/mlp-attribute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image, k }),
    });
    const data = await resp.json();

    const container = document.getElementById("attrib-results");
    container.innerHTML = "";

    renderPredictionHeader(container, data);

    // find global max for bar normalization across attributed neurons
    let globalMaxAct = 0;
    if (neuronProfilesCache) {
        for (const n of data.neurons) {
            const profile = neuronProfilesCache.neurons.find(p => p.neuron_idx === n.neuron);
            if (profile) {
                for (const cls of profile.top_classes)
                    globalMaxAct = Math.max(globalMaxAct, cls.mean_act);
            }
        }
    }

    for (const neuron of data.neurons) {
        const profile = neuronProfilesCache
            ? neuronProfilesCache.neurons.find(p => p.neuron_idx === neuron.neuron)
            : null;
        renderProfileCard(
            container,
            `Neuron ${neuron.neuron} — activation: ${neuron.activation} — contribution: ${neuron.contribution}`,
            neuron.template,
            profile,
            globalMaxAct,
        );
    }
});


// load sample for attrib tab
document.getElementById("attrib-load-sample-btn").addEventListener("click", async () => {
    const label = document.getElementById("attrib-sample-class").value;
    const index = parseInt(document.getElementById("attrib-sample-index").value) || 0;
    if (!label) return;

    const resp = await fetch(`/api/sample?label=${encodeURIComponent(label)}&index=${index}`);
    if (!resp.ok) { alert(await resp.text()); return; }
    const data = await resp.json();

    for (let y = 0; y < GRID; y++)
        for (let x = 0; x < GRID; x++)
            attribGrid[y][x] = data.image[y][x];
    renderAttrib();
});

renderAttrib();

// --- SAE attribution tab ---

const saeCanvas = document.getElementById("sae-canvas");
const saeCtx = saeCanvas.getContext("2d");
const saeGrid = Array.from({ length: GRID }, () => new Float32Array(GRID));
let saeDrawing = false;

function renderSae() {
    saeCtx.clearRect(0, 0, saeCanvas.width, saeCanvas.height);
    for (let y = 0; y < GRID; y++) {
        for (let x = 0; x < GRID; x++) {
            const v = Math.round(saeGrid[y][x] * 255);
            saeCtx.fillStyle = `rgb(${v},${v},${v})`;
            saeCtx.fillRect(x * CELL, y * CELL, CELL, CELL);
        }
    }
}

function saeBrush(gx, gy) {
    if (gx >= 0 && gx < GRID && gy >= 0 && gy < GRID)
        saeGrid[gy][gx] = Math.max(saeGrid[gy][gx], 1.0);
    for (const [dx, dy] of [[0,1],[0,-1],[1,0],[-1,0]])
        if (gx+dx >= 0 && gx+dx < GRID && gy+dy >= 0 && gy+dy < GRID)
            saeGrid[gy+dy][gx+dx] = Math.max(saeGrid[gy+dy][gx+dx], 0.4);
    for (const [dx, dy] of [[1,1],[1,-1],[-1,1],[-1,-1]])
        if (gx+dx >= 0 && gx+dx < GRID && gy+dy >= 0 && gy+dy < GRID)
            saeGrid[gy+dy][gx+dx] = Math.max(saeGrid[gy+dy][gx+dx], 0.2);
}

saeCanvas.addEventListener("mousedown", (e) => {
    saeDrawing = true;
    const rect = saeCanvas.getBoundingClientRect();
    saeBrush(Math.floor((e.clientX - rect.left) / CELL), Math.floor((e.clientY - rect.top) / CELL));
    renderSae();
});
saeCanvas.addEventListener("mousemove", (e) => {
    if (!saeDrawing) return;
    const rect = saeCanvas.getBoundingClientRect();
    saeBrush(Math.floor((e.clientX - rect.left) / CELL), Math.floor((e.clientY - rect.top) / CELL));
    renderSae();
});
saeCanvas.addEventListener("mouseup", () => { saeDrawing = false; });
saeCanvas.addEventListener("mouseleave", () => { saeDrawing = false; });

document.getElementById("sae-clear-btn").addEventListener("click", () => {
    for (let y = 0; y < GRID; y++) saeGrid[y].fill(0);
    renderSae();
    document.getElementById("sae-features").innerHTML = "";
    document.getElementById("sae-status").textContent = "";
});

// drag-and-drop image onto SAE canvas
saeCanvas.addEventListener("dragover", (e) => { e.preventDefault(); });
saeCanvas.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = () => {
        const img = new Image();
        img.onload = () => {
            // draw image onto a 28x28 offscreen canvas to extract pixel data
            const c = document.createElement("canvas");
            c.width = GRID; c.height = GRID;
            const cx = c.getContext("2d");
            cx.drawImage(img, 0, 0, GRID, GRID);
            const data = cx.getImageData(0, 0, GRID, GRID).data;
            for (let y = 0; y < GRID; y++) {
                for (let x = 0; x < GRID; x++) {
                    const i = (y * GRID + x) * 4;
                    // grayscale from RGB
                    saeGrid[y][x] = (data[i] * 0.299 + data[i+1] * 0.587 + data[i+2] * 0.114) / 255;
                }
            }
            renderSae();
        };
        img.src = reader.result;
    };
    reader.readAsDataURL(file);
});

document.getElementById("sae-attrib-btn").addEventListener("click", async () => {
    const k = parseInt(document.getElementById("sae-k").value);
    const status = document.getElementById("sae-status");
    status.textContent = "Loading...";

    const image = [];
    for (let y = 0; y < GRID; y++) image.push(Array.from(saeGrid[y]));

    // fetch SAE feature profiles if not cached
    if (!saeProfilesCache) {
        const profileResp = await fetch(`/interp_probes/layer${CONFIG.sae.layer}_exp${CONFIG.sae.expansion}_profiles.json`);
        if (profileResp.ok) saeProfilesCache = await profileResp.json();
    }

    const resp = await fetch("/api/sae-attribute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            config_name: CONFIG.model,
            image, layer: CONFIG.sae.layer, expansion: CONFIG.sae.expansion, k,
        }),
    });
    const data = await resp.json();
    status.textContent = `Top-${k} explains ${data.r_squared}% of activations`;

    const resultsContainer = document.getElementById("sae-results");
    resultsContainer.innerHTML = "";
    renderPredictionHeader(resultsContainer, data);

    const container = document.getElementById("sae-features");
    container.innerHTML = "";

    // find global max for bar normalization across attributed features
    let globalMaxAct = 0;
    if (saeProfilesCache) {
        for (const f of data.features) {
            const profile = saeProfilesCache.features.find(p => p.feature_idx === f.feature_idx);
            if (profile) {
                for (const cls of profile.top_classes)
                    globalMaxAct = Math.max(globalMaxAct, cls.mean_act);
            }
        }
    }

    for (const feat of data.features) {
        const profile = saeProfilesCache
            ? saeProfilesCache.features.find(p => p.feature_idx === feat.feature_idx)
            : null;
        renderProfileCard(
            container,
            `Feature ${feat.feature_idx} — activation: ${feat.activation} — contribution: ${feat.contribution}`,
            feat.template,
            profile,
            globalMaxAct,
        );
    }
});

function drawWeightTemplate(template, size) {
    const c = document.createElement("canvas");
    c.width = size;
    c.height = size;
    const cx = c.getContext("2d");
    let vmax = 0;
    for (let y = 0; y < GRID; y++)
        for (let x = 0; x < GRID; x++)
            vmax = Math.max(vmax, Math.abs(template[y][x]));
    if (vmax === 0) vmax = 1;
    const cs = size / GRID;
    for (let y = 0; y < GRID; y++) {
        for (let x = 0; x < GRID; x++) {
            const v = Math.round(128 + (template[y][x] / vmax) * 127);
            cx.fillStyle = `rgb(${v},${v},${v})`;
            cx.fillRect(x * cs, y * cs, cs + 1, cs + 1);
        }
    }
    return c;
}

// load sample for SAE tab
document.getElementById("sae-load-sample-btn").addEventListener("click", async () => {
    const label = document.getElementById("sae-sample-class").value;
    const index = parseInt(document.getElementById("sae-sample-index").value) || 0;
    if (!label) return;
    const resp = await fetch(`/api/sample?label=${encodeURIComponent(label)}&index=${index}`);
    if (!resp.ok) { alert(await resp.text()); return; }
    const data = await resp.json();
    for (let y = 0; y < GRID; y++)
        for (let x = 0; x < GRID; x++)
            saeGrid[y][x] = data.image[y][x];
    renderSae();
});

renderSae();

// --- MLP SAE profile tab ---

const profileLayerSelect = document.getElementById("sae-profile-layer");
const profileExpSelect = document.getElementById("sae-profile-exp");
const profileStatsDiv = document.getElementById("sae-profile-stats");
const profileFeaturesDiv = document.getElementById("sae-profile-features");

async function loadProfiles() {
    const layer = profileLayerSelect.value;
    const exp = profileExpSelect.value;
    profileStatsDiv.textContent = "Loading...";
    profileFeaturesDiv.innerHTML = "";

    let data;
    try {
        const resp = await fetch(`/interp_probes/layer${layer}_exp${exp}_profiles.json`);
        if (!resp.ok) throw new Error("not found");
        data = await resp.json();
    } catch {
        profileStatsDiv.textContent = `No profile found for layer ${layer}, ${exp}x expansion. Run analyze.py first.`;
        return;
    }

    // stats header
    profileStatsDiv.textContent =
        `${data.tag} — dict_size: ${data.dict_size} | dead: ${data.num_dead} | mean active/sample: ${data.mean_active} | recon MSE: ${data.recon_mse}`;

    // find max mean_act across all features for normalizing bars
    let globalMaxAct = 0;
    for (const feat of data.features) {
        for (const cls of feat.top_classes) {
            globalMaxAct = Math.max(globalMaxAct, cls.mean_act);
        }
    }

    for (const feat of data.features) {
        const card = document.createElement("div");
        card.className = "profile-card";

        // header
        const header = document.createElement("div");
        header.className = "profile-header";
        header.textContent = `Feature ${feat.feature_idx} — ${feat.pct_active}% active (${feat.num_active} samples)`;
        card.appendChild(header);

        const body = document.createElement("div");
        body.className = "profile-body";

        // weight template (leftmost)
        if (feat.template) {
            const tmpl = drawWeightTemplate(feat.template, 84);
            tmpl.className = "profile-template";
            body.appendChild(tmpl);
        }

        // class bars
        const bars = document.createElement("div");
        bars.className = "profile-bars";
        for (const cls of feat.top_classes) {
            const pct = globalMaxAct > 0 ? (cls.mean_act / globalMaxAct * 100) : 0;
            const row = document.createElement("div");
            row.className = "pred-row";
            row.innerHTML = `
                <span class="pred-label" style="font-size:16px">${cls.label}</span>
                <div class="pred-bar-bg">
                    <div class="pred-bar" style="width: ${pct}%"></div>
                </div>
                <span class="pred-pct">${cls.mean_act.toFixed(3)}</span>
            `;
            bars.appendChild(row);
        }
        body.appendChild(bars);

        // classifier weight bars (effective: W_cls @ decoder_col)
        if (feat.top_classifier_weights) {
            const clsBars = document.createElement("div");
            clsBars.className = "profile-bars";
            const clsHeader = document.createElement("div");
            clsHeader.style.cssText = "color:#b97;font-size:11px;margin-bottom:2px";
            clsHeader.textContent = "pushes toward ↓";
            clsBars.appendChild(clsHeader);
            const maxW = Math.max(...feat.top_classifier_weights.map(w => Math.abs(w.weight)));
            for (const cls of feat.top_classifier_weights) {
                const pct = maxW > 0 ? (Math.abs(cls.weight) / maxW * 100) : 0;
                const row = document.createElement("div");
                row.className = "pred-row";
                row.innerHTML = `
                    <span class="pred-label" style="font-size:16px">${cls.label}</span>
                    <div class="pred-bar-bg">
                        <div class="pred-bar" style="width: ${pct}%; background: #d4956a"></div>
                    </div>
                    <span class="pred-pct">${cls.weight.toFixed(3)}</span>
                `;
                clsBars.appendChild(row);
            }
            body.appendChild(clsBars);
        }

        // top real images
        const imgsDiv = document.createElement("div");
        imgsDiv.className = "profile-images";
        for (const imgData of feat.top_images) {
            const c = document.createElement("canvas");
            c.width = 56;
            c.height = 56;
            const cx = c.getContext("2d");
            // 28x28 float data -> 56x56 canvas (2x upscale)
            const cs = 56 / GRID;
            for (let y = 0; y < GRID; y++) {
                for (let x = 0; x < GRID; x++) {
                    const v = Math.round(imgData[y][x] * 255);
                    cx.fillStyle = `rgb(${v},${v},${v})`;
                    cx.fillRect(x * cs, y * cs, cs + 1, cs + 1);
                }
            }
            imgsDiv.appendChild(c);
        }
        body.appendChild(imgsDiv);
        card.appendChild(body);
        profileFeaturesDiv.appendChild(card);
    }
}

profileLayerSelect.addEventListener("change", loadProfiles);
profileExpSelect.addEventListener("change", loadProfiles);
loadProfiles();

// --- MLP neuron profiles ---

const neuronLayerSelect = document.getElementById("neuron-profile-layer");
const neuronFeaturesDiv = document.getElementById("neuron-profile-features");

async function loadNeuronProfiles() {
    const layer = neuronLayerSelect.value;
    neuronFeaturesDiv.innerHTML = "<p style='color:#888;text-align:center'>Loading...</p>";

    let data;
    try {
        const resp = await fetch(`/interp_model/layer${layer}_profiles.json`);
        if (!resp.ok) throw new Error("not found");
        data = await resp.json();
    } catch {
        neuronFeaturesDiv.innerHTML = `<p style='color:#888;text-align:center'>No neuron profiles for layer ${layer}. Run run_interp.py first.</p>`;
        return;
    }

    neuronFeaturesDiv.innerHTML = "";

    // max mean_act across all neurons for normalizing bars
    let globalMaxAct = 0;
    for (const n of data.neurons) {
        for (const cls of n.top_classes) {
            globalMaxAct = Math.max(globalMaxAct, cls.mean_act);
        }
    }

    for (const neuron of data.neurons) {
        const card = document.createElement("div");
        card.className = "profile-card";

        const header = document.createElement("div");
        header.className = "profile-header";
        header.textContent = `Neuron ${neuron.neuron_idx} — ${neuron.pct_active}% active (${neuron.num_active} samples)`;
        card.appendChild(header);

        const body = document.createElement("div");
        body.className = "profile-body";

        // weight template (leftmost)
        if (neuron.template) {
            const tmpl = drawWeightTemplate(neuron.template, 84);
            tmpl.className = "profile-template";
            body.appendChild(tmpl);
        }

        // class bars
        const bars = document.createElement("div");
        bars.className = "profile-bars";
        for (const cls of neuron.top_classes) {
            const pct = globalMaxAct > 0 ? (cls.mean_act / globalMaxAct * 100) : 0;
            const row = document.createElement("div");
            row.className = "pred-row";
            row.innerHTML = `
                <span class="pred-label" style="font-size:16px">${cls.label}</span>
                <div class="pred-bar-bg">
                    <div class="pred-bar" style="width: ${pct}%"></div>
                </div>
                <span class="pred-pct">${cls.mean_act.toFixed(3)}</span>
            `;
            bars.appendChild(row);
        }
        body.appendChild(bars);

        // classifier weight bars (layer 1 neurons only)
        if (neuron.top_classifier_weights) {
            const clsBars = document.createElement("div");
            clsBars.className = "profile-bars";
            const clsHeader = document.createElement("div");
            clsHeader.style.cssText = "color:#b97;font-size:11px;margin-bottom:2px";
            clsHeader.textContent = "pushes toward ↓";
            clsBars.appendChild(clsHeader);
            const maxW = Math.max(...neuron.top_classifier_weights.map(w => Math.abs(w.weight)));
            for (const cls of neuron.top_classifier_weights) {
                const pct = maxW > 0 ? (Math.abs(cls.weight) / maxW * 100) : 0;
                const row = document.createElement("div");
                row.className = "pred-row";
                row.innerHTML = `
                    <span class="pred-label" style="font-size:16px">${cls.label}</span>
                    <div class="pred-bar-bg">
                        <div class="pred-bar" style="width: ${pct}%; background: #d4956a"></div>
                    </div>
                    <span class="pred-pct">${cls.weight.toFixed(3)}</span>
                `;
                clsBars.appendChild(row);
            }
            body.appendChild(clsBars);
        }

        // top real images
        const imgsDiv = document.createElement("div");
        imgsDiv.className = "profile-images";
        for (const imgData of neuron.top_images) {
            const c = document.createElement("canvas");
            c.width = 56;
            c.height = 56;
            const cx = c.getContext("2d");
            const cs = 56 / GRID;
            for (let y = 0; y < GRID; y++) {
                for (let x = 0; x < GRID; x++) {
                    const v = Math.round(imgData[y][x] * 255);
                    cx.fillStyle = `rgb(${v},${v},${v})`;
                    cx.fillRect(x * cs, y * cs, cs + 1, cs + 1);
                }
            }
            imgsDiv.appendChild(c);
        }
        body.appendChild(imgsDiv);
        card.appendChild(body);
        neuronFeaturesDiv.appendChild(card);
    }
}

if (neuronLayerSelect) {
    neuronLayerSelect.addEventListener("change", loadNeuronProfiles);
    loadNeuronProfiles();
}

// --- tabs ---

for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
        document.querySelector(".tab.active").classList.remove("active");
        document.querySelector(".tab-content.active").classList.remove("active");
        tab.classList.add("active");
        document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
    });
}

// --- highlight toggle ---
// precompute highlighted versions once, then swap src instantly
// {original_src: highlighted_data_url}
const highlightCache = new Map();
const originalSrcs = new Map();
const HIGHLIGHT_THRESH = 0.7 * 255;  // 70% brightness

function buildHighlightCache() {
    const images = document.querySelectorAll(".interp-section img");
    let loaded = 0;
    for (const img of images) {
        if (highlightCache.has(img.src)) continue;
        const src = img.src;
        const source = new Image();
        source.crossOrigin = "anonymous";
        source.src = src;
        source.onload = () => {
            const c = document.createElement("canvas");
            c.width = source.width;
            c.height = source.height;
            const cx = c.getContext("2d");
            cx.drawImage(source, 0, 0);
            const data = cx.getImageData(0, 0, c.width, c.height);
            const d = data.data;
            for (let i = 0; i < d.length; i += 4) {
                const brightness = d[i] * 0.299 + d[i+1] * 0.587 + d[i+2] * 0.114;
                // only highlight grayscale pixels (skip colored text)
                const isGray = Math.abs(d[i] - d[i+1]) < 30 && Math.abs(d[i+1] - d[i+2]) < 30;
                if (isGray && brightness > HIGHLIGHT_THRESH) {
                    d[i] = 255;
                    d[i+1] = 40;
                    d[i+2] = 40;
                }
            }
            cx.putImageData(data, 0, 0);
            highlightCache.set(src, c.toDataURL());
        };
    }
}

// build cache after images load
window.addEventListener("load", buildHighlightCache);

document.getElementById("highlight-toggle").addEventListener("change", () => {
    const on = document.getElementById("highlight-toggle").checked;
    for (const img of document.querySelectorAll(".interp-section img")) {
        if (!originalSrcs.has(img)) originalSrcs.set(img, img.src);
        img.src = on ? (highlightCache.get(originalSrcs.get(img)) || img.src) : originalSrcs.get(img);
    }
});

// init — fetch config then render
(async () => {
    const resp = await fetch("/api/config");
    CONFIG = await resp.json();
    render();
})();
