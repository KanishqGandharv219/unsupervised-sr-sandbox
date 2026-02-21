
// viewer.js

// Global state
let currentSampleId = null;
let currentMethod = 'sr_hybrid'; // Default method

// DOM Elements
const sampleListEl = document.getElementById('sample-list');
const mainImageEl = document.getElementById('main-image');
const methodBtns = document.querySelectorAll('.method-btn');
const arcSharpnessValEl = document.getElementById('arc-sharpness-val');
const ringContrastValEl = document.getElementById('ring-contrast-val');
const descTextEl = document.getElementById('desc-text');
const loadingIndicator = document.getElementById('loading-indicator');

// Initialize Viewer
function initViewer() {
    if (typeof demoConfig === 'undefined') {
        console.error("Config not loaded!");
        return;
    }

    // Populate Sample List
    const sampleIds = Object.keys(demoConfig.samples);
    if (sampleIds.length === 0) return;

    sampleIds.forEach((id, index) => {
        const sample = demoConfig.samples[id];
        const item = document.createElement('div');
        item.className = 'sample-item';
        item.dataset.id = id;
        item.onclick = () => selectSample(id);

        item.innerHTML = `
            <img src="assets/samples/${id}/lr.png" class="sample-thumb" alt="Thumb">
            <div class="sample-meta">
                <strong>${sample.name}</strong>
            </div>
        `;
        sampleListEl.appendChild(item);

        if (index === 0) {
            selectSample(id);
        }
    });

    // Setup Method Buttons
    methodBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            selectMethod(btn.dataset.method);
        });
    });
}

// Select a Sample
function selectSample(id) {
    if (currentSampleId === id) return;
    currentSampleId = id;

    // Update active state in list
    document.querySelectorAll('.sample-item').forEach(el => {
        el.classList.toggle('active', el.dataset.id === id);
    });

    updateDisplay();
}

// Select a Method
function selectMethod(method) {
    if (currentMethod === method) return;
    currentMethod = method;

    // Update active button
    methodBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.method === method);
    });

    updateDisplay();
}

// Update Image and Metrics
function updateDisplay() {
    if (!currentSampleId) return;

    const sample = demoConfig.samples[currentSampleId];
    const metrics = sample.metrics[currentMethod];

    // Update Description
    descTextEl.textContent = sample.description || "No description available.";

    // Update Image
    // Construct path: assets/samples/{id}/{method}.png
    const imagePath = `assets/samples/${currentSampleId}/${currentMethod}.png`;

    // Show loading? (Optional, browser cache usually fast enough for local static)
    // loadingIndicator.style.display = 'block';

    mainImageEl.onload = () => {
        // loadingIndicator.style.display = 'none';
        mainImageEl.classList.remove('error');
    };

    mainImageEl.onerror = () => {
        console.error(`Failed to load ${imagePath}`);
        // Fallback or error state
        if (currentMethod !== 'lr') {
            // Try fallback to LR if SR fails
            mainImageEl.src = `assets/samples/${currentSampleId}/lr.png`;
            alert("Image missing, falling back to LR.");
        }
    };

    mainImageEl.src = imagePath;

    // Update Metrics
    // Fallback gracefully since individual sample sharpness isn't inside config.js per element yet (we only extracted aggregate table stats). We'll pull from global stats.
    if (demoConfig.stats && demoConfig.stats.arc_sharpness && demoConfig.stats.arc_sharpness[currentMethod]) {
        arcSharpnessValEl.textContent = demoConfig.stats.arc_sharpness[currentMethod].toFixed(2);
        ringContrastValEl.textContent = demoConfig.stats.ring_contrast[currentMethod].toFixed(2);

        arcSharpnessValEl.className = 'metric-val';
        ringContrastValEl.className = 'metric-val';
    } else {
        arcSharpnessValEl.textContent = "--";
        ringContrastValEl.textContent = "--";
    }
}

// Run init when DOM ready
document.addEventListener('DOMContentLoaded', initViewer);
