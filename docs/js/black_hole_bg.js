// black_hole_bg.js - Rendering the Black Hole GLTF

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({
    canvas: document.querySelector('#bg-canvas'),
    alpha: true,
    antialias: true
});

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputEncoding = THREE.sRGBEncoding;

// --- Starfield Removed per Request ---

// --- Black Hole GLTF Loader ---
const loader = new THREE.GLTFLoader();
const targetGroup = new THREE.Group();
scene.add(targetGroup);

loader.load('black_hole/scene.gltf', (gltf) => {
    const model = gltf.scene;

    // Automatically scale the model to fit a standard viewport size
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    // Normalize size
    const targetSize = 6.0;
    const scale = targetSize / maxDim;
    model.scale.set(scale, scale, scale);

    // Center model at origin
    model.position.sub(center.multiplyScalar(scale));

    // Enhance emissiveness if present for glow effect
    model.traverse((child) => {
        if (child.isMesh) {
            // Check if material has an emissive map and bump its intensity
            if (child.material && child.material.emissiveIntensity !== undefined) {
                child.material.emissiveIntensity *= 2.0;
            }
        }
    });

    targetGroup.add(model);
}, undefined, (error) => {
    console.error('An error happened while loading the black hole model:', error);
});

// Ambient lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 2.0); // Boosted ambient light for visibility
scene.add(ambientLight);

// Slightly angled directional light to give the accretion disk some shape
const dirLight = new THREE.DirectionalLight(0xffbbaa, 3.0);
dirLight.position.set(5, 5, 2);
scene.add(dirLight);

// --- Positioning ---
camera.position.z = 15; // Moved camera further back just in case the scale logic clipped it
camera.position.y = 2;

// --- Orbit Controls ---
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.0;

// --- Mouse Parallax ---
let mouseX = 0;
let mouseY = 0;
document.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX / window.innerWidth) * 2 - 1;
    mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
});

// --- Animation Loop ---
function animate() {
    requestAnimationFrame(animate);

    controls.update(); // Update orbital controls

    // Subtle drift rotation for the black hole
    if (targetGroup) {
        targetGroup.rotation.y += 0.001;
        // Introduce mouse parallax
        targetGroup.rotation.x += (mouseY * 0.1 - targetGroup.rotation.x) * 0.05;
        targetGroup.rotation.z += (mouseX * 0.1 - targetGroup.rotation.z) * 0.05;
    }



    renderer.render(scene, camera);
}

// Handle Window Resizing
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

animate();
