// background.js - Phenomenal Earth (Three.js Journey Style)

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({
    canvas: document.querySelector('#bg-canvas'),
    alpha: true,
    antialias: true
});

renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

// --- Load Textures (Three.js Journey Standards) ---
const textureLoader = new THREE.TextureLoader();

const dayTexture = textureLoader.load('assets/textures/day.jpg');
dayTexture.colorSpace = THREE.SRGBColorSpace; // CRITICAL: Correct color profile via the User's lesson

const nightTexture = textureLoader.load('assets/textures/night.jpg');
nightTexture.colorSpace = THREE.SRGBColorSpace; // CRITICAL: Correct color profile

const specularCloudsTexture = textureLoader.load('assets/textures/specularClouds.jpg');
// specularClouds is Linear (default), which is correct for data textures

// --- Earth Material ---
const earthMaterial = new THREE.ShaderMaterial({
    uniforms: {
        uDayTexture: { value: dayTexture },
        uNightTexture: { value: nightTexture },
        uSpecularCloudsTexture: { value: specularCloudsTexture },
        uSunDirection: { value: new THREE.Vector3(0, 0, 1) }, // Default to front
        uAtmosphereDayColor: { value: new THREE.Color(0x4db2ff) },
        uAtmosphereTwilightColor: { value: new THREE.Color(0x652706) }

    },
    vertexShader: `
        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vPosition;

        void main() {
            vUv = uv;
            vNormal = normalize(normalMatrix * normal);
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            vPosition = mvPosition.xyz;
            gl_Position = projectionMatrix * mvPosition;
        }
    `,
    fragmentShader: `
        uniform sampler2D uDayTexture;
        uniform sampler2D uNightTexture;
        uniform sampler2D uSpecularCloudsTexture;
        uniform vec3 uSunDirection;
        uniform vec3 uAtmosphereDayColor;
        uniform vec3 uAtmosphereTwilightColor;

        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vPosition;

        void main() {
            vec3 viewDirection = normalize(-vPosition);
            vec3 normal = normalize(vNormal);
            vec3 sunDir = normalize(uSunDirection);

            // --- Textures ---
            vec3 dayColor = texture2D(uDayTexture, vUv).rgb;
            vec3 nightColor = texture2D(uNightTexture, vUv).rgb;
            
            // Channel Packing from Lesson: R=Specular, G=Clouds
            vec2 specularClouds = texture2D(uSpecularCloudsTexture, vUv).rg;
            float specularMask = specularClouds.r;
            float cloudStrength = specularClouds.g;

            // --- Day / Night Mix ---
            float sunOrientation = dot(normal, sunDir);
            float dayMix = smoothstep(-0.25, 0.25, sunOrientation);

            // --- Lighting Mechanics ---
            
            // 1. Day Color
            vec3 color = dayColor;

            // 2. Night Color (Cities visible only at night)
            // Cities logic: mix night texture based on inverse dayMix
            color = mix(nightColor, color, dayMix);

            // 3. Specular (Sun Reflection on Ocean)
            vec3 reflection = reflect(-sunDir, normal);
            // Sharp specular highlight
            float specular = -dot(reflection, viewDirection);
            specular = max(0.0, specular);
            specular = pow(specular, 32.0);
            
            // Mask by texture (only ocean) && dayMix (only day side)
            vec3 specularColor = vec3(1.0) * specular * specularMask * dayMix;
            color += specularColor;

            // 4. Clouds
            // Clouds are white. They hide the surface.
            // Mix current color with white based on cloud strength.
            // But clouds are also affected by day/night!
            // Night clouds should be dark (or block city lights).
            
            // Three.js Journey usually does:
            // Mix surface with cloud color (white).
            // Then apply lighting (shadows on clouds).
            // For simplicity/impact:
            vec3 cloudColor = vec3(1.0) * dayMix; // Illuminated clouds
            // Or just add them on top?
            
            // Let's do:
            // Blend surface to white based on cloud density
            color = mix(color, vec3(1.0), cloudStrength * dayMix); 


            // 5. Atmosphere
            float fresnel = dot(viewDirection, normal);
            fresnel = clamp(1.0 - fresnel, 0.0, 1.0);
            fresnel = pow(fresnel, 3.0);

            // Atmosphere Color Mixing (Day vs Twilight)
            float atmosMix = smoothstep(-0.5, 0.5, sunOrientation);
            vec3 atmosphereColor = mix(uAtmosphereTwilightColor, uAtmosphereDayColor, atmosMix);

            color += atmosphereColor * fresnel * 0.6; // 0.6 = Intensity

            gl_FragColor = vec4(color, 1.0);
            
            // Color Correction (Three.js Journey uses specific logic, but simple Linear output is fine if Renderer handles encoding)
            // Since we use r128, manual encoding might be needed if outputEncoding isn't set.
            // But let's stick to this for now.
        }
    `
});

// --- Mesh & Real Physics ---
const earthGeometry = new THREE.SphereGeometry(2, 64, 64);
const earth = new THREE.Mesh(earthGeometry, earthMaterial);

// Earth is an oblate spheroid. 
// Equatorial radius: 6378.137 km, Polar radius: 6356.752 km
// Ratio (Polar / Equatorial) = 0.996647189
// Assuming X and Z are equatorial axes (y is polar)
earth.scale.set(1.0, 0.996647189, 1.0);

// Nest Earth in a group to handle Axial Tilt independently from rotation
const earthGroup = new THREE.Group();
earthGroup.add(earth);

// Real Earth Axial Tilt is ~23.5 degrees
// Rotate the group so the entire axis is tilted
earthGroup.rotation.z = 23.5 * Math.PI / 180;

scene.add(earthGroup);

// --- Sun Helper ---
// A visual representation of where the sun is
const sunGeometry = new THREE.SphereGeometry(0.1, 16, 16);
const sunMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });
const sunMesh = new THREE.Mesh(sunGeometry, sunMaterial);
sunMesh.visible = false; // Hidden per request
scene.add(sunMesh);

// --- Position ---
earthGroup.position.x = 2.5;
camera.position.z = 6;

// --- Animation ---
const sunSpherical = new THREE.Spherical(1, 1.64305295782746, -1.784424627239); // Using Spherical coords for easier rotation
const sunDirection = new THREE.Vector3();

// Mouse Parallax
let mouseX = 0;
let mouseY = 0;
document.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX / window.innerWidth) * 2 - 1;
    mouseY = -(e.clientY / window.innerHeight) * 2 + 1;
});

function animate() {
    requestAnimationFrame(animate);

    // Rotate Earth
    earth.rotation.y += 0.0005;

    // Mouse Parallax
    earth.rotation.x += (mouseY * 0.05 - earth.rotation.x) * 0.05;

    // Update Sun Position (Orbiting)
    // To make it dynamic so user sees the "Source of Light" moving?
    // Or fixed? Let's make it fixed but controllable via GUI.
    sunDirection.setFromSpherical(sunSpherical);

    // Update Shader
    earthMaterial.uniforms.uSunDirection.value.copy(sunDirection);

    // Update Helper
    sunMesh.position.copy(sunDirection).multiplyScalar(5); // Put explicit distance for helper

    renderer.render(scene, camera);
    stats.update(); // Update FPS
}

// Handle Resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});

// --- Starfield (Procedural) ---
const starGeometry = new THREE.BufferGeometry();
const starCount = 3000;
const starPos = new Float32Array(starCount * 3);
const starColors = new Float32Array(starCount * 3);

for (let i = 0; i < starCount; i++) {
    const x = (Math.random() - 0.5) * 100; // Wide range
    const y = (Math.random() - 0.5) * 100;
    const z = (Math.random() - 0.5) * 100;

    // Push away from center so stars aren't inside the Earth
    // If distance is too small, push it out
    // Simple way: just spawn them far away
    const dist = Math.sqrt(x * x + y * y + z * z);
    if (dist < 10) {
        // Just discard or push?
        // Let's just spawn them in a shell 
        // e.g. radius 20 to 50
    }
    // Better: Spherical coords
    const r = 30 + Math.random() * 60;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);

    starPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    starPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    starPos[i * 3 + 2] = r * Math.cos(phi);

    // Color: mostly white, some slight blue/yellow tint
    const starType = Math.random();
    if (starType > 0.9) { // Blueish
        starColors[i * 3] = 0.8;
        starColors[i * 3 + 1] = 0.8;
        starColors[i * 3 + 2] = 1.0;
    } else if (starType > 0.8) { // Yellowish
        starColors[i * 3] = 1.0;
        starColors[i * 3 + 1] = 1.0;
        starColors[i * 3 + 2] = 0.8;
    } else {
        starColors[i * 3] = 1.0;
        starColors[i * 3 + 1] = 1.0;
        starColors[i * 3 + 2] = 1.0;
    }
}

starGeometry.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
starGeometry.setAttribute('color', new THREE.BufferAttribute(starColors, 3));

const starMaterial = new THREE.PointsMaterial({
    size: 0.2, // Small dots
    vertexColors: true,
    transparent: true,
    opacity: 0.8,
    sizeAttenuation: true
});

const stars = new THREE.Points(starGeometry, starMaterial);
scene.add(stars);


// --- Custom premium FPS Counter ---
const fpsElement = document.createElement('div');
fpsElement.style.position = 'fixed';
fpsElement.style.bottom = '10px'; // Revert to bottom
fpsElement.style.left = '10px';   // Revert to left
fpsElement.style.color = 'var(--text-primary)';
fpsElement.style.fontFamily = 'Inter, sans-serif';
fpsElement.style.fontSize = '12px';
fpsElement.style.fontWeight = '600';
fpsElement.style.background = 'var(--bg-elevated)';
fpsElement.style.border = '1px solid var(--accent-primary)';
fpsElement.style.padding = '8px 12px';
fpsElement.style.borderRadius = '8px';
fpsElement.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.5)';
fpsElement.style.backdropFilter = 'blur(10px)';
fpsElement.style.letterSpacing = '1px';
fpsElement.innerText = '120 FPS'; // Initial placeholder
document.body.appendChild(fpsElement);

let frameCount = 0;
let lastTime = performance.now();


// --- GUI Controls (Simplified) ---
let gui;
// Declare debugObject outside so it's globally accessible to animate()
const debugObject = {
    sunPhi: 1.64305295782746,
    sunTheta: -1.784424627239,
    atmosDay: '#4db2ff',
    atmosTwilight: '#652706',
    bgHex: '#000000',
    rotationSpeed: 0.0005
};

try {
    gui = new lil.GUI({ title: 'Earth Control' });
    gui.close(); // Collapse by default

    gui.add(debugObject, 'rotationSpeed', 0, 0.01, 0.0001).name('Rotation Speed');
    gui.add(debugObject, 'sunPhi', 0, Math.PI).name('Sun Latitude').onChange(val => {
        sunSpherical.phi = val;
    });
    gui.add(debugObject, 'sunTheta', -Math.PI, Math.PI).name('Sun Longitude').onChange(val => {
        sunSpherical.theta = val;
    });
    gui.addColor(debugObject, 'atmosDay').name('Day Atmosphere').onChange(val => {
        earthMaterial.uniforms.uAtmosphereDayColor.value.set(val);
    });
    gui.addColor(debugObject, 'atmosTwilight').name('Twilight Color').onChange(val => {
        earthMaterial.uniforms.uAtmosphereTwilightColor.value.set(val);
    });
    gui.addColor(debugObject, 'bgHex').name('Space Color').onChange(val => {
        // Change canvas background
        renderer.domElement.style.background = val;
    });
} catch (e) {
    console.warn("GUI failed", e);
}

function animate() {
    requestAnimationFrame(animate);

    // Rotate Earth based on GUI control
    if (typeof debugObject !== 'undefined') {
        earth.rotation.y += debugObject.rotationSpeed;
    } else {
        earth.rotation.y += 0.0005;
    }

    // Rotate Stars (Slowly)
    if (stars) stars.rotation.y -= 0.0002;

    // Mouse Parallax applied to the Group so tilt is preserved
    earthGroup.rotation.x += (mouseY * 0.05 - earthGroup.rotation.x) * 0.05;

    // Update Sun Position (Orbiting)
    sunDirection.setFromSpherical(sunSpherical);

    // Update Shader
    earthMaterial.uniforms.uSunDirection.value.copy(sunDirection);

    // Update Helper
    if (sunMesh) sunMesh.position.copy(sunDirection).multiplyScalar(5);

    renderer.render(scene, camera);

    // FPS Calculation
    frameCount++;
    const currentTime = performance.now();
    if (currentTime > lastTime + 1000) {
        const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
        fpsElement.innerText = 'FPS: ' + fps;
        frameCount = 0;
        lastTime = currentTime;
    }
}

animate();
