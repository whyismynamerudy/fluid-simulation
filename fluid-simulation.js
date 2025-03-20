// Enhanced Body Detection Fluid Simulation
// Uses Three.js for rendering and TensorFlow.js PoseNet for body detection

// Global variables
let camera, scene, renderer;
let videoElement, videoTexture, videoMaterial;
let particleSystem, particleUniforms;
let bodyOutlineMesh;
let canvasForProcessing, ctxForProcessing;
let debugCanvas, debugCtx;
let poseNetModel;
let currentPose = null;
let lastPoseTime = 0;
let bodySegmentation = null;
let lastInteractionPoints = [];
let fluidField;

// Configuration
const config = {
  particleCount: 5000,
  particleSize: 0.05,
  particleColor: 0x4FC3F7, // Light blue particles
  particleOpacity: 0.6,
  viscosity: 0.97,
  repulsionForce: 0.08,
  repulsionDistance: 1.0,
  bodyOutlineColor: 0xFFFFFF,
  trailEffect: true,
  colorVariation: true,
  flowFieldStrength: 0.02,
  particleSpeedLimit: 0.1,
  glowEffect: true,
  particleLifeSpan: { min: 2000, max: 6000 }, // milliseconds
  useShaders: true,
  modelConfidenceThreshold: 0.3,
  debugMode: false
};

// Initialize the system
async function init() {
  updateStatus("Initializing system...");
  
  // Create debug canvas
  setupDebugCanvas();
  
  // Create scene first (before renderer tries to use it)
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(
    75, window.innerWidth / window.innerHeight, 0.1, 1000
  );
  camera.position.z = 5;
  
  // Create renderer with post-processing capabilities
  setupRenderer();

  // Initialize canvas for video processing
  canvasForProcessing = document.createElement('canvas');
  ctxForProcessing = canvasForProcessing.getContext('2d');
  
  // Set up video camera input
  await setupCamera();
  
  // Load TensorFlow.js models
  await loadTensorFlowModels();
  
  // Initialize fluid simulation with shaders
  if (config.useShaders) {
    initFluidSimulationWithShaders();
  } else {
    initFluidSimulation();
  }
  
  // Initialize body outline visualization
  initBodyOutline();
  
  // Create fluid field for more complex interactions
  initFluidField();
  
  // Handle window resize
  window.addEventListener('resize', onWindowResize);
  
  // Start animation loop
  animate();
  
  // Update UI to match actual config values
  updateUIControls();
  
  updateStatus("Initialization complete. Move in front of the camera.");
  
  // Hide loading indicator
  document.getElementById('loading').style.display = 'none';
}

// Set up debug canvas
function setupDebugCanvas() {
  debugCanvas = document.createElement('canvas');
  document.body.appendChild(debugCanvas);
  debugCanvas.style.position = 'absolute';
  debugCanvas.style.top = '10px';
  debugCanvas.style.right = '10px';
  debugCanvas.style.width = '160px';
  debugCanvas.style.height = '120px';
  debugCanvas.style.border = '1px solid white';
  debugCanvas.style.display = config.debugMode ? 'block' : 'none';
  debugCtx = debugCanvas.getContext('2d');
}

// Set up renderer with post-processing
function setupRenderer() {
  renderer = new THREE.WebGLRenderer({ 
    antialias: true,
    alpha: true
  });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  document.body.appendChild(renderer.domElement);
  
  // Add basic ambient lighting (scene is now defined before this is called)
  // const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  // scene.add(ambientLight);
}

// Set up webcam
async function setupCamera() {
  try {
    updateStatus("Setting up camera...");
    // Request camera access with higher resolution
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      } 
    });
    
    // Create video element
    videoElement = document.createElement('video');
    videoElement.srcObject = stream;
    videoElement.play();
    videoElement.style.display = 'none';
    document.body.appendChild(videoElement);
    
    // Wait for video metadata to load
    await new Promise(resolve => {
      videoElement.onloadedmetadata = () => {
        canvasForProcessing.width = videoElement.videoWidth;
        canvasForProcessing.height = videoElement.videoHeight;
        debugCanvas.width = videoElement.videoWidth;
        debugCanvas.height = videoElement.videoHeight;
        resolve();
      };
    });
    
    // Create video texture and material
    videoTexture = new THREE.VideoTexture(videoElement);
    videoTexture.minFilter = THREE.LinearFilter;
    videoTexture.magFilter = THREE.LinearFilter;
    
    videoMaterial = new THREE.MeshBasicMaterial({ 
      map: videoTexture,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.8
    });
    
    // Create a plane for the video background
    const aspectRatio = videoElement.videoHeight / videoElement.videoWidth;
    const planeGeometry = new THREE.PlaneGeometry(10, 10 * aspectRatio);
    const plane = new THREE.Mesh(planeGeometry, videoMaterial);
    plane.position.z = -0.5; // Put behind particles
    scene.add(plane);
    
    // Flip the video so it appears mirror-like
    plane.scale.x = -1;
    
    updateStatus("Camera setup complete");
  } catch (error) {
    updateStatus("Error accessing camera: " + error.message);
    console.error('Error accessing camera:', error);
  }
}

// Load TensorFlow.js models
async function loadTensorFlowModels() {
  updateStatus("Loading TensorFlow models...");
  
  try {
    // Dynamically import TensorFlow.js and models
    const tf = await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.11.0/dist/tf.min.js');
    const posenet = await import('https://cdn.jsdelivr.net/npm/@tensorflow-models/posenet@2.2.2/dist/posenet.min.js');
    
    // Load PoseNet model with mobilenet architecture (faster)
    poseNetModel = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: { width: 640, height: 480 },
      multiplier: 0.75,
      quantBytes: 2
    });
    
    updateStatus("TensorFlow models loaded successfully");
  } catch (error) {
    updateStatus("Error loading TensorFlow models: " + error.message);
    console.error('Error loading TensorFlow models:', error);
  }
}

// Initialize fluid simulation with shaders
function initFluidSimulationWithShaders() {
  updateStatus(`Creating advanced particle system with ${config.particleCount} particles`);
  
  // Define custom particle shader material
  const particleShaderMaterial = new THREE.ShaderMaterial({
    uniforms: {
      color: { value: new THREE.Color(config.particleColor) },
      pointTexture: { value: createParticleTexture() },
      time: { value: 0 },
      opacity: { value: config.particleOpacity }
    },
    vertexShader: `
      attribute float size;
      attribute vec3 customColor;
      attribute float life;
      
      varying vec3 vColor;
      varying float vLife;
      
      void main() {
        vColor = customColor;
        vLife = life;
        
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * (300.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      uniform vec3 color;
      uniform sampler2D pointTexture;
      uniform float opacity;
      
      varying vec3 vColor;
      varying float vLife;
      
      void main() {
        // Radial gradient texture
        vec4 texColor = texture2D(pointTexture, gl_PointCoord);
        
        // Apply color and fade based on life
        gl_FragColor = vec4(vColor, opacity * vLife) * texColor;
        
        // Add glow effect
        if (length(gl_PointCoord - vec2(0.5)) > 0.5) discard;
      }
    `,
    blending: THREE.AdditiveBlending,
    depthTest: false,
    transparent: true
  });
  
  // Create particle system using buffer geometry for better performance
  const particles = new THREE.BufferGeometry();
  
  // Positions array (xyz per particle)
  const positions = new Float32Array(config.particleCount * 3);
  
  // Velocities array (xy per particle - stored in userData)
  const velocities = new Float32Array(config.particleCount * 2);
  
  // Additional attributes for shader
  const sizes = new Float32Array(config.particleCount);
  const colors = new Float32Array(config.particleCount * 3);
  const lives = new Float32Array(config.particleCount);
  const lifeTimes = new Float32Array(config.particleCount);
  
  // Initialize particles with random positions and properties
  for (let i = 0; i < config.particleCount; i++) {
    const i3 = i * 3;
    const i2 = i * 2;
    
    // Position
    positions[i3] = (Math.random() - 0.5) * 8;
    positions[i3 + 1] = (Math.random() - 0.5) * 8 * (window.innerHeight / window.innerWidth);
    positions[i3 + 2] = 0.5; // Put particles slightly in front of video
    
    // Velocity
    velocities[i2] = (Math.random() - 0.5) * 0.02;
    velocities[i2 + 1] = (Math.random() - 0.5) * 0.02;
    
    // Size - vary slightly for more natural look
    sizes[i] = config.particleSize * (0.8 + Math.random() * 0.4);
    
    // Color - slight variations of the base color
    const hsl = new THREE.Color(config.particleColor).getHSL({});
    const color = new THREE.Color().setHSL(
      hsl.h + (Math.random() * 0.1 - 0.05),
      hsl.s * (0.7 + Math.random() * 0.6),
      hsl.l * (0.7 + Math.random() * 0.6)
    );
    
    colors[i3] = color.r;
    colors[i3 + 1] = color.g;
    colors[i3 + 2] = color.b;
    
    // Life values - each particle has a random lifetime
    lives[i] = 1.0; // Full life at start
    lifeTimes[i] = config.particleLifeSpan.min + 
      Math.random() * (config.particleLifeSpan.max - config.particleLifeSpan.min);
  }
  
  // Add attributes to the geometry
  particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  particles.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
  particles.setAttribute('customColor', new THREE.BufferAttribute(colors, 3));
  particles.setAttribute('life', new THREE.BufferAttribute(lives, 1));
  
  // Store additional data in userData
  particles.userData = {
    velocities,
    lifeTimes,
    creationTimes: new Float32Array(config.particleCount).fill(Date.now())
  };
  
  // Create the particle system
  if (particleSystem) {
    scene.remove(particleSystem);
  }
  
  particleSystem = new THREE.Points(particles, particleShaderMaterial);
  scene.add(particleSystem);
  
  // Store uniforms for animation updates
  particleUniforms = particleShaderMaterial.uniforms;
}

// Create circular particle texture
function createParticleTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  
  const context = canvas.getContext('2d');
  
  // Create radial gradient
  const gradient = context.createRadialGradient(
    64, 64, 0,
    64, 64, 64
  );
  
  // Add color stops for a glowing effect
  gradient.addColorStop(0, 'rgba(255, 255, 255, 1.0)');
  gradient.addColorStop(0.2, 'rgba(255, 255, 255, 0.8)');
  gradient.addColorStop(0.4, 'rgba(255, 255, 255, 0.4)');
  gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
  
  // Draw circle
  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);
  
  // Create texture
  const texture = new THREE.Texture(canvas);
  texture.needsUpdate = true;
  return texture;
}

// Initialize standard fluid simulation (non-shader version)
function initFluidSimulation() {
  updateStatus(`Creating particle system with ${config.particleCount} particles`);
  
  // Create particle system for fluid visualization
  const particles = new THREE.BufferGeometry();
  const positions = new Float32Array(config.particleCount * 3);
  const velocities = new Float32Array(config.particleCount * 2);
  const lifetimes = new Float32Array(config.particleCount);
  const startTimes = new Float32Array(config.particleCount);
  
  // Current time
  const now = Date.now();
  
  // Distribute particles across the scene
  for (let i = 0; i < config.particleCount; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 8;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 8 * (window.innerHeight / window.innerWidth);
    positions[i * 3 + 2] = 0.5; // Put particles slightly in front of video
    
    velocities[i * 2] = (Math.random() - 0.5) * 0.02; // Small initial velocity
    velocities[i * 2 + 1] = (Math.random() - 0.5) * 0.02;
    
    // Set random lifetime between min and max
    lifetimes[i] = config.particleLifeSpan.min + 
      Math.random() * (config.particleLifeSpan.max - config.particleLifeSpan.min);
    
    // Set creation time
    startTimes[i] = now;
  }
  
  particles.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  particles.userData = {
    velocities,
    lifetimes,
    startTimes,
    lastUpdate: now
  };
  
  // Load particle texture
  const textureLoader = new THREE.TextureLoader();
  let particleTexture;
  
  try {
    // Try to create a custom texture
    particleTexture = createParticleTexture();
  } catch (error) {
    console.error('Error creating particle texture:', error);
    // Fallback to a simple circle
    particleTexture = null;
  }
  
  // Create material with additive blending for glow effect
  const material = new THREE.PointsMaterial({ 
    color: config.particleColor,
    size: config.particleSize,
    transparent: true,
    opacity: config.particleOpacity,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true,
    map: particleTexture,
    depthWrite: false
  });
  
  // Remove any existing particle system
  if (particleSystem) {
    scene.remove(particleSystem);
  }
  
  particleSystem = new THREE.Points(particles, material);
  scene.add(particleSystem);
}

// Initialize fluid field for more complex flow behaviors
function initFluidField() {
  const fieldResolution = 50; // Number of cells in each direction
  const fieldSize = { width: 10, height: 10 * (window.innerHeight / window.innerWidth) };
  
  // Create a grid of vectors representing the fluid flow
  fluidField = {
    resolution: fieldResolution,
    size: fieldSize,
    cells: new Array(fieldResolution * fieldResolution),
    
    // Initialize field with random small vectors
    init() {
      for (let i = 0; i < this.cells.length; i++) {
        this.cells[i] = {
          x: (Math.random() * 2 - 1) * 0.01,
          y: (Math.random() * 2 - 1) * 0.01
        };
      }
    },
    
    // Get cell index from world position
    getCellIndex(x, y) {
      // Convert world coordinates to grid coordinates
      const gridX = Math.floor(((x + this.size.width / 2) / this.size.width) * this.resolution);
      const gridY = Math.floor(((y + this.size.height / 2) / this.size.height) * this.resolution);
      
      // Clamp to grid boundaries
      const clampedX = Math.max(0, Math.min(this.resolution - 1, gridX));
      const clampedY = Math.max(0, Math.min(this.resolution - 1, gridY));
      
      return clampedY * this.resolution + clampedX;
    },
    
    // Get flow vector at a specific world position
    getFlowAt(x, y) {
      const index = this.getCellIndex(x, y);
      return this.cells[index] || { x: 0, y: 0 };
    },
    
    // Apply force at a specific position
    applyForce(posX, posY, forceX, forceY, radius) {
      const radiusSquared = radius * radius;
      
      // Determine affected cells
      const minX = Math.max(0, Math.floor(((posX - radius + this.size.width / 2) / this.size.width) * this.resolution));
      const maxX = Math.min(this.resolution - 1, Math.ceil(((posX + radius + this.size.width / 2) / this.size.width) * this.resolution));
      const minY = Math.max(0, Math.floor(((posY - radius + this.size.height / 2) / this.size.height) * this.resolution));
      const maxY = Math.min(this.resolution - 1, Math.ceil(((posY + radius + this.size.height / 2) / this.size.height) * this.resolution));
      
      // Apply force to each affected cell
      for (let y = minY; y <= maxY; y++) {
        for (let x = minX; x <= maxX; x++) {
          // Convert grid coordinates to world coordinates (cell center)
          const worldX = (x / this.resolution) * this.size.width - this.size.width / 2 + (this.size.width / this.resolution / 2);
          const worldY = (y / this.resolution) * this.size.height - this.size.height / 2 + (this.size.height / this.resolution / 2);
          
          // Calculate distance to force application point
          const dx = worldX - posX;
          const dy = worldY - posY;
          const distSquared = dx * dx + dy * dy;
          
          // Apply force with falloff based on distance
          if (distSquared < radiusSquared) {
            const index = y * this.resolution + x;
            const falloff = 1 - Math.sqrt(distSquared) / radius;
            
            this.cells[index].x += forceX * falloff * 0.5;
            this.cells[index].y += forceY * falloff * 0.5;
          }
        }
      }
    },
    
    // Update the field (apply diffusion, dissipation, etc.)
    update() {
      // Simple dissipation - gradually reduce flow strength
      for (let i = 0; i < this.cells.length; i++) {
        this.cells[i].x *= 0.99;
        this.cells[i].y *= 0.99;
      }
      
      // Could add more complex fluid dynamics here (diffusion, advection, etc.)
    },
    
    // Debug visualization
    debugDraw(ctx, canvasWidth, canvasHeight) {
      if (!ctx) return;
      
      const cellWidth = canvasWidth / this.resolution;
      const cellHeight = canvasHeight / this.resolution;
      
      ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
      ctx.lineWidth = 1;
      
      for (let y = 0; y < this.resolution; y++) {
        for (let x = 0; x < this.resolution; x++) {
          const index = y * this.resolution + x;
          const cell = this.cells[index];
          
          const centerX = (x + 0.5) * cellWidth;
          const centerY = (y + 0.5) * cellHeight;
          
          // Scale the vector for visibility
          const endX = centerX + cell.x * 200;
          const endY = centerY + cell.y * 200;
          
          ctx.beginPath();
          ctx.moveTo(centerX, centerY);
          ctx.lineTo(endX, endY);
          ctx.stroke();
        }
      }
    }
  };
  
  // Initialize the field
  fluidField.init();
}

// Initialize body outline visualization
function initBodyOutline() {
  // Create empty geometry for the body outline
  const lineGeometry = new THREE.BufferGeometry();
  const lineMaterial = new THREE.LineBasicMaterial({
    color: config.bodyOutlineColor,
    linewidth: 2,
    transparent: true,
    opacity: 0.7
  });
  
  // Add initial points (will be updated later)
  const points = [];
  for (let i = 0; i < 100; i++) {
    points.push(0, 0, 0.7);
  }
  
  lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  
  bodyOutlineMesh = new THREE.LineLoop(lineGeometry, lineMaterial);
  scene.add(bodyOutlineMesh);
  
  console.log("Body outline initialized");
}

// Process video frame to detect body using TensorFlow
async function processVideoFrame() {
  if (!videoElement || videoElement.readyState !== videoElement.HAVE_ENOUGH_DATA || !poseNetModel) {
    return { pose: null, interactionPoints: [] };
  }
  
  // Only run pose estimation every few frames for performance
  const now = Date.now();
  let pose = currentPose;
  
  // Run pose detection every 100ms
  if (now - lastPoseTime > 100) {
    try {
      pose = await poseNetModel.estimateSinglePose(videoElement, {
        flipHorizontal: true
      });
      
      // Only update if confidence is high enough
      if (pose && pose.score >= config.modelConfidenceThreshold) {
        currentPose = pose;
        lastPoseTime = now;
      }
    } catch (error) {
      console.error('Error estimating pose:', error);
    }
  }
  
  // Draw debug view
  if (config.debugMode && debugCtx) {
    drawDebugView(pose);
  }
  
  // Extract interaction points from pose
  const interactionPoints = extractInteractionPoints(pose);
  
  return { pose, interactionPoints };
}

// Draw debug visualization
function drawDebugView(pose) {
  if (!debugCtx || !videoElement) return;
  
  // Draw current video frame
  debugCtx.drawImage(videoElement, 0, 0, debugCanvas.width, debugCanvas.height);
  
  // Draw pose keypoints if available
  if (pose && pose.keypoints) {
    pose.keypoints.forEach(keypoint => {
      if (keypoint.score > config.modelConfidenceThreshold) {
        const { x, y } = keypoint.position;
        
        // Draw keypoint
        debugCtx.beginPath();
        debugCtx.arc(x, y, 5, 0, 2 * Math.PI);
        debugCtx.fillStyle = 'red';
        debugCtx.fill();
        
        // Draw label
        debugCtx.fillStyle = 'white';
        debugCtx.font = '10px Arial';
        debugCtx.fillText(keypoint.part, x + 7, y + 7);
      }
    });
    
    // Draw skeleton lines between connected joints
    drawSkeleton(pose.keypoints, debugCtx);
  }
  
  // Draw fluid field for debug purposes
  if (fluidField) {
    fluidField.debugDraw(debugCtx, debugCanvas.width, debugCanvas.height);
  }
}

// Draw skeleton connections
function drawSkeleton(keypoints, ctx) {
  // Define connected body parts (pairs of indices)
  const adjacentPairs = [
    ['leftShoulder', 'rightShoulder'], 
    ['leftShoulder', 'leftElbow'], 
    ['leftElbow', 'leftWrist'],
    ['rightShoulder', 'rightElbow'], 
    ['rightElbow', 'rightWrist'],
    ['leftShoulder', 'leftHip'], 
    ['rightShoulder', 'rightHip'],
    ['leftHip', 'rightHip'],
    ['leftHip', 'leftKnee'], 
    ['leftKnee', 'leftAnkle'],
    ['rightHip', 'rightKnee'], 
    ['rightKnee', 'rightAnkle']
  ];
  
  // Create lookup map
  const keypointMap = {};
  keypoints.forEach(keypoint => {
    keypointMap[keypoint.part] = keypoint;
  });
  
  // Draw lines
  ctx.strokeStyle = 'lime';
  ctx.lineWidth = 2;
  
  adjacentPairs.forEach(pair => {
    const startPoint = keypointMap[pair[0]];
    const endPoint = keypointMap[pair[1]];
    
    if (startPoint && endPoint && 
        startPoint.score > config.modelConfidenceThreshold && 
        endPoint.score > config.modelConfidenceThreshold) {
      ctx.beginPath();
      ctx.moveTo(startPoint.position.x, startPoint.position.y);
      ctx.lineTo(endPoint.position.x, endPoint.position.y);
      ctx.stroke();
    }
  });
}

// Extract interaction points from pose data
function extractInteractionPoints(pose) {
  const points = [];
  
  if (!pose || !pose.keypoints || pose.score < config.modelConfidenceThreshold) {
    return points;
  }
  
  // Create normalized conversion functions from pixel space to world space
  const videoWidth = videoElement.videoWidth;
  const videoHeight = videoElement.videoHeight;
  
  // Convert a point from pixel to world coordinates
  const convertToWorldCoords = (x, y) => {
    // Normalize to 0-1
    const normX = x / videoWidth;
    const normY = y / videoHeight;
    
    // Convert to world coords (-5 to 5 for x)
    const worldX = (normX * 2 - 1) * 5;
    
    // Adjust for aspect ratio and flip y axis
    const worldY = (normY * 2 - 1) * -5 * (videoHeight / videoWidth);
    
    return { x: worldX, y: worldY };
  };
  
  // Track important keypoints
  const keyPointsToTrack = [
    'leftWrist', 'rightWrist', 'leftElbow', 'rightElbow',
    'leftShoulder', 'rightShoulder', 'leftHip', 'rightHip'
  ];
  
  // Add all valid keypoints above threshold to interaction points
  pose.keypoints.forEach(keypoint => {
    if (keypoint.score >= config.modelConfidenceThreshold && 
        keyPointsToTrack.includes(keypoint.part)) {
      const { x, y } = keypoint.position;
      const worldCoords = convertToWorldCoords(x, y);
      
      // Add world space coordinates and type of point
      points.push({
        ...worldCoords,
        z: 0.7,
        type: keypoint.part,
        score: keypoint.score,
        radius: keypoint.part.includes('Wrist') ? 0.8 : 0.5 // Hands have larger influence
      });
    }
  });
  
  // If needed, interpolate between current and previous points for smoother interaction
  if (lastInteractionPoints.length > 0 && points.length > 0) {
    // Interpolation logic could be added here
  }
  
  // Store for next frame
  lastInteractionPoints = [...points];
  
  return points;
}

// Update body outline visualization based on pose
function updateBodyOutline(pose, interactionPoints) {
  if (!bodyOutlineMesh) return;
  
  // If no valid pose, hide the outline
  if (!pose || pose.score < config.modelConfidenceThreshold || interactionPoints.length === 0) {
    bodyOutlineMesh.visible = false;
    return;
  }
  
  bodyOutlineMesh.visible = true;
  
  // Get position buffer
  const positions = bodyOutlineMesh.geometry.attributes.position.array;
  
  // Create a convex hull around the detected body parts
  const hull = generateConvexHull(interactionPoints);
  
  // Update positions of the line geometry
  for (let i = 0; i < hull.length; i++) {
    const point = hull[i];
    positions[i * 3] = point.x;
    positions[i * 3 + 1] = point.y;
    positions[i * 3 + 2] = point.z || 0.7;
  }
  
  // Fill remaining points with last point if needed
  if (hull.length < positions.length / 3) {
    const lastPoint = hull[hull.length - 1];
    for (let i = hull.length; i < positions.length / 3; i++) {
      positions[i * 3] = lastPoint.x;
      positions[i * 3 + 1] = lastPoint.y;
      positions[i * 3 + 2] = lastPoint.z || 0.7;
    }
  }
  
  bodyOutlineMesh.geometry.attributes.position.needsUpdate = true;
}

// Generate a convex hull from a set of points
function generateConvexHull(points) {
  if (points.length < 3) return points;
  
  // Find the point with the lowest y value
  let startPoint = points[0];
  points.forEach(p => {
    if (p.y < startPoint.y || (p.y === startPoint.y && p.x < startPoint.x)) {
      startPoint = p;
    }
  });
  
  // Sort the points by angle relative to the start point
  const sortedPoints = [...points].sort((a, b) => {
    if (a === startPoint) return -1;
    if (b === startPoint) return 1;
    
    const angleA = Math.atan2(a.y - startPoint.y, a.x - startPoint.x);
    const angleB = Math.atan2(b.y - startPoint.y, b.x - startPoint.x);
    
    return angleA - angleB;
  });
  
  // Graham scan algorithm
  const hull = [sortedPoints[0], sortedPoints[1]];
  
  for (let i = 2; i < sortedPoints.length; i++) {
    while (hull.length > 1 && !isLeftTurn(
      hull[hull.length - 2], 
      hull[hull.length - 1], 
      sortedPoints[i]
    )) {
      hull.pop();
    }
    hull.push(sortedPoints[i]);
  }
  
  return hull;
}

// Check if three points make a left turn (used for convex hull)
function isLeftTurn(p1, p2, p3) {
  return (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x) > 0;
}

// Add this code at the end of your fluid-simulation-enhanced.js file

// Expose fluid simulation functions to window object
window.fluidFunctions = {
  init,
  resetParticles,
  toggleDebugView,
  updateParticleSize,
  updateParticleOpacity,
  updateParticleCount,
  updateViscosity,
  updateParticleColor,
  toggleShaders,
  toggleColorVariation,
  toggleGlowEffect
};

// Add these missing functions that are referenced in the UI

// Reset particles to random positions
function resetParticles() {
  if (!particleSystem) return;
  
  const positions = particleSystem.geometry.attributes.position.array;
  const velocities = particleSystem.geometry.userData.velocities;
  
  for (let i = 0; i < config.particleCount; i++) {
    const i3 = i * 3;
    const i2 = i * 2;
    
    // Position
    positions[i3] = (Math.random() - 0.5) * 8;
    positions[i3 + 1] = (Math.random() - 0.5) * 8 * (window.innerHeight / window.innerWidth);
    positions[i3 + 2] = 0.5;
    
    // Velocity
    velocities[i2] = (Math.random() - 0.5) * 0.02;
    velocities[i2 + 1] = (Math.random() - 0.5) * 0.02;
    
    // Reset life if using shaders
    if (particleSystem.geometry.attributes.life) {
      particleSystem.geometry.attributes.life.array[i] = 1.0;
      particleSystem.geometry.attributes.life.needsUpdate = true;
    }
    
    // Reset creation time
    if (particleSystem.geometry.userData.creationTimes) {
      particleSystem.geometry.userData.creationTimes[i] = Date.now();
    }
    if (particleSystem.geometry.userData.startTimes) {
      particleSystem.geometry.userData.startTimes[i] = Date.now();
    }
  }
  
  particleSystem.geometry.attributes.position.needsUpdate = true;
  updateStatus("Particles reset");
}

// Toggle debug view
function toggleDebugView() {
  config.debugMode = !config.debugMode;
  if (debugCanvas) {
    debugCanvas.style.display = config.debugMode ? 'block' : 'none';
  }
  updateStatus(`Debug view ${config.debugMode ? 'enabled' : 'disabled'}`);
}

// Update particle size
function updateParticleSize(size) {
  config.particleSize = size;
  
  if (particleSystem) {
    if (config.useShaders) {
      // Update size attribute for shader-based particles
      const sizes = particleSystem.geometry.attributes.size.array;
      for (let i = 0; i < config.particleCount; i++) {
        sizes[i] = config.particleSize * (0.8 + Math.random() * 0.4);
      }
      particleSystem.geometry.attributes.size.needsUpdate = true;
    } else {
      // Update material size for standard particles
      particleSystem.material.size = config.particleSize;
    }
  }
}

// Update particle opacity
function updateParticleOpacity(opacity) {
  config.particleOpacity = opacity;
  
  if (particleSystem) {
    if (config.useShaders && particleUniforms) {
      particleUniforms.opacity.value = config.particleOpacity;
    } else {
      particleSystem.material.opacity = config.particleOpacity;
    }
  }
}

// Update particle count
function updateParticleCount(count) {
  // Store old count
  const oldCount = config.particleCount;
  config.particleCount = count;
  
  // If the simulation is already running, reinitialize with new count
  if (particleSystem) {
    if (config.useShaders) {
      initFluidSimulationWithShaders();
    } else {
      initFluidSimulation();
    }
    updateStatus(`Particle count updated to ${count}`);
  }
}

// Update viscosity
function updateViscosity(value) {
  config.viscosity = value;
  updateStatus(`Viscosity updated to ${value.toFixed(2)}`);
}

// Update particle color
function updateParticleColor(colorValue) {
  // Convert hex string to numeric color
  config.particleColor = new THREE.Color(colorValue).getHex();
  
  if (particleSystem) {
    if (config.useShaders && particleUniforms) {
      particleUniforms.color.value = new THREE.Color(config.particleColor);
    } else {
      particleSystem.material.color = new THREE.Color(config.particleColor);
    }
  }
}

// Toggle shaders
function toggleShaders(useShaders) {
  if (config.useShaders === useShaders) return;
  
  config.useShaders = useShaders;
  
  // Reinitialize with new setting
  if (useShaders) {
    initFluidSimulationWithShaders();
  } else {
    initFluidSimulation();
  }
  
  updateStatus(`${useShaders ? 'Advanced' : 'Standard'} rendering mode enabled`);
}

// Toggle color variation
function toggleColorVariation(enabled) {
  config.colorVariation = enabled;
  
  if (particleSystem && config.useShaders) {
    // Update colors in the shader-based system
    const colors = particleSystem.geometry.attributes.customColor.array;
    
    for (let i = 0; i < config.particleCount; i++) {
      const i3 = i * 3;
      
      if (enabled) {
        // Apply color variation
        const hsl = new THREE.Color(config.particleColor).getHSL({});
        const color = new THREE.Color().setHSL(
          hsl.h + (Math.random() * 0.1 - 0.05),
          hsl.s * (0.7 + Math.random() * 0.6),
          hsl.l * (0.7 + Math.random() * 0.6)
        );
        
        colors[i3] = color.r;
        colors[i3 + 1] = color.g;
        colors[i3 + 2] = color.b;
      } else {
        // Use uniform color
        const color = new THREE.Color(config.particleColor);
        colors[i3] = color.r;
        colors[i3 + 1] = color.g;
        colors[i3 + 2] = color.b;
      }
    }
    
    particleSystem.geometry.attributes.customColor.needsUpdate = true;
  }
  
  updateStatus(`Color variation ${enabled ? 'enabled' : 'disabled'}`);
}

// Toggle glow effect
function toggleGlowEffect(enabled) {
  config.glowEffect = enabled;
  
  if (particleSystem && !config.useShaders) {
    // For non-shader particles, we can change the blending mode
    particleSystem.material.blending = enabled ? 
      THREE.AdditiveBlending : THREE.NormalBlending;
  }
  
  updateStatus(`Glow effect ${enabled ? 'enabled' : 'disabled'}`);
}

// Add missing update UI controls function
function updateUIControls() {
  // Update sliders to match current config
  document.getElementById('particle-size').value = config.particleSize;
  document.getElementById('size-value').textContent = config.particleSize.toFixed(2);
  
  document.getElementById('particle-opacity').value = config.particleOpacity;
  document.getElementById('opacity-value').textContent = config.particleOpacity.toFixed(2);
  
  document.getElementById('particle-count').value = config.particleCount;
  document.getElementById('count-value').textContent = config.particleCount;
  
  document.getElementById('viscosity').value = config.viscosity;
  document.getElementById('viscosity-value').textContent = config.viscosity.toFixed(2);
  
  // Update color picker
  document.getElementById('particle-color').value = '#' + new THREE.Color(config.particleColor).getHexString();
  
  // Update checkboxes
  document.getElementById('use-shaders').checked = config.useShaders;
  document.getElementById('color-variation').checked = config.colorVariation;
  document.getElementById('glow-effect').checked = config.glowEffect;
}

// Add missing onWindowResize function
function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

// Add status update function
function updateStatus(message) {
  const statusEl = document.getElementById('status');
  if (statusEl) {
    statusEl.textContent = message;
  }
  console.log(message);
}

// Add missing animate function for the animation loop
function animate() {
  requestAnimationFrame(animate);
  
  // Process video frame to get body pose and interaction points
  processVideoFrame().then(({ pose, interactionPoints }) => {
    // Update fluid simulation based on interaction points
    updateFluidSimulation(interactionPoints);
    
    // Update body outline visualization
    updateBodyOutline(pose, interactionPoints);
    
    // Update fluid field
    if (fluidField) {
      fluidField.update();
    }
  });
  
  // Update the renderer
  renderer.render(scene, camera);
}

// Add missing updateFluidSimulation function
function updateFluidSimulation(interactionPoints) {
  if (!particleSystem) return;
  
  const now = Date.now();
  const positions = particleSystem.geometry.attributes.position.array;
  const velocities = particleSystem.geometry.userData.velocities;
  
  // Time step for simulation
  const deltaTime = (now - (particleSystem.geometry.userData.lastUpdate || now)) / 1000;
  particleSystem.geometry.userData.lastUpdate = now;
  
  // Limit delta time to prevent large jumps
  const dt = Math.min(deltaTime, 0.05);
  
  // Update time uniform for shaders
  if (config.useShaders && particleUniforms) {
    particleUniforms.time.value = now / 1000;
  }
  
  // Apply fluid dynamics to particles
  for (let i = 0; i < config.particleCount; i++) {
    const i3 = i * 3;
    const i2 = i * 2;
    
    // Apply viscosity (damping)
    velocities[i2] *= config.viscosity;
    velocities[i2 + 1] *= config.viscosity;
    
    // Current position
    let x = positions[i3];
    let y = positions[i3 + 1];
    
    // Apply forces from interaction points (body parts)
    for (const point of interactionPoints) {
      const dx = x - point.x;
      const dy = y - point.y;
      const distSquared = dx * dx + dy * dy;
      const radius = point.radius || config.repulsionDistance;
      
      // Apply repulsion force if within range
      if (distSquared < radius * radius && distSquared > 0.01) {
        const dist = Math.sqrt(distSquared);
        const force = config.repulsionForce * (1 - dist / radius);
        
        // Normalize direction and apply force
        velocities[i2] += (dx / dist) * force;
        velocities[i2 + 1] += (dy / dist) * force;
      }
    }
    
    // Apply flow field influence
    if (fluidField) {
      const flow = fluidField.getFlowAt(x, y);
      velocities[i2] += flow.x * config.flowFieldStrength;
      velocities[i2 + 1] += flow.y * config.flowFieldStrength;
    }
    
    // Apply speed limit
    const speedSq = velocities[i2] * velocities[i2] + velocities[i2 + 1] * velocities[i2 + 1];
    if (speedSq > config.particleSpeedLimit * config.particleSpeedLimit) {
      const speed = Math.sqrt(speedSq);
      velocities[i2] = (velocities[i2] / speed) * config.particleSpeedLimit;
      velocities[i2 + 1] = (velocities[i2 + 1] / speed) * config.particleSpeedLimit;
    }
    
    // Update position
    positions[i3] += velocities[i2] * dt * 60; // Scale by 60 to make it frame-rate independent
    positions[i3 + 1] += velocities[i2 + 1] * dt * 60;
    
    // Boundary handling - bounce off edges
    const bounds = {
      x: 5,
      y: 5 * (window.innerHeight / window.innerWidth)
    };
    
    if (positions[i3] > bounds.x) {
      positions[i3] = bounds.x;
      velocities[i2] *= -0.5; // Dampen on bounce
    } else if (positions[i3] < -bounds.x) {
      positions[i3] = -bounds.x;
      velocities[i2] *= -0.5;
    }
    
    if (positions[i3 + 1] > bounds.y) {
      positions[i3 + 1] = bounds.y;
      velocities[i2 + 1] *= -0.5;
    } else if (positions[i3 + 1] < -bounds.y) {
      positions[i3 + 1] = -bounds.y;
      velocities[i2 + 1] *= -0.5;
    }
    
    // Handle particle life cycle if using lifespan
    if (config.useShaders && particleSystem.geometry.attributes.life) {
      const life = particleSystem.geometry.attributes.life.array;
      const lifeTimes = particleSystem.geometry.userData.lifeTimes;
      const creationTimes = particleSystem.geometry.userData.creationTimes;
      
      // Calculate remaining life ratio
      const elapsed = now - creationTimes[i];
      const totalLife = lifeTimes[i];
      
      if (elapsed > totalLife) {
        // Respawn particle
        positions[i3] = (Math.random() - 0.5) * 8;
        positions[i3 + 1] = (Math.random() - 0.5) * 8 * (window.innerHeight / window.innerWidth);
        velocities[i2] = (Math.random() - 0.5) * 0.02;
        velocities[i2 + 1] = (Math.random() - 0.5) * 0.02;
        creationTimes[i] = now;
        life[i] = 1.0;
      } else {
        // Update life ratio
        life[i] = 1.0 - (elapsed / totalLife);
      }
    }
  }
  
  // Update attributes to reflect changes
  particleSystem.geometry.attributes.position.needsUpdate = true;
  
  if (config.useShaders) {
    if (particleSystem.geometry.attributes.life) {
      particleSystem.geometry.attributes.life.needsUpdate = true;
    }
  }
}