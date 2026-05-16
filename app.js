import * as THREE from "./vendor/three/three.module.js";
import { OBJLoader } from "./vendor/three/OBJLoader.js";
import loadMujoco from "./vendor/mujoco/mujoco.js";

const canvas = document.querySelector("#viewport");
const loading = document.querySelector("#loading");
const joystick = document.querySelector("#joystick");
const joystickKnob = document.querySelector("#joystick-knob");
const outputs = {
  forward: document.querySelector("#forward"),
  turn: document.querySelector("#turn"),
  runtime: document.querySelector("#runtime"),
  status: document.querySelector("#status"),
  step: document.querySelector("#step"),
  controlRate: document.querySelector("#control-rate"),
  physicsRate: document.querySelector("#physics-rate"),
};

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x14181b, 1);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x14181b, 18, 48);

const camera = new THREE.PerspectiveCamera(48, 1, 0.05, 140);
camera.up.set(0, 0, 1);

const root = new THREE.Group();
scene.add(root);

const hemi = new THREE.HemisphereLight(0xdce8ff, 0x2d332a, 1.35);
scene.add(hemi);

const sun = new THREE.DirectionalLight(0xffffff, 2.5);
sun.position.set(-6, -8, 12);
scene.add(sun);

function createCheckerTexture() {
  const tileCount = 8;
  const tilePixels = 64;
  const textureCanvas = document.createElement("canvas");
  textureCanvas.width = tileCount * tilePixels;
  textureCanvas.height = tileCount * tilePixels;
  const context = textureCanvas.getContext("2d");
  const colors = ["#0e4004", "#aaba8d"];

  for (let y = 0; y < tileCount; y += 1) {
    for (let x = 0; x < tileCount; x += 1) {
      context.fillStyle = colors[(x + y) % 2];
      context.fillRect(x * tilePixels, y * tilePixels, tilePixels, tilePixels);
    }
  }

  const texture = new THREE.CanvasTexture(textureCanvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(3, 3);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  return texture;
}

const grid = new THREE.GridHelper(48, 48, 0x9a8e67, 0x30465d);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(48, 48),
  new THREE.MeshStandardMaterial({
    map: createCheckerTexture(),
    roughness: 0.82,
    metalness: 0.0,
  }),
);
ground.receiveShadow = true;
scene.add(ground);

const boneMaterial = new THREE.MeshStandardMaterial({
  color: 0xd6d0bd,
  roughness: 0.72,
  metalness: 0.02,
});
const darkBoneMaterial = new THREE.MeshStandardMaterial({
  color: 0x9f9b8d,
  roughness: 0.78,
  metalness: 0.02,
});

const loader = new OBJLoader();
const meshObjects = new Map();
const bodyObjects = new Map();
const bodyDefs = [];
const keys = new Set();
const command = { forward: 0, turn: 0 };
const joystickState = {
  active: false,
  pointerId: null,
  forward: 0,
  turn: 0,
};
const lastRoot = {
  position: new THREE.Vector3(0, 0, 1.6),
  quaternion: new THREE.Quaternion(),
};
let limits = { maxForward: 0.8, maxReverse: 0.25, maxTurn: 0.25 };
let runtime = null;
let followCamera = true;
let pointerDrag = null;
const orbit = {
  target: new THREE.Vector3(0, 0, 1.4),
  azimuth: -2.35,
  elevation: 0.38,
  distance: 10.5,
};

function setStatus(message) {
  outputs.status.value = message;
  loading.textContent = message;
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} ${response.status}`);
  }
  return response.json();
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} ${response.status}`);
  }
  return response.text();
}

async function fetchArrayBuffer(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} ${response.status}`);
  }
  return response.arrayBuffer();
}

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) {
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(1, height);
    camera.updateProjectionMatrix();
  }
}

function createBodyGroups(bodies) {
  bodyDefs.length = 0;
  bodyObjects.clear();
  for (const body of bodies) {
    const group = new THREE.Group();
    group.name = body.name;
    group.userData.body = body;
    bodyObjects.set(body.name, group);
  }

  for (const body of bodies) {
    const group = bodyObjects.get(body.name);
    const parent = body.parent ? bodyObjects.get(body.parent) : root;
    parent.add(group);
    bodyDefs.push(body);
  }
  globalThis.__trexBodyCount = bodyDefs.length;
}

function loadObj(mesh) {
  return new Promise((resolve, reject) => {
    loader.load(
      mesh.mesh,
      (object) => {
        object.name = mesh.name;
        object.traverse((child) => {
          if (child.isMesh) {
            child.material = mesh.name.includes("toe")
              ? darkBoneMaterial
              : boneMaterial;
            child.frustumCulled = false;
          }
        });
        object.position.fromArray(mesh.origin_position);
        object.quaternion.fromArray(mesh.origin_quaternion);
        const parent = bodyObjects.get(mesh.link) || root;
        parent.add(object);
        meshObjects.set(mesh.name, object);
        globalThis.__trexMeshCount = meshObjects.size;
        resolve();
      },
      undefined,
      reject,
    );
  });
}

function applyQpos(qpos) {
  for (const body of bodyDefs) {
    const group = bodyObjects.get(body.name);
    if (!group) {
      continue;
    }

    group.position.fromArray(body.position);
    group.quaternion.fromArray(body.quaternion);

    for (const joint of body.joints) {
      if (joint.type === "free") {
        group.position.set(qpos[0], qpos[1], qpos[2]);
        group.quaternion.set(qpos[4], qpos[5], qpos[6], qpos[3]);
      } else if (joint.type === "hinge") {
        const rotation = new THREE.Quaternion().setFromAxisAngle(
          new THREE.Vector3().fromArray(joint.axis).normalize(),
          qpos[joint.qposAdr],
        );
        group.quaternion.multiply(rotation);
      }
    }
  }
}

function silu(x) {
  return x / (1 + Math.exp(-x));
}

function makeArrayMap(policyMeta, policyBuffer) {
  const floats = new Float32Array(policyBuffer);
  const arrays = new Map();
  for (const item of policyMeta.arrays) {
    arrays.set(item.name, floats.subarray(item.offset, item.offset + item.length));
  }
  return arrays;
}

function dense(input, kernel, bias, inSize, outSize, activate) {
  const output = new Float32Array(outSize);
  for (let j = 0; j < outSize; j += 1) {
    let sum = bias[j];
    for (let i = 0; i < inSize; i += 1) {
      sum += input[i] * kernel[i * outSize + j];
    }
    output[j] = activate ? silu(sum) : sum;
  }
  return output;
}

function makePolicy(policyMeta, policyBuffer) {
  const arrays = makeArrayMap(policyMeta, policyBuffer);
  const mean = arrays.get("normalizer.state.mean");
  const std = arrays.get("normalizer.state.std");
  const layers = [
    { name: "hidden_0", inSize: 88, outSize: 512, activate: true },
    { name: "hidden_1", inSize: 512, outSize: 256, activate: true },
    { name: "hidden_2", inSize: 256, outSize: 128, activate: true },
    { name: "hidden_3", inSize: 128, outSize: 20, activate: false },
  ];
  return {
    act(observation) {
      let values = new Float32Array(observation.length);
      for (let i = 0; i < observation.length; i += 1) {
        values[i] = (observation[i] - mean[i]) / std[i];
      }
      for (const layer of layers) {
        values = dense(
          values,
          arrays.get(`${layer.name}.kernel`),
          arrays.get(`${layer.name}.bias`),
          layer.inSize,
          layer.outSize,
          layer.activate,
        );
      }
      const action = new Float32Array(policyMeta.actionSize);
      for (let i = 0; i < action.length; i += 1) {
        action[i] = Math.tanh(values[i]);
      }
      return action;
    },
  };
}

function slice3(values, start) {
  return [values[start], values[start + 1], values[start + 2]];
}

function matTransposeVec(siteXmat, siteId, vector) {
  const offset = siteId * 9;
  const m00 = siteXmat[offset];
  const m01 = siteXmat[offset + 1];
  const m02 = siteXmat[offset + 2];
  const m10 = siteXmat[offset + 3];
  const m11 = siteXmat[offset + 4];
  const m12 = siteXmat[offset + 5];
  const m20 = siteXmat[offset + 6];
  const m21 = siteXmat[offset + 7];
  const m22 = siteXmat[offset + 8];
  return [
    m00 * vector[0] + m10 * vector[1] + m20 * vector[2],
    m01 * vector[0] + m11 * vector[1] + m21 * vector[2],
    m02 * vector[0] + m12 * vector[1] + m22 * vector[2],
  ];
}

function norm2(a, b) {
  return Math.hypot(a, b);
}

function clip(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function makeTrexRuntime(mujoco, scenePayload, policy, xml) {
  const model = mujoco.MjModel.from_xml_string(xml);
  const data = new mujoco.MjData(model);
  const qpos = data.qpos;
  const qvel = data.qvel;
  const ctrl = data.ctrl;
  const sensordata = data.sensordata;
  const siteXmat = data.site_xmat;
  const siteXpos = data.site_xpos;
  const actuatorForce = data.actuator_force;
  const control = scenePayload.control;
  const sensors = scenePayload.sensors;
  const actionSize = scenePayload.model.actionSize;
  const lastAct = new Float32Array(actionSize);
  const observation = new Float32Array(88);
  let gaitPhase = 0;
  let step = 0;
  let lastActionMeanAbs = 0;

  function reset() {
    qpos.set(scenePayload.model.initialQpos);
    qvel.fill(0);
    ctrl.fill(0);
    lastAct.fill(0);
    gaitPhase = 0;
    step = 0;
    command.forward = 0;
    command.turn = 0;
    mujoco.mj_forward(model, data);
  }

  function gravity() {
    return matTransposeVec(siteXmat, sensors.imuSiteId, [0, 0, -1]);
  }

  function localSensor(sensorDef) {
    return matTransposeVec(
      siteXmat,
      sensors.imuSiteId,
      slice3(sensordata, sensorDef.start),
    );
  }

  function fillObservation() {
    const gyro = slice3(sensordata, sensors.gyro.start);
    const grav = gravity();
    const localLinvel = localSensor(sensors.globalLinvel);
    const localAngvel = localSensor(sensors.globalAngvel);
    let index = 0;
    for (const value of gyro) observation[index++] = value;
    for (const value of grav) observation[index++] = value;
    for (let i = 7; i < qpos.length; i += 1) observation[index++] = qpos[i];
    for (let i = 6; i < qvel.length; i += 1) observation[index++] = qvel[i];
    for (const value of lastAct) observation[index++] = value;
    for (const value of localLinvel) observation[index++] = value;
    for (const value of localAngvel) observation[index++] = value;
    observation[index++] = command.forward;
    observation[index++] = command.turn;
    observation[index++] = Math.sin(gaitPhase);
    observation[index++] = Math.cos(gaitPhase);
    return observation;
  }

  function standingCommandGate() {
    return norm2(command.forward, command.turn) < 0.05 ? 1 : 0;
  }

  function runningSpeedGate() {
    const speed = norm2(command.forward, command.turn);
    const width = Math.max(control.runningGateFull - control.runningGateStart, 1e-6);
    return clip((speed - control.runningGateStart) / width, 0, 1);
  }

  function commandResidualScale() {
    if (control.isWalkTask) {
      return control.walkActionResidualScale;
    }
    const speedGate = runningSpeedGate();
    return control.actionResidualScale.map(
      (value, index) =>
        (1 - speedGate) * value +
        speedGate * control.runningActionResidualScale[index],
    );
  }

  function postureGate() {
    const grav = gravity();
    let diffSq = 0;
    for (let i = 0; i < 3; i += 1) {
      const diff = control.uprightGravity[i] - grav[i];
      diffSq += diff * diff;
    }
    const orientation = Math.exp(-2 * diffSq);
    const torsoHeight = siteXpos[sensors.imuSiteId * 3 + 2];
    const heightStart =
      control.targetTorsoHeight * control.locomotionHeightGateFraction;
    const heightWidth = Math.max(control.targetTorsoHeight - heightStart, 1e-6);
    const heightGate = clip((torsoHeight - heightStart) / heightWidth, 0, 1);
    const orientationStart = control.locomotionOrientationGateThreshold;
    const orientationWidth = Math.max(1 - orientationStart, 1e-6);
    const orientationGate = clip(
      (orientation - orientationStart) / orientationWidth,
      0,
      1,
    );
    return heightGate * orientationGate;
  }

  function phaseActionCenterAction() {
    const shape = control.phaseActionCenterShape;
    if (!shape.length || shape[0] <= 0) {
      return null;
    }
    const phaseCount = shape[0];
    const scaledPhase = ((gaitPhase % (2 * Math.PI)) * phaseCount) / (2 * Math.PI);
    const lower = Math.floor(scaledPhase);
    const upper = (lower + 1) % phaseCount;
    const alpha = scaledPhase - lower;
    const out = new Float32Array(actionSize);
    for (let i = 0; i < actionSize; i += 1) {
      const lowValue = control.phaseActionCenter[lower * actionSize + i];
      const highValue = control.phaseActionCenter[upper * actionSize + i];
      out[i] = (1 - alpha) * lowValue + alpha * highValue;
    }
    return out;
  }

  function actionCenter() {
    const phaseCenter = phaseActionCenterAction();
    if (phaseCenter) {
      const movingGate = 1 - standingCommandGate();
      const speedGate =
        control.isMarchTask || control.isWalkTask ? 1 : runningSpeedGate();
      const phaseGate = movingGate * speedGate;
      return control.standPoseAction.map((value, index) =>
        clip((1 - phaseGate) * value + phaseGate * phaseCenter[index], -1, 1),
      );
    }
    const gaitPrior = new Float32Array(actionSize);
    if (control.applyGaitPriorAction) {
      const right = Math.sin(gaitPhase);
      const left = -right;
      gaitPrior[2] = right;
      gaitPrior[3] = left;
      gaitPrior[4] = -right;
      gaitPrior[5] = -left;
      gaitPrior[6] = 0.75 * right;
      gaitPrior[7] = 0.75 * left;
      const movingGate = 1 - standingCommandGate();
      const speedGate =
        control.isMarchTask || control.isWalkTask ? 1 : runningSpeedGate();
      const gaitScale = movingGate * speedGate * control.gaitPriorScale;
      for (let i = 0; i < actionSize; i += 1) {
        gaitPrior[i] *= gaitScale;
      }
    }
    return control.standPoseAction.map((value, index) =>
      clip(value + gaitPrior[index], -1, 1),
    );
  }

  function residualScale() {
    const commandScale = commandResidualScale();
    const recoveryGate = 1 - postureGate();
    return commandScale.map(
      (value, index) =>
        (1 - recoveryGate) * value +
        recoveryGate * Math.max(value, control.recoveryActionResidualScale[index]),
    );
  }

  function updateGaitPhase() {
    if (standingCommandGate()) {
      gaitPhase = 0;
      return;
    }
    const frequency = clip(
      control.gaitFrequencyMin +
        control.gaitFrequencyPerMps * Math.max(command.forward, 0),
      control.gaitFrequencyMin,
      control.gaitFrequencyMax,
    );
    gaitPhase =
      (gaitPhase + 2 * Math.PI * frequency * scenePayload.dt) % (2 * Math.PI);
  }

  function stepControl() {
    const rawAction = policy.act(fillObservation());
    const center = actionCenter();
    const scale = residualScale();
    let meanAbs = 0;
    ctrl.fill(0);
    for (let i = 0; i < actionSize; i += 1) {
      const clipped = clip(rawAction[i], -1, 1);
      const applied = clip(center[i] + clipped * scale[i], -1, 1);
      lastAct[i] = applied;
      meanAbs += Math.abs(rawAction[i]);
      const targetScale =
        applied >= 0
          ? control.actionCtrlPositiveScale[i]
          : control.actionCtrlNegativeScale[i];
      ctrl[control.actionActuatorIds[i]] =
        control.actionCtrlNeutral[i] + applied * targetScale * control.actionScale;
    }
    for (let i = 0; i < scenePayload.nSubsteps; i += 1) {
      mujoco.mj_step(model, data);
    }
    updateGaitPhase();
    lastActionMeanAbs = meanAbs / actionSize;
    step += 1;
  }

  reset();
  return {
    dt: scenePayload.dt,
    qpos,
    get step() {
      return step;
    },
    get lastActionMeanAbs() {
      return lastActionMeanAbs;
    },
    reset,
    stepControl,
  };
}

function updateCommandFromKeys() {
  let forward = 0;
  let turn = 0;
  if (keys.has("KeyW") || keys.has("ArrowUp")) {
    forward += limits.maxForward;
  }
  if (keys.has("KeyS") || keys.has("ArrowDown")) {
    forward -= limits.maxReverse;
  }
  if (keys.has("KeyA") || keys.has("ArrowLeft")) {
    turn += limits.maxTurn;
  }
  if (keys.has("KeyD") || keys.has("ArrowRight")) {
    turn -= limits.maxTurn;
  }
  command.forward = clip(
    forward + joystickState.forward,
    -limits.maxReverse,
    limits.maxForward,
  );
  command.turn = clip(turn + joystickState.turn, -limits.maxTurn, limits.maxTurn);
}

function setJoystickCommand(event) {
  const rect = joystick.getBoundingClientRect();
  const radius = rect.width / 2;
  const centerX = rect.left + radius;
  const centerY = rect.top + radius;
  const dx = event.clientX - centerX;
  const dy = event.clientY - centerY;
  const distance = Math.hypot(dx, dy);
  const normalizedDistance = Math.min(1, distance / radius);
  const angle = Math.atan2(dy, dx);
  const x = Math.cos(angle) * normalizedDistance;
  const y = Math.sin(angle) * normalizedDistance;
  joystickState.forward =
    y < 0 ? -y * limits.maxForward : -y * limits.maxReverse;
  joystickState.turn = -x * limits.maxTurn;
  joystickKnob.style.transform = `translate(calc(-50% + ${x * 38}px), calc(-50% + ${
    y * 38
  }px))`;
  updateCommandFromKeys();
}

function resetJoystick() {
  joystickState.active = false;
  joystickState.pointerId = null;
  joystickState.forward = 0;
  joystickState.turn = 0;
  joystickKnob.style.transform = "translate(-50%, -50%)";
  updateCommandFromKeys();
}

function resetRuntime() {
  keys.clear();
  resetJoystick();
  runtime?.reset();
}

function updateTelemetry() {
  outputs.forward.value = command.forward.toFixed(2);
  outputs.turn.value = command.turn.toFixed(2);
  outputs.step.value = runtime ? String(runtime.step) : "0";
  if (runtime) {
    lastRoot.position.fromArray(runtime.qpos.slice(0, 3));
    lastRoot.quaternion.set(
      runtime.qpos[4],
      runtime.qpos[5],
      runtime.qpos[6],
      runtime.qpos[3],
    );
  }
}

function updateCamera() {
  if (followCamera) {
    orbit.target.lerp(
      new THREE.Vector3(
        lastRoot.position.x,
        lastRoot.position.y,
        lastRoot.position.z + 1.25,
      ),
      0.08,
    );
  }

  const cosElevation = Math.cos(orbit.elevation);
  const offset = new THREE.Vector3(
    Math.cos(orbit.azimuth) * cosElevation * orbit.distance,
    Math.sin(orbit.azimuth) * cosElevation * orbit.distance,
    Math.sin(orbit.elevation) * orbit.distance,
  );
  camera.position.copy(orbit.target).add(offset);
  camera.lookAt(orbit.target);
}

let nextControlAt = 0;
function animate(now = 0) {
  if (runtime) {
    if (nextControlAt === 0) {
      nextControlAt = now;
    }
    const controlMs = 1000 * runtime.dt;
    let guard = 0;
    while (now >= nextControlAt && guard < 4) {
      runtime.stepControl();
      nextControlAt += controlMs;
      guard += 1;
    }
    applyQpos(runtime.qpos);
    updateTelemetry();
  }
  resize();
  updateCamera();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

window.addEventListener("keydown", (event) => {
  if (event.repeat) {
    return;
  }
  if (event.code === "Space") {
    keys.clear();
    command.forward = 0;
    command.turn = 0;
    return;
  }
  if (event.code === "KeyR") {
    resetRuntime();
    return;
  }
  keys.add(event.code);
  updateCommandFromKeys();
});

window.addEventListener("keyup", (event) => {
  keys.delete(event.code);
  updateCommandFromKeys();
});

joystick.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  try {
    joystick.setPointerCapture(event.pointerId);
  } catch {
    // Synthetic pointer events used by smoke tests do not always have capture.
  }
  joystickState.active = true;
  joystickState.pointerId = event.pointerId;
  setJoystickCommand(event);
});

joystick.addEventListener("pointermove", (event) => {
  if (!joystickState.active || joystickState.pointerId !== event.pointerId) {
    return;
  }
  event.preventDefault();
  setJoystickCommand(event);
});

joystick.addEventListener("pointerup", (event) => {
  if (joystickState.pointerId === event.pointerId) {
    resetJoystick();
  }
});

joystick.addEventListener("pointercancel", (event) => {
  if (joystickState.pointerId === event.pointerId) {
    resetJoystick();
  }
});

joystick.addEventListener("lostpointercapture", resetJoystick);

canvas.addEventListener("pointerdown", (event) => {
  canvas.setPointerCapture(event.pointerId);
  followCamera = false;
  document.querySelector("#camera").textContent = "Orbit";
  pointerDrag = {
    id: event.pointerId,
    x: event.clientX,
    y: event.clientY,
  };
});

canvas.addEventListener("pointermove", (event) => {
  if (!pointerDrag || pointerDrag.id !== event.pointerId) {
    return;
  }
  const dx = event.clientX - pointerDrag.x;
  const dy = event.clientY - pointerDrag.y;
  pointerDrag.x = event.clientX;
  pointerDrag.y = event.clientY;
  orbit.azimuth -= dx * 0.006;
  orbit.elevation = THREE.MathUtils.clamp(
    orbit.elevation + dy * 0.004,
    -0.15,
    1.15,
  );
});

canvas.addEventListener("pointerup", (event) => {
  if (pointerDrag?.id === event.pointerId) {
    pointerDrag = null;
  }
});

canvas.addEventListener(
  "wheel",
  (event) => {
    event.preventDefault();
    followCamera = false;
    document.querySelector("#camera").textContent = "Orbit";
    orbit.distance = THREE.MathUtils.clamp(
      orbit.distance * (1 + Math.sign(event.deltaY) * 0.08),
      3.5,
      28,
    );
  },
  { passive: false },
);

document.querySelector("#reset").addEventListener("click", resetRuntime);
document.querySelector("#camera").addEventListener("click", (event) => {
  followCamera = !followCamera;
  event.currentTarget.textContent = followCamera ? "Follow" : "Orbit";
});

async function main() {
  setStatus("Loading scene metadata");
  const [scenePayload, policyMeta, policyBuffer, xml] = await Promise.all([
    fetchJson("./generated/scene.json"),
    fetchJson("./generated/policy.json"),
    fetchArrayBuffer("./generated/policy.bin"),
    fetchText("./generated/trex.xml"),
  ]);
  limits = scenePayload.limits;
  outputs.runtime.value = scenePayload.runtime;
  outputs.controlRate.value = `${scenePayload.controlHz.toFixed(0)} Hz`;
  outputs.physicsRate.value = `${scenePayload.physicsHz.toFixed(0)} Hz`;

  setStatus("Loading visual meshes");
  createBodyGroups(scenePayload.bodies);
  await Promise.all(scenePayload.meshes.map(loadObj));

  setStatus("Loading MuJoCo WASM");
  const mujoco = await loadMujoco({
    locateFile: (path) => `./vendor/mujoco/${path}`,
  });

  setStatus("Starting simulation");
  const policy = makePolicy(policyMeta, policyBuffer);
  runtime = makeTrexRuntime(mujoco, scenePayload, policy, xml);
  applyQpos(runtime.qpos);
  loading.classList.add("hidden");
  outputs.status.value = "running";
  globalThis.__trexRuntime = "wasm";
  globalThis.__trexStep = () => runtime?.step ?? 0;
  animate();
}

main().catch((error) => {
  loading.textContent = error.message;
  outputs.status.value = "error";
  console.error(error);
});
