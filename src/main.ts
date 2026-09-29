import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import type { MainModule, MjData, MjModel } from "@mujoco/mujoco";
import "./style.css";

// One entry per position actuator, in the model's order.
const JOINTS = [
  ["shoulder_pan", "Shoulder pan"],
  ["shoulder_lift", "Shoulder lift"],
  ["elbow_flex", "Elbow flex"],
  ["wrist_flex", "Wrist flex"],
  ["wrist_roll", "Wrist roll"],
  ["gripper", "Gripper"],
] as const;

// Poses in radians, one value per joint (from examples/so101.py).
const REST = [0, -1.57, 1.57, 0.8, 0, 0];
const POSES = [
  { name: "Rest", short: "Rest", ctrl: REST },
  { name: "Reach forward", short: "Forward", ctrl: [0, 0.6, -0.6, 0.4, 0, 1.2] },
  { name: "Reach right", short: "Right", ctrl: [1.2, 0.3, -0.2, 0.6, 1.0, 1.2] },
  { name: "Reach up", short: "Up", ctrl: [0, 0.4, -1.0, -0.4, 1.5, 0.6] },
];
const MOVE_SECONDS = 0.9;
const BG = 0xf1f1ee;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const view = $("view");

// Load the WASM engine and the packed model in parallel while the empty stage renders.
const loading = Promise.all([
  import("@mujoco/mujoco").then((m) => m.default()),
  fetchPack("/so101.pack.gz", (f) => ($("loading-bar").style.width = `${Math.round(f * 100)}%`)),
]);

// Stage: renderer, camera, lights and floor. MuJoCo is z-up, so the three.js scene is too.
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
view.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);
scene.fog = new THREE.Fog(BG, 2.5, 7);
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.25;
scene.environmentRotation.set(Math.PI / 2, 0, 0);

const sky = new THREE.HemisphereLight(0xffffff, 0xb8b4a8, 0.6);
sky.position.set(0, 0, 1);
scene.add(sky);

const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(0.8, -0.45, 1.7);
sun.target.position.set(0.12, -0.05, 0);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -0.8;
sun.shadow.camera.right = sun.shadow.camera.top = 0.8;
sun.shadow.camera.near = 0.5;
sun.shadow.camera.far = 3.5;
sun.shadow.bias = -0.0002;
sun.shadow.normalBias = 0.004;
sun.shadow.radius = 4;
sun.shadow.intensity = 0.6;
scene.add(sun, sun.target);

const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ color: BG, roughness: 1 }));
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(2, 20, 0xd6d6cf, 0xe0e0da);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.0005;
scene.add(grid);

const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 50);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.12, -0.05, 0.12);
controls.enableDamping = true;
controls.minDistance = 0.2;
controls.maxDistance = 3;
controls.maxPolarAngle = Math.PI / 2 - 0.03;

// Start from a 3/4 front-left view far enough back to fit every pose and the box.
// On wide screens the panel floats on the right, so the view centre shifts left.
let framed = false;
function resize() {
  const { clientWidth: w, clientHeight: h } = view;
  const panel = $("panel");
  const shift = panel.offsetTop < h / 2 ? (panel.offsetWidth + 20) / 2 : 0;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.setViewOffset(w, h, shift, 0, w, h);
  if (!framed) {
    const half = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const distance = Math.max(0.28 / half, (0.36 * h) / (half * (w - 2 * shift)));
    camera.position.copy(controls.target).addScaledVector(new THREE.Vector3(0.75, 0.45, 0.5).normalize(), distance);
    controls.update();
  }
}
new ResizeObserver(resize).observe(view);
resize();
controls.addEventListener("start", () => (framed = true));

// Simulation state, set once MuJoCo is ready.
let sim: Sim | null = null;
let refreshControls = () => {};
let tween: { from: number[]; to: number[]; start: number } | null = null;
let last = performance.now();
let budget = 0;

renderer.setAnimationLoop((now) => {
  const dt = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (sim) {
    if (tween) {
      const k = Math.min((now - tween.start) / 1000 / MOVE_SECONDS, 1);
      const e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
      sim.setCtrl(tween.from.map((a, i) => a + (tween!.to[i] - a) * e));
      if (k === 1) tween = null;
    }
    budget += dt;
    while (budget >= sim.timestep) {
      sim.mujoco.mj_step(sim.model, sim.data);
      budget -= sim.timestep;
    }
    sim.sync();
    refreshControls();
  }
  controls.update();
  renderer.render(scene, camera);
});

loading
  .then(([mujoco, files]) => {
    sim = createSim(mujoco, files);
    scene.add(sim.root);
    $("loading").classList.add("hidden");
    refreshControls = buildControls(sim);
  })
  .catch((error) => {
    console.error(error);
    $("loading-text").textContent = `Could not start the simulation: ${error instanceof Error ? error.message : error}`;
  });

type Sim = ReturnType<typeof createSim>;

function createSim(mujoco: MainModule, files: [string, Uint8Array][]) {
  mujoco.FS.mkdir("/so101");
  mujoco.FS.mkdir("/so101/assets");
  for (const [name, bytes] of files) mujoco.FS.writeFile(`/so101/${name}`, bytes);
  const model: MjModel = mujoco.MjModel.from_xml_path("/so101/scene_web.xml");
  for (const [name] of files) mujoco.FS.unlink(`/so101/${name}`); // compiled, no longer needed
  const data: MjData = new mujoco.MjData(model);

  const actuators = JOINTS.map(([name]) => {
    const id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, name);
    if (id < 0) throw new Error(`Actuator ${name} not found`);
    return { id, qpos: model.jnt_qposadr[model.actuator_trnid[2 * id]] as number };
  });
  const range = actuators.map(({ id }) => [model.actuator_ctrlrange[2 * id], model.actuator_ctrlrange[2 * id + 1]]);
  const { root, items } = buildMeshes(mujoco, model);

  const sim = {
    mujoco,
    model,
    data,
    timestep: model.opt.timestep,
    root,
    range,
    target: [...REST],
    setCtrl(values: number[]) {
      const ctrl = data.ctrl;
      values.forEach((v, i) => {
        sim.target[i] = THREE.MathUtils.clamp(v, range[i][0], range[i][1]);
        ctrl[actuators[i].id] = sim.target[i];
      });
    },
    joints: () => actuators.map(({ qpos }) => data.qpos[qpos] as number),
    reset() {
      mujoco.mj_resetData(model, data);
      const qpos = data.qpos;
      actuators.forEach(({ qpos: adr }, i) => (qpos[adr] = REST[i]));
      sim.setCtrl(REST);
      mujoco.mj_forward(model, data);
    },
    // Copy every drawn geom's world pose from MuJoCo into its three.js object.
    sync() {
      const pos = data.geom_xpos;
      const mat = data.geom_xmat;
      for (const { geom, object } of items) {
        const p = 3 * geom;
        const r = 9 * geom;
        object.matrix.set(
          mat[r], mat[r + 1], mat[r + 2], pos[p],
          mat[r + 3], mat[r + 4], mat[r + 5], pos[p + 1],
          mat[r + 6], mat[r + 7], mat[r + 8], pos[p + 2],
          0, 0, 0, 1,
        );
        object.matrixWorldNeedsUpdate = true;
      }
    },
  };
  sim.reset();
  sim.sync();
  return sim;
}

// One three.js mesh per visible MuJoCo geom (groups 0 to 2, like MuJoCo's own viewer).
// The floor plane is skipped: the stage has its own.
function buildMeshes(mujoco: MainModule, model: MjModel) {
  const T = mujoco.mjtGeom;
  const root = new THREE.Group();
  const items: { geom: number; object: THREE.Object3D }[] = [];
  const meshCache = new Map<number, THREE.BufferGeometry>();
  const materials = new Map<string, THREE.Material>();

  for (let g = 0; g < model.ngeom; g++) {
    if (model.geom_group[g] > 2) continue;
    const type = model.geom_type[g];
    const s = model.geom_size.subarray(3 * g, 3 * g + 3);
    let geometry: THREE.BufferGeometry;
    if (type === T.mjGEOM_MESH.value) {
      const id = model.geom_dataid[g];
      geometry = meshCache.get(id) ?? meshGeometry(model, id);
      meshCache.set(id, geometry);
    } else if (type === T.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]);
    } else if (type === T.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 32).rotateX(Math.PI / 2);
    } else if (type === T.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(s[0], 24, 16);
    } else if (type === T.mjGEOM_CAPSULE.value) {
      geometry = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 24).rotateX(Math.PI / 2);
    } else {
      continue;
    }

    const matId = model.geom_matid[g];
    const rgba = matId >= 0 ? model.mat_rgba.subarray(4 * matId, 4 * matId + 4) : model.geom_rgba.subarray(4 * g, 4 * g + 4);
    const key = Array.from(rgba).join();
    let material = materials.get(key);
    if (!material) {
      const color = new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace);
      const dark = color.getHSL({ h: 0, s: 0, l: 0 }).l < 0.2;
      material = new THREE.MeshStandardMaterial({ color, roughness: dark ? 0.45 : 0.6, metalness: 0 });
      materials.set(key, material);
    }

    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.matrixAutoUpdate = false;
    root.add(mesh);
    items.push({ geom: g, object: mesh });
  }
  return { root, items };
}

// Mesh vertices and per-corner normals as MuJoCo computed them, in the geom frame.
function meshGeometry(model: MjModel, id: number) {
  const vert = model.mesh_vert;
  const face = model.mesh_face;
  const normal = model.mesh_normal;
  const faceNormal = model.mesh_facenormal;
  const vertAdr = model.mesh_vertadr[id];
  const normalAdr = model.mesh_normaladr[id];
  const faceAdr = model.mesh_faceadr[id];
  const faceNum = model.mesh_facenum[id];
  const position = new Float32Array(9 * faceNum);
  const normals = new Float32Array(9 * faceNum);
  for (let i = 0; i < 3 * faceNum; i++) {
    const v = 3 * (vertAdr + face[3 * faceAdr + i]);
    const n = 3 * (normalAdr + faceNormal[3 * faceAdr + i]);
    position.set(vert.subarray(v, v + 3), 3 * i);
    normals.set(normal.subarray(n, n + 3), 3 * i);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  return geometry;
}

function buildControls(sim: Sim) {
  const poses = $("poses");
  const buttons = POSES.map((pose) => {
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `<span class="long">${pose.name}</span><span class="short">${pose.short}</span>`;
    button.title = pose.name;
    button.onclick = () => {
      tween = { from: [...sim.target], to: pose.ctrl, start: performance.now() };
      buttons.forEach((b) => b.classList.toggle("active", b === button));
    };
    poses.append(button);
    return button;
  });
  buttons[0].classList.add("active");

  const joints = $("joints");
  const sliders = JOINTS.map(([name, label], i) => {
    const row = document.createElement("div");
    row.className = "joint";
    row.innerHTML = `<label for="j-${name}">${label}</label><input id="j-${name}" type="range" step="any"><output></output>`;
    const input = row.querySelector("input")!;
    input.min = String(sim.range[i][0]);
    input.max = String(sim.range[i][1]);
    input.oninput = () => {
      tween = null;
      buttons.forEach((b) => b.classList.remove("active"));
      sim.setCtrl(sim.target.map((v, j) => (j === i ? Number(input.value) : v)));
    };
    joints.append(row);
    return { input, output: row.querySelector("output")! };
  });

  const reset = $<HTMLButtonElement>("reset");
  reset.disabled = false;
  reset.onclick = () => {
    tween = null;
    sim.reset();
    buttons.forEach((b, i) => b.classList.toggle("active", i === 0));
  };

  // Sliders show the target, the numbers show where each joint actually is.
  const clock = $("clock");
  return () => {
    const angles = sim.joints();
    sliders.forEach(({ input, output }, i) => {
      input.value = String(sim.target[i]);
      output.value = `${Math.round(THREE.MathUtils.radToDeg(angles[i]))}°`;
    });
    clock.textContent = `${sim.data.time.toFixed(1)} s simulated`;
  };
}

// Fetch the gzip model pack with progress, then split it into [name, bytes] files.
async function fetchPack(url: string, onProgress: (fraction: number) => void): Promise<[string, Uint8Array][]> {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 6.5e6;
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (const reader = response.body.getReader(); ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(Math.min(loaded / total, 1));
  }
  let bytes: Uint8Array = new Uint8Array(await new Blob(chunks as BlobPart[]).arrayBuffer());
  // Unpack unless a server already removed the gzip layer.
  if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream("gzip"));
    bytes = new Uint8Array(await new Response(stream).arrayBuffer());
  }
  const length = new DataView(bytes.buffer).getUint32(0, true);
  const header: [string, number][] = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + length)));
  let offset = 4 + length;
  return header.map(([name, size]) => [name, bytes.subarray(offset, (offset += size))]);
}
