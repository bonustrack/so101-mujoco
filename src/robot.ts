// The SO-101 in MuJoCo without any drawing: physics, timed motions, inverse
// kinematics and the scene facts the agent reads. main.ts draws it, and
// scripts/check-agent.ts runs it headless with Bun.
import type { MainModule, MjData, MjModel } from "@mujoco/mujoco";

// One entry per position actuator, in the model's order.
export const JOINTS = [
  ["shoulder_pan", "Shoulder pan"],
  ["shoulder_lift", "Shoulder lift"],
  ["elbow_flex", "Elbow flex"],
  ["wrist_flex", "Wrist flex"],
  ["wrist_roll", "Wrist roll"],
  ["gripper", "Gripper"],
] as const;

// Poses in radians, one value per joint (from examples/so101.py).
export const REST = [0, -1.57, 1.57, 0.8, 0, 0];
export const GRIPPER = 5;
export const OPEN = 0.8; // jaws about 7 cm apart
export const CLOSED = -0.17; // fully shut: the jaw stalls on whatever is between them

const ARM = 5; // shoulder pan to wrist roll; the IK moves these, never the gripper
export const LIFTED = 0.05; // the box counts as lifted when its lowest corner is 5 cm above the table

export type Vec3 = [number, number, number];
export type Robot = ReturnType<typeof createRobot>;

// Files are [name, bytes] pairs: scene_web.xml, so101.xml and the meshes under assets/.
export function loadModel(mujoco: MainModule, files: [string, Uint8Array][]) {
  mujoco.FS.mkdir("/so101");
  mujoco.FS.mkdir("/so101/assets");
  for (const [name, bytes] of files) mujoco.FS.writeFile(`/so101/${name}`, bytes);
  const model: MjModel = mujoco.MjModel.from_xml_path("/so101/scene_web.xml");
  for (const [name] of files) mujoco.FS.unlink(`/so101/${name}`); // compiled, no longer needed
  return model;
}

export function createRobot(mujoco: MainModule, model: MjModel) {
  const data: MjData = new mujoco.MjData(model);
  const scratch: MjData = new mujoco.MjData(model); // IK works here, so the live physics is untouched
  const obj = mujoco.mjtObj;
  const id = (type: { value: number }, name: string) => {
    const i = mujoco.mj_name2id(model, type.value, name);
    if (i < 0) throw new Error(`${name} not found in the model`);
    return i;
  };

  const actuators = JOINTS.map(([name]) => {
    const a = id(obj.mjOBJ_ACTUATOR, name);
    const joint = model.actuator_trnid[2 * a];
    return { id: a, qpos: model.jnt_qposadr[joint] as number, dof: model.jnt_dofadr[joint] as number };
  });
  const range = actuators.map(({ id }) => [model.actuator_ctrlrange[2 * id], model.actuator_ctrlrange[2 * id + 1]]);
  const tcp = id(obj.mjOBJ_SITE, "gripperframe"); // between the jaw tips
  const hand = id(obj.mjOBJ_BODY, "gripper"); // fixed jaw; +z runs from the tips up the wrist, +x toward the moving jaw
  const jaw = id(obj.mjOBJ_BODY, "moving_jaw_so101_v1");
  const boxBody = id(obj.mjOBJ_BODY, "box");
  const boxGeom = id(obj.mjOBJ_GEOM, "box");
  const boxQpos = model.jnt_qposadr[model.body_jntadr[boxBody]] as number;
  const boxDof = model.jnt_dofadr[model.body_jntadr[boxBody]] as number;
  const boxSize = Array.from(model.geom_size.subarray(3 * boxGeom, 3 * boxGeom + 3)) as Vec3;
  const jacp = new mujoco.DoubleBuffer(3 * model.nv);
  const jacr = new mujoco.DoubleBuffer(3 * model.nv);

  let motion: { path: number[][]; start: number; duration: number; settle: number; done: () => void } | null = null;

  const robot = {
    mujoco,
    model,
    data,
    range,
    boxSize,
    target: [...REST],
    setCtrl(values: number[]) {
      values.forEach((v, i) => {
        robot.target[i] = Math.min(Math.max(v, range[i][0]), range[i][1]);
        data.ctrl[actuators[i].id] = robot.target[i];
      });
    },
    joints: () => actuators.map(({ qpos }) => data.qpos[qpos] as number),

    // One physics step. A running motion sets the actuator targets first.
    step() {
      if (motion) {
        const t = data.time - motion.start;
        const k = Math.min(t / motion.duration, 1);
        const e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2; // ease in and out
        const s = e * (motion.path.length - 1);
        const i = Math.min(Math.floor(s), motion.path.length - 2);
        const a = motion.path[i];
        const b = motion.path[i + 1];
        robot.setCtrl(a.map((v, j) => v + (b[j] - v) * (s - i)));
        if (t >= motion.duration + motion.settle) {
          const { done } = motion;
          motion = null;
          done();
        }
      }
      mujoco.mj_step(model, data);
    },
    // Glide the actuator targets through `path` (full 6-joint targets), then hold for `settle` seconds.
    // Resolves in simulated time, so it works the same live and headless.
    play(path: number[][], duration: number, settle = 0.25) {
      robot.cancel();
      return new Promise<void>((done) => {
        motion = { path: [[...robot.target], ...path], start: data.time, duration: Math.max(duration, 1e-3), settle, done };
      });
    },
    hold: (seconds: number) => robot.play([[...robot.target]], 1e-3, seconds),
    cancel() {
      const done = motion?.done;
      motion = null;
      done?.();
    },

    reset() {
      robot.cancel();
      mujoco.mj_resetData(model, data);
      actuators.forEach(({ qpos }, i) => (data.qpos[qpos] = REST[i]));
      robot.setCtrl(REST);
      mujoco.mj_forward(model, data);
    },
    // Put the box somewhere on the table, upright, turned by `yaw` radians.
    placeBox(x: number, y: number, yaw: number) {
      const q = data.qpos;
      q[boxQpos] = x;
      q[boxQpos + 1] = y;
      q[boxQpos + 2] = boxSize[2] + 0.001;
      q[boxQpos + 3] = Math.cos(yaw / 2);
      q[boxQpos + 4] = 0;
      q[boxQpos + 5] = 0;
      q[boxQpos + 6] = Math.sin(yaw / 2);
      for (let i = 0; i < 6; i++) data.qvel[boxDof + i] = 0;
      mujoco.mj_forward(model, data);
    },
    // A random spot in front of the arm that a top-down grasp can reach.
    randomBox(random = Math.random) {
      const r = 0.16 + 0.08 * random();
      const a = (random() - 0.5) * 1.6;
      robot.placeBox(r * Math.cos(a), r * Math.sin(a), (random() - 0.5) * Math.PI);
    },

    // Scene facts, all computed from the physics state.
    box() {
      const p = data.xpos.subarray(3 * boxBody, 3 * boxBody + 3);
      const R = data.xmat.subarray(9 * boxBody, 9 * boxBody + 9);
      // Lowest corner: half-extents projected on the vertical.
      const bottom = p[2] - (Math.abs(R[6]) * boxSize[0] + Math.abs(R[7]) * boxSize[1] + Math.abs(R[8]) * boxSize[2]);
      return {
        pos: [p[0], p[1], p[2]] as Vec3,
        yaw: Math.atan2(R[3], R[0]),
        upright: R[8] > 0.95,
        bottom,
        top: 2 * p[2] - bottom,
      };
    },
    tcp: () => Array.from(data.site_xpos.subarray(3 * tcp, 3 * tcp + 3)) as Vec3,
    // Direction the jaws close along, as a heading in radians.
    handYaw: () => Math.atan2(data.xmat[9 * hand + 3], data.xmat[9 * hand]),
    // Which jaws touch the box right now.
    contacts() {
      let fixed = false;
      let moving = false;
      for (let i = 0; i < data.ncon; i++) {
        const c = data.contact.get(i);
        if (!c) continue;
        const other = c.geom1 === boxGeom ? c.geom2 : c.geom2 === boxGeom ? c.geom1 : -1;
        if (other >= 0) {
          const body = model.geom_bodyid[other];
          if (body === hand) fixed = true;
          if (body === jaw) moving = true;
        }
        c.delete();
      }
      return { fixed, moving };
    },
    // Box centre in the hand frame: x across the jaws, z along the fingers (tips near z = -0.1).
    boxInHand() {
      const p = data.xpos.subarray(3 * boxBody, 3 * boxBody + 3);
      const o = data.xpos.subarray(3 * hand, 3 * hand + 3);
      const R = data.xmat.subarray(9 * hand, 9 * hand + 9);
      const d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
      return [0, 1, 2].map((c) => R[c] * d[0] + R[3 + c] * d[1] + R[6 + c] * d[2]) as Vec3;
    },
    // Lifted and held: code decides success, not the model.
    held() {
      const { fixed, moving } = robot.contacts();
      return robot.box().bottom >= LIFTED && fixed && moving;
    },

    // Inverse kinematics: arm joints that put the jaw tips at `pos` with the fingers pointing
    // straight down and the jaws closing along `yaw`. Damped least squares on mj_jacSite.
    // A lower `weight` trades wrist pitch for position when the pose is out of reach.
    solve(pos: Vec3, yaw: number, seed: number[] = robot.joints(), weight = 0.3) {
      scratch.qpos.set(data.qpos);
      actuators.slice(0, ARM).forEach(({ qpos }, i) => (scratch.qpos[qpos] = seed[i]));
      const want = [
        [Math.cos(yaw), Math.sin(yaw), 0], // hand x: across the jaws
        [-Math.sin(yaw), Math.cos(yaw), 0], // hand y
        [0, 0, 1], // hand z: fingers point down
      ];
      const W = weight; // orientation weight against metres of position error
      let error = Infinity;
      for (let iter = 0; iter < 200; iter++) {
        mujoco.mj_kinematics(model, scratch);
        mujoco.mj_comPos(model, scratch);
        const p = scratch.site_xpos.subarray(3 * tcp, 3 * tcp + 3);
        const R = scratch.xmat.subarray(9 * hand, 9 * hand + 9);
        const e = [pos[0] - p[0], pos[1] - p[1], pos[2] - p[2], 0, 0, 0];
        for (let c = 0; c < 3; c++) {
          // 0.5 * sum(current axis x wanted axis): the small rotation that aligns the frames.
          const a = [R[c], R[3 + c], R[6 + c]];
          const b = want[c];
          e[3] += 0.5 * W * (a[1] * b[2] - a[2] * b[1]);
          e[4] += 0.5 * W * (a[2] * b[0] - a[0] * b[2]);
          e[5] += 0.5 * W * (a[0] * b[1] - a[1] * b[0]);
        }
        error = Math.hypot(e[0], e[1], e[2]) + Math.hypot(e[3], e[4], e[5]) / W / 10;
        if (Math.hypot(e[0], e[1], e[2]) < 5e-4 && Math.hypot(e[3], e[4], e[5]) < 0.003 * W) break;
        mujoco.mj_jacSite(model, scratch, jacp, jacr, tcp);
        const jp = jacp.GetView() as Float64Array;
        const jr = jacr.GetView() as Float64Array;
        const J = [0, 1, 2, 3, 4, 5].map((r) =>
          actuators.slice(0, ARM).map(({ dof }) => (r < 3 ? jp[r * model.nv + dof] : W * jr[(r - 3) * model.nv + dof])),
        );
        const dq = dampedStep(J, e, 0.02);
        const scale = Math.min(1, 0.2 / Math.max(...dq.map(Math.abs)));
        actuators.slice(0, ARM).forEach(({ qpos }, i) => {
          const [lo, hi] = range[i];
          scratch.qpos[qpos] = Math.min(Math.max(scratch.qpos[qpos] + scale * dq[i], lo), hi);
        });
      }
      mujoco.mj_kinematics(model, scratch);
      const p = scratch.site_xpos.subarray(3 * tcp, 3 * tcp + 3);
      const q = actuators.slice(0, ARM).map(({ qpos }) => scratch.qpos[qpos] as number);
      return { q, miss: Math.hypot(pos[0] - p[0], pos[1] - p[1], pos[2] - p[2]), error };
    },
    // The jaw direction that matches a box face and needs the least wrist roll from here.
    graspYaw(pos: Vec3, boxYaw: number) {
      let best = { q: [] as number[], miss: Infinity, yaw: 0, cost: Infinity };
      for (let k = 0; k < 4; k++) {
        const yaw = boxYaw + (k * Math.PI) / 2;
        const s = robot.solve(pos, yaw, [Math.atan2(pos[1], pos[0]), 0, 0, 1.2, 0]);
        const cost = s.error + 0.01 * Math.abs(s.q[4]);
        if (cost < best.cost) best = { ...s, yaw, cost };
      }
      return best;
    },
  };
  robot.reset();
  return robot;
}

// Solve (JᵀJ + λ²I) dq = Jᵀe for a small dense J (rows x n).
function dampedStep(J: number[][], e: number[], lambda: number) {
  const n = J[0].length;
  const A = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => J.reduce((s, row) => s + row[i] * row[j], 0) + (i === j ? lambda * lambda : 0)),
  );
  const b = Array.from({ length: n }, (_, i) => J.reduce((s, row, r) => s + row[i] * e[r], 0));
  for (let c = 0; c < n; c++) {
    let pivot = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[pivot][c])) pivot = r;
    [A[c], A[pivot]] = [A[pivot], A[c]];
    [b[c], b[pivot]] = [b[pivot], b[c]];
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / A[c][c];
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k];
    x[r] = s / A[r][r];
  }
  return x;
}
