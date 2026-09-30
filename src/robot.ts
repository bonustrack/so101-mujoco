// The SO-101 on its wheeled base in MuJoCo, without any drawing: physics, timed motions, driving,
// inverse kinematics and the scene facts the agent reads. main.ts draws it, and
// scripts/check-agent.ts runs it headless with Bun.
//
// Frames: every position the agent reads or asks for is in the arm's frame on the floor: x forward,
// y left, from the arm's base, z up from the floor. The arm rides on the chassis, so this frame moves
// when the base drives. Only the editor and the drawing work in world coordinates.
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

// Rest: folded, fingers down, the jaws 12 cm above the floor and 15 cm ahead, so the base can drive
// over and up to objects without the arm touching them. Radians, one value per joint.
export const REST = [0, -1.14, 0.92, 1.6, 0, 0];
export const GRIPPER = 5;
export const OPEN = 0.8; // jaws about 7 cm apart
export const CLOSED = -0.17; // fully shut: the jaw stalls on whatever is between them

const ARM = 5; // shoulder pan to wrist roll; the IK moves these, never the gripper
export const LIFTED = 0.05; // an object counts as lifted when its lowest point is 5 cm above the table

export type Vec3 = [number, number, number];
export type Robot = ReturnType<typeof createRobot>;

// Objects come from a fixed pool in scene_web.xml (box0 to box5, ball0 to ball5), so the
// scene can change without recompiling: unused ones wait below the floor with collisions off.
export type Kind = "box" | "ball";
export const POOL = 6; // objects of each kind
export type Obj = {
  kind: Kind;
  label: string; // "Box 1"
  body: number;
  geom: number;
  qpos: number;
  dof: number;
  bvh: number;
  size: Vec3; // half extents in metres; a ball's three values are its radius
  active: boolean;
  home: [number, number, number]; // x, y and yaw that Reset puts it back to
  order: number; // when it was added, so "the first box" is well defined
};
const DENSITY = 1000; // kg/m³: the 3 x 3 x 4 cm box weighs 36 g
export const DEFAULT_SIZE: Record<Kind, Vec3> = { box: [0.015, 0.015, 0.02], ball: [0.02, 0.02, 0.02] };
export const SIZE_RANGE = [0.01, 0.04]; // half extents: 2 to 8 cm across, the smallest the jaws still hold
// Grab limits, measured with scripted picks: the open jaws take a box side up to 6 cm, and fingers-down
// reach needs a box at most 6 cm high. Balls of every size in range work: the jaws close around them.
export const JAWS = 0.06;
export const TALLEST = 0.06;
const FIRST_BOX: [number, number, number] = [0.2, 0.08, 0]; // where the scene's first box starts
const PARKED = [0, 0, -1, 1, 0, 0, 0]; // unused objects: 1 m below the floor, collisions off
const DRAG_LIFT = 0.004; // a dragged object floats 4 mm above the floor

// The wheeled base (model/scene_web.xml). One drive step is 10 cm, one turn step 15°.
export const STEP = 0.1;
export const TURN = (15 * Math.PI) / 180;
// The robot's outline on the floor in the arm's frame, wheels included.
export const BODY = { front: 0.08, back: -0.13, side: 0.095 };
const WHEEL = 0.03; // wheel radius
const TRACK = 0.17; // between the left and right wheels
const SPEED = 0.12; // m/s when driving straight
const SPIN = 3; // rad/s asked of the chassis when turning; the wheels skid sideways, so it turns slower
const SETTLE = 0.3; // seconds the wheels hold still after a move
export type Moved = { moved: number; turned: number; bumped: Obj | null };

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
  const jacp = new mujoco.DoubleBuffer(3 * model.nv);
  const jacr = new mujoco.DoubleBuffer(3 * model.nv);
  const chassis = id(obj.mjOBJ_BODY, "chassis");
  const wheels = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"].map((name) => ({ id: id(obj.mjOBJ_ACTUATOR, name), left: name.endsWith("l") }));
  // Where the arm's frame stands on the floor, and where it faces (radians, counterclockwise from world x).
  const base = () => {
    const R = data.xmat.subarray(9 * chassis, 9 * chassis + 9);
    return { x: data.xpos[3 * chassis] as number, y: data.xpos[3 * chassis + 1] as number, yaw: Math.atan2(R[3], R[0]) };
  };
  const toArm = (p: ArrayLike<number>): Vec3 => {
    const { x, y, yaw } = base();
    const [c, s, dx, dy] = [Math.cos(yaw), Math.sin(yaw), p[0] - x, p[1] - y];
    return [c * dx + s * dy, -s * dx + c * dy, p[2]];
  };
  const toWorld = (p: ArrayLike<number>): Vec3 => {
    const { x, y, yaw } = base();
    const [c, s] = [Math.cos(yaw), Math.sin(yaw)];
    return [x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2]];
  };

  const objects: Obj[] = (["box", "ball"] as const).flatMap((kind) =>
    Array.from({ length: POOL }, (_, i) => {
      const body = id(obj.mjOBJ_BODY, `${kind}${i}`);
      const joint = model.body_jntadr[body];
      return {
        kind,
        label: `${kind === "box" ? "Box" : "Ball"} ${i + 1}`,
        body,
        geom: model.body_geomadr[body] as number,
        qpos: model.jnt_qposadr[joint] as number,
        dof: model.jnt_dofadr[joint] as number,
        bvh: model.body_bvhadr[body] as number,
        size: [...DEFAULT_SIZE[kind]] as Vec3,
        active: false,
        home: [0, 0, 0] as [number, number, number],
        order: 0,
      };
    }),
  );
  // The robot's collision geoms (arm, chassis and wheels), which a new object must not land on.
  const armGeoms = Array.from({ length: model.ngeom }, (_, g) => g).filter(
    (g) => model.geom_bodyid[g] > 0 && !objects.some((o) => o.geom === g) && (model.geom_contype[g] || model.geom_conaffinity[g]),
  );
  const byGeom = new Map(objects.map((o) => [o.geom, o]));
  const floor = id(obj.mjOBJ_GEOM, "floor");
  let added = 0;
  let dragged: { o: Obj; pose: number[] } | null = null;
  let motion: { path: number[][]; start: number; duration: number; settle: number; done: () => void; until?: () => boolean } | null = null;
  // A base move: its start pose, how far it has turned so far, and when it stopped (-1 while it runs).
  // `before`: the objects it touched at the start, with where they stood then.
  type Rolling = { kind: "drive" | "turn"; amount: number; x: number; y: number; yaw: number; last: number; turned: number; start: number; stopped: number; before: Map<Obj, number[]>; bumped: Obj | null; done: (moved: Moved) => void };
  let rolling: Rolling | null = null;
  let ticks = 0;
  // Parked, the base is braked: the "brake" weld in scene_web.xml holds the chassis where it stopped. Driving
  // turns it off.
  const chassisQpos = model.jnt_qposadr[model.body_jntadr[chassis]] as number;
  const brake = id(obj.mjOBJ_EQUALITY, "brake");
  const EQUALITY = mujoco.mjtDisableBit.mjDSBL_EQUALITY.value;
  const park = () => {
    wheelSpeeds(0, 0);
    const [x, y, z, w, qx, qy, qz] = data.qpos.subarray(chassisQpos, chassisQpos + 7);
    // The weld holds the world's pose as seen from the chassis: the chassis pose, inverted.
    const inverse = [w, -qx, -qy, -qz];
    model.eq_data.set([0, 0, 0, ...rotate([-x, -y, -z], inverse), ...inverse, 1], 11 * brake);
    model.opt.disableflags &= ~EQUALITY;
  };
  const release = () => (model.opt.disableflags |= EQUALITY);

  // Write a free joint's pose (x y z and quaternion) and stop it.
  const pin = (o: Obj, pose: number[]) => {
    data.qpos.set(pose, o.qpos);
    data.qvel.fill(0, o.dof, o.dof + 6);
  };
  const wheelSpeeds = (v: number, w: number) => {
    for (const { id, left } of wheels) data.ctrl[id] = (v + (left ? -w : w) * (TRACK / 2)) / WHEEL;
  };
  // The objects the robot touches right now, other than one gripped by both jaws. One pass over the contacts.
  const bumping = () => {
    const touched = new Set<Obj>();
    const gripped = new Map<Obj, number>();
    const contacts = data.contact;
    for (let i = 0; i < data.ncon; i++) {
      const c = contacts.get(i);
      if (!c) continue;
      const [a, b] = [byGeom.get(c.geom1), byGeom.get(c.geom2)];
      const o = a ?? b;
      const other = a ? c.geom2 : c.geom1;
      if (o && o.active && armGeoms.includes(other)) {
        touched.add(o);
        const body = model.geom_bodyid[other];
        if (body === hand) gripped.set(o, (gripped.get(o) ?? 0) | 1);
        if (body === jaw) gripped.set(o, (gripped.get(o) ?? 0) | 2);
      }
      c.delete();
    }
    contacts.delete();
    for (const [o, jaws] of gripped) if (jaws === 3) touched.delete(o);
    return touched;
  };
  // How far the current base move has gone: along its start heading, and turned.
  const progress = (r: Rolling): Moved => {
    const b = base();
    return { moved: (b.x - r.x) * Math.cos(r.yaw) + (b.y - r.y) * Math.sin(r.yaw), turned: r.turned, bumped: r.bumped };
  };
  // One control tick of a base move: wheel speeds from the chassis pose, a feedback loop that ramps up,
  // cruises, slows down on the target and holds the heading when driving straight. It stops on target,
  // on a timeout, when the robot bumps into an object it was not touching at the start, or when an object
  // it touched at the start moves 4 mm: it never pushes.
  const roll = (r: Rolling) => {
    const b = base();
    r.turned += wrap(b.yaw - r.last);
    r.last = b.yaw;
    const t = data.time - r.start;
    const drive = r.kind === "drive";
    const left = r.amount - (drive ? progress(r).moved : r.turned);
    // Near the target, the floor speed keeps the wheels turning against the skid; once there, only a soft pull remains.
    const near = r.stopped >= 0;
    const speed = (cap: number, gain: number, floor: number) => Math.sign(left) * Math.min(cap, gain * Math.abs(left) + (near ? 0 : floor));
    let v = drive ? speed(Math.min(SPEED, 0.4 * t + 0.02), 3, 0.01) : 0;
    let w = drive ? -8 * r.turned : speed(Math.min(SPIN, 3 * t + 0.2), 6, 0.15);
    if (!near) {
      if (Math.abs(left) < (drive ? 0.002 : 0.004)) r.stopped = data.time;
      if (t > 2 + 1.5 * Math.abs(r.amount) * (drive ? 1 / SPEED : 1)) r.stopped = data.time;
      if (r.stopped < 0 && ticks % 2 === 0) {
        const hit =
          [...bumping()].find((o) => !r.before.has(o)) ??
          [...r.before].find(([o, [x, y]]) => Math.hypot(data.xpos[3 * o.body] - x, data.xpos[3 * o.body + 1] - y) > 0.004)?.[0];
        if (hit) {
          r.bumped = hit;
          r.stopped = data.time;
          v = w = 0;
        }
      }
    }
    if (r.bumped) v = w = 0;
    wheelSpeeds(v, w);
    if (r.stopped >= 0 && data.time - r.stopped >= SETTLE) {
      rolling = null;
      park();
      r.done(progress(r));
    }
  };
  // An object's heading in the world.
  const heading = (o: Obj) => Math.atan2(data.xmat[9 * o.body + 3], data.xmat[9 * o.body]);
  const upright = (o: Obj, x: number, y: number, yaw: number, lift: number) => [
    x,
    y,
    o.size[2] + lift,
    Math.cos(yaw / 2),
    0,
    0,
    Math.sin(yaw / 2),
  ];
  const collide = (o: Obj, on: boolean) => {
    const v = on ? 1 : 0;
    model.geom_contype[o.geom] = model.geom_conaffinity[o.geom] = v;
    model.body_contype[o.body] = model.body_conaffinity[o.body] = v;
  };
  // Size, bounds, mass and inertia together, then the constants that depend on mass.
  // setConst runs on the IK scratch data, so the live state is untouched.
  const setSize = (o: Obj, size: Vec3) => {
    const [a, b, c] = (o.size = o.kind === "ball" ? [size[0], size[0], size[0]] : ([...size] as Vec3));
    model.geom_size.set(o.size, 3 * o.geom);
    model.geom_rbound[o.geom] = o.kind === "box" ? Math.hypot(a, b, c) : a;
    model.geom_aabb.set([0, 0, 0, a, b, c], 6 * o.geom);
    model.bvh_aabb.set([0, 0, 0, a, b, c], 6 * o.bvh);
    const mass = DENSITY * (o.kind === "box" ? 8 * a * b * c : (4 / 3) * Math.PI * a ** 3);
    model.body_mass[o.body] = mass;
    model.body_inertia.set(
      o.kind === "box" ? [(b * b + c * c) / 3, (a * a + c * c) / 3, (a * a + b * b) / 3].map((k) => k * mass) : [0.4 * mass * a * a, 0.4 * mass * a * a, 0.4 * mass * a * a],
      3 * o.body,
    );
    mujoco.mj_setConst(model, scratch);
  };

  const robot = {
    mujoco,
    model,
    data,
    range,
    objects,
    // The object the pick commands, the observation and the success check work on.
    focus: null as Obj | null,
    target: [...REST],
    setCtrl(values: number[]) {
      values.forEach((v, i) => {
        robot.target[i] = Math.min(Math.max(v, range[i][0]), range[i][1]);
        data.ctrl[actuators[i].id] = robot.target[i];
      });
    },
    joints: () => actuators.map(({ qpos }) => data.qpos[qpos] as number),

    // One physics step. Unused and dragged objects are held in place; a running motion sets the actuator targets,
    // a running base move the wheel speeds.
    step() {
      for (const o of objects) if (!o.active) pin(o, PARKED);
      if (dragged) pin(dragged.o, dragged.pose);
      ticks++;
      if (rolling) roll(rolling);
      if (motion) {
        const t = data.time - motion.start;
        const k = Math.min(t / motion.duration, 1);
        const e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2; // ease in and out
        const s = e * (motion.path.length - 1);
        const i = Math.min(Math.floor(s), motion.path.length - 2);
        const a = motion.path[i];
        const b = motion.path[i + 1];
        robot.setCtrl(a.map((v, j) => v + (b[j] - v) * (s - i)));
        // Stop where the arm is the moment `until` holds, then settle there.
        if (motion.until?.()) Object.assign(motion, { until: undefined, path: [[...robot.target], [...robot.target]], start: data.time - motion.duration });
        if (t >= motion.duration + motion.settle) {
          const { done } = motion;
          motion = null;
          done();
        }
      }
      mujoco.mj_step(model, data);
    },
    // Glide the actuator targets through `path` (full 6-joint targets), then hold for `settle` seconds.
    // Resolves in simulated time, so it works the same live and headless. `until`, checked every step, ends the glide early.
    play(path: number[][], duration: number, settle = 0.25, until?: () => boolean) {
      robot.stopArm();
      return new Promise<void>((done) => {
        motion = { path: [[...robot.target], ...path], start: data.time, duration: Math.max(duration, 1e-3), settle, done, until };
      });
    },
    hold: (seconds: number) => robot.play([[...robot.target]], 1e-3, seconds),
    stopArm() {
      const done = motion?.done;
      motion = null;
      done?.();
    },
    stopBase() {
      const r = rolling;
      rolling = null;
      park();
      r?.done(progress(r));
    },
    // Stop everything: the arm where it is, the wheels at once.
    cancel() {
      robot.stopArm();
      robot.stopBase();
    },

    // The base: drive `amount` metres straight ahead (negative: back), or turn `amount` radians on the spot
    // (positive: left). The arm holds its pose meanwhile. Resolves in simulated time, once the wheels have
    // held still for 0.3 s, with how far it went and the object it bumped into or pushed, if any.
    move(kind: "drive" | "turn", amount: number) {
      robot.stopBase();
      return new Promise<Moved>((done) => {
        const { x, y, yaw } = base();
        const before = new Map([...bumping()].map((o) => [o, [data.xpos[3 * o.body], data.xpos[3 * o.body + 1]]]));
        rolling = { kind, amount, x, y, yaw, last: yaw, turned: 0, start: data.time, stopped: -1, before, bumped: null, done };
        release();
      });
    },
    base,
    toArm,
    toWorld,
    // A point in the arm's frame: how far from the arm's base, and its angle, left positive.
    polar: (p: ArrayLike<number>) => ({ r: Math.hypot(p[0], p[1]), a: Math.atan2(p[1], p[0]) }),

    // Arm back to rest, the base back to its start and every object back to where it was placed.
    reset() {
      robot.cancel();
      dragged = null;
      mujoco.mj_resetData(model, data);
      park();
      actuators.forEach(({ qpos }, i) => (data.qpos[qpos] = REST[i]));
      robot.setCtrl(REST);
      for (const o of objects) pin(o, o.active ? upright(o, ...o.home, 0.001) : PARKED);
      mujoco.mj_forward(model, data);
    },
    // Back to the first scene: one 3 x 3 x 4 cm box.
    resetScene() {
      dragged = null;
      for (const o of objects) robot.remove(o);
      const box = robot.add("box")!;
      robot.place(box, ...FIRST_BOX);
    },

    // Scene editing. Every change runs mj_forward, so the drawing and the facts follow at once.
    active: () => objects.filter((o) => o.active).sort((a, b) => a.order - b.order),
    // The next free object of that kind, dropped from 2 cm onto a clear spot in reach, or from above the
    // others when the table is full. Null when all 6 are out.
    add(kind: Kind) {
      const o = objects.find((o) => o.kind === kind && !o.active);
      if (!o) return null;
      setSize(o, DEFAULT_SIZE[kind]);
      o.active = true;
      o.order = ++added;
      collide(o, true);
      const { x, y, room } = robot.clearSpot(o);
      const tops = robot.active().map((other) => (other === o ? 0 : robot.object(other).top));
      robot.place(o, x, y, 0, room > 0 ? 0.02 : Math.max(...tops) + 0.01);
      return o;
    },
    remove(o: Obj) {
      o.active = false;
      collide(o, false);
      if (dragged?.o === o) dragged = null;
      if (robot.focus === o) robot.focus = null;
      pin(o, PARKED);
      mujoco.mj_forward(model, data);
    },
    // New half extents. The object keeps its spot and its lowest point, so it never sinks into the table.
    resize(o: Obj, size: Vec3) {
      const before = robot.object(o).bottom;
      setSize(o, size);
      data.qpos[o.qpos + 2] += before - robot.object(o).bottom;
      data.qvel.fill(0, o.dof, o.dof + 6);
      mujoco.mj_forward(model, data);
    },
    // Put an object upright at (x, y) in the arm's frame, turned by `yaw` radians, `lift` above the floor.
    // Reset brings it back to this spot on the floor, wherever the base is then.
    place(o: Obj, x: number, y: number, yaw: number, lift = 0.001) {
      const [wx, wy] = toWorld([x, y, 0]);
      o.home = [wx, wy, yaw + base().yaw];
      pin(o, upright(o, ...o.home, lift));
      mujoco.mj_forward(model, data);
    },
    // Hold an object upright just above the floor at world (x, y) while the pointer drags it: the physics
    // keeps running around it, and it pushes what it meets. drop() lets it fall and settle.
    drag(o: Obj, x: number, y: number) {
      const yaw = dragged?.o === o || o.kind === "box" ? heading(o) : 0;
      dragged = { o, pose: upright(o, x, y, yaw, DRAG_LIFT) };
      pin(o, dragged.pose);
      mujoco.mj_forward(model, data);
    },
    drop() {
      if (!dragged) return;
      const { o, pose } = dragged;
      dragged = null;
      o.home = [pose[0], pose[1], heading(o)];
      data.qvel.fill(0, o.dof, o.dof + 6);
    },
    // How much room `o` would have at (x, y): the gap to the nearest other object or robot part it could touch.
    // Negative when it would overlap one. The robot counts where it stands now, unless `arm` is false.
    roomFinder(o: Obj, arm = true) {
      const height = 2 * o.size[2] + 0.03;
      const obstacles = [
        ...robot.active().flatMap((other) => (other === o ? [] : [[...robot.object(other).pos, robot.footprint(other)]])),
        ...(arm ? armGeoms : []).flatMap((g) => {
          const p = toArm(data.geom_xpos.subarray(3 * g, 3 * g + 3));
          const r = model.geom_rbound[g];
          return p[2] - r < height ? [[p[0], p[1], p[2], r]] : [];
        }),
      ];
      return (x: number, y: number) => Math.min(...obstacles.map(([ox, oy, , r]) => Math.hypot(ox - x, oy - y) - r - robot.footprint(o)));
    },
    // A reachable spot with 1 cm to spare, going outward from the front of the arm, else the one with the most room.
    clearSpot(o: Obj) {
      const room = robot.roomFinder(o);
      let best = { x: 0.2, y: 0, room: -Infinity };
      for (const r of [0.2, 0.17, 0.23, 0.14, 0.26])
        for (let i = 0; i <= 10; i++) {
          const a = (i % 2 ? 1 : -1) * Math.ceil(i / 2) * 0.2; // 0, then ±0.2 rad outward to ±1
          const spot = { x: r * Math.cos(a), y: r * Math.sin(a), room: room(r * Math.cos(a), r * Math.sin(a)) };
          if (spot.room >= 0.01) return spot;
          if (spot.room > best.room) best = spot;
        }
      return best;
    },
    // A random clear spot in front of the arm that a top-down grasp can reach, turned at random.
    randomize(o: Obj, random = Math.random) {
      const room = robot.roomFinder(o);
      let best = { x: 0.2, y: 0, room: -Infinity };
      for (let i = 0; i < 50 && best.room < 0.005; i++) {
        const r = 0.16 + 0.08 * random();
        const a = (random() - 0.5) * 1.6;
        const spot = { x: r * Math.cos(a), y: r * Math.sin(a), room: room(r * Math.cos(a), r * Math.sin(a)) };
        if (spot.room > best.room) best = spot;
      }
      robot.place(o, best.x, best.y, (random() - 0.5) * Math.PI);
    },
    // Radius of the circle it covers on the table, standing upright.
    footprint: (o: Obj) => (o.kind === "ball" ? o.size[0] : Math.hypot(o.size[0], o.size[1])),
    // Jev's target: the selected object, else the first box, else the first object.
    pickTarget(selected: Obj | null) {
      const active = robot.active();
      return selected?.active ? selected : (active.find((o) => o.kind === "box") ?? active[0] ?? null);
    },
    // Why the arm cannot grab it, or null when it can.
    tooBig(o: Obj) {
      if (o.kind === "ball") return null;
      if (2 * Math.min(o.size[0], o.size[1]) > JAWS + 1e-9) return `every side is over ${JAWS * 100} cm, wider than the open jaws`;
      if (2 * o.size[2] > TALLEST + 1e-9) return `over ${TALLEST * 100} cm high, out of reach from above`;
      return null;
    },

    // Scene facts, all computed from the physics state, in the arm's frame. `world` is for the editor.
    object(target?: Obj) {
      const o = target ?? robot.focus!;
      const world = Array.from(data.xpos.subarray(3 * o.body, 3 * o.body + 3)) as Vec3;
      const p = toArm(world);
      const R = data.xmat.subarray(9 * o.body, 9 * o.body + 9);
      const [a, b, c] = o.size;
      const ball = o.kind === "ball";
      // Lowest point: half extents projected on the vertical.
      const bottom = p[2] - (ball ? a : Math.abs(R[6]) * a + Math.abs(R[7]) * b + Math.abs(R[8]) * c);
      return {
        pos: p,
        world,
        // A ball has no faces: the jaws line up with the direction from the base instead.
        yaw: ball ? Math.atan2(p[1], p[0]) : wrap(heading(o) - base().yaw),
        upright: ball || R[8] > 0.95,
        bottom,
        top: 2 * p[2] - bottom,
      };
    },
    tcp: () => toArm(data.site_xpos.subarray(3 * tcp, 3 * tcp + 3)),
    // Direction the jaws close along, as a heading in radians.
    handYaw: () => wrap(Math.atan2(data.xmat[9 * hand + 3], data.xmat[9 * hand]) - base().yaw),
    // Which jaws touch the target right now, and whether any part of the arm does.
    contacts: () => robot.touching(robot.focus),
    // Held between both jaws.
    gripped: (o: Obj) => {
      const t = robot.touching(o);
      return t.fixed && t.moving;
    },
    // Everything that touches `o` right now: each jaw, any arm part, other objects, the table.
    touching(o: Obj | null) {
      let fixed = false;
      let moving = false;
      let arm = false;
      let table = false;
      const others = new Set<Obj>();
      const geom = o?.geom ?? -1;
      // data.contact is a fresh copy of every contact on each read: read it once and free it.
      const contacts = data.contact;
      for (let i = 0; i < data.ncon; i++) {
        const c = contacts.get(i);
        if (!c) continue;
        const other = c.geom1 === geom ? c.geom2 : c.geom2 === geom ? c.geom1 : -1;
        if (other >= 0) {
          const body = model.geom_bodyid[other];
          if (body === hand) fixed = true;
          if (body === jaw) moving = true;
          if (armGeoms.includes(other)) arm = true;
          if (other === floor) table = true;
          const touched = byGeom.get(other);
          if (touched) others.add(touched);
        }
        c.delete();
      }
      contacts.delete();
      return { fixed, moving, arm, table, others };
    },
    // Linear speed in m/s.
    speed: (o: Obj) => Math.hypot(...data.qvel.subarray(o.dof, o.dof + 3)),
    // Target centre in the hand frame: x across the jaws, z along the fingers (tips near z = -0.1).
    objectInHand() {
      const p = robot.object().world;
      const o = data.xpos.subarray(3 * hand, 3 * hand + 3);
      const R = data.xmat.subarray(9 * hand, 9 * hand + 9);
      const d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
      return [0, 1, 2].map((c) => R[c] * d[0] + R[3 + c] * d[1] + R[6 + c] * d[2]) as Vec3;
    },
    // Lifted and held: code decides success, not the model.
    held() {
      const { fixed, moving } = robot.contacts();
      return robot.object().bottom >= LIFTED && fixed && moving;
    },

    // Inverse kinematics: arm joints that put the jaw tips at `target` with the fingers pointing
    // straight down and the jaws closing along `yaw`. Damped least squares on mj_jacSite.
    // A lower `weight` trades wrist pitch for position when the pose is out of reach.
    solve(target: Vec3, yaw: number, seed: number[] = robot.joints(), weight = 0.3) {
      scratch.qpos.set(data.qpos);
      actuators.slice(0, ARM).forEach(({ qpos }, i) => (scratch.qpos[qpos] = seed[i]));
      // The IK works in the world, where the arm stands now.
      const pos = toWorld(target);
      const turn = yaw + base().yaw;
      const want = [
        [Math.cos(turn), Math.sin(turn), 0], // hand x: across the jaws
        [-Math.sin(turn), Math.cos(turn), 0], // hand y
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
    // The target's width between jaws that close along `yaw`: its x side, or its y side a quarter turn later.
    across(yaw: number) {
      const o = robot.focus!;
      const k = Math.abs(Math.round((yaw - robot.object().yaw) / (Math.PI / 2))) % 2;
      return 2 * o.size[k];
    },
    // Where the jaw tips go to grip the target at height `z`, jaws closing along `yaw`. The tips sit
    // nearer the fixed jaw, so a wide object is gripped off centre to clear that jaw by 5 mm.
    gripPoint(yaw: number, z: number): Vec3 {
      const [x, y] = robot.object().pos;
      const shift = Math.max(0, robot.across(yaw) / 2 - 0.015);
      return [x - shift * Math.cos(yaw), y - shift * Math.sin(yaw), z];
    },
    // The jaw direction that matches a face of the target, fits between the open jaws, keeps the open jaws
    // off every other object once lowered, and needs the least wrist roll from here, with the tip position
    // for it at height `z`.
    graspYaw(z: number) {
      const o = robot.object();
      const faceYaw = o.yaw;
      let best = { q: [] as number[], miss: Infinity, yaw: 0, pos: [0, 0, 0] as Vec3, cost: Infinity };
      for (let k = 0; k < 4; k++) {
        const yaw = faceYaw + (k * Math.PI) / 2;
        const pos = robot.gripPoint(yaw, z);
        const s = robot.solve(pos, yaw, [Math.atan2(pos[1], pos[0]), 0, 0, 1.2, 0]);
        const low = robot.solve(robot.gripPoint(yaw, Math.max(o.bottom + 0.01, o.pos[2] - 0.005)), yaw, s.q);
        const bump = robot.active().length > 1 && robot.collides(low.q, OPEN, robot.focus) ? 0.5 : 0;
        const cost = s.error + 0.01 * Math.abs(s.q[4]) + (robot.across(yaw) > JAWS + 1e-9 ? 1 : 0) + bump;
        if (cost < best.cost) best = { ...s, yaw, pos, cost };
      }
      return best;
    },
    // Whether the arm posed at `q`, gripper at `grip`, would touch any object but `except`. Checked on the
    // IK scratch data, so the live physics is untouched.
    collides(q: number[], grip: number, except: Obj | null) {
      scratch.qpos.set(data.qpos);
      actuators.slice(0, ARM).forEach(({ qpos }, i) => (scratch.qpos[qpos] = q[i]));
      scratch.qpos[actuators[GRIPPER].qpos] = grip;
      mujoco.mj_fwdPosition(model, scratch);
      const contacts = scratch.contact;
      let hit = false;
      for (let i = 0; i < scratch.ncon; i++) {
        const c = contacts.get(i);
        if (!c) continue;
        const a = byGeom.get(c.geom1) ?? byGeom.get(c.geom2);
        const arm = armGeoms.includes(c.geom1) || armGeoms.includes(c.geom2);
        if (a && a !== except && a.active && arm) hit = true;
        c.delete();
      }
      contacts.delete();
      return hit;
    },
  };
  robot.resetScene();
  robot.reset();
  return robot;
}

// An angle wrapped to -180° to 180°.
export function wrap(angle: number) {
  return angle - 2 * Math.PI * Math.round(angle / (2 * Math.PI));
}

// v turned by the unit quaternion q = (w, x, y, z).
function rotate([vx, vy, vz]: number[], [w, x, y, z]: number[]) {
  const [tx, ty, tz] = [2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)];
  return [vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx];
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
