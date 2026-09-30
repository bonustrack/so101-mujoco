// The closed loop: read the goal, plan, then per turn observe the scene, ask Jev for the next command,
// run it with IK and physics, check the result in code, repeat. Jev is a decision model, not a chat
// model: it answers Choice questions over fixed lists, with a probability per option. Numbers,
// geometry and the plan stay in code.
//
// 1. Goal: one Jev call maps the goal text to a task (take, stack the boxes, stack everything, put X on Y,
//    drive or turn, drive to an object) and its parameters (which object, onto which, stack order, where,
//    which way and how far).
// 2. Plan: code expands the task into subgoals (plan.ts), again after every turn: pick-and-place, and a
//    drive up to anything out of the arm's reach.
// 3. Turns: for the current subgoal, Jev picks the next command among the ones that fit it.
// 4. Check: code decides when a subgoal is achieved and when the whole task is.
import { CLOSED, GRIPPER, LIFTED, OPEN, REST, STEP, TURN, wrap, type Obj, type Robot, type Vec3 } from "./robot";
import { clear, driveSteps, driven, inReach, parseRequest, plan, readTask, restingOn, taskSize, toGo, type ChoiceQuestion, type Drive, type Plan, type Subgoal, type Target, type Task, type TaskName } from "./plan";
import { MARGIN, clearance as floorClearance, floorMap, follow, goalsAround, goalsForMove, nearest, pivot, planPath, type FloorMap } from "./nav";

export type ChoiceAnswer = { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number };
export type JevRequest = { model: string; state: Record<string, unknown>; questions: Record<string, ChoiceQuestion> };
export type JevResponse = { model: string; answers: Record<string, ChoiceAnswer>; usage?: { input_tokens: number; output_tokens: number } };
export type Decide = (request: JevRequest, signal: AbortSignal) => Promise<JevResponse>;
export type Observation = ReturnType<typeof observe>;
type Step = { command: string; result: string };
type Place = Extract<Subgoal, { kind: "place" }>;
type Approach = Extract<Subgoal, { kind: "approach" }>;
type DriveSub = Extract<Subgoal, { kind: "drive" }>;

export type Stage = "observe" | "jev" | "ik" | "physics" | "drive" | "check";
export type AgentEvent =
  | { type: "stage"; stage: Stage }
  | { type: "task"; request: JevRequest; answers: Record<string, ChoiceAnswer>; name: TaskName; confidence: number; text: string; ok: boolean; maxTurns: number; model: string; ms: number; usage: JevResponse["usage"] }
  | { type: "plan"; plan: Plan; maxTurns: number }
  | { type: "turn"; turn: number; request: JevRequest; subgoal: string }
  // A step code runs without asking Jev: driving, for now.
  | { type: "skill"; turn: number; skill: string; subgoal: string }
  // The path the chassis centre is about to follow, world x and y.
  | { type: "path"; points: [number, number][] }
  | { type: "decision"; turn: number; answer: ChoiceAnswer; model: string; ms: number; tokens: number; usage: JevResponse["usage"] }
  | { type: "result"; turn: number; command: string; text: string }
  | { type: "check"; turn: number; success: boolean; text: string }
  | { type: "end"; outcome: "success" | "done" | "stopped" | "turns" | "stuck" | "budget" | "error"; text: string };

const HOVER = 0.03; // jaw tips 3 cm above the object top
const CARRY = 0.015; // a carried object 1.5 cm above its target: towers get close to the arm's reach
const LIFT = 0.08;
// The turn budget grows with the objects a task moves: 8 per object, plus 4. An object that falls and has
// to be placed again adds 8, twice at most. A drive gets one turn per step, plus 2; every drive up to
// something out of reach adds 3.
export const maxTurns = (task: Task) => (task.kind === "drive" ? driveSteps(task.drive) + 2 : 8 * taskSize(task) + 4);
const REDO = 8;
const APPROACH = 3;
// The time budget: seconds of motion for the whole task.
export const budget = (task: Task) => 60 + 20 * Math.max(1, taskSize(task));
const SPEEDS = { drive: 0.12, turn: 1.4 }; // what the path planner counts on: m/s straight, rad/s on the spot

const cm = (m: number) => Math.round(m * 1000) / 10;
const deg = (rad: number) => Math.round((rad * 180) / Math.PI);
type Run = (robot: Robot, stage: (s: Stage) => void, target?: Target, sub?: Subgoal) => Promise<string>;
// The base's single steps, which the page's drive pad runs, and the drives code runs for a task.
export const BASE_COMMANDS = ["drive_forward", "drive_backward", "turn_left", "turn_right", "drive", "drive_to_object"];

// The commands Jev chooses from. The text is what Jev reads; `run` is what the arm does.
// Each text names the exact observation values that make the command right, because Jev reads
// criteria literally: vaguer wording let lower_to_box come close to winning while the gripper was away.
// They all act on the subgoal's object, robot.focus.
export const COMMANDS: Record<string, { text: string; run?: Run }> = {
  open_gripper: {
    text: 'Open the jaws wide. Right when `observation.gripper` is "closed" or "partly open" and `observation.gripper_vs_object` is not "holding the object". Wrong when `observation.gripper` is already "open".',
    run: async (robot, stage) => {
      stage("physics");
      const held = robot.contacts();
      await robot.play([[...robot.target.slice(0, GRIPPER), OPEN]], 0.5);
      return held.fixed && held.moving ? "Jaws open. The object was let go." : "Jaws open.";
    },
  },
  move_above_object: {
    text: 'Move the gripper to hover just above the object, fingers pointing down. Right when `observation.gripper_vs_object` is "away from the object".',
    run: async (robot, stage) => {
      stage("ik");
      const object = robot.object();
      let pose = robot.graspYaw(object.top + HOVER);
      if (pose.miss > 0.003) pose = { ...pose, ...robot.solve(pose.pos, pose.yaw, pose.q, 0.05) };
      // Go through a point high above the object, so the arm never sweeps through it.
      const via = robot.solve([pose.pos[0], pose.pos[1], Math.max(0.16, object.top + 0.08)], pose.yaw, pose.q, 0.05);
      const grip = robot.target[GRIPPER];
      // Open jaws still around an object it just set down rise straight up first, so they do not drag it along.
      const tip = robot.tcp();
      const around = robot.active().filter((o) => o !== robot.focus && !clear(robot, o));
      const rise = around.length ? line(robot, tip, [tip[0], tip[1], Math.max(...around.map((o) => robot.object(o).top)) + 0.04], 0.3, robot.handYaw()) : [];
      stage("physics");
      await robot.play([...rise, [...via.q, grip], [...pose.q, grip]], rise.length ? 2.4 : 1.8);
      return reached(robot, pose.pos, "Above the object");
    },
  },
  lower_to_object: {
    text: 'Lower the gripper straight down so the object ends up between the jaws. Right only when `observation.gripper_vs_object` is "above the object" and `observation.gripper` is "open". Wrong when the jaws are closed or the gripper is away from the object: it would hit the object.',
    run: async (robot, stage, target) => {
      stage("ik");
      const object = robot.object();
      const yaw = squareYaw(robot);
      // To take it, grip just under its middle. To set a box on something, grip 1 cm above its bottom (at
      // most 3.5 cm under its top): the higher the tower, the less the arm has to reach up.
      const z = target && robot.focus!.kind === "box" ? Math.max(object.bottom + 0.01, object.top - 0.035) : Math.max(object.bottom + 0.01, object.pos[2] - 0.005);
      const goal = robot.gripPoint(yaw, z);
      const path = line(robot, robot.tcp(), goal, 0.3);
      stage("physics");
      await robot.play(path, 1.0);
      return reached(robot, goal, "Lowered around the object");
    },
  },
  close_gripper: {
    text: 'Close the jaws to grip. Right when `observation.gripper_vs_object` is "the object is between the jaws".',
    run: async (robot, stage) => {
      stage("physics");
      await robot.play([[...robot.target.slice(0, GRIPPER), CLOSED]], 0.8, 0.4);
      const { fixed, moving } = robot.contacts();
      return fixed && moving ? "Jaws closed on the object." : "Jaws closed on nothing.";
    },
  },
  lift: {
    text: 'Raise the gripper 8 cm, carrying what it holds. Right when `observation.gripper_vs_object` is "holding the object" and `observation.object_state` is "on the table".',
    run: async (robot, stage) => {
      stage("ik");
      const tip = robot.tcp();
      const path = line(robot, tip, [tip[0], tip[1], tip[2] + LIFT], 0.05);
      stage("physics");
      await robot.play(path, 1.2);
      return `Raised ${cm(robot.tcp()[2] - tip[2])} cm.`;
    },
  },
  move_above_target: {
    text: 'Carry the held object up, over to the target and hold it just above it. Right when `observation.gripper_vs_object` is "holding the object" and `observation.object_vs_target` is "away from the target".',
    run: async (robot, stage, target) => {
      stage("ik");
      const o = robot.object();
      const tip = robot.tcp();
      const yaw0 = robot.handYaw();
      // Carry it around the arm's base, not across it: the pan swings, the reach stays in the good ring.
      const pan = wrap(Math.atan2(target!.y, target!.x) - Math.atan2(tip[1], tip[0]));
      const turn = landingTurn(robot.focus!, o.yaw, target!, pan);
      const d = [o.pos[0] - tip[0], o.pos[1] - tip[1], o.pos[2] - tip[2]]; // the object's centre from the jaw tips
      const [c, s] = [Math.cos(turn), Math.sin(turn)];
      const below = o.pos[2] - o.bottom; // its centre above its lowest point
      const goal: Vec3 = [target!.x - (c * d[0] - s * d[1]), target!.y - (s * d[0] + c * d[1]), target!.top + CARRY + below - d[2]];
      // Travel high enough that the carried object clears everything on the table.
      const tallest = Math.max(0, ...robot.active().filter((x) => x !== robot.focus).map((x) => robot.object(x).top));
      const safe = Math.max(tip[2], goal[2], tallest + 0.02 + below - d[2]);
      const up = safe - tip[2] > 0.005 ? line(robot, tip, [tip[0], tip[1], safe], TILT, yaw0) : [];
      const over = line(robot, [tip[0], tip[1], safe], [goal[0], goal[1], safe], TILT, yaw0, yaw0 + turn, up.at(-1), true);
      const down = safe - goal[2] > 0.005 ? line(robot, [goal[0], goal[1], safe], goal, TILT, yaw0 + turn, yaw0 + turn, over.at(-1)) : [];
      stage("physics");
      await robot.play([...up, ...over, ...down], 1.2 + 0.4 * (up.length + down.length) / 6 + Math.min(1.2, 8 * Math.hypot(goal[0] - tip[0], goal[1] - tip[1])));
      const a = robot.object();
      const grip = robot.contacts();
      if (!grip.fixed || !grip.moving) return `Dropped ${robot.focus!.label} on the way.`;
      return `Carried over ${target!.label}: ${cm(Math.hypot(a.pos[0] - target!.x, a.pos[1] - target!.y))} cm off centre, ${cm(a.bottom - target!.top)} cm above it.`;
    },
  },
  lower_to_place: {
    text: 'Lower the held object straight down until it sits on the target. Right only when `observation.gripper_vs_object` is "holding the object" and `observation.object_vs_target` is "above the target".',
    run: async (robot, stage, target) => {
      stage("ik");
      const o = robot.object();
      const tip = robot.tcp();
      const yaw = robot.handYaw();
      // Aim 5 mm below contact, and stop at the first touch.
      const goal: Vec3 = [tip[0] + target!.x - o.pos[0], tip[1] + target!.y - o.pos[1], tip[2] - (o.bottom - target!.top) - 0.005];
      const path = line(robot, tip, goal, TILT, yaw);
      const focus = robot.focus!;
      const landed = () => {
        const t = robot.touching(focus);
        return target!.object ? t.others.has(target!.object) : t.table;
      };
      stage("physics");
      await robot.play(path, 1.2, 0.3, landed);
      const v = versus(robot, focus, target!);
      return v.sitting ? `Set down on ${target!.label}, ${cm(v.off)} cm off centre.` : `Lowered, ${cm(v.gap)} cm above ${target!.label}, not on it.`;
    },
  },
  release: {
    text: 'Open the jaws to let go of the object. Right only when `observation.gripper_vs_object` is "holding the object" and `observation.object_vs_target` is "sitting on the target". Wrong when `observation.object_vs_target` is "above the target" or "away from the target": the object would fall.',
    run: async (robot, stage, target) => {
      stage("physics");
      await robot.play([[...robot.target.slice(0, GRIPPER), OPEN]], 0.6, 0.4);
      const o = robot.focus!;
      const on = target!.object ? restingOn(robot, o, target!.object) : robot.touching(o).table;
      return on ? `Let go. ${o.label} sits on ${target!.label}.` : `Let go. ${o.label} is not on ${target!.label}.`;
    },
  },
  retreat: {
    text: 'Move the open jaws away from the object they just let go of. Right when `observation.gripper_vs_object` is "open around the placed object", which is right after release.',
    run: async (robot, stage) => {
      stage("ik");
      const tip = robot.tcp();
      const o = robot.object();
      const yaw = robot.handYaw();
      // Step 6 mm off the fixed jaw, which the object still leans on: the object sits on the jaw's +x side,
      // toward the open moving jaw. Then slide out sideways along the gap between the jaws, toward the arm's
      // base, so the jaws pass by the object even where the arm cannot reach higher. Then rise a little.
      const off: Vec3 = [tip[0] - 0.006 * Math.cos(yaw), tip[1] - 0.006 * Math.sin(yaw), tip[2]];
      let side: Vec3 = [-Math.sin(yaw), Math.cos(yaw), 0];
      if (side[0] * o.pos[0] + side[1] * o.pos[1] > 0) side = [-side[0], -side[1], 0];
      const out = robot.footprint(robot.focus!) + 0.025;
      const away: Vec3 = [off[0] + side[0] * out, off[1] + side[1] * out, tip[2] + 0.01];
      const back = line(robot, tip, off, TILT, yaw);
      const slide = line(robot, off, away, TILT, yaw, yaw, back.at(-1));
      const rise = line(robot, away, [away[0], away[1], Math.max(away[2] + 0.02, o.top + 0.02)], TILT, yaw, yaw, slide.at(-1));
      stage("physics");
      await robot.play([...back, ...slide, ...rise], 1.6);
      return clear(robot, robot.focus!) ? "Moved out, jaws clear." : "Moved out, but the jaws are still around the object.";
    },
  },
  // The drive pad's steps. Driving for a task is code's job: see driveTo.
  drive_forward: { text: "Drive the wheeled base straight forward 10 cm.", run: (robot, stage) => driveStep(robot, stage, false, 1) },
  drive_backward: { text: "Drive the wheeled base straight back 10 cm.", run: (robot, stage) => driveStep(robot, stage, false, -1) },
  turn_left: { text: "Turn the wheeled base left 15 degrees on the spot.", run: (robot, stage) => driveStep(robot, stage, true, 1) },
  turn_right: { text: "Turn the wheeled base right 15 degrees on the spot.", run: (robot, stage) => driveStep(robot, stage, true, -1) },
  go_home: {
    text: "Fold the arm back to its rest pose. Right only when `goal` asks for the rest pose.",
    run: async (robot, stage) => {
      stage("physics");
      await robot.play([[...REST.slice(0, GRIPPER), robot.target[GRIPPER]]], 1.5);
      return "Back at rest.";
    },
  },
  done: {
    text: 'Stop, nothing left to do. Right when `observation` shows `goal` is achieved. To take the object, that means `observation.object_state` says lifted and `observation.gripper_vs_object` is "holding the object".',
  },
};
const DONE_PLACE =
  'Stop, `subgoal` is achieved. Right only when `observation.gripper_vs_object` is "clear of the placed object". Wrong when `observation.gripper_vs_object` is "holding the object" or "open around the placed object": the jaws must let go and move away first.';

// The commands that fit each kind of arm subgoal. Taking one object keeps the original 7.
const KINDS = {
  take: ["open_gripper", "move_above_object", "lower_to_object", "close_gripper", "lift", "go_home", "done"],
  place: ["open_gripper", "move_above_object", "lower_to_object", "close_gripper", "move_above_target", "lower_to_place", "release", "retreat", "done"],
};
const INSTRUCTIONS = {
  take: "A robot arm is working toward `goal`. `observation` is the scene right now and `history` lists the commands already run, oldest first. Which command should the arm run next?",
  place:
    "A robot arm is working toward `goal`, one step at a time. Right now it must do `subgoal`: pick up `observation.object`, set it down on `observation.target`, let go, and move the jaws away from it. `observation` is the scene right now and `history` lists the commands already run for this subgoal, oldest first. Which command should the arm run next?",
};

// One question per turn. Instructions point at state fields by name, as the TypeSafe docs advise.
export function question(kind: keyof typeof KINDS): Record<string, ChoiceQuestion> {
  return {
    next_command: {
      type: "choice",
      instructions: INSTRUCTIONS[kind],
      criteria: Object.fromEntries(KINDS[kind].map((name) => [name, kind === "place" && name === "done" ? DONE_PLACE : COMMANDS[name].text])),
    },
  };
}

// What Jev sees: plain words computed in code, plus the object's type, size and raw positions for context.
export function observe(robot: Robot) {
  const target = robot.focus!;
  const object = robot.object();
  const tip = robot.tcp();
  const angle = robot.joints()[GRIPPER];
  const { fixed, moving } = robot.contacts();
  const holding = fixed && moving;
  const h = robot.objectInHand();
  // Where the object sits in the hand when it is ready to grip (see robot.gripPoint).
  const x = 0.012 + Math.max(0, robot.across(robot.handYaw()) / 2 - 0.015);
  const between = Math.abs(h[0] - x) < 0.015 && Math.abs(h[1]) < 0.015 && h[2] > -0.11 && h[2] < -0.05;
  const sideways = Math.hypot(tip[0] - object.pos[0], tip[1] - object.pos[1]);
  const aboveTop = tip[2] - object.top;
  const [w, d, hgt] = target.size.map((s) => cm(2 * s));
  return {
    object: target.kind,
    object_size_cm: target.kind === "ball" ? { diameter: w } : { width: w, depth: d, height: hgt },
    gripper: holding ? "closed on the object" : angle > 0.5 ? "open" : angle < 0.15 ? "closed" : "partly open",
    gripper_vs_object: holding
      ? "holding the object"
      : between
        ? "the object is between the jaws"
        : sideways < 0.015 + Math.max(0, x - 0.012) && aboveTop > 0
          ? "above the object"
          : "away from the object",
    object_state:
      !object.upright && !holding
        ? "knocked over on the table"
        : object.bottom > 0.005
          ? `lifted ${cm(object.bottom)} cm off the table`
          : "on the table",
    object_position_cm: { forward: cm(object.pos[0]), left: cm(object.pos[1]), up: cm(object.pos[2]) },
    ...(target.kind === "box" && { object_turned_deg: Math.round((object.yaw * 180) / Math.PI) }),
    gripper_tip_cm: { forward: cm(tip[0]), left: cm(tip[1]), up: cm(tip[2]) },
    gripper_to_object_cm: { sideways: cm(sideways), above_object_top: cm(aboveTop) },
  };
}

// The object against its target: sitting on it, held above it, or away from it. A held object counts as
// sitting when it is within 3 mm of the target: the jaws hold a ball a hair above it, and letting go drops it there.
function versus(robot: Robot, o: Obj, target: Target) {
  const a = robot.object(o);
  const touch = robot.touching(o);
  const off = Math.hypot(a.pos[0] - target.x, a.pos[1] - target.y);
  const gap = a.bottom - target.top;
  const t = target.object;
  const reach = !t ? 0.015 : t.kind === "ball" ? 0.006 : Math.max(0.006, 0.7 * Math.min(t.size[0], t.size[1]));
  const held = touch.fixed && touch.moving;
  const down = held ? gap > -0.006 && gap < 0.003 : Math.abs(gap) < 0.006 && (t ? touch.others.has(t) : touch.table);
  const sitting = a.upright && off < reach && down;
  return { sitting, above: !sitting && off < reach && gap > 0, off, gap };
}

// A place subgoal adds the target to the same facts: where it is and where the object is against it.
export function observePlace(robot: Robot, sub: Place) {
  const base = observe(robot);
  const o = sub.object;
  const holding = base.gripper_vs_object === "holding the object";
  const v = versus(robot, o, sub.target);
  const under = holding ? null : robot.active().find((b) => b !== o && restingOn(robot, o, b));
  return {
    ...base,
    object: `${base.object}, ${o.label}`,
    gripper_vs_object: !holding && v.sitting ? (clear(robot, o) ? "clear of the placed object" : "open around the placed object") : base.gripper_vs_object,
    object_state: under ? `resting on ${under.label}` : base.object_state,
    target: sub.target.object ? `the top of ${sub.target.label}` : sub.target.label,
    target_top_cm: { forward: cm(sub.target.x), left: cm(sub.target.y), up: cm(sub.target.top) },
    object_vs_target: v.sitting ? "sitting on the target" : v.above && holding ? "above the target" : "away from the target",
    object_to_target_cm: { sideways: cm(v.off), above_target_top: cm(v.gap) },
  };
}

// Before the base moves: an object in the jaws is raised clear of everything on the floor, its target
// included, else the arm folds to rest with the jaws open, ready for what it drives to, rising first out of
// anything it is near, so the robot drags nothing along and sweeps nothing over.
async function stow(robot: Robot, stage: (s: Stage) => void) {
  const tip = robot.tcp();
  const held = robot.active().find(robot.gripped);
  const near = robot.active().filter((o) => Math.hypot(robot.object(o).pos[0] - tip[0], robot.object(o).pos[1] - tip[1]) < 0.35);
  const tallest = Math.max(0, ...(held ? robot.active() : near).filter((o) => o !== held).map((o) => robot.object(o).top));
  stage("ik");
  if (held) {
    const up = Math.max(0.05, tallest + 0.02) - robot.object(held).bottom;
    if (up < 0.005) return;
    const path = line(robot, tip, [tip[0], tip[1], tip[2] + up], TILT, robot.handYaw());
    stage("physics");
    await robot.play(path, 1.0);
    return;
  }
  const q = robot.joints();
  const folded = REST.slice(0, GRIPPER).every((v, i) => Math.abs(q[i] - v) < 0.05);
  if (folded && q[GRIPPER] > 0.5) return;
  const rise = !folded && tip[2] < tallest + 0.04 ? line(robot, tip, [tip[0], tip[1], tallest + 0.04], TILT, robot.handYaw()) : [];
  stage("physics");
  await robot.play([...rise, [...REST.slice(0, GRIPPER), OPEN]], folded ? 0.5 : rise.length ? 2.2 : 1.5);
}

// One step of the drive pad: 10 cm or 15°.
async function driveStep(robot: Robot, stage: (s: Stage) => void, turn: boolean, sign: 1 | -1) {
  await stow(robot, stage);
  stage("drive");
  const r = await robot.move(turn ? "turn" : "drive", sign * (turn ? TURN : STEP));
  const text = turn ? `Turned ${Math.abs(deg(r.turned))}° ${r.turned > 0 ? "left" : "right"}.` : `Drove ${Math.abs(cm(r.moved))} cm ${r.moved > 0 ? "forward" : "back"}.`;
  return r.bumped ? `${text} Stopped early: the base touched ${r.bumped.label}.` : text;
}

// Drive the base where a subgoal needs it, around everything: a move asked for in the goal, or up to what is
// out of the arm's reach. Code plans the path on a map of the floor (nav.ts) and follows it. At the first touch
// the base stops, the map is made again from where things are now, and it plans again. The same way at most
// twice: after two touches it keeps a wider berth, after three it stops and says why. With no way at all, a
// plain drive goes as far as it can and stops before what is in the way.
async function driveTo(robot: Robot, stage: (s: Stage) => void, emit: (e: AgentEvent) => void, sub: DriveSub | Approach, signal: AbortSignal) {
  await stow(robot, stage);
  stage("drive");
  const what = sub.kind === "drive" ? sub.text.toLowerCase() : `get ${sub.label} in reach`;
  // Carrying something to put down, the chassis keeps clear of the target but the load passes over it.
  const held = robot.active().find(robot.gripped);
  const where = () => robot.toWorld([...(sub.kind === "approach" ? sub.where() : [0, 0]), 0]);
  const soft = () => (sub.kind === "approach" && held ? robot.active().filter((o) => o !== held && Math.hypot(robot.object(o).world[0] - where()[0], robot.object(o).world[1] - where()[1]) < 0.03) : []);
  const goals = (map: FloorMap, yaw: number) => {
    if (sub.kind === "drive") return goalsForMove(map, sub.drive.from, sub.drive.turn, sub.drive.amount);
    const [x, y] = where();
    return goalsAround(map, x, y, sub.distance, yaw, [sub.object, ...soft()]);
  };
  let margin = MARGIN;
  const touched: Obj[] = [];
  for (;;) {
    if (signal.aborted) return { ok: false, text: "Stopped." };
    const map = floorMap(robot, soft(), margin);
    const start = pivot(robot.base());
    stage("ik");
    const path = planPath(map, start, goals(map, start.yaw), SPEEDS);
    if (!path) {
      if (margin > 0.012) {
        margin = 0.012; // a tighter squeeze before giving up
        continue;
      }
      const blocker = nearest(map, start);
      if (sub.kind === "drive" && !sub.drive.turn) return { ok: false, text: await edgeForward(robot, stage, map, sub, blocker?.obj ?? null) };
      return { ok: false, text: `No free way to ${what}${blocker ? `: ${blocker.obj.label} and the others leave no room` : ""}.` };
    }
    emit({ type: "path", points: path.points });
    stage("drive");
    const r = await follow(robot, path, signal);
    if (r.ok) return { ok: true, text: `${touched.length ? `Planned again after touching ${labels(touched)}. ` : ""}Drove ${cm(r.moved)} cm on a path of ${path.segments.length} ${path.segments.length === 1 ? "move" : "moves"}.` };
    if (!r.bumped) return { ok: false, text: "Stopped." };
    touched.push(r.bumped);
    if (touched.length >= 3) return { ok: false, text: `Stopped: the base touched ${labels(touched)} on the way, three times. It does not push, so it gives up.` };
    if (touched.length === 2) margin = 0.035;
  }
}
const labels = (objs: Obj[]) => [...new Set(objs.map((o) => o.label))].join(" and ");

// No way around to a plain drive's end: go straight as far as stays clear, and stop there.
async function edgeForward(robot: Robot, stage: (s: Stage) => void, map: FloorMap, sub: DriveSub, blocker: Obj | null) {
  const sign = Math.sign(toGo(robot, sub.drive));
  const c = pivot(robot.base());
  let d = 0;
  while (d + 0.01 <= Math.abs(toGo(robot, sub.drive)) && clearanceAt(map, c, sign * (d + 0.01)) >= MARGIN) d += 0.01;
  stage("drive");
  if (d > 0.005) await robot.move("drive", sign * d);
  return `Drove ${cm(d)} cm and stopped before ${blocker?.label ?? "what is in the way"}: there is no way to ${sub.text.toLowerCase().replace(/\.$/, "")} without touching it.`;
}
const clearanceAt = (map: FloorMap, c: { x: number; y: number; yaw: number }, d: number) =>
  Math.min(...[0.5, 1].map((k) => floorClearance(map, { x: c.x + k * d * Math.cos(c.yaw), y: c.y + k * d * Math.sin(c.yaw), yaw: c.yaw })));

// Driving: the base is where the goal asked, still 0.5 s later. Approaching: the destination is in reach.
async function checkBase(robot: Robot, sub: DriveSub | Approach) {
  if (sub.kind === "approach") {
    const { r, a } = robot.polar(sub.where());
    const ok = inReach(robot, sub.where());
    return { success: ok, text: `${sub.label} is ${cm(r)} cm away, ${Math.abs(deg(a))}° ${a > 0 ? "left" : "right"}: ${ok ? "in reach" : "not in reach yet"}.` };
  }
  const left = toGo(robot, sub.drive);
  if (!driven(robot, sub.drive)) return { success: false, text: `${sub.drive.turn ? `${Math.abs(deg(left))}°` : `${Math.abs(cm(left))} cm`} to go.` };
  await robot.hold(0.5);
  return { success: true, text: baseMoved(robot, sub.drive) };
}
// How far the base went on the goal's move, in words, and how far it ended beside the straight line.
function baseMoved(robot: Robot, drive: Drive) {
  const done = drive.amount - toGo(robot, drive);
  const { x, y } = robot.base();
  const side = -(x - drive.from.x) * Math.sin(drive.from.yaw) + (y - drive.from.y) * Math.cos(drive.from.yaw);
  const beside = !drive.turn && Math.abs(side) > 0.02 ? `, ending ${Math.abs(cm(side))} cm to the ${side > 0 ? "left" : "right"} of the straight line, around what was in the way` : "";
  return `The base ${drive.turn ? `turned ${Math.abs(deg(done))}° ${done > 0 ? "left" : "right"}` : `moved ${Math.abs(cm(done))} cm ${done > 0 ? "forward" : "back"}`}${beside}.`;
}

// Taking: the target's lowest point 5 cm up, gripped by both jaws, for 1 s.
async function checkTake(robot: Robot) {
  const name = robot.focus!.label;
  if (!robot.held()) {
    const object = robot.object();
    return { success: false, text: object.bottom > 0.005 ? `${name} ${cm(object.bottom)} cm up, not lifted ${cm(LIFTED)} cm yet.` : `${name} not lifted.` };
  }
  await robot.hold(1);
  const success = robot.held();
  return { success, text: success ? `${name} lifted ${cm(robot.object().bottom)} cm and held for 1 s.` : `${name} slipped.` };
}

// Placing: the object sits on its target, let go, jaws clear, and still there 0.5 s later.
async function checkPlace(robot: Robot, task: Task, sub: Place) {
  const o = sub.object;
  const achieved = () => {
    const now = plan(robot, task).steps.find((s) => s.text === stepOf(task, sub));
    return !!now?.done;
  };
  if (!achieved()) {
    const v = versus(robot, o, sub.target);
    const held = robot.touching(o);
    const text = held.fixed && held.moving
      ? v.sitting ? `${o.label} sits on ${sub.target.label}, still in the jaws.` : `${o.label} in the jaws, ${cm(v.off)} cm from ${sub.target.label} sideways, ${cm(v.gap)} cm above it.`
      : v.sitting ? `${o.label} sits on ${sub.target.label}, the jaws are still around it.` : `${o.label} is not on ${sub.target.label}.`;
    return { success: false, text };
  }
  await robot.hold(0.5);
  if (!achieved() || robot.speed(o) > 0.01) return { success: false, text: `${o.label} did not stay on ${sub.target.label}.` };
  const v = versus(robot, o, sub.target);
  return { success: true, text: `${o.label} sits on ${sub.target.label}, ${cm(v.off)} cm off centre, still after 0.5 s.` };
}
// The plan step a subgoal belongs to (the subgoal text names the current top; the step names the tower).
const stepOf = (task: Task, sub: Place) =>
  task.kind === "stack" && sub.object !== task.objects[0] ? `Put ${sub.object.label} on the tower` : sub.text;

// The whole task: every step done, still true 1 s later.
async function checkAll(robot: Robot, task: Task) {
  await robot.hold(1);
  const now = plan(robot, task);
  if (now.current) return { success: false, text: "Something moved during the last second." };
  if (task.kind === "stack") {
    const top = robot.object(task.objects.at(-1)!).top;
    return { success: true, text: `${task.objects.length} objects stacked, ${cm(top)} cm tall, standing for 1 s.` };
  }
  if (task.kind === "drive") return { success: true, text: baseMoved(robot, task.drive) };
  if (task.kind === "drive_to") {
    const { r } = robot.polar(robot.object(task.object).pos);
    return { success: true, text: `${task.object.label} is in reach, ${cm(r)} cm from the arm.` };
  }
  return { success: true, text: "Done and still for 1 s." };
}

export async function runAgent(options: {
  robot: Robot;
  goal: string;
  model: string;
  decide: Decide;
  signal: AbortSignal;
  onEvent: (event: AgentEvent) => void;
  selected?: Obj | null; // the object selected in the scene, if any
}) {
  const { robot, goal, model, decide, signal, onEvent: emit, selected = null } = options;
  const stage = (s: Stage) => emit({ type: "stage", stage: s });
  let history: Step[] = [];
  let tokens = 0;
  let calls = 0;
  let turn = 0;
  const end = (outcome: Extract<AgentEvent, { type: "end" }>["outcome"], text: string) => emit({ type: "end", outcome, text });
  const ask = async (request: JevRequest) => {
    const started = performance.now();
    const response = await decide(request, signal);
    calls++;
    tokens += response.usage?.input_tokens ?? 0;
    return { response, ms: performance.now() - started };
  };
  const tally = () => `${turn} turns, ${calls} Jev calls, ${tokens.toLocaleString("en-US")} input tokens.`;
  try {
    // 1. The goal: one call, the task and its parameters.
    stage("observe");
    const parse = parseRequest(robot, goal, selected);
    const parseRequestBody: JevRequest = { model, ...parse };
    stage("jev");
    const { response: parsed, ms } = await ask(parseRequestBody);
    for (const q of Object.keys(parse.questions))
      if (parsed.answers?.[q]?.type !== "choice") throw new Error(`Unexpected answer from Jev: ${JSON.stringify(parsed).slice(0, 200)}`);
    const reading = readTask(robot, parsed.answers, selected);
    const task = reading.task;
    let limit = task ? maxTurns(task) : 0;
    let most = limit + 2 * REDO;
    emit({ type: "task", request: parseRequestBody, answers: parsed.answers, name: reading.name, confidence: reading.confidence, text: reading.text, ok: !!task, maxTurns: limit, model: parsed.model, ms, usage: parsed.usage });
    if (!task) return end("error", reading.text);

    // 2. Turns, each on the current subgoal of a plan made again from the physics state.
    let current = plan(robot, task);
    emit({ type: "plan", plan: current, maxTurns: limit });
    if (!current.current) {
      stage("check");
      const all = await checkAll(robot, task);
      return end(all.success ? "success" : "done", all.success ? `Already done. ${all.text}` : all.text);
    }
    // The progress monitor: one command runs twice at most from the same state. Then code starts the subgoal
    // afresh, once; the next time it stops and says why. The whole task has a time budget.
    const started = robot.data.time;
    const seconds = budget(task);
    const idle = new Map<string, number>();
    let fresh = 0;
    let subgoal = "";
    for (turn = 1; turn <= limit; turn++) {
      const sub = current.current!;
      if (sub.kind !== "drive" && !sub.object.active) return end("error", `${sub.object.label} was deleted.`);
      if (robot.data.time - started > seconds) return end("budget", `Out of time: ${Math.round(robot.data.time - started)} s of motion, over the ${seconds} s this task gets. Left: ${left(current)}.`);
      if (sub.text !== subgoal) {
        history = [];
        // Each drive up to something out of reach gets its own turns.
        if (sub.kind === "approach") [limit, most] = [limit + APPROACH, most + APPROACH];
      }
      subgoal = sub.text;
      robot.focus = sub.kind === "drive" ? null : sub.object;

      if (sub.kind === "drive" || sub.kind === "approach") {
        // Driving is code's job: no Jev call.
        emit({ type: "skill", turn, skill: "drive", subgoal: sub.text });
        const drove = await driveTo(robot, stage, emit, sub, signal);
        if (signal.aborted) break;
        emit({ type: "result", turn, command: sub.kind === "drive" ? "drive" : "drive_to_object", text: drove.text });
        stage("check");
        const result = await checkBase(robot, sub);
        if (signal.aborted) break;
        emit({ type: "check", turn, ...result });
        if (!result.success && !drove.ok) return end("stuck", drove.text);
      } else {
        stage("observe");
        const observation = sub.kind === "take" ? observe(robot) : observePlace(robot, sub);
        const request: JevRequest = {
          model,
          state: { goal, ...(sub.kind !== "take" && { subgoal: sub.text }), observation, history: history.slice(-4) },
          questions: question(sub.kind),
        };
        emit({ type: "turn", turn, request, subgoal: sub.text });

        stage("jev");
        const { response, ms } = await ask(request);
        const answer = response.answers?.next_command;
        if (answer?.type !== "choice" || !(answer.choice in request.questions.next_command.criteria))
          throw new Error(`Unexpected answer from Jev: ${JSON.stringify(response).slice(0, 200)}`);
        emit({ type: "decision", turn, answer, model: response.model, ms, tokens, usage: response.usage });

        const command = COMMANDS[answer.choice];
        let text: string;
        if (!command.run) {
          stage("check");
          if (sub.kind === "take") {
            const result = await checkTake(robot);
            return end(result.success ? "success" : "done", `Jev chose done. ${result.text}`);
          }
          // A subgoal is over when code says so: done alone never skips one.
          const result = await checkPlace(robot, task, sub);
          text = result.success ? "Checked." : `Not done yet: ${result.text}`;
        } else {
          // In the world: a base move shifts every position seen from the arm, not only the object's.
          const before = robot.object().world;
          const grip = robot.contacts();
          text = await command.run(robot, stage, sub.kind === "place" ? sub.target : undefined, sub);
          if (signal.aborted) break;
          const after = robot.object().world;
          const moved = Math.hypot(after[0] - before[0], after[1] - before[1]);
          // Knocked, not carried.
          if (moved > 0.01 && !(grip.fixed && grip.moving) && !robot.contacts().moving) text += ` The object moved ${cm(moved)} cm.`;
        }
        history.push({ command: answer.choice, result: text });
        emit({ type: "result", turn, command: answer.choice, text });
        // No progress: the same command from the same facts a second time, whether it changed nothing or the
        // arm goes round in a circle.
        const key = `${sub.text}|${answer.choice}|${words(observation)}`;
        const times = (idle.get(key) ?? 0) + 1;
        idle.set(key, times);
        if (times >= 2) {
          if (fresh >= 1) return end("stuck", `Stuck on "${sub.text}": ${answer.choice} ran twice from the same state, again after a fresh start. Last: ${text}`);
          fresh++;
          idle.clear();
          history = [];
          emit({ type: "result", turn, command: "fresh_start", text: `${answer.choice} ran twice from the same state: raising the arm to start "${sub.text}" afresh.` });
          await freshStart(robot, stage);
          if (signal.aborted) break;
        }
      }

      stage("check");
      if (sub.kind === "take") {
        const result = await checkTake(robot);
        if (signal.aborted) break;
        emit({ type: "check", turn, ...result });
        if (result.success) return end("success", `Done in ${tally()}`);
        current = plan(robot, task);
        continue;
      }
      if (sub.kind === "place") {
        const result = await checkPlace(robot, task, sub);
        if (signal.aborted) break;
        emit({ type: "check", turn, ...result });
      }
      const next = plan(robot, task);
      const done = (p: Plan) => p.steps.filter((s) => s.done).length;
      if (done(next) < done(current)) limit = Math.min(most, limit + REDO);
      if (JSON.stringify(next.steps) !== JSON.stringify(current.steps)) emit({ type: "plan", plan: next, maxTurns: limit });
      current = next;
      if (!current.current) {
        const all = await checkAll(robot, task);
        if (signal.aborted) break;
        if (all.success) return end("success", `${all.text} Done in ${tally()}`);
        emit({ type: "check", turn, ...all });
        current = plan(robot, task);
        limit = Math.min(most, limit + REDO);
        emit({ type: "plan", plan: current, maxTurns: limit });
        if (!current.current) return end("done", all.text);
      }
    }
    if (signal.aborted) return end("stopped", `Stop pressed during turn ${turn}.`);
    end("turns", `Stopped after ${limit} turns. ${current.current?.kind === "take" ? `${current.current.object.label} not lifted.` : `Left: ${left(current)}.`}`);
  } catch (error) {
    if (signal.aborted) return end("stopped", `Stop pressed during turn ${turn}.`);
    end("error", error instanceof Error ? error.message : String(error));
  }
}
const left = (p: Plan) => p.steps.filter((s) => !s.done).map((s) => s.text).join(", ");
// The facts in words that Jev reads, without the numbers: the state the monitor compares.
const words = (o: object) => JSON.stringify(Object.entries(o).filter(([, v]) => typeof v === "string"));

// Raise the jaws 5 cm straight up, with or without what they hold: a clean start for the subgoal.
async function freshStart(robot: Robot, stage: (s: Stage) => void) {
  stage("ik");
  const tip = robot.tcp();
  const path = line(robot, tip, [tip[0], tip[1], tip[2] + 0.05], TILT, robot.handYaw());
  stage("physics");
  await robot.play(path, 0.8);
}

// The jaw heading snapped to the nearest face of the target, so the jaws stay square to it.
function squareYaw(robot: Robot) {
  const face = robot.object().yaw;
  const q = Math.PI / 2;
  return face + q * Math.round((robot.handYaw() - face) / q);
}

// How far to turn the jaws, in the world, so a box lands square on the box below it: its long side along
// the long side below when either is clearly longer than wide, else any matching face. Of the turns that
// do it, the one closest to the arm's pan swing, so the wrist rolls least. A ball, or a spot on the table,
// turns with the pan and the wrist does not roll at all.
function landingTurn(o: Obj, yaw: number, target: Target, pan: number) {
  const t = target.object;
  if (target.yaw === null || o.kind === "ball" || !t) return pan;
  const long = (b: Obj) => b.size[0] / b.size[1];
  const square = Math.max(long(o), 1 / long(o), long(t), 1 / long(t)) < 1.15;
  const along = square || long(o) >= 1 === long(t) >= 1 ? 0 : Math.PI / 2; // which of its sides goes along the target's x side
  const step = square ? Math.PI / 2 : Math.PI;
  const base = target.yaw + along - yaw;
  return base + step * Math.round((pan - base) / step);
}

// Straight line for the jaw tips, solved as IK waypoints. The jaws stay square to the target, or turn
// from `yaw0` to `yaw1` along the way. `seed` is the joint pose it starts from. With `weight` TILT, each
// point keeps the fingers pointing down when the arm can, and tilts them only as much as it must to get there.
// With `arc`, the path goes around the arm's base instead, at a radius that changes evenly.
const TILT = -1;
function line(robot: Robot, from: Vec3, to: Vec3, weight: number, yaw0 = squareYaw(robot), yaw1 = yaw0, seed?: number[], arc = false) {
  const grip = robot.target[GRIPPER];
  const path: number[][] = [];
  let q = seed?.slice(0, GRIPPER) ?? robot.joints();
  const [r0, a0] = [Math.hypot(from[0], from[1]), Math.atan2(from[1], from[0])];
  const [r1, sweep] = [Math.hypot(to[0], to[1]), wrap(Math.atan2(to[1], to[0]) - a0)];
  const n = arc ? Math.max(6, Math.ceil(Math.abs(sweep) / 0.15)) : 6;
  for (let i = 1; i <= n; i++) {
    const k = i / n;
    const [r, a] = [r0 + (r1 - r0) * k, a0 + sweep * k];
    const p = (arc ? [r * Math.cos(a), r * Math.sin(a), from[2] + (to[2] - from[2]) * k] : from.map((v, j) => v + (to[j] - v) * k)) as Vec3;
    const yaw = yaw0 + (yaw1 - yaw0) * k;
    q = weight === TILT ? reach(robot, p, yaw, q).q : robot.solve(p, yaw, q, weight).q;
    path.push([...q, grip]);
  }
  return path;
}
function reach(robot: Robot, p: Vec3, yaw: number, seed: number[]) {
  let best = robot.solve(p, yaw, seed, 0.3);
  for (const weight of [0.1, 0.03, 0.01]) {
    if (best.miss < 0.002) break;
    const s = robot.solve(p, yaw, best.q, weight);
    if (s.miss < best.miss) best = s;
  }
  return best;
}

function reached(robot: Robot, target: Vec3, text: string) {
  const miss = Math.hypot(...robot.tcp().map((v, i) => v - target[i]));
  return miss < 0.01 ? `${text}, ${cm(miss)} cm off.` : `Stopped ${cm(miss)} cm short of the target.`;
}
