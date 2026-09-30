// The skills code runs for a task. GoTo drives the base to where a step needs it, Pick takes an object, Place
// sets one down. Each is a short sequence of guarded moves, checked in code as it goes, that says what it did and,
// when it fails, why. The loop (agent.ts) picks the skill for the plan's current step, and a variant of it after
// a failure: the jaws turned 90°, another side, a wider berth.
//
// Skills read the scene from the World (world.ts) and act through the robot: joint targets, the gripper and
// the wheels. Numbers and geometry stay in code.
import { CLOSED, GRIPPER, LIFTED, OPEN, REST, STEP, TURN, WALL, wrap, type Obj, type Robot, type Vec3 } from "./robot";
import { clear, driven, inReach, intoTarget, onTop, restingOn, toGo, type Subgoal, type Target } from "./plan";
import { MARGIN, armOf, clearance, floorMap, follow, goalsAround, goalsForMove, nearest, pivot, planPath, type FloorMap } from "./nav";
import type { World } from "./world";

export type Stage = "plan" | "jev" | "ik" | "physics" | "drive" | "check";
export type Ctx = {
  robot: Robot;
  world: World;
  stage: (s: Stage) => void;
  say: (action: string, text: string) => void; // one action and its result, for the Flow log
  path: (points: [number, number][]) => void; // the path the base is about to drive
  signal: AbortSignal;
};
export type Outcome = { ok: boolean; text: string };
export type Place = Extract<Subgoal, { kind: "place" }>;
export type Approach = Extract<Subgoal, { kind: "approach" }>;
export type DriveSub = Extract<Subgoal, { kind: "drive" }>;

const HOVER = 0.03; // jaw tips 3 cm above the object top
const CARRY = 0.015; // a carried object 1.5 cm above its target: towers get close to the arm's reach
const LIFT = 0.08;
// The base: up to 30 cm/s on straight parts, gentler with something in the jaws. The path planner counts on these.
const EMPTY = { speed: 0.3, accel: 0.6 };
const LOADED = { speed: 0.15, accel: 0.2 };
const TURN_RATE = 1.4; // rad/s turning on the spot
// Arm moves take this share of the time they took at first: measured, 3 times faster keeps every pick.
const PACE = 0.35;
const play = (robot: Robot, path: number[][], seconds: number, settle?: number, until?: () => boolean) => robot.play(path, PACE * seconds, settle, until);

export const cm = (m: number) => Math.round(m * 1000) / 10;
export const deg = (rad: number) => Math.round((rad * 180) / Math.PI);

// ---- arm moves ----
// Each acts on robot.focus, the object of the current step, and returns what happened in words.

export async function openJaws({ robot, stage }: Ctx) {
  stage("physics");
  const held = robot.contacts();
  await play(robot, [[...robot.target.slice(0, GRIPPER), OPEN]], 0.5);
  return held.fixed && held.moving ? "Jaws open. The object was let go." : "Jaws open.";
}

// Hover just above the object, fingers down, jaws square to a face that fits between them, through a point
// high above it so the arm never sweeps through it. `turned`: the jaws a quarter turn from the best angle.
// `open`: the jaws open on the way.
export async function moveAbove({ robot, stage }: Ctx, turned = false, open = false) {
  stage("ik");
  const object = robot.object();
  let pose = robot.graspYaw(object.top + HOVER);
  if (turned) {
    const yaw = pose.yaw + Math.PI / 2;
    const pos = robot.gripPoint(yaw, object.top + HOVER);
    pose = { ...pose, ...robot.solve(pos, yaw, pose.q), yaw, pos };
  }
  if (pose.miss > 0.003) pose = { ...pose, ...robot.solve(pose.pos, pose.yaw, pose.q, 0.05) };
  const via = robot.solve([pose.pos[0], pose.pos[1], Math.max(0.16, object.top + 0.08)], pose.yaw, pose.q, 0.05);
  const grip = open ? OPEN : robot.target[GRIPPER];
  // Open jaws still around an object it just set down rise straight up first, so they do not drag it along.
  const tip = robot.tcp();
  const around = robot.active().filter((o) => o !== robot.focus && !clear(robot, o));
  const rise = around.length ? line(robot, tip, [tip[0], tip[1], Math.max(...around.map((o) => robot.object(o).top)) + 0.04], 0.3, robot.handYaw()) : [];
  stage("physics");
  await play(robot, [...rise, [...via.q, grip], [...pose.q, grip]], rise.length ? 2.4 : 1.8);
  return reached(robot, pose.pos, "Above the object");
}

// Lower straight down so the object ends up between the jaws: just under its middle to take it, or 1 cm above
// the bottom of a box that goes on something (at most 3.5 cm under its top), so the arm reaches less high.
export async function lowerToObject({ robot, stage }: Ctx, placing = false) {
  stage("ik");
  const object = robot.object();
  const yaw = squareYaw(robot);
  const z = placing && robot.focus!.kind === "box" ? Math.max(object.bottom + 0.01, object.top - 0.035) : Math.max(object.bottom + 0.01, object.pos[2] - 0.005);
  const goal = robot.gripPoint(yaw, z);
  const path = line(robot, robot.tcp(), goal, 0.3);
  stage("physics");
  await play(robot, path, 1.0);
  return reached(robot, goal, "Lowered around the object");
}

// Gentler than the other moves too: jaws that close in under 0.6 s drop a 6 cm wide box 1 time in 8, measured.
export async function closeJaws({ robot, stage }: Ctx) {
  stage("physics");
  await robot.play([[...robot.target.slice(0, GRIPPER), CLOSED]], 0.6, 0.4);
  const { fixed, moving } = robot.contacts();
  return fixed && moving ? "Jaws closed on the object." : "Jaws closed on nothing.";
}

// A gentler pace than the other moves: a wide, flat box held by the jaw tips slips when jerked up.
export async function lift({ robot, stage }: Ctx) {
  stage("ik");
  const tip = robot.tcp();
  const path = line(robot, tip, [tip[0], tip[1], tip[2] + LIFT], 0.05);
  stage("physics");
  await robot.play(path, 0.6);
  return `Raised ${cm(robot.tcp()[2] - tip[2])} cm.`;
}

// Carry the held object up, over to the target, and hold it just above it: around the arm's base, not across
// it, high enough to clear everything, turned to land square on a box below. `turned`: a quarter turn more.
export async function carryAbove({ robot, stage }: Ctx, target: Target, turned = false) {
  stage("ik");
  const o = robot.object();
  const tip = robot.tcp();
  const yaw0 = robot.handYaw();
  const pan = wrap(Math.atan2(target.y, target.x) - Math.atan2(tip[1], tip[0]));
  const turn = landingTurn(robot.focus!, o.yaw, target, pan) + (turned ? Math.PI / 2 : 0);
  const d = [o.pos[0] - tip[0], o.pos[1] - tip[1], o.pos[2] - tip[2]]; // the object's centre from the jaw tips
  const [c, s] = [Math.cos(turn), Math.sin(turn)];
  const below = o.pos[2] - o.bottom; // its centre above its lowest point
  const goal: Vec3 = [target.x - (c * d[0] - s * d[1]), target.y - (s * d[0] + c * d[1]), target.top + CARRY + below - d[2]];
  const tallest = Math.max(0, ...robot.active().filter((x) => x !== robot.focus).map((x) => robot.object(x).top));
  // Travel high enough that the load clears everything: its lowest point, the object or the jaw tips under a ball.
  const safe = Math.max(tip[2], goal[2], tip[2] + tallest + 0.02 - Math.min(o.bottom, tip[2]));
  const up = safe - tip[2] > 0.005 ? line(robot, tip, [tip[0], tip[1], safe], TILT, yaw0) : [];
  const over = line(robot, [tip[0], tip[1], safe], [goal[0], goal[1], safe], TILT, yaw0, yaw0 + turn, up.at(-1), true);
  const down = safe - goal[2] > 0.005 ? line(robot, [goal[0], goal[1], safe], goal, TILT, yaw0 + turn, yaw0 + turn, over.at(-1)) : [];
  stage("physics");
  await play(robot, [...up, ...over, ...down], 1.2 + (0.4 * (up.length + down.length)) / 6 + Math.min(1.2, 8 * Math.hypot(goal[0] - tip[0], goal[1] - tip[1])));
  const a = robot.object();
  const grip = robot.contacts();
  if (!grip.fixed || !grip.moving) return `Dropped ${robot.focus!.label} on the way.`;
  return `Carried over ${target.label}: ${cm(Math.hypot(a.pos[0] - target.x, a.pos[1] - target.y))} cm off centre, ${cm(a.bottom - target.top)} cm above it.`;
}

// Lower the held object straight down onto the target, aiming 5 mm below contact, and stop at the first touch.
export async function lowerToPlace({ robot, stage }: Ctx, target: Target) {
  stage("ik");
  const o = robot.object();
  const tip = robot.tcp();
  const yaw = robot.handYaw();
  const goal: Vec3 = [tip[0] + target.x - o.pos[0], tip[1] + target.y - o.pos[1], tip[2] - (o.bottom - target.top) - 0.005];
  const path = line(robot, tip, goal, TILT, yaw);
  const focus = robot.focus!;
  // The first touch: the object on the target, or the jaws on it (a small ball sits above the jaw tips).
  const landed = () => {
    const t = robot.touching(focus);
    return target.object ? t.others.has(target.object) || robot.touching(target.object).arm : t.table;
  };
  stage("physics");
  await play(robot, path, 1.2, 0.3, landed);
  const v = versus(robot, focus, target);
  return v.sitting ? `Set down on ${target.label}, ${cm(v.off)} cm off centre.` : `Lowered, ${cm(v.gap)} cm above ${target.label}, not on it.`;
}

export async function releaseJaws({ robot, stage }: Ctx, target: Target) {
  stage("physics");
  await play(robot, [[...robot.target.slice(0, GRIPPER), OPEN]], 0.6, 0.4);
  const o = robot.focus!;
  const on = target.object ? restingOn(robot, o, target.object) : robot.touching(o).table;
  return on ? `Let go. ${o.label} sits on ${target.label}.` : `Let go. ${o.label} is not on ${target.label}.`;
}

// Move the open jaws away from what they just let go of: 6 mm off the fixed jaw it still leans on, then out
// sideways along the gap between the jaws, toward the arm's base, then up a little. A ball was dropped from
// above with the jaw tips on the target: the jaws rise straight up instead, so they drag nothing sideways.
export async function retreat({ robot, stage }: Ctx) {
  stage("ik");
  const tip = robot.tcp();
  const o = robot.object();
  const yaw = robot.handYaw();
  if (robot.focus!.kind === "ball") {
    const up = line(robot, tip, [tip[0], tip[1], Math.max(tip[2], o.top) + 0.03], TILT, yaw);
    stage("physics");
    await play(robot, up, 1.0);
    return clear(robot, robot.focus!) ? "Rose straight up, jaws clear." : "Rose, but the jaws are still around the ball.";
  }
  const off: Vec3 = [tip[0] - 0.006 * Math.cos(yaw), tip[1] - 0.006 * Math.sin(yaw), tip[2]];
  let side: Vec3 = [-Math.sin(yaw), Math.cos(yaw), 0];
  if (side[0] * o.pos[0] + side[1] * o.pos[1] > 0) side = [-side[0], -side[1], 0];
  const out = robot.footprint(robot.focus!) + 0.025;
  const away: Vec3 = [off[0] + side[0] * out, off[1] + side[1] * out, tip[2] + 0.01];
  const back = line(robot, tip, off, TILT, yaw);
  const slide = line(robot, off, away, TILT, yaw, yaw, back.at(-1));
  const rise = line(robot, away, [away[0], away[1], Math.max(away[2] + 0.02, o.top + 0.02)], TILT, yaw, yaw, slide.at(-1));
  stage("physics");
  // Slower than the other moves: sliding out fast, a jaw catches the box it just set down and tips the stack.
  await robot.play([...back, ...slide, ...rise], 0.9);
  return clear(robot, robot.focus!) ? "Moved out, jaws clear." : "Moved out, but the jaws are still around the object.";
}

// Raise the jaws 5 cm straight up, with or without what they hold.
async function raise({ robot, stage }: Ctx) {
  stage("ik");
  const tip = robot.tcp();
  const path = line(robot, tip, [tip[0], tip[1], tip[2] + 0.05], TILT, robot.handYaw());
  stage("physics");
  await play(robot, path, 0.8);
}

// ---- the base ----

// Before the base moves: an object in the jaws is raised clear of everything on the floor, its target
// included, else the arm folds to rest with the jaws open, rising first out of anything it is near, so the
// robot drags nothing along and sweeps nothing over.
export async function stow({ robot, stage }: Ctx) {
  const tip = robot.tcp();
  const held = robot.active().find(robot.gripped);
  const near = robot.active().filter((o) => Math.hypot(robot.object(o).pos[0] - tip[0], robot.object(o).pos[1] - tip[1]) < 0.35);
  const tallest = Math.max(0, ...(held ? robot.active() : near).filter((o) => o !== held).map((o) => robot.object(o).top));
  stage("ik");
  if (held) {
    // The load's lowest point: the object, or the jaw tips under a small ball.
    const up = Math.max(0.05, tallest + 0.02) - Math.min(robot.object(held).bottom, tip[2]);
    if (up < 0.005) return;
    const path = line(robot, tip, [tip[0], tip[1], tip[2] + up], TILT, robot.handYaw());
    stage("physics");
    await play(robot, path, 1.0);
    return;
  }
  const q = robot.joints();
  const folded = REST.slice(0, GRIPPER).every((v, i) => Math.abs(q[i] - v) < 0.05);
  if (folded && q[GRIPPER] > 0.5) return;
  const rise = !folded && tip[2] < tallest + 0.04 ? line(robot, tip, [tip[0], tip[1], tallest + 0.04], TILT, robot.handYaw()) : [];
  stage("physics");
  await play(robot, [...rise, [...REST.slice(0, GRIPPER), OPEN]], folded ? 0.5 : rise.length ? 2.2 : 1.5);
}

// The drive pad: one step of 10 cm or 15°.
export const PAD: Record<string, (robot: Robot) => Promise<string>> = Object.fromEntries(
  (
    [
      ["drive_forward", false, 1],
      ["drive_backward", false, -1],
      ["turn_left", true, 1],
      ["turn_right", true, -1],
    ] as const
  ).map(([name, turn, sign]) => [
    name,
    async (robot: Robot) => {
      const ctx = { robot, stage: () => {} } as unknown as Ctx;
      await stow(ctx);
      const r = await robot.move(turn ? "turn" : "drive", sign * (turn ? TURN : STEP));
      const text = turn ? `Turned ${Math.abs(deg(r.turned))}° ${r.turned > 0 ? "left" : "right"}.` : `Drove ${Math.abs(cm(r.moved))} cm ${r.moved > 0 ? "forward" : "back"}.`;
      return r.bumped ? `${text} Stopped early: the base touched ${r.bumped.label}.` : text;
    },
  ]),
);

// GoTo: drive the base where a step needs it, around everything. A move asked for in the goal, or up to what is
// out of the arm's reach. It plans a path on a map of the floor (nav.ts) and follows it. At the first touch the
// base stops, the map is made again from where things are now, and it plans again. After two touches it keeps
// a wider berth, after three it stops and says why. With no way at all, a plain drive goes as far as it can.
// `avoid`: a heading to come from another side than. `backOff`: first back away 10 cm from what is nearest.
export async function goTo(ctx: Ctx, sub: DriveSub | Approach, opts: { avoid?: number; backOff?: boolean; preshape?: Obj } = {}): Promise<Outcome> {
  const { robot, world, stage, signal } = ctx;
  await stow(ctx);
  const what = sub.kind === "drive" ? sub.text.toLowerCase() : `get ${sub.label} in reach`;
  const held = world.objects().find((o) => o.held);
  const where = () => robot.toWorld([...(sub.kind === "approach" ? sub.where() : [0, 0]), 0]);
  // Carrying something to put down, the chassis keeps clear of the target but the load passes over it.
  const soft = () => (sub.kind === "approach" && held ? world.objects().filter((o) => !o.held && Math.hypot(o.world[0] - where()[0], o.world[1] - where()[1]) < 0.03).map((o) => o.obj) : []);
  const goals = (map: FloorMap, yaw: number) => {
    if (sub.kind === "drive") return goalsForMove(map, sub.drive.from, sub.drive.turn, sub.drive.amount);
    const [x, y] = where();
    const all = goalsAround(map, x, y, sub.distance, yaw, [sub.object, ...soft()]);
    return opts.avoid === undefined ? all : all.filter((g) => Math.abs(wrap(g.yaw - opts.avoid!)) > 0.7);
  };
  const how = held ? LOADED : EMPTY;
  if (opts.backOff) await backOff(ctx);
  let margin = MARGIN;
  const touched: Obj[] = [];
  let slips = 0;
  for (;;) {
    if (signal.aborted) return { ok: false, text: "Stopped." };
    const map = floorMap(world, soft(), margin);
    const start = pivot(world.base());
    stage("ik");
    const path = planPath(map, start, goals(map, start.yaw), { drive: how.speed, turn: TURN_RATE });
    if (!path) {
      if (margin > 0.012) {
        margin = 0.012; // a tighter squeeze before giving up
        continue;
      }
      const blocker = nearest(map, start);
      if (sub.kind === "drive" && !sub.drive.turn) return { ok: false, text: await edgeForward(ctx, map, sub, blocker?.obj ?? null) };
      return { ok: false, text: `No free way to ${what}${blocker ? `: ${blocker.obj.label} and the others leave no room` : ""}.` };
    }
    ctx.path(path.points);
    stage("drive");
    const near = opts.preshape && preshape(ctx, opts.preshape, armOf(path.goal));
    // Close enough already: the object well in reach, or the move asked for done. Then no last corrections.
    const enough = () => {
      if (sub.kind === "drive") return driven(robot, sub.drive);
      const { r, a } = robot.polar(sub.where());
      return Math.abs(r - sub.distance) < 0.03 && Math.abs(a) < 0.35;
    };
    const r = await follow(robot, path, { ...how, signal, near: near ? { within: 0.15, run: near } : undefined, enough }, map);
    if (r.ok) return { ok: true, text: `${touched.length ? `Planned again after touching ${labels(touched)}. ` : ""}Drove ${cm(r.moved)} cm on a path of ${path.segments.length} ${path.segments.length === 1 ? "move" : "moves"}.` };
    if (signal.aborted) return { ok: false, text: "Stopped." };
    if (!r.bumped) {
      // Slipped off the path, or no progress for 2 s: plan again from here, once.
      if (++slips > 1) return { ok: false, text: `Stopped: ${r.text ?? "the base did not get there"}.` };
      continue;
    }
    touched.push(r.bumped);
    if (touched.length >= 3) return { ok: false, text: `Stopped: the base touched ${labels(touched)} on the way, three times. It does not push, so it gives up.` };
    if (touched.length === 2) margin = 0.035;
  }
}
const labels = (objs: Obj[]) => [...new Set(objs.map((o) => o.label))].join(" and ");

// While the base drives its last 15 cm up to an object to take, the arm rises to a point high above where the
// object will be, jaws open. Only when nothing near the object stands that high.
function preshape(ctx: Ctx, o: Obj, goal: { x: number; y: number; yaw: number }) {
  const { robot, world } = ctx;
  const f = world.see(o);
  const [c, s] = [Math.cos(goal.yaw), Math.sin(goal.yaw)];
  const dx = f.world[0] - goal.x;
  const dy = f.world[1] - goal.y;
  const [x, y] = [c * dx + s * dy, -s * dx + c * dy];
  const z = Math.max(0.16, f.top + 0.08);
  const nearby = world.objects().filter((n) => n.obj !== o && Math.hypot(n.world[0] - f.world[0], n.world[1] - f.world[1]) < 0.15);
  if (nearby.some((n) => n.top > z - 0.03)) return null;
  const face = o.kind === "box" ? wrap(f.yaw + world.base().yaw - goal.yaw) : Math.atan2(y, x);
  const q = Math.PI / 2;
  const yaw = face + q * Math.round((Math.atan2(y, x) - face) / q);
  return () => {
    const pose = robot.solve([x, y, z], yaw, [Math.atan2(y, x), 0, 0, 1.2, 0], 0.05);
    play(robot, [[...pose.q, OPEN]], 1.0, 0);
  };
}

// Back away 10 cm from the nearest object, straight back or forward, whichever keeps clear.
async function backOff(ctx: Ctx) {
  const { robot, world } = ctx;
  const map = floorMap(world, [], 0.01);
  const c = pivot(world.base());
  const at = (d: number) => clearance(map, { x: c.x + d * Math.cos(c.yaw), y: c.y + d * Math.sin(c.yaw), yaw: c.yaw });
  const d = at(-0.1) >= at(0.1) ? -0.1 : 0.1;
  if (at(d) < 0.005) return;
  ctx.stage("drive");
  const r = await robot.move("drive", d);
  ctx.say("back_off", `Backed ${cm(Math.abs(r.moved))} cm ${d < 0 ? "back" : "forward"} to start again from there.`);
}

// No way around to a plain drive's end: go straight as far as stays clear, and stop there.
async function edgeForward(ctx: Ctx, map: FloorMap, sub: DriveSub, blocker: Obj | null) {
  const { robot } = ctx;
  const sign = Math.sign(toGo(robot, sub.drive));
  const c = pivot(robot.base());
  const free = (d: number) => Math.min(...[0.5, 1].map((k) => clearance(map, { x: c.x + k * d * Math.cos(c.yaw), y: c.y + k * d * Math.sin(c.yaw), yaw: c.yaw })));
  let d = 0;
  while (d + 0.01 <= Math.abs(toGo(robot, sub.drive)) && free(sign * (d + 0.01)) >= MARGIN) d += 0.01;
  ctx.stage("drive");
  if (d > 0.005) await robot.move("drive", sign * d);
  return `Drove ${cm(d)} cm and stopped before ${blocker?.label ?? "what is in the way"}: there is no way to ${sub.text.toLowerCase().replace(/\.$/, "")} without touching it.`;
}

// ---- Pick and Place ----

// Pick: open, hover, lower around it, close, check both jaws hold it, lift. Variants: 0 the best jaw angle, 1 the
// jaws turned 90°, 2 from another side of the object first.
export async function pick(ctx: Ctx, o: Obj, variant: number, approach?: Approach): Promise<Outcome> {
  const { robot } = ctx;
  robot.focus = o;
  if (variant >= 2 && approach) {
    const moved = await goTo(ctx, approach, { avoid: ctx.world.base().yaw, preshape: o });
    ctx.say("drive_to_object", moved.text);
    if (!moved.ok) return moved;
    robot.focus = o;
  }
  // The jaws open on the way above it, unless they hold something to let go of first.
  const steps: [string, () => Promise<string>][] = [
    ["open_gripper", () => openJaws(ctx)],
    ["move_above_object", () => moveAbove(ctx, variant === 1, true)],
    ["lower_to_object", () => lowerToObject(ctx, false)],
    ["close_gripper", () => closeJaws(ctx)],
  ];
  const holding = robot.active().some(robot.gripped);
  for (const [name, run] of steps) {
    if (name === "open_gripper" && !holding) continue;
    if (ctx.signal.aborted) return { ok: false, text: "Stopped." };
    const text = await run();
    ctx.say(name, text);
    if (short(text)) return retry(ctx, `The arm cannot get there: ${text}`);
  }
  // Both jaws on it, and the gripper not shut on nothing.
  const t = robot.contacts();
  if (!(t.fixed && t.moving) || robot.joints()[GRIPPER] < CLOSED + 0.03) return retry(ctx, `The jaws missed ${o.label}.`);
  ctx.say("lift", await lift(ctx));
  const a = robot.object(o);
  if (!robot.gripped(o) || a.bottom < LIFTED - 0.01) return retry(ctx, `${o.label} slipped out of the jaws.`);
  return { ok: true, text: `Took ${o.label}.` };
}
// A move that ended over 2.5 cm from where it aimed: the arm cannot get there from here.
const short = (text: string) => /Stopped ([\d.]+) cm short/.test(text) && parseFloat(/Stopped ([\d.]+) cm short/.exec(text)![1]) > 2.5;
// A failed pick lets go and raises the arm, so the next try starts clean.
async function retry(ctx: Ctx, why: string): Promise<Outcome> {
  await openJaws(ctx);
  await raise(ctx);
  return { ok: false, text: why };
}

// Place: take the object if it is not in the jaws, carry it over the target, lower it until it touches, let go
// once it sits there, and move the jaws away. Variants: 0 as planned, 1 landing a quarter turn around, 2 the
// pick from another side.
export async function place(ctx: Ctx, sub: Place, variant: number, approach?: Approach): Promise<Outcome> {
  const { robot } = ctx;
  const o = sub.object;
  robot.focus = o;
  // The target where it is now, from wherever the base ends up.
  const [wx, wy] = robot.toWorld([sub.target.x, sub.target.y, 0]);
  const fresh = (): Target => {
    if (sub.target.into) return intoTarget(robot, sub.target.object!);
    if (sub.target.object) return onTop(robot, sub.target.object);
    const [x, y] = robot.toArm([wx, wy, 0]);
    return { ...sub.target, x, y };
  };
  if (!robot.gripped(o)) {
    // Gripped low, a box sets down on a high tower with less reach; gripped at its middle, it rides a drive steady.
    const took = await pickToPlace(ctx, o, variant, approach, !sub.target.into && inReach(robot, [fresh().x, fresh().y]));
    if (!took.ok) return took;
  } else if (variant >= 2 && approach) {
    const moved = await goTo(ctx, approach, { avoid: ctx.world.base().yaw });
    ctx.say("drive_to_object", moved.text);
    if (!moved.ok) return moved;
    robot.focus = o;
  }
  const target = fresh();
  if (!inReach(robot, [target.x, target.y])) return { ok: false, text: `Took ${o.label}. ${target.label} is out of reach from here.` };
  if (target.into) return drop(ctx, o, target.object!);
  for (const [name, run] of [
    ["move_above_target", () => carryAbove(ctx, target, variant === 1)],
    ["lower_to_place", () => lowerToPlace(ctx, target)],
  ] as const) {
    if (ctx.signal.aborted) return { ok: false, text: "Stopped." };
    const text = await run();
    ctx.say(name, text);
    if (/Dropped/.test(text)) return { ok: false, text };
  }
  const v = versus(robot, o, target);
  if (!v.sitting && !v.close) {
    await raise(ctx);
    return { ok: false, text: `${o.label} did not land on ${target.label}: ${cm(v.gap)} cm above it, ${cm(v.off)} cm off centre.` };
  }
  ctx.say("release", await releaseJaws(ctx, target));
  ctx.say("retreat", await retreat(ctx));
  return { ok: true, text: `Set ${o.label} on ${target.label}.` };
}
// Into the container: a free spot inside that the arm reaches, the object carried over the walls, lowered to just
// above what is under it, let go, and the jaws raised straight up.
async function drop(ctx: Ctx, o: Obj, container: Obj): Promise<Outcome> {
  const { robot } = ctx;
  const spot = dropSpot(robot, o, container);
  if (!spot) return { ok: false, text: "No spot inside the container is in the arm's reach from here." };
  const target: Target = { label: "the container", x: spot[0], y: spot[1], top: spot[2] + 0.01, yaw: null, object: container, into: true };
  for (const [name, run] of [
    ["move_above_target", () => carryAbove(ctx, target)],
    ["lower_to_place", () => lowerToPlace(ctx, target)],
  ] as const) {
    if (ctx.signal.aborted) return { ok: false, text: "Stopped." };
    const text = await run();
    ctx.say(name, text.replace(/Set down on the container.*|Lowered, .*/, "Lowered into the container."));
    if (/Dropped/.test(text)) return { ok: false, text };
  }
  ctx.stage("physics");
  await play(robot, [[...robot.target.slice(0, GRIPPER), OPEN]], 0.6, 0.4);
  ctx.say("release", robot.inside(o, container) ? `Let go. ${o.label} is in the container.` : `Let go. ${o.label} is not in the container.`);
  await raise(ctx);
  ctx.say("retreat", "Rose straight up.");
  return robot.inside(o, container) ? { ok: true, text: `Put ${o.label} in the container.` } : { ok: false, text: `${o.label} did not land in the container.` };
}

// A spot inside the container for `o`, in the arm's frame, with the height of what is under it: at least 2 cm
// from the walls, in the arm's reach, clear of what is already inside, nearest where the arm reaches best. When
// nothing inside is clear, on top of what is lowest.
function dropSpot(robot: Robot, o: Obj, container: Obj): Vec3 | null {
  const c = robot.object(container);
  const [cs, sn] = [Math.cos(c.yaw), Math.sin(c.yaw)];
  const half = container.size[0] - WALL - robot.footprint(o) - 0.02;
  const inside = robot.active().filter((x) => x !== o && x !== container && robot.inside(x, container));
  let best: { p: Vec3; score: number } | null = null;
  for (let u = -half; u <= half + 1e-9; u += 0.01)
    for (let v = -half; v <= half + 1e-9; v += 0.01) {
      const x = c.pos[0] + cs * u - sn * v;
      const y = c.pos[1] + sn * u + cs * v;
      if (!inReach(robot, [x, y])) continue;
      const under = inside.filter((i) => Math.hypot(robot.object(i).pos[0] - x, robot.object(i).pos[1] - y) < robot.footprint(i) + robot.footprint(o) + 0.005);
      const top = Math.max(c.bottom + 0.004, ...under.map((i) => robot.object(i).top));
      const score = Math.hypot(x - 0.2, y) + (under.length ? 1 + top : 0);
      if (!best || score < best.score) best = { p: [x, y, top], score };
    }
  return best?.p ?? null;
}

// The pick inside a place: grip a box low when it goes on something in reach, so the arm reaches less high.
async function pickToPlace(ctx: Ctx, o: Obj, variant: number, approach?: Approach, low = true) {
  const { robot } = ctx;
  if (variant >= 2 && approach) {
    const moved = await goTo(ctx, approach, { avoid: ctx.world.base().yaw, preshape: o });
    ctx.say("drive_to_object", moved.text);
    if (!moved.ok) return moved;
    robot.focus = o;
  }
  const steps: [string, () => Promise<string>][] = [
    ["open_gripper", () => openJaws(ctx)],
    ["move_above_object", () => moveAbove(ctx, variant === 1 && o.kind === "box", true)],
    ["lower_to_object", () => lowerToObject(ctx, low)],
    ["close_gripper", () => closeJaws(ctx)],
  ];
  const holding = robot.active().some(robot.gripped);
  for (const [name, run] of steps) {
    if (name === "open_gripper" && !holding) continue;
    if (ctx.signal.aborted) return { ok: false, text: "Stopped." };
    const text = await run();
    ctx.say(name, text);
    if (short(text)) return retry(ctx, `The arm cannot get there: ${text}`);
  }
  const t = robot.contacts();
  if (!(t.fixed && t.moving) || robot.joints()[GRIPPER] < CLOSED + 0.03) return retry(ctx, `The jaws missed ${o.label}.`);
  return { ok: true, text: `Took ${o.label}.` };
}

// ---- geometry helpers ----

// The object against its target: sitting on it, held above it, or away from it. A held object counts as
// sitting when it is within 3 mm of the target: the jaws hold a ball a hair above it, and letting go drops it there.
export function versus(robot: Robot, o: Obj, target: Target) {
  const a = robot.object(o);
  const touch = robot.touching(o);
  const off = Math.hypot(a.pos[0] - target.x, a.pos[1] - target.y);
  const gap = a.bottom - target.top;
  const t = target.object;
  const reach = !t ? 0.015 : t.kind === "ball" ? 0.006 : Math.max(0.006, 0.7 * Math.min(t.size[0], t.size[1]));
  const held = touch.fixed && touch.moving;
  const down = held ? gap > -0.006 && gap < 0.003 : Math.abs(gap) < 0.006 && (t ? touch.others.has(t) : touch.table);
  const sitting = a.upright && off < reach && down;
  // Close enough to let go: centred, and within 6 mm of the target's top, even a little tilted in the jaws. A
  // ball sits above the jaw tips, so they touch the target first: it drops the last few cm.
  const close = off < reach && gap > -0.006 && gap < (o.kind === "ball" ? 0.045 : 0.006);
  return { sitting, close, above: !sitting && off < reach && gap > 0, off, gap };
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
