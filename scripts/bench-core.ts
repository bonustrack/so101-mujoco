// The benchmark behind `bun run bench` and the quick subset in `bun run check-agent`: random scenes in fixed
// families, one task each, run headless with the page's own code. A run records its outcome, what else it
// moved or knocked over, what the base touched, the chassis tilt, the time and the Jev calls.
//
// Two deciders answer the Jev questions:
// - rules: a perfect decider. It reads the goal right and follows the command criteria to the letter, so it
//   shows the limits of the code itself.
// - jev: the real Jev, through the page's own Netlify Function.
//
// "Clean" means success, and nothing else moved more than 2 cm, knocked over or rolled under the base, and
// the base pushed nothing more than 2 cm. What the task moves on purpose does not count.
import { runAgent, type AgentEvent, type Decide, type JevRequest } from "../src/agent";
import type { Obj, Robot, Vec3 } from "../src/robot";

export type Mode = "rules" | "jev";
// Jev's measured latency through the live Netlify Function, added per call to the rules decider's time,
// so both modes estimate what the page shows.
export const RULES_LATENCY = 0.2;

type Spec = { kind: "box" | "ball"; x: number; y: number; yaw: number; size: Vec3 };
type Truth = { task: string; picks: Record<string, string> };
// A scene, a goal, the true reading, and which objects the task may move.
type Case = { family: string; goal: string; truth: Truth; objs: Obj[]; target?: Obj; selected?: Obj | null; moves: Obj[] };
type Ev = { t: number; kind: string; detail: string };
export type Record = ReturnType<typeof finish>;

export function createBench(robot: Robot) {
  const { mujoco, model, data } = robot;
  const bodyId = (name: string) => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, name);
  const chassis = bodyId("chassis");
  const baseBodies = new Set([chassis, ...["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"].map(bodyId)]);
  const objectGeoms = new Map(robot.objects.map((o) => [o.geom, o]));
  const floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, "floor");
  const geomKind = (g: number) => (objectGeoms.has(g) ? "object" : g === floor ? "floor" : baseBodies.has(model.geom_bodyid[g]) ? "base" : "arm");
  const chassisDof = model.jnt_dofadr[model.body_jntadr[chassis]] as number;

  // Seeded random, reset per run, so a run is the same whatever ran before it.
  let seed = 1;
  const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const uni = (a: number, b: number) => a + (b - a) * random();
  const pick = <T>(xs: T[]) => xs[Math.floor(random() * xs.length)];

  const boxSize = (): Vec3 => [uni(0.012, 0.026), uni(0.012, 0.026), uni(0.012, 0.025)];
  const ballSize = (): Vec3 => {
    const r = uni(0.015, 0.03);
    return [r, r, r];
  };
  const foot = (s: Spec) => (s.kind === "ball" ? s.size[0] : Math.hypot(s.size[0], s.size[1]));
  // Clear of the robot's outline (at the start the arm's frame is the world's) and of the others.
  const fits = (s: Spec, others: Spec[]) => {
    const m = foot(s) + 0.03;
    const inside = s.x < 0.08 + m && s.x > -0.13 - m && Math.abs(s.y) < 0.095 + m;
    return !inside && others.every((o) => Math.hypot(o.x - s.x, o.y - s.y) > foot(o) + foot(s) + 0.035);
  };
  function spot(kind: "box" | "ball", others: Spec[], r: [number, number], a: [number, number]): Spec {
    for (let i = 0; i < 200; i++) {
      const rr = uni(...r);
      const aa = uni(...a);
      const s: Spec = { kind, x: rr * Math.cos(aa), y: rr * Math.sin(aa), yaw: uni(-1.5, 1.5), size: kind === "box" ? boxSize() : ballSize() };
      if (fits(s, others)) return s;
    }
    throw new Error("no spot");
  }
  const NEAR: [[number, number], [number, number]] = [[0.16, 0.25], [-0.9, 0.9]];
  const FAR: [[number, number], [number, number]] = [[0.35, 0.9], [-Math.PI, Math.PI]];

  function build(specs: Spec[]) {
    robot.resetScene();
    robot.reset();
    robot.remove(robot.active()[0]);
    const objs = specs.map((s) => {
      const o = robot.add(s.kind)!;
      robot.resize(o, s.size);
      robot.place(o, s.x, s.y, s.yaw);
      return o;
    });
    robot.reset();
    for (let i = 0; i < 100; i++) robot.step();
    return objs;
  }

  const FAMILY: { [name: string]: () => Case } = {
    // Take the only ball among boxes, all in reach: the object has to come from the goal.
    near_take_ball() {
      const specs: Spec[] = [];
      for (const k of ["box", "box", "ball"] as const) specs.push(spot(k, specs, ...NEAR));
      const objs = build(specs);
      const ball = objs[2];
      return { family: "near_take_ball", goal: pick(["take the ball", "pick up the ball", "grab the ball"]), truth: { task: "take", picks: { object: ball.label } }, objs, target: ball, moves: [ball] };
    },
    // The same, with a box selected in the scene, as after Add box or a drag on the page. Must pass.
    near_take_ball_selected() {
      const c = FAMILY.near_take_ball();
      return { ...c, family: "near_take_ball_selected", selected: c.objs[0] };
    },
    near_take_named() {
      const specs: Spec[] = [];
      for (const k of ["box", "box", "ball"] as const) specs.push(spot(k, specs, ...NEAR));
      const objs = build(specs);
      const t = pick(objs);
      return { family: "near_take_named", goal: `pick up ${t.label}`, truth: { task: "take", picks: { object: t.label } }, objs, target: t, moves: [t] };
    },
    near_stack() {
      const specs: Spec[] = [];
      for (const k of ["box", "box", "box"] as const) specs.push(spot(k, specs, ...NEAR));
      const objs = build(specs);
      return { family: "near_stack", goal: "stack all the boxes", truth: { task: "stack_boxes", picks: {} }, objs, moves: objs };
    },
    near_put() {
      const specs: Spec[] = [];
      for (const k of ["box", "box", "ball"] as const) specs.push(spot(k, specs, ...NEAR));
      const objs = build(specs);
      const [x, y] = random() < 0.5 ? [objs[2], objs[0]] : [objs[1], objs[0]];
      return { family: "near_put", goal: `put ${x.label} on ${y.label}`, truth: { task: "put_on", picks: { object: x.label, onto: y.label } }, objs, target: x, moves: [x] };
    },
    // A far object anywhere around, with others spread on the floor.
    far_take() {
      const specs: Spec[] = [];
      const n = 2 + Math.floor(random() * 3);
      for (let i = 0; i < n; i++) specs.push(spot(random() < 0.6 ? "box" : "ball", specs, ...FAR));
      const t = specs.length - 1;
      const objs = build(specs);
      return { family: "far_take", goal: `drive to ${objs[t].label} and pick it up`, truth: { task: "take", picks: { object: objs[t].label } }, objs, target: objs[t], moves: [objs[t]] };
    },
    // A far object with a box right on the straight line to it. Must pass.
    blocked_take() {
      const b = uni(-0.7, 0.7);
      const r = uni(0.55, 0.8);
      const target: Spec = { kind: random() < 0.5 ? "box" : "ball", x: r * Math.cos(b), y: r * Math.sin(b), yaw: uni(-1, 1), size: [0, 0, 0] };
      target.size = target.kind === "box" ? boxSize() : ballSize();
      const br = uni(0.28, 0.42);
      const off = uni(-0.03, 0.03);
      const blocker: Spec = { kind: "box", x: br * Math.cos(b) - off * Math.sin(b), y: br * Math.sin(b) + off * Math.cos(b), yaw: uni(-1, 1), size: boxSize() };
      const specs = [blocker, target];
      if (random() < 0.5) specs.push(spot("ball", specs, ...FAR));
      const objs = build(specs);
      return { family: "blocked_take", goal: `take ${objs[1].label}`, truth: { task: "take", picks: { object: objs[1].label } }, objs, target: objs[1], moves: [objs[1]] };
    },
    far_put() {
      const specs: Spec[] = [spot("box", [], ...NEAR)];
      specs.push(spot(random() < 0.5 ? "box" : "ball", specs, [0.4, 0.8], [-Math.PI, Math.PI]));
      if (random() < 0.7) specs.push(spot("box", specs, ...FAR));
      const objs = build(specs);
      return { family: "far_put", goal: `put ${objs[1].label} on ${objs[0].label}`, truth: { task: "put_on", picks: { object: objs[1].label, onto: objs[0].label } }, objs, target: objs[1], moves: [objs[1]] };
    },
    spread_stack() {
      const specs: Spec[] = [];
      for (let i = 0; i < 3; i++) specs.push(spot("box", specs, [0.25, 0.7], [-Math.PI, Math.PI]));
      const objs = build(specs);
      return { family: "spread_stack", goal: "stack all the boxes", truth: { task: "stack_boxes", picks: {} }, objs, moves: objs };
    },
    // Turn around with objects close around the robot. Must pass.
    turn_clutter() {
      const specs: Spec[] = [];
      for (let i = 0; i < 3; i++) specs.push(spot(random() < 0.5 ? "box" : "ball", specs, [0.17, 0.3], [-Math.PI, Math.PI]));
      const objs = build(specs);
      return { family: "turn_clutter", goal: "turn around", truth: { task: "drive", picks: { move: "left", angle: "180_deg" } }, objs, moves: [] };
    },
    // Drive forward 50 cm with a box on the way. Must pass.
    forward_blocked() {
      const box: Spec = { kind: "box", x: uni(0.2, 0.4), y: uni(-0.06, 0.06), yaw: uni(-1, 1), size: boxSize() };
      const specs = [box];
      specs.push(spot("ball", specs, ...FAR));
      const objs = build(specs);
      return { family: "forward_blocked", goal: "go forward 50 cm", truth: { task: "drive", picks: { move: "forward", distance: "50_cm" } }, objs, moves: [] };
    },
    drive_to() {
      const specs: Spec[] = [];
      for (let i = 0; i < 4; i++) specs.push(spot(random() < 0.5 ? "box" : "ball", specs, ...FAR));
      const objs = build(specs);
      const t = pick(objs);
      return { family: "drive_to", goal: `go to ${t.label}`, truth: { task: "drive_to", picks: { object: t.label } }, objs, target: t, moves: [] };
    },
  };
  const families = Object.keys(FAMILY);

  // ---- the rules decider ----
  const choice = (name: string, options: string[], confidence = 0.9) => ({
    type: "choice" as const,
    choice: name,
    probabilities: Object.fromEntries(options.map((c) => [c, c === name ? 0.9 : 0.1 / Math.max(1, options.length - 1)])),
    confidence,
  });
  // The reading is the truth; each other question is read the way its criteria say.
  function rules(truth: Truth): Decide {
    return async (r: JevRequest) => {
      const usage = { input_tokens: 0, output_tokens: 0 };
      if (r.questions.task) {
        const answers = Object.fromEntries(Object.entries(r.questions).map(([q, { criteria }]) => [q, choice(q === "task" ? truth.task : (truth.picks[q] ?? Object.keys(criteria)[0]), Object.keys(criteria))]));
        return { model: "rules", answers, usage };
      }
      const o = r.state.observation as { [k: string]: string };
      const options = Object.keys(r.questions.next_command.criteria);
      const name = o.to_go
        ? /forward/.test(o.to_go) ? "drive_forward" : /back/.test(o.to_go) ? "drive_backward" : /left/.test(o.to_go) ? "turn_left" : "turn_right"
        : o.destination ? "drive_to_object"
        : o.gripper_vs_object === "holding the object"
          ? options.includes("lift") ? "lift"
          : o.object_vs_target === "away from the target" ? "move_above_target" : o.object_vs_target === "above the target" ? "lower_to_place" : "release"
        : o.gripper_vs_object === "open around the placed object" ? "retreat"
        : o.gripper_vs_object === "clear of the placed object" ? "done"
        : o.gripper_vs_object === "the object is between the jaws" ? "close_gripper"
        : o.gripper !== "open" ? "open_gripper"
        : o.gripper_vs_object === "above the object" ? "lower_to_object"
        : "move_above_object";
      return { model: "rules", answers: { next_command: choice(name, options) }, usage };
    };
  }

  // ---- one run ----
  async function run(family: string, runSeed: number, mode: Mode, jev?: Decide, maxSim = 240) {
    seed = runSeed * 7919 + 13;
    for (let k = 0; k < 5; k++) random();
    const c = FAMILY[family]();
    const events: AgentEvent[] = [];
    const evs: Ev[] = [];
    const note = (kind: string, detail: string) => evs.push({ t: Math.round(data.time * 100) / 100, kind, detail });
    const start = snap();
    let thinking = false;
    let jevMs = 0;
    let calls = 0;
    let inputTokens = 0;
    const truth = rules(c.truth);
    const inner = mode === "jev" ? jev! : truth;
    const decide: Decide = async (r, s) => {
      thinking = true;
      const t0 = performance.now();
      try {
        const out = await inner(r, s);
        if (mode === "jev" && r.questions.next_command && !r.questions.task) {
          const want = (await truth(r, s)).answers.next_command.choice;
          if (out.answers.next_command?.choice !== want) note("jev_off_rules", `Jev chose ${out.answers.next_command?.choice}, the rules say ${want}`);
        }
        calls++;
        inputTokens += out.usage?.input_tokens ?? 0;
        return out;
      } finally {
        jevMs += performance.now() - t0;
        thinking = false;
      }
    };
    // The base moves when the chassis does: the brake holds it still otherwise. What it pushes while moving
    // counts from the start of each move to its end.
    let moving = false;
    let still = 0;
    let atMove = start;
    let maxTilt = 0;
    const touched = new Set<string>();
    const rolled = new Set<string>();
    let steps = 0;
    const controller = new AbortController();
    const simStart = data.time;
    const cpu0 = performance.now();
    let finished = false;
    const task = runAgent({ robot, goal: c.goal, model: "jev-latest", decide, signal: controller.signal, onEvent: (e) => events.push(e), selected: c.selected ?? null }).finally(() => (finished = true));
    while (!finished) {
      if (thinking) {
        await new Promise((r) => setTimeout(r, 2));
        continue;
      }
      for (let i = 0; i < 10; i++) {
        robot.step();
        steps++;
        if (steps % 5) continue;
        const R = data.xmat.subarray(9 * chassis, 9 * chassis + 9);
        const tilt = (Math.acos(Math.min(1, R[8])) * 180) / Math.PI;
        maxTilt = Math.max(maxTilt, tilt);
        const v = Math.hypot(data.qvel[chassisDof], data.qvel[chassisDof + 1]);
        const w = Math.abs(data.qvel[chassisDof + 5]);
        if (v > 0.005 || w > 0.03) {
          if (!moving) atMove = snap();
          moving = true;
          still = 0;
        } else if (moving && (still += 5 * model.opt.timestep) > 0.2) {
          moving = false;
          for (const [o, d] of moved(atMove)) if (d > 0.01) note("base_push", `${o.label}${o === c.target ? " (target)" : ""} moved ${(d * 100).toFixed(1)} cm while the base moved`);
        }
        if (!moving) continue;
        const contacts = data.contact;
        for (let k = 0; k < data.ncon; k++) {
          const ct = contacts.get(k);
          if (!ct) continue;
          const [g1, g2] = [ct.geom1, ct.geom2];
          ct.delete();
          const o = objectGeoms.get(g1) ?? objectGeoms.get(g2);
          if (!o || !o.active || robot.gripped(o)) continue;
          const kind = geomKind(objectGeoms.has(g1) ? g2 : g1);
          if (kind !== "base" && kind !== "arm") continue;
          const key = `${o.label}:${kind}`;
          if (!touched.has(key)) {
            touched.add(key);
            note(kind === "base" ? "base_touch" : "arm_touch_driving", `${o.label}${o === c.target ? " (target)" : ""} touched by the ${kind === "base" ? "chassis or a wheel" : "arm or what it holds"} while driving`);
          }
          const a = robot.object(o);
          if (kind === "base" && (a.bottom > 0.006 || !a.upright || tilt > 3) && !rolled.has(o.label)) {
            rolled.add(o.label);
            note("rollover", `${o.label} under the base: bottom ${(a.bottom * 100).toFixed(1)} cm, upright ${a.upright}, chassis tilt ${tilt.toFixed(1)}°`);
          }
        }
        contacts.delete();
      }
      if (data.time - simStart > maxSim) {
        note("sim_cap", `cut after ${maxSim} s of motion`);
        controller.abort();
        robot.cancel();
      }
      await new Promise((r) => setImmediate(r));
    }
    await task;
    return finish({ c, mode, runSeed, events, evs, start, calls, inputTokens, sim: data.time - simStart, jevMs, cpuMs: performance.now() - cpu0 - jevMs, maxTilt });
  }

  // Every active object's world position, whether the jaws hold it, and whether it stands upright.
  function snap() {
    return new Map(robot.active().map((o) => [o, { p: robot.object(o).world, held: robot.gripped(o), upright: robot.object(o).upright }]));
  }
  // How far each object not held now or then moved since `then`.
  function moved(then: ReturnType<typeof snap>) {
    const now = snap();
    const out: [Obj, number][] = [];
    for (const [o, a] of then) {
      const b = now.get(o);
      if (!b || a.held || b.held) continue;
      out.push([o, Math.hypot(b.p[0] - a.p[0], b.p[1] - a.p[1])]);
    }
    return out;
  }

  function finish(r: {
    c: Case; mode: Mode; runSeed: number; events: AgentEvent[]; evs: Ev[]; start: ReturnType<typeof snap>;
    calls: number; inputTokens: number; sim: number; jevMs: number; cpuMs: number; maxTilt: number;
  }) {
    const { c, events, evs } = r;
    const end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
    const taskEv = events.find((e) => e.type === "task") as Extract<AgentEvent, { type: "task" }> | undefined;
    // Collateral: what the task should not move, moved more than 2 cm; anything upright at the start knocked
    // over; anything rolled under the base; anything the base pushed more than 2 cm.
    for (const [o, d] of moved(r.start)) if (d > 0.02 && !c.moves.includes(o)) note("moved", `${o.label} moved ${(d * 100).toFixed(1)} cm, the task does not move it`);
    const now = snap();
    for (const [o, a] of r.start) if (a.upright && !now.get(o)?.upright && !now.get(o)?.held) note("topple", `${o.label} knocked over`);
    function note(kind: string, detail: string) {
      evs.push({ t: Math.round(r.sim * 100) / 100, kind, detail });
    }
    const heldNow = robot.active().find((o) => robot.gripped(o));
    const wrongObject = !!c.target && c.truth.task === "take" && end.outcome === "success" && heldNow !== c.target;
    if (wrongObject) note("wrong_object", `holding ${heldNow?.label ?? "nothing"}, asked for ${c.target!.label}`);
    const readingOk = !taskEv || r.mode === "rules" ? true : readingMatches(taskEv.text, c);
    if (!readingOk) note("bad_reading", `${taskEv!.name}: ${taskEv!.text}`);
    const collateral = evs.some((e) => e.kind === "moved" || e.kind === "topple" || e.kind === "rollover" || (e.kind === "base_push" && parseFloat(e.detail.split("moved ")[1]) > 2));
    const success = end.outcome === "success" && !wrongObject && readingOk;
    const jevSeconds = r.jevMs / 1000;
    return {
      mode: r.mode,
      family: c.family,
      seed: r.runSeed,
      goal: c.goal,
      reading: taskEv ? `${taskEv.name} (${taskEv.confidence.toFixed(2)}): ${taskEv.text}` : "",
      selected: c.selected?.label ?? null,
      outcome: end.outcome,
      success,
      clean: success && !collateral,
      endText: end.text,
      calls: r.calls,
      inputTokens: r.inputTokens,
      simSeconds: round(r.sim),
      jevSeconds: round(jevSeconds),
      // What the page shows: motion in real time, plus Jev.
      taskSeconds: round(r.sim + (r.mode === "rules" ? RULES_LATENCY * r.calls : jevSeconds)),
      cpuSeconds: round(r.cpuMs / 1000),
      maxTilt: round(r.maxTilt),
      baseTouch: evs.some((e) => e.kind === "base_touch"),
      repeats: repeats(events),
      commands: events.flatMap((e) => (e.type === "decision" ? [e.answer.choice] : [])).join(" > "),
      events: evs,
    };
  }

  return { families, run };
}

const round = (x: number) => Math.round(x * 10) / 10;

// The most times one arm command, or one drive up to something, ran in a row within one subgoal with nothing
// changing. Steps of a plain drive or turn do not count: they are meant to repeat.
function repeats(events: AgentEvent[]) {
  const STEPS = ["drive_forward", "drive_backward", "turn_left", "turn_right"];
  let best = 0;
  let run = 0;
  let last = "";
  let sub = "";
  for (const e of events) {
    if (e.type === "turn" || e.type === "skill") sub = e.subgoal;
    if (e.type !== "result") continue;
    const key = `${sub}|${e.command}|${e.text}`;
    const counts = !STEPS.includes(e.command) || /Stopped early/.test(e.text);
    run = counts && key === last ? run + 1 : counts ? 1 : 0;
    last = key;
    best = Math.max(best, run);
  }
  return best;
}

// The reading names the right task and objects.
function readingMatches(text: string, c: Case) {
  const t = c.truth;
  if (t.task === "take") return text.startsWith(`Take ${t.picks.object} `);
  if (t.task === "put_on") return text.startsWith(`Put ${t.picks.object} on ${t.picks.onto}`);
  if (t.task === "stack_boxes") return text.startsWith("Stack ");
  if (t.task === "drive_to") return text.startsWith(`Drive up to ${t.picks.object}`);
  if (t.task === "drive") return t.picks.angle === "180_deg" ? /Turn left 180/.test(text) : /Drive forward 50 cm/.test(text);
  return true;
}

// The quick subset `bun run check-agent` runs: families and seeds that must end clean.
export const QUICK: { family: string; seeds: number[] }[] = [
  { family: "near_take_ball_selected", seeds: [1001, 1013] },
  { family: "blocked_take", seeds: [1000, 1012] },
  { family: "forward_blocked", seeds: [1010, 1022] },
  { family: "turn_clutter", seeds: [1002, 1014] },
  { family: "far_take", seeds: [1005] },
];

// ---- the summary and the gate ----
const median = (xs: number[]) => quantile(xs, 0.5);
function quantile(xs: number[], q: number) {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.ceil(q * s.length) - 1)];
}
const pct = (n: number, d: number) => (d ? (100 * n) / d : NaN);

export function summarize(records: Record[]) {
  const by = new Map<string, Record[]>();
  for (const r of records) by.set(r.family, [...(by.get(r.family) ?? []), r]);
  const families = [...by].map(([family, rs]) => ({
    family,
    runs: rs.length,
    success: pct(rs.filter((r) => r.success).length, rs.length),
    clean: pct(rs.filter((r) => r.clean).length, rs.length),
    time: median(rs.map((r) => r.taskSeconds)),
    calls: median(rs.map((r) => r.calls)),
    baseTouch: pct(rs.filter((r) => r.baseTouch).length, rs.length),
  }));
  const n = records.length;
  const container = records.filter((r) => r.family.startsWith("container"));
  return {
    runs: n,
    success: pct(records.filter((r) => r.success).length, n),
    clean: pct(records.filter((r) => r.clean).length, n),
    timeMedian: median(records.map((r) => r.taskSeconds)),
    timeP90: quantile(records.map((r) => r.taskSeconds), 0.9),
    callsMedian: median(records.map((r) => r.calls)),
    callsP90: quantile(records.map((r) => r.calls), 0.9),
    callsMax: Math.max(...records.map((r) => r.calls)),
    noBaseTouch: pct(records.filter((r) => !r.baseTouch).length, n),
    maxTilt: Math.max(...records.map((r) => r.maxTilt)),
    tiltOver5: records.filter((r) => r.maxTilt >= 5).length,
    maxRepeats: Math.max(...records.map((r) => r.repeats)),
    wrongObject: families.find((f) => f.family === "near_take_ball_selected")?.clean ?? NaN,
    container: container.length ? pct(container.filter((r) => r.success).length, container.length) : NaN,
    containerClean: container.length ? pct(container.filter((r) => r.clean).length, container.length) : NaN,
    cpuPerRun: records.reduce((s, r) => s + r.cpuSeconds, 0) / n,
    families,
  };
}
export type Summary = ReturnType<typeof summarize>;

// The gate: what a phase must keep. Targets from docs/PLAN-smart-robot.md.
export const TARGETS = {
  clean: 95, // % of runs
  timeMedian: 7, // seconds
  callsMedian: 1,
  callsP90: 2,
  noBaseTouch: 98, // % of runs
  maxTilt: 5, // degrees, every run
  wrongObject: 100, // % clean
  container: 90, // % success
};
// The floor each phase ships on: raised as the phases land.
export const GATE = { clean: 92, wrongObject: 100, noBaseTouch: 98, maxTilt: 5 };

export function gate(s: Summary, floor: { [k: string]: number } = GATE) {
  const fails: string[] = [];
  if (floor.clean !== undefined && !(s.clean >= floor.clean)) fails.push(`clean ${s.clean.toFixed(0)}% under ${floor.clean}%`);
  if (floor.wrongObject !== undefined && !(s.wrongObject >= floor.wrongObject)) fails.push(`wrong-object family ${s.wrongObject.toFixed(0)}% clean, under ${floor.wrongObject}%`);
  if (floor.timeMedian !== undefined && !(s.timeMedian <= floor.timeMedian)) fails.push(`median task ${s.timeMedian} s over ${floor.timeMedian} s`);
  if (floor.callsMedian !== undefined && !(s.callsMedian <= floor.callsMedian)) fails.push(`median Jev calls ${s.callsMedian} over ${floor.callsMedian}`);
  if (floor.callsP90 !== undefined && !(s.callsP90 <= floor.callsP90)) fails.push(`p90 Jev calls ${s.callsP90} over ${floor.callsP90}`);
  if (floor.noBaseTouch !== undefined && !(s.noBaseTouch >= floor.noBaseTouch)) fails.push(`runs without base contact ${s.noBaseTouch.toFixed(0)}% under ${floor.noBaseTouch}%`);
  if (floor.maxTilt !== undefined && !(s.maxTilt < floor.maxTilt)) fails.push(`chassis tilt ${s.maxTilt}° not under ${floor.maxTilt}°`);
  if (floor.container !== undefined && !(s.container >= floor.container)) fails.push(`container tasks ${s.container.toFixed(0)}% under ${floor.container}%`);
  return fails;
}

export function report(s: Summary) {
  const f = (x: number, d = 0) => (Number.isNaN(x) ? "-" : x.toFixed(d));
  const lines = [`${"family".padEnd(24)} ${"runs".padStart(4)} ${"ok%".padStart(5)} ${"clean%".padStart(6)} ${"time s".padStart(6)} ${"calls".padStart(5)} ${"touch%".padStart(6)}`];
  for (const r of [...s.families].sort((a, b) => a.family.localeCompare(b.family)))
    lines.push(`${r.family.padEnd(24)} ${String(r.runs).padStart(4)} ${f(r.success).padStart(5)} ${f(r.clean).padStart(6)} ${f(r.time, 1).padStart(6)} ${f(r.calls).padStart(5)} ${f(r.baseTouch).padStart(6)}`);
  const t = TARGETS;
  const row = (label: string, value: string, target: string) => `${label.padEnd(34)} ${value.padEnd(14)} target ${target}`;
  lines.push(
    "",
    row("runs", String(s.runs), ""),
    row("success", `${f(s.success)}%`, ""),
    row("clean success", `${f(s.clean)}%`, `>= ${t.clean}%`),
    row("task time median / p90", `${f(s.timeMedian, 1)} / ${f(s.timeP90, 1)} s`, `median <= ${t.timeMedian} s`),
    row("Jev calls median / p90 / max", `${f(s.callsMedian)} / ${f(s.callsP90)} / ${f(s.callsMax)}`, `${t.callsMedian} / ${t.callsP90}`),
    row("runs without base contact", `${f(s.noBaseTouch)}%`, `>= ${t.noBaseTouch}%`),
    row("max chassis tilt (runs at 5° or more)", `${f(s.maxTilt, 1)}° (${s.tiltOver5})`, `< ${t.maxTilt}°`),
    row("wrong-object family, clean", `${f(s.wrongObject)}%`, `${t.wrongObject}%`),
    row("container tasks, success (clean)", `${f(s.container)}% (${f(s.containerClean)}%)`, `>= ${t.container}%`),
    row("most repeats of one command", String(s.maxRepeats), "<= 2"),
    row("CPU per run", `${f(s.cpuPerRun, 1)} s`, ""),
  );
  return lines.join("\n");
}
