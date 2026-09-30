// Headless checks for the robot, the scene editor and the Jev loop, with the same code the page runs.
// Usage: bun run check-agent (no API key needed: Jev is mocked).
import loadMujoco from "@mujoco/mujoco";
import { readdirSync, readFileSync } from "node:fs";
import { COMMANDS, maxTurns, observe, question, runAgent, type AgentEvent, type Decide, type JevRequest, type Observation } from "../src/agent";
import { plan, type Task } from "../src/plan";
import { STEP, TURN, createRobot, loadModel, type Obj, type Vec3 } from "../src/robot";
import jevFunction from "../netlify/functions/jev";
import { QUICK, createBench } from "./bench-core";

const dir = new URL("../model/", import.meta.url).pathname;
const files = ["scene_web.xml", "so101.xml", ...readdirSync(dir + "assets").map((f) => "assets/" + f)].map(
  (name) => [name, readFileSync(dir + name)] as [string, Uint8Array],
);
const mujoco = await loadMujoco();
const robot = createRobot(mujoco, loadModel(mujoco, files));
const heap = () => robot.data.qpos.buffer.byteLength; // the WASM memory: it grows, never shrinks
const heapStart = heap();
let failures = 0;
const report = (ok: boolean, label: string, detail = "") => {
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? `  (${detail})` : ""}`);
};

// Step the physics while `task` runs, as the page's animation loop does.
async function drive<T>(task: Promise<T>): Promise<T> {
  let finished = false;
  task.finally(() => (finished = true));
  while (!finished) {
    for (let i = 0; i < 20; i++) robot.step();
    await new Promise((r) => setTimeout(r, 0));
  }
  return task;
}
const run = (name: string) => COMMANDS[name].run!(robot, () => {});
let seed = 42;
const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;

// Back to the first scene (one 3 x 3 x 4 cm box), arm at rest, the box as the target.
const fresh = () => {
  robot.resetScene();
  robot.reset();
  robot.focus = robot.pickTarget(null);
  return robot.focus!;
};

// 1. Scripted pick with the same commands Jev picks from: default box, then random boxes.
const PLAN = ["open_gripper", "move_above_object", "lower_to_object", "close_gripper", "lift"];
async function scriptedPick() {
  for (const name of PLAN) await drive(run(name));
  await drive(robot.hold(1));
  return robot.held();
}
fresh();
report(await scriptedPick(), "scripted pick, default box", `box ${(robot.object().bottom * 100).toFixed(1)} cm up`);
let picked = 0;
const N = 40;
for (let i = 0; i < N; i++) {
  robot.randomize(fresh(), random);
  if (await scriptedPick()) picked++;
}
report(picked === N, "scripted pick, random boxes", `${picked}/${N}`);
// The folded arm's jaws rest inside the random area: a box must never start in them, whatever its size.
let touching = 0;
for (let i = 0; i < 500; i++) {
  const box = fresh();
  if (i % 2) robot.resize(box, [0, 1, 2].map(() => 0.01 + 0.03 * random()) as Vec3);
  robot.randomize(box, random);
  if (robot.contacts().arm) touching++;
}
report(touching === 0, "random boxes of any size start clear of the arm", `${touching}/500 touching`);

// 2. Scene editor: fill the pool, move, resize and delete, and the physics stays calm.
const settle = (seconds: number) => {
  for (let i = 0; i < seconds / robot.model.opt.timestep; i++) robot.step();
};
// No object flies off, spins or sinks into the table.
const calm = () => {
  let speed = 0;
  let sink = 0;
  for (const o of robot.active()) {
    speed = Math.max(speed, Math.hypot(...robot.data.qvel.subarray(o.dof, o.dof + 3)));
    sink = Math.min(sink, robot.object(o).bottom);
  }
  return { ok: speed < 0.02 && sink > -0.002, text: `max speed ${(speed * 100).toFixed(2)} cm/s, lowest point ${(sink * 1000).toFixed(1)} mm` };
};
const size = (): Vec3 => [0, 1, 2].map(() => 0.01 + 0.03 * random()) as Vec3;
fresh();
const added: Obj[] = [];
for (let i = 0; i < 12; i++) {
  const o = robot.add(i % 2 ? "ball" : "box");
  if (o) added.push(o);
}
report(added.length === 11 && robot.active().length === 12 && robot.add("box") === null, "add fills the pool of 6 boxes and 6 balls, then refuses", `${robot.active().length} out`);
settle(2);
let state = calm();
report(state.ok, "physics calm after adding 11 objects", state.text);
for (const o of robot.active()) robot.resize(o, size());
settle(2);
state = calm();
report(state.ok, "physics calm after resizing all 12 at random", state.text);
// Drag one box straight through the others, as a mouse would, then let go.
const [dragged, ...others] = robot.active();
for (let t = 0; t <= 1; t += 0.02) {
  const a = -0.8 + 1.6 * t;
  robot.drag(dragged, 0.2 * Math.cos(a), 0.2 * Math.sin(a));
  for (let i = 0; i < 4; i++) robot.step();
}
robot.drop();
settle(2);
state = calm();
const landed = robot.object(dragged).pos;
report(state.ok && Math.abs(landed[0] - 0.2 * Math.cos(0.8)) < 0.02 && Math.abs(landed[1] - 0.2 * Math.sin(0.8)) < 0.02, "drag through the others, drop, settle", state.text);
for (const o of others.slice(0, 6)) robot.remove(o);
settle(1);
state = calm();
report(state.ok && robot.active().length === 6 && others.slice(0, 6).every((o) => robot.object(o).pos[2] < -0.5), "delete parks objects under the table", state.text);
// Grow a box right next to another: they push apart without flying off.
fresh();
const grown = robot.focus!;
const neighbour = robot.add("box")!;
robot.place(neighbour, robot.object(grown).pos[0], robot.object(grown).pos[1] - 0.035, 0);
settle(1);
robot.resize(grown, [0.04, 0.04, 0.04]);
settle(2);
state = calm();
report(state.ok, "grow a box into its neighbour", state.text);
report(robot.tooBig(grown) !== null && robot.tooBig(robot.add("ball")!) === null, "8 cm box too big to grab, balls fit");

// 3. Scripted picks on resized boxes and on balls, at random spots.
const pickCases: [string, "box" | "ball", Vec3][] = [
  ["box 5 x 2.5 x 5 cm", "box", [0.025, 0.0125, 0.025]],
  ["box 6 x 6 x 2 cm", "box", [0.03, 0.03, 0.01]],
  ["box 8 x 3 x 4 cm", "box", [0.04, 0.015, 0.02]],
  ["ball 4 cm", "ball", [0.02, 0.02, 0.02]],
  ["ball 2 cm", "ball", [0.01, 0.01, 0.01]],
  ["ball 7 cm", "ball", [0.035, 0.035, 0.035]],
];
for (const [label, kind, half] of pickCases) {
  let ok = 0;
  for (let i = 0; i < 5; i++) {
    fresh();
    robot.remove(robot.focus!);
    const o = robot.add(kind)!;
    robot.resize(o, half);
    robot.randomize(o, random);
    robot.focus = o;
    settle(0.3);
    if (await scriptedPick()) ok++;
  }
  report(ok === 5, `scripted pick, ${label}`, `${ok}/5`);
}

// 4. The full loop with a mock Jev that replays the plan, in Jev's answer format.
const choice = (name: string, options: string[], confidence = 0.88) => ({
  type: "choice" as const,
  choice: name,
  probabilities: Object.fromEntries(options.map((c) => [c, c === name ? 0.9 : 0.1 / (options.length - 1)])),
  confidence,
});
const answer = (name: string) => ({
  model: "mock",
  answers: { next_command: choice(name, Object.keys(COMMANDS)) },
  usage: { input_tokens: 700, output_tokens: 20 },
});
// The goal reading: `task`, and any parameter the test names; the others take their first option.
let parsed: JevRequest | null = null;
const reading = (task: string, picks: Record<string, string> = {}, confidence = 0.95) => (r: JevRequest) => {
  parsed = r;
  const answers = Object.fromEntries(
    Object.entries(r.questions).map(([q, { criteria }]) => [q, choice(q === "task" ? task : (picks[q] ?? Object.keys(criteria)[0]), Object.keys(criteria), confidence)]),
  );
  return { model: "mock", answers, usage: { input_tokens: 900, output_tokens: 30 } };
};
const validRequest = (r: JevRequest) => {
  const o = r.state.observation as Observation;
  return (
    r.model === "jev-latest" &&
    r.state.goal === "Take the box" &&
    typeof o.gripper_vs_object === "string" &&
    ["box", "ball"].includes(o.object) &&
    typeof o.object_size_cm === "object" &&
    r.questions.next_command.type === "choice" &&
    Object.keys(r.questions.next_command.criteria).join() === Object.keys(question("take").next_command.criteria).join()
  );
};

async function loop(decide: Decide, signal = new AbortController().signal, goal = "Take the box", task = reading("take"), selected: Obj | null = null) {
  const events: AgentEvent[] = [];
  const both: Decide = async (request, s) => (request.questions.task ? task(request) : decide(request, s));
  await drive(runAgent({ robot, goal, model: "jev-latest", decide: both, signal, onEvent: (e) => events.push(e), selected }));
  return events;
}
robot.randomize(fresh(), random);
let requestsOk = true;
let turn = 0;
let events = await loop(async (request) => {
  requestsOk &&= validRequest(request);
  return answer(PLAN[turn++] ?? "done");
});
let end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
report(end.outcome === "success" && turn === 5, "loop with scripted mock", `${end.outcome} after ${turn} turns: ${end.text}`);
report(requestsOk, "every request has goal, observation with the object type and size, and the Choice question");

// 5. Closed loop: a mock that reads the observation like Jev would, and the box jumps mid-run.
const reactive: Decide = async ({ state }) => {
  const o = state.observation as Observation;
  return answer(
    o.gripper_vs_object === "holding the object" ? "lift"
    : o.gripper_vs_object === "the object is between the jaws" ? "close_gripper"
    : o.gripper !== "open" ? "open_gripper"
    : o.gripper_vs_object === "above the object" ? "lower_to_object"
    : "move_above_object",
  );
};
fresh();
let moved = false;
events = await loop(async (request) => {
  const choice = (await reactive(request, new AbortController().signal)).answers.next_command.choice;
  if (choice === "lower_to_object" && !moved) {
    moved = true;
    robot.place(robot.focus!, 0.17, -0.09, 0.6); // the box jumps while the gripper hovers
    return reactive({ ...request, state: { ...request.state, observation: observe(robot) } }, new AbortController().signal);
  }
  return answer(choice);
});
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
const chosen = events.flatMap((e) => (e.type === "decision" ? [e.answer.choice] : []));
report(end.outcome === "success" && chosen.filter((c) => c === "move_above_object").length === 2, "closed loop re-plans after the box moves", chosen.join(" > "));

// 6. The loop on a ball that was dragged to a new spot and selected, next to the first box.
fresh();
const ball = robot.add("ball")!;
robot.resize(ball, [0.025, 0.025, 0.025]);
robot.drag(ball, 0.18, -0.12);
for (let i = 0; i < 20; i++) robot.step();
robot.drop();
settle(0.5);
robot.focus = robot.pickTarget(ball);
let seen = "";
events = await loop(
  async (request) => {
    const o = request.state.observation as Observation;
    seen ||= `${o.object} ${JSON.stringify(o.object_size_cm)}`;
    return reactive(request, new AbortController().signal);
  },
  undefined,
  "Take the ball",
  reading("take", { object: ball.label }),
  ball,
);
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
report(end.outcome === "success" && robot.focus === ball && seen === `ball {"diameter":5}`, "loop takes a dragged, resized ball", `${seen}: ${end.text}`);

// The goal wins over the selection: "take the ball" with a box selected takes the ball. Must pass.
fresh();
robot.add("box");
const wanted = robot.add("ball")!;
robot.focus = null;
events = await loop(reactive, undefined, "take the ball", reading("take", { object: wanted.label }), robot.active()[0]);
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
report(end.outcome === "success" && robot.gripped(wanted), "take the ball with a box selected takes the ball", `holding ${robot.active().find(robot.gripped)?.label ?? "nothing"}: ${end.text}`);
// "Take it" with a box selected: Jev names the selection, the robot takes it.
fresh();
const it = robot.add("box")!;
events = await loop(reactive, undefined, "take it", reading("take", { object: it.label }), it);
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
report(end.outcome === "success" && robot.gripped(it), "take it with a box selected takes that box", end.text);

// 7. Stop mid-run.
fresh();
const controller = new AbortController();
turn = 0;
events = await loop(async () => {
  if (turn === 2) controller.abort();
  return answer(PLAN[turn++]);
}, controller.signal);
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
report(end.outcome === "stopped", "Stop ends the run", end.text);

// 8. Reading the goal: the request carries the objects in words, and code checks Jev's reading.
fresh();
robot.add("box");
events = await loop(async () => answer("done"), undefined, "line them up", reading("other"));
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
const sentParse = parsed as JevRequest | null; // set inside the mock, which TypeScript cannot follow
const labels = (sentParse?.state.objects as string[] | undefined) ?? [];
report(
  end.outcome === "error" && /Not something this arm can do yet/.test(end.text) && !events.some((e) => e.type === "turn") &&
    labels.length === 2 && labels.every((l) => /^Box \d: a box, [\d.]+ x [\d.]+ x [\d.]+ cm, the /.test(l)) &&
    Object.keys(sentParse!.questions).join() === "task,object,onto,order,spot,move,distance,angle",
  "an unsupported goal ends before any turn, and says what works",
  end.text,
);
// Only the reading matters here: the first turn stops the run.
const stopAtFirstTurn = () => {
  const c = new AbortController();
  return [async () => (c.abort(), answer("done")), c.signal] as const;
};
events = await loop(...stopAtFirstTurn(), "stack", reading("stack_boxes", {}, 0.4));
const taskEvent = events.find((e) => e.type === "task") as Extract<AgentEvent, { type: "task" }>;
report(taskEvent.confidence === 0.4 && taskEvent.maxTurns === 8 * 2 + 4, "a low-confidence reading is passed on with the turn budget", `confidence ${taskEvent.confidence}, ${taskEvent.maxTurns} turns`);

// 9. Pick and place: a mock that reads the observation the way Jev's criteria describe. Whatever is out of
// the arm's reach, it drives up to.
const placer: Decide = async ({ state }) => {
  const o = state.observation as ReturnType<typeof observe> & { object_vs_target?: string; destination?: string };
  return answer(
    o.destination ? "drive_to_object"
    : o.gripper_vs_object === "holding the object"
      ? o.object_vs_target === "away from the target" ? "move_above_target" : o.object_vs_target === "above the target" ? "lower_to_place" : "release"
      : o.gripper_vs_object === "open around the placed object" ? "retreat"
      : o.gripper_vs_object === "the object is between the jaws" ? "close_gripper"
      : o.gripper !== "open" ? "open_gripper"
      : o.gripper_vs_object === "above the object" ? "lower_to_object"
      : "move_above_object",
  );
};
// A scene of `kinds`, each at a random spot, boxes at random sizes unless `sized` is false.
const scene = (kinds: ("box" | "ball")[], sized = true) => {
  robot.resetScene();
  robot.reset();
  robot.remove(robot.active()[0]);
  const objects = kinds.map((k) => robot.add(k)!);
  if (sized) for (const o of objects) if (o.kind === "box") robot.resize(o, [0.012 + 0.016 * random(), 0.012 + 0.016 * random(), 0.012 + 0.013 * random()]);
  for (const o of objects) robot.randomize(o, random);
  robot.reset();
  settle(0.5);
  return objects;
};
let stacked = 0;
const worst: string[] = [];
for (let i = 0; i < 10; i++) {
  scene(["box", "box", "box"]);
  events = await loop(placer, undefined, "stack all the boxes", reading("stack_boxes"));
  end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
  if (end.outcome === "success") stacked++;
  else worst.push(end.text);
}
report(stacked >= 8, "stack 3 boxes of different sizes, 10 random layouts", `${stacked}/10${worst.length ? `; ${worst[0]}` : ""}`);
// Three boxes as Add box makes them, 3 x 3 x 4 cm: a 12 cm tower.
let plain = 0;
const why: string[] = [];
for (let i = 0; i < 5; i++) {
  scene(["box", "box", "box"], false);
  events = await loop(placer, undefined, "stack all the boxes", reading("stack_boxes"));
  end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
  if (end.outcome === "success") plain++;
  else why.push(end.text);
}
report(plain === 5, "stack 3 default boxes, 5 random layouts", `${plain}/5${why.length ? `; ${why[0]}` : ""}`);
// A box knocked off the tower goes back on the list and back on the tower.
const [b1, b2, b3] = scene(["box", "box", "box"], false);
let knocked = false;
events = await loop(
  async (request) => {
    const task: Task = { kind: "stack", objects: [b1, b2, b3], spot: "here", skipped: [] };
    if (!knocked && plan(robot, task).steps[1].done) {
      knocked = true;
      const spot = robot.clearSpot(b2);
      robot.place(b2, spot.x, spot.y, 0); // Box 2 is swept off the tower onto the table
      settle(0.5);
    }
    return placer(request, new AbortController().signal);
  },
  undefined,
  "stack all the boxes",
  reading("stack_boxes"),
);
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
const replans = events.filter((e) => e.type === "plan").map((e) => (e as Extract<AgentEvent, { type: "plan" }>).plan.steps.filter((s) => s.done).length);
report(knocked && end.outcome === "success" && replans.some((n, i) => i > 0 && n < replans[i - 1]), "a box knocked off the tower is queued again", `done steps over time ${replans.join(" > ")}: ${end.text}`);
// Put X on Y, and a ball on top of a stack of everything.
const [x, y] = scene(["box", "box"]);
events = await loop(placer, undefined, `put ${x.label} on ${y.label}`, reading("put_on", { object: x.label, onto: y.label }));
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
report(end.outcome === "success", "put one box on another", end.text);
scene(["box", "box", "ball"], false);
events = await loop(placer, undefined, "stack everything", reading("stack_all"));
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
report(end.outcome === "success", "stack 2 boxes and a ball on top", end.text);
// Too tall: what goes over 10 cm stays out, and the budget follows the objects that go in.
const tall = scene(["box", "box", "box", "box"], false);
for (const o of tall) robot.resize(o, [0.015, 0.015, 0.025]);
events = await loop(...stopAtFirstTurn(), "stack", reading("stack_boxes"));
const read = events.find((e) => e.type === "task") as Extract<AgentEvent, { type: "task" }>;
report(/Box 4 stays out/.test(read.text) && read.maxTurns === maxTurns({ kind: "stack", objects: tall.slice(0, 3), spot: "here", skipped: [] }), "a tower over 10 cm leaves the last box out", read.text);

// 10. The wheeled base: fixed steps with the wheel servos, Jev's drive commands, and driving up to what is out of reach.
const pose = () => robot.base();
const deg = (rad: number) => (rad * 180) / Math.PI;
fresh();
let p0 = pose();
let base = await drive(robot.move("drive", STEP));
let p1 = pose();
const along = (p1.x - p0.x) * Math.cos(p0.yaw) + (p1.y - p0.y) * Math.sin(p0.yaw);
report(Math.abs(along - STEP) < 0.01 && Math.abs(deg(p1.yaw - p0.yaw)) < 1 && !base.bumped, "drive forward one step moves the base about 10 cm", `${(along * 100).toFixed(1)} cm, heading ${deg(p1.yaw - p0.yaw).toFixed(1)}°`);
p0 = pose();
base = await drive(robot.move("turn", TURN));
p1 = pose();
report(Math.abs(deg(p1.yaw - p0.yaw) - 15) < 2 && Math.hypot(p1.x - p0.x, p1.y - p0.y) < 0.02, "turn left one step turns the base about 15°", `${deg(p1.yaw - p0.yaw).toFixed(1)}°, the arm's base shifted ${(Math.hypot(p1.x - p0.x, p1.y - p0.y) * 100).toFixed(1)} cm`);
// Parked, the brake holds the base while the arm works.
p0 = pose();
robot.randomize(robot.focus!, random);
await scriptedPick();
p1 = pose();
report(Math.hypot(p1.x - p0.x, p1.y - p0.y) < 0.001 && Math.abs(deg(p1.yaw - p0.yaw)) < 0.1, "the parked base holds still while the arm picks", `${(Math.hypot(p1.x - p0.x, p1.y - p0.y) * 1000).toFixed(2)} mm, ${deg(p1.yaw - p0.yaw).toFixed(2)}°`);
// Driving into a box stops the base at the touch, and moving on from there still drives.
fresh();
robot.place(robot.focus!, 0.12, 0, 0);
base = await drive(robot.move("drive", STEP));
report(base.bumped === robot.focus && base.moved > 0.01 && base.moved < 0.05, "the base stops when it bumps into a box", `${(base.moved * 100).toFixed(1)} cm, touched ${base.bumped?.label ?? "nothing"}`);

// Jev's drive commands, with a mock that reads the observation the way the criteria describe.
const driver: Decide = async (request, signal) => {
  const o = request.state.observation as { to_go?: string; destination?: string };
  if (o.to_go) return answer(/forward/.test(o.to_go) ? "drive_forward" : /back/.test(o.to_go) ? "drive_backward" : /left/.test(o.to_go) ? "turn_left" : "turn_right");
  return request.questions.next_command.criteria.lift ? reactive(request, signal) : placer(request, signal);
};
fresh();
robot.remove(robot.focus!); // a clear floor ahead
p0 = pose();
events = await loop(driver, undefined, "go forward 20 cm", reading("drive", { move: "forward", distance: "20_cm" }));
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
p1 = pose();
let steps = events.flatMap((e) => (e.type === "decision" ? [e.answer.choice] : []));
report(end.outcome === "success" && Math.abs(Math.hypot(p1.x - p0.x, p1.y - p0.y) - 0.2) < 0.01 && steps.join() === "drive_forward,drive_forward", "Jev drives forward 20 cm in two steps", `${steps.join(" > ")}: ${end.text}`);
p0 = pose();
events = await loop(driver, undefined, "turn left", reading("drive", { move: "left" }));
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
p1 = pose();
report(end.outcome === "success" && Math.abs(deg(p1.yaw - p0.yaw) - 15) < 2, "Jev turns left 15°", `${deg(p1.yaw - p0.yaw).toFixed(1)}°: ${end.text}`);
// Drive, then pick: a box 60 cm away to the front left, out of reach.
fresh();
robot.place(robot.focus!, 0.55, 0.3, 0.4);
robot.reset();
events = await loop(driver, undefined, "drive to the box and pick it up", reading("take"));
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
steps = events.flatMap((e) => (e.type === "decision" ? [e.answer.choice] : []));
const drove = events.some((e) => e.type === "plan" && e.plan.steps[0]?.text === "Drive to Box 1");
report(end.outcome === "success" && robot.held() && drove && steps[0] === "drive_to_object", "drive to a box out of reach, then pick it up", `${steps.join(" > ")}: ${end.text}`);
// Put a far box on a near one: drive to it, take it, carry it back on the base, set it down.
const [near, far] = scene(["box", "box"], false);
robot.place(near, 0.2, 0.12, 0);
robot.place(far, 0.6, -0.35, 0.3);
robot.reset();
events = await loop(driver, undefined, `put ${far.label} on ${near.label}`, reading("put_on", { object: far.label, onto: near.label }));
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
steps = events.flatMap((e) => (e.type === "decision" ? [e.answer.choice] : []));
report(end.outcome === "success" && steps.filter((c) => c === "drive_to_object").length === 2, "put a far box on a near one, driving both ways", `${steps.join(" > ")}: ${end.text}`);
// Stacking still works after the base has moved: three boxes in front of the base where it now stands.
scene(["box", "box", "box"], false);
await drive(robot.move("turn", -2 * TURN));
for (const o of robot.active()) robot.randomize(o, random);
settle(0.5);
events = await loop(placer, undefined, "stack all the boxes", reading("stack_boxes"));
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
report(end.outcome === "success", "stack 3 boxes after the base has turned", end.text);
// The quick bench: random scenes from the bench families that must end clean.
const bench = createBench(robot);
for (const { family, seeds } of QUICK)
  for (const seed of seeds) {
    const r = await bench.run(family, seed, "rules");
    report(r.clean, `bench ${family} #${seed} ends clean`, `${r.outcome} in ${r.taskSeconds} s: ${r.endText}${r.events.length ? ` | ${r.events.map((e) => e.kind).join(",")}` : ""}`);
  }
// Every run above frees what it reads from the WASM heap.
const growth = (heap() - heapStart) / 1e6;
report(growth < 64, "WASM memory stays flat across all runs", `${(heapStart / 1e6).toFixed(0)} MB at start, +${growth.toFixed(0)} MB`);

// 11. The proxy: no key is refused locally, a wrong key reaches TypeSafe and comes back refused.
const local = await jevFunction(new Request("http://x/.netlify/functions/jev", { method: "POST", body: "{}" }));
report(local.status === 401, "proxy without a key", `HTTP ${local.status}`);
const upstream = await jevFunction(new Request("http://x/.netlify/functions/jev", { headers: { authorization: "Bearer not-a-key" } }));
report(upstream.status === 401 || upstream.status === 403, "proxy forwards to api.typesafe.ai", `HTTP ${upstream.status} ${(await upstream.text()).slice(0, 80)}`);

console.log(failures ? `${failures} failed` : "All checks passed");
process.exit(failures ? 1 : 0);
