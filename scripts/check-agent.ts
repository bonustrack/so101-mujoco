// Headless checks for the robot, the scene editor and the Jev loop, with the same code the page runs.
// Usage: bun run check-agent (no API key needed: Jev is mocked).
import loadMujoco from "@mujoco/mujoco";
import { readdirSync, readFileSync } from "node:fs";
import { COMMANDS, observe, runAgent, type AgentEvent, type Decide, type JevRequest } from "../src/agent";
import { createRobot, loadModel, type Obj, type Vec3 } from "../src/robot";
import jevFunction from "../netlify/functions/jev";

const dir = new URL("../model/", import.meta.url).pathname;
const files = ["scene_web.xml", "so101.xml", ...readdirSync(dir + "assets").map((f) => "assets/" + f)].map(
  (name) => [name, readFileSync(dir + name)] as [string, Uint8Array],
);
const mujoco = await loadMujoco();
const robot = createRobot(mujoco, loadModel(mujoco, files));
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
const answer = (choice: string) => ({
  model: "mock",
  answers: {
    next_command: {
      type: "choice" as const,
      choice,
      probabilities: Object.fromEntries(Object.keys(COMMANDS).map((c) => [c, c === choice ? 0.9 : 0.1 / 6])),
      confidence: 0.88,
    },
  },
  usage: { input_tokens: 700, output_tokens: 20 },
});
const validRequest = (r: JevRequest) =>
  r.model === "jev-latest" &&
  r.state.goal === "Take the box" &&
  typeof r.state.observation.gripper_vs_object === "string" &&
  ["box", "ball"].includes(r.state.observation.object) &&
  typeof r.state.observation.object_size_cm === "object" &&
  r.questions.next_command.type === "choice" &&
  Object.keys(r.questions.next_command.criteria).join() === Object.keys(COMMANDS).join();

async function loop(decide: Decide, signal = new AbortController().signal) {
  const events: AgentEvent[] = [];
  await drive(runAgent({ robot, goal: "Take the box", model: "jev-latest", decide, signal, onEvent: (e) => events.push(e) }));
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
report(end.outcome === "success" && turn === 5, "loop with scripted mock", `${end.outcome} after ${turn} calls: ${end.text}`);
report(requestsOk, "every request has goal, observation with the object type and size, and the Choice question");

// 5. Closed loop: a mock that reads the observation like Jev would, and the box jumps mid-run.
const reactive: Decide = async ({ state: { observation: o } }) =>
  answer(
    o.gripper_vs_object === "holding the object" ? "lift"
    : o.gripper_vs_object === "the object is between the jaws" ? "close_gripper"
    : o.gripper !== "open" ? "open_gripper"
    : o.gripper_vs_object === "above the object" ? "lower_to_object"
    : "move_above_object",
  );
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
events = await loop(async (request) => {
  seen ||= `${request.state.observation.object} ${JSON.stringify(request.state.observation.object_size_cm)}`;
  return reactive(request, new AbortController().signal);
});
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
report(end.outcome === "success" && robot.focus === ball && seen === `ball {"diameter":5}`, "loop takes a dragged, resized ball", `${seen}: ${end.text}`);

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

// 8. The proxy: no key is refused locally, a wrong key reaches TypeSafe and comes back refused.
const local = await jevFunction(new Request("http://x/.netlify/functions/jev", { method: "POST", body: "{}" }));
report(local.status === 401, "proxy without a key", `HTTP ${local.status}`);
const upstream = await jevFunction(new Request("http://x/.netlify/functions/jev", { headers: { authorization: "Bearer not-a-key" } }));
report(upstream.status === 401 || upstream.status === 403, "proxy forwards to api.typesafe.ai", `HTTP ${upstream.status} ${(await upstream.text()).slice(0, 80)}`);

console.log(failures ? `${failures} failed` : "All checks passed");
process.exit(failures ? 1 : 0);
