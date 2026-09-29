// Headless checks for the robot and the Jev loop, with the same code the page runs.
// Usage: bun run check-agent (no API key needed: Jev is mocked).
import loadMujoco from "@mujoco/mujoco";
import { readdirSync, readFileSync } from "node:fs";
import { COMMANDS, observe, runAgent, type AgentEvent, type Decide, type JevRequest } from "../src/agent";
import { createRobot, loadModel } from "../src/robot";
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

// 1. Scripted pick with the same commands Jev picks from: default box, then random boxes.
const PLAN = ["open_gripper", "move_above_box", "lower_to_box", "close_gripper", "lift"];
async function scriptedPick() {
  for (const name of PLAN) await drive(run(name));
  await drive(robot.hold(1));
  return robot.held();
}
robot.reset();
report(await scriptedPick(), "scripted pick, default box", `box ${(robot.box().bottom * 100).toFixed(1)} cm up`);
let picked = 0;
const N = 40;
for (let i = 0; i < N; i++) {
  robot.reset();
  robot.randomBox(random);
  if (await scriptedPick()) picked++;
}
report(picked === N, "scripted pick, random boxes", `${picked}/${N}`);
// The folded arm's jaws rest inside the random area: a box must never start in them.
let touching = 0;
for (let i = 0; i < 500; i++) {
  robot.reset();
  robot.randomBox(random);
  if (robot.contacts().arm) touching++;
}
report(touching === 0, "random boxes start clear of the arm", `${touching}/500 touching`);

// 2. The full loop with a mock Jev that replays the plan, in Jev's answer format.
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
  typeof r.state.observation.gripper_vs_box === "string" &&
  r.questions.next_command.type === "choice" &&
  Object.keys(r.questions.next_command.criteria).join() === Object.keys(COMMANDS).join();

async function loop(decide: Decide, signal = new AbortController().signal) {
  const events: AgentEvent[] = [];
  await drive(runAgent({ robot, goal: "Take the box", model: "jev-latest", decide, signal, onEvent: (e) => events.push(e) }));
  return events;
}
robot.reset();
robot.randomBox(random);
let requestsOk = true;
let turn = 0;
let events = await loop(async (request) => {
  requestsOk &&= validRequest(request);
  return answer(PLAN[turn++] ?? "done");
});
let end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
report(end.outcome === "success" && turn === 5, "loop with scripted mock", `${end.outcome} after ${turn} calls: ${end.text}`);
report(requestsOk, "every request has goal, observation and the Choice question");

// 3. Closed loop: a mock that reads the observation like Jev would, and the box jumps mid-run.
const reactive: Decide = async ({ state: { observation: o } }) =>
  answer(
    o.gripper_vs_box === "holding the box" ? "lift"
    : o.gripper_vs_box === "the box is between the jaws" ? "close_gripper"
    : o.gripper !== "open" ? "open_gripper"
    : o.gripper_vs_box === "above the box" ? "lower_to_box"
    : "move_above_box",
  );
robot.reset();
let moved = false;
events = await loop(async (request) => {
  const choice = (await reactive(request, new AbortController().signal)).answers.next_command.choice;
  if (choice === "lower_to_box" && !moved) {
    moved = true;
    robot.placeBox(0.17, -0.09, 0.6); // the box jumps while the gripper hovers
    return reactive({ ...request, state: { ...request.state, observation: observe(robot) } }, new AbortController().signal);
  }
  return answer(choice);
});
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
const chosen = events.flatMap((e) => (e.type === "decision" ? [e.answer.choice] : []));
report(end.outcome === "success" && chosen.filter((c) => c === "move_above_box").length === 2, "closed loop re-plans after the box moves", chosen.join(" > "));

// 4. Stop mid-run.
robot.reset();
const controller = new AbortController();
turn = 0;
events = await loop(async () => {
  if (turn === 2) controller.abort();
  return answer(PLAN[turn++]);
}, controller.signal);
end = events.at(-1) as Extract<AgentEvent, { type: "end" }>;
report(end.outcome === "stopped", "Stop ends the run", end.text);

// 5. The proxy: no key is refused locally, a wrong key reaches TypeSafe and comes back refused.
const local = await jevFunction(new Request("http://x/.netlify/functions/jev", { method: "POST", body: "{}" }));
report(local.status === 401, "proxy without a key", `HTTP ${local.status}`);
const upstream = await jevFunction(new Request("http://x/.netlify/functions/jev", { headers: { authorization: "Bearer not-a-key" } }));
report(upstream.status === 401 || upstream.status === 403, "proxy forwards to api.typesafe.ai", `HTTP ${upstream.status} ${(await upstream.text()).slice(0, 80)}`);

console.log(failures ? `${failures} failed` : "All checks passed");
process.exit(failures ? 1 : 0);
