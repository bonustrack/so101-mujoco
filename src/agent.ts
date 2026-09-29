// The closed loop: observe the scene, ask Jev for the next command, run it with
// IK and physics, check the result in code, repeat. Jev is a decision model, not
// a chat model: it answers a Choice question over the fixed command list below,
// with a probability per command. Numbers and geometry stay in code.
import { CLOSED, GRIPPER, LIFTED, OPEN, REST, type Robot, type Vec3 } from "./robot";

export type ChoiceAnswer = { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number };
export type JevRequest = { model: string; state: { goal: string; observation: Observation; history: Step[] }; questions: typeof QUESTIONS };
export type JevResponse = { model: string; answers: Record<string, ChoiceAnswer>; usage?: { input_tokens: number; output_tokens: number } };
export type Decide = (request: JevRequest, signal: AbortSignal) => Promise<JevResponse>;
export type Observation = ReturnType<typeof observe>;
type Step = { command: string; result: string };

export type Stage = "observe" | "jev" | "ik" | "physics" | "check";
export type AgentEvent =
  | { type: "stage"; stage: Stage }
  | { type: "turn"; turn: number; request: JevRequest }
  | { type: "decision"; turn: number; answer: ChoiceAnswer; model: string; ms: number; tokens: number; usage: JevResponse["usage"] }
  | { type: "result"; turn: number; command: string; text: string }
  | { type: "check"; turn: number; success: boolean; text: string }
  | { type: "end"; outcome: "success" | "done" | "stopped" | "turns" | "error"; text: string };

const HOVER = 0.03; // jaw tips 3 cm above the object top
const LIFT = 0.08;
export const MAX_TURNS = 12;

const cm = (m: number) => Math.round(m * 1000) / 10;

// The commands Jev chooses from. The text is what Jev reads; `run` is what the arm does.
// Each text names the exact observation values that make the command right, because Jev reads
// criteria literally: vaguer wording let lower_to_box come close to winning while the gripper was away.
// They all act on the target object, robot.focus.
export const COMMANDS: Record<string, { text: string; run?: (robot: Robot, stage: (s: Stage) => void) => Promise<string> }> = {
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
      stage("physics");
      await robot.play([[...via.q, grip], [...pose.q, grip]], 1.8);
      return reached(robot, pose.pos, "Above the object");
    },
  },
  lower_to_object: {
    text: 'Lower the gripper straight down so the object ends up between the jaws. Right only when `observation.gripper_vs_object` is "above the object" and `observation.gripper` is "open". Wrong when the jaws are closed or the gripper is away from the object: it would hit the object.',
    run: async (robot, stage) => {
      stage("ik");
      const object = robot.object();
      const yaw = squareYaw(robot);
      const goal = robot.gripPoint(yaw, Math.max(object.bottom + 0.01, object.pos[2] - 0.005));
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

// One question per turn. Instructions point at state fields by name, as the TypeSafe docs advise.
export const QUESTIONS = {
  next_command: {
    type: "choice" as const,
    instructions:
      "A robot arm is working toward `goal`. `observation` is the scene right now and `history` lists the commands already run, oldest first. Which command should the arm run next?",
    criteria: Object.fromEntries(Object.entries(COMMANDS).map(([name, { text }]) => [name, text])),
  },
};

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

// Success is decided here, not by the model: the target's lowest point 5 cm up, gripped by both jaws, for 1 s.
async function check(robot: Robot) {
  const name = robot.focus!.label;
  if (!robot.held()) {
    const object = robot.object();
    return { success: false, text: object.bottom > 0.005 ? `${name} ${cm(object.bottom)} cm up, not lifted ${cm(LIFTED)} cm yet.` : `${name} not lifted.` };
  }
  await robot.hold(1);
  const success = robot.held();
  return { success, text: success ? `${name} lifted ${cm(robot.object().bottom)} cm and held for 1 s.` : `${name} slipped.` };
}

export async function runAgent(options: {
  robot: Robot;
  goal: string;
  model: string;
  decide: Decide;
  signal: AbortSignal;
  onEvent: (event: AgentEvent) => void;
}) {
  const { robot, goal, model, decide, signal, onEvent: emit } = options;
  const stage = (s: Stage) => emit({ type: "stage", stage: s });
  const history: Step[] = [];
  let tokens = 0;
  let turn = 0;
  const end = (outcome: Extract<AgentEvent, { type: "end" }>["outcome"], text: string) => emit({ type: "end", outcome, text });
  try {
    for (turn = 1; turn <= MAX_TURNS; turn++) {
      if (!robot.focus?.active) return end("error", "No object to take. Add a box or a ball.");
      stage("observe");
      const request: JevRequest = { model, state: { goal, observation: observe(robot), history: history.slice(-4) }, questions: QUESTIONS };
      emit({ type: "turn", turn, request });

      stage("jev");
      const started = performance.now();
      const response = await decide(request, signal);
      const answer = response.answers?.next_command;
      if (answer?.type !== "choice" || !(answer.choice in COMMANDS)) throw new Error(`Unexpected answer from Jev: ${JSON.stringify(response).slice(0, 200)}`);
      tokens += response.usage?.input_tokens ?? 0;
      emit({ type: "decision", turn, answer, model: response.model, ms: performance.now() - started, tokens, usage: response.usage });

      const command = COMMANDS[answer.choice];
      if (!command.run) {
        stage("check");
        const result = await check(robot);
        return end(result.success ? "success" : "done", `Jev chose done. ${result.text}`);
      }
      const before = robot.object().pos;
      let text = await command.run(robot, stage);
      if (signal.aborted) break;
      const after = robot.object().pos;
      const moved = Math.hypot(after[0] - before[0], after[1] - before[1]);
      if (moved > 0.01 && !robot.contacts().moving) text += ` The object moved ${cm(moved)} cm.`;
      history.push({ command: answer.choice, result: text });
      emit({ type: "result", turn, command: answer.choice, text });

      stage("check");
      const result = await check(robot);
      if (signal.aborted) break;
      emit({ type: "check", turn, ...result });
      if (result.success) return end("success", `Done in ${turn} turns, ${turn} Jev calls, ${tokens.toLocaleString("en-US")} input tokens.`);
    }
    if (signal.aborted) return end("stopped", `Stop pressed during turn ${turn}.`);
    end("turns", `Stopped after ${MAX_TURNS} turns without lifting ${robot.focus!.label}.`);
  } catch (error) {
    if (signal.aborted) return end("stopped", `Stop pressed during turn ${turn}.`);
    end("error", error instanceof Error ? error.message : String(error));
  }
}

// The jaw heading snapped to the nearest face of the target, so the jaws stay square to it.
function squareYaw(robot: Robot) {
  const face = robot.object().yaw;
  const quarter = Math.PI / 2;
  return face + quarter * Math.round((robot.handYaw() - face) / quarter);
}

// Straight line for the jaw tips, solved as IK waypoints that keep the jaws square to the target.
function line(robot: Robot, from: Vec3, to: Vec3, weight: number) {
  const yaw = squareYaw(robot);
  const grip = robot.target[GRIPPER];
  const path: number[][] = [];
  let seed = robot.joints();
  for (let i = 1; i <= 6; i++) {
    const p = from.map((v, j) => v + ((to[j] - v) * i) / 6) as Vec3;
    seed = robot.solve(p, yaw, seed, weight).q;
    path.push([...seed, grip]);
  }
  return path;
}

function reached(robot: Robot, target: Vec3, text: string) {
  const miss = Math.hypot(...robot.tcp().map((v, i) => v - target[i]));
  return miss < 0.01 ? `${text}, ${cm(miss)} cm off.` : `Stopped ${cm(miss)} cm short of the target.`;
}
