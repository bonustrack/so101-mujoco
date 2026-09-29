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
  | { type: "decision"; turn: number; answer: ChoiceAnswer; model: string; ms: number; tokens: number }
  | { type: "result"; turn: number; command: string; text: string }
  | { type: "check"; turn: number; success: boolean; text: string }
  | { type: "end"; outcome: "success" | "done" | "stopped" | "turns" | "error"; text: string };

const HOVER = 0.03; // jaw tips 3 cm above the box top
const LIFT = 0.08;
export const MAX_TURNS = 12;

const cm = (m: number) => Math.round(m * 1000) / 10;

// The commands Jev chooses from. The text is what Jev reads; `run` is what the arm does.
// Each text names the exact observation values that make the command right, because Jev reads
// criteria literally: vaguer wording let lower_to_box come close to winning while the gripper was away.
export const COMMANDS: Record<string, { text: string; run?: (robot: Robot, stage: (s: Stage) => void) => Promise<string> }> = {
  open_gripper: {
    text: 'Open the jaws wide. Right when `observation.gripper` is "closed" or "partly open" and `observation.gripper_vs_box` is not "holding the box". Wrong when `observation.gripper` is already "open".',
    run: async (robot, stage) => {
      stage("physics");
      const held = robot.contacts();
      await robot.play([[...robot.target.slice(0, GRIPPER), OPEN]], 0.5);
      return held.fixed && held.moving ? "Jaws open. The box was let go." : "Jaws open.";
    },
  },
  move_above_box: {
    text: 'Move the gripper to hover just above the box, fingers pointing down. Right when `observation.gripper_vs_box` is "away from the box".',
    run: async (robot, stage) => {
      stage("ik");
      const box = robot.box();
      const above: Vec3 = [box.pos[0], box.pos[1], box.top + HOVER];
      let pose = robot.graspYaw(above, box.yaw);
      if (pose.miss > 0.003) pose = { ...pose, ...robot.solve(above, pose.yaw, pose.q, 0.05) };
      // Go through a point high above the box, so the arm never sweeps through it.
      const via = robot.solve([box.pos[0], box.pos[1], 0.16], pose.yaw, pose.q, 0.05);
      const grip = robot.target[GRIPPER];
      stage("physics");
      await robot.play([[...via.q, grip], [...pose.q, grip]], 1.8);
      return reached(robot, above, "Above the box");
    },
  },
  lower_to_box: {
    text: 'Lower the gripper straight down so the box ends up between the jaws. Right only when `observation.gripper_vs_box` is "above the box" and `observation.gripper` is "open". Wrong when the jaws are closed or the gripper is away from the box: it would hit the box.',
    run: async (robot, stage) => {
      stage("ik");
      const box = robot.box();
      const goal: Vec3 = [box.pos[0], box.pos[1], Math.max(box.bottom + 0.01, box.pos[2] - 0.005)];
      const path = line(robot, robot.tcp(), goal, 0.3);
      stage("physics");
      await robot.play(path, 1.0);
      return reached(robot, goal, "Lowered around the box");
    },
  },
  close_gripper: {
    text: 'Close the jaws to grip. Right when `observation.gripper_vs_box` is "the box is between the jaws".',
    run: async (robot, stage) => {
      stage("physics");
      await robot.play([[...robot.target.slice(0, GRIPPER), CLOSED]], 0.8, 0.4);
      const { fixed, moving } = robot.contacts();
      return fixed && moving ? "Jaws closed on the box." : "Jaws closed on nothing.";
    },
  },
  lift: {
    text: 'Raise the gripper 8 cm, carrying what it holds. Right when `observation.gripper_vs_box` is "holding the box" and `observation.box` is "standing on the table".',
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
    text: 'Stop, nothing left to do. Right when `observation` shows `goal` is achieved. To take the box, that means `observation.box` says lifted and `observation.gripper_vs_box` is "holding the box".',
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

// What Jev sees: plain words computed in code, plus the raw positions for context.
export function observe(robot: Robot) {
  const box = robot.box();
  const tip = robot.tcp();
  const angle = robot.joints()[GRIPPER];
  const { fixed, moving } = robot.contacts();
  const holding = fixed && moving;
  const h = robot.boxInHand();
  const between = Math.abs(h[0] - 0.012) < 0.015 && Math.abs(h[1]) < 0.015 && h[2] > -0.11 && h[2] < -0.05;
  const sideways = Math.hypot(tip[0] - box.pos[0], tip[1] - box.pos[1]);
  const aboveTop = tip[2] - box.top;
  const [w, d, hgt] = robot.boxSize.map((s) => cm(2 * s));
  return {
    gripper: holding ? "closed on the box" : angle > 0.5 ? "open" : angle < 0.15 ? "closed" : "partly open",
    gripper_vs_box: holding
      ? "holding the box"
      : between
        ? "the box is between the jaws"
        : sideways < 0.015 && aboveTop > 0
          ? "above the box"
          : "away from the box",
    box:
      !box.upright && !holding
        ? "knocked over on the table"
        : box.bottom > 0.005
          ? `lifted ${cm(box.bottom)} cm off the table`
          : "standing on the table",
    box_position_cm: { forward: cm(box.pos[0]), left: cm(box.pos[1]), up: cm(box.pos[2]) },
    box_size_cm: { width: w, depth: d, height: hgt },
    box_turned_deg: Math.round((box.yaw * 180) / Math.PI),
    gripper_tip_cm: { forward: cm(tip[0]), left: cm(tip[1]), up: cm(tip[2]) },
    gripper_to_box_cm: { sideways: cm(sideways), above_box_top: cm(aboveTop) },
  };
}

// Success is decided here, not by the model: box lowest corner 5 cm up, gripped by both jaws, for 1 s.
async function check(robot: Robot) {
  if (!robot.held()) {
    const box = robot.box();
    return { success: false, text: box.bottom > 0.005 ? `Box ${cm(box.bottom)} cm up, not lifted ${cm(LIFTED)} cm yet.` : "Box not lifted." };
  }
  await robot.hold(1);
  const success = robot.held();
  return { success, text: success ? `Box lifted ${cm(robot.box().bottom)} cm and held for 1 s.` : "The box slipped." };
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
      stage("observe");
      const request: JevRequest = { model, state: { goal, observation: observe(robot), history: history.slice(-4) }, questions: QUESTIONS };
      emit({ type: "turn", turn, request });

      stage("jev");
      const started = performance.now();
      const response = await decide(request, signal);
      const answer = response.answers?.next_command;
      if (answer?.type !== "choice" || !(answer.choice in COMMANDS)) throw new Error(`Unexpected answer from Jev: ${JSON.stringify(response).slice(0, 200)}`);
      tokens += response.usage?.input_tokens ?? 0;
      emit({ type: "decision", turn, answer, model: response.model, ms: performance.now() - started, tokens });

      const command = COMMANDS[answer.choice];
      if (!command.run) {
        stage("check");
        const result = await check(robot);
        return end(result.success ? "success" : "done", `Jev chose done. ${result.text}`);
      }
      const before = robot.box().pos;
      let text = await command.run(robot, stage);
      if (signal.aborted) break;
      const after = robot.box().pos;
      const moved = Math.hypot(after[0] - before[0], after[1] - before[1]);
      if (moved > 0.01 && !robot.contacts().moving) text += ` The box moved ${cm(moved)} cm.`;
      history.push({ command: answer.choice, result: text });
      emit({ type: "result", turn, command: answer.choice, text });

      stage("check");
      const result = await check(robot);
      if (signal.aborted) break;
      emit({ type: "check", turn, ...result });
      if (result.success) return end("success", `Done in ${turn} turns, ${turn} Jev calls, ${tokens.toLocaleString("en-US")} input tokens.`);
    }
    if (signal.aborted) return end("stopped", `Stop pressed during turn ${turn}.`);
    end("turns", `Stopped after ${MAX_TURNS} turns without lifting the box.`);
  } catch (error) {
    if (signal.aborted) return end("stopped", `Stop pressed during turn ${turn}.`);
    end("error", error instanceof Error ? error.message : String(error));
  }
}

// Straight line for the jaw tips, solved as IK waypoints that keep the current jaw heading.
function line(robot: Robot, from: Vec3, to: Vec3, weight: number) {
  const box = robot.box();
  // Snap the jaw heading to the nearest box face, so the jaws stay square to it.
  const quarter = Math.PI / 2;
  const yaw = box.yaw + quarter * Math.round((robot.handYaw() - box.yaw) / quarter);
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
