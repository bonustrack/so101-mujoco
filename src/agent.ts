// The loop: Jev decides what, code decides how. Jev is a decision model, not a chat model: it answers Choice
// questions over fixed lists, with a probability per option.
//
// 1. Goal: one Jev call maps the goal text to a task (take, stack the boxes, stack everything, put X on Y, drive
//    or turn, drive to an object) and its parameters (which object, onto which, stack order, where, which way
//    and how far).
// 2. Plan: code expands the task into steps (plan.ts), again after every step, from the physics state: a box
//    that fell off a tower simply shows up as not done again.
// 3. Skills: code runs each step with a skill (skills.ts): GoTo, Pick or Place, checked as it goes.
// 4. Check: code decides when a step is done and when the whole task is.
// 5. Monitor: a step that fails is tried another way (the jaws turned, another side, backing off first), each
//    way once. When code has no way left, one Jev call chooses: try once more, leave this object out, or stop.
//    Every task has a time budget.
import { LIFTED, type Obj, type Robot } from "./robot";
import { approachTo, driven, inReach, parseRequest, plan, readTask, taskSize, toGo, type ChoiceQuestion, type Drive, type Plan, type Task, type TaskName } from "./plan";
import { cm, deg, goTo, pick, place, versus, type Approach, type Ctx, type DriveSub, type Place, type Stage } from "./skills";
import { simWorld } from "./world";

export type { Stage };
export type ChoiceAnswer = { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number };
export type JevRequest = { model: string; state: Record<string, unknown>; questions: Record<string, ChoiceQuestion> };
export type JevResponse = { model: string; answers: Record<string, ChoiceAnswer>; usage?: { input_tokens: number; output_tokens: number } };
export type Decide = (request: JevRequest, signal: AbortSignal) => Promise<JevResponse>;
export type Skill = "drive" | "pick" | "place";

export type AgentEvent =
  | { type: "stage"; stage: Stage }
  | { type: "task"; request: JevRequest; answers: Record<string, ChoiceAnswer>; name: TaskName; confidence: number; text: string; ok: boolean; budget: number; model: string; ms: number; usage: JevResponse["usage"] }
  | { type: "plan"; plan: Plan }
  // A step starts: which skill, on which step of the plan, the how-many-th try, and how this try differs.
  | { type: "skill"; step: number; skill: Skill; subgoal: string; attempt: number; how: string }
  // The path the base's pivot is about to follow, world x and y.
  | { type: "path"; points: [number, number][] }
  | { type: "result"; action: string; text: string }
  | { type: "check"; success: boolean; text: string }
  // Code ran out of ways: Jev chose what to do.
  | { type: "recovery"; request: JevRequest; answer: ChoiceAnswer; model: string; ms: number; usage: JevResponse["usage"] }
  | { type: "end"; outcome: "success" | "done" | "stopped" | "stuck" | "budget" | "error"; text: string };

// The time budget: seconds of motion for the whole task.
export const budget = (task: Task) => 60 + 20 * Math.max(1, taskSize(task));
// The ways each skill is tried, in order, once each.
export const HOW: Record<Skill, string[]> = {
  drive: ["the shortest free path", "backing off 10 cm first", "the shortest free path again"],
  pick: ["the best jaw angle", "the jaws turned 90°", "from another side"],
  place: ["as planned", "landing a quarter turn around", "from another side"],
};

// When code has no way left: what Jev chooses from.
export function recoveryQuestion(canSkip: boolean): Record<string, ChoiceQuestion> {
  return {
    recovery: {
      type: "choice",
      instructions: "A robot arm on wheels working toward `goal` could not do `step`. It tried every way it knows, and `tried` says how each try ended. What should it do now?",
      criteria: {
        retry: 'Start `step` over, once more. Right when `already_retried` is "no" and `tried` looks like bad luck: something slipped, moved, fell or missed by a little.',
        ...(canSkip && { skip: 'Leave this object out and go on with `steps_left`. Right when `already_retried` is "yes" and `goal` can still be done in part without it, like the other boxes of a tower.' }),
        stop: `Stop and say why. Right when \`step\` cannot work the way things stand${canSkip ? ", or when `goal` makes no sense without it" : ', or when `already_retried` is "yes"'}.`,
      },
    },
  };
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
  const ctx: Ctx = {
    robot,
    world: simWorld(robot),
    stage,
    say: (action, text) => emit({ type: "result", action, text }),
    path: (points) => emit({ type: "path", points }),
    signal,
  };
  let tokens = 0;
  let calls = 0;
  let steps = 0;
  const end = (outcome: Extract<AgentEvent, { type: "end" }>["outcome"], text: string) => emit({ type: "end", outcome, text });
  const ask = async (request: JevRequest) => {
    const started = performance.now();
    const response = await decide(request, signal);
    calls++;
    tokens += response.usage?.input_tokens ?? 0;
    return { response, ms: performance.now() - started };
  };
  const tally = () => `${steps} ${steps === 1 ? "step" : "steps"}, ${calls} Jev ${calls === 1 ? "call" : "calls"}, ${tokens.toLocaleString("en-US")} input tokens.`;
  try {
    // 1. The goal: one call, the task and its parameters.
    stage("plan");
    const parse = parseRequest(robot, goal, selected);
    const parseRequestBody: JevRequest = { model, ...parse };
    stage("jev");
    const { response: parsed, ms } = await ask(parseRequestBody);
    for (const q of Object.keys(parse.questions))
      if (parsed.answers?.[q]?.type !== "choice") throw new Error(`Unexpected answer from Jev: ${JSON.stringify(parsed).slice(0, 200)}`);
    const reading = readTask(robot, parsed.answers, selected);
    const task = reading.task;
    const seconds = task ? budget(task) : 0;
    emit({ type: "task", request: parseRequestBody, answers: parsed.answers, name: reading.name, confidence: reading.confidence, text: reading.text, ok: !!task, budget: seconds, model: parsed.model, ms, usage: parsed.usage });
    if (!task) return end("error", reading.text);

    // 2. Steps, each the current step of a plan made again from the physics state.
    stage("plan");
    let current = plan(robot, task);
    emit({ type: "plan", plan: current });
    const started = robot.data.time;
    const tried = new Map<string, string[]>(); // step text -> how each failed try ended
    const retried = new Set<string>();
    for (;;) {
      if (signal.aborted) return end("stopped", "Stop pressed.");
      if (!current.current) {
        stage("check");
        const all = await checkAll(robot, task);
        if (signal.aborted) return end("stopped", "Stop pressed.");
        if (all.success) return end("success", `${steps ? "" : "Already done. "}${all.text} Done in ${tally()}`);
        emit({ type: "check", ...all });
        current = plan(robot, task);
        emit({ type: "plan", plan: current });
        if (!current.current) return end("done", all.text);
      }
      const used = robot.data.time - started;
      if (used > seconds) return end("budget", `Out of time: ${Math.round(used)} s of motion, over the ${seconds} s this task gets. Left: ${left(current)}.`);
      const sub = current.current!;
      if (sub.kind !== "drive" && !sub.object.active) return end("error", `${sub.object.label} was deleted.`);
      const skill: Skill = sub.kind === "drive" || sub.kind === "approach" ? "drive" : sub.kind === "take" ? "pick" : "place";
      const failed = tried.get(sub.text) ?? [];
      if (failed.length >= HOW[skill].length) {
        // 3. Code has no way left: Jev chooses.
        const canSkip = sub.kind === "place" && ((task.kind === "stack" && task.objects.indexOf(sub.object) > 0 && task.objects.length > 2) || (task.kind === "put_in" && task.objects.length > 1));
        const request: JevRequest = {
          model,
          state: { goal, step: sub.text, tried: failed, steps_left: current.steps.filter((s) => !s.done && s.text !== sub.text).map((s) => s.text), already_retried: retried.has(sub.text) ? "yes" : "no" },
          questions: recoveryQuestion(canSkip),
        };
        stage("jev");
        const { response, ms } = await ask(request);
        const answer = response.answers?.recovery;
        if (answer?.type !== "choice" || !(answer.choice in request.questions.recovery.criteria)) throw new Error(`Unexpected answer from Jev: ${JSON.stringify(response).slice(0, 200)}`);
        emit({ type: "recovery", request, answer, model: response.model, ms, usage: response.usage });
        if (answer.choice === "retry" && !retried.has(sub.text)) {
          retried.add(sub.text);
          tried.set(sub.text, []);
        } else if (answer.choice === "skip" && (task.kind === "stack" || task.kind === "put_in") && sub.kind === "place") {
          task.objects = task.objects.filter((o) => o !== sub.object);
          task.skipped.push(`${sub.object.label} left out: ${failed.at(-1)}`);
          current = plan(robot, task);
          emit({ type: "plan", plan: current });
          continue;
        } else return end("stuck", `Could not ${sub.text.charAt(0).toLowerCase() + sub.text.slice(1)}. ${failed.at(-1) ?? ""}`.trim());
      }
      const attempt = (tried.get(sub.text) ?? []).length;
      steps++;
      robot.focus = sub.kind === "drive" ? null : sub.object;
      emit({ type: "skill", step: steps, skill, subgoal: sub.text, attempt: attempt + 1, how: HOW[skill][attempt] });
      const outcome = await run(ctx, sub, attempt);
      if (signal.aborted) return end("stopped", "Stop pressed.");

      // 4. Check the step, then plan again from what the physics shows now.
      stage("check");
      const result = sub.kind === "take" ? await checkTake(robot) : sub.kind === "place" ? await checkPlace(robot, task, sub) : await checkBase(robot, sub);
      if (signal.aborted) return end("stopped", "Stop pressed.");
      emit({ type: "check", ...result });
      if (sub.kind === "take" && result.success) return end("success", `${result.text} Done in ${tally()}`);
      stage("plan");
      const next = plan(robot, task);
      if (!result.success && next.current?.text === sub.text) tried.set(sub.text, [...(tried.get(sub.text) ?? []), outcome.ok ? result.text : outcome.text]);
      if (JSON.stringify(next.steps) !== JSON.stringify(current.steps)) emit({ type: "plan", plan: next });
      current = next;
    }
  } catch (error) {
    if (signal.aborted) return end("stopped", "Stop pressed.");
    end("error", error instanceof Error ? error.message : String(error));
  }
}
const left = (p: Plan) => p.steps.filter((s) => !s.done).map((s) => s.text).join(", ");

// The skill for a step, in the way `attempt` says.
async function run(ctx: Ctx, sub: NonNullable<Plan["current"]>, attempt: number) {
  const { robot } = ctx;
  if (sub.kind === "drive" || sub.kind === "approach") {
    const next = sub.kind === "approach" && !robot.gripped(sub.object) ? sub.object : undefined;
    const r = await goTo(ctx, sub, { backOff: attempt === 1, preshape: next });
    ctx.say(sub.kind === "drive" ? "drive" : "drive_to_object", r.text);
    return r;
  }
  const around = approachTo(robot, sub.kind === "place" && robot.gripped(sub.object) ? sub.target : sub.object);
  return sub.kind === "take" ? pick(ctx, sub.object, attempt, around) : place(ctx, sub, attempt, around);
}

// ---- checks: code decides when a step is done ----

// Driving: the base is where the goal asked, still 0.5 s later. Approaching: the destination is in reach.
async function checkBase(robot: Robot, sub: DriveSub | Approach) {
  if (sub.kind === "approach") {
    const { r, a } = robot.polar(sub.where());
    const ok = inReach(robot, sub.where());
    return { success: ok, text: `${sub.label} is ${cm(r)} cm away, ${Math.abs(deg(a))}° ${a > 0 ? "left" : "right"}: ${ok ? "in reach" : "not in reach yet"}.` };
  }
  const togo = toGo(robot, sub.drive);
  if (!driven(robot, sub.drive)) return { success: false, text: `${sub.drive.turn ? `${Math.abs(deg(togo))}°` : `${Math.abs(cm(togo))} cm`} to go.` };
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
    if (sub.target.into) return { success: false, text: `${o.label} is not in the container.` };
    const v = versus(robot, o, sub.target);
    const held = robot.touching(o);
    const text =
      held.fixed && held.moving
        ? v.sitting
          ? `${o.label} sits on ${sub.target.label}, still in the jaws.`
          : `${o.label} in the jaws, ${cm(v.off)} cm from ${sub.target.label} sideways, ${cm(v.gap)} cm above it.`
        : v.sitting
          ? `${o.label} sits on ${sub.target.label}, the jaws are still around it.`
          : `${o.label} is not on ${sub.target.label}.`;
    return { success: false, text };
  }
  await robot.hold(0.5);
  if (!achieved() || robot.speed(o) > 0.01) return { success: false, text: `${o.label} did not stay ${sub.target.into ? "in" : "on"} ${sub.target.label}.` };
  if (sub.target.into) return { success: true, text: `${o.label} is in the container, still after 0.5 s.` };
  const v = versus(robot, o, sub.target);
  return { success: true, text: `${o.label} sits on ${sub.target.label}, ${cm(v.off)} cm off centre, still after 0.5 s.` };
}
// The plan step a subgoal belongs to (the subgoal text names the current top; the step names the tower).
const stepOf = (task: Task, sub: Place) => (task.kind === "stack" && sub.object !== task.objects[0] ? `Put ${sub.object.label} on the tower` : sub.text);

// The whole task: every step done, still true 1 s later.
async function checkAll(robot: Robot, task: Task) {
  await robot.hold(1);
  const now = plan(robot, task);
  if (now.current) return { success: false, text: "Something moved during the last second." };
  if (task.kind === "stack") {
    const top = robot.object(task.objects.at(-1)!).top;
    return { success: true, text: `${task.objects.length} objects stacked, ${cm(top)} cm tall, standing for 1 s.${task.skipped.length ? " " + task.skipped.join(" ") : ""}` };
  }
  if (task.kind === "drive") return { success: true, text: baseMoved(robot, task.drive) };
  if (task.kind === "put_in") {
    const n = task.objects.length;
    return { success: true, text: `${n === 1 ? `${task.objects[0].label} is` : `${n} objects are`} in the container, still for 1 s.${task.skipped.length ? " " + task.skipped.join(" ") : ""}` };
  }
  if (task.kind === "drive_to") {
    const { r } = robot.polar(robot.object(task.object).pos);
    return { success: true, text: `${task.object.label} is in reach, ${cm(r)} cm from the arm.` };
  }
  return { success: true, text: "Done and still for 1 s." };
}
