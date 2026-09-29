// From the goal text to an ordered list of subgoals. Jev reads the goal once and maps it to a task
// with a few closed-set parameters (the "function calling" pattern in the TypeSafe docs). Code then
// expands the task into pick-and-place subgoals, and expands it again from the physics state after
// every turn: a box that fell off the stack simply shows up as not done again.
import type { Obj, Robot } from "./robot";

export type ChoiceQuestion = { type: "choice"; instructions: string; criteria: Record<string, string> };

// Where a held object goes: the top of an object, or a spot on the table.
export type Target = { label: string; x: number; y: number; top: number; yaw: number | null; object: Obj | null };
export type Subgoal = { kind: "take"; object: Obj; text: string } | { kind: "place"; object: Obj; target: Target; text: string };
export type Step = { text: string; done: boolean };
export type Plan = { steps: Step[]; current: Subgoal | null };

export type Task =
  | { kind: "take"; object: Obj }
  | { kind: "stack"; objects: Obj[]; spot: Spot; skipped: string[] } // bottom to top
  | { kind: "put_on"; object: Obj; onto: Obj };
// Where a tower stands: where its base is, or a spot on the table the base moves to first.
export type Spot = "here" | { x: number; y: number; label: string };
// With the fingers pointing down, the SO-101 reaches highest 20 to 24 cm from its base: about 9.5 cm up
// at the jaw tips, against 8 cm at 16 cm out. A tower is built in that ring.
export const BEST = 0.22;
const RING = [0.18, 0.26];
const ROOM = 0.04; // free table around a tower's base
// Measured with scripted runs: the arm sets an object down reliably on a top up to 10 cm high, not above.
export const MAX_TOP = 0.1;
const height = (o: Obj) => 2 * (o.kind === "ball" ? o.size[0] : o.size[2]);

export const TASKS = {
  take: 'Pick up one object and hold it up. Right for goals like "take the box", "lift the ball", "grab Box 2" or "pick it up".',
  stack_boxes:
    'Stack all the boxes into one tower. Right for goals like "stack the boxes", "stack all the boxes together", "pile up the boxes" or "make a tower with the boxes". Balls stay where they are.',
  stack_all: 'Stack every object, boxes and balls, into one tower. Right only when `goal` asks to stack everything or all the objects, balls included.',
  put_on: 'Put one object on top of one other object. Right for goals like "put Box 2 on Box 1", "put the ball on the big box" or "place the small box on the other one".',
  other:
    "Anything else: lining objects up, clearing the table, sorting, pushing, throwing, moving the arm without an object, or a goal that is not about these objects.",
};
export type TaskName = keyof typeof TASKS;
export const SUPPORTED = 'Take an object, stack the boxes, stack everything, or put one object on another ("put Box 2 on Box 1").';

// Size words computed in code, since Jev does not compare numbers well.
const area = (o: Obj) => (o.kind === "ball" ? Math.PI * o.size[0] ** 2 : 4 * o.size[0] * o.size[1]);
const bigFirst = (a: Obj, b: Obj) => area(b) - area(a) || b.size[2] - a.size[2] || a.order - b.order;
const sizeText = (o: Obj) =>
  o.kind === "ball" ? `${Math.round(o.size[0] * 200)} cm across` : o.size.map((s) => Math.round(s * 200)).join(" x ") + " cm";

function describe(robot: Robot, o: Obj) {
  const same = robot.active().filter((other) => other.kind === o.kind).sort(bigFirst);
  const rank = same.length < 2 ? `the only ${o.kind}` : same[0] === o ? `the largest ${o.kind}` : same.at(-1) === o ? `the smallest ${o.kind}` : `a middle-sized ${o.kind}`;
  const { pos } = robot.object(o);
  const side = pos[1] > 0.04 ? "on the arm's left" : pos[1] < -0.04 ? "on the arm's right" : "in front of the arm";
  return `${o.label}: a ${o.kind}, ${sizeText(o)}, ${rank}, ${side}`;
}

// The one parse request: the task and every parameter any task needs, in one call.
export function parseRequest(robot: Robot, goal: string, selected: Obj | null) {
  const objects = robot.active();
  const labels = Object.fromEntries(objects.map((o) => [o.label, describe(robot, o)]));
  const state = { goal, objects: Object.values(labels), selected: selected?.label ?? "none" };
  const questions: Record<string, ChoiceQuestion> = {
    task: {
      type: "choice",
      instructions: "`goal` is what a person asked a robot arm to do with the objects on a table, listed in `objects`. Which task does `goal` ask for?",
      criteria: TASKS,
    },
    object: {
      type: "choice",
      instructions:
        "Which object in `objects` does `goal` name as the one to pick up or to move? Match the name, the type (box or ball), the size words and the side. When `goal` does not single one out, the object named in `selected`, else the first box.",
      criteria: labels,
    },
    onto: {
      type: "choice",
      instructions: "Which object in `objects` does `goal` name as the one to put something on? When `goal` names none, the largest box.",
      criteria: labels,
    },
    order: {
      type: "choice",
      instructions: "In which order should the objects be stacked?",
      criteria: {
        largest_at_bottom: "Largest at the bottom, smallest on top. Right when `goal` says nothing about the order, or asks for the biggest at the bottom.",
        smallest_at_bottom: "Smallest at the bottom, largest on top. Right only when `goal` asks for the smallest at the bottom or an upside-down tower.",
      },
    },
    spot: {
      type: "choice",
      instructions: "Where should the tower stand?",
      criteria: {
        here: "Where the bottom object already stands. Right when `goal` names no place.",
        front: "In the middle, right in front of the arm. Right only when `goal` asks for the middle, the centre or the front.",
      },
    },
  };
  return { state, questions };
}

export type ParseAnswers = Record<string, { choice: string; confidence: number }>;
// The questions that matter for each task: their lowest confidence is the reading's confidence.
const USES: Record<TaskName, string[]> = { take: ["task", "object"], stack_boxes: ["task", "order", "spot"], stack_all: ["task", "order", "spot"], put_on: ["task", "object", "onto"], other: ["task"] };

// Jev's answers to a task, checked in code. Returns an error text when the task cannot run.
export function readTask(robot: Robot, answers: ParseAnswers, selected: Obj | null): { task: Task | null; name: TaskName; confidence: number; text: string } {
  const name = (answers.task?.choice ?? "other") as TaskName;
  const confidence = Math.min(...USES[name].map((q) => answers[q]?.confidence ?? 0));
  const find = (label?: string) => robot.active().find((o) => o.label === label) ?? null;
  const fail = (text: string) => ({ task: null, name, confidence, text });
  if (name === "take") {
    // The selected object wins, as before; Jev's pick only counts when nothing is selected.
    const object = selected?.active ? selected : (find(answers.object?.choice) ?? robot.pickTarget(null));
    if (!object) return fail("No object to take. Add a box or a ball.");
    const why = robot.tooBig(object);
    if (why) return fail(`${object.label} is too big to grab: ${why}.`);
    return { task: { kind: "take", object }, name, confidence, text: `Take ${object.label} and hold it up.` };
  }
  if (name === "put_on") {
    const object = find(answers.object?.choice);
    const onto = find(answers.onto?.choice);
    if (!object || !onto) return fail("Jev did not name both objects.");
    if (object === onto) return fail(`Jev read both objects as ${object.label}. Name them, like "put Box 2 on Box 1".`);
    if (onto.kind === "ball") return fail(`Nothing stays on a ball, so ${object.label} cannot go on ${onto.label}.`);
    if (robot.object(onto).top > MAX_TOP) return fail(`The top of ${onto.label} is over ${MAX_TOP * 100} cm high, more than the arm reaches.`);
    const why = robot.tooBig(object);
    if (why) return fail(`${object.label} is too big to grab: ${why}.`);
    return { task: { kind: "put_on", object, onto }, name, confidence, text: `Put ${object.label} on ${onto.label}.` };
  }
  if (name === "stack_boxes" || name === "stack_all") {
    const skipped: string[] = [];
    const boxes = robot.active().filter((o) => o.kind === "box").sort(bigFirst);
    if (answers.order?.choice === "smallest_at_bottom") boxes.reverse();
    // A box the jaws cannot take can only be the bottom one.
    const fixed = boxes.filter((o) => robot.tooBig(o));
    for (const o of fixed.slice(1)) skipped.push(`${o.label} stays: too big to grab.`);
    const objects = fixed.length ? [fixed[0], ...boxes.filter((o) => !robot.tooBig(o))] : boxes;
    // Nothing stays on a ball, so only one ball can go, on top.
    if (name === "stack_all") {
      const [ball, ...rest] = robot.active().filter((o) => o.kind === "ball");
      if (ball) objects.push(ball);
      for (const o of rest) skipped.push(`${o.label} stays: nothing stays on a ball, so only one ball goes on top.`);
    }
    if (!boxes.length) return fail("No box to build on. Add a box.");
    // The arm sets objects down on a tower up to about 10 cm high: the rest stay out.
    let top = 0;
    const fit = objects.findIndex((o) => {
      const over = top > MAX_TOP + 1e-9;
      top += height(o);
      return over;
    });
    if (fit > 0) {
      const out = objects.splice(fit).map((o) => o.label);
      skipped.push(`${out.join(", ")} ${out.length === 1 ? "stays" : "stay"} out: the tower would be over ${MAX_TOP * 100} cm high, more than the arm reaches.`);
    }
    if (objects.length < 2) return fail(`Only ${objects[0].label} can go in a tower: nothing to stack.${skipped.length ? " " + skipped.join(" ") : ""}`);
    const spot = towerSpot(robot, objects[0], answers.spot?.choice === "front", objects.length);
    const where = spot === "here" ? `where ${objects[0].label} stands` : `at ${spot.label}`;
    const order = objects.map((o) => o.label).join(", ");
    return {
      task: { kind: "stack", objects, spot, skipped },
      name,
      confidence,
      text: `Stack ${objects.length} objects ${where}, bottom to top: ${order}.${skipped.length ? " " + skipped.join(" ") : ""}`,
    };
  }
  return fail(`Not something this arm can do yet. It can: ${SUPPORTED}`);
}

// The base stays unless the goal asks for the front, or the tower needs the arm's best reach and the base is
// outside that ring. The spot is the free place in the ring closest to where it is asked for.
function towerSpot(robot: Robot, base: Obj, front: boolean, count: number): Spot {
  const { pos } = robot.object(base);
  const r = Math.hypot(pos[0], pos[1]);
  const room = robot.roomFinder(base, false); // the arm moves away once it works
  // A tower needs room around it, or picking up its neighbours knocks it over.
  const ok = (x: number, y: number) => room(x, y) >= ROOM;
  if (!front && (count < 3 || (r >= RING[0] && r <= RING[1])) && ok(pos[0], pos[1])) return "here";
  if (robot.tooBig(base)) return "here";
  const want = front ? 0 : Math.atan2(pos[1], pos[0]);
  for (const need of [ROOM, 0.01])
    for (let i = 0; i <= 16; i++) {
      const a = want + (i % 2 ? 1 : -1) * Math.ceil(i / 2) * 0.12;
      const x = BEST * Math.cos(a);
      const y = BEST * Math.sin(a);
      if (room(x, y) >= need) return { x, y, label: front ? "the spot in front of the arm" : "a clear spot where the arm reaches highest" };
    }
  return "here";
}

// How many objects a task touches: the turn budget grows with it.
export const taskSize = (task: Task) => (task.kind === "take" ? 1 : task.kind === "put_on" ? 2 : task.objects.length);

// ---- Scene facts for placing, all from the physics state ----

// Resting on `below`: upright, its lowest point on the other's top, its centre over it, touching it,
// and not gripped by both jaws.
export function restingOn(robot: Robot, o: Obj, below: Obj) {
  const a = robot.object(o);
  const b = robot.object(below);
  const touch = robot.touching(o);
  const reach = below.kind === "ball" ? 0.006 : Math.max(0.006, 0.7 * Math.min(below.size[0], below.size[1]));
  return (
    a.upright &&
    Math.abs(a.bottom - b.top) < 0.006 &&
    Math.hypot(a.pos[0] - b.pos[0], a.pos[1] - b.pos[1]) < reach &&
    touch.others.has(below) &&
    !(touch.fixed && touch.moving)
  );
}

// On the table at a spot: upright, on the table, its centre within 1.5 cm of the spot.
function onSpot(robot: Robot, o: Obj, spot: { x: number; y: number }) {
  const a = robot.object(o);
  const touch = robot.touching(o);
  return a.upright && a.bottom < 0.004 && touch.table && Math.hypot(a.pos[0] - spot.x, a.pos[1] - spot.y) < 0.015 && !(touch.fixed && touch.moving);
}

// The jaws are clear of `o`: nothing of the arm touches it and the jaw tips are not around it.
export function clear(robot: Robot, o: Obj) {
  const a = robot.object(o);
  const tip = robot.tcp();
  const around = Math.hypot(tip[0] - a.pos[0], tip[1] - a.pos[1]) < robot.footprint(o) + 0.01 && tip[2] < a.top + 0.005;
  const t = robot.touching(o);
  return !t.fixed && !t.moving && !t.arm && !around;
}

// The objects stacked on `base` from the bottom up, among `among`.
export function tower(robot: Robot, base: Obj, among: Obj[]) {
  const chain = [base];
  for (;;) {
    const top = chain.at(-1)!;
    const next = among.find((o) => !chain.includes(o) && restingOn(robot, o, top));
    if (!next) return chain;
    chain.push(next);
  }
}

export const onTop = (robot: Robot, o: Obj): Target => {
  const { pos, top, yaw } = robot.object(o);
  return { label: o.label, x: pos[0], y: pos[1], top, yaw: o.kind === "box" ? yaw : null, object: o };
};

// Where the robot stands on a task right now: every step with done or not, and the first open one.
export function plan(robot: Robot, task: Task): Plan {
  if (task.kind === "take") {
    const current: Subgoal = { kind: "take", object: task.object, text: `Take ${task.object.label}` };
    return { steps: [{ text: current.text, done: robot.focus === task.object && robot.held() }], current };
  }
  if (task.kind === "put_on") {
    const done = restingOn(robot, task.object, task.onto) && clear(robot, task.object);
    const text = `Put ${task.object.label} on ${task.onto.label}`;
    return { steps: [{ text, done }], current: done ? null : { kind: "place", object: task.object, target: onTop(robot, task.onto), text } };
  }
  const [base, ...rest] = task.objects;
  const steps: Step[] = [];
  let current: Subgoal | null = null;
  if (task.spot !== "here") {
    const text = `Move ${base.label} to ${task.spot.label}`;
    const done = onSpot(robot, base, task.spot) && clear(robot, base);
    steps.push({ text, done });
    if (!done) current = { kind: "place", object: base, target: { ...task.spot, top: 0, yaw: null, object: null }, text };
  } else steps.push({ text: `${base.label} stays as the base`, done: true });
  const chain = tower(robot, base, task.objects);
  const top = chain.at(-1)!;
  for (const o of rest) {
    const text = `Put ${o.label} on the tower`;
    const done = chain.includes(o) && clear(robot, o);
    steps.push({ text, done });
    // Already set down but the jaws are still around it: its target stays the object below it.
    const below = chain.includes(o) ? chain[chain.indexOf(o) - 1] : top;
    if (!done && !current) current = { kind: "place", object: o, target: onTop(robot, below), text: `Put ${o.label} on ${below === base ? "" : "the tower, on "}${below.label}` };
  }
  return { steps, current };
}
