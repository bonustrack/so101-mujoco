// The Jev and Flow sections: key and model, the goal, Run and Stop, a live log of every turn, and what the calls cost.
import { BASE_COMMANDS, question, runAgent, type AgentEvent, type ChoiceAnswer, type JevResponse } from "./agent";
import { SUPPORTED, TASKS, type Plan } from "./plan";
import { PRICE_NOTE, askJev, jevCost, listModels } from "./jev";
import type { Obj, Robot } from "./robot";

const KEY = "jev-api-key";
const MODEL = "jev-model";
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const round = (n: number) => n.toFixed(2);
const count = (n: number) => n.toLocaleString("en-US");
const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumSignificantDigits: 2 });

// Jev calls, tokens and dollars, for one run or the whole visit. Calls on a model with no published price stay unpriced.
type Tally = { calls: number; input: number; output: number; usd: number; unpriced: number };
const tally = (): Tally => ({ calls: 0, input: 0, output: 0, usd: 0, unpriced: 0 });
const price = (t: Tally) =>
  t.unpriced === 0 ? usd.format(t.usd) : t.unpriced === t.calls ? "cost unknown" : `${usd.format(t.usd)} + ${t.unpriced} unpriced`;
const spent = (t: Tally) => `${t.calls} ${t.calls === 1 ? "call" : "calls"} · ${count(t.input)} in / ${count(t.output)} out · ${price(t)}`;

// Taking one object, Jev's target is the one the goal names, else the selected object, else the first box.
export function setupAgent(robot: Robot, selected: () => Obj | null, lock: (locked: boolean) => void) {
  const key = $<HTMLInputElement>("jev-key");
  const model = $<HTMLSelectElement>("jev-model");
  const goal = $<HTMLTextAreaElement>("goal");
  const run = $<HTMLButtonElement>("run");
  const stopButton = $<HTMLButtonElement>("stop");
  const randomize = $<HTMLButtonElement>("randomize");
  const status = $("status");
  const flow = $("flow");
  const stages = [...$("diagram").querySelectorAll<HTMLElement>("li")];
  $("question").textContent = JSON.stringify(
    { read_the_goal: { task: TASKS }, take_one_object: question("take"), put_one_on_another: question("place") },
    null,
    2,
  );
  const planBox = $("plan");
  const cost = $("cost");
  const costShort = $("cost-short"); // the session total, in view when Flow is folded
  $("price").textContent = `Cost uses TypeSafe's published price. ${PRICE_NOTE}. Other models show their tokens and "cost unknown".`;
  const sessionCost = tally();
  let runCost = tally();
  const charge = (model: string, usage: JevResponse["usage"]) => {
    const dollars = jevCost(model, usage);
    for (const t of [runCost, sessionCost]) {
      t.calls++;
      t.input += usage?.input_tokens ?? 0;
      t.output += usage?.output_tokens ?? 0;
      if (dollars === null) t.unpriced++;
      else t.usd += dollars;
    }
    showCost();
    return dollars;
  };
  const showCost = () => {
    cost.innerHTML = `<span>Run</span>${spent(runCost)}<br><span>Session</span>${spent(sessionCost)}`;
    costShort.textContent = price(sessionCost);
  };

  let controller: AbortController | null = null;
  // The log follows new turns only while the reader stays at its bottom.
  let follow = true;
  flow.onscroll = () => (follow = flow.scrollHeight - flow.scrollTop - flow.clientHeight < 40);
  const target = () => robot.pickTarget(selected());
  // An empty floor still lets Jev drive the base.
  const idle = () => {
    run.disabled = controller !== null || !key.value.trim();
    stopButton.disabled = controller === null;
    randomize.disabled = !target();
  };
  const hint = () => {
    if (controller) return;
    status.textContent = key.value.trim() ? `Ready. Press Run. ${SUPPORTED}` : `Paste a key, then press Run. ${SUPPORTED}`;
  };
  // Driving the base is physics too.
  const highlight = (stage: string | null) => stages.forEach((li) => li.classList.toggle("on", li.dataset.stage === (stage === "drive" ? "physics" : stage)));

  // Fill the model list from GET /v1/models once a key is present; keep the saved choice.
  let asked = "";
  async function loadModels() {
    const value = key.value.trim();
    if (!value || value === asked) return;
    asked = value;
    try {
      const names = await listModels(value);
      const saved = localStorage.getItem(MODEL) ?? "jev-latest";
      model.replaceChildren(...names.map((name) => new Option(name, name, false, name === saved)));
    } catch (error) {
      if (!controller) status.textContent = error instanceof Error ? error.message : String(error);
    }
  }

  key.value = localStorage.getItem(KEY) ?? "";
  model.value = localStorage.getItem(MODEL) ?? "jev-latest";
  key.oninput = () => {
    const value = key.value.trim();
    if (value) localStorage.setItem(KEY, value);
    else localStorage.removeItem(KEY);
    idle();
    hint();
  };
  key.onchange = loadModels;
  $("jev-clear").onclick = () => {
    key.value = "";
    asked = "";
    localStorage.removeItem(KEY);
    idle();
    hint();
  };
  model.onchange = () => localStorage.setItem(MODEL, model.value);
  randomize.onclick = () => {
    const next = controller ? robot.focus : target();
    if (next?.active) robot.randomize(next);
  };
  stopButton.onclick = () => stop();
  loadModels();
  idle();
  hint();

  function stop() {
    controller?.abort();
    robot.cancel();
  }

  run.onclick = async () => {
    const apiKey = key.value.trim();
    const next = target();
    if (!apiKey || controller) return;
    // The selection is read once, at the start: selecting another object mid-run changes nothing.
    const chosen = selected();
    robot.focus = next;
    controller = new AbortController();
    lock(true);
    idle();
    flow.replaceChildren();
    planBox.hidden = true;
    let turns = 0;
    let turn = 0;
    let task = "";
    follow = true;
    runCost = tally();
    showCost();
    $<HTMLDetailsElement>("how").open = false; // room for the log; the diagram stays in view
    highlight("goal");
    let current: HTMLElement = flow;
    const add = (html: string, className = "") => {
      const row = document.createElement(current === flow ? "li" : "div");
      row.className = `step ${className}`.trim();
      row.innerHTML = html;
      current.append(row);
      if (follow) flow.scrollTop = flow.scrollHeight;
      return row;
    };
    // Jev's answer: its choice, confidence, time, tokens and cost, and its top 3 probabilities.
    const decision = (label: string, answer: ChoiceAnswer, model: string, ms: number, usage: JevResponse["usage"], note = "") => {
      const top = Object.entries(answer.probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3);
      const bars = top
        .map(
          ([name, p]) =>
            `<div class="bar-row" title="${escape(name)}: probability ${round(p)}"><span>${escape(name)}</span><i style="width:${Math.max(p * 100, 1)}%"></i><b>${round(p)}</b></div>`,
        )
        .join("");
      const dollars = charge(model, usage);
      const tokens = usage ? `${count(usage.input_tokens)} in + ${count(usage.output_tokens)} out tokens` : "no token count";
      return (
        `<p>${label} <strong>${escape(answer.choice)}</strong><span class="meta">confidence ${round(answer.confidence)} · ${Math.round(ms)} ms · ${escape(model)}</span><span class="meta">${tokens} · ${dollars === null ? "cost unknown" : usd.format(dollars)}</span></p><div class="bars" aria-label="Jev's top probabilities">${bars}</div>` +
        note
      );
    };
    const sent = (what: string, state: unknown) => {
      const row = add(`<details><summary>Sent to Jev: ${what}</summary><pre></pre></details>`);
      row.querySelector("pre")!.textContent = JSON.stringify(state, null, 2);
    };
    // The plan: the task as read, then each step, ticked once code has checked it.
    // A drive step leaves the plan once the base is there: it stays on the list, ticked, before the step it led to.
    const drove: { text: string; before: string }[] = [];
    const showPlan = (plan: Plan) => {
      plan.steps.forEach((s, i) => {
        const before = plan.steps[i + 1]?.text;
        if (s.drive && before && !drove.some((d) => d.text === s.text && d.before === before)) drove.push({ text: s.text, before });
      });
      const shown = [...plan.steps];
      for (const d of drove) {
        const at = shown.findIndex((s) => s.text === d.before);
        if (at >= 0 && !shown.some((s) => s.text === d.text)) shown.splice(at, 0, { text: d.text, done: true });
      }
      const now = shown.findIndex((s) => !s.done);
      const steps = shown.map((s, i) => `<li class="${s.done ? "done" : i === now ? "now" : ""}">${escape(s.text)}</li>`).join("");
      planBox.querySelector("ol")!.innerHTML = steps;
      planBox.hidden = false;
    };
    const onEvent = (event: AgentEvent) => {
      switch (event.type) {
        case "stage":
          highlight(event.stage);
          status.textContent = turn ? `Turn ${turn} of ${turns}: ${STAGE_TEXT[event.stage]}` : `Reading the goal: ${STAGE_TEXT[event.stage]}`;
          break;
        case "task": {
          task = event.name;
          turns = event.maxTurns;
          current = document.createElement("li");
          current.innerHTML = `<h3>Goal</h3>`;
          flow.append(current);
          sent("goal, objects, selection", event.request.state);
          const unsure = event.confidence < 0.6;
          add(
            decision("Jev read it as", event.answers.task, event.model, event.ms, event.usage, unsure ? `<p class="meta">Low confidence ${round(event.confidence)}: Jev is not sure about this reading. Check it in the plan above.</p>` : ""),
            unsure ? "unsure" : "",
          );
          add(`<p><span class="tag">Task</span>${escape(event.text)}</p>`, event.ok ? "" : "error");
          const head = planBox.querySelector("p")!;
          head.innerHTML = `<strong>${escape(event.text)}</strong>` + (unsure ? ` <span class="unsure">Jev is not sure (confidence ${round(event.confidence)}).</span>` : "");
          planBox.querySelector("ol")!.replaceChildren();
          planBox.hidden = false;
          break;
        }
        case "plan":
          turns = event.maxTurns;
          showPlan(event.plan);
          break;
        case "turn": {
          turn = event.turn;
          current = document.createElement("li");
          current.innerHTML = `<h3>Turn ${event.turn} <span class="meta-inline">${escape(event.subgoal)}</span></h3>`;
          flow.append(current);
          sent("goal, observation, history", event.request.state);
          break;
        }
        case "skill": {
          // Code drives the base on its own: a path on a map of the floor, no Jev call.
          turn = event.turn;
          current = document.createElement("li");
          current.innerHTML = `<h3>Turn ${event.turn} <span class="meta-inline">${escape(event.subgoal)}, planned in code</span></h3>`;
          flow.append(current);
          break;
        }
        case "decision": {
          highlight("command");
          const unsure = event.answer.confidence < 0.5;
          add(
            decision("Jev chose", event.answer, event.model, event.ms, event.usage, unsure ? `<p class="meta">Low confidence: Jev is not sure, the top choice runs anyway.</p>` : ""),
            unsure ? "unsure" : "",
          );
          break;
        }
        case "result":
          add(`<p><span class="tag">${BASE_COMMANDS.includes(event.command) ? "Base" : "Arm"}</span>${escape(event.text)}</p>`);
          break;
        case "check":
          add(`<p><span class="tag">Check</span>${escape(event.text)}</p>`, event.success ? "ok" : "");
          break;
        case "end": {
          current = flow;
          const word = event.outcome === "success" ? (DONE_TEXT[task] ?? "Done.") : END_TEXT[event.outcome];
          add(`<p><strong>${word}</strong> ${escape(event.text)}</p>`, `end ${event.outcome}`);
          status.textContent = `${word} ${event.text}`;
          break;
        }
      }
    };
    await runAgent({
      robot,
      goal: goal.value.trim() || "Take the box",
      model: model.value || "jev-latest",
      decide: (request, signal) => askJev(apiKey, request, signal),
      signal: controller.signal,
      onEvent,
      selected: chosen,
    });
    lock(false); // before clearing the controller, so the end message stays up
    controller = null;
    highlight(null);
    idle();
  };

  return {
    stop,
    // The scene changed: the target, Run and the hint may too.
    refresh() {
      idle();
      hint();
    },
  };
}

const STAGE_TEXT: Record<string, string> = {
  observe: "reading the scene",
  jev: "waiting for Jev",
  ik: "solving the joint angles",
  physics: "moving the arm",
  drive: "driving the base",
  check: "checking the result",
};
const DONE_TEXT: Record<string, string> = { take: "Lifted.", stack_boxes: "Stacked.", stack_all: "Stacked.", put_on: "Placed.", drive: "Moved.", drive_to: "Arrived." };
const END_TEXT: Record<Extract<AgentEvent, { type: "end" }>["outcome"], string> = {
  success: "Done.",
  done: "Jev stopped.",
  stopped: "Stopped.",
  turns: "Out of turns.",
  stuck: "Stuck.",
  budget: "Out of time.",
  error: "Error.",
};

function escape(text: string) {
  return text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}
