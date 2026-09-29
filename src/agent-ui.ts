// The left panel: Jev key and model, the goal, Run and Stop, and a live log of every turn.
import { MAX_TURNS, QUESTIONS, runAgent, type AgentEvent } from "./agent";
import { askJev, listModels } from "./jev";
import type { Obj, Robot } from "./robot";

const KEY = "jev-api-key";
const MODEL = "jev-model";
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const round = (n: number) => n.toFixed(2);

// Jev's target is the selected object, else the first box, else the first object.
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
  $("question").textContent = JSON.stringify(QUESTIONS, null, 2);

  let controller: AbortController | null = null;
  // The log follows new turns only while the reader stays at its bottom.
  let follow = true;
  flow.onscroll = () => (follow = flow.scrollHeight - flow.scrollTop - flow.clientHeight < 40);
  const target = () => robot.pickTarget(selected());
  const idle = () => {
    const next = target();
    run.disabled = controller !== null || !key.value.trim() || !next || !!robot.tooBig(next);
    stopButton.disabled = controller === null;
    randomize.disabled = !next;
  };
  const hint = () => {
    if (controller) return;
    const next = target();
    const why = next && robot.tooBig(next);
    status.textContent = !next
      ? "Add a box or a ball, then press Run."
      : why
        ? `${next.label} is too big to grab: ${why}. Resize it or select another object.`
        : key.value.trim()
          ? `Ready. Press Run: Jev takes ${next.label}.`
          : `Paste a key, then press Run. Jev takes ${next.label}.`;
  };
  const highlight = (stage: string | null) => stages.forEach((li) => li.classList.toggle("on", li.dataset.stage === stage));

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
    if (!apiKey || controller || !next) return;
    // The target stays the same for the whole run, whatever gets selected meanwhile.
    robot.focus = next;
    if (/^Take the (box|ball)$/.test(goal.value.trim())) goal.value = `Take the ${next.kind}`;
    controller = new AbortController();
    lock(true);
    idle();
    flow.replaceChildren();
    follow = true;
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
    const onEvent = (event: AgentEvent) => {
      switch (event.type) {
        case "stage":
          highlight(event.stage);
          status.textContent = `Turn ${flow.children.length} of ${MAX_TURNS}: ${STAGE_TEXT[event.stage]}`;
          break;
        case "turn": {
          current = document.createElement("li");
          current.innerHTML = `<h3>Turn ${event.turn}</h3>`;
          flow.append(current);
          const sent = add(`<details><summary>Sent to Jev: goal, observation, history</summary><pre></pre></details>`);
          sent.querySelector("pre")!.textContent = JSON.stringify(event.request.state, null, 2);
          break;
        }
        case "decision": {
          highlight("command");
          const top = Object.entries(event.answer.probabilities)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3);
          const bars = top
            .map(
              ([name, p]) =>
                `<div class="bar-row" title="${name}: probability ${round(p)}"><span>${name}</span><i style="width:${Math.max(p * 100, 1)}%"></i><b>${round(p)}</b></div>`,
            )
            .join("");
          const unsure = event.answer.confidence < 0.5;
          add(
            `<p>Jev chose <strong>${event.answer.choice}</strong><span class="meta">confidence ${round(event.answer.confidence)} · ${Math.round(event.ms)} ms · ${escape(event.model)}</span></p><div class="bars" aria-label="Jev's top probabilities">${bars}</div>` +
              (unsure ? `<p class="meta">Low confidence: Jev is not sure, the top choice runs anyway.</p>` : ""),
            unsure ? "unsure" : "",
          );
          break;
        }
        case "result":
          add(`<p><span class="tag">Arm</span>${escape(event.text)}</p>`);
          break;
        case "check":
          add(`<p><span class="tag">Check</span>${escape(event.text)}</p>`, event.success ? "ok" : "");
          break;
        case "end":
          current = flow;
          add(`<p><strong>${END_TEXT[event.outcome]}</strong> ${escape(event.text)}</p>`, `end ${event.outcome}`);
          status.textContent = `${END_TEXT[event.outcome]} ${event.text}`;
          break;
      }
    };
    await runAgent({
      robot,
      goal: goal.value.trim() || "Take the box",
      model: model.value || "jev-latest",
      decide: (request, signal) => askJev(apiKey, request, signal),
      signal: controller.signal,
      onEvent,
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
  check: "checking the target",
};
const END_TEXT: Record<Extract<AgentEvent, { type: "end" }>["outcome"], string> = {
  success: "Lifted.",
  done: "Jev stopped.",
  stopped: "Stopped.",
  turns: "Out of turns.",
  error: "Error.",
};

function escape(text: string) {
  return text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}
