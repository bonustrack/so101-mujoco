// The benchmark gate: `bun run bench`. Random scenes in every family (scripts/bench-core.ts), run headless
// with the page's own code, split over the CPU cores. Prints a table per family and the numbers against the
// targets, writes one JSON line per run, and fails when the numbers fall under the gate.
//
//   bun run bench                                  # 20 seeds per family, the rules decider
//   bun run bench --seeds 4 --mode jev             # the real Jev: reads the key from ~/.secrets/jev-key
//   bun run bench --families blocked_take,drive_to --seeds 10
//   bun run bench --out /tmp/so101-bench/run.jsonl --no-gate
import loadMujoco from "@mujoco/mujoco";
import { appendFileSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { availableParallelism, homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import type { Decide, JevResponse } from "../src/agent";
import { createRobot, loadModel } from "../src/robot";
import jevFunction from "../netlify/functions/jev";
import { createBench, gate, report, summarize, type Mode, type Record } from "./bench-core";

const arg = (name: string, fallback: string) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const MODE = arg("mode", "rules") as Mode;
const SEEDS = Number(arg("seeds", "20"));
const SEED0 = Number(arg("seed", "1000"));
const FAMILIES = arg("families", "");
const JOBS = Number(arg("jobs", String(Math.min(availableParallelism(), 4))));
const OUT = arg("out", join(tmpdir(), "so101-bench", `${MODE}-${Date.now()}.jsonl`));
const SHARD = arg("shard", ""); // "k/n": this process runs every n-th run from k, and writes JSON lines only

async function worker(k: number, n: number, out: string) {
  const dir = new URL("../model/", import.meta.url).pathname;
  const files = ["scene_web.xml", "so101.xml", ...readdirSync(dir + "assets").map((f) => "assets/" + f)].map((name) => [name, readFileSync(dir + name)] as [string, Uint8Array]);
  const mujoco = await loadMujoco();
  const robot = createRobot(mujoco, loadModel(mujoco, files));
  const bench = createBench(robot);
  const families = FAMILIES ? FAMILIES.split(",") : bench.families;
  const jev = MODE === "jev" ? jevDecider() : undefined;
  const total = families.length * SEEDS;
  for (let i = k; i < total; i += n) {
    const family = families[i % families.length];
    const record = await bench.run(family, SEED0 + i, MODE, jev);
    appendFileSync(out, JSON.stringify(record) + "\n");
    console.log(`${record.clean ? "clean" : record.success ? "dirty" : "FAIL "} ${family} #${record.seed} "${record.goal}" ${record.outcome} ${record.taskSeconds}s ${record.calls} calls | ${record.events.map((e) => e.kind).join(",")}`);
  }
}

// The real Jev through the page's own Netlify Function, with retries on busy. The key comes from a file and
// is never printed.
function jevDecider(): Decide {
  const key = readFileSync(process.env.JEV_KEY_FILE ?? join(homedir(), ".secrets/jev-key"), "utf8").trim();
  return async (request) => {
    for (let attempt = 0; ; attempt++) {
      const response = await jevFunction(new Request("http://x/.netlify/functions/jev", { method: "POST", headers: { authorization: `Bearer ${key}` }, body: JSON.stringify(request) }));
      const text = await response.text();
      if (response.ok) return JSON.parse(text) as JevResponse;
      if (attempt < 4 && (response.status === 429 || response.status >= 500)) {
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
        continue;
      }
      throw new Error(`Jev HTTP ${response.status}: ${text.slice(0, 200)}`);
    }
  };
}

if (SHARD) {
  const [k, n] = SHARD.split("/").map(Number);
  await worker(k, n, OUT);
  process.exit(0);
}

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, "");
const started = performance.now();
const parts = Array.from({ length: JOBS }, (_, k) => `${OUT}.${k}`);
const children = parts.map((part, k) =>
  Bun.spawn([process.execPath, import.meta.path, ...process.argv.slice(2).filter((a, i, all) => a !== "--out" && all[i - 1] !== "--out"), "--out", part, "--shard", `${k}/${JOBS}`], {
    stdout: "inherit",
    stderr: "inherit",
  }),
);
const codes = await Promise.all(children.map((c) => c.exited));
const records: Record[] = [];
for (const part of parts) {
  const text = (() => {
    try {
      return readFileSync(part, "utf8");
    } catch {
      return "";
    }
  })();
  for (const line of text.split("\n")) if (line) records.push(JSON.parse(line));
  rmSync(part, { force: true });
}
records.sort((a, b) => a.seed - b.seed);
writeFileSync(OUT, records.map((r) => JSON.stringify(r)).join("\n") + "\n");
const summary = summarize(records);
console.log(`\n${MODE} decider, ${records.length} runs in ${((performance.now() - started) / 1000).toFixed(0)} s on ${JOBS} cores. Runs: ${OUT}\n`);
console.log(report(summary));
const fails = codes.some((c) => c !== 0) ? ["a worker crashed"] : process.argv.includes("--no-gate") ? [] : gate(summary);
console.log(fails.length ? `\nGate failed: ${fails.join("; ")}` : "\nGate passed");
process.exit(fails.length ? 1 : 0);
