# Plan: a smarter, faster SO-101 on wheels

Status: all phases built, safety follow-up passes both 300-scene release checks (2026-09-30). Exact results and remaining limits are in section 6.

## 1. What goes wrong today

Measured headless with the same code as the page (`scripts/bench.ts`). 12 scene families, random layouts and sizes, one task each. Two deciders:

- **Rules**: a perfect decider that follows the command criteria to the letter. It shows the limits of the code itself. 252 runs.
- **Real Jev** (`jev-latest`): 96 runs, 1,004 turn decisions.

"Clean" means success and nothing else pushed more than 2 cm, rolled over or knocked over.

| | Rules | Real Jev |
|---|---|---|
| Success | 79% | 78% |
| Clean success | 63% | 58% |
| Task time, median / p90 (motion + Jev) | 14.6 s / 39 s | 16.5 s / 50 s |
| Jev calls per task, median / p90 / max | 8 / 19 / 66 | 8 / 21 / 52 |

Clean success by family (rules / Jev):

| Family | Goal | Rules | Jev |
|---|---|---|---|
| blocked_take | take an object with a box on the straight line to it | 0% | 0% |
| forward_blocked | "go forward 50 cm", a box ahead | 0% | 0% |
| near_take_ball_selected | "take the ball", a box selected in the scene | 0% | 0% |
| spread_stack | stack 3 boxes spread 25 to 70 cm around | 52% | 38% |
| far_put | put a far object on a near box | 57% | 50% |
| turn_clutter | "turn around" with objects close by | 71% | 62% |
| near_put | put a box or a ball on a box, all in reach | 76% | 75% |
| drive_to | "go to Box 3" | 95% | 100% |
| far_take, near_take_ball, near_take_named | take one object | 100% | 100% |
| near_stack | stack 3 boxes in reach | 100% | 75% |

### The six failure types

1. **Driving into or over boxes: the main cause.** 51% of the runs that drive touch an object with the base (77 of 150 rules runs, 29 of 57 Jev runs). 34 runs rolled a box under the chassis or a wheel. 192 pushes, median 9 cm, up to 40 cm. It is the first cause in 68 of the 94 rules runs that are not clean (72%), and 28 of 40 for Jev.
   Why: `drive_to_object` turns on the spot and drives straight, with no map. A bump stops the base, but the next move ignores any object it already touches at the start (`before: bumping()`, src/robot.ts:349). So the second try pushes through or climbs over.
2. **Loops.** 23% of rules runs and 24% of Jev runs get stuck: they repeat a command with no progress (up to 12 times) or run out of turns. In 49 of the 54 rules runs with a repeat, a base contact came first, and 41 of them repeat a drive command. Most others follow a place miss (type 5). Nothing tracks progress, and the same observation gets the same command.
3. **Grasp failures.** Rare in themselves: 0 of 252 rules runs, and 30 of 30 scripted picks in reach. With Jev, 8 of 96 runs, all with a Jev step pick off the rules, like lowering the jaws while they were still closed (see type 4).
4. **Bad plans and wrong object.**
   - "Take the ball" with a box selected takes the box: 29 of 29 runs. Jev read "Ball 1" right 8 times of 8. Code overrides it: the selected object wins over the goal (src/plan.ts:166), and Run even rewrites "Take the ball" to "Take the box" (src/agent-ui.ts:132). Adding an object selects it, so this happens often on the page.
   - Jev read the goal right in 88 of 88 other runs.
   - Jev's turn picks differ from the rules in 69 of 1,004 turns (6.9%). Mostly `move_above_object` or `lower_to_object` while the jaws are still closed (52 times). That made 4 runs fail.
5. **Out of reach.** After a drive, "not in reach" only happened after a bump (0 cases without one). One real gap: a ball on a box fails 5 times of 13 ("Lowered, 2 cm above Box 1, not on it", then the same move again). In a scripted check, a 3 cm ball sat 1.6 cm above the jaw tips, so the fixed jaw hit the box first. Balls of 4 to 6 cm landed fine.
6. **Tipping.** The robot never tipped. Highest chassis tilt: 13°. Over 10° in 2 runs of 348, both while climbing a box. Objects knocked over: 26 of 252 rules runs, 20 of them by the base.

### Where the time goes

- Take in reach: 8.0 s. Five arm moves with fixed durations, 2.4 s of it fixed waits.
- Far take: 16 s (half driving at 12 cm/s with stops, half the pick). Stack of 3 in reach: 27 s. Stack of 3 spread out: 65 s (median, rules).
- Jev: 0.16 s per call direct, about 0.2 s through the live Netlify function (1.2 s when cold). 1,185 input tokens per call, about $0.00005.

Headroom, measured:

- Arm moves 3 times faster (durations x0.35, waits x0.2): 30 of 30 picks, 2.7 s per pick instead of 8.1 s.
- Base: 35 cm/s straight works (tilt under 1.2°, stops within 0.5 cm). Today 12 cm/s. Turning on the spot reaches about 90°/s.
- Timestep 0.01 instead of 0.005: 30 of 30 picks, half the CPU.

## 2. Design: Jev decides what, code decides how

Today Jev picks every small step. The rules decider proves the right step follows from the observation in code, so per-step calls add time, cost and a 6.9% error rate, and no intelligence. The fix for type 1 and 2 is in code anyway.

- **Jev**: reads the goal (1 call). It chooses again only when there is a real choice: after code has tried twice and failed (try another way, skip this object, stop).
- **Code**: the map, the path, the drive, the pick, the place, the checks. Deterministic and tested by the bench.
- **One world interface**: the sim fills it with true poses now, a camera can fill it later.

```
 goal text
    |
 [Jev] read the goal ................ 1 call: task, objects, target
    |
 [Planner] task -> skills ........... planned again from the world after each skill
    |   e.g. GoTo Ball 1 > Pick Ball 1 > GoTo Box 1 > Place on Box 1
    |
 [Skills]  GoTo(pose)       Pick(object)       Place(object, target)
           map + A* path    grasp choice       place choice
           path follower    guarded moves      release, retreat
    |                 ^
    |                 |  World: robot pose, objects (sim now, camera later)
 [Monitor] progress, contacts, time
    |   no progress for 2 s, an unplanned contact, or a failed check
    v
 recovery: plan again > back off, other side or other yaw
           > [Jev] choose: retry another way, skip, stop > stop and say why
```

### Navigation (fixes types 1 and 2)

- **Map**: each object is a circle on the floor (its footprint), from the world state. Container walls too. A 2 cm grid in a 2.5 m window around the robot.
- **Robot shape**: the real rectangle, 21 x 19 cm with the wheels, plus 2 cm margin, plus what the jaws hold.
- **Path**: A* over position and heading (16 headings). Moves: forward, back, turn on the spot by one heading. A turn is allowed only where the swept rectangle is free. So it finds "back up, then turn" by itself. A few ms per plan.
- **Goal pose**: try 16 sides around the object at 20 cm. Keep the ones that are free and where the open jaws clear the neighbours. Take the cheapest path.
- **Follow**: continuous speed on straight parts, up to 30 cm/s, turns up to 90°/s. No more 10 cm steps. Curve smoothing (pure pursuit) only if the bench shows the stops cost much.
- **Safety net**: any contact with an object that is not the target stops the base. The map updates, and it plans again. It never pushes on.
- **Plain moves** ("go forward 50 cm"): go to the pose 50 cm ahead, around what is in the way. If there is no way, stop before the object and say so.
- Not potential fields: they get stuck in U-shaped clutter and swing in narrow gaps.

### Progress monitor

- Each skill reports progress (path left, phase done). No progress for 2 s means stuck.
- The same skill with the same settings runs at most twice. A third try must change something: another side, the jaws turned 90°, or back off 10 cm first.
- The budget is time (60 s plus 20 s per object), not turns.
- When code has no option left: one Jev Choice call (retry another way, skip, stop), then a clear end message.

### Grasp and place (types 3 and 5)

- Before a pick: check the arm reaches hover and grip height with margin, and the open jaws clear the neighbours. If not, move the base to another side first.
- Close: check both jaws touch and the gripper did not close fully. If not: open, center on the object again, retry with the jaws turned 90°.
- Small balls: the tips stick out below them. Release from 1 cm above once centered, or grip them lower.
- Option, sim only: try a pick in a scratch copy of the physics first and keep the best of 2 or 3 candidates.

### Speed

- Arm moves 2 to 3 times faster, and wait until the joints stop instead of fixed waits.
- The arm moves to hover while the base drives the last 15 cm.
- Base up to 30 cm/s.
- 1 Jev call per task.
- Keep timestep 0.005 on the page. Use 0.01 in the bench if it stays green.

### Perception

- Today the skills use true poses from MuJoCo. Fast and exact. Keep it.
- A real robot needs a camera. Skills read only `World` (robot pose, and each object's pose, size and kind) and send only wheel speeds, joint targets and the gripper. A camera module can fill `World` later.
- The bench can add pose noise (5 mm, 5°) so the skills do not depend on perfect data.

### What stays

The task logic in plan.ts, the IK, the brake, the Drive pad and the Flow panel. Flow shows skills instead of turns, and the planned path is drawn on the floor.

## 3. Phases

Each phase ships only when the bench passes.

| Phase | What | Expected gain |
|---|---|---|
| 0 | The bench as a gate (`bun run bench`, 12 families x 20 seeds, rules decider, plus a real Jev run before each push). The goal wins over the selection, which only fills "it" or "this". | Wrong object: 29 of 29 runs to 0. |
| 1 | Map, A* path, safe follow, contact stop and plan again. Progress monitor with the "twice at most" rule. | Base contacts: 51% of driving runs to under 2%. Loops: 23% to 0. Clean success: 63% to about 90%. |
| 2 | Skills run by code end to end. Jev reads the goal and handles recovery only. Continuous drive up to 30 cm/s. | Jev calls: 8 to 1 per task. The 6.9% wrong step picks disappear. Far take: 16 s to about 9 s. |
| 3 | Grasp and place checks and retries, small balls, the container: "put X in the container", "fill the container with the balls". | Ball on box: 8 of 13 to 13 of 13. Clean success 95% or more. The container joins the bench. |
| 4 | Speed: faster arm moves, waits that end when the joints stop, arm moving while driving. | Take in reach: 8 s to under 4 s. Stack of 3 in reach: 27 s to under 15 s. |

## 4. Targets

On the bench (240 runs or more, the real pipeline):

- Clean success 95% or more. 100% on the must-pass cases: "take the ball" with a box selected, a box on the path, "turn around" near objects.
- The base never touches an object it did not aim for, in 98% of runs or more.
- Nothing repeated more than twice with the same settings. Every task ends within 60 s plus 20 s per object, with a reason when it fails.
- Median task time on the page 7 s or less (today 15 to 16.5 s). Take in reach 4 s, far take 9 s, stack of 3 in reach 15 s.
- Jev calls: 1 at the median, 2 at p90 (today 8 and 21). About $0.00005 per task.
- Chassis tilt always under 5°.

## 5. Reproduce

```sh
bun run bench --mode rules --seeds 20 --jobs 2 --seed 1000 --out /tmp/so101-bench/rules-1.jsonl
bun run bench --mode rules --seeds 20 --jobs 2 --seed 8000 --out /tmp/so101-bench/heldout-1.jsonl
bun run bench --mode jev --seeds 2 --jobs 2 --seed 1500 --no-gate --out /tmp/so101-bench/jev-1.jsonl
```

The rules benchmark now covers 15 families, including filling the container with boxes: 300 runs at 20 seeds per family. The small real Jev sample checks the provider and goal readings, not the full release gate. It reads the key from `~/.secrets/jev-key` without printing it.

Times are simulated motion time plus Jev latency (0.2 s per call for rules, measured for real Jev). Browser wall time also depends on rendering speed. Software rendering on this box is much slower than real time.

## 6. Results

`bun run bench`, the rules decider, 20 random scenes per family. 12 families (240 runs) until phase 2, 14 with the two container families from phase 3 (280 runs). Times are motion plus 0.2 s per Jev call.

| | Before (36c6988) | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Target |
|---|---|---|---|---|---|---|---|
| Clean success | 62% | 70% | 98% | 98% | 100% | 279/280 (99.6%) | 95% or more |
| Task time, median / p90 | 16.4 / 44.3 s | 16.4 / 44.3 s | 15.0 / 43.7 s | 12.7 / 48.9 s | 14.1 / 60.8 s | 6.5 / 31.6 s | 7 s or less |
| Jev calls, median / p90 / max | 8 / 23 / 84 | 8 / 23 / 84 | 6 / 15 / 23 | 1 / 1 / 3 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 2 |
| Runs without base contact | 71% | 71% | 100% | 100% | 100% | 99.6% | 98% or more |
| Max chassis tilt | 15.3° | 15.3° | 1.2° | 0.9° | 1.2° | 1.4° | under 5° |
| "Take the ball", a box selected | 0% | 100% | 100% | 100% | 100% | 100% | 100% |
| Container tasks | | | | | 100% | 100% | 90% or more |

Phase 4 by family, median time: take in reach 2.9 s, turn around 3.7 s, go forward 50 cm around a box 4.7 s, drive to an object 4.8 s, put on a box in reach 4.9 s, far take 7.2 s, blocked take 8.0 s, stack 3 in reach 10.5 s, far put 13.8 s, put in the container 14.5 s, stack 3 spread out 30.7 s, fill the container with 3 balls 48.5 s.

Real Jev (`jev-latest`) on the same bench after phase 4: 28 runs, 2 per family, 100% clean, median 6.0 s, 1 call per task.

The original phase 4 report rounded 279/280 clean runs to 100%. The table now gives the exact count. Those earlier measurements did not map every container wall in the contact checks. The safety follow-up covers all wall geoms, stops arm motion on driving contact, preserves held loads during GoTo, checks Stop between phases and enforces the deadline during physics steps. The gate now checks all mandatory families, repeat limits and motion budgets. Box filling has its own family, without changing the earlier families' seeds.

### Safety follow-up, 2026-09-30

Both release checks pass with the original targets unchanged. Container approach keeps 25 cm from its centre. Empty-arm driving reuses its hover pose instead of repeating the same move. Gentle grasp, carry and release timings keep the safer arm pace.

| | Default, seed 1000 | Heldout, seed 8000 | Target |
|---|---|---|---|
| Success | 299/300 (99.7%) | 300/300 (100%) | |
| Clean success | 290/300 (96.7%) | 296/300 (98.7%) | at least 95% |
| Task time, median / p90 | 7.0 / 46.1 s | 6.9 / 42.2 s | median at most 7 s |
| Jev calls, median / p90 / max | 1 / 1 / 3 | 1 / 1 / 2 | median 1, p90 at most 2 |
| Runs without base contact | 300/300 (100%) | 295/300 (98.3%) | at least 98% |
| Mandatory safety families, clean | 80/80 (100%) | 80/80 (100%) | 100% |
| Most repeats of the same settings | 2 | 2 | at most 2 |
| Motion-budget overruns | 0 | 0 | 0 |
| Near take / far take / near 3-box stack | 2.9 / 6.6 / 10.5 s | 2.9 / 6.8 / 10.5 s | at most 4 / 9 / 15 s |
| Max chassis tilt | 1.3° | 3.2° | under 5° |
| Container tasks, success | 60/60 (100%) | 60/60 (100%) | at least 90% |
| Container tasks, clean | 51/60 (85%) | 56/60 (93.3%) | included in overall clean target |
| Box-fill family, clean | 13/20 (65%) | 17/20 (85%) | included in overall clean target |

The tests check Stop during release, no later retreat, retaining a load during GoTo, contact with every container wall, two 6 cm cubes side by side, the active final-skill deadline and restarting after cancellation. They also reject a gate missing mandatory families or hiding excess repeats and budget overruns. Typecheck, `check-agent` and the production build pass.

A separate real Jev sample, seed 1500, has 30/30 successful and clean tasks, one call each, no repeated settings and no budget overruns. One run has a base contact, so this small sample does not meet the 98% no-contact target and is not a full release gate (`--no-gate`). Its printed median is 7.0 s and p90 45.2 s. The benchmark uses nearest-rank quantiles. The midpoint median for these 30 samples is 7.25 s. Median input is 2,236 tokens, about $0.000094 at the documented input price.

Remaining physical limits are real: difficult container scenes can shift the container or tip a box, and stopping on contact does not undo movement already caused by that contact. The default sample includes a 21.5 cm container shift. One spread-stack run safely stops because the arm cannot clear the objects. Filling succeeds in every container case above, but it is not always clean. Emptying a container and arbitrary multi-step goals are not supported by this task reader.

Rejected iterations are not release evidence: the first strict check was 295/300 clean but 7.5 s median. A faster arm with corner-first packing was 282/300 clean and 7.4 s. An intermediate safer setting was 297/300 clean and 7.2 s, while its heldout sample missed the no-contact target. The final settings above replace them.

Not done as planned:
- Timestep 0.01: CPU per run went from 1.7 to 1.2 s only, and the base got worse (a base contact, tilt 3°, median 7.7 s). The model keeps 0.005.
- The pivot: the base turns about a point 2.5 cm ahead of the arm's base, not its chassis centre (the arm loads the front wheels). The planner uses the measured point.
- Pose noise in the bench (5 mm, 5°) is not built: the skills still read true poses.
- The cost target is not met in the measured live examples: about $0.0001 per task, not about $0.00005. They use about 2,350 input tokens in one Jev call.

