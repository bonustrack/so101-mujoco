# SO-101 in MuJoCo

The SO-101 robot arm on a wheeled base, simulated live in the browser with MuJoCo physics and drawn with three.js.

- Drag to orbit, scroll or pinch to zoom. The view follows the base.
- One slider per joint drives the arm's position actuators. The number next to each slider is the joint's actual angle.
- Preset poses: rest, reach forward, reach right, reach up. Reset puts the arm, the base and the objects back.
- A 3 x 3 x 4 cm box sits on the floor within reach of a top-down grasp.

## The wheeled base

The arm rides on a 21 x 14 cm chassis of 2 kg on 4 wheels, a free body on the floor (`model/scene_web.xml` attaches `so101.xml` on top, unchanged). Each wheel is a hinge joint with a velocity actuator, rubber on the floor (friction 1). The left pair and the right pair get the same speed (skid steering): together they drive, against each other they turn on the spot.

- A move is a feedback loop on the chassis pose (`src/robot.ts`): it ramps up, cruises at 12 cm/s, slows down on the target and holds the heading. One step drives 10 cm or turns 15°, within about 2 mm and 0.3°.
- It stops at the first touch and never pushes: an object it already touched when the move started stops it as soon as it moves 4 mm.
- Parked, the base is braked: a weld holds the chassis where it stopped, and driving lets go. Left free, the chassis rocks on MuJoCo's soft wheel contacts whenever the arm swings, and that shakes small objects out of the jaws.
- Before the base moves, the arm folds to rest with its jaws open, or lifts what it holds clear of what is around.
- The Drive pad in the panel runs single steps: forward, back, turn left, turn right.

### Driving for a task

Driving is code's job, not Jev's (`src/nav.ts`):

- A map of the floor from the physics state: every object is a circle. The robot is its real outline, 21 x 19 cm with the wheels, with 2 cm kept free, plus what the jaws hold (it passes over what is lower) and the folded jaws (they pass over anything under 11.5 cm).
- A* over position and 24 headings, 15° apart. Moves: 4 cm forward or back, or a turn on the spot, each allowed only where the outline stays clear all along. So it finds "back up, then turn" by itself. The base turns about a point 2.5 cm ahead of the arm's base, measured: the arm loads the front wheels.
- Driving up to something: 24 poses around it, 20 cm from the arm's base and facing it (22 cm for a tower's top). It keeps the free ones, prefers those with nothing between the jaws and it, and takes the cheapest path.
- A move asked in the goal ("go forward 50 cm", "turn around") goes around what is in the way. It may end beside the straight line, or shift a little to find room to turn. With no way at all, it goes as far as it can and stops before what is in the way.
- The follower drives the path move by move, facing each line's end from where it really is. At the first touch it stops, maps the floor again and plans again. After two touches it keeps a wider berth, after three it stops and says why.

## Jev drives the arm

The panel lets [Jev](https://docs.typesafe.ai) (TypeSafe AI's System One model) drive the arm. Paste a TypeSafe API key, type a goal and press Run. Goals it handles:

- Take one object: "Take the box", "lift the ball", "grab Box 2". The goal wins over the selection: the selected object only fills in "it" or "this".
- Stack the boxes: "Stack all the boxes together". Biggest at the bottom unless the goal asks otherwise.
- Stack everything: boxes, then one ball on top (nothing stays on a ball).
- Put one object on another: "Put Box 2 on Box 1".
- Drive or turn: "go forward", "drive back 20 cm", "turn left", "turn around". Distances snap to 10, 20, 30, 50 cm or 1 m, angles to 15, 30, 45, 90 or 180°.
- Drive up to an object: "go to Box 2". Taking, stacking and putting drive by themselves when what they need is out of reach: "drive to the ball and pick it up".

Anything else ("line them up", "clear the floor") ends at once with the list above.

Jev is a decision model, not a chat model: it answers typed questions and does not write text or call tools. So code stays in control and gives Jev narrow decisions:

1. Read the goal. One `POST /v1/systemone` call with the goal and the objects in words (type, size, largest or smallest, side, out of reach) and eight Choice questions: the task, which object, onto which, the stack order, the spot, which way to drive, how far, how much to turn. Code checks the answers (both objects named, nothing on a ball, the box fits the jaws) and shows the reading and its confidence in the Flow panel.
2. Plan. Code turns the task into pick-and-place steps and makes the plan again from the physics state after every turn (`src/plan.ts`). A tower is built where its biggest box stands, or at a clear spot 22 cm from the base when that box has a neighbour closer than 4 cm or, for a tower of 3 or more, stands outside 18 to 26 cm from the base. A box that falls off goes back on the list. When the object to take or the spot to set it on is out of the arm's reach (14 to 26 cm from its base, up to 77° to either side), the next step is to drive up to it: the object ends 20 cm straight ahead, a tower's top 22 cm.
3. Turns. For the current step, code turns the physics state into plain facts: jaws open or closed, gripper away from, above or around the object, holding it or not, the object away from, above or sitting on its target. One Choice question over the commands that fit the step: `open_gripper`, `move_above_object`, `lower_to_object`, `close_gripper`, `lift` and `go_home` to take an object; `move_above_target`, `lower_to_place`, `release` and `retreat` to set it down; `done`. A drive, asked in the goal or up to what is out of reach, is code's step: see "Driving for a task".
4. Act. Code solves the joint angles with damped least squares inverse kinematics on `mj_jacSite` (fingers pointing down, tilted only as much as a high target needs) and MuJoCo runs the motion. A carry goes around the arm's base, turns a box square to the one below it, and stops lowering at the first touch. The retreat slides the open jaws out sideways, since the arm cannot always rise.
5. Check. Code decides when a step is done: an object taken is 5 cm up in both jaws for 1 s; an object placed rests on its target, let go, jaws clear, still for 0.5 s. A tower counts once every box is on it and it stands for 1 s. The budget is 8 turns per object plus 4, 8 more when a box has to go back (twice at most), 3 more per drive up to something, one per step for a move asked in the goal plus 2, or until Stop. A move is done within 1 cm or 2°.
6. Watch progress. One command runs at most twice from the same state. Then code raises the arm and starts the step afresh, once. The next time, the run stops and says why. A task also stops after 60 s of motion plus 20 s per object.

With the fingers pointing down, the SO-101 reaches about 9.5 cm up at the jaw tips, 20 to 24 cm from its base. So a box goes on a tower only while the tower is at most 10 cm high: the plan leaves out what would go higher, and says so.

"Randomize" moves the current object at any time, even mid-run, and the next observation shows Jev the new spot. The Flow panel shows the reading of the goal, the plan with a tick per step done, and each turn: what was sent, Jev's top probabilities, what the arm did and the check.

The key is kept in the browser's localStorage. api.typesafe.ai does not allow calls from other sites, so the page calls `netlify/functions/jev.ts`, which forwards GET (models) and POST (questions) with the key from the request header. It stores nothing and logs nothing. `bun run dev` and `bun run preview` serve the same function locally.

## Stack

- [@mujoco/mujoco](https://www.npmjs.com/package/@mujoco/mujoco) 3.14.0: the official MuJoCo WebAssembly bindings from Google DeepMind, single-threaded build.
- [three](https://threejs.org) 0.186.1 for rendering, Vite 8 for the build, Bun for scripts.

The page loads the WASM engine and the model lazily, with a loading bar. `scripts/pack-model.ts` packs the scene XML and the STL meshes into one gzip file (`public/so101.pack.gz`, about 6.5 MB) that the browser unpacks with `DecompressionStream`.

## Run it

```sh
bun install
bun run dev      # http://localhost:5173
bun run build    # static site in dist/
bun run check    # type check
bun run check-agent  # headless: scripted picks, the scene editor, the loop with a mocked Jev (take, stack, put on, a fallen box), Stop, the base (steps, brake, bumps, drive then pick), a quick bench, WASM memory, the proxy
bun run bench        # the benchmark gate: 20 random scenes per family, headless, on all cores
```

`bun run bench` runs every task family of `scripts/bench-core.ts` on random scenes with a perfect "rules" decider, prints clean success, task time, Jev calls, base contacts and chassis tilt against the targets, and fails under the gate. `--mode jev` asks the real Jev instead (key from `~/.secrets/jev-key`, or the file in `JEV_KEY_FILE`). A run is clean when it succeeds and nothing else moved more than 2 cm, fell over or rolled under the base.

## Deploy

Netlify reads `netlify.toml`: build `bun install && bun run build`, publish `dist`, functions from `netlify/functions`.

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/bonustrack/so101-mujoco)

## Credits

- Model: [SO-101](https://github.com/TheRobotStudio/SO-ARM100) by The Robot Studio, from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) (`robotstudio_so101`), Apache-2.0. The files in `model/` are unchanged, see `model/LICENSE`. `model/scene_web.xml` is this site's scene, based on the Menagerie `scene_box.xml`.
- Physics: [MuJoCo](https://github.com/google-deepmind/mujoco), Apache-2.0.
