# SO-101 in MuJoCo

The SO-101 robot arm, simulated live in the browser with MuJoCo physics and drawn with three.js.

- Drag to orbit, scroll or pinch to zoom.
- One slider per joint drives the arm's position actuators. The number next to each slider is the joint's actual angle.
- Preset poses: rest, reach forward, reach right, reach up. Reset puts the arm and the box back.
- A 3 x 3 x 4 cm box sits within reach of a top-down grasp.

## Jev drives the arm

The panel lets [Jev](https://docs.typesafe.ai) (TypeSafe AI's System One model) drive the arm. Paste a TypeSafe API key, type a goal and press Run. Goals it handles:

- Take one object: "Take the box", "lift the ball", "grab Box 2".
- Stack the boxes: "Stack all the boxes together". Biggest at the bottom unless the goal asks otherwise.
- Stack everything: boxes, then one ball on top (nothing stays on a ball).
- Put one object on another: "Put Box 2 on Box 1".

Anything else ("line them up", "clear the table") ends at once with the list above.

Jev is a decision model, not a chat model: it answers typed questions and does not write text or call tools. So code stays in control and gives Jev narrow decisions:

1. Read the goal. One `POST /v1/systemone` call with the goal and the objects in words (type, size, largest or smallest, side) and five Choice questions: the task, which object, onto which, the stack order, the spot. Code checks the answers (both objects named, nothing on a ball, the box fits the jaws) and shows the reading and its confidence in the Flow panel.
2. Plan. Code turns the task into pick-and-place steps and makes the plan again from the physics state after every turn (`src/plan.ts`). A tower is built where its biggest box stands, or at a clear spot 22 cm from the base when that box has a neighbour closer than 4 cm or, for a tower of 3 or more, stands outside 18 to 26 cm from the base. A box that falls off goes back on the list.
3. Turns. For the current step, code turns the physics state into plain facts: jaws open or closed, gripper away from, above or around the object, holding it or not, the object away from, above or sitting on its target. One Choice question over the commands that fit the step: `open_gripper`, `move_above_object`, `lower_to_object`, `close_gripper`, `lift` and `go_home` to take an object; `move_above_target`, `lower_to_place`, `release` and `retreat` to set it down; `done`.
4. Act. Code solves the joint angles with damped least squares inverse kinematics on `mj_jacSite` (fingers pointing down, tilted only as much as a high target needs) and MuJoCo runs the motion. A carry goes around the arm's base, turns a box square to the one below it, and stops lowering at the first touch. The retreat slides the open jaws out sideways, since the arm cannot always rise.
5. Check. Code decides when a step is done: an object taken is 5 cm up in both jaws for 1 s; an object placed rests on its target, let go, jaws clear, still for 0.5 s. A tower counts once every box is on it and it stands for 1 s. The budget is 8 turns per object plus 4, 8 more when a box has to go back (twice at most), or until Stop.

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
bun run check-agent  # headless: scripted picks, the scene editor, the loop with a mocked Jev (take, stack, put on, a fallen box), Stop, the proxy
```

## Deploy

Netlify reads `netlify.toml`: build `bun install && bun run build`, publish `dist`, functions from `netlify/functions`.

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/bonustrack/so101-mujoco)

## Credits

- Model: [SO-101](https://github.com/TheRobotStudio/SO-ARM100) by The Robot Studio, from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) (`robotstudio_so101`), Apache-2.0. The files in `model/` are unchanged, see `model/LICENSE`. `model/scene_web.xml` is this site's scene, based on the Menagerie `scene_box.xml`.
- Physics: [MuJoCo](https://github.com/google-deepmind/mujoco), Apache-2.0.
