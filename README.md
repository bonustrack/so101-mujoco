# SO-101 in MuJoCo

The SO-101 robot arm, simulated live in the browser with MuJoCo physics and drawn with three.js.

- Drag to orbit, scroll or pinch to zoom.
- One slider per joint drives the arm's position actuators. The number next to each slider is the joint's actual angle.
- Preset poses: rest, reach forward, reach right, reach up. Reset puts the arm and the box back.
- A 3 x 3 x 4 cm box sits within reach of a top-down grasp.

## Jev takes the box

The left panel lets [Jev](https://docs.typesafe.ai) (TypeSafe AI's System One model) drive the arm. Paste a TypeSafe API key, type a goal such as "Take the box" and press Run.

Jev is a decision model, not a chat model: it answers typed questions and does not write text or call tools. So the loop keeps code in control and gives Jev one narrow decision per turn:

1. Observe. Code turns the physics state into plain facts: jaws open or closed, gripper away from, above or around the box, box standing or lifted, plus positions in cm.
2. Decide. One `POST /v1/systemone` call with the goal, the observation and the last commands as `state`, and one Choice question over 7 commands: `open_gripper`, `move_above_box`, `lower_to_box`, `close_gripper`, `lift`, `go_home`, `done`. Jev returns the choice, a probability per command and a confidence.
3. Act. Code solves the joint angles with damped least squares inverse kinematics on `mj_jacSite` (jaw tips on target, fingers pointing down, jaws square to the box) and MuJoCo runs the motion.
4. Check. Code decides success: the box's lowest corner 5 cm above the table, touched by both jaws, for 1 s. Otherwise the loop observes again, up to 12 turns or until Stop.

"Randomize box" moves the box at any time, even mid-run, and the next observation shows Jev the new spot. The Flow panel shows each turn: what was sent, Jev's top probabilities, what the arm did and the check.

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
bun run check-agent  # headless: scripted picks on 40 random boxes, the loop with a mocked Jev, Stop, the proxy
```

## Deploy

Netlify reads `netlify.toml`: build `bun install && bun run build`, publish `dist`, functions from `netlify/functions`.

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/bonustrack/so101-mujoco)

## Credits

- Model: [SO-101](https://github.com/TheRobotStudio/SO-ARM100) by The Robot Studio, from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) (`robotstudio_so101`), Apache-2.0. The files in `model/` are unchanged, see `model/LICENSE`. `model/scene_web.xml` is this site's scene, based on the Menagerie `scene_box.xml`.
- Physics: [MuJoCo](https://github.com/google-deepmind/mujoco), Apache-2.0.
