# SO-101 in MuJoCo

The SO-101 robot arm, simulated live in the browser with MuJoCo physics and drawn with three.js.

- Drag to orbit, scroll or pinch to zoom.
- One slider per joint drives the arm's position actuators. The number next to each slider is the joint's actual angle.
- Preset poses: rest, reach forward, reach right, reach up. Reset puts the arm and the box back.
- A box sits within reach, so the gripper can push it.

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
```

## Deploy

Netlify reads `netlify.toml`: build `bun install && bun run build`, publish `dist`.

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/bonustrack/so101-mujoco)

## Credits

- Model: [SO-101](https://github.com/TheRobotStudio/SO-ARM100) by The Robot Studio, from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101) (`robotstudio_so101`), Apache-2.0. The files in `model/` are unchanged, see `model/LICENSE`. `model/scene_web.xml` is this site's scene, based on the Menagerie `scene_box.xml`.
- Physics: [MuJoCo](https://github.com/google-deepmind/mujoco), Apache-2.0.
