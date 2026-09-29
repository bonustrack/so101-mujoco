// The scene editor: add boxes and balls, select one by clicking it, drag it along the table,
// resize or delete it. Physics stays in robot.ts; this file handles the pointer, the keys,
// the selection ring and the Scene section of the right panel.
import * as THREE from "three";
import { SIZE_RANGE, type Kind, type Obj, type Robot, type Vec3 } from "./robot";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const cm = (m: number) => `${Math.round(m * 1000) / 10} cm`;
const SIDES = { box: ["Width", "Depth", "Height"], ball: ["Diameter"] };
const NUDGE = 0.005; // arrow keys move the selection 5 mm, 2 cm with Shift

export type Editor = ReturnType<typeof setupEditor>;

export function setupEditor(options: {
  robot: Robot;
  camera: THREE.Camera;
  canvas: HTMLCanvasElement;
  scene: THREE.Scene;
  meshes: Map<Obj, THREE.Object3D>;
  onChange: () => void; // the selection or a size changed: Jev's target may have too
}) {
  const { robot, camera, canvas, scene, meshes, onChange } = options;
  let selected: Obj | null = null;
  let locked = false;

  // A thin ring on the table around the selection.
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(1, 1.08, 64),
    new THREE.MeshBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.45, depthWrite: false }),
  );
  ring.visible = false;
  scene.add(ring);

  // Picking and dragging. The capture listener runs before OrbitControls, so a press on an
  // object moves the object and a press anywhere else still orbits the camera.
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const hit = new THREE.Vector3();
  const aim = (event: PointerEvent) => {
    const rect = canvas.getBoundingClientRect();
    pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
  };
  let drag: { o: Obj; id: number; plane: THREE.Plane; offset: THREE.Vector2; start: THREE.Vector2; moving: boolean } | null = null;
  let press: THREE.Vector2 | null = null;

  canvas.parentElement!.addEventListener(
    "pointerdown",
    (event) => {
      if (event.target !== canvas || drag || !event.isPrimary) return;
      press = new THREE.Vector2(event.clientX, event.clientY);
      aim(event);
      const shown = robot.active().map((o) => meshes.get(o)!);
      const first = raycaster.intersectObjects(shown, false)[0];
      if (!first) return;
      const o = [...meshes].find(([, mesh]) => mesh === first.object)![0];
      event.stopPropagation();
      canvas.setPointerCapture(event.pointerId);
      select(o);
      const p = robot.object(o).pos;
      drag = {
        o,
        id: event.pointerId,
        plane: new THREE.Plane(new THREE.Vector3(0, 0, 1), -first.point.z),
        offset: new THREE.Vector2(p[0] - first.point.x, p[1] - first.point.y),
        start: press,
        moving: false,
      };
    },
    { capture: true },
  );
  canvas.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.id) return;
    // A few pixels of slack, so a plain click selects without nudging.
    if (!drag.moving && drag.start.distanceTo(new THREE.Vector2(event.clientX, event.clientY)) < 4) return;
    drag.moving = true;
    aim(event);
    if (!raycaster.ray.intersectPlane(drag.plane, hit)) return;
    move(drag.o, hit.x + drag.offset.x, hit.y + drag.offset.y);
  });
  const release = (event: PointerEvent) => {
    if (drag && event.pointerId === drag.id) {
      if (drag.moving) robot.drop();
      drag = null;
      press = null;
      return;
    }
    // A click on empty space, not an orbit: clear the selection.
    if (press && event.type === "pointerup" && press.distanceTo(new THREE.Vector2(event.clientX, event.clientY)) < 4) select(null);
    press = null;
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);

  // Keep dragged objects on the table in front of the arm, clear of its base.
  function move(o: Obj, x: number, y: number) {
    const min = 0.07 + robot.footprint(o);
    const r = Math.min(Math.max(Math.hypot(x, y), min), 0.45);
    const a = Math.atan2(y, x);
    robot.drag(o, r * Math.cos(a), r * Math.sin(a));
  }

  // Arrow keys nudge the selection along the view: up moves it away from the camera.
  addEventListener("keydown", (event) => {
    if (!selected || event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target instanceof HTMLSelectElement) return;
    if (event.key === "Escape") return select(null);
    if ((event.key === "Delete" || event.key === "Backspace") && !locked) return remove();
    const turn = { ArrowUp: 0, ArrowLeft: 1, ArrowDown: 2, ArrowRight: 3 }[event.key];
    if (turn === undefined || drag) return;
    event.preventDefault();
    const view = camera.getWorldDirection(new THREE.Vector3());
    const heading = Math.atan2(view.y, view.x) + (turn * Math.PI) / 2;
    const step = event.shiftKey ? 4 * NUDGE : NUDGE;
    const p = robot.object(selected).pos;
    move(selected, p[0] + step * Math.cos(heading), p[1] + step * Math.sin(heading));
    robot.drop();
  });

  // The Scene section.
  const addBox = $<HTMLButtonElement>("add-box");
  const addBall = $<HTMLButtonElement>("add-ball");
  const resetScene = $<HTMLButtonElement>("reset-scene");
  const panel = $("selection");
  const name = $("selected-name");
  const del = $<HTMLButtonElement>("delete");
  const sizes = $("sizes");
  const grab = $("grab");
  const hint = $("scene-hint");
  let sliders: { input: HTMLInputElement; output: HTMLOutputElement }[] = [];

  const add = (kind: Kind) => {
    const o = robot.add(kind);
    if (o) select(o);
  };
  addBox.onclick = () => add("box");
  addBall.onclick = () => add("ball");
  resetScene.onclick = () => {
    select(null);
    robot.resetScene();
    refresh();
  };
  del.onclick = () => remove();

  function remove() {
    if (!selected) return;
    robot.remove(selected);
    select(null);
  }

  function select(o: Obj | null) {
    selected = o;
    panel.hidden = !o;
    hint.hidden = !!o;
    sliders = [];
    sizes.replaceChildren();
    if (o) {
      name.textContent = o.label;
      sliders = SIDES[o.kind].map((label, i) => {
        const row = document.createElement("div");
        row.className = "joint size";
        row.innerHTML = `<label for="size-${i}">${label}</label><input id="size-${i}" type="range" step="0.005"><output></output>`;
        const input = row.querySelector("input")!;
        input.min = String(2 * SIZE_RANGE[0]);
        input.max = String(2 * SIZE_RANGE[1]);
        input.value = String(2 * o.size[i]);
        input.disabled = locked;
        input.oninput = () => {
          const size = [...o.size] as Vec3;
          size[i] = Number(input.value) / 2;
          robot.resize(o, size);
          refresh();
        };
        sizes.append(row);
        return { input, output: row.querySelector("output")! };
      });
    }
    refresh();
  }

  // Buttons, sliders and the grab note follow the scene.
  function refresh() {
    const left = (kind: Kind) => robot.objects.some((o) => o.kind === kind && !o.active);
    addBox.disabled = locked || !left("box");
    addBall.disabled = locked || !left("ball");
    addBox.title = left("box") ? "Add a box" : "All 6 boxes are out";
    addBall.title = left("ball") ? "Add a ball" : "All 6 balls are out";
    resetScene.disabled = locked;
    del.disabled = locked;
    if (selected) {
      sliders.forEach(({ input, output }, i) => {
        input.disabled = locked;
        output.value = cm(2 * selected!.size[i]);
      });
      const why = robot.tooBig(selected);
      grab.textContent = why ? `Too big to grab: ${why}.` : "The arm can grab it.";
      grab.classList.toggle("warn", !!why);
    }
    onChange();
  }

  select(null);
  return {
    selected: () => selected,
    // While Jev drives the arm, objects can still be selected and dragged, but not added, resized or removed.
    lock(value: boolean) {
      locked = value;
      refresh();
    },
    refresh,
    // Once per frame: the ring follows the selection.
    frame() {
      ring.visible = !!selected?.active;
      if (!selected?.active) return;
      const p = robot.object(selected).pos;
      ring.position.set(p[0], p[1], 0.0008);
      ring.scale.setScalar(robot.footprint(selected) + 0.006);
    },
  };
}
