// Driving without touching anything. A map of what stands on the floor, a path for the robot's real outline
// (A* over position and heading, so it finds "back up, then turn" by itself), and a follower that stops at
// the first touch and plans again from where things are now. It never pushes on.
//
// Poses here are world coordinates on the floor. Paths are planned for the point the base turns about on the
// spot: measured, 2.5 cm ahead of the arm's base, since the arm loads the front wheels. `pivot` converts.
import { BODY, wrap, type Obj, type Robot } from "./robot";

export type Pose = { x: number; y: number; yaw: number };
const PIVOT = 0.025; // in the arm's frame; a 90° turn drifts about 5 mm around it
const CX = (BODY.front + BODY.back) / 2 - PIVOT; // the outline's centre, from the pivot: 5 cm behind
const HALF_L = (BODY.front - BODY.back) / 2; // 10.5 cm
const HALF_W = BODY.side; // 9.5 cm, wheels included
export const MARGIN = 0.02; // kept free around the outline
const CELL = 0.02; // the grid A* remembers poses on
const HEADINGS = 24; // 15° apart: every turn a goal can ask for
const DH = (2 * Math.PI) / HEADINGS;
const STRIDE = 0.04; // one straight move
const REACHED = 0.03; // a path ends this close to its goal; the follower corrects the rest
const EXPANSIONS = 200_000;

// The pivot of the base whose arm's frame stands at `p`.
export const pivot = (p: Pose): Pose => ({ x: p.x + PIVOT * Math.cos(p.yaw), y: p.y + PIVOT * Math.sin(p.yaw), yaw: p.yaw });

// ---- the map ----

// An object on the floor as a circle. `soft`: only the chassis must clear it, not what the jaws hold (the
// held object is carried over its target).
type Circle = { x: number; y: number; r: number; top: number; obj: Obj; soft: boolean };
// A part that sticks out of the chassis outline, in the pivot's frame: it only meets obstacles taller than
// `above`.
type Part = { x: number; y: number; r: number; above: number };
export type FloorMap = { circles: Circle[]; parts: Part[]; margin: number };

// What stands on the floor now, from the physics state: every object but the one in the jaws. The folded
// jaws, 12 cm up and 15 cm ahead, only meet a tower; an object in the jaws meets what stands higher than
// its bottom.
export function floorMap(robot: Robot, soft: Obj[] = [], margin = MARGIN): FloorMap {
  const held = robot.active().find(robot.gripped);
  const circles = robot.active().flatMap((o) => {
    if (o === held) return [];
    const f = robot.object(o);
    return [{ x: f.world[0], y: f.world[1], r: robot.footprint(o), top: f.top, obj: o, soft: soft.includes(o) }];
  });
  const parts: Part[] = [{ x: 0.15 - PIVOT, y: 0, r: 0.03, above: 0.115 }];
  if (held) {
    const f = robot.object(held);
    parts.push({ x: f.pos[0] - PIVOT, y: f.pos[1], r: robot.footprint(held), above: f.bottom - 0.01 });
  }
  return { circles, parts, margin };
}

// The gap between the robot with its pivot at `c` and the nearest obstacle; negative when they overlap.
export function clearance(map: FloorMap, c: Pose) {
  const cs = Math.cos(c.yaw);
  const sn = Math.sin(c.yaw);
  let gap = Infinity;
  for (const o of map.circles) {
    const dx = o.x - c.x;
    const dy = o.y - c.y;
    const lx = cs * dx + sn * dy;
    const ly = -sn * dx + cs * dy;
    const qx = Math.abs(lx - CX) - HALF_L;
    const qy = Math.abs(ly) - HALF_W;
    const d = qx > 0 || qy > 0 ? Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) : Math.max(qx, qy);
    gap = Math.min(gap, d - o.r);
    if (o.soft) continue;
    for (const p of map.parts) {
      if (o.top <= p.above) continue;
      gap = Math.min(gap, Math.hypot(lx - p.x, ly - p.y) - p.r - o.r);
    }
  }
  return gap;
}
// The obstacle nearest the robot at `c`.
export function nearest(map: FloorMap, c: Pose) {
  let best: Circle | null = null;
  let gap = Infinity;
  for (const o of map.circles) {
    const g = clearance({ ...map, circles: [o] }, c);
    if (g < gap) [best, gap] = [o, g];
  }
  return best;
}

// ---- the path ----

// A goal for the pivot, with an extra cost in seconds (a less good side, a sideways offset).
export type Goal = Pose & { cost: number };
type Move = "start" | "forward" | "back" | "left" | "right";
// A turn keeps its signed amount: left or right matters, 180° left is not 180° right.
export type Segment = { kind: "turn"; yaw: number; amount: number } | { kind: "line"; x: number; y: number; back: boolean };
export type Path = { segments: Segment[]; goal: Goal; points: [number, number][]; seconds: number };
export type Speeds = { drive: number; turn: number }; // m/s, rad/s
const SWITCH = 0.4; // seconds lost stopping and starting again between moves of another kind

// The cheapest path from the pivot at `start` to any goal, A* over position and 24 headings. Moves:
// 4 cm forward or back, or a turn of 15° on the spot, each allowed only where the outline stays `margin` clear
// of everything along the way. From a pose already too close, any move that gets farther away is allowed too.
export function planPath(map: FloorMap, start: Pose, goals: Goal[], speeds: Speeds): Path | null {
  if (!goals.length) return null;
  const pad = 0.6;
  const x0 = Math.min(start.x, ...goals.map((g) => g.x)) - pad;
  const y0 = Math.min(start.y, ...goals.map((g) => g.y)) - pad;
  const nx = Math.ceil((Math.max(start.x, ...goals.map((g) => g.x)) + pad - x0) / CELL) + 1;
  const ny = Math.ceil((Math.max(start.y, ...goals.map((g) => g.y)) + pad - y0) / CELL) + 1;
  const size = nx * ny * HEADINGS;
  const g = new Float64Array(size).fill(Infinity);
  const px = new Float64Array(size);
  const py = new Float64Array(size);
  const gap = new Float64Array(size);
  const parent = new Int32Array(size).fill(-1);
  const move = new Uint8Array(size);
  const closed = new Uint8Array(size);
  const MOVES: Move[] = ["start", "forward", "back", "left", "right"];
  const headingOf = (yaw: number) => (((Math.round(wrap(yaw - start.yaw) / DH) % HEADINGS) + HEADINGS) % HEADINGS);
  const goalH = goals.map((goal) => headingOf(goal.yaw));
  const keyOf = (x: number, y: number, h: number) => {
    const ix = Math.round((x - x0) / CELL);
    const iy = Math.round((y - y0) / CELL);
    return ix < 0 || iy < 0 || ix >= nx || iy >= ny ? -1 : (ix * ny + iy) * HEADINGS + h;
  };
  const turns = (a: number, b: number) => Math.min((a - b + HEADINGS) % HEADINGS, (b - a + HEADINGS) % HEADINGS);
  const guess = (x: number, y: number, h: number) => {
    let best = Infinity;
    goals.forEach((goal, i) => {
      const t = Math.max(0, Math.hypot(goal.x - x, goal.y - y) - REACHED) / speeds.drive + (turns(h, goalH[i]) * DH) / speeds.turn + goal.cost;
      if (t < best) best = t;
    });
    return best;
  };
  const heap = new Heap();
  const s = keyOf(start.x, start.y, 0);
  g[s] = 0;
  px[s] = start.x;
  py[s] = start.y;
  gap[s] = clearance(map, start);
  heap.push(guess(start.x, start.y, 0), s);
  const finals = new Map<number, number>(); // final marker -> goal index
  let expansions = 0;
  while (heap.size) {
    const [, key] = heap.pop();
    if (key < 0) return build(-key - 1, goals[finals.get(key)!]);
    if (closed[key]) continue;
    closed[key] = 1;
    if (++expansions > EXPANSIONS) return null;
    const h = key % HEADINGS;
    const [x, y] = [px[key], py[key]];
    goals.forEach((goal, i) => {
      if (goalH[i] === h && Math.hypot(goal.x - x, goal.y - y) <= REACHED) {
        const marker = -key - 1;
        if (!finals.has(marker) || goals[finals.get(marker)!].cost > goal.cost) {
          finals.set(marker, i);
          heap.push(g[key] + goal.cost, marker);
        }
      }
    });
    const yaw = start.yaw + h * DH;
    const last = MOVES[move[key]];
    for (let m = 1; m <= 4; m++) {
      const kind = MOVES[m];
      let nxp = x;
      let nyp = y;
      let nh = h;
      let cost: number;
      let mid: Pose;
      if (kind === "forward" || kind === "back") {
        const d = kind === "forward" ? STRIDE : -STRIDE;
        nxp = x + d * Math.cos(yaw);
        nyp = y + d * Math.sin(yaw);
        mid = { x: (x + nxp) / 2, y: (y + nyp) / 2, yaw };
        cost = (STRIDE / speeds.drive) * (kind === "back" ? 1.5 : 1);
      } else {
        nh = (h + (kind === "left" ? 1 : HEADINGS - 1)) % HEADINGS;
        mid = { x, y, yaw: yaw + (kind === "left" ? DH / 2 : -DH / 2) };
        cost = DH / speeds.turn;
      }
      if (last !== "start" && last !== kind) cost += SWITCH;
      const nk = keyOf(nxp, nyp, nh);
      if (nk < 0 || closed[nk]) continue;
      const ng = g[key] + cost;
      if (ng >= g[nk]) continue;
      const end = { x: nxp, y: nyp, yaw: start.yaw + nh * DH };
      const c = Math.min(clearance(map, mid), clearance(map, end));
      if (c < map.margin && !(gap[key] < map.margin && c > gap[key] + 1e-4)) continue;
      g[nk] = ng;
      px[nk] = nxp;
      py[nk] = nyp;
      gap[nk] = c;
      parent[nk] = key;
      move[nk] = m;
      heap.push(ng + guess(nxp, nyp, nh), nk);
    }
  }
  return null;

  // From the start to `key`: merged into turns and straight lines, and the pivot's track for drawing.
  function build(key: number, goal: Goal): Path {
    const chain: number[] = [];
    for (let k = key; k >= 0; k = parent[k]) chain.unshift(k);
    const segments: Segment[] = [];
    const points: [number, number][] = [[start.x, start.y]];
    for (const k of chain.slice(1)) {
      const kind = MOVES[move[k]];
      const lastSeg = segments.at(-1);
      if (kind === "left" || kind === "right") {
        const yaw = start.yaw + (k % HEADINGS) * DH;
        const step = kind === "left" ? DH : -DH;
        if (lastSeg?.kind === "turn") Object.assign(lastSeg, { yaw, amount: lastSeg.amount + step });
        else segments.push({ kind: "turn", yaw, amount: step });
      } else {
        const back = kind === "back";
        if (lastSeg?.kind === "line" && lastSeg.back === back) Object.assign(lastSeg, { x: px[k], y: py[k] });
        else segments.push({ kind: "line", x: px[k], y: py[k], back });
        points.push([px[k], py[k]]);
      }
    }
    // The exact goal: the follower ends there.
    points.push([goal.x, goal.y]);
    return { segments, goal, points, seconds: g[key] + goal.cost };
  }
}

// A binary min-heap of (priority, key).
class Heap {
  private f: number[] = [];
  private k: number[] = [];
  get size() {
    return this.f.length;
  }
  push(f: number, k: number) {
    const { f: F, k: K } = this;
    let i = F.length;
    F.push(f);
    K.push(k);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (F[p] <= F[i]) break;
      [F[p], F[i]] = [F[i], F[p]];
      [K[p], K[i]] = [K[i], K[p]];
      i = p;
    }
  }
  pop(): [number, number] {
    const { f: F, k: K } = this;
    const top: [number, number] = [F[0], K[0]];
    const lf = F.pop()!;
    const lk = K.pop()!;
    if (F.length) {
      F[0] = lf;
      K[0] = lk;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < F.length && F[l] < F[m]) m = l;
        if (r < F.length && F[r] < F[m]) m = r;
        if (m === i) break;
        [F[m], F[i]] = [F[i], F[m]];
        [K[m], K[i]] = [K[i], K[m]];
        i = m;
      }
    }
    return top;
  }
}

// ---- goals ----

// Where the base stands to work on something at world (x, y), `distance` from the arm's base, facing it: one
// pose per heading, 24 around it. The free ones, cheaper where no other object stands between the jaws and it.
export function goalsAround(map: FloorMap, x: number, y: number, distance: number, from: number, skip: Obj[] = []): Goal[] {
  const goals: Goal[] = [];
  for (let h = 0; h < HEADINGS; h++) {
    const yaw = from + h * DH;
    const c = pivot({ x: x - distance * Math.cos(yaw), y: y - distance * Math.sin(yaw), yaw });
    if (clearance(map, c) < map.margin) continue;
    // Others next to it on the robot's side are in the way of the arm.
    let cost = 0;
    for (const o of map.circles) {
      if (skip.includes(o.obj) || Math.hypot(o.x - x, o.y - y) < 1e-6) continue;
      const along = -((o.x - x) * Math.cos(yaw) + (o.y - y) * Math.sin(yaw));
      const side = Math.abs(-(o.x - x) * Math.sin(yaw) + (o.y - y) * Math.cos(yaw));
      if (along > -0.02 && along < distance && side < o.r + 0.05) cost += 1.5;
    }
    goals.push({ ...c, cost });
  }
  return goals;
}

// A move asked for in the goal, from the arm's pose `from`: `amount` metres straight ahead (negative: back) or
// radians turned. Where the exact end is taken, the robot may end beside it: side offsets cost more the
// farther they are. A turn may shift the robot a little, to find room to turn.
export function goalsForMove(map: FloorMap, from: Pose, turn: boolean, amount: number): Goal[] {
  const goals: Goal[] = [];
  const u = [Math.cos(from.yaw), Math.sin(from.yaw)];
  const n = [-u[1], u[0]];
  if (turn) {
    const yaw = from.yaw + amount;
    for (const [a, b] of [[0, 0], ...[0.05, 0.1, 0.15].flatMap((d) => [[d, 0], [-d, 0], [0, d], [0, -d]])])
      goals.push({ ...pivot({ x: from.x + a * u[0] + b * n[0], y: from.y + a * u[1] + b * n[1], yaw }), cost: 4 * Math.hypot(a, b) });
  } else
    for (let k = 0; k <= 8; k++)
      for (const side of k ? [1, -1] : [1]) {
        const off = side * k * 0.05;
        goals.push({ ...pivot({ x: from.x + amount * u[0] + off * n[0], y: from.y + amount * u[1] + off * n[1], yaw: from.yaw }), cost: 10 * Math.abs(off) });
      }
  return goals.filter((goal) => clearance(map, goal) >= map.margin);
}

// ---- the follower ----

export type Drove = { ok: boolean; bumped: Obj | null; moved: number };

// Drive a path segment by segment with the base's own controller. Before each straight line it turns to face
// the line's end from where it really is, so small slips do not add up. At the end it drives what is left along
// its heading and turns to the goal's heading. Stops at the first touch.
export async function follow(robot: Robot, path: Path, signal?: AbortSignal): Promise<Drove> {
  let moved = 0;
  const here = () => pivot(robot.base());
  // Turn to `yaw`, the planned way round when `amount` is given.
  const turnTo = async (yaw: number, amount?: number) => {
    const d = amount === undefined ? wrap(yaw - robot.base().yaw) : amount + wrap(yaw - amount - robot.base().yaw);
    if (Math.abs(d) < 0.01) return null;
    return (await robot.move("turn", d)).bumped;
  };
  const drive = async (d: number) => {
    if (Math.abs(d) < 0.004) return null;
    const r = await robot.move("drive", d);
    moved += Math.abs(r.moved);
    return r.bumped;
  };
  for (const seg of path.segments) {
    if (signal?.aborted) return { ok: false, bumped: null, moved };
    let bumped: Obj | null;
    if (seg.kind === "turn") bumped = await turnTo(seg.yaw, seg.amount);
    else {
      const c = here();
      const d = Math.hypot(seg.x - c.x, seg.y - c.y);
      const face = Math.atan2(seg.y - c.y, seg.x - c.x) + (seg.back ? Math.PI : 0);
      bumped = d > 0.02 && Math.abs(wrap(face - c.yaw)) > 0.03 ? await turnTo(face) : null;
      if (!bumped) {
        const e = here();
        bumped = await drive((seg.back ? -1 : 1) * Math.hypot(seg.x - e.x, seg.y - e.y));
      }
    }
    if (bumped) return { ok: false, bumped, moved };
  }
  const c = here();
  const along = (path.goal.x - c.x) * Math.cos(c.yaw) + (path.goal.y - c.y) * Math.sin(c.yaw);
  const bumped = (await drive(along)) ?? (await turnTo(path.goal.yaw));
  return { ok: !bumped, bumped, moved };
}
