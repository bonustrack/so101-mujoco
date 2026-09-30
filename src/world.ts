// What the path planner and the skills know about the scene: where the robot stands, and each object's pose,
// size and kind. The simulation fills it with the true poses from MuJoCo. A camera module could fill the same
// interface on a real robot: nothing that reads it knows where the numbers come from.
import type { Kind, Obj, Robot, Vec3 } from "./robot";

export type Pose = { x: number; y: number; yaw: number }; // on the floor, world coordinates
export type Seen = {
  obj: Obj;
  label: string;
  kind: Kind;
  size: Vec3; // half extents; a ball's three values are its radius
  world: Vec3; // centre, world coordinates
  pos: Vec3; // centre, in the arm's frame
  yaw: number; // heading in the arm's frame
  upright: boolean;
  bottom: number; // lowest point above the floor
  top: number;
  footprint: number; // radius of the circle it covers on the floor
  held: boolean; // between both jaws
};
export type World = {
  base(): Pose; // where the arm's frame stands and faces
  objects(): Seen[];
  see(o: Obj): Seen;
};

export function simWorld(robot: Robot): World {
  const see = (o: Obj): Seen => {
    const f = robot.object(o);
    return { obj: o, label: o.label, kind: o.kind, size: o.size, world: f.world, pos: f.pos, yaw: f.yaw, upright: f.upright, bottom: f.bottom, top: f.top, footprint: robot.footprint(o), held: robot.gripped(o) };
  };
  return { base: robot.base, objects: () => robot.active().map(see), see };
}
