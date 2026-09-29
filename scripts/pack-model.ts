// Packs the MuJoCo model (scene, arm XML and the meshes it uses) into one
// gzip file, public/so101.pack.gz. Netlify does not compress .stl files, so
// this cuts the download from about 17 MB to about 6.5 MB. The browser
// unpacks it with DecompressionStream.
// Format: u32 little-endian header length, JSON header [[name, size], ...], then the files.
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";

const dir = new URL("../model/", import.meta.url).pathname;
const out = new URL("../public/", import.meta.url).pathname;

const arm = readFileSync(dir + "so101.xml", "utf8");
const meshes = [...arm.matchAll(/<mesh [^>]*file="([^"]+)"/g)].map((m) => "assets/" + m[1]);
const names = ["scene_web.xml", "so101.xml", ...meshes];
const files = names.map((name) => readFileSync(dir + name));

const header = new TextEncoder().encode(JSON.stringify(names.map((name, i) => [name, files[i].length])));
const size = new Uint8Array(4);
new DataView(size.buffer).setUint32(0, header.length, true);
const pack = gzipSync(Buffer.concat([size, header, ...files]), { level: 9 });

mkdirSync(out + "so101", { recursive: true });
writeFileSync(out + "so101.pack.gz", pack);
copyFileSync(dir + "LICENSE", out + "so101/LICENSE.txt");
console.log(`so101.pack.gz: ${names.length} files, ${(pack.length / 1e6).toFixed(1)} MB`);
