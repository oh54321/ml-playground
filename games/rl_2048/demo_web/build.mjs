// Package a static site; no Python server or external CDN is needed at runtime.
import { copyFile, mkdir, readFile, rm, writeFile } from "node:fs/promises";

const source = new URL("./", import.meta.url);
const output = new URL("dist/", source);
await rm(output, { recursive: true, force: true });
await mkdir(new URL("vendor/", output), { recursive: true });
for (const file of ["index.html", "style.css", "app.js", "backend.js", "game.js", "policy.onnx", "model.json"]) {
  await copyFile(new URL(file, source), new URL(file, output));
}
const runtime = new URL("node_modules/onnxruntime-web/", source);
for (const file of ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  await copyFile(new URL(`dist/${file}`, runtime), new URL(`vendor/${file}`, output));
}
// The npm package omits the license files; include its matching release notices.
const { version } = JSON.parse(await readFile(new URL("package.json", runtime), "utf8"));
for (const file of ["LICENSE", "ThirdPartyNotices.txt"]) {
  const response = await fetch(`https://raw.githubusercontent.com/microsoft/onnxruntime/v${version}/${file}`);
  if (!response.ok) throw new Error(`Could not download ${file}: ${response.status}`);
  await writeFile(new URL(`vendor/${file}`, output), await response.text());
}
console.log("Static demo written to games/rl_2048/demo_web/dist/");
