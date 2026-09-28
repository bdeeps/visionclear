// VisionClear's maths: a small computer-vision lab written from scratch (no downloads, no datasets).
// Everything the box shows is computed here, live, in your browser:
//  - a seeded random number generator, so every visit sees the same pictures;
//  - a tiny ray-marching renderer that turns a 3D mango, a light and a camera into RGB pixels
//    (chapter 1), using signed distance functions (Hart 1996, "Sphere tracing") and Lambert shading;
//  - a 2D picture maker that draws four made-up objects (mango, banana, chilli, coin) with random
//    size, angle, colour, background, light and sensor noise, plus a pixel mask and a box for each;
//  - hand-made filters: Sobel edges, the Harris corner response (Harris & Stephens 1988, k = 0.05)
//    and Laws' texture energy (Laws 1980, the L5/E5/S5/R5 masks);
//  - a convolutional neural network (CNN) with 3×3 convolutions (optionally dilated, Yu & Koltun
//    2016), ReLU, 2×2 max pooling and a dense softmax layer, trained with backpropagation and Adam
//    (Kingma & Ba 2014, arXiv:1412.6980). It can also return the gradient with respect to the
//    picture itself, which powers the saliency map and the adversarial attack (Goodfellow, Shlens &
//    Szegedy 2014, arXiv:1412.6572; iterative version: Kurakin, Goodfellow & Bengio 2016);
//  - data augmentation (random turn, flip, zoom, shift, brightness, colour cast and noise);
//  - intersection over union (IoU) and greedy non-maximum suppression (NMS), as in R-CNN and YOLO.
// This file has no three.js in it, so it also runs under Node for testing.

export const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const TAU = Math.PI * 2;

// ---------------------------------------------------------------- random numbers
// mulberry32: a tiny, well-known 32-bit PRNG. rng(seed)() gives numbers in [0, 1).
export function rng(seed = 1) {
  let a = seed >>> 0;
  const r = () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  r.gauss = () => { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * v); };
  r.int = (n) => Math.floor(r() * n);
  r.pick = (arr) => arr[Math.floor(r() * arr.length)];
  r.range = (a, b) => a + (b - a) * r();
  return r;
}
export function shuffle(arr, r) { for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; }
export const argmax = (a, from = 0) => { let k = from; for (let i = from + 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; };

// ---------------------------------------------------------------- pictures
// A picture is 3 planes (red, green, blue) of W×H numbers from 0 to 1, stored plane after plane
// ("CHW" order, the way most CNN libraries store them).
export const image = (w, h) => ({ w, h, d: new Float32Array(3 * w * h) });
export const px = (im, x, y) => { const n = im.w * im.h, i = y * im.w + x; return [im.d[i], im.d[n + i], im.d[2 * n + i]]; };
export const to255 = (v) => Math.round(clamp(v, 0, 1) * 255);
export function copyImage(im) { return { w: im.w, h: im.h, d: Float32Array.from(im.d) }; }
// Brightness as the eye weighs it (ITU-R BT.601 luma weights: 0.299 R + 0.587 G + 0.114 B).
export function grey(im) {
  const n = im.w * im.h, g = new Float32Array(n);
  for (let i = 0; i < n; i++) g[i] = 0.299 * im.d[i] + 0.587 * im.d[n + i] + 0.114 * im.d[2 * n + i];
  return g;
}
// Bilinear sample of one channel with the edge pixels repeated outside the picture.
function sample(im, c, x, y) {
  const W = im.w, H = im.h, o = c * W * H;
  x = clamp(x, 0, W - 1); y = clamp(y, 0, H - 1);
  const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1), fx = x - x0, fy = y - y0;
  return (im.d[o + y0 * W + x0] * (1 - fx) + im.d[o + y0 * W + x1] * fx) * (1 - fy) + (im.d[o + y1 * W + x0] * (1 - fx) + im.d[o + y1 * W + x1] * fx) * fy;
}
// Shrink or grow a picture to a new size (bilinear).
export function resize(im, w, h) {
  const out = image(w, h), sx = im.w / w, sy = im.h / h;
  for (let c = 0; c < 3; c++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out.d[c * w * h + y * w + x] = sample(im, c, (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5);
  return out;
}
export function crop(im, x0, y0, w, h) {
  const out = image(w, h);
  for (let c = 0; c < 3; c++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const X = clamp(x0 + x, 0, im.w - 1), Y = clamp(y0 + y, 0, im.h - 1);
    out.d[c * w * h + y * w + x] = im.d[c * im.w * im.h + Y * im.w + X];
  }
  return out;
}

// ---------------------------------------------------------------- chapter 1: a 3D mango, ray-marched
// A mango is an ellipsoid (half-widths 0.55, 0.8, 0.46 m-ish units) bent sideways a little, with a
// small stalk, sitting on a table in front of a wall. For each pixel we shoot a ray from the camera,
// step along it by the signed distance (sphere tracing), and shade what it hits: Lambert (diffuse)
// light, a soft ambient term, and a shadow ray towards the lamp. Colours are linear light, turned
// into 0–1 sRGB numbers at the end (IEC 61966-2-1 transfer curve), like a camera's image processor.
export const MANGO = { rx: 0.55, ry: 0.8, rz: 0.46, bend: 0.16, cy: 0.8 };
export const LIGHTS = {
  day: { name: 'Daylight', rgb: [1, 0.98, 0.95] },
  tube: { name: 'Tube light', rgb: [0.86, 0.97, 1.08] },
  sunset: { name: 'Sunset', rgb: [1.2, 0.78, 0.48] },
};
const srgb = (l) => (l <= 0.0031308 ? 12.92 * l : 1.055 * Math.pow(l, 1 / 2.4) - 0.055);
function sdMango(x, y, z) {
  const M = MANGO, py = y - M.cy, t = py / M.ry;
  const qx = (x - M.bend * (1 - t * t)) / M.rx, qy = py / M.ry, qz = z / M.rz;
  const k0 = Math.hypot(qx, qy, qz), k1 = Math.hypot(qx / M.rx, qy / M.ry, qz / M.rz);
  const body = k1 > 1e-6 ? (k0 * (k0 - 1)) / k1 : -Math.min(M.rx, M.ry, M.rz);
  // stalk: a short capsule on top
  const sx = x - 0.02, sy = clamp(y - (M.cy + M.ry - 0.04), 0, 0.12), sz = z;
  const stalk = Math.hypot(sx, y - (M.cy + M.ry - 0.04) - sy, sz) - 0.035;
  return body < stalk ? [body, 0] : [stalk, 1];
}
// The mango's own colour at a point: green near the stalk, yellow in the middle, a red blush on one cheek.
export function mangoAlbedo(x, y, z, part) {
  if (part === 1) return [0.12, 0.08, 0.03];
  const t = clamp((y - MANGO.cy) / MANGO.ry, -1, 1);
  const g = clamp((t - 0.25) * 1.6, 0, 1), blush = clamp((x + 0.15) * 1.8, 0, 1) * clamp(1 - Math.abs(t + 0.1) * 1.4, 0, 1) * clamp(z * 2 + 0.6, 0, 1);
  const base = [0.78 - 0.42 * g, 0.5 + 0.02 * g, 0.04 + 0.03 * g];
  return [base[0] + (0.62 - base[0]) * blush * 0.7, base[1] * (1 - 0.55 * blush), base[2]];
}
// opts: { w, h, turn (radians, mango spins about its upright axis), dist (camera distance), light
// (brightness 0.2–1.6), tint (LIGHTS key), elev (camera height angle) }. Returns an image (sRGB 0–1)
// and a mask of which pixels show the mango.
export function renderMango({ w = 32, h = 24, turn = 0, dist = 3.4, light = 1, tint = 'day', elev = 0.22, aa = 2 } = {}) {
  const im = image(w, h), mask = new Uint8Array(w * h), n = w * h;
  const LC = LIGHTS[tint]?.rgb || LIGHTS.day.rgb;
  const ct = Math.cos(turn), st = Math.sin(turn);
  // camera looks at the mango's middle from a point on a circle; the mango spins (we turn the ray instead)
  const tgt = [0, MANGO.cy - 0.05, 0], cam = [0, tgt[1] + Math.sin(elev) * dist, Math.cos(elev) * dist];
  const fw = [tgt[0] - cam[0], tgt[1] - cam[1], tgt[2] - cam[2]], fl = Math.hypot(...fw); fw[0] /= fl; fw[1] /= fl; fw[2] /= fl;
  const rt = [1, 0, 0], up = [rt[1] * fw[2] - rt[2] * fw[1], rt[2] * fw[0] - rt[0] * fw[2], rt[0] * fw[1] - rt[1] * fw[0]];
  const fov = 0.62, aspect = w / h;                                   // about 35° tall field of view
  const Ldir = [-0.45, 0.8, 0.55], Ll = Math.hypot(...Ldir); Ldir[0] /= Ll; Ldir[1] /= Ll; Ldir[2] /= Ll;
  const TABLE = 0, WALL = -1.6;
  // scene distance in "world" coordinates, with the mango turned by `turn`
  const sd = (x, y, z) => { const mx = ct * x + st * z, mz = -st * x + ct * z; return sdMango(mx, y, mz); };
  const march = (o, d, maxT) => {
    let t = 0.01;
    for (let k = 0; k < 64; k++) { const [s] = sd(o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t); if (s < 0.002) return t; t += s * 0.8; if (t > maxT) break; }
    return -1;
  };
  const shade = (o, d) => {
    let tHit = march(o, d, 12), col, hitM = false;
    // planes: table (y = 0) and wall (z = WALL)
    const tT = d[1] < 0 ? (TABLE - o[1]) / d[1] : 1e9, tW = d[2] < 0 ? (WALL - o[2]) / d[2] : 1e9;
    const tP = Math.min(tT, tW);
    let P, N, alb;
    if (tHit > 0 && tHit < tP) {
      hitM = true; P = [o[0] + d[0] * tHit, o[1] + d[1] * tHit, o[2] + d[2] * tHit];
      const e = 0.002, f0 = sd(P[0], P[1], P[2])[0];
      N = [sd(P[0] + e, P[1], P[2])[0] - f0, sd(P[0], P[1] + e, P[2])[0] - f0, sd(P[0], P[1], P[2] + e)[0] - f0];
      const nl = Math.hypot(...N) || 1; N = N.map((v) => v / nl);
      const part = sd(P[0], P[1], P[2])[1], mx = ct * P[0] + st * P[2], mz = -st * P[0] + ct * P[2];
      alb = mangoAlbedo(mx, P[1], mz, part);
    } else if (tP < 1e8) {
      P = [o[0] + d[0] * tP, o[1] + d[1] * tP, o[2] + d[2] * tP];
      if (tT < tW) { N = [0, 1, 0]; const wood = 0.5 + 0.5 * Math.sin(P[0] * 9 + Math.sin(P[2] * 3) * 2); alb = [0.34 + 0.06 * wood, 0.22 + 0.04 * wood, 0.13 + 0.02 * wood]; }
      else { N = [0, 0, 1]; alb = [0.42, 0.44, 0.47]; }
    } else return [[0.05, 0.05, 0.06], false];
    const ndl = Math.max(0, N[0] * Ldir[0] + N[1] * Ldir[1] + N[2] * Ldir[2]);
    let sh = 1;
    if (ndl > 0) { const so = [P[0] + N[0] * 0.01, P[1] + N[1] * 0.01, P[2] + N[2] * 0.01]; if (march(so, Ldir, 6) > 0) sh = 0.12; }
    const diff = ndl * sh * light * 1.1, amb = 0.12 * (0.6 + 0.4 * light);
    col = [0, 1, 2].map((c) => alb[c] * (diff * LC[c] + amb));
    // a little specular shine on the mango's waxy skin (Blinn-Phong, exponent 40)
    if (hitM && sh > 0.5) { const hv = [Ldir[0] - d[0], Ldir[1] - d[1], Ldir[2] - d[2]], hl = Math.hypot(...hv); const sp = Math.pow(Math.max(0, (N[0] * hv[0] + N[1] * hv[1] + N[2] * hv[2]) / hl), 40) * 0.25 * light; col = col.map((v, c) => v + sp * LC[c]); }
    return [col, hitM];
  };
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const acc = [0, 0, 0]; let hits = 0;
    for (let sy = 0; sy < aa; sy++) for (let sx = 0; sx < aa; sx++) {
      const u = ((x + (sx + 0.5) / aa) / w * 2 - 1) * Math.tan(fov / 2) * aspect, v = (1 - (y + (sy + 0.5) / aa) / h * 2) * Math.tan(fov / 2);
      const d = [fw[0] + rt[0] * u + up[0] * v, fw[1] + rt[1] * u + up[1] * v, fw[2] + rt[2] * u + up[2] * v], dl = Math.hypot(...d);
      const [c, hm] = shade(cam, [d[0] / dl, d[1] / dl, d[2] / dl]);
      acc[0] += c[0]; acc[1] += c[1]; acc[2] += c[2]; if (hm) hits++;
    }
    const i = y * w + x, k = aa * aa;
    for (let c = 0; c < 3; c++) im.d[c * n + i] = clamp(srgb(Math.max(0, acc[c] / k)), 0, 1);
    mask[i] = hits * 2 >= k ? 1 : 0;
  }
  return { im, mask };
}

// ---------------------------------------------------------------- the four made-up objects
// Each shape is a signed distance in its own units (negative inside), about 1 unit from middle to
// edge. `part` picks a colour: 0 = body, 1 = stalk or cap, 2 = rim, 3 = emblem.
export const CLASSES = ['nothing', 'mango', 'banana', 'chilli', 'coin'];
export const CLASS_COL = ['#8a93a6', '#ffb547', '#ffe45c', '#ff5a5a', '#b9c3d6'];
function sdArc(u, v, cx, R, span, thick, capR) {
  const dx = u - cx, r = Math.hypot(dx, v), a = Math.atan2(v, -dx), t = clamp(a / span, -1, 1);
  if (Math.abs(a) <= span) return [Math.abs(r - R) - thick(t), t];
  const ex = cx - R * Math.cos(span * Math.sign(a)), ey = R * Math.sin(span * Math.sign(a));
  return [Math.hypot(u - ex, v - ey) - capR, t];
}
export function shapeSD(cls, u, v) {
  if (cls === 1) {                                                   // mango: a bent oval with a stalk
    const t = clamp(v / 0.9, -1, 1), qx = (u - 0.14 * (1 - t * t) + 0.05) / 0.62, qy = v / 0.9;
    const body = (Math.hypot(qx, qy) - 1) * 0.62, stalk = Math.hypot(u - 0.0, v + 0.93) - 0.08;
    return stalk < body ? [stalk, 1] : [body, 0];
  }
  if (cls === 2) {                                                   // banana: a thick curved band, both ends tapered
    const [d, t] = sdArc(u + 0.15, v, 1.35, 1.35, 0.62, (t) => 0.05 + 0.25 * Math.sqrt(Math.max(0, 1 - t * t)), 0.05);
    return [d, t > 0.86 ? 1 : 0];
  }
  if (cls === 3) {                                                   // chilli: long, thin, fat at the stalk end
    const [d, t] = sdArc(u + 0.02, v, 2.4, 2.4, 0.38, (t) => 0.02 + 0.17 * Math.pow((1 - t) / 2, 0.55), 0.02);
    const sx = u - 0.2, sy = v + 0.92, stalk = Math.hypot(sx - clamp(sx, 0, 0.16), sy - clamp(sy, -0.16, 0) * 1) - 0.045;
    const cap = Math.hypot(u - 0.13, v + 0.86) - 0.17;
    const g = Math.min(stalk, cap);
    return g < d ? [g, 1] : [d, 0];
  }
  if (cls === 4) {                                                   // coin: a disc with a rim and an emblem
    const r = Math.hypot(u, v), d = r - 0.72;
    const part = r > 0.6 ? 2 : Math.abs(r - 0.32) < 0.05 || r < 0.1 ? 3 : 0;
    return [d, part];
  }
  return [1e9, 0];
}
// Typical colours [body, part1, rim, emblem] for each class, before random tweaks. Several look alike
// on purpose (a green mango, a green banana, a green chilli), so colour alone can't give the answer.
export const PALETTE = {
  1: [[[0.95, 0.72, 0.16], [0.3, 0.2, 0.05]], [[0.45, 0.62, 0.16], [0.25, 0.18, 0.05]], [[0.96, 0.5, 0.12], [0.3, 0.2, 0.05]]],
  2: [[[0.98, 0.86, 0.25], [0.28, 0.2, 0.08]], [[0.62, 0.78, 0.22], [0.28, 0.2, 0.08]], [[0.93, 0.8, 0.3], [0.2, 0.15, 0.06]]],
  3: [[[0.86, 0.1, 0.08], [0.2, 0.5, 0.12]], [[0.2, 0.55, 0.12], [0.15, 0.4, 0.1]], [[0.75, 0.08, 0.1], [0.25, 0.5, 0.15]]],
  4: [[[0.76, 0.77, 0.8], [0.9, 0.9, 0.92]], [[0.83, 0.66, 0.3], [0.93, 0.78, 0.4]], [[0.72, 0.45, 0.28], [0.85, 0.55, 0.36]]],
};
export const BACKGROUNDS = [[0.4, 0.27, 0.16], [0.25, 0.3, 0.38], [0.72, 0.7, 0.66], [0.2, 0.36, 0.26], [0.14, 0.14, 0.16], [0.55, 0.42, 0.3], [0.36, 0.22, 0.3]];
// Smooth random texture (value noise) so backgrounds look like cloth or wood, not flat paint.
function valueNoise(r, gw, gh) {
  const g = Array.from({ length: (gw + 1) * (gh + 1) }, () => r() * 2 - 1);
  return (x, y) => { const X = clamp(x, 0, gw - 0.001), Y = clamp(y, 0, gh - 0.001), x0 = Math.floor(X), y0 = Math.floor(Y), fx = X - x0, fy = Y - y0, sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy), a = g[y0 * (gw + 1) + x0], b = g[y0 * (gw + 1) + x0 + 1], c = g[(y0 + 1) * (gw + 1) + x0], d = g[(y0 + 1) * (gw + 1) + x0 + 1]; return (a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy; };
}
// Make an object description with random looks.
export function makeObj(cls, r, { x, y, s, ang, col = null } = {}) {
  const pal = col ?? r.pick(PALETTE[cls]);
  const j = 0.85 + 0.3 * r(), hue = [1 + (r() - 0.5) * 0.12, 1 + (r() - 0.5) * 0.12, 1 + (r() - 0.5) * 0.12];
  return { cls, x, y, s, ang, body: pal[0].map((v, c) => clamp(v * j * hue[c], 0, 1)), part: pal[1], lx: -0.5 + (r() - 0.5) * 0.4, ly: -0.7 };
}
// Draw a scene: a background, some objects, a light level and colour cast, and sensor noise.
// Returns the picture, a mask (which object, 0 = none, at each pixel) and a tight box for each object.
export function drawScene(w, h, objs, { bg = [0.4, 0.3, 0.2], tex = 0.07, grad = 0.12, light = 1, cast = [1, 1, 1], noise = 0.02, seed = 1 } = {}) {
  const r = rng(seed), im = image(w, h), n = w * h, mask = new Uint8Array(n), cover = new Float32Array(n);
  const nz = valueNoise(r, Math.ceil(w / 6) + 1, Math.ceil(h / 6) + 1), gdir = r() * TAU, gx = Math.cos(gdir), gy = Math.sin(gdir);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x, k = 1 + tex * nz(x / 6, y / 6) * 2 + grad * (((x / w) - 0.5) * gx + ((y / h) - 0.5) * gy) * 2;
    for (let c = 0; c < 3; c++) im.d[c * n + i] = bg[c] * k;
  }
  const boxes = [];
  objs.forEach((o, oi) => {
    const ca = Math.cos(o.ang), sa = Math.sin(o.ang), R = Math.ceil(o.s * 1.25) + 1;
    let bx0 = 1e9, by0 = 1e9, bx1 = -1, by1 = -1;
    for (let y = Math.max(0, Math.floor(o.y - R)); y < Math.min(h, Math.ceil(o.y + R)); y++) for (let x = Math.max(0, Math.floor(o.x - R)); x < Math.min(w, Math.ceil(o.x + R)); x++) {
      const dx = (x + 0.5 - o.x) / o.s, dy = (y + 0.5 - o.y) / o.s, u = ca * dx + sa * dy, v = -sa * dx + ca * dy;
      const [d, part] = shapeSD(o.cls, u, v), a = clamp(0.5 - d * o.s, 0, 1);
      if (a <= 0) continue;
      const i = y * w + x;
      // fake 3D: lighter towards the light, darker at the rim
      const inside = clamp(-d * 5, 0, 1), sh = 0.72 + 0.4 * clamp(-(dx * o.lx + dy * o.ly) * 0.8 + 0.3, 0, 1) * inside + 0.08 * inside;
      let col = part === 1 ? o.part : part === 2 ? o.body.map((v) => v * 1.12) : part === 3 ? o.body.map((v) => v * 0.8) : o.body;
      if (o.cls === 4) { const shine = Math.max(0, 1 - Math.hypot(dx + 0.35, dy + 0.35) * 1.6); col = col.map((v) => v + 0.25 * shine); }
      for (let c = 0; c < 3; c++) im.d[c * n + i] = im.d[c * n + i] * (1 - a) + clamp(col[c] * sh, 0, 1.2) * a;
      if (a >= 0.5) { mask[i] = o.cls; cover[i] = oi + 1; bx0 = Math.min(bx0, x); by0 = Math.min(by0, y); bx1 = Math.max(bx1, x + 1); by1 = Math.max(by1, y + 1); }
    }
    // a soft shadow under the object, to the lower right
    if (bx1 > 0) boxes.push({ cls: o.cls, x0: bx0, y0: by0, x1: bx1, y1: by1 });
  });
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) im.d[c * n + i] = clamp(im.d[c * n + i] * light * cast[c] + r.gauss() * noise, 0, 1);
  return { im, mask, boxes };
}

// ---------------------------------------------------------------- datasets
export let BOXW = 4; export const setBoxW = (v) => { BOXW = v; };
export const SIZE = 20;            // the recogniser looks at 20 × 20 colour pictures: 1,200 numbers
// One training or test picture of class cls. 'train' pictures are fairly tidy: upright-ish, normal
// light. 'wild' pictures are the real world: any angle, dim or bright light, colour casts, noise.
export function makeSample(cls, r, mode = 'train', opt = {}) {
  const W = SIZE, wild = mode === 'wild';
  const light = opt.light ?? (wild ? r.range(0.62, 1.35) : r.range(0.82, 1.18));
  const cast = opt.cast ?? (wild ? [r.range(0.85, 1.15), r.range(0.88, 1.1), r.range(0.8, 1.2)] : [r.range(0.95, 1.05), 1, r.range(0.95, 1.05)]);
  const bg = opt.bg ?? r.pick(BACKGROUNDS).map((v) => clamp(v * r.range(0.8, 1.2), 0, 1));
  const noise = opt.noise ?? (wild ? r.range(0.01, 0.06) : r.range(0.005, 0.025));
  const objs = [];
  if (cls > 0) {
    const s = opt.s ?? (wild ? r.range(5.2, 8.4) : r.range(6, 7.6)), ang = opt.ang ?? (wild ? r() * TAU : (r() - 0.5) * 0.7);
    objs.push(makeObj(cls, r, { x: W / 2 + (opt.dx ?? (r() - 0.5) * 4), y: W / 2 + (opt.dy ?? (r() - 0.5) * 4), s, ang, col: opt.col }));
  } else if (r() < 0.6) {
    // "nothing": an object that is mostly out of the frame, or only a corner of one
    // (near misses, 5 to 9 pixels off-centre, teach the detector to fire only on an object's middle)
    const a = r() * TAU, dist = r() < 0.5 ? r.range(5, 9) : r.range(9, 15);
    objs.push(makeObj(1 + r.int(4), r, { x: W / 2 + Math.cos(a) * dist, y: W / 2 + Math.sin(a) * dist, s: r.range(5.5, 8.5), ang: r() * TAU }));
  }
  const sc = drawScene(W, W, objs, { bg, light, cast, noise, seed: r.int(1e9) });
  if (opt.full) return sc;
  return sc.im;
}
// perClass pictures of each class. X: pictures, Y: labels, M: masks (for boxes after augmentation).
export function dataset(perClass, seed, mode, classes = [0, 1, 2, 3, 4]) {
  const r = rng(seed), X = [], Y = [], M = [];
  let scene = null;
  for (let k = 0; k < perClass; k++) for (const c of classes) {
    let sc;
    // Half the "nothing" pictures are cut from busy scenes, away from any object's middle: the
    // bits between objects that a sliding-window detector must learn to ignore.
    if (c === 0 && k % 2 && mode === 'train') {
      if (!scene || k % 8 === 1) scene = makeDetScene(r.int(1e9));
      for (let t = 0; t < 30; t++) {
        const x = r.int(SCENE_W - SIZE + 1), y = r.int(SCENE_H - SIZE + 1);
        if (scene.objs.every((o) => Math.hypot(o.x - x - SIZE / 2, o.y - y - SIZE / 2) > 5)) { sc = { im: crop(scene.im, x, y, SIZE, SIZE), mask: new Uint8Array(SIZE * SIZE) }; break; }
      }
    }
    sc ||= makeSample(c, r, mode, { full: true });
    X.push(sc.im); Y.push(c); M.push(sc.mask);
  }
  return { X, Y, M };
}
// The tight box around the mask pixels, as fractions of the picture: [x0, y0, x1, y1].
export function maskBox(mask, w, h) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x + 1); y1 = Math.max(y1, y + 1); }
  return x1 < 0 ? null : [x0 / w, y0 / h, x1 / w, y1 / h];
}
// A biased dataset, for the "shortcut" lesson: every picture of a class sits on that class's own
// background colour. A network can score well by learning the background, not the object.
export const BIAS_BG = { 1: [0.2, 0.3, 0.62], 2: [0.18, 0.5, 0.22], 3: [0.5, 0.26, 0.55], 4: [0.52, 0.34, 0.18] };
export function biasedSample(cls, r, bgCls = cls) {
  return makeSample(cls, r, 'train', { bg: BIAS_BG[bgCls].map((v) => clamp(v * r.range(0.9, 1.1), 0, 1)), s: r.range(4.6, 6.4), ang: r() * TAU });
}

// Data augmentation: make a new, slightly different picture from an old one, so the network sees
// the same object turned, flipped, zoomed, shifted, lit differently and noisier.
export function augment(im, r, strength = 1, mask = null) {
  const W = im.w, H = im.h, n = W * H, out = image(W, H), om = mask ? new Uint8Array(n) : null;
  const a = r() * TAU * strength, flip = r() < 0.5 ? -1 : 1, zoom = 1 + (r() - 0.5) * 0.24 * strength;
  const tx = (r() - 0.5) * 2 * strength, ty = (r() - 0.5) * 2 * strength;
  const bright = 1 + (r() - 0.5) * 0.8 * strength, cast = [0, 1, 2].map(() => 1 + (r() - 0.5) * 0.2 * strength), noise = r() * 0.04 * strength;
  const ca = Math.cos(a), sa = Math.sin(a), cx = W / 2, cy = H / 2;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const dx = (x + 0.5 - cx - tx) / zoom * flip, dy = (y + 0.5 - cy - ty) / zoom;
    const sx = cx + ca * dx - sa * dy - 0.5, sy = cy + sa * dx + ca * dy - 0.5;
    for (let c = 0; c < 3; c++) out.d[c * n + y * W + x] = clamp(sample(im, c, sx, sy) * bright * cast[c] + r.gauss() * noise, 0, 1);
    if (om) { const X = Math.round(sx), Y = Math.round(sy); om[y * W + x] = X >= 0 && Y >= 0 && X < W && Y < H ? mask[Y * W + X] : 0; }
  }
  if (om) out.mask = om;
  return out;
}
// Change a picture's light: multiply every number (and optionally tint it).
export function relight(im, k, cast = [1, 1, 1]) {
  const out = copyImage(im), n = im.w * im.h;
  for (let c = 0; c < 3; c++) for (let i = 0; i < n; i++) out.d[c * n + i] = clamp(im.d[c * n + i] * k * cast[c], 0, 1);
  return out;
}

// ---------------------------------------------------------------- hand-made filters (chapter 2)
// All work on a brightness map g (W×H), with the edge pixels repeated outside the picture.
export const SOBEL_X = [-1, 0, 1, -2, 0, 2, -1, 0, 1], SOBEL_Y = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
export function conv2(g, w, h, k) {
  const K = Math.round(Math.sqrt(k.length)), c = (K - 1) / 2, out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = 0; j < K; j++) for (let i = 0; i < K; i++) s += k[j * K + i] * g[clamp(y + j - c, 0, h - 1) * w + clamp(x + i - c, 0, w - 1)];
    out[y * w + x] = s;
  }
  return out;
}
export function edges(g, w, h) {
  const gx = conv2(g, w, h, SOBEL_X), gy = conv2(g, w, h, SOBEL_Y), m = new Float32Array(w * h);
  for (let i = 0; i < m.length; i++) m[i] = Math.hypot(gx[i], gy[i]);
  return { gx, gy, m };
}
// Harris corner response R = det(M) − k·trace(M)², where M sums gx², gx·gy and gy² over a 3×3
// neighbourhood. R is big and positive only where the picture changes in two directions at once.
export function harris(g, w, h, k = 0.05) {
  const { gx, gy } = edges(g, w, h), n = w * h, a = new Float32Array(n), b = new Float32Array(n), c = new Float32Array(n);
  for (let i = 0; i < n; i++) { a[i] = gx[i] * gx[i]; b[i] = gx[i] * gy[i]; c[i] = gy[i] * gy[i]; }
  const box = [1, 2, 1, 2, 4, 2, 1, 2, 1].map((v) => v / 16);
  const A = conv2(a, w, h, box), B = conv2(b, w, h, box), C = conv2(c, w, h, box), R = new Float32Array(n);
  for (let i = 0; i < n; i++) R[i] = A[i] * C[i] - B[i] * B[i] - k * (A[i] + C[i]) ** 2;
  return R;
}
// Laws' texture energy: filter with the 5×5 "ripple" mask R5ᵀ·R5 (R5 = [1, −4, 6, −4, 1]), which
// answers strongly to fine, busy patterns, then average the size of the answer over 5×5.
export function lawsTexture(g, w, h) {
  const R5 = [1, -4, 6, -4, 1], E5 = [-1, -2, 0, 2, 1];
  const outer = (a, b) => a.flatMap((x) => b.map((y) => x * y));
  const f1 = conv2(g, w, h, outer(R5, R5)), f2 = conv2(g, w, h, outer(E5, E5));
  const e = new Float32Array(w * h);
  for (let i = 0; i < e.length; i++) e[i] = Math.abs(f1[i]) / 36 + Math.abs(f2[i]) / 9;
  return conv2(e, w, h, new Array(25).fill(1 / 25));
}
// 2×2 max pooling: keep the biggest of each 2×2 block, so the map halves in width and height.
export function maxPool(m, w, h) {
  const W = w >> 1, H = h >> 1, out = new Float32Array(W * H), arg = new Int32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let best = -1e9, bi = 0;
    for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) { const k = (2 * y + j) * w + 2 * x + i; if (m[k] > best) { best = m[k]; bi = k; } }
    out[y * W + x] = best; arg[y * W + x] = bi;
  }
  return { out, arg, w: W, h: H };
}

// ---------------------------------------------------------------- the CNN
// Layers work on "tensors": { c, h, w, d: Float32Array(c·h·w) }.
const T = (c, h, w) => ({ c, h, w, d: new Float32Array(c * h * w) });
class Conv {
  constructor(inC, outC, k, r, dil = 1) {
    this.inC = inC; this.outC = outC; this.k = k; this.dil = dil;
    const fan = inC * k * k, sc = Math.sqrt(2 / fan);                 // He initialisation (He et al. 2015)
    this.W = Float32Array.from({ length: outC * fan }, () => r.gauss() * sc); this.b = new Float32Array(outC);
    this.gW = new Float32Array(this.W.length); this.gb = new Float32Array(outC);
  }
  forward(x) {
    const { inC, outC, k, dil } = this, H = x.h, Wd = x.w, n = H * Wd, y = T(outC, H, Wd), c0 = (k - 1) / 2;
    this.x = x;
    for (let o = 0; o < outC; o++) {
      const Y = y.d, Xd = x.d, yo = o * n; Y.fill(this.b[o], yo, yo + n);
      for (let i = 0; i < inC; i++) for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
        const wv = this.W[((o * inC + i) * k + ky) * k + kx], oy = (ky - c0) * dil, ox = (kx - c0) * dil, xi = i * n;
        const ya = Math.max(0, -oy), yb = Math.min(H, H - oy), xa = Math.max(0, -ox), xb = Math.min(Wd, Wd - ox);
        for (let yy = ya; yy < yb; yy++) { const yr = yo + yy * Wd, xr = xi + (yy + oy) * Wd + ox; for (let xx = xa; xx < xb; xx++) Y[yr + xx] += wv * Xd[xr + xx]; }
      }
    }
    return y;
  }
  backward(dy, needDx = true) {
    const { inC, outC, k, dil } = this, x = this.x, H = x.h, Wd = x.w, n = H * Wd, c0 = (k - 1) / 2, dx = needDx ? T(inC, H, Wd) : null;
    for (let o = 0; o < outC; o++) {
      const yo = o * n; let sb = 0; for (let q = 0; q < n; q++) sb += dy.d[yo + q]; this.gb[o] += sb;
      for (let i = 0; i < inC; i++) for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
        const wi = ((o * inC + i) * k + ky) * k + kx, wv = this.W[wi], oy = (ky - c0) * dil, ox = (kx - c0) * dil, xi = i * n;
        const ya = Math.max(0, -oy), yb = Math.min(H, H - oy), xa = Math.max(0, -ox), xb = Math.min(Wd, Wd - ox);
        let g = 0; const D = dy.d, Xd = x.d;
        for (let yy = ya; yy < yb; yy++) {
          const yr = yo + yy * Wd, xr = xi + (yy + oy) * Wd + ox;
          for (let xx = xa; xx < xb; xx++) g += D[yr + xx] * Xd[xr + xx];
          if (dx) { const DX = dx.d; for (let xx = xa; xx < xb; xx++) DX[xr + xx] += D[yr + xx] * wv; }
        }
        this.gW[wi] += g;
      }
    }
    return dx;
  }
  get params() { return [[this.W, this.gW], [this.b, this.gb]]; }
}
class ReLU {
  forward(x) { const d = new Float32Array(x.d.length), s = x.d; for (let i = 0; i < d.length; i++) d[i] = s[i] > 0 ? s[i] : 0; this.y = { c: x.c, h: x.h, w: x.w, d }; return this.y; }
  backward(dy) { const d = new Float32Array(dy.d.length), y = this.y.d, g = dy.d; for (let i = 0; i < d.length; i++) d[i] = y[i] > 0 ? g[i] : 0; return { c: dy.c, h: dy.h, w: dy.w, d }; }
  get params() { return []; }
}
class Pool {
  forward(x) {
    const p = T(x.c, x.h >> 1, x.w >> 1), arg = new Int32Array(p.d.length), H = p.h, W = p.w;
    for (let c = 0; c < x.c; c++) for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) {
      let best = -1e30, bi = 0;
      for (let j = 0; j < 2; j++) for (let i = 0; i < 2; i++) { const k = c * x.h * x.w + (2 * y + j) * x.w + 2 * xx + i; if (x.d[k] > best) { best = x.d[k]; bi = k; } }
      const o = c * H * W + y * W + xx; p.d[o] = best; arg[o] = bi;
    }
    this.x = x; this.arg = arg; return p;
  }
  backward(dy) { const dx = T(this.x.c, this.x.h, this.x.w); for (let o = 0; o < dy.d.length; o++) dx.d[this.arg[o]] += dy.d[o]; return dx; }
  get params() { return []; }
}
class Dense {
  constructor(nIn, nOut, r) {
    this.nIn = nIn; this.nOut = nOut; const sc = Math.sqrt(1 / nIn);
    this.W = Float32Array.from({ length: nIn * nOut }, () => r.gauss() * sc); this.b = new Float32Array(nOut);
    this.gW = new Float32Array(this.W.length); this.gb = new Float32Array(nOut);
  }
  forward(x) {
    this.x = x; const y = T(this.nOut, 1, 1);
    for (let j = 0; j < this.nOut; j++) { let s = this.b[j]; const o = j * this.nIn; for (let i = 0; i < this.nIn; i++) s += this.W[o + i] * x.d[i]; y.d[j] = s; }
    return y;
  }
  backward(dy) {
    const x = this.x, dx = { ...x, d: new Float32Array(x.d.length) };
    for (let j = 0; j < this.nOut; j++) { const g = dy.d[j], o = j * this.nIn; this.gb[j] += g; for (let i = 0; i < this.nIn; i++) { this.gW[o + i] += g * x.d[i]; dx.d[i] += g * this.W[o + i]; } }
    return dx;
  }
  get params() { return [[this.W, this.gW], [this.b, this.gb]]; }
}
export function softmax(z) { let m = -1e30; for (const v of z) m = Math.max(m, v); const e = Array.from(z, (v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return Float32Array.from(e, (v) => v / s); }

// A network: a list of layers. kind 'class' ends in softmax over classes; 'seg' gives one
// object-or-not score per pixel (sigmoid), for masks.
export class Net {
  constructor(layers, kind = 'class') { this.layers = layers; this.kind = kind; this.t = 0; this.opt = null; }
  // withBox: 4 extra outputs guess the object's box inside the picture (x0, y0, x1, y1 as fractions),
  // like the box regression of R-CNN detectors (Girshick et al. 2014).
  static classifier(nClass, seed = 1, { c1 = 8, c2 = 16, hidden = 0, withBox = false } = {}) {
    const r = rng(seed), nF = c2 * (SIZE / 4) ** 2, nO = nClass + (withBox ? 4 : 0);
    const head = hidden ? [new Dense(nF, hidden, r), new ReLU(), new Dense(hidden, nO, r)] : [new Dense(nF, nO, r)];
    const net = new Net([new Conv(3, c1, 3, r), new ReLU(), new Pool(), new Conv(c1, c2, 3, r), new ReLU(), new Pool(), ...head]);
    net.nC = nClass; net.withBox = withBox; return net;
  }
  // Fully convolutional: 3×3 convolutions with growing gaps (dilation 1, 2, 4) see a 15-pixel-wide
  // patch around each pixel, then a 1×1 convolution scores "object or not" for every pixel.
  static segmenter(seed = 2) {
    const r = rng(seed);
    return new Net([new Conv(3, 8, 3, r, 1), new ReLU(), new Conv(8, 8, 3, r, 2), new ReLU(), new Conv(8, 8, 3, r, 4), new ReLU(), new Conv(8, 1, 1, r)], 'seg');
  }
  get params() { let n = 0; for (const L of this.layers) for (const [p] of L.params) n += p.length; return n; }
  // Run the picture through; keep every layer's output (for feature-map pictures).
  forward(im) {
    let x = { c: 3, h: im.h, w: im.w, d: im.d }; this.acts = [x];
    for (const L of this.layers) { x = L.forward(x); this.acts.push(x); }
    if (this.kind === 'class') { const nC = this.nC || x.d.length; this.out = softmax(x.d.subarray(0, nC)); this.box = this.withBox ? Array.from(x.d.subarray(nC, nC + 4)) : null; }
    else this.out = x.d.map((v) => 1 / (1 + Math.exp(-v)));
    return this.out;
  }
  // Gradient of the loss for label y (a class, or a mask for 'seg'); returns d(loss)/d(picture).
  backward(y, needInput = false, box = null) {
    const last = this.acts[this.acts.length - 1], g = { ...last, d: new Float32Array(last.d.length) };
    if (this.kind === 'class') {
      for (let j = 0; j < this.out.length; j++) g.d[j] = this.out[j] - (j === y ? 1 : 0);
      if (box && this.withBox) for (let j = 0; j < 4; j++) g.d[this.nC + j] = BOXW * (this.box[j] - box[j]);   // squared-error loss, weight 2
    }
    else { const n = g.d.length; for (let j = 0; j < n; j++) g.d[j] = (this.out[j] - (y[j] ? 1 : 0)) / n; }
    let d = g;
    for (let l = this.layers.length - 1; l >= 0; l--) { const L = this.layers[l]; d = L instanceof Conv ? L.backward(d, l > 0 || needInput) : L.backward(d); }
    return d;
  }
  loss(y) { if (this.kind === 'class') return -Math.log(Math.max(1e-7, this.out[y])); let s = 0; for (let j = 0; j < this.out.length; j++) s -= Math.log(Math.max(1e-7, y[j] ? this.out[j] : 1 - this.out[j])); return s / this.out.length; }
  zero() { for (const L of this.layers) for (const [, g] of L.params) g.fill(0); }
  // Adam: each weight gets its own step size from running averages of its gradient and its square.
  step(lr, n = 1) {
    if (!this.opt) this.opt = this.layers.flatMap((L) => L.params.map(([p]) => [new Float32Array(p.length), new Float32Array(p.length)]));
    this.t++;
    const b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** this.t, c2 = 1 - b2 ** this.t; let k = 0;
    for (const L of this.layers) for (const [p, g] of L.params) {
      const [m, v] = this.opt[k++];
      for (let i = 0; i < p.length; i++) { const gi = g[i] / n; m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi; p[i] -= (lr * m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8); }
    }
  }
  predict(im) { return this.forward(im); }
  // Saliency: how much each pixel would change the winning score (size of the input gradient).
  saliency(im, cls) {
    this.forward(im); const d = this.backward(cls, true).d, n = im.w * im.h, s = new Float32Array(n);
    for (let i = 0; i < n; i++) s[i] = Math.abs(d[i]) + Math.abs(d[n + i]) + Math.abs(d[2 * n + i]);
    return s;
  }
  firstKernels() { const L = this.layers[0]; return { W: L.W, n: L.outC, k: L.k }; }
}

// Iterative fast gradient sign attack: nudge every number by ±eps/steps in the direction that makes
// the right answer less likely, several times, never moving any number by more than eps in total.
export function attack(net, im, label, eps, steps = 10) {
  let adv = copyImage(im); const a = eps / Math.max(1, steps) * 1.5;
  for (let s = 0; s < steps; s++) {
    net.forward(adv); const g = net.backward(label, true).d;
    for (let i = 0; i < adv.d.length; i++) adv.d[i] = clamp(clamp(adv.d[i] + a * Math.sign(g[i]), im.d[i] - eps, im.d[i] + eps), 0, 1);
  }
  return adv;
}

// ---------------------------------------------------------------- boxes
export const area = (b) => Math.max(0, b.x1 - b.x0) * Math.max(0, b.y1 - b.y0);
export function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0)), iy = Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0)), inter = ix * iy;
  return inter / Math.max(1e-9, area(a) + area(b) - inter);
}
// Greedy NMS, recorded step by step: take the most confident box left, keep it, and throw away every
// other box of the same class that overlaps it by more than the IoU limit. Repeat.
export function nmsSteps(dets, iouMax = 0.4) {
  const order = dets.map((d, i) => i).sort((a, b) => dets[b].conf - dets[a].conf), state = dets.map(() => 'wait'), steps = [];
  for (const i of order) {
    if (state[i] !== 'wait') continue;
    state[i] = 'keep'; const killed = [];
    for (const j of order) if (state[j] === 'wait' && dets[j].cls === dets[i].cls && iou(dets[i], dets[j]) > iouMax) { state[j] = 'drop'; killed.push(j); }
    steps.push({ keep: i, killed });
  }
  return steps;
}

// ---------------------------------------------------------------- the detection scene (chapter 4)
export const SCENE_W = 96, SCENE_H = 64;
export function makeDetScene(seed) {
  const r = rng(seed), objs = [], n = 3 + r.int(3);
  let tries = 0;
  while (objs.length < n && tries++ < 200) {
    const s = r.range(5.6, 8.2), x = r.range(10, SCENE_W - 10), y = r.range(10, SCENE_H - 10);
    if (objs.some((o) => Math.hypot(o.x - x, o.y - y) < (o.s + s) * 1.35)) continue;
    objs.push(makeObj(1 + r.int(4), r, { x, y, s, ang: r() * TAU }));
  }
  const bg = r.pick(BACKGROUNDS);
  return { ...drawScene(SCENE_W, SCENE_H, objs, { bg, light: r.range(0.9, 1.1), noise: 0.02, seed: r.int(1e9) }), objs };
}
// Training crops for the mask network: 24×24 pieces of busy scenes, with their true masks.
export function segSample(r) {
  const W = 24, objs = [], n = 1 + r.int(3);
  for (let k = 0; k < n; k++) objs.push(makeObj(1 + r.int(4), r, { x: r.range(2, W - 2), y: r.range(2, W - 2), s: r.range(5, 8.5), ang: r() * TAU }));
  const bg = r.pick(BACKGROUNDS).map((v) => clamp(v * r.range(0.8, 1.2), 0, 1));
  const { im, mask } = drawScene(W, W, objs, { bg, light: r.range(0.75, 1.25), noise: r.range(0.01, 0.04), seed: r.int(1e9) });
  return { im, mask };
}

// ---------------------------------------------------------------- vision transformer patches (chapter 6)
// ViT cuts a picture into square patches and treats each one as a "word" (Dosovitskiy et al. 2020:
// 224 × 224 pictures in 16 × 16 patches give 196 tokens).
export function patches(im, p) {
  const out = [];
  for (let y = 0; y + p <= im.h; y += p) for (let x = 0; x + p <= im.w; x += p) out.push({ x, y, im: crop(im, x, y, p, p) });
  return out;
}

// ---------------------------------------------------------------- QR-style finder patterns (chapter 6)
// A QR code's three corner squares (7 × 7 modules: dark ring, light ring, dark 3 × 3 core) cross any
// straight line through their middle as dark–light–dark–light–dark runs in the ratio 1:1:3:1:1
// (ISO/IEC 18004). Scanners look for exactly that, row by row, then check the column too. This makes
// a QR-style symbol (version 1 layout, 21 × 21 modules, random data, so it is not a real, readable
// code) and finds its finder patterns with that classic, non-learning method.
export function qrSymbol(seed) {
  const r = rng(seed), N = 21, m = new Uint8Array(N * N);
  for (let i = 0; i < N * N; i++) m[i] = r() < 0.5 ? 1 : 0;
  const finder = (ox, oy) => { for (let y = -1; y <= 7; y++) for (let x = -1; x <= 7; x++) { const X = ox + x, Y = oy + y; if (X < 0 || Y < 0 || X >= N || Y >= N) continue; const d = Math.max(Math.abs(x - 3), Math.abs(y - 3)); m[Y * N + X] = d === 4 ? 0 : d === 3 ? 1 : d === 2 ? 0 : 1; } };
  finder(0, 0); finder(N - 7, 0); finder(0, N - 7);
  for (let i = 8; i < N - 8; i++) { m[6 * N + i] = i % 2 ? 0 : 1; m[i * N + 6] = i % 2 ? 0 : 1; }   // timing patterns
  return { N, m };
}
// Render the symbol at size W×W with a quiet zone, a turn and blur, as a camera might see it.
export function qrImage(sym, W, ang, seed) {
  const r = rng(seed), im = image(W, W), n = W * W, q = sym.N + 8, ca = Math.cos(ang), sa = Math.sin(ang);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
    let v = 0;
    for (let sy = 0; sy < 2; sy++) for (let sx = 0; sx < 2; sx++) {
      const u = (x + 0.25 + sx * 0.5) / W - 0.5, w = (y + 0.25 + sy * 0.5) / W - 0.5;
      const mx = (ca * u + sa * w) * q * 1.05 + sym.N / 2, my = (-sa * u + ca * w) * q * 1.05 + sym.N / 2;
      const X = Math.floor(mx), Y = Math.floor(my);
      v += X >= 0 && Y >= 0 && X < sym.N && Y < sym.N && sym.m[Y * sym.N + X] ? 0.1 : 0.92;
    }
    v /= 4; const i = y * W + x;
    im.d[i] = clamp(v + r.gauss() * 0.03, 0, 1); im.d[n + i] = clamp(v * 0.98 + r.gauss() * 0.03, 0, 1); im.d[2 * n + i] = clamp(v * 0.95 + r.gauss() * 0.03, 0, 1);
  }
  return im;
}
// Scan rows for 1:1:3:1:1 runs, confirm down the column, and merge nearby hits.
export function findFinders(im) {
  const W = im.w, H = im.h, g = grey(im); let mean = 0; for (const v of g) mean += v; mean /= g.length;
  const dark = (x, y) => g[y * W + x] < mean;
  const ratioOK = (runs) => { const tot = runs.reduce((a, b) => a + b, 0); if (tot < 7) return false; const u = tot / 7, tol = u * 0.7; return Math.abs(runs[0] - u) < tol && Math.abs(runs[1] - u) < tol && Math.abs(runs[2] - 3 * u) < 3 * tol && Math.abs(runs[3] - u) < tol && Math.abs(runs[4] - u) < tol; };
  const runsAt = (get, len) => { const out = []; let cur = get(0), n = 0, start = 0; for (let i = 0; i <= len; i++) { const d = i < len ? get(i) : !cur; if (d === cur) n++; else { out.push({ dark: cur, n, start }); cur = d; n = 1; start = i; } } return out; };
  const hits = [], rowsHit = [];
  for (let y = 0; y < H; y++) {
    const R = runsAt((x) => dark(x, y), W);
    for (let k = 0; k + 4 < R.length; k++) {
      if (!R[k].dark) continue;
      const five = R.slice(k, k + 5); if (!ratioOK(five.map((q) => q.n))) continue;
      const cx = Math.round(five[2].start + five[2].n / 2);
      const C = runsAt((yy) => dark(cx, yy), H), j = C.findIndex((q) => q.start <= y && q.start + q.n > y);
      rowsHit.push({ y, x0: five[0].start, x1: five[4].start + five[4].n });
      if (j >= 2 && j + 2 < C.length && ratioOK(C.slice(j - 2, j + 3).map((q) => q.n))) hits.push([cx, C[j].start + C[j].n / 2, five.reduce((a, q) => a + q.n, 0) / 7]);
    }
  }
  const centres = [];
  for (const [x, y, u] of hits) { const c = centres.find((q) => Math.hypot(q.x - x, q.y - y) < u * 3); if (c) { c.x = (c.x * c.n + x) / (c.n + 1); c.y = (c.y * c.n + y) / (c.n + 1); c.n++; } else centres.push({ x, y, n: 1, u }); }
  centres.sort((a, b) => b.n - a.n);
  return { centres: centres.slice(0, 3), rowsHit };
}
