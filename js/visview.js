// VisionClear's drawing helpers: pictures as walls of 3D pixel blocks, canvas boards for charts,
// probability bars and feature maps, and small layout helpers shared by the chapters.
import { THREE, M, clamp } from './kit.js';
import { to255, CLASSES, CLASS_COL } from './vision.js';

export const COL = {
  white: '#e8eef8', soft: 'rgba(255,255,255,.6)', dim: 'rgba(255,255,255,.3)', hot: '#ffd166', good: '#7be08c', bad: '#ff5a7a',
  pos: '#ff9a4a', neg: '#4aa8ff', teal: '#b8f2e6', grid: 'rgba(255,255,255,.08)', r: '#ff5a5a', g: '#5ce17a', b: '#5a9bff',
};
export const F = (px, w = '') => `${w} ${px}px system-ui, -apple-system, "Segoe UI", sans-serif`.trim();

// ---------------------------------------------------------------- boards
export function panel(g, w, h, a = 0.92) { g.clearRect(0, 0, w, h); g.fillStyle = `rgba(9,11,16,${a})`; g.fillRect(0, 0, w, h); }
export function txt(g, s, x, y, font = F(20), col = COL.white, align = 'left') { g.font = font; g.fillStyle = col; g.textAlign = align; g.fillText(s, x, y); g.textAlign = 'left'; }
export function boardMesh(tex, w, h) { return new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, transparent: true, toneMapped: false, side: THREE.DoubleSide })); }
export const inReel = () => document.body.classList.contains('gb-reel');
// On a narrow (phone) stage, hide minor labels. Outside the video, nudge the picture down a little
// (negative y), so the readout in the top-left corner covers less of it.
export function fitNarrow(stage, minor = [], y0 = -0.1, yWide = -0.07) {
  const narrow = stage.host.clientWidth < 560;
  minor.forEach((l) => { if (l) l.visible = !narrow; });
  const y = inReel() ? 0 : narrow ? y0 : yWide;
  if (!stage.shift || stage.shift[1] !== y) stage.setShift(0, y);
  return narrow;
}

// Draw a picture (vision.js image) into a rectangle on a canvas, with crisp square pixels.
const off = typeof document !== 'undefined' ? document.createElement('canvas') : null;
export function drawImage(g, im, x, y, w, h, { chan = -1, alpha = 1 } = {}) {
  off.width = im.w; off.height = im.h;
  const o = off.getContext('2d'), id = o.createImageData(im.w, im.h), n = im.w * im.h;
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) id.data[i * 4 + c] = chan < 0 || chan === c ? to255(im.d[c * n + i]) : 0;
  for (let i = 0; i < n; i++) id.data[i * 4 + 3] = 255;
  o.putImageData(id, 0, 0);
  g.save(); g.globalAlpha = alpha; g.imageSmoothingEnabled = false; g.drawImage(off, x, y, w, h); g.restore();
}
// A single-number map (feature map, edges, saliency) as a picture. signed: orange for +, blue for −.
export function drawMap(g, m, mw, mh, x, y, w, h, { signed = false, scale = null, cmap = null } = {}) {
  off.width = mw; off.height = mh;
  const o = off.getContext('2d'), id = o.createImageData(mw, mh);
  let top = scale; if (!top) { top = 1e-6; for (const v of m) top = Math.max(top, Math.abs(v)); }
  for (let i = 0; i < mw * mh; i++) {
    const v = clamp(m[i] / top, -1, 1);
    let c;
    if (cmap) c = cmap(v);
    else if (signed) c = v >= 0 ? [18 + 237 * v, 20 + 134 * v, 26 + 48 * v] : [18 + 56 * -v, 20 + 148 * -v, 26 + 229 * -v];
    else c = heat(Math.max(0, v));
    id.data[i * 4] = c[0]; id.data[i * 4 + 1] = c[1]; id.data[i * 4 + 2] = c[2]; id.data[i * 4 + 3] = 255;
  }
  o.putImageData(id, 0, 0);
  g.save(); g.imageSmoothingEnabled = false; g.drawImage(off, x, y, w, h); g.restore();
}
// Dark → purple → orange → yellow, easy to read on a dark board (in the spirit of "inferno").
export function heat(v) {
  const s = [[10, 10, 18], [80, 20, 110], [200, 60, 70], [250, 150, 40], [255, 240, 150]];
  const f = clamp(v, 0, 1) * (s.length - 1), i = Math.min(s.length - 2, Math.floor(f)), k = f - i;
  return [0, 1, 2].map((c) => s[i][c] + (s[i + 1][c] - s[i][c]) * k);
}
// Horizontal probability bars for the five classes.
export function probBars(g, probs, x, y, w, rowH = 40, { labels = CLASSES, font = 22, truth = -1 } = {}) {
  let best = 0; for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  for (let k = 0; k < probs.length; k++) {
    const yy = y + k * rowH, p = probs[k];
    txt(g, labels[k], x, yy + rowH * 0.62, F(font, k === best ? 'bold' : ''), k === best ? COL.white : COL.soft);
    const bx = x + font * 5.2, bw = w - font * 5.2 - font * 3.2;
    g.fillStyle = 'rgba(255,255,255,.08)'; g.fillRect(bx, yy + rowH * 0.18, bw, rowH * 0.6);
    g.fillStyle = CLASS_COL[k] || COL.hot; g.globalAlpha = k === best ? 1 : 0.6; g.fillRect(bx, yy + rowH * 0.18, bw * clamp(p, 0, 1), rowH * 0.6); g.globalAlpha = 1;
    txt(g, `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`, x + w, yy + rowH * 0.62, F(font * 0.85), COL.soft, 'right');
    if (k === truth) txt(g, '✓', bx - 20, yy + rowH * 0.62, F(font * 0.9, 'bold'), COL.good, 'right');
  }
}
export function roundRect(g, x, y, w, h, r) { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }
// Line chart. series: [{ pts: [numbers], col, label }], values 0..ymax.
export function lineChart(g, R, series, { ymax = 1, xmax = null, yfmt = (v) => `${Math.round(v * 100)}%`, xlabel = '' } = {}) {
  g.strokeStyle = COL.grid; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) { const y = R.y + (k / 4) * R.h; g.beginPath(); g.moveTo(R.x, y); g.lineTo(R.x + R.w, y); g.stroke(); txt(g, yfmt(ymax * (1 - k / 4)), R.x - 8, y + 6, F(16), COL.soft, 'right'); }
  const n = xmax ?? Math.max(2, ...series.map((s) => s.pts.length));
  for (const s of series) {
    if (!s.pts.length) continue;
    g.strokeStyle = s.col; g.lineWidth = s.lw || 3; g.setLineDash(s.dash || []); g.beginPath();
    s.pts.forEach((v, i) => { const x = R.x + (i / Math.max(1, n - 1)) * R.w, y = R.y + R.h - clamp(v / ymax, 0, 1) * R.h; i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.stroke(); g.setLineDash([]);
    const i = s.pts.length - 1, v = s.pts[i]; g.fillStyle = s.col; g.beginPath(); g.arc(R.x + (i / Math.max(1, n - 1)) * R.w, R.y + R.h - clamp(v / ymax, 0, 1) * R.h, 5, 0, 7); g.fill();
  }
  if (xlabel) txt(g, xlabel, R.x + R.w, R.y + R.h + 24, F(16), COL.soft, 'right');
}
export function legend(g, x, y, items, font = F(17)) {
  let cx = x;
  for (const [col, label] of items) { g.fillStyle = col; g.fillRect(cx, y - 12, 18, 12); txt(g, label, cx + 26, y, font, COL.soft); g.font = font; cx += 26 + g.measureText(label).width + 22; }
}

// ---------------------------------------------------------------- pictures as 3D pixel blocks
// A wall of W×H blocks, one per pixel. Each block takes the pixel's colour; its depth can show one
// number (brightness, or a single channel), so you can see a picture as a landscape of numbers.
export class PixelWall {
  constructor(parent, w, h, cell, { depth = 0.25, gap = 0.9, lambert = true } = {}) {
    this.w = w; this.h = h; this.cell = cell; this.depth = depth;
    const n = w * h;
    const mat = lambert ? new THREE.MeshLambertMaterial({ emissive: 0x080808 }) : new THREE.MeshBasicMaterial({ toneMapped: false });
    this.mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(cell * gap, cell * gap, 1), mat, n);
    this.mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
    this.group = new THREE.Group(); this.group.add(this.mesh); parent.add(this.group);
    this._o = new THREE.Object3D(); this._c = new THREE.Color();
  }
  // pixel (x, y) → local position of its centre
  pos(x, y) { return [(x + 0.5 - this.w / 2) * this.cell, (this.h / 2 - y - 0.5) * this.cell]; }
  // im: a vision.js picture of the same size. hFn(i) → 0..1 block height. chan: -1 colour, 0/1/2 one channel.
  set(im, { chan = -1, hFn = null, dim = null } = {}) {
    const o = this._o, c = this._c, n = this.w * this.h;
    for (let y = 0; y < this.h; y++) for (let x = 0; x < this.w; x++) {
      const i = y * this.w + x, r = im.d[i], gg = im.d[n + i], b = im.d[2 * n + i];
      const hv = hFn ? hFn(i) : 0.299 * r + 0.587 * gg + 0.114 * b, d = 0.01 + this.depth * clamp(hv, 0, 1);
      const [X, Y] = this.pos(x, y);
      o.position.set(X, Y, d / 2); o.scale.set(1, 1, d); o.updateMatrix(); this.mesh.setMatrixAt(i, o.matrix);
      if (chan < 0) c.setRGB(r, gg, b, THREE.SRGBColorSpace);
      else { const v = [r, gg, b][chan]; c.setRGB(chan === 0 ? v : 0.03, chan === 1 ? v : 0.03, chan === 2 ? v : 0.03, THREE.SRGBColorSpace); }
      if (dim && dim(i)) c.multiplyScalar(0.25);
      this.mesh.setColorAt(i, c);
    }
    this.mesh.instanceMatrix.needsUpdate = true; this.mesh.instanceColor.needsUpdate = true;
  }
  // a one-number map (e.g. a feature map): signed orange/blue or heat colours; block height = size.
  setMap(m, { scale = null, signed = false } = {}) {
    const o = this._o, c = this._c; let top = scale; if (!top) { top = 1e-6; for (const v of m) top = Math.max(top, Math.abs(v)); }
    for (let y = 0; y < this.h; y++) for (let x = 0; x < this.w; x++) {
      const i = y * this.w + x, v = clamp(m[i] / top, -1, 1), d = 0.01 + this.depth * Math.abs(v), [X, Y] = this.pos(x, y);
      o.position.set(X, Y, d / 2); o.scale.set(1, 1, d); o.updateMatrix(); this.mesh.setMatrixAt(i, o.matrix);
      const col = signed ? (v >= 0 ? [18 + 237 * v, 20 + 134 * v, 26 + 48 * v] : [18 + 56 * -v, 20 + 148 * -v, 26 + 229 * -v]) : heat(Math.max(0, v));
      c.setRGB(col[0] / 255, col[1] / 255, col[2] / 255, THREE.SRGBColorSpace); this.mesh.setColorAt(i, c);
    }
    this.mesh.instanceMatrix.needsUpdate = true; this.mesh.instanceColor.needsUpdate = true;
  }
}
// A dark backing plate behind a pixel wall.
export function backPlate(w, h, color = 0x151922) { const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.04), M.matte(color)); m.receiveShadow = true; return m; }
// A frame of thin lines (for windows, boxes).
export function frameLines(w, h, color = 0xffd166, opacity = 1) {
  const g = new THREE.BufferGeometry().setFromPoints([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]].map(([x, y]) => new THREE.Vector3(x * w / 2, y * h / 2, 0)));
  return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, toneMapped: false }));
}
export function hexCol(s) { return new THREE.Color(s); }
