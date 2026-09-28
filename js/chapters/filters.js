// Chapter 2: filters and features. Three views:
//  - Hand-made filters sliding over a live picture (48 × 32, drawn by vision.js drawScene): Sobel
//    edge kernels (Sobel & Feldman 1968), edge strength √(gx² + gy²), the Harris corner response
//    R = det(M) − 0.05·trace(M)² (Harris & Stephens 1988) and Laws' texture energy (Laws 1980).
//  - 2×2 max pooling, twice: 48 × 32 → 24 × 16 → 12 × 8.
//  - The trained CNN from lab.js, opened up: its 8 first-layer filters (the learned 3×3×3 weights),
//    the 8 feature maps they make, pooling, the 16 second-layer maps, and the answer. The idea that
//    early layers find edges and later layers combine them into parts comes from Hubel & Wiesel's
//    cat experiments (1959, 1962), Fukushima's Neocognitron (1980) and, for big trained networks,
//    Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks" (2014). A 2-layer toy
//    shows only the start of that ladder.
import { THREE, M, canvasTexture, clamp } from '../kit.js';
import { drawScene, makeObj, rng, grey, edges, harris, lawsTexture, maxPool, SOBEL_X, SOBEL_Y, makeSample, CLASSES, argmax, SIZE } from '../vision.js';
import { lab, labStep, labFinish, mainReady, progress, trainingNote } from '../lab.js';
import { panel, txt, F, COL, boardMesh, fitNarrow, inReel, PixelWall, backPlate, frameLines, drawMap, drawImage, probBars } from '../visview.js';

const PW = 48, PH = 32;
const FILTERS = {
  edgeX: { name: 'Up-down edges', k: 3, note: 'Sobel filter: big where brightness jumps from left to right.' },
  edgeY: { name: 'Side edges', k: 3, note: 'Sobel filter turned on its side: finds tops and bottoms.' },
  edges: { name: 'All edges', k: 3, note: 'Both Sobel answers combined: √(gx² + gy²). Outlines light up.' },
  corners: { name: 'Corners', k: 3, note: 'Harris corner score: big only where edges run two ways at once, like a tip or a stalk.' },
  texture: { name: 'Texture', k: 5, note: 'Laws\' texture energy: a 5×5 "ripple" filter that answers to busy, fine patterns.' },
};
const VIEWS = {
  hand: { pos: [0, 3.0, 8.9], target: [0, 2.2, 0] },
  pool: { pos: [0.2, 2.9, 8.4], target: [0.2, 2.15, 0] },
  cnn: { pos: [0.45, 2.8, 10.8], target: [0.45, 2.15, 0] },
};
function makePicture(seed) {
  const r = rng(seed), kinds = [1, 4, 2, 3].sort(() => r() - 0.5).slice(0, 3);
  const objs = kinds.map((c, i) => makeObj(c, r, { x: 9 + i * 15 + r.range(-1.5, 1.5), y: r.range(13, 19), s: r.range(8, 10), ang: r.range(-0.7, 0.7) }));
  return drawScene(PW, PH, objs, { bg: [0.32, 0.36, 0.42], light: 1, noise: 0.012, seed }).im;
}
const NET_PICKS = { 1: 'Mango', 2: 'Banana', 3: 'Chilli', 4: 'Coin' };

export default {
  id: 'filters',
  short: 'Filters and features',
  title: 'Filters find edges, then shapes',
  subtitle: 'A small grid of weights slides over the picture. Stack layers of them and the network sees parts, then objects.',
  view: VIEWS.hand,
  learn: `<p>If a picture is just numbers, how do you find a mango in them? Start small. Look at a <b>3 × 3</b> patch, multiply each pixel by a weight, and add up. This little grid of weights is a <b>filter</b> (or kernel), and sliding it across the whole picture is called <b>convolution</b>. NeuralNetClear shows it on a digit; here it runs on a colour scene.</p>
    <p>The right weights make the sum big only at an <b>edge</b>, where brightness jumps. Combine an up-down and a side-to-side edge filter and you get outlines. Where edges run <b>two ways at once</b>, you have a <b>corner</b>, like the tip of a chilli. Other filters answer to <b>texture</b>: busy, fine patterns. Each answer, spot by spot, makes a new picture called a <b>feature map</b>.</p>
    <p><b>Pooling</b> then shrinks each map: keep only the biggest number in every 2 × 2 block. Four numbers become one, so the next layer looks at a wider area, and a small shift of the object hardly changes the answer.</p>
    <p>A <b>convolutional neural network</b> (CNN) stacks these steps, but nobody sets its weights by hand: it <b>learns</b> them from examples. Open "Inside the trained CNN" to see the network this box trains in your browser. Its first layer learns edge and colour filters on its own. Its second layer combines them into bigger parts. Big networks such as ResNet repeat this 50 or more times, climbing from edges to textures to parts to whole objects. The idea was loosely inspired by cells in the cat's visual cortex (see BrainClear), but a CNN is maths, not a brain.</p>
    <p class="tip"><b>Try it:</b> switch between the filters and watch which parts of the picture light up. Then open "Inside the trained CNN", pick a banana, and compare its maps with a coin's.</p>`,
  terms: [
    { t: 'Filter (kernel)', d: 'A small grid of weights, like 3 × 3, that is multiplied with each patch of the picture.' },
    { t: 'Convolution', d: 'Sliding a filter over the whole picture, adding up at every spot.' },
    { t: 'Feature map', d: 'The picture of a filter\'s answers: bright where it found its pattern.' },
    { t: 'Edge', d: 'A place where brightness or colour changes sharply.' },
    { t: 'Max pooling', d: 'Keeping the biggest number in each 2 × 2 block, which halves the map\'s width and height.' },
    { t: 'CNN', d: 'Convolutional neural network: layers of learned filters and pooling, then a final guess.' },
  ],
  defaults: { show: 'hand', filter: 'edges', scan: 4, seed: 3, pick: 2, sample: 0 },
  controls: [
    { key: 'show', type: 'seg', label: 'Show', options: [{ v: 'hand', label: 'Hand-made filters' }, { v: 'pool', label: 'Pooling' }, { v: 'cnn', label: 'Inside the trained CNN' }] },
    { key: 'filter', type: 'seg', label: 'Filter', options: Object.entries(FILTERS).map(([v, f]) => ({ v, label: f.name })), hint: 'Used by "Hand-made filters" and "Pooling".' },
    { key: 'scan', type: 'range', label: 'Scan speed', min: 0.2, max: 6, step: 0.1, fmt: (v) => `${Math.round(v * 48)} spots a second` },
    { key: 'pick', type: 'seg', label: 'CNN picture', options: Object.entries(NET_PICKS).map(([v, l]) => ({ v: +v, label: l })) },
    { key: 'go', type: 'buttons', label: 'Pictures', items: [
      { label: 'New scene', act: (s) => { s.seed = (s.seed || 1) + 1; } },
      { label: 'Another CNN picture', act: (s) => { s.sample = (s.sample || 0) + 1; } },
    ] },
  ],
  quiz: [
    { q: 'What is a feature map?', options: ['A map of where the camera was', 'The picture of a filter\'s answers at every spot', 'A list of object names', 'The network\'s weights'], answer: 1, why: 'Slide a filter over the picture and write down its sum at every spot: that new picture is a feature map.' },
    { q: 'What does 2 × 2 max pooling do to a 48 × 32 map?', options: ['Makes it 96 × 64', 'Makes it 24 × 16, keeping the biggest of each 2 × 2 block', 'Blurs it but keeps the size', 'Deletes the edges'], answer: 1, why: 'Every 2 × 2 block becomes one number, its maximum, so width and height both halve.' },
    { q: 'Who chooses the filters in a CNN?', options: ['A programmer writes each one', 'They are learned from examples during training', 'They are random and never change', 'The camera'], answer: 1, why: 'Training adjusts the filter weights so the network\'s answers get better. Edge filters appear on their own.' },
  ],
  reel: [
    { ms: 5600, caption: 'A CNN learns its own filters: edges first, then bigger parts, then an answer.', set: { show: 'cnn', pick: 2, sample: 0 }, act: () => labFinish('seg'), spin: 0, view: { pos: [0.45, 2.7, 10.4], target: [0.45, 2.1, 0] } },
  ],

  build({ stage }) {
    const root = new THREE.Group(); stage.root.add(root);
    const handG = new THREE.Group(), poolG = new THREE.Group(), cnnG = new THREE.Group(); root.add(handG, poolG, cnnG);

    // ================= hand-made filters
    const C1 = 0.07, IX = -2.0, OX = 2.0, WY = 2.25;
    const inG = new THREE.Group(); inG.position.set(IX, WY, 0); handG.add(inG);
    const outG = new THREE.Group(); outG.position.set(OX, WY, 0); handG.add(outG);
    const inWall = new PixelWall(inG, PW, PH, C1, { depth: 0.18 }), outWall = new PixelWall(outG, PW, PH, C1, { depth: 0.35 });
    const p1 = backPlate(PW * C1 + 0.1, PH * C1 + 0.1); p1.position.z = -0.03; inG.add(p1);
    const p2 = backPlate(PW * C1 + 0.1, PH * C1 + 0.1); p2.position.z = -0.03; outG.add(p2);
    const win = frameLines(C1 * 3, C1 * 3, 0xffd166); win.position.z = 0.25; inG.add(win);
    const cellF = frameLines(C1, C1, 0xffd166); cellF.position.z = 0.4; outG.add(cellF);
    const beamGeo = new THREE.BufferGeometry(); beamGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3));
    const beams = new THREE.LineSegments(beamGeo, new THREE.LineBasicMaterial({ color: 0xffd166, transparent: true, opacity: 0.6, toneMapped: false })); handG.add(beams);
    const kern = canvasTexture(420, 420, (g, w, h, s) => {
      panel(g, w, h); if (!s) return;
      const f = s.filter; txt(g, FILTERS[f].name, w / 2, 38, F(26, 'bold'), COL.white, 'center');
      const grid = (k, K, x0, y0, S) => k.forEach((v, i) => {
        const x = x0 + (i % K) * S, y = y0 + Math.floor(i / K) * S, a = clamp(Math.abs(v) / (K === 5 ? 36 : 2), 0, 1);
        g.fillStyle = v > 0 ? `rgba(255,154,74,${0.15 + 0.7 * a})` : v < 0 ? `rgba(74,168,255,${0.15 + 0.7 * a})` : 'rgba(255,255,255,.06)'; g.fillRect(x + 2, y + 2, S - 4, S - 4);
        txt(g, String(v), x + S / 2, y + S / 2 + 8, F(K === 5 ? 18 : 26, 'bold'), COL.white, 'center');
      });
      if (f === 'edgeX') grid(SOBEL_X, 3, 90, 70, 80);
      else if (f === 'edgeY') grid(SOBEL_Y, 3, 90, 70, 80);
      else if (f === 'edges') { grid(SOBEL_X, 3, 20, 90, 58); grid(SOBEL_Y, 3, 226, 90, 58); txt(g, 'gx', 107, 80, F(20), COL.soft, 'center'); txt(g, 'gy', 313, 80, F(20), COL.soft, 'center'); txt(g, 'strength = √(gx² + gy²)', w / 2, 300, F(22), COL.hot, 'center'); }
      else if (f === 'corners') { grid(SOBEL_X, 3, 20, 90, 58); grid(SOBEL_Y, 3, 226, 90, 58); txt(g, 'M sums gx², gx·gy, gy² nearby', w / 2, 300, F(20), COL.soft, 'center'); txt(g, 'R = det M − 0.05 (trace M)²', w / 2, 332, F(22, 'bold'), COL.hot, 'center'); }
      else { const R5 = [1, -4, 6, -4, 1]; grid(R5.flatMap((a) => R5.map((b) => a * b)), 5, 60, 64, 60); }
      txt(g, 'multiply the patch by these weights, add', w / 2, h - 44, F(18), COL.soft, 'center');
      txt(g, 'orange = plus, blue = minus', w / 2, h - 18, F(18), COL.soft, 'center');
    });
    const kMesh = boardMesh(kern.tex, 1.35, 1.35); kMesh.position.set(0, 0.62, 0.05); handG.add(kMesh);
    const lIn = stage.label('The picture: 48 × 32 pixels', [IX, WY - 1.38, 0], handG), lOut = stage.label('Feature map', [OX, WY - 1.38, 0], handG, 'hot');

    // ================= pooling
    const C2 = 0.066, PY = 2.1, PXs = [-2.2, 0.55, 2.55];
    const pools = [[PW, PH], [PW / 2, PH / 2], [PW / 4, PH / 4]].map(([w, h], k) => {
      const g = new THREE.Group(); g.position.set(PXs[k], PY, 0); poolG.add(g);
      const wall = new PixelWall(g, w, h, C2 * (1 << k), { depth: 0.35 });
      const pl = backPlate(w * C2 * (1 << k) + 0.08, h * C2 * (1 << k) + 0.08); pl.position.z = -0.03; g.add(pl);
      return { g, wall, w, h };
    });
    const pwin = frameLines(C2 * 2, C2 * 2, 0xffd166); pwin.position.z = 0.4; pools[0].g.add(pwin);
    const pcell = frameLines(C2 * 2, C2 * 2, 0xffd166); pcell.position.z = 0.4; pools[1].g.add(pcell);
    const lP = [stage.label('48 × 32 = 1,536', [PXs[0], PY - 1.3, 0], poolG), stage.label('24 × 16 = 384', [PXs[1], PY - 1.3, 0], poolG), stage.label('12 × 8 = 96', [PXs[2], PY - 1.3, 0], poolG, 'hot')];
    const poolNote = canvasTexture(640, 150, (g, w, h) => { panel(g, w, h, 0.85); txt(g, 'Max pooling: each 2 × 2 block → its biggest number', w / 2, 58, F(26, 'bold'), COL.hot, 'center'); txt(g, 'Half the width, half the height, a quarter of the numbers', w / 2, 104, F(22), COL.soft, 'center'); });
    const pnMesh = boardMesh(poolNote.tex, 3.2, 0.75); pnMesh.position.set(0.2, 0.35, 0.05); poolG.add(pnMesh);

    // ================= inside the trained CNN
    const MAPS = [
      { name: 'Picture', x: -3.55, n: 1, cols: 1, S: 1.25 },
      { name: 'Layer 1: 8 filters', x: -1.95, n: 8, cols: 2, S: 0.56, act: 2 },
      { name: 'Pool', x: -0.6, n: 8, cols: 2, S: 0.56, act: 3 },
      { name: 'Layer 2: 16 filters', x: 0.95, n: 16, cols: 4, S: 0.4, act: 5 },
      { name: 'Pool', x: 2.55, n: 16, cols: 4, S: 0.4, act: 6 },
    ];
    const tiles = MAPS.map((L) => {
      const rows = Math.ceil(L.n / L.cols), arr = [];
      for (let i = 0; i < L.n; i++) {
        const c = canvasTexture(40, 40, () => {}); c.tex.magFilter = THREE.NearestFilter; c.tex.minFilter = THREE.NearestFilter;
        const m = boardMesh(c.tex, L.S * 0.92, L.S * 0.92);
        m.position.set(L.x + ((i % L.cols) - (L.cols - 1) / 2) * L.S, 1.85 + ((rows - 1) / 2 - Math.floor(i / L.cols)) * L.S, (i % L.cols) * -0.05);
        cnnG.add(m); arr.push(c);
      }
      return arr;
    });
    const lMaps = MAPS.map((L, k) => stage.label(L.name, [L.x, k === 0 ? 1.0 : 0.55, 0], cnnG, k === 3 ? 'hot' : ''));
    let cnnProbs = new Float32Array(5), cnnKey = '';
    const answer = canvasTexture(420, 470, (g, w, h) => {
      panel(g, w, h);
      if (!mainReady()) { txt(g, 'Training in your browser…', 20, 44, F(24, 'bold')); g.fillStyle = 'rgba(255,255,255,.12)'; g.fillRect(20, 70, w - 40, 20); g.fillStyle = COL.hot; g.fillRect(20, 70, (w - 40) * progress(), 20); txt(g, trainingNote(), 20, 124, F(18), COL.soft); return; }
      txt(g, 'Answer', 20, 40, F(26, 'bold')); txt(g, 'dense layer → softmax', 20, 70, F(18), COL.soft);
      probBars(g, cnnProbs, 20, 90, w - 40, 52, { font: 22 });
      txt(g, 'Learned first-layer filters:', 20, 372, F(18), COL.soft);
      const K = lab.net.firstKernels();
      for (let o = 0; o < K.n; o++) {
        let mx = 1e-6; for (let q = 0; q < 27; q++) mx = Math.max(mx, Math.abs(K.W[o * 27 + q]));
        for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) {
          const c = [0, 1, 2].map((ch) => Math.round(clamp(0.5 + 0.5 * K.W[o * 27 + ch * 9 + y * 3 + x] / mx, 0, 1) * 255));
          g.fillStyle = `rgb(${c})`; g.fillRect(20 + o * 48 + x * 13, 388 + y * 13, 13, 13);
        }
      }
      txt(g, 'grey = 0, colours = which colour each filter likes', 20, h - 14, F(15), COL.soft);
    });
    const ansMesh = boardMesh(answer.tex, 1.6, 1.79); ansMesh.position.set(4.25, 1.9, 0); cnnG.add(ansMesh);
    const arrowsGeo = new THREE.BufferGeometry().setFromPoints([-2.85, -1.3, 0.05, 1.7, 3.3].flatMap((x) => [new THREE.Vector3(x - 0.12, 1.85, 0), new THREE.Vector3(x + 0.12, 1.85, 0)]));
    cnnG.add(new THREE.LineSegments(arrowsGeo, new THREE.LineBasicMaterial({ color: 0xffd166, toneMapped: false })));
    const showCNN = (s) => {
      const r = rng(500 + s.pick * 97 + s.sample * 13), im = makeSample(s.pick, r, 'train');
      drawImage(tiles[0][0].canvas.getContext('2d'), im, 0, 0, 40, 40); tiles[0][0].tex.needsUpdate = true;
      if (!mainReady()) return;
      const net = lab.net; cnnProbs = net.forward(im);
      MAPS.forEach((L, k) => {
        if (!k) return;
        const A = net.acts[L.act], n = A.h * A.w; let top = 1e-6; for (const v of A.d) top = Math.max(top, v);
        for (let i = 0; i < L.n; i++) { const c = tiles[k][i], g = c.canvas.getContext('2d'); drawMap(g, A.d.subarray(i * n, (i + 1) * n), A.w, A.h, 0, 0, 40, 40, { scale: top * 0.8 }); c.tex.needsUpdate = true; }
      });
      answer.redraw();
    };

    // ================= state
    let pic = null, picSeed = -1, maps = {}, scanPos = 0, fKey = '', view = '', t = 0;
    const computeMaps = () => {
      const g = grey(pic), e = edges(g, PW, PH);
      maps = { edgeX: e.gx, edgeY: e.gy, edges: e.m, corners: harris(g, PW, PH).map((v) => Math.max(0, v)), texture: lawsTexture(g, PW, PH) };
    };
    const signed = (f) => f === 'edgeX' || f === 'edgeY';
    const topOf = (m) => { let a = 1e-6; for (const v of m) a = Math.max(a, Math.abs(v)); return a; };

    return {
      update(dt, s) {
        dt = Math.max(0, dt); t += dt;
        if (!mainReady()) labStep(inReel() ? 1000 : 9); else labStep(4);
        const narrow = fitNarrow(stage, [lIn, ...lP.slice(0, 2), ...lMaps.slice(1)], -0.12);
        handG.visible = s.show === 'hand'; poolG.visible = s.show === 'pool'; cnnG.visible = s.show === 'cnn';
        if (view !== s.show) { if (view) stage.setView(VIEWS[s.show].pos, VIEWS[s.show].target, 0.9); view = s.show; }
        if (s.seed !== picSeed) { picSeed = s.seed; pic = makePicture(s.seed); computeMaps(); inWall.set(pic); fKey = ''; }
        const f = s.filter, m = maps[f], top = topOf(m) * (f === 'corners' ? 0.5 : 1);
        if (s.show === 'hand') {
          const fk = `hand|${f}|${picSeed}`;
          if (fk !== fKey) { fKey = fk; scanPos = 0; kern.redraw(s); lOut.element.textContent = `Feature map: ${FILTERS[f].name.toLowerCase()}`; }
          scanPos += dt * s.scan * 48; if (scanPos > PW * PH + 60) scanPos = 0;
          const cur = Math.floor(scanPos), shown = new Float32Array(PW * PH);
          for (let i = 0; i < PW * PH; i++) shown[i] = i <= cur ? m[i] : 0;
          outWall.setMap(shown, { scale: top, signed: signed(f) });
          const i = Math.min(PW * PH - 1, cur), x = i % PW, y = Math.floor(i / PW), K = FILTERS[f].k;
          win.scale.set(K / 3, K / 3, 1);
          const [wx, wy] = inWall.pos(x, y), [ox, oy] = outWall.pos(x, y);
          win.position.set(wx, wy, 0.25); cellF.position.set(ox, oy, 0.4);
          const bp = beamGeo.attributes.position, h = (C1 * K) / 2;
          bp.setXYZ(0, IX + wx + h, WY + wy + h, 0.25); bp.setXYZ(1, OX + ox, WY + oy, 0.4); bp.setXYZ(2, IX + wx + h, WY + wy - h, 0.25); bp.setXYZ(3, OX + ox, WY + oy, 0.4); bp.needsUpdate = true;
          this._cur = [x, y, m[i]];
        } else if (s.show === 'pool') {
          const pk = `pool|${f}|${picSeed}`;
          const base = signed(f) ? m.map(Math.abs) : m;
          if (pk !== fKey) { fKey = pk; const a = maxPool(base, PW, PH), b = maxPool(a.out, a.w, a.h); this._pool = [base, a.out, b.out]; this._pt = topOf(base); }
          pools.forEach((P, k) => P.wall.setMap(this._pool[k], { scale: this._pt }));
          scanPos += dt * s.scan * 12; if (scanPos > (PW / 2) * (PH / 2) + 20) scanPos = 0;
          const i = Math.min((PW / 2) * (PH / 2) - 1, Math.floor(scanPos)), x = i % (PW / 2), y = Math.floor(i / (PW / 2));
          const [a0, b0] = pools[0].wall.pos(2 * x, 2 * y); pwin.position.set(a0 + C2 / 2, b0 - C2 / 2, 0.4);
          const [a1, b1] = pools[1].wall.pos(x, y); pcell.position.set(a1, b1, 0.4); pcell.scale.set(1, 1, 1);
          this._pc = [x, y, this._pool[1][i]];
        } else {
          const ck = `${s.pick}|${s.sample}|${mainReady()}|${lab.run}`;
          if (ck !== cnnKey) { cnnKey = ck; showCNN(s); }
          else if (!mainReady() && Math.floor(t * 4) !== this._tq) { this._tq = Math.floor(t * 4); answer.redraw(); }
        }
        lMaps[0].visible = !narrow;
      },
      readout(s) {
        if (s.show === 'hand') {
          const [x, y, v] = this._cur || [0, 0, 0], K = FILTERS[s.filter].k;
          return `<div class="big">${FILTERS[s.filter].name}</div>
            <div class="row"><span>Window at column ${x + 1}, row ${y + 1}</span><b>answer ${v.toFixed(2)}</b></div>
            <div class="row"><span>Multiply-adds for the whole picture</span><b>${PW} × ${PH} × ${K * K} = ${(PW * PH * K * K).toLocaleString('en')}</b></div>
            <small>${FILTERS[s.filter].note}</small>`;
        }
        if (s.show === 'pool') {
          const [x, y, v] = this._pc || [0, 0, 0];
          return `<div class="big">2 × 2 max pooling</div>
            <div class="row"><span>Block ${x + 1}, ${y + 1} keeps its biggest</span><b>${v.toFixed(2)}</b></div>
            <div class="row"><span>Numbers after two poolings</span><b>1,536 → 384 → 96</b></div>
            <small>Pooling ${FILTERS[s.filter].name.toLowerCase()}. A nudge of one pixel barely changes the pooled map.</small>`;
        }
        if (!mainReady()) return `<div class="big">Training… ${Math.round(progress() * 100)}%</div><div class="row"><span>Network</span><b>${(lab.net?.params || 21081).toLocaleString('en')} weights</b></div><small>${trainingNote()}</small>`;
        const best = argmax(cnnProbs);
        return `<div class="big">"${CLASSES[best]}", ${Math.round(cnnProbs[best] * 100)}% sure</div>
          <div class="row"><span>Layer 1</span><b>8 maps of 20 × 20</b></div>
          <div class="row"><span>Layer 2</span><b>16 maps of 10 × 10</b></div>
          <div class="row"><span>Weights learned in your browser</span><b>${lab.net.params.toLocaleString('en')}</b></div>
          <small>Bright = the filter found its pattern there.</small>`;
      },
    };
  },
};
