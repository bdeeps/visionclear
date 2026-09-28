// Chapter 4: detection and segmentation: not just "what" but "where".
// A 96 × 64 scene holds 3 to 5 made-up objects. The detector is the main CNN from lab.js used as a
// sliding window (as in Viola & Jones 2001 and the early R-CNN family): it looks at every 20 × 20
// window, 2 pixels apart (39 × 23 = 897 windows), and for each gives class probabilities and a box
// (its 4 box outputs, trained like R-CNN's box regression, Girshick et al. 2014).
// Windows whose best object score beats the confidence threshold become candidate boxes. Greedy
// non-maximum suppression (NMS) then keeps the most confident box, removes boxes that overlap it by
// more than the IoU limit, and repeats. Each kept box is averaged with the boxes it removed, weighted
// by confidence ("box voting", Gidaris & Komodakis 2015), and needs at least 3 windows to agree
// (like the minimum-neighbours rule of the Viola–Jones detector in OpenCV).
// IoU = area of overlap ÷ area of union. PASCAL VOC counted a detection as correct at IoU ≥ 0.5
// (Everingham et al. 2010). Masks come from the separate mask network in lab.js: every pixel gets an
// object-or-not score, and each detection colours the object pixels inside its box.
import { THREE, M, canvasTexture, clamp } from '../kit.js';
import { makeDetScene, crop, argmax, iou, nmsSteps, CLASSES, CLASS_COL, SCENE_W, SCENE_H, SIZE } from '../vision.js';
import { lab, labStep, labFinish, mainReady, progress, trainingNote } from '../lab.js';
import { panel, txt, F, COL, boardMesh, fitNarrow, inReel, drawImage, heat } from '../visview.js';

const STRIDE = 2, NX = (SCENE_W - SIZE) / STRIDE + 1, NY = (SCENE_H - SIZE) / STRIDE + 1, VOTES = 3;
const BW = 4.8, U = BW / SCENE_W, BH = SCENE_H * U, BX = -0.65, BY = 2.05, ZK = 1.0;
const STEPS = { scan: '1. Slide a window', boxes: '2. Candidate boxes', nms: '3. Non-max suppression', masks: '4. Pixel masks' };
const toW = (x, y) => [BX + (x - SCENE_W / 2) * U, BY + (SCENE_H / 2 - y) * U];

export default {
  id: 'detect',
  short: 'Detection and masks',
  title: 'Where is it? Boxes, then masks',
  subtitle: 'Slide the recogniser across a whole scene, keep the confident boxes, remove the duplicates, then colour every pixel.',
  view: { pos: [2.2, 3.5, 7.4], target: [0.1, 1.85, 0] },
  learn: `<p>A recogniser says <b>what</b> is in a picture. A <b>detector</b> must also say <b>where</b>, for every object, even when there are several. The simplest way: slide the recogniser across the scene like a magnifying glass. This one checks <b>897 windows</b> of 20 × 20 pixels, each 2 pixels from the last, and each window guesses a class, a <b>confidence</b> and a tight <b>box</b>.</p>
    <p>Keep only windows above a <b>confidence threshold</b>. Here each box floats above the picture at a height equal to its confidence, and the glass sheet is the threshold. Lower it and you find more objects, but also more junk.</p>
    <p>One mango sets off many overlapping windows. <b>Non-max suppression</b> (NMS) cleans up: take the most confident box, delete every box that overlaps it too much, and repeat. Overlap is measured by <b>IoU</b>, intersection over union: the shared area divided by the total area covered. 1 means identical, 0 means no overlap. A detection usually counts as right if its IoU with the true box is at least 0.5.</p>
    <p>Boxes are rough. <b>Segmentation</b> labels every single pixel instead: a second small network here scores each pixel "object or not", and each detection paints its own object's pixels. Modern detectors such as <b>YOLO</b> look at the whole picture once instead of sliding, which is how they run at video speed in cars and phones. This little detector trained in your browser in seconds, so it misses some objects and raises some false alarms: the readout counts both honestly.</p>
    <p class="tip"><b>Try it:</b> step through NMS with the slider and watch duplicates drop. Then drag the threshold down to 0.4 and up to 0.95. Which gives misses, and which gives false alarms?</p>`,
  terms: [
    { t: 'Object detection', d: 'Finding every object in a picture, with a class and a box for each.' },
    { t: 'Bounding box', d: 'The smallest upright rectangle around an object.' },
    { t: 'Confidence', d: 'How sure the model is, from 0 to 1. A threshold decides which guesses to keep.' },
    { t: 'IoU', d: 'Intersection over union: overlap area ÷ combined area of two boxes.' },
    { t: 'Non-max suppression', d: 'Keep the most confident box, remove boxes that overlap it a lot, repeat.' },
    { t: 'Segmentation', d: 'Labelling every pixel with what it belongs to, giving an outline (mask) instead of a box.' },
  ],
  defaults: { step: 'nms', thr: 0.8, nmsIoU: 0.3, nmsK: 1, scene: 11, truth: true, play: false },
  onChange(s, key) { if (key === 'scene' || key === 'thr' || key === 'nmsIoU') { s.nmsK = key === 'scene' ? 1 : s.nmsK; } },
  controls: [
    { key: 'step', type: 'seg', label: 'Step', options: Object.entries(STEPS).map(([v, l]) => ({ v, label: l.slice(3) })) },
    { key: 'thr', type: 'range', label: 'Confidence threshold', min: 0.3, max: 0.99, step: 0.01, fmt: (v) => v.toFixed(2) },
    { key: 'nmsK', type: 'range', label: 'Non-max suppression progress', min: 0, max: 1, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%` },
    { key: 'nmsIoU', type: 'range', label: 'NMS overlap limit (IoU)', min: 0.1, max: 0.8, step: 0.01, fmt: (v) => v.toFixed(2), hint: 'Boxes that overlap a kept box more than this are removed.' },
    { key: 'truth', type: 'toggle', label: 'Show the true boxes (green)' },
    { key: 'go', type: 'buttons', label: 'Scene', items: [{ label: 'New scene', act: (s) => { s.scene = (s.scene || 1) + 1; } }, { label: 'Play NMS', act: (s) => { s.step = 'nms'; s.nmsK = 0; s.play = true; } }] },
  ],
  quiz: [
    { q: 'Two boxes share an overlap of 20 square pixels, and together cover 80 square pixels. What is their IoU?', options: ['0.2', '0.25', '0.8', '4'], answer: 1, why: 'IoU = intersection ÷ union = 20 ÷ 80 = 0.25.' },
    { q: 'What does non-max suppression remove?', options: ['The most confident box', 'Boxes that overlap a more confident box too much', 'All boxes below 0.5 IoU with the truth', 'Every box of the same colour'], answer: 1, why: 'It keeps the best box and deletes its near-duplicates, then repeats with the next best.' },
    { q: 'You lower the confidence threshold a lot. What usually happens?', options: ['Fewer boxes and no mistakes', 'More objects found, but more false alarms too', 'The boxes get tighter', 'Nothing changes'], answer: 1, why: 'A lower bar lets in weak true detections and weak junk alike: a trade-off between misses and false alarms.' },
  ],
  reel: [
    { ms: 5600, caption: 'Slide the recogniser over the scene: every confident window becomes a box.', set: { step: 'boxes', thr: 0.8, scene: 11, truth: false }, act: () => labFinish('seg'), anim: { thr: [0.97, 0.6] }, spin: 0, view: { pos: [2.6, 3.4, 7.0], target: [0.0, 1.8, 0] } },
    { ms: 5400, caption: 'Non-max suppression keeps the best box and deletes its duplicates.', set: { step: 'nms', thr: 0.8, nmsIoU: 0.3, truth: true, scene: 11 }, anim: { nmsK: [0, 1] }, spin: 0, view: { pos: [1.6, 3.0, 6.8], target: [-0.2, 1.9, 0] } },
  ],

  build({ stage }) {
    const root = new THREE.Group(); stage.root.add(root);
    let sc = null, sceneSeed = -1, W = [], done = 0, run = -1, segP = null, segRun = false, t = 0, scanT = 0;
    let cands = [], steps = [], fused = [], shownK = -1, state = [], cacheKey = '';

    // ---- the scene board: picture plus overlays (heat map, masks, true boxes, labels)
    const board = canvasTexture(768, 512, (g, w, h, s) => {
      g.clearRect(0, 0, w, h); if (!sc || !s) return;
      drawImage(g, sc.im, 0, 0, w, h, { alpha: s.step === 'masks' ? 0.55 : 1 });
      const k = w / SCENE_W;
      if (s.step === 'scan') {
        g.globalAlpha = 0.6;
        for (let i = 0; i < Math.min(done, Math.floor(scanT)); i++) { const x = (i % NX) * STRIDE + SIZE / 2, y = Math.floor(i / NX) * STRIDE + SIZE / 2, c = heat(W[i].conf); g.fillStyle = `rgb(${c})`; g.fillRect((x - 1) * k, (y - 1) * k, STRIDE * k, STRIDE * k); }
        g.globalAlpha = 1;
      }
      if (s.step === 'masks' && segP) {
        const id = g.getImageData(0, 0, w, h);
        for (const f of fused) {
          const col = new THREE.Color(CLASS_COL[f.cls]);
          for (let y = Math.max(0, Math.floor(f.y0)); y < Math.min(SCENE_H, Math.ceil(f.y1)); y++) for (let x = Math.max(0, Math.floor(f.x0)); x < Math.min(SCENE_W, Math.ceil(f.x1)); x++) {
            if (segP[y * SCENE_W + x] < 0.5) continue;
            for (let yy = Math.floor(y * k); yy < Math.floor((y + 1) * k); yy++) for (let xx = Math.floor(x * k); xx < Math.floor((x + 1) * k); xx++) {
              const q = (yy * w + xx) * 4; id.data[q] = id.data[q] * 0.35 + col.r * 255 * 0.65; id.data[q + 1] = id.data[q + 1] * 0.35 + col.g * 255 * 0.65; id.data[q + 2] = id.data[q + 2] * 0.35 + col.b * 255 * 0.65;
            }
          }
        }
        g.putImageData(id, 0, 0);
      }
      if (s.truth) { g.strokeStyle = '#5ce17a'; g.lineWidth = 3; g.setLineDash([8, 6]); for (const b of sc.boxes) g.strokeRect(b.x0 * k, b.y0 * k, (b.x1 - b.x0) * k, (b.y1 - b.y0) * k); g.setLineDash([]); }
      if ((s.step === 'nms' && shownK >= steps.length) || s.step === 'masks') for (const f of fused) {
        const tb = bestTruth(f);
        g.fillStyle = 'rgba(0,0,0,.7)'; const label = `${CLASSES[f.cls]} ${f.conf.toFixed(2)}${tb ? ` · IoU ${tb.v.toFixed(2)}` : ''}`; g.font = F(19, 'bold'); const tw = g.measureText(label).width;
        const lx = clamp(f.x0 * k, 0, w - tw - 10), ly = f.y0 * k > 28 ? f.y0 * k - 28 : f.y1 * k + 2;
        g.fillRect(lx, ly, tw + 10, 26); txt(g, label, lx + 5, ly + 19, F(19, 'bold'), CLASS_COL[f.cls]);
      }
    });
    board.tex.magFilter = THREE.LinearFilter;
    const bMesh = boardMesh(board.tex, BW, BH); bMesh.position.set(BX, BY, 0); root.add(bMesh);
    const frame = new THREE.Mesh(new THREE.BoxGeometry(BW + 0.12, BH + 0.12, 0.04), M.matte(0x151922)); frame.position.set(BX, BY, -0.03); root.add(frame);
    // threshold sheet
    const sheet = new THREE.Mesh(new THREE.PlaneGeometry(BW, BH), new THREE.MeshBasicMaterial({ color: 0xb8f2e6, transparent: true, opacity: 0.1, depthWrite: false, side: THREE.DoubleSide }));
    sheet.position.set(BX, BY, 0); root.add(sheet);
    const sheetEdge = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(BW, BH)), new THREE.LineBasicMaterial({ color: 0xb8f2e6, transparent: true, opacity: 0.6 })); sheet.add(sheetEdge);
    const lSheet = stage.label('threshold', [BX + BW / 2, BY + BH / 2, 0], root, 'hot');
    // the sliding window
    const win = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(SIZE * U, SIZE * U)), new THREE.LineBasicMaterial({ color: 0xffd166, toneMapped: false })); win.position.z = 0.03; root.add(win);
    // the boxes: one line-segment mesh for all of them (4 edges = 8 points each), plus posts to the picture
    const MAXB = 400, bp = new Float32Array(MAXB * 16 * 3), bc = new Float32Array(MAXB * 16 * 3), bg = new THREE.BufferGeometry();
    bg.setAttribute('position', new THREE.BufferAttribute(bp, 3)); bg.setAttribute('color', new THREE.BufferAttribute(bc, 3));
    const boxes = new THREE.LineSegments(bg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95, toneMapped: false })); root.add(boxes);
    const lAxis = stage.label('height = confidence', [BX - BW / 2 - 0.1, BY - BH / 2, 0.6], root);

    // ---- info board: IoU picture and the NMS list
    const info = canvasTexture(400, 700, (g, w, h, s) => {
      panel(g, w, h); if (!s) return;
      txt(g, STEPS[s.step], 20, 40, F(24, 'bold'), COL.hot);
      // IoU diagram
      txt(g, 'IoU = overlap ÷ union', 20, 84, F(20, 'bold'));
      g.fillStyle = 'rgba(92,225,122,.35)'; g.fillRect(40, 104, 150, 110); g.fillStyle = 'rgba(255,209,102,.35)'; g.fillRect(110, 140, 150, 110);
      g.fillStyle = 'rgba(255,255,255,.7)'; g.fillRect(110, 140, 80, 74);
      g.strokeStyle = '#5ce17a'; g.lineWidth = 3; g.strokeRect(40, 104, 150, 110); g.strokeStyle = COL.hot; g.strokeRect(110, 140, 150, 110);
      txt(g, 'overlap', 150, 184, F(16, 'bold'), '#111', 'center');
      txt(g, '≥ 0.5 usually counts as a hit', 20, 284, F(17), COL.soft);
      let y = 330;
      if (s.step === 'nms' || s.step === 'masks') {
        txt(g, `Steps: ${Math.min(shownK, steps.length)} of ${steps.length}`, 20, y, F(20, 'bold')); y += 32;
        steps.slice(0, Math.min(shownK, 9)).forEach((st, i) => {
          const c = cands[st.keep], ok = st.killed.length + 1 >= VOTES;
          txt(g, `${i + 1}. keep ${CLASSES[c.cls]} ${c.conf.toFixed(2)}`, 20, y, F(17, 'bold'), ok ? CLASS_COL[c.cls] : COL.dim);
          txt(g, ok ? `−${st.killed.length}` : 'too few votes', w - 20, y, F(16), ok ? COL.bad : COL.dim, 'right'); y += 26;
        });
        if (steps.length > 9 && shownK > 9) txt(g, `… and ${Math.min(shownK, steps.length) - 9} more`, 20, y, F(16), COL.soft);
      } else if (s.step === 'scan') {
        txt(g, `${NX} × ${NY} = ${NX * NY} windows`, 20, y, F(20, 'bold')); y += 30;
        txt(g, 'each one: a full CNN run', 20, y, F(17), COL.soft); y += 26; txt(g, 'heat = best object score', 20, y, F(17), COL.soft);
      } else {
        txt(g, `${cands.length} windows above ${s.thr.toFixed(2)}`, 20, y, F(20, 'bold')); y += 30;
        txt(g, 'many boxes per object:', 20, y, F(17), COL.soft); y += 26; txt(g, 'next, NMS removes duplicates', 20, y, F(17), COL.soft);
      }
      txt(g, 'green dashes = true boxes', 20, h - 18, F(16), '#5ce17a');
    });
    const iMesh = boardMesh(info.tex, 1.5, 2.62); iMesh.position.set(BX + BW / 2 + 0.95, BY, 0); root.add(iMesh);

    const bestTruth = (f) => { let best = null; sc.boxes.forEach((b) => { const v = iou(f, b); if (b.cls === f.cls && (!best || v > best.v)) best = { b, v }; }); return best; };
    const detect = (budget) => {
      const t0 = performance.now(), net = lab.net;
      while (done < NX * NY && performance.now() - t0 < budget) {
        const i = done, x = (i % NX) * STRIDE, y = Math.floor(i / NX) * STRIDE, p = net.forward(crop(sc.im, x, y, SIZE, SIZE)), c = argmax(p, 1), b = net.box;
        W[i] = { cls: c, conf: p[c], x0: x + clamp(b[0], -0.2, 1.2) * SIZE, y0: y + clamp(b[1], -0.2, 1.2) * SIZE, x1: x + clamp(b[2], -0.2, 1.2) * SIZE, y1: y + clamp(b[3], -0.2, 1.2) * SIZE, wx: x, wy: y };
        if (W[i].x1 < W[i].x0 + 1) W[i].x1 = W[i].x0 + 1; if (W[i].y1 < W[i].y0 + 1) W[i].y1 = W[i].y0 + 1;
        done++;
      }
    };
    const prepare = (s) => {
      cands = W.filter((d) => d.conf > s.thr).sort((a, b) => b.conf - a.conf);
      steps = nmsSteps(cands, s.nmsIoU);
      fused = steps.filter((st) => st.killed.length + 1 >= VOTES).map((st) => {
        const grp = [st.keep, ...st.killed]; let w = 0; const f = { cls: cands[st.keep].cls, conf: cands[st.keep].conf, x0: 0, y0: 0, x1: 0, y1: 0, votes: grp.length };
        for (const i of grp) { const d = cands[i]; w += d.conf; f.x0 += d.x0 * d.conf; f.y0 += d.y0 * d.conf; f.x1 += d.x1 * d.conf; f.y1 += d.y1 * d.conf; }
        f.x0 /= w; f.y0 /= w; f.x1 /= w; f.y1 /= w; return f;
      });
    };
    const setBox = (n, b, z, col, a = 1) => {
      const [x0, y1] = toW(b.x0, b.y0), [x1, y0] = toW(b.x1, b.y1);
      const P = [[x0, y0], [x1, y0], [x1, y0], [x1, y1], [x1, y1], [x0, y1], [x0, y1], [x0, y0]];
      const o = n * 48;
      P.forEach(([x, y], k) => { bp[o + k * 3] = x; bp[o + k * 3 + 1] = y; bp[o + k * 3 + 2] = z; bc[o + k * 3] = col.r * a; bc[o + k * 3 + 1] = col.g * a; bc[o + k * 3 + 2] = col.b * a; });
      // two posts down to the picture, so the height reads as confidence
      const Q = [[x0, y1, z], [x0, y1, 0.01], [x1, y1, z], [x1, y1, 0.01], [x0, y0, z], [x0, y0, 0.01], [x1, y0, z], [x1, y0, 0.01]];
      Q.forEach(([x, y, zz], k) => { const q = o + 24 + k * 3; bp[q] = x; bp[q + 1] = y; bp[q + 2] = zz; bc[q] = col.r * a * 0.35; bc[q + 1] = col.g * a * 0.35; bc[q + 2] = col.b * a * 0.35; });
    };
    const cGrey = new THREE.Color(0x8a93a6), cBad = new THREE.Color(0xff5a7a), cols = CLASS_COL.map((c) => new THREE.Color(c));

    return {
      update(dt, s) {
        dt = Math.max(0, dt); t += dt;
        if (!mainReady() || !lab.segDone) labStep(inReel() ? 1000 : 10);
        const narrow = fitNarrow(stage, [lAxis, lSheet], -0.12);
        if (s.scene !== sceneSeed) { sceneSeed = s.scene; sc = makeDetScene(90 + s.scene * 7); W = []; done = 0; segP = null; segRun = false; scanT = 0; cacheKey = ''; }
        if (mainReady() && run !== lab.run) { run = lab.run; W = []; done = 0; cacheKey = ''; }
        if (mainReady() && done < NX * NY) detect(inReel() ? 1e9 : 10);
        if (lab.segDone && !segRun && sc) { segRun = true; segP = lab.seg.forward(sc.im).slice(); cacheKey = ''; }
        if (s.play) { s.nmsK = Math.min(1, s.nmsK + dt * 0.18); if (s.nmsK >= 1) s.play = false; this._syncK = true; }
        if (this._syncK) { const el = [...document.querySelectorAll('#panel .ctl')].find((c) => c.querySelector('label')?.textContent.startsWith('Non-max')); const inp = el?.querySelector('input'); if (inp && document.activeElement !== inp) { inp.value = s.nmsK; inp.style.setProperty('--p', s.nmsK * 100 + '%'); el.querySelector('output').textContent = `${Math.round(s.nmsK * 100)}%`; } if (!s.play) this._syncK = false; }
        scanT = s.step === 'scan' ? (scanT + dt * 120) % (NX * NY + 80) : NX * NY;
        const ready = done >= NX * NY;
        const key = `${ready}|${s.thr.toFixed(3)}|${s.nmsIoU.toFixed(3)}|${sceneSeed}|${run}`;
        if (ready && key !== cacheKey) { cacheKey = key; prepare(s); shownK = -1; }
        const K = s.step === 'nms' ? Math.round(s.nmsK * steps.length) : s.step === 'masks' ? steps.length : 0;
        // box states for this NMS step
        if (ready && (K !== shownK || this._stepWas !== s.step || this._bk !== cacheKey || this._truth !== s.truth)) {
          shownK = K; this._stepWas = s.step; this._bk = cacheKey; this._truth = s.truth;
          state = cands.map(() => 'wait');
          steps.slice(0, K).forEach((st) => { state[st.keep] = st.killed.length + 1 >= VOTES ? 'keep' : 'lone'; st.killed.forEach((j) => { state[j] = 'drop'; }); });
          board.redraw(s); info.redraw(s);
        } else if (!ready || s.step === 'scan') { if (Math.floor(t * 8) !== this._q) { this._q = Math.floor(t * 8); board.redraw(s); info.redraw(s); } }
        // draw boxes
        let n = 0;
        if (ready && s.step !== 'scan') {
          if (s.step === 'masks' || (s.step === 'nms' && K >= steps.length)) fused.forEach((f) => { if (n < MAXB) setBox(n++, f, s.step === 'masks' ? 0.04 : 0.05 + f.conf * ZK, cols[f.cls], 1); });
          else cands.forEach((d, i) => {
            if (n >= MAXB) return; const st = state[i];
            if (st === 'drop') { const fall = 0.05 + d.conf * ZK * 0.15; setBox(n++, d, fall, cBad, 0.35); }
            else if (st === 'lone') setBox(n++, d, 0.05 + d.conf * ZK * 0.3, cGrey, 0.4);
            else setBox(n++, d, 0.05 + d.conf * ZK, st === 'keep' ? cols[d.cls] : cols[d.cls], st === 'keep' ? 1 : 0.45);
          });
        }
        bg.setDrawRange(0, n * 16); bg.attributes.position.needsUpdate = true; bg.attributes.color.needsUpdate = true;
        sheet.visible = s.step === 'boxes' || s.step === 'nms'; sheet.position.z = 0.05 + s.thr * ZK; lSheet.position.set(BX + BW / 2, BY + BH / 2, sheet.position.z); lSheet.visible = sheet.visible && !narrow;
        lAxis.visible = sheet.visible && !narrow;
        win.visible = s.step === 'scan';
        if (win.visible) { const i = Math.min(NX * NY - 1, Math.floor(scanT)), x = (i % NX) * STRIDE + SIZE / 2, y = Math.floor(i / NX) * STRIDE + SIZE / 2, [wx, wy] = toW(x, y); win.position.set(wx, wy, 0.03); }
      },
      readout(s) {
        if (!mainReady() || done < NX * NY) return `<div class="big">${mainReady() ? `Scanning… ${Math.round((100 * done) / (NX * NY))}%` : `Training… ${Math.round(progress() * 100)}%`}</div><div class="row"><span>Windows to check</span><b>${NX} × ${NY} = ${NX * NY}</b></div><small>${mainReady() ? 'Each window is one run of the recogniser.' : trainingNote()}</small>`;
        const hits = fused.filter((f) => (bestTruth(f)?.v || 0) >= 0.5), mIoU = hits.length ? hits.reduce((a, f) => a + bestTruth(f).v, 0) / hits.length : 0;
        const found = new Set(hits.map((f) => bestTruth(f).b)).size;
        if (s.step === 'scan') return `<div class="big">${NX * NY} windows checked</div><div class="row"><span>Above the threshold</span><b>${cands.length}</b></div><small>A 20 × 20 window every 2 pixels. Hot = the recogniser thinks an object is centred there.</small>`;
        if (s.step === 'masks') {
          let tp = 0, fp = 0, fn = 0; if (segP) for (let i = 0; i < segP.length; i++) { const m = sc.mask[i] > 0, q = segP[i] > 0.5; if (m && q) tp++; else if (q) fp++; else if (m) fn++; }
          return `<div class="big">Pixel masks</div><div class="row"><span>Mask IoU with the true object pixels</span><b>${segP ? (tp / Math.max(1, tp + fp + fn)).toFixed(2) : lab.segDone ? '…' : 'training…'}</b></div><div class="row"><span>Mask network</span><b>${lab.seg ? lab.seg.params.toLocaleString('en') : '1,401'} weights</b></div><small>${lab.segDone ? 'Each detection paints the object pixels inside its box.' : 'The mask network is still training in your browser.'}</small>`;
        }
        return `<div class="big">${s.step === 'nms' && shownK >= steps.length ? `${fused.length} detections` : `${cands.length} candidate boxes`}</div>
          <div class="row"><span>Objects really there</span><b>${sc.boxes.length}</b></div>
          <div class="row"><span>Found (IoU ≥ 0.5)</span><b>${found} of ${sc.boxes.length}</b></div>
          <div class="row"><span>False alarms</span><b>${fused.length - hits.length}</b></div>
          <div class="row"><span>Average IoU of hits</span><b>${hits.length ? mIoU.toFixed(2) : '–'}</b></div>
          <small>${s.step === 'nms' && shownK < steps.length ? `NMS step ${Math.max(0, shownK)} of ${steps.length}` : 'After NMS and box voting.'}</small>`;
      },
    };
  },
};
