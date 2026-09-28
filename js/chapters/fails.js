// Chapter 5: when it fails. Four views, three of them run live on this box's own networks:
//  - Adversarial noise: an iterative fast-gradient-sign attack (Goodfellow, Shlens & Szegedy 2014;
//    Kurakin, Goodfellow & Bengio 2016), 10 steps, never moving any of the 1,200 numbers by more than
//    ε. The noise is picked using the network's own gradient, so it is not random: random noise of
//    the same size rarely changes the answer (shown for comparison).
//  - Shortcut learning: the shortcut network in lab.js was trained where every fruit sat on its own
//    background colour. Swap the backgrounds and it follows the background, not the object. The
//    saliency map (size of the gradient at each pixel, Simonyan, Vedaldi & Zisserman 2013) shows where
//    it "looks". Real cases: Geirhos et al. 2020, "Shortcut learning in deep neural networks"; Zech et
//    al. 2018 (PLOS Medicine 15(11):e1002683): chest X-ray CNNs could identify the hospital system.
//  - Light: the main network tested on the same 300 wild pictures made darker or brighter.
//  - Faces: sourced numbers, not a model.
//    Gender Shades (Buolamwini & Gebru 2018, PMLR 81:77–91), Pilot Parliaments Benchmark of 1,270
//    people from Rwanda, Senegal, South Africa, Iceland, Finland and Sweden. Error rates (%) of three
//    commercial gender classifiers, table 4:
//      Microsoft: darker female 20.8, darker male 6.0, lighter female 1.7, lighter male 0.0
//      Face++:    darker female 34.5, darker male 0.7, lighter female 9.8, lighter male 0.8
//      IBM:       darker female 34.7, darker male 12.0, lighter female 7.1, lighter male 0.3
//    NIST FRVT Part 3: Demographic Effects (NISTIR 8280, December 2019, Grother, Ngan & Hanaoka):
//    false positive rates often differ by factors of 10 to beyond 100 across demographic groups;
//    false negatives usually by factors below 3; differences depend strongly on the algorithm.
//    EU AI Act (Regulation (EU) 2024/1689), Article 5: real-time remote biometric identification in
//    public spaces for law enforcement is prohibited except in narrowly defined cases.
import { THREE, M, box, canvasTexture, clamp } from '../kit.js';
import { attack, argmax, rng, copyImage, relight, biasedSample, CLASSES, CLASS_COL } from '../vision.js';
import { lab, labStep, labFinish, mainReady, progress, trainingNote } from '../lab.js';
import { panel, txt, F, COL, boardMesh, fitNarrow, inReel, PixelWall, backPlate, drawImage, drawMap, probBars, lineChart } from '../visview.js';

const VIEWS = {
  adv: { pos: [0.2, 2.95, 7.6], target: [0.2, 2.3, 0] },
  shortcut: { pos: [0.2, 2.95, 7.8], target: [0.2, 2.3, 0] },
  light: { pos: [0.2, 2.95, 7.6], target: [0.2, 2.3, 0] },
  faces: { pos: [1.6, 3.1, 7.9], target: [0.7, 1.55, 0] },
};
const SHADES = [
  { co: 'Microsoft', v: [20.8, 6.0, 1.7, 0.0] },
  { co: 'Face++', v: [34.5, 0.7, 9.8, 0.8] },
  { co: 'IBM', v: [34.7, 12.0, 7.1, 0.3] },
];
const GROUPS = ['Darker-skinned women', 'Darker-skinned men', 'Lighter-skinned women', 'Lighter-skinned men'];
const GCOL = [0xff7a59, 0xffb547, 0x8ef0ff, 0x7aa2ff];
const LEVELS = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8];
const SC = { 1: 'Mango', 2: 'Banana', 3: 'Chilli', 4: 'Coin' };

export default {
  id: 'fails',
  short: 'When it fails',
  title: 'How image recognition fails',
  subtitle: 'Invisible noise, lazy shortcuts, bad light and unbalanced data. Knowing the failures is part of knowing the tool.',
  view: VIEWS.adv,
  learn: `<p>A network that scores 90% in a test can still fail in strange ways, because it never "understands" a mango. It found number patterns that happened to work. Here are four ways that goes wrong, three of them run live on this box's own networks.</p>
    <p><b>Adversarial noise.</b> Using the network's own maths, you can work out a tiny change for every pixel that pushes the answer the wrong way. Each number moves by only a few steps out of 255, too little for you to notice, yet the label flips. Random noise of the same size usually does nothing. Researchers have shown the same with stickers on real stop signs.</p>
    <p><b>Shortcuts.</b> A second network here was trained where every banana sat on green and every mango on blue. It scored almost 100%. Swap the backgrounds and it follows the colour behind the fruit, because that was the easiest pattern. The <b>saliency map</b> shows which pixels its answer depends on. Real models have done this too: one reading chest X-rays was found to pick up clues about which hospital took the picture, which was linked to how sick patients there tended to be.</p>
    <p><b>Light and data bias.</b> A network only knows the world its training pictures showed. Dim the light far below anything it trained on and accuracy falls. The same thing happens with people. The 2018 <b>Gender Shades</b> study tested three commercial face-analysis systems: they got the gender of lighter-skinned men wrong at most 0.8% of the time, but of darker-skinned women up to 34.7%. Much of the gap came from unbalanced training data. The companies improved their systems after the study, and a large 2019 test by the US standards agency NIST found error gaps between groups in many (not all) algorithms.</p>
    <p><b>Privacy and surveillance.</b> Face recognition can find missing children, unlock phones and speed up airport queues, like India's DigiYatra. It can also track people without their consent, and a false match can wrongly accuse someone. Countries are drawing different lines: the EU's AI Act bans most live face recognition by police in public places, while others are expanding its use. Where to draw the line is a debate for everyone, not only engineers.</p>
    <p class="tip"><b>Try it:</b> raise the attack strength until the answer flips, then compare "Random noise". In "Shortcut", put a banana on the mango's background and look at the saliency map.</p>`,
  terms: [
    { t: 'Adversarial example', d: 'A picture changed on purpose, often invisibly, so that a model gets it wrong.' },
    { t: 'Gradient', d: 'For each pixel, which way and how much a change would move the answer.' },
    { t: 'Shortcut learning', d: 'When a model uses an easy clue (like the background) instead of the real thing.' },
    { t: 'Saliency map', d: 'A picture of which pixels the answer depends on most.' },
    { t: 'Dataset bias', d: 'When training data under-represents some conditions or groups, so the model does worse on them.' },
    { t: 'False match', d: 'A face recognition system wrongly saying two different people are the same person.' },
  ],
  defaults: { show: 'adv', eps: 0.02, noiseKind: 'attack', advPick: 0, shortCls: 2, shortBg: 1, lvl: 1, face: 0 },
  controls: [
    { key: 'show', type: 'seg', label: 'Show', options: [{ v: 'adv', label: 'Adversarial noise' }, { v: 'shortcut', label: 'Shortcut' }, { v: 'light', label: 'Light' }, { v: 'faces', label: 'Faces and fairness' }] },
    { key: 'eps', type: 'range', label: 'Attack strength ε', min: 0, max: 0.06, step: 0.001, fmt: (v) => `${(v * 255).toFixed(1)} of 255 per number` },
    { key: 'noiseKind', type: 'seg', label: 'Noise', options: [{ v: 'attack', label: 'Worked out (attack)' }, { v: 'random', label: 'Random' }] },
    { key: 'shortCls', type: 'seg', label: 'Shortcut: the object', options: Object.entries(SC).map(([v, l]) => ({ v: +v, label: l })) },
    { key: 'shortBg', type: 'seg', label: 'Shortcut: the background it had in training', options: Object.entries(SC).map(([v, l]) => ({ v: +v, label: l + "'s" })) },
    { key: 'lvl', type: 'range', label: 'Light level', min: 0.1, max: 1.8, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%` },
    { key: 'go', type: 'buttons', label: 'Pictures', items: [{ label: 'Another picture', act: (s) => { s.advPick = (s.advPick || 0) + 1; } }] },
  ],
  quiz: [
    { q: 'What makes adversarial noise different from random noise?', options: ['It is much stronger', 'It is worked out from the network\'s own gradient to push the answer the wrong way', 'It is coloured', 'It only works on phones'], answer: 1, why: 'Each tiny change is chosen to hurt the answer most. Random changes of the same size mostly cancel out.' },
    { q: 'A model trained with every banana on green calls a banana on blue a "mango". Why?', options: ['Bananas look like mangoes', 'It learned the background colour as a shortcut', 'The picture was too small', 'Blue is hard for cameras'], answer: 1, why: 'In its training data the background alone gave the right answer, so it learned that instead of the shape.' },
    { q: 'What did the Gender Shades study find?', options: ['All systems were equally accurate for everyone', 'Error rates were far higher for darker-skinned women than for lighter-skinned men', 'Face analysis never makes mistakes', 'Only one company was tested'], answer: 1, why: 'Up to 34.7% error for darker-skinned women against at most 0.8% for lighter-skinned men, across three commercial systems.' },
  ],
  reel: [
    { ms: 5400, caption: 'Change every pixel by a hair, in just the right way, and the answer flips.', set: { show: 'adv', noiseKind: 'attack', advPick: 0 }, act: () => labFinish('seg'), anim: { eps: [0, 0.03] }, spin: 0, view: { pos: [0.2, 2.95, 7.4], target: [0.2, 2.3, 0] } },
    { ms: 5000, caption: 'Face analysis erred up to 34.7% for darker-skinned women, but under 1% for lighter-skinned men.', set: { show: 'faces' }, spin: 0.35, view: { pos: [1.6, 3.1, 7.6], target: [0.7, 1.55, 0] } },
  ],

  build({ stage }) {
    const root = new THREE.Group(); stage.root.add(root);
    const G = { adv: new THREE.Group(), shortcut: new THREE.Group(), light: new THREE.Group(), faces: new THREE.Group() };
    Object.values(G).forEach((g) => root.add(g));
    let t = 0, view = '';
    const wallAt = (parent, x, y, cell = 0.09) => { const g = new THREE.Group(); g.position.set(x, y, 0); parent.add(g); const w = new PixelWall(g, 20, 20, cell, { depth: 0.2 }); const p = backPlate(20 * cell + 0.1, 20 * cell + 0.1); p.position.z = -0.03; g.add(p); return w; };
    const signTxt = (html, pos, parent) => stage.label(html, pos, parent, 'hot');

    // ================= adversarial
    const aW = [wallAt(G.adv, -2.6, 2.6), wallAt(G.adv, -0.55, 2.6), wallAt(G.adv, 1.5, 2.6)];
    const aL = [stage.label('The picture', [-2.6, 1.55, 0], G.adv), stage.label('Noise × 10', [-0.55, 1.55, 0], G.adv), stage.label('Picture + noise', [1.5, 1.55, 0], G.adv, 'hot')];
    signTxt('+', [-1.58, 2.6, 0], G.adv); signTxt('=', [0.48, 2.6, 0], G.adv);
    let adv = null, orig = null, pOrig = null, pAdv = null, aKey = '';
    const aBars = canvasTexture(900, 320, (g, w, h) => {
      panel(g, w, h); if (!pOrig) return;
      const b0 = argmax(pOrig), b1 = argmax(pAdv), flipped = b0 !== b1;
      txt(g, 'Before', 20, 38, F(24, 'bold')); probBars(g, pOrig, 20, 50, 400, 52, { font: 20 });
      txt(g, flipped ? 'After: fooled!' : 'After', 470, 38, F(24, 'bold'), flipped ? COL.bad : COL.white); probBars(g, pAdv, 470, 50, 410, 52, { font: 20 });
    });
    const abMesh = boardMesh(aBars.tex, 3.9, 1.39); abMesh.position.set(-0.55, 0.78, 0.05); G.adv.add(abMesh);
    const pickAdv = (k) => {
      // the k-th test picture the network gets right, cycling through the classes
      const want = 1 + (k % 4); let n = Math.floor(k / 4);
      // prefer pictures that a small attack (ε = 0.025) can flip, so the slider shows the effect
      let first = -1;
      for (let i = 0; i < lab.TX.length; i++) {
        if (lab.TY[i] !== want || argmax(lab.net.forward(lab.TX[i])) !== want) continue;
        if (first < 0) first = i;
        if (argmax(lab.net.forward(attack(lab.net, lab.TX[i], want, 0.025, 10))) !== want && n-- <= 0) return i;
      }
      return Math.max(0, first);
    };

    // ================= shortcut
    const sW = wallAt(G.shortcut, -2.4, 2.45, 0.1), salW = wallAt(G.shortcut, -0.15, 2.45, 0.1);
    const sL = [stage.label('The test picture', [-2.4, 1.3, 0], G.shortcut), stage.label('Saliency: where the answer comes from', [-0.15, 1.3, 0], G.shortcut, 'hot')];
    let sIm = null, sP = null, shortStats = null, sKey = '';
    const sBoard = canvasTexture(560, 700, (g, w, h, s) => {
      panel(g, w, h);
      if (!lab.shortDone) { txt(g, 'Training the shortcut network…', 20, 44, F(22, 'bold')); txt(g, lab.mainDone ? 'almost there' : 'waits for the main network first', 20, 80, F(18), COL.soft); return; }
      txt(g, 'Its training pictures', 20, 36, F(22, 'bold'));
      for (let c = 0; c < 4; c++) for (let k = 0; k < 5; k++) drawImage(g, lab.shortX[c + 4 * k], 20 + k * 64, 52 + c * 64, 58, 58);
      for (let c = 0; c < 4; c++) txt(g, CLASSES[c + 1], 350, 90 + c * 64, F(18, 'bold'), CLASS_COL[c + 1]);
      txt(g, 'every fruit on its own colour', 20, 332, F(17), COL.soft);
      if (shortStats) {
        txt(g, 'Tested on 200 new pictures', 20, 380, F(22, 'bold'));
        const bar = (y, label, v, col) => { txt(g, label, 20, y, F(18), COL.soft); g.fillStyle = 'rgba(255,255,255,.08)'; g.fillRect(20, y + 10, w - 120, 26); g.fillStyle = col; g.fillRect(20, y + 10, (w - 120) * v, 26); txt(g, `${Math.round(v * 100)}%`, w - 20, y + 31, F(20, 'bold'), COL.white, 'right'); };
        bar(420, 'Usual backgrounds: right', shortStats.same, COL.good);
        bar(490, 'Swapped backgrounds: right', shortStats.swap, COL.bad);
        bar(560, 'Swapped: it named the background\'s fruit', shortStats.follow, COL.hot);
      }
      if (sP) { const b = argmax(sP); txt(g, `This picture: "${CLASSES[b + 1]}" (${Math.round(sP[b] * 100)}%)`, 20, h - 24, F(22, 'bold'), b + 1 === s.shortCls ? COL.good : COL.bad); }
    });
    const sbMesh = boardMesh(sBoard.tex, 2.05, 2.56); sbMesh.position.set(2.3, 2.1, 0); G.shortcut.add(sbMesh);

    // ================= light
    const lW = wallAt(G.light, -2.3, 2.45, 0.1);
    const lL = stage.label('A test picture in this light', [-2.3, 1.3, 0], G.light);
    let lAcc = null, lPos = 0, lIm = null, lP = null, lRun = -1;
    const lBoard = canvasTexture(640, 560, (g, w, h, s) => {
      panel(g, w, h); if (!s) return;
      txt(g, 'Right answers vs light level', 20, 38, F(24, 'bold'));
      const R = { x: 70, y: 70, w: w - 100, h: h - 200 };
      // the range of light seen in training (tidy pictures 0.82–1.18, stretched by augmentation to about 0.5–1.6)
      const X = (v) => R.x + ((v - 0.1) / 1.7) * R.w;
      g.fillStyle = 'rgba(123,224,140,.1)'; g.fillRect(X(0.5), R.y, X(1.6) - X(0.5), R.h); txt(g, 'seen in training', X(1.05), R.y + 22, F(16), COL.good, 'center');
      if (lAcc) {
        g.strokeStyle = COL.hot; g.lineWidth = 4; g.beginPath();
        LEVELS.forEach((v, i) => { if (lAcc[i] === undefined) return; const x = X(v), y = R.y + R.h * (1 - lAcc[i]); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
        LEVELS.forEach((v, i) => { if (lAcc[i] === undefined) return; g.fillStyle = COL.hot; g.beginPath(); g.arc(X(v), R.y + R.h * (1 - lAcc[i]), 5, 0, 7); g.fill(); });
      }
      for (let k = 0; k <= 4; k++) txt(g, `${k * 25}%`, R.x - 8, R.y + R.h * (1 - k / 4) + 6, F(16), COL.soft, 'right');
      [0.1, 0.5, 1, 1.5].forEach((v) => txt(g, `${Math.round(v * 100)}%`, X(v), R.y + R.h + 24, F(16), COL.soft, 'center'));
      txt(g, 'light level →', R.x + R.w, R.y + R.h + 48, F(16), COL.soft, 'right');
      g.strokeStyle = COL.white; g.setLineDash([5, 5]); g.beginPath(); g.moveTo(X(s.lvl), R.y); g.lineTo(X(s.lvl), R.y + R.h); g.stroke(); g.setLineDash([]);
      if (lP) { const b = argmax(lP); txt(g, `This picture: "${CLASSES[b]}", ${Math.round(lP[b] * 100)}% sure`, 20, h - 60, F(22, 'bold'), b === lab.TY[3] ? COL.good : COL.bad); }
      txt(g, lAcc && lAcc.length === LEVELS.length ? '300 wild test pictures at each level' : 'testing…', 20, h - 24, F(17), COL.soft);
    });
    const lbMesh = boardMesh(lBoard.tex, 2.9, 2.54); lbMesh.position.set(0.95, 2.1, 0); G.light.add(lbMesh);

    // ================= faces: Gender Shades as 3D bars
    const fg = new THREE.Group(); fg.position.set(-0.7, 0, 0); G.faces.add(fg);
    const base = box(5.2, 0.06, 2.6, M.matte(0x1b202b)); base.position.set(0, 0.03, 0); fg.add(base);
    const fLabels = [];
    SHADES.forEach((c, i) => {
      const x = -1.7 + i * 1.7;
      c.v.forEach((v, j) => {
        const hgt = 0.02 + v * 0.062, b = box(0.3, hgt, 0.3, M.plastic(GCOL[j], { emissive: GCOL[j], emissiveIntensity: 0.15 }));
        b.position.set(x - 0.51 + j * 0.34, 0.06 + hgt / 2, 0.2); fg.add(b);
        if (j === 0 || j === 3) fLabels.push(stage.label(`${v.toFixed(1)}%`, [x - 0.51 + j * 0.34 - 0.7, 0.2 + hgt + 0.12, 0.2], G.faces, j === 0 ? 'hot' : ''));
      });
      fLabels.push(stage.label(c.co, [x - 0.7, 0.06, 1.25], G.faces));
    });
    const fKey = canvasTexture(620, 560, (g, w, h) => {
      panel(g, w, h);
      txt(g, 'Gender Shades (2018)', 20, 40, F(26, 'bold'), COL.hot);
      txt(g, 'How often 3 commercial systems got', 20, 74, F(19), COL.soft); txt(g, 'a person\'s gender wrong, in %', 20, 100, F(19), COL.soft);
      GROUPS.forEach((gr, j) => { g.fillStyle = '#' + GCOL[j].toString(16).padStart(6, '0'); g.fillRect(20, 128 + j * 36, 22, 22); txt(g, gr, 54, 146 + j * 36, F(19)); });
      txt(g, '1,270 faces of members of parliament', 20, 300, F(18), COL.soft); txt(g, 'from 3 African and 3 Nordic countries.', 20, 326, F(18), COL.soft);
      txt(g, 'NIST 2019, 189 algorithms: false matches', 20, 380, F(18, 'bold')); txt(g, 'often 10 to 100+ times more common for', 20, 406, F(18), COL.soft); txt(g, 'some groups; the size of the gap depends', 20, 432, F(18), COL.soft); txt(g, 'strongly on the algorithm.', 20, 458, F(18), COL.soft);
      txt(g, 'Sources: Buolamwini & Gebru 2018;', 20, h - 46, F(16), COL.dim); txt(g, 'NIST IR 8280 (2019).', 20, h - 22, F(16), COL.dim);
    });
    const fkMesh = boardMesh(fKey.tex, 2.2, 1.99); fkMesh.position.set(3.2, 1.25, 0.3); fkMesh.rotation.y = -0.2; G.faces.add(fkMesh);

    const setView = (s) => { if (view !== s.show) { if (view) stage.setView(VIEWS[s.show].pos, VIEWS[s.show].target, 0.9); view = s.show; } };

    return {
      update(dt, s) {
        dt = Math.max(0, dt); t += dt;
        if (!lab.shortDone) labStep(inReel() ? 1000 : 10);
        const narrow = fitNarrow(stage, [...aL.slice(0, 2), sL[0], lL, ...fLabels.slice(0, 0)], -0.12);
        Object.entries(G).forEach(([k, g]) => { g.visible = s.show === k; });
        setView(s);
        if (!mainReady()) { if (Math.floor(t * 3) !== this._q) { this._q = Math.floor(t * 3); } return; }
        if (s.show === 'adv') {
          const k = `${s.advPick}|${s.eps.toFixed(4)}|${s.noiseKind}|${lab.run}`;
          if (k !== aKey) {
            aKey = k; const idx = pickAdv(s.advPick); orig = lab.TX[idx]; pOrig = lab.net.forward(orig).slice();
            if (s.noiseKind === 'attack') adv = attack(lab.net, orig, lab.TY[idx], s.eps, 10);
            else { const r = rng(77 + s.advPick); adv = copyImage(orig); for (let i = 0; i < adv.d.length; i++) adv.d[i] = clamp(orig.d[i] + (r() < 0.5 ? -1 : 1) * s.eps, 0, 1); }
            pAdv = lab.net.forward(adv).slice();
            const diff = copyImage(orig); for (let i = 0; i < diff.d.length; i++) diff.d[i] = clamp(0.5 + (adv.d[i] - orig.d[i]) * 10, 0, 1);
            aW[0].set(orig); aW[1].set(diff, { hFn: (i) => { let m = 0; for (let c = 0; c < 3; c++) m = Math.max(m, Math.abs(adv.d[c * 400 + i] - orig.d[c * 400 + i])); return m * 12; } }); aW[2].set(adv);
            aBars.redraw(s); this._maxd = adv.d.reduce((m, v, i) => Math.max(m, Math.abs(v - orig.d[i])), 0);
          }
        } else if (s.show === 'shortcut' && lab.shortDone) {
          if (!shortStats) {
            const r = rng(99); let same = 0, swap = 0, follow = 0;
            for (let k = 0; k < 50; k++) for (let c = 1; c <= 4; c++) {
              if (argmax(lab.short.forward(biasedSample(c, r))) === c - 1) same++;
              const b = 1 + ((c + r.int(3)) % 4), g = argmax(lab.short.forward(biasedSample(c, r, b))); if (g === c - 1) swap++; if (g === b - 1) follow++;
            }
            shortStats = { same: same / 200, swap: swap / 200, follow: follow / 200 };
          }
          const k = `${s.shortCls}|${s.shortBg}|${s.advPick}`;
          if (k !== sKey) {
            sKey = k; sIm = biasedSample(s.shortCls, rng(300 + s.shortCls * 11 + s.shortBg * 5 + s.advPick * 3), s.shortBg);
            sP = lab.short.forward(sIm).slice(); const raw = lab.short.saliency(sIm, argmax(sP)), sal = new Float32Array(400);
            for (let y = 0; y < 20; y++) for (let x = 0; x < 20; x++) { let a = 0, n = 0; for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) { const X = x + i, Y = y + j; if (X >= 0 && Y >= 0 && X < 20 && Y < 20) { a += raw[Y * 20 + X]; n++; } } sal[y * 20 + x] = a / n; }
            sW.set(sIm); salW.setMap(sal); sBoard.redraw(s);
          }
        } else if (s.show === 'shortcut' && Math.floor(t * 3) !== this._q3) { this._q3 = Math.floor(t * 3); sBoard.redraw(s); }
        else if (s.show === 'light') {
          if (lRun !== lab.run) { lRun = lab.run; lAcc = []; lPos = 0; }
          const t0 = performance.now();
          while (lAcc.length < LEVELS.length && performance.now() - t0 < (inReel() ? 1e9 : 8)) {
            const L = LEVELS[lAcc.length]; let ok = 0;
            for (let i = 0; i < lab.TX.length; i++) if (argmax(lab.net.forward(relight(lab.TX[i], L))) === lab.TY[i]) ok++;
            lAcc.push(ok / lab.TX.length);
          }
          const k = `${s.lvl.toFixed(3)}|${lAcc.length}|${lab.run}`;
          if (k !== this._lk) { this._lk = k; lIm = relight(lab.TX[3], s.lvl); lP = lab.net.forward(lIm).slice(); lW.set(lIm); lBoard.redraw(s); }
        }
      },
      readout(s) {
        if (s.show === 'faces') return `<div class="big">Up to 34.7% vs 0.8%</div><div class="row"><span>Worst error: darker-skinned women</span><b>20.8–34.7%</b></div><div class="row"><span>Lighter-skinned men</span><b>0.0–0.8%</b></div><small>Gender Shades, 2018: three commercial systems. They were later updated.</small>`;
        if (!mainReady()) return `<div class="big">Training… ${Math.round(progress() * 100)}%</div><small>${trainingNote()}</small>`;
        if (s.show === 'adv' && pOrig) {
          const b0 = argmax(pOrig), b1 = argmax(pAdv);
          return `<div class="big">${b0 === b1 ? `Still "${CLASSES[b1]}"` : `"${CLASSES[b0]}" → "${CLASSES[b1]}"`}</div>
            <div class="row"><span>Biggest change to any number</span><b>${((this._maxd || 0) * 255).toFixed(1)} of 255</b></div>
            <div class="row"><span>Numbers changed</span><b>1,200</b></div>
            <small>${s.noiseKind === 'attack' ? 'Each change is chosen with the network\'s gradient.' : 'Random ± changes of the same size, for comparison.'}</small>`;
        }
        if (s.show === 'shortcut') {
          if (!lab.shortDone) return `<div class="big">Training the shortcut network…</div><small>It trains after the main network.</small>`;
          const b = sP ? argmax(sP) : 0;
          return `<div class="big">"${CLASSES[b + 1]}"</div><div class="row"><span>Really</span><b>a ${CLASSES[s.shortCls]} on the ${CLASSES[s.shortBg]}'s background</b></div>
            <div class="row"><span>Right with usual backgrounds</span><b>${shortStats ? Math.round(shortStats.same * 100) + '%' : '…'}</b></div>
            <div class="row"><span>Right with swapped backgrounds</span><b>${shortStats ? Math.round(shortStats.swap * 100) + '%' : '…'}</b></div>
            <small>It learned the background, not the fruit.</small>`;
        }
        const i = LEVELS.findIndex((v) => v >= s.lvl - 1e-6), a = lAcc && lAcc[i] !== undefined ? lAcc[i] : null;
        return `<div class="big">Light ${Math.round(s.lvl * 100)}%</div><div class="row"><span>Right answers near this level</span><b>${a === null ? '…' : Math.round(a * 100) + '%'}</b></div><div class="row"><span>At normal light (100%)</span><b>${lAcc && lAcc[8] !== undefined ? Math.round(lAcc[8] * 100) + '%' : '…'}</b></div><small>It only knows the light levels its training pictures had.</small>`;
      },
    };
  },
};
