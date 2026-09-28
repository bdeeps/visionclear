// Chapter 3: train a recogniser. The main CNN (lab.js) learns, live, from 1,200 made-up training
// pictures (240 each of nothing, mango, banana, chilli and coin). Every half pass it is tested on 300
// "wild" pictures it never trains on: any angle, dimmer or brighter light, colour casts, more noise.
// With data augmentation, each training picture is randomly turned, flipped, zoomed, shifted,
// re-lit, tinted and sprinkled with noise every time it is used (vision.js augment). Without it, the
// network sees the same 1,200 tidy, mostly upright pictures every pass, and memorises them: training
// accuracy near 100%, test accuracy much lower. (In tests of this exact code: 82% on the wild
// test pictures with augmentation, against 63% without, while training accuracy reached 97%.) The confusion matrix counts, for each true class, what the
// network guessed. "Try your own" makes a new picture with your settings and asks the network.
import { THREE, canvasTexture, clamp } from '../kit.js';
import { makeSample, augment, rng, argmax, CLASSES, CLASS_COL, PALETTE, BACKGROUNDS } from '../vision.js';
import { lab, labStep, labFinish, restart, mainReady, progress, trainingNote, EPOCHS, PER } from '../lab.js';
import { panel, txt, F, COL, boardMesh, fitNarrow, inReel, PixelWall, backPlate, drawImage, probBars, lineChart, legend } from '../visview.js';

const VIEWS = {
  train: { pos: [0.1, 2.75, 7.3], target: [0.1, 2.05, 0] },
  try: { pos: [0.3, 2.65, 6.6], target: [0.3, 1.95, 0] },
};
const TRY_CLASSES = [{ v: 1, label: 'Mango' }, { v: 2, label: 'Banana' }, { v: 3, label: 'Chilli' }, { v: 4, label: 'Coin' }, { v: 0, label: 'Nothing' }];
function tryPicture(s) {
  const r = rng(4000 + s.bgSeed * 31 + s.cls * 7);
  const col = s.cls ? PALETTE[s.cls][s.look % 3] : null;
  return makeSample(s.cls, r, 'train', { ang: s.turn, s: 6.8 * s.size, light: s.light, noise: s.noise, col, bg: BACKGROUNDS[s.bgSeed % BACKGROUNDS.length], dx: 0, dy: 0, cast: [1, 1, 1] });
}

export default {
  id: 'train',
  short: 'Train a recogniser',
  title: 'Train a recogniser, live',
  subtitle: 'Show a network 1,200 labelled pictures, again and again, and it learns to tell a mango from a chilli.',
  view: VIEWS.train,
  learn: `<p>Now let's teach a network to recognise things. When you opened this box, the page drew <b>1,200 made-up pictures</b>, 240 each of a <b>mango</b>, a <b>banana</b>, a <b>chilli</b>, a <b>coin</b> and <b>nothing</b>, each 20 × 20 pixels with its own colour, size, angle and background. Each one comes with its right answer, a <b>label</b>.</p>
    <p>Training works as in NeuralNetClear: show a batch of 16 pictures, measure how wrong the answers are, and nudge all <b>21,081 weights</b> a little to be less wrong (backpropagation). One pass through all 1,200 is an <b>epoch</b>. This network gets 8.</p>
    <p>The real test is pictures it has <b>never seen</b>. Every half pass, it is tested on 300 "wild" pictures: turned any way, in dim or bright light, with colour casts and more noise. The <b>confusion matrix</b> shows what it mixes up: each row is the true answer, each column its guess.</p>
    <p><b>Data augmentation</b> is a cheap trick with a big effect. Each time a training picture is used, it is randomly <b>turned, flipped, zoomed, shifted, re-lit and made noisy</b>. The network never sees exactly the same picture twice, so it can't just memorise. Turn augmentation off and train again: training accuracy shoots towards 100%, but on wild pictures it does clearly worse. It learned the pictures, not the fruit.</p>
    <p class="tip"><b>Try it:</b> watch the test line climb. Then switch off augmentation, press "Train again" and compare the gap between the two lines. Finally open "Try your own" and look for a turn or light level that fools it.</p>`,
  terms: [
    { t: 'Label', d: 'The right answer attached to a training picture.' },
    { t: 'Epoch', d: 'One pass through every training picture.' },
    { t: 'Test set', d: 'Pictures kept aside and never trained on, to check the network on something new.' },
    { t: 'Accuracy', d: 'The share of pictures it gets right.' },
    { t: 'Confusion matrix', d: 'A table of true answers against guesses: it shows which things get mixed up.' },
    { t: 'Data augmentation', d: 'Making new training pictures by randomly turning, flipping, re-lighting and adding noise to old ones.' },
  ],
  defaults: { show: 'train', aug: true, cls: 2, turn: 0.4, size: 1, light: 1, noise: 0.02, look: 0, bgSeed: 1 },
  controls: [
    { key: 'show', type: 'seg', label: 'Show', options: [{ v: 'train', label: 'Training' }, { v: 'try', label: 'Try your own' }] },
    { key: 'aug', type: 'toggle', label: 'Data augmentation', hint: 'Takes effect when you press "Train again".' },
    { key: 'go', type: 'buttons', label: 'Training', items: [{ label: 'Train again', act: (s) => restart(!!s.aug) }] },
    { key: 'cls', type: 'seg', label: 'Your picture', options: TRY_CLASSES },
    { key: 'turn', type: 'range', label: 'Turn', min: 0, max: Math.PI * 2, step: 0.01, fmt: (v) => `${Math.round((v * 180) / Math.PI)}°` },
    { key: 'size', type: 'range', label: 'Size', min: 0.55, max: 1.4, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%` },
    { key: 'light', type: 'range', label: 'Light', min: 0.15, max: 1.8, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%` },
    { key: 'noise', type: 'range', label: 'Sensor noise', min: 0, max: 0.2, step: 0.005, fmt: (v) => `${Math.round(v * 255)} levels` },
    { key: 'look', type: 'seg', label: 'Colour', options: [{ v: 0, label: 'Usual' }, { v: 1, label: 'Unripe / other' }, { v: 2, label: 'Third' }] },
    { key: 'bg', type: 'buttons', label: 'Background', items: [{ label: 'New background', act: (s) => { s.bgSeed = (s.bgSeed || 0) + 1; } }] },
  ],
  quiz: [
    { q: 'Why is the network tested on pictures it never trained on?', options: ['To save time', 'To check it learned the objects, not just memorised its examples', 'Because test pictures are easier', 'It isn\'t, all pictures are used for training'], answer: 1, why: 'Only new pictures show whether it will work in the real world.' },
    { q: 'What does data augmentation do?', options: ['Adds more classes', 'Randomly turns, flips, re-lights and adds noise to training pictures', 'Makes the network bigger', 'Deletes wrong answers'], answer: 1, why: 'The network never sees the same picture twice, so it must learn what stays the same.' },
    { q: 'In a confusion matrix, what do the numbers off the diagonal show?', options: ['Correct answers', 'Mistakes: pictures of one thing guessed as another', 'The weights', 'The training time'], answer: 1, why: 'The diagonal is right answers. Everything else is a mix-up, like a banana called a chilli.' },
  ],
  reel: [
    { ms: 6000, caption: 'Show it 1,200 labelled pictures, again and again, and test it on new ones.', set: { show: 'train', aug: true }, act: () => { labFinish('train'); restart(true); }, spin: 0, view: { pos: [0.1, 2.75, 7.6], target: [0.1, 2.05, 0] } },
  ],

  build({ stage, s: s0 }) {
    s0.aug = lab.aug;
    let view = '', tick = 0, lastTick = -1, histN = -1, key = '', run = -1;
    const root = new THREE.Group(); stage.root.add(root);
    const trainG = new THREE.Group(), tryG = new THREE.Group(); root.add(trainG, tryG);

    // ---- training pictures: one row per class, re-augmented a few times a second
    const COLS = 6;
    const data = canvasTexture(560, 560, (g, w, h) => {
      panel(g, w, h);
      txt(g, `Training pictures (${PER} of each)`, 20, 36, F(24, 'bold'));
      txt(g, lab.aug ? 'augmented: new random changes each time' : 'no augmentation: always the same', 20, 64, F(18), lab.aug ? COL.hot : COL.soft);
      const r = rng(Math.floor(tick * 1.5) + 1), S = 72, X0 = 140, Y0 = 84;
      for (let c = 0; c < 5; c++) {
        txt(g, CLASSES[c], 20, Y0 + c * (S + 22) + S / 2 + 8, F(20, 'bold'), CLASS_COL[c]);
        let k = 0;
        for (let i = c; i < lab.X.length && k < COLS; i += 5, k++) {
          const im = lab.aug ? augment(lab.X[i], r) : lab.X[i];
          drawImage(g, im, X0 + k * (S + 6), Y0 + c * (S + 22), S, S);
        }
      }
    });
    const dMesh = boardMesh(data.tex, 2.3, 2.3); dMesh.position.set(-2.75, 2.1, 0); trainG.add(dMesh);
    // ---- accuracy chart
    const chart = canvasTexture(560, 520, (g, w, h) => {
      panel(g, w, h);
      txt(g, 'How often it is right', 20, 36, F(24, 'bold'));
      const R = { x: 70, y: 70, w: w - 100, h: h - 190 };
      lineChart(g, R, [{ pts: [0.2, ...lab.hist.map((p) => p.train)], col: COL.soft, lw: 2, dash: [6, 5] }, { pts: [0.2, ...lab.hist.map((p) => p.test)], col: COL.good, lw: 4 }], { ymax: 1, xmax: EPOCHS * 2 + 1, xlabel: 'training passes (epochs) →' });
      for (let e = 0; e <= EPOCHS; e += 2) txt(g, String(e), R.x + (e / EPOCHS) * R.w, R.y + R.h + 22, F(16), COL.soft, 'center');
      legend(g, 24, h - 62, [[COL.good, 'new "wild" pictures (test)'], [COL.soft, 'training pictures']]);
      g.fillStyle = 'rgba(255,255,255,.12)'; g.fillRect(24, h - 38, w - 48, 14); g.fillStyle = COL.hot; g.fillRect(24, h - 38, (w - 48) * progress(), 14);
      txt(g, trainingNote(), 24, h - 8, F(16), COL.soft);
    });
    const cMesh = boardMesh(chart.tex, 2.3, 2.14); cMesh.position.set(-0.3, 2.1, 0); trainG.add(cMesh);
    // ---- confusion matrix
    const conf = canvasTexture(560, 560, (g, w, h) => {
      panel(g, w, h);
      txt(g, 'Confusion matrix (300 test pictures)', 20, 36, F(22, 'bold'));
      const S = 76, X0 = 140, Y0 = 110;
      txt(g, 'the network guessed →', X0, 70, F(17), COL.soft);
      for (let j = 0; j < 5; j++) { g.save(); g.translate(X0 + j * S + S / 2 + 6, Y0 - 8); g.rotate(-0.6); txt(g, CLASSES[j], 0, 0, F(16), CLASS_COL[j]); g.restore(); }
      for (let i = 0; i < 5; i++) {
        txt(g, CLASSES[i], X0 - 10, Y0 + i * S + S / 2 + 7, F(18, 'bold'), CLASS_COL[i], 'right');
        for (let j = 0; j < 5; j++) {
          const n = lab.cm ? lab.cm[i][j] : 0, tot = lab.cm ? lab.cm[i].reduce((a, b) => a + b, 0) || 1 : 1, a = n / tot;
          g.fillStyle = i === j ? `rgba(123,224,140,${0.08 + 0.85 * a})` : `rgba(255,90,122,${a > 0 ? 0.12 + 1.6 * a : 0.04})`;
          g.fillRect(X0 + j * S + 2, Y0 + i * S + 2, S - 4, S - 4);
          if (lab.cm) txt(g, String(n), X0 + j * S + S / 2, Y0 + i * S + S / 2 + 8, F(22, i === j ? 'bold' : ''), n ? COL.white : COL.dim, 'center');
        }
      }
      txt(g, '↑ true answer (60 of each)', 20, h - 18, F(17), COL.soft);
    });
    const xMesh = boardMesh(conf.tex, 2.3, 2.3); xMesh.position.set(2.15, 2.1, 0); trainG.add(xMesh);
    const lTrain = stage.label('The network: 20×20×3 → 8 filters → pool → 16 filters → pool → 48 → answer', [-0.3, 0.72, 0], trainG);

    // ---- try your own
    const TW = 20, TC = 0.13, TX = -1.55, TY = 2.1;
    const tg = new THREE.Group(); tg.position.set(TX, TY, 0); tryG.add(tg);
    const wall = new PixelWall(tg, TW, TW, TC, { depth: 0.3 }); const pl = backPlate(TW * TC + 0.12, TW * TC + 0.12); pl.position.z = -0.03; tg.add(pl);
    let probs = new Float32Array(5), tryIm = null;
    const bars = canvasTexture(520, 460, (g, w, h, s) => {
      panel(g, w, h);
      if (!mainReady()) { txt(g, 'Training in your browser…', 20, 44, F(24, 'bold')); g.fillStyle = 'rgba(255,255,255,.12)'; g.fillRect(20, 70, w - 40, 20); g.fillStyle = COL.hot; g.fillRect(20, 70, (w - 40) * progress(), 20); txt(g, trainingNote(), 20, 124, F(18), COL.soft); return; }
      const best = argmax(probs), ok = s && best === s.cls;
      txt(g, `It says: ${CLASSES[best]}`, 20, 44, F(30, 'bold'), ok ? COL.good : COL.bad);
      txt(g, `${Math.round(probs[best] * 100)}% sure · ${ok ? 'right' : 'wrong'}`, 20, 78, F(20), COL.soft);
      probBars(g, probs, 20, 100, w - 40, 58, { font: 24, truth: s ? s.cls : -1 });
      txt(g, 'what the network sees:', 20, h - 16, F(16), COL.soft);
      if (tryIm) drawImage(g, tryIm, w - 80, h - 70, 60, 60);
    });
    const bMesh = boardMesh(bars.tex, 2.6, 2.3); bMesh.position.set(1.75, TY, 0); tryG.add(bMesh);
    const lTry = stage.label('Your picture: 20 × 20 × 3 = 1,200 numbers', [TX, TY - 1.55, 0], tryG, 'hot');

    return {
      update(dt, s) {
        dt = Math.max(0, dt); tick += dt;
        labStep(inReel() ? 55 : s.show === 'train' ? 14 : 8);
        const narrow = fitNarrow(stage, [lTrain], s.show === 'train' ? -0.24 : -0.12);
        // on a phone, stack the three boards: chart on top, pictures and confusion matrix below
        if (narrow !== this._nw) {
          this._nw = narrow;
          if (narrow) { cMesh.position.set(0, 3.25, 0); dMesh.position.set(-1.18, 1.1, 0); xMesh.position.set(1.18, 1.1, 0); dMesh.scale.setScalar(0.94); xMesh.scale.setScalar(0.94); }
          else { cMesh.position.set(-0.3, 2.1, 0); dMesh.position.set(-2.75, 2.1, 0); xMesh.position.set(2.15, 2.1, 0); dMesh.scale.setScalar(1); xMesh.scale.setScalar(1); }
        }
        trainG.visible = s.show === 'train'; tryG.visible = s.show === 'try';
        if (view !== s.show) { if (view) stage.setView(VIEWS[s.show].pos, VIEWS[s.show].target, 0.9); view = s.show; }
        if (s.show === 'train') {
          if (Math.floor(tick * 1.5) !== lastTick || lab.X.length < PER * 5) { lastTick = Math.floor(tick * 1.5); data.redraw(); }
          if (lab.hist.length !== histN || run !== lab.run || Math.floor(tick * 5) !== this._q) { this._q = Math.floor(tick * 5); histN = lab.hist.length; run = lab.run; chart.redraw(); conf.redraw(); }
        } else {
          const k = `${s.cls}|${s.turn.toFixed(3)}|${s.size.toFixed(3)}|${s.light.toFixed(3)}|${s.noise.toFixed(3)}|${s.look}|${s.bgSeed}|${mainReady()}|${lab.run}|${lab.epoch}`;
          if (k !== key) { key = k; tryIm = tryPicture(s); wall.set(tryIm); if (mainReady()) probs = lab.net.forward(tryIm); bars.redraw(s); }
          else if (!mainReady() && Math.floor(tick * 4) !== this._q2) { this._q2 = Math.floor(tick * 4); bars.redraw(s); }
        }
      },
      readout(s) {
        if (s.show === 'try' && mainReady()) {
          const best = argmax(probs);
          return `<div class="big">"${CLASSES[best]}", ${Math.round(probs[best] * 100)}% sure</div>
            <div class="row"><span>You made</span><b>${s.cls ? 'a ' + CLASSES[s.cls] : 'nothing'}</b></div>
            <div class="row"><span>Test accuracy of this network</span><b>${Math.round(lab.acc * 100)}%</b></div>
            <small>${best === s.cls ? 'Right. Try a stranger turn, light or colour.' : 'Fooled! Wild pictures like this are where it still makes mistakes.'}</small>`;
        }
        const h = lab.hist.at(-1);
        return `<div class="big">${mainReady() ? 'Trained' : `Training… ${Math.round(progress() * 100)}%`}</div>
          <div class="row"><span>Augmentation</span><b>${lab.aug ? 'on' : 'off'}</b></div>
          <div class="row"><span>Right on training pictures</span><b>${h ? Math.round(h.train * 100) + '%' : '…'}</b></div>
          <div class="row"><span>Right on new wild pictures</span><b>${h ? Math.round(h.test * 100) + '%' : '…'}</b></div>
          <small>${trainingNote()}. Chance would be 20%.</small>`;
      },
    };
  },
};
