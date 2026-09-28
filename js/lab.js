// The recogniser lab: makes the made-up pictures and trains VisionClear's three small networks right
// here in the browser, a few milliseconds at a time so the page stays smooth. It is shared by every
// chapter and survives chapter changes.
//  1. The main CNN (20×20 colour picture → nothing / mango / banana / chilli / coin, plus a box):
//     conv 3×3 ×8 → ReLU → max-pool → conv 3×3 ×16 → ReLU → max-pool → dense 48 → ReLU → 9 outputs.
//     Trained on 1,200 tidy pictures (240 per class), with or without augmentation, for 8 passes
//     (epochs) in mini-batches of 16 with Adam, then tested on 300 "wild" pictures it never saw.
//  2. The mask network (chapter 4): three dilated 3×3 convolutions and a 1×1 convolution that score
//     every pixel "object or not", trained on 304 busy 24×24 scene crops for 4 epochs.
//  3. The shortcut network (chapter 5): the same CNN design, trained on a biased set in which each
//     fruit always sits on its own background colour.
import { Net, rng, shuffle, argmax, dataset, augment, maskBox, segSample, biasedSample, setBoxW } from './vision.js';

setBoxW(10);        // weight of the box loss against the class loss
export const PER = 240, TEST_PER = 60, EPOCHS = 8, BATCH = 16, LR = 0.004;
let r = rng(5);

export const lab = {
  aug: true, run: 0,
  X: [], Y: [], M: [], TX: [], TY: [], genK: 0,
  net: null, epoch: 0, pos: 0, order: [], hist: [], cm: null, acc: 0, trainAcc: 0, _ok: 0, _seen: 0, evalPos: -1, _cm: null, _okT: 0,
  seg: null, segData: [], segEpoch: 0, segPos: 0, segLoss: [], segDone: false,
  short: null, shortX: [], shortY: [], shortEpoch: 0, shortPos: 0, shortDone: false,
  mainDone: false,
  get phase() { return this.genK < PER ? 'gen' : !this.mainDone ? 'train' : !this.segDone ? 'seg' : !this.shortDone ? 'short' : 'done'; },
};

function newMain() {
  r = rng(5);                                  // same random stream every time, so retraining is repeatable
  lab.net = Net.classifier(5, 3, { withBox: true, hidden: 48 }); lab.epoch = 0; lab.pos = 0; lab.hist = []; lab.cm = null; lab.acc = 0; lab.trainAcc = 0;
  lab._ok = 0; lab._seen = 0; lab.evalPos = -1; lab.order = lab.X.map((_, i) => i); lab.mainDone = false; lab.run++;
}
// Start the main network again from random weights (chapter 3's "Train again").
export function restart(aug = lab.aug) { lab.aug = aug; if (lab.genK >= PER) newMain(); }

function evalChunk(n) {
  const net = lab.net;
  if (lab.evalPos === 0) { lab._cm = Array.from({ length: 5 }, () => new Array(5).fill(0)); lab._okT = 0; }
  for (let k = 0; k < n && lab.evalPos < lab.TX.length; k++, lab.evalPos++) {
    const i = lab.evalPos, g = argmax(net.forward(lab.TX[i])); lab._cm[lab.TY[i]][g]++; if (g === lab.TY[i]) lab._okT++;
  }
  if (lab.evalPos >= lab.TX.length) {
    lab.cm = lab._cm; lab.acc = lab._okT / lab.TX.length;
    lab.hist.push({ t: lab.epoch + lab.pos / lab.X.length, test: lab.acc, train: lab._seen ? lab._ok / lab._seen : 0 });
    lab._ok = 0; lab._seen = 0; lab.evalPos = -1;
    if (lab.epoch >= EPOCHS) lab.mainDone = true;
  }
}

export function labStep(budget = 8) {
  const t0 = performance.now();
  while (performance.now() - t0 < budget && lab.phase !== 'done') {
    const ph = lab.phase;
    if (ph === 'gen') {
      const tr = dataset(8, 1000 + lab.genK, 'train'), te = lab.genK % 32 === 0 ? dataset(8, 9000 + lab.genK, 'wild') : null;
      lab.X.push(...tr.X); lab.Y.push(...tr.Y); lab.M.push(...tr.M); lab.genK += 8;
      if (te && lab.TX.length < TEST_PER * 5) { lab.TX.push(...te.X); lab.TY.push(...te.Y); }
      if (lab.genK >= PER) { lab.TX.length = Math.min(lab.TX.length, TEST_PER * 5); lab.TY.length = lab.TX.length; while (lab.TX.length < TEST_PER * 5) { const t = dataset(4, 7000 + lab.TX.length, 'wild'); lab.TX.push(...t.X); lab.TY.push(...t.Y); } newMain(); }
    } else if (ph === 'train') {
      if (lab.evalPos >= 0) { evalChunk(12); continue; }
      const net = lab.net;
      if (lab.pos === 0) shuffle(lab.order, r);
      net.zero(); let n = 0;
      for (let k = lab.pos; k < Math.min(lab.X.length, lab.pos + BATCH); k++, n++) {
        const i = lab.order[k], a = lab.aug ? augment(lab.X[i], r, 1, lab.M[i]) : { ...lab.X[i], mask: lab.M[i] };
        const p = net.forward(a); if (argmax(p) === lab.Y[i]) lab._ok++; lab._seen++;
        net.backward(lab.Y[i], false, lab.Y[i] ? maskBox(a.mask, a.w, a.h) : null);
      }
      const half = lab.X.length / 2, before = lab.pos;
      // the step size shrinks smoothly over training (cosine schedule, Loshchilov & Hutter 2016), to settle
      const prog = (lab.epoch + lab.pos / lab.X.length) / EPOCHS;
      net.step(LR * (0.15 + 0.85 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, prog)))), n); lab.pos += BATCH;
      if (lab.pos >= lab.X.length) { lab.pos = 0; lab.epoch++; lab.evalPos = 0; }
      else if (before < half && lab.pos >= half) lab.evalPos = 0;   // also test half-way through each pass
    } else if (ph === 'seg') {
      if (!lab.seg) { lab.seg = Net.segmenter(2); const rs = rng(8); lab.segData = Array.from({ length: 304 }, () => segSample(rs)); }
      const net = lab.seg; net.zero(); let L = 0;
      for (let k = lab.segPos; k < lab.segPos + 8; k++) { const s = lab.segData[k]; net.forward(s.im); L += net.loss(s.mask); net.backward(s.mask); }
      net.step(0.01, 8); lab.segLoss.push(L / 8); lab.segPos += 8;
      if (lab.segPos >= lab.segData.length) { lab.segPos = 0; lab.segEpoch++; if (lab.segEpoch >= 4) lab.segDone = true; }
    } else if (ph === 'short') {
      if (!lab.short) {
        lab.short = Net.classifier(4, 9); const rs = rng(12);
        for (let k = 0; k < 100; k++) for (let c = 1; c <= 4; c++) { lab.shortX.push(biasedSample(c, rs)); lab.shortY.push(c - 1); }
        lab.shortOrder = lab.shortX.map((_, i) => i);
      }
      const net = lab.short; if (lab.shortPos === 0) shuffle(lab.shortOrder, r);
      net.zero();
      for (let k = lab.shortPos; k < lab.shortPos + 16; k++) { const i = lab.shortOrder[k]; net.forward(lab.shortX[i]); net.backward(lab.shortY[i]); }
      net.step(LR, 16); lab.shortPos += 16;
      if (lab.shortPos >= lab.shortX.length) { lab.shortPos = 0; lab.shortEpoch++; if (lab.shortEpoch >= 3) lab.shortDone = true; }
    }
  }
}
export const labFinish = (until = 'done') => {
  const order = ['gen', 'train', 'seg', 'short', 'done'];
  while (order.indexOf(lab.phase) < order.indexOf(until)) labStep(200);
};
export const mainReady = () => lab.mainDone || lab.phase === 'seg' || lab.phase === 'short' || lab.phase === 'done';
// 0..1 progress of the main network (picture making counts for the first 10%).
export function progress() {
  if (lab.genK < PER) return 0.1 * lab.genK / PER;
  if (lab.mainDone) return 1;
  return 0.1 + 0.9 * Math.min(1, (lab.epoch + lab.pos / lab.X.length) / EPOCHS);
}
export function trainingNote() {
  if (lab.genK < PER) return `Drawing made-up pictures: ${(lab.genK * 5).toLocaleString('en')} of ${(PER * 5).toLocaleString('en')}`;
  if (!lab.mainDone) return `Training pass ${Math.min(EPOCHS, lab.epoch + 1)} of ${EPOCHS}${lab.aug ? ', with augmentation' : ', no augmentation'}`;
  return 'Trained';
}
