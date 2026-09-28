// Chapter 6: where image recognition is used, and what comes next. Each app shows what goes in and
// what comes out. Three are live: the photo search runs this box's own trained CNN on 16 new pictures;
// the QR view runs the classic 1:1:3:1:1 finder-pattern scan (vision.js findFinders, no learning);
// the vision-transformer view really cuts a picture into patches. The others are labelled illustrations.
// Sources for the claims in the text:
//  - Diabetic retinopathy in India: Gulshan et al., JAMA Ophthalmology 2019;137(9):987–993. 3,049
//    patients at Aravind Eye Hospital and Sankara Nethralaya. Algorithm sensitivity/specificity for
//    referable DR: 88.9% / 92.2% (Aravind), 92.1% / 95.2% (Sankara Nethralaya).
//  - Crop disease: Mohanty, Hughes & Salathé, Frontiers in Plant Science 2016, 7:1419: 99.35% on
//    held-out PlantVillage lab photos (54,306 images, 14 crops, 26 diseases), but about 31% on photos
//    taken elsewhere. Farmer apps such as Plantix (PEAT, Berlin) are widely used in India (as reported).
//  - QR codes: invented at Denso Wave, Japan, in 1994 (Masahiro Hara's team); ISO/IEC 18004 defines
//    the finder patterns and Reed–Solomon error correction. UPI payments in India use QR codes
//    specified by NPCI (Bharat QR / UPI QR).
//  - Vision Transformer: Dosovitskiy et al. 2020, arXiv:2010.11929 ("An Image is Worth 16x16 Words").
//  - CLIP: Radford et al. 2021, arXiv:2103.00020: trained on 400 million image–text pairs.
//  - Number-plate reading (ANPR) for e-challans is used by traffic police in several Indian cities (as
//    reported); FASTag toll payment uses RFID tags, not cameras.
import { THREE, M, box, canvasTexture, clamp } from '../kit.js';
import { makeSample, rng, argmax, CLASSES, CLASS_COL, qrSymbol, qrImage, findFinders, patches, crop } from '../vision.js';
import { lab, labStep, labFinish, mainReady, progress, trainingNote } from '../lab.js';
import { panel, txt, F, COL, boardMesh, fitNarrow, inReel, drawImage, roundRect } from '../visview.js';

const APPS = {
  photos: { name: 'Photo search', model: 'cnn', in: 'your gallery', out: 'the pictures that match', note: 'Live: this box\'s own network searches 16 new pictures.' },
  eyes: { name: 'Eye screening', model: 'cnn', in: 'a photo of the back of the eye', out: '"refer to a doctor" or "looks fine"', note: 'Illustration. Tested in India at Aravind and Sankara Nethralaya (2019).' },
  crops: { name: 'Crop disease', model: 'cnn', in: 'a leaf photo from a farmer\'s phone', out: 'a likely disease, and what to do', note: 'Illustration. Great in the lab, much weaker in real fields.' },
  roads: { name: 'Traffic cameras', model: 'cnn', in: 'a road camera frame', out: 'vehicles, their speed, a number plate', note: 'Illustration. Plate text here is made up.' },
  factory: { name: 'Factory checks', model: 'cnn', in: 'every item on a conveyor', out: 'pass, or reject with the flaw marked', note: 'Illustration.' },
  qr: { name: 'UPI QR codes', model: 'rules', in: 'a phone camera frame', out: 'three corner squares found, then the code', note: 'Live: the classic finder-pattern scan. No learning at all.' },
  next: { name: 'What\'s next', model: 'vit', in: 'a picture, cut into patches', out: 'words: a caption or an answer', note: 'Live patches; the caption is an illustration.' },
};

function fundus(g, X, Y, S) {
  const cx = X + S / 2, cy = Y + S / 2, R = S * 0.46, gr = g.createRadialGradient(cx, cy, 10, cx, cy, R);
  gr.addColorStop(0, '#f2a24a'); gr.addColorStop(1, '#7a2a10'); g.fillStyle = gr; g.beginPath(); g.arc(cx, cy, R, 0, 7); g.fill();
  g.fillStyle = '#ffe2a0'; g.beginPath(); g.arc(cx - R * 0.35, cy, R * 0.14, 0, 7); g.fill();
  g.strokeStyle = 'rgba(120,20,10,.85)'; g.lineWidth = 3;
  for (let k = 0; k < 6; k++) { const a = -1.3 + k * 0.5; g.beginPath(); g.moveTo(cx - R * 0.35, cy); g.quadraticCurveTo(cx + Math.cos(a) * R * 0.4, cy + Math.sin(a) * R * 0.7, cx + Math.cos(a) * R * 0.95, cy + Math.sin(a) * R * 0.9); g.stroke(); }
  g.fillStyle = '#fff09a'; [[0.2, -0.2], [0.35, 0.1], [0.1, 0.3], [0.42, -0.05]].forEach(([u, v]) => { g.beginPath(); g.arc(cx + u * R, cy + v * R, 4, 0, 7); g.fill(); });
  g.fillStyle = '#b01010'; [[0.25, 0.22], [0.05, -0.3]].forEach(([u, v]) => { g.beginPath(); g.arc(cx + u * R, cy + v * R, 5, 0, 7); g.fill(); });
}
function leaf(g, X, Y, S, t) {
  g.save(); g.translate(X + S / 2, Y + S / 2); g.rotate(-0.5);
  g.fillStyle = '#3f8f3a'; g.beginPath(); g.moveTo(0, -S * 0.42); g.quadraticCurveTo(S * 0.34, -S * 0.1, 0, S * 0.42); g.quadraticCurveTo(-S * 0.34, -S * 0.1, 0, -S * 0.42); g.fill();
  g.strokeStyle = '#2a6a28'; g.lineWidth = 3; g.beginPath(); g.moveTo(0, -S * 0.4); g.lineTo(0, S * 0.42); g.stroke();
  for (let k = -3; k <= 3; k++) { g.beginPath(); g.moveTo(0, k * S * 0.1); g.lineTo(S * 0.18, k * S * 0.1 - S * 0.08); g.moveTo(0, k * S * 0.1); g.lineTo(-S * 0.18, k * S * 0.1 - S * 0.08); g.stroke(); }
  [[0.08, -0.1], [-0.1, 0.08], [0.06, 0.2], [-0.05, -0.22]].forEach(([u, v], i) => { g.fillStyle = '#7a5a1a'; g.beginPath(); g.arc(u * S, v * S, S * 0.035, 0, 7); g.fill(); g.strokeStyle = '#d8c040'; g.lineWidth = 2; g.stroke(); });
  g.restore();
}
function road(g, X, Y, W, H, t) {
  g.fillStyle = '#2b2f36'; g.fillRect(X, Y, W, H); g.strokeStyle = '#e8e8e8'; g.setLineDash([22, 18]); g.lineWidth = 4;
  g.beginPath(); g.moveTo(X + W / 2, Y); g.lineTo(X + W / 2, Y + H); g.stroke(); g.setLineDash([]);
  const cars = [[0.28, (t * 0.12) % 1.3 - 0.2, '#d04040'], [0.72, (t * 0.18 + 0.5) % 1.3 - 0.2, '#4080d0'], [0.3, (t * 0.12 + 0.6) % 1.3 - 0.2, '#e0c040']];
  for (const [u, v, c] of cars) { const cx = X + u * W, cy = Y + v * H; g.fillStyle = c; roundRect(g, cx - 26, cy - 40, 52, 80, 10); g.fill(); g.fillStyle = 'rgba(180,220,255,.8)'; g.fillRect(cx - 18, cy - 28, 36, 16); g.fillStyle = '#f4f4f0'; g.fillRect(cx - 16, cy + 30, 32, 9); }
}
function conveyor(g, X, Y, W, H, t) {
  g.fillStyle = '#23272e'; g.fillRect(X, Y + H * 0.3, W, H * 0.4);
  for (let k = 0; k < 12; k++) { const x = X + ((k * 60 + t * 40) % (W + 60)) - 30; g.fillStyle = 'rgba(255,255,255,.06)'; g.fillRect(x, Y + H * 0.3, 4, H * 0.4); }
  for (let k = 0; k < 4; k++) {
    const x = X + ((k * 110 + t * 40) % (W + 110)) - 55, y = Y + H / 2, bad = k === 2;
    g.fillStyle = '#c9ccd4'; g.beginPath(); g.arc(x, y, 34, 0, 7); g.fill(); g.strokeStyle = '#8a8f99'; g.lineWidth = 4; g.beginPath(); g.arc(x, y, 26, 0, 7); g.stroke();
    if (bad) { g.fillStyle = '#23272e'; g.beginPath(); g.moveTo(x + 20, y - 28); g.lineTo(x + 36, y - 8); g.lineTo(x + 30, y - 30); g.fill(); g.strokeStyle = COL.bad; g.lineWidth = 3; g.strokeRect(x + 10, y - 40, 34, 36); }
  }
}

export default {
  id: 'uses',
  short: 'Where it\'s used',
  title: 'Where computers see, and what comes next',
  subtitle: 'From photo search to eye screening, farms, roads, factories and UPI payments. Plus the models that look and talk.',
  view: { pos: [0.2, 2.95, 7.8], target: [0.2, 2.3, 0] },
  learn: `<p>The same ideas, pixels in, filters, layers, an answer out, run in many places around you. Pick one.</p>
    <p><b>Phone photo search.</b> Type "mango" in a gallery app and a network like this box's finds the pictures, without anyone having labelled them. The search here is live: this box's own network looks through 16 new pictures.</p>
    <p><b>Eye screening.</b> People with diabetes can slowly lose sight from diabetic retinopathy. In a 2019 study of 3,049 patients at Aravind Eye Hospital and Sankara Nethralaya in India, a network spotted cases that needed a doctor about as well as, or better than, human graders. It is a <b>screening aid</b>: doctors still decide.</p>
    <p><b>Crops.</b> Phone apps can suggest a leaf disease from a photo, and millions of farmers in India have reportedly tried them. But be careful: one well-known network scored 99% on tidy lab photos and only about 31% on photos taken elsewhere. Chapter 5's lesson again.</p>
    <p><b>Roads and factories.</b> Traffic cameras count vehicles and read <b>number plates</b> for fines. Factory cameras check every tablet, circuit board or coin for flaws, far faster than people.</p>
    <p><b>UPI QR codes.</b> Surprise: scanning a QR code mostly uses <b>no learning</b>. The scanner hunts for the three corner squares, whose stripes always come in the ratio 1:1:3:1:1, then straightens the grid and fixes errors with clever maths. Classic vision is still everywhere.</p>
    <p><b>What's next.</b> <b>Vision transformers</b> cut a picture into patches and treat them like words (see LLMClear). <b>Vision-language models</b> learn from hundreds of millions of pictures with captions, so you can ask a question about a photo and get an answer in words. They can also describe things wrongly and confidently, so check what matters. Video models (VideoModelClear) and running models fast on phones (InferenceClear) are the next steps.</p>
    <p class="tip"><b>Try it:</b> in Photo search, switch the search word and count the mistakes. In UPI QR codes, turn the code and watch the corner squares still get found.</p>`,
  terms: [
    { t: 'Screening aid', d: 'A tool that flags who should see an expert. It supports, and does not replace, the doctor.' },
    { t: 'ANPR', d: 'Automatic number-plate recognition: finding a plate in a frame and reading its letters.' },
    { t: 'Finder pattern', d: 'The three big squares in a QR code\'s corners, easy to find at any angle.' },
    { t: 'Vision transformer (ViT)', d: 'A model that splits a picture into patches and uses attention between them, like words in a sentence.' },
    { t: 'Vision-language model', d: 'A model trained on pictures with text, which can caption images or answer questions about them.' },
  ],
  defaults: { app: 'photos', query: 2, gal: 0, qrAng: 0.35, patch: 5 },
  controls: [
    { key: 'app', type: 'seg', label: 'Where', options: Object.entries(APPS).map(([v, a]) => ({ v, label: a.name })) },
    { key: 'query', type: 'seg', label: 'Photo search for', options: [1, 2, 3, 4].map((v) => ({ v, label: CLASSES[v] })) },
    { key: 'qrAng', type: 'range', label: 'QR code turn', min: -1.2, max: 1.2, step: 0.01, fmt: (v) => `${Math.round((v * 180) / Math.PI)}°` },
    { key: 'patch', type: 'seg', label: 'ViT patch size', options: [{ v: 4, label: '4 × 4' }, { v: 5, label: '5 × 5' }, { v: 10, label: '10 × 10' }] },
    { key: 'go', type: 'buttons', label: 'Pictures', items: [{ label: 'New gallery', act: (s) => { s.gal = (s.gal || 0) + 1; } }] },
  ],
  quiz: [
    { q: 'In the 2019 Indian eye-screening study, what role did the network play?', options: ['It replaced eye doctors', 'A screening aid that flags who should see a doctor', 'It performed surgery', 'It only counted patients'], answer: 1, why: 'It matched or beat human graders at spotting referable cases, but doctors still decide.' },
    { q: 'How does a phone mainly find a UPI QR code in a camera frame?', options: ['A huge neural network', 'By looking for corner squares whose stripes run 1:1:3:1:1', 'By asking a server', 'By reading the colour'], answer: 1, why: 'The finder patterns give that ratio at any angle. Error-correcting maths then reads the code. No learning needed.' },
    { q: 'What does a vision transformer do first with a picture?', options: ['Turns it into sound', 'Cuts it into small patches and treats them like words', 'Blurs it', 'Prints it'], answer: 1, why: 'Each patch becomes a token, and attention lets every patch look at every other, as in language models.' },
  ],
  reel: [
    { ms: 5000, caption: 'Scanning a UPI QR code needs no learning: just find three squares with 1:1:3:1:1 stripes.', set: { app: 'qr' }, act: () => labFinish('done'), anim: { qrAng: [-0.6, 0.6] }, spin: 0, view: { pos: [0.2, 2.9, 7.6], target: [0.2, 2.3, 0] } },
  ],

  build({ stage }) {
    const root = new THREE.Group(); stage.root.add(root);
    let t = 0, gal = null, galP = null, galKey = '', qr = null, qrKey = '', found = null, vitPatches = [], vitKey = '', vitIm = null;

    // ---- the input and output boards
    const inB = canvasTexture(560, 560, (g, w, h, s) => {
      panel(g, w, h); if (!s) return; const A = APPS[s.app];
      txt(g, 'In', 20, 40, F(26, 'bold'), COL.hot); txt(g, A.in, 66, 40, F(20), COL.soft);
      const X = 30, Y = 64, S = w - 60, H = h - 110;
      if (s.app === 'photos' && gal) gal.forEach((im, i) => drawImage(g, im, X + (i % 4) * (S / 4) + 4, Y + Math.floor(i / 4) * (H / 4) + 4, S / 4 - 8, H / 4 - 8));
      else if (s.app === 'eyes') fundus(g, X + (S - H) / 2, Y, H);
      else if (s.app === 'crops') leaf(g, X + (S - H) / 2, Y, H, t);
      else if (s.app === 'roads') road(g, X, Y, S, H, t);
      else if (s.app === 'factory') conveyor(g, X, Y, S, H, t);
      else if (s.app === 'qr' && qr) {
        drawImage(g, qr, X + (S - H) / 2, Y, H, H); const k = H / qr.w, ox = X + (S - H) / 2;
        const y = Math.floor((t * 30) % qr.h), rows = found.rowsHit.filter((r) => r.y <= y);
        g.fillStyle = 'rgba(255,209,102,.35)'; rows.forEach((r) => g.fillRect(ox + r.x0 * k, Y + r.y * k, (r.x1 - r.x0) * k, k));
        g.strokeStyle = COL.hot; g.lineWidth = 2; g.beginPath(); g.moveTo(ox, Y + y * k); g.lineTo(ox + H, Y + y * k); g.stroke();
      } else if (s.app === 'next' && vitIm) {
        const P = s.patch, n = vitIm.w / P, gap = 6, cs = (H - gap * (n - 1)) / n;
        vitPatches.forEach((p, i) => drawImage(g, p.im, X + (S - H) / 2 + (i % n) * (cs + gap), Y + Math.floor(i / n) * (cs + gap), cs, cs));
      }
      txt(g, A.note, 20, h - 18, F(16), COL.soft);
    });
    const outB = canvasTexture(560, 560, (g, w, h, s) => {
      panel(g, w, h); if (!s) return; const A = APPS[s.app];
      txt(g, 'Out', 20, 40, F(26, 'bold'), COL.good); txt(g, A.out, 78, 40, F(19), COL.soft);
      const X = 30, Y = 70, S = w - 60, H = h - 120;
      if (s.app === 'photos') {
        if (!mainReady()) { txt(g, 'Training in your browser…', X, Y + 40, F(22, 'bold')); g.fillStyle = COL.hot; g.fillRect(X, Y + 60, S * progress(), 14); return; }
        txt(g, `Search: "${CLASSES[s.query]}"`, X, Y + 10, F(24, 'bold'), CLASS_COL[s.query]);
        gal.forEach((im, i) => {
          const hit = argmax(galP[i]) === s.query, x = X + (i % 4) * (S / 4) + 4, y = Y + 30 + Math.floor(i / 4) * ((H - 40) / 4) + 4, cw = S / 4 - 8, ch = (H - 40) / 4 - 8;
          drawImage(g, im, x, y, cw, ch, { alpha: hit ? 1 : 0.18 });
          if (hit) { const right = gal.truth[i] === s.query; g.strokeStyle = right ? COL.good : COL.bad; g.lineWidth = 5; g.strokeRect(x, y, cw, ch); if (!right) txt(g, '✗', x + cw - 18, y + 26, F(24, 'bold'), COL.bad); }
        });
      } else if (s.app === 'eyes') {
        const bars = [['refer', 0.86, COL.pos], ['looks fine', 0.14, COL.neg]];
        bars.forEach(([k, p, c], i) => { const y = Y + 20 + i * 70; txt(g, k, X, y + 30, F(24, 'bold')); g.fillStyle = 'rgba(255,255,255,.08)'; g.fillRect(X + 150, y + 8, S - 220, 32); g.fillStyle = c; g.fillRect(X + 150, y + 8, (S - 220) * p, 32); txt(g, `${Math.round(p * 100)}%`, X + S, y + 32, F(20), COL.soft, 'right'); });
        txt(g, 'In the 2019 Indian study (3,049 patients):', X, Y + 200, F(19, 'bold'));
        txt(g, 'Aravind: caught 88.9% of cases,', X, Y + 234, F(19)); txt(g, '92.2% right on healthy eyes', X, Y + 260, F(19), COL.soft);
        txt(g, 'Sankara Nethralaya: 92.1% and 95.2%', X, Y + 296, F(19)); txt(g, 'A screening aid. A doctor decides.', X, Y + 346, F(20, 'bold'), COL.hot);
      } else if (s.app === 'crops') {
        txt(g, 'Likely: leaf spot (illustration)', X, Y + 20, F(22, 'bold')); txt(g, 'Advice: ask the local agriculture office', X, Y + 52, F(18), COL.soft);
        txt(g, 'One famous test (2016):', X, Y + 120, F(20, 'bold'));
        const bar = (y, l, v, c) => { txt(g, l, X, y, F(18)); g.fillStyle = 'rgba(255,255,255,.08)'; g.fillRect(X, y + 10, S - 80, 28); g.fillStyle = c; g.fillRect(X, y + 10, (S - 80) * v, 28); txt(g, `${v === 0.9935 ? '99.35' : '31'}%`, X + S, y + 32, F(20, 'bold'), COL.white, 'right'); };
        bar(Y + 160, 'Tidy lab photos like its training set', 0.9935, COL.good); bar(Y + 240, 'Photos taken elsewhere', 0.314, COL.bad);
        txt(g, 'Real fields are messier than the lab.', X, Y + 330, F(19, 'bold'), COL.hot);
      } else if (s.app === 'roads') {
        txt(g, '3 vehicles · lane 1 · 42 km/h', X, Y + 20, F(22, 'bold'));
        g.fillStyle = '#f4f4f0'; roundRect(g, X + 40, Y + 70, S - 80, 90, 10); g.fill(); g.strokeStyle = '#111'; g.lineWidth = 4; g.stroke();
        txt(g, 'AB 12 CD 3456', X + S / 2, Y + 132, F(46, 'bold'), '#111', 'center'); txt(g, 'made-up plate', X + S / 2, Y + 188, F(16), COL.soft, 'center');
        txt(g, 'Find the plate → straighten it →', X, Y + 250, F(19)); txt(g, 'split into characters → read each one', X, Y + 278, F(19)); txt(g, 'Night, rain and dirty plates cause misreads.', X, Y + 330, F(18), COL.hot);
      } else if (s.app === 'factory') {
        txt(g, 'Item 3: REJECT', X, Y + 20, F(26, 'bold'), COL.bad); txt(g, 'chipped rim, top right', X, Y + 54, F(19), COL.soft);
        txt(g, 'Items 1, 2, 4: pass', X, Y + 100, F(22, 'bold'), COL.good);
        txt(g, 'Cameras check tablets, circuit boards,', X, Y + 170, F(19)); txt(g, 'bottles and fabric, many items a second.', X, Y + 198, F(19));
        txt(g, 'Flaws are rare, so there are few "bad"', X, Y + 250, F(18), COL.soft); txt(g, 'examples to learn from: a hard problem.', X, Y + 276, F(18), COL.soft);
      } else if (s.app === 'qr' && found) {
        const c = found.centres, k = H / qr.w, ox = X + (S - H) / 2;
        drawImage(g, qr, ox, Y, H, H, { alpha: 0.4 });
        c.forEach((q, i) => { g.strokeStyle = COL.good; g.lineWidth = 4; g.beginPath(); g.arc(ox + q.x * k, Y + q.y * k, q.u * 4 * k, 0, 7); g.stroke(); txt(g, String(i + 1), ox + q.x * k, Y + q.y * k + 8, F(24, 'bold'), COL.good, 'center'); });
        if (c.length === 3) { g.strokeStyle = COL.hot; g.lineWidth = 3; g.beginPath(); c.forEach((q, i) => (i ? g.lineTo(ox + q.x * k, Y + q.y * k) : g.moveTo(ox + q.x * k, Y + q.y * k))); g.closePath(); g.stroke(); }
        txt(g, c.length === 3 ? '3 corners found: now straighten and read' : `${c.length} corners found`, X, h - 44, F(19, 'bold'), c.length === 3 ? COL.good : COL.bad);
      } else if (s.app === 'next') {
        const n = vitPatches.length;
        txt(g, `${n} patches → ${n} tokens`, X, Y + 16, F(22, 'bold'));
        txt(g, 'attention between every pair:', X, Y + 50, F(18), COL.soft); txt(g, `${n} × ${n} = ${(n * n).toLocaleString('en')} scores per head`, X, Y + 76, F(18), COL.soft);
        g.fillStyle = 'rgba(184,242,230,.1)'; roundRect(g, X, Y + 110, S, 120, 12); g.fill();
        txt(g, 'A vision-language model might say:', X + 16, Y + 146, F(18), COL.soft); txt(g, '"A banana on a cloth."', X + 16, Y + 190, F(28, 'bold'), COL.teal);
        txt(g, 'illustration: no language model runs here', X, Y + 262, F(16), COL.dim);
        txt(g, 'CLIP (2021) learned from 400 million', X, Y + 312, F(18)); txt(g, 'pictures with captions from the web.', X, Y + 338, F(18));
        txt(g, 'It can be confidently wrong: check it.', X, Y + 380, F(18, 'bold'), COL.hot);
      }
    });
    const inM = boardMesh(inB.tex, 2.4, 2.4); inM.position.set(-2.55, 2.05, 0); root.add(inM);
    const outM = boardMesh(outB.tex, 2.4, 2.4); outM.position.set(2.75, 2.05, 0); root.add(outM);

    // ---- the model in the middle
    const mid = new THREE.Group(); mid.position.set(0.1, 2.05, 0); root.add(mid);
    const cnn = new THREE.Group(); mid.add(cnn);
    const slabs = [[1.1, 1.1, 0.07], [0.9, 0.9, 0.14], [0.7, 0.7, 0.22], [0.5, 0.5, 0.3], [0.3, 0.3, 0.36]].map(([w, h, d], i) => {
      const m = box(d, h, w, M.plastic(0x2a3346, { emissive: 0x8ef0ff, emissiveIntensity: 0.05 })); m.position.x = -0.62 + i * 0.3; cnn.add(m); return m;
    });
    const rules = new THREE.Group(); mid.add(rules);
    const rb = box(1.1, 1.1, 0.5, M.plastic(0x2a2f3a)); rules.add(rb);
    const rulesT = canvasTexture(256, 256, (g, w, h) => { g.clearRect(0, 0, w, h); txt(g, '1:1:3:1:1', w / 2, 110, F(40, 'bold'), COL.hot, 'center'); txt(g, 'geometry +', w / 2, 160, F(28), COL.white, 'center'); txt(g, 'error-fixing maths', w / 2, 196, F(24), COL.white, 'center'); });
    const rf = new THREE.Mesh(new THREE.PlaneGeometry(1.0, 1.0), new THREE.MeshBasicMaterial({ map: rulesT.tex, transparent: true, toneMapped: false })); rf.position.z = 0.26; rules.add(rf);
    const vit = new THREE.Group(); mid.add(vit);
    const tb = new THREE.Mesh(new THREE.BoxGeometry(0.5, 1.5, 0.5), M.clear(0xb8f2e6, 0.25)); vit.add(tb);
    const tokens = []; for (let i = 0; i < 25; i++) { const c = canvasTexture(20, 20, () => {}); c.tex.magFilter = THREE.NearestFilter; const m = new THREE.Mesh(new THREE.PlaneGeometry(0.16, 0.16), new THREE.MeshBasicMaterial({ map: c.tex, toneMapped: false, side: THREE.DoubleSide })); vit.add(m); tokens.push({ m, c }); }
    const lMid = stage.label('', [0.1, 0.95, 0], root, 'hot');
    const beams = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-1.32, 2.05, 0), new THREE.Vector3(-0.7, 2.05, 0), new THREE.Vector3(0.85, 2.05, 0), new THREE.Vector3(1.52, 2.05, 0)]), new THREE.LineBasicMaterial({ color: 0xffd166, toneMapped: false }));
    root.add(beams);

    return {
      update(dt, s) {
        dt = Math.max(0, dt); t += dt;
        if (!mainReady()) labStep(inReel() ? 1000 : 10); else labStep(3);
        fitNarrow(stage, [], -0.12);
        const A = APPS[s.app];
        cnn.visible = A.model === 'cnn'; rules.visible = A.model === 'rules'; vit.visible = A.model === 'vit';
        lMid.element.textContent = A.model === 'cnn' ? 'A trained CNN' : A.model === 'rules' ? 'Classic vision: no learning' : 'Vision transformer';
        slabs.forEach((m, i) => { m.material.emissiveIntensity = 0.05 + 0.4 * Math.max(0, Math.sin(t * 4 - i * 0.8)); });
        if (s.app === 'photos') {
          const k = `${s.gal}|${mainReady()}|${lab.run}`;
          if (k !== galKey) {
            galKey = k; const r = rng(2024 + s.gal * 17); gal = []; gal.truth = [];
            for (let i = 0; i < 16; i++) { const c = 1 + (i * 7 + s.gal) % 4; gal.push(makeSample(c, r, 'wild')); gal.truth.push(c); }
            galP = mainReady() ? gal.map((im) => lab.net.forward(im).slice()) : null;
          }
        }
        if (s.app === 'qr') {
          const k = s.qrAng.toFixed(3);
          if (k !== qrKey) { qrKey = k; qr = qrImage(qrSymbol(21), 84, s.qrAng, 5); found = findFinders(qr); }
        }
        if (s.app === 'next') {
          const k = `${s.patch}`;
          if (k !== vitKey) {
            vitKey = k; vitIm = makeSample(2, rng(606), 'train', { ang: 0.5, s: 7.4, dx: 0, dy: 0 }); vitPatches = patches(vitIm, s.patch);
            tokens.forEach((tk, i) => { tk.m.visible = i < vitPatches.length; if (i < vitPatches.length) { drawImage(tk.c.canvas.getContext('2d'), vitPatches[i].im, 0, 0, 20, 20); tk.c.tex.needsUpdate = true; } });
          }
          const n = Math.min(25, vitPatches.length);
          tokens.forEach((tk, i) => { if (i >= n) return; const a = (i / n) * Math.PI * 2 + t * 0.6, y = -0.65 + (i / Math.max(1, n - 1)) * 1.3; tk.m.position.set(Math.cos(a) * 0.55, y, Math.sin(a) * 0.55); tk.m.lookAt(stage.camera.position); });
        }
        const live = s.app === 'qr' || s.app === 'roads' || s.app === 'factory' || s.app === 'crops';
        const key = `${s.app}|${s.query}|${galKey}|${qrKey}|${vitKey}|${mainReady()}`;
        if (key !== this._k || (live && Math.floor(t * 15) !== this._q) || (!mainReady() && Math.floor(t * 3) !== this._q2)) { this._k = key; this._q = Math.floor(t * 15); this._q2 = Math.floor(t * 3); inB.redraw(s); outB.redraw(s); }
      },
      readout(s) {
        const A = APPS[s.app];
        if (s.app === 'photos') {
          if (!mainReady() || !galP) return `<div class="big">Training… ${Math.round(progress() * 100)}%</div><small>${trainingNote()}</small>`;
          let hits = 0, right = 0, total = 0; galP.forEach((p, i) => { if (argmax(p) === s.query) { hits++; if (gal.truth[i] === s.query) right++; } if (gal.truth[i] === s.query) total++; });
          return `<div class="big">${hits} found for "${CLASSES[s.query]}"</div><div class="row"><span>Really ${CLASSES[s.query]}s</span><b>${right} of ${hits}</b></div><div class="row"><span>${CLASSES[s.query][0].toUpperCase() + CLASSES[s.query].slice(1)}s in the gallery</span><b>${total}</b></div><small>Live: the network from chapter 3 on 16 new wild pictures.</small>`;
        }
        if (s.app === 'qr' && found) return `<div class="big">${found.centres.length} of 3 corner squares found</div><div class="row"><span>Rows with a 1:1:3:1:1 stripe</span><b>${found.rowsHit.length}</b></div><div class="row"><span>Learned weights used</span><b>0</b></div><small>QR-style pattern with random data, not a real payment code.</small>`;
        if (s.app === 'next') return `<div class="big">${vitPatches.length} patches of ${s.patch} × ${s.patch}</div><div class="row"><span>Numbers per patch</span><b>${s.patch * s.patch * 3}</b></div><div class="row"><span>Original ViT</span><b>224×224 → 196 patches of 16×16</b></div><small>Each patch becomes a token, like a word in LLMClear.</small>`;
        return `<div class="big">${A.name}</div><div class="row"><span>In</span><b>${A.in}</b></div><div class="row"><span>Out</span><b>${A.out}</b></div><small>${A.note}</small>`;
      },
    };
  },
};
