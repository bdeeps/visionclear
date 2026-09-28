// Chapter 1: pixels are numbers. A 3D mango sits on a table under a lamp. A tiny ray-marching
// renderer (vision.js renderMango) takes its picture: for every pixel it shoots rays from the camera,
// finds the mango, table or wall, and works out the light (Lambert shading, a shadow ray, a little
// shine), then turns linear light into 0–255 sRGB numbers (IEC 61966-2-1). The 3D model on the
// left and the picture on the right are the same scene: turn the mango, move the camera or change
// the lamp and every number changes, although it is still the same mango.
import { THREE, M, box, sphere, canvasTexture, clamp } from '../kit.js';
import { renderMango, mangoAlbedo, MANGO, LIGHTS, to255, px } from '../vision.js';
import { labStep } from '../lab.js';
import { panel, txt, F, COL, boardMesh, fitNarrow, inReel, PixelWall, backPlate, frameLines } from '../visview.js';

const RES = { 8: [8, 6], 16: [16, 12], 32: [32, 24], 48: [48, 36] };
const REF = { turn: 0.5, dist: 3.4, light: 1, tint: 'day' };
const WALL_W = 3.3, WALL_X = 0.75, WALL_Y = 2.15, SCN = [-2.75, 0.05, 0.1], SCN_ROT = 0.95, SCN_K = 0.78;
const CHAN = ['Red', 'Green', 'Blue'];

export default {
  id: 'pixels',
  short: 'Pixels are numbers',
  title: 'To a computer, a picture is just numbers',
  subtitle: 'A camera turns a mango into a grid of red, green and blue values. That grid is all the computer gets.',
  view: { pos: [0.5, 3.1, 9.8], target: [0.45, 2.1, 0] },
  learn: `<p>You look at this mango and just <b>see</b> a mango. A computer never does. A camera's sensor (CameraClear shows how) measures light in a grid of tiny squares called <b>pixels</b>, and each pixel becomes three numbers: how much <b>red</b>, <b>green</b> and <b>blue</b> light arrived, each from 0 to 255.</p>
    <p>So this 32 × 24 picture is <b>2,304 numbers</b> in a row. A phone photo of 12 megapixels is about <b>36 million</b>. Somewhere in those numbers is a mango, but no single number says "mango".</p>
    <p>The picture on the right is made live by a tiny <b>renderer</b> in this page: for every pixel it shoots rays from the camera into the 3D scene on the left, finds what they hit, and works out the light. Click any pixel to read its numbers.</p>
    <p>Now the hard part. <b>Turn</b> the mango, <b>move</b> the camera or <b>dim</b> the lamp, and most of the numbers change. A different angle, size or light gives a completely different grid, for the same mango. Recognising things means finding what stays the same while all the numbers move. Your eyes and brain do this without effort (EyeClear and BrainClear show how). Computers had to learn it.</p>
    <p class="tip"><b>Try it:</b> click a pixel on the mango, then drag the lamp brightness down. Watch its numbers fall and the "numbers changed" readout climb. Try a sunset light: the mango turns orange in the numbers, even though the mango didn't change.</p>`,
  terms: [
    { t: 'Pixel', d: 'One tiny square of a picture, with a single colour.' },
    { t: 'RGB', d: 'Red, green and blue: three numbers per pixel, each 0 to 255 in most pictures.' },
    { t: 'Resolution', d: 'How many pixels across and down. More pixels, more detail, more numbers.' },
    { t: 'Channel', d: 'One of the three layers of a colour picture: all the reds, all the greens or all the blues.' },
    { t: 'Rendering', d: 'Working out a picture of a 3D scene, pixel by pixel, from light and shapes.' },
    { t: 'Invariance', d: 'Giving the same answer when things that don\'t matter (angle, size, light) change.' },
  ],
  defaults: { res: 32, chan: -1, turn: 0.5, dist: 3.4, light: 1, tint: 'day' },
  controls: [
    { key: 'res', type: 'seg', label: 'Resolution', options: Object.entries(RES).map(([v, [w, h]]) => ({ v: +v, label: `${w}×${h}` })), fmt: (v) => `${(RES[v][0] * RES[v][1] * 3).toLocaleString('en')} numbers` },
    { key: 'chan', type: 'seg', label: 'Show', options: [{ v: -1, label: 'Colour' }, { v: 0, label: 'Red' }, { v: 1, label: 'Green' }, { v: 2, label: 'Blue' }], hint: 'Block height shows brightness (or the one channel you pick).' },
    { key: 'turn', type: 'range', label: 'Turn the mango', min: 0, max: Math.PI * 2, step: 0.01, fmt: (v) => `${Math.round((v * 180) / Math.PI)}°` },
    { key: 'dist', type: 'range', label: 'Camera distance', min: 2.3, max: 7, step: 0.01, fmt: (v) => `${v.toFixed(1)} (mango ${Math.round(100 * 3.4 / v)}% size)` },
    { key: 'light', type: 'range', label: 'Lamp brightness', min: 0.15, max: 1.6, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%` },
    { key: 'tint', type: 'seg', label: 'Light colour', options: Object.entries(LIGHTS).map(([v, l]) => ({ v, label: l.name })) },
    { key: 'go', type: 'buttons', label: 'View', items: [{ label: 'Back to the first view', act: (s) => Object.assign(s, REF) }] },
  ],
  quiz: [
    { q: 'What does a computer actually receive from a colour camera?', options: ['The names of the objects', 'A grid of numbers: red, green and blue for every pixel', 'A description of the shapes', 'Only the outline'], answer: 1, why: 'The sensor measures light in each pixel. The picture is just those numbers.' },
    { q: 'How many numbers are in a 32 × 24 colour picture?', options: ['56', '768', '2,304', '32,000'], answer: 2, why: '32 × 24 = 768 pixels, and each has 3 numbers: 768 × 3 = 2,304.' },
    { q: 'Why is recognising a mango hard for a computer?', options: ['Mangoes are rare', 'Turning it, moving it or changing the light changes most of the numbers', 'Cameras cannot see yellow', 'Computers are slow at adding'], answer: 1, why: 'The same mango can give completely different grids. The computer must find what stays the same.' },
  ],
  reel: [
    { ms: 5200, caption: 'To a computer, a mango is not a mango. It is a grid of numbers.', set: { res: 32, chan: -1, turn: 0.5, dist: 3.4, light: 1, tint: 'day' }, anim: { turn: [0.2, 1.1] }, spin: 0, view: { pos: [0.5, 3.2, 9.6], target: [0.45, 2.25, 0] } },
    { ms: 5200, caption: 'Each pixel is three numbers: red, green and blue, from 0 to 255.', set: { res: 16, chan: -1, turn: 0.6, light: 1, tint: 'day' }, anim: { light: [1, 0.35] }, spin: 0, view: { pos: [2.6, 2.3, 6.2], target: [2.2, 2.05, 0] } },
  ],

  build({ stage }) {
    const root = new THREE.Group(); stage.root.add(root);
    // ---- the 3D scene: table, backdrop, mango, lamp and camera, in the renderer's own coordinates
    const scene = new THREE.Group(); scene.position.set(...SCN); scene.rotation.y = SCN_ROT; scene.scale.setScalar(SCN_K); root.add(scene);
    const table = box(3.4, 0.08, 3.2, M.matte(0x6b4a2e)); table.position.set(0, -0.04, -0.1); scene.add(table);
    const back = box(3.4, 2.2, 0.06, M.matte(0x6f7277)); back.position.set(0, 1.1, -1.63); scene.add(back);
    // mango: a sphere pushed into the same bent ellipsoid the renderer uses, coloured by the same albedo
    const mg = new THREE.SphereGeometry(1, 64, 40), P = mg.attributes.position, cols = new Float32Array(P.count * 3);
    for (let i = 0; i < P.count; i++) {
      const x = P.getX(i), y = P.getY(i), z = P.getZ(i), X = MANGO.rx * x + MANGO.bend * (1 - y * y), Y = MANGO.ry * y + MANGO.cy, Z = MANGO.rz * z;
      P.setXYZ(i, X, Y, Z); const a = mangoAlbedo(X, Y, Z, 0); cols.set(a.map((v) => Math.min(1, v * 0.85)), i * 3);
    }
    mg.setAttribute('color', new THREE.BufferAttribute(cols, 3)); mg.computeVertexNormals();
    const mango = new THREE.Group(), mangoMesh = new THREE.Mesh(mg, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.45 })); mangoMesh.castShadow = true;
    const stalk = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.035, 0.14, 10), M.matte(0x2a1c0c)); stalk.position.set(0.02, MANGO.cy + MANGO.ry + 0.02, 0);
    mango.add(mangoMesh, stalk); scene.add(mango);
    const Ld = new THREE.Vector3(-0.45, 0.8, 0.55).normalize();
    const lamp = sphere(0.13, M.glow(0xfff4d8)); lamp.position.copy(Ld.clone().multiplyScalar(2.1)).add(new THREE.Vector3(0, MANGO.cy, 0)); scene.add(lamp);
    const halo = new THREE.Mesh(new THREE.SphereGeometry(0.3, 24, 12), M.ghost(0xfff0c0, 0.25)); lamp.add(halo);
    const cam = new THREE.Group(); scene.add(cam);
    const body = box(0.42, 0.3, 0.3, M.plastic(0x2a2f3a)); const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.12, 0.2, 24), M.metal(0x9aa3b0)); lens.rotation.x = Math.PI / 2; lens.position.z = -0.22;
    cam.add(body, lens);
    const fr = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffd166, transparent: true, opacity: 0.55, toneMapped: false })); scene.add(fr);
    const lScene = stage.label('The 3D scene', [SCN[0] + 0.3, -0.25, SCN[2] + 0.9], root);
    const lCam = stage.label('Camera', [0, 0.42, 0], cam);

    // ---- the picture: a wall of pixel blocks, rebuilt when the resolution changes
    const wallG = new THREE.Group(); wallG.position.set(WALL_X, WALL_Y, 0); root.add(wallG);
    let wall = null, plate = null, res = 0, cell = 0;
    const sel = frameLines(1, 1, 0xffd166); sel.position.z = 0.32; wallG.add(sel);
    const lWall = stage.label('', [WALL_X, WALL_Y - 1.55, 0], root, 'hot');
    // ---- the numbers board: a 5×5 patch around the chosen pixel
    let pick = null, im = null, mask = null, ref = null, changed = 0, stamp = '', lastRender = -1, want = '';
    const numbers = canvasTexture(520, 600, (g, w, h, s) => {
      panel(g, w, h);
      if (!im || !s) return;
      const [x0, y0] = pick; txt(g, `Pixel (${x0 + 1}, ${y0 + 1}) and its neighbours`, 20, 36, F(22, 'bold'));
      const N = 5, S = 94, X0 = (w - N * S) / 2, Y0 = 56;
      for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
        const x = x0 + i - 2, y = y0 + j - 2, X = X0 + i * S, Y = Y0 + j * S;
        if (x < 0 || y < 0 || x >= im.w || y >= im.h) { g.fillStyle = 'rgba(255,255,255,.04)'; g.fillRect(X + 2, Y + 2, S - 4, S - 4); continue; }
        const [r, gg, b] = px(im, x, y).map(to255);
        g.fillStyle = s.chan < 0 ? `rgb(${r},${gg},${b})` : `rgb(${s.chan === 0 ? r : 12},${s.chan === 1 ? gg : 12},${s.chan === 2 ? b : 12})`; g.fillRect(X + 2, Y + 2, S - 4, S - 4);
        g.fillStyle = 'rgba(0,0,0,.55)'; g.fillRect(X + 6, Y + 14, S - 12, S - 28);
        [r, gg, b].forEach((v, c) => txt(g, String(v), X + S / 2, Y + 34 + c * 22, F(19, s.chan === c || (s.chan < 0 && i === 2 && j === 2) ? 'bold' : ''), s.chan >= 0 && s.chan !== c ? COL.dim : [COL.r, COL.g, COL.b][c], 'center'));
        if (i === 2 && j === 2) { g.strokeStyle = COL.hot; g.lineWidth = 4; g.strokeRect(X + 1, Y + 1, S - 2, S - 2); }
      }
      txt(g, 'Each square: red, green, blue (0–255)', w / 2, h - 18, F(18), COL.soft, 'center');
    });
    const nb = boardMesh(numbers.tex, 2.0, 2.31); nb.position.set(3.72, WALL_Y, 0.05); root.add(nb);
    const linkGeo = new THREE.BufferGeometry(); linkGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3));
    const link = new THREE.LineSegments(linkGeo, new THREE.LineBasicMaterial({ color: 0xffd166, transparent: true, opacity: 0.5, toneMapped: false })); root.add(link);

    const rebuild = (r) => {
      if (wall) { wallG.remove(wall.group); wall.mesh.geometry.dispose(); wall.mesh.material.dispose(); wallG.remove(plate); }
      const [w, h] = RES[r]; res = r; cell = WALL_W / w;
      wall = new PixelWall(wallG, w, h, cell, { depth: 0.3 }); plate = backPlate(WALL_W + 0.12, cell * h + 0.12); plate.position.z = -0.03; wallG.add(plate);
      sel.scale.set(cell * 1.08, cell * 1.08, 1); pick = null; ref = null;
      lWall.position.set(WALL_X, WALL_Y - (cell * h) / 2 - 0.22, 0);
    };
    const render = (s) => {
      const [w, h] = RES[s.res];
      const out = renderMango({ w, h, turn: s.turn, dist: s.dist, light: s.light, tint: s.tint, aa: w > 32 ? 1 : 2 });
      im = out.im; mask = out.mask;
      if (!ref) ref = renderMango({ w, h, ...REF, aa: w > 32 ? 1 : 2 }).im;
      let ch = 0; for (let i = 0; i < im.d.length; i++) if (Math.abs(im.d[i] - ref.d[i]) > 0.1) ch++; changed = ch / im.d.length;
      if (!pick) { let sx = 0, sy = 0, n = 0; for (let i = 0; i < w * h; i++) if (mask[i]) { sx += i % w; sy += Math.floor(i / w); n++; } pick = n ? [Math.round(sx / n), Math.round(sy / n)] : [w >> 1, h >> 1]; }
      pick = [clamp(pick[0], 0, w - 1), clamp(pick[1], 0, h - 1)];
    };

    // ---- click a pixel to read it
    const ray = new THREE.Raycaster(), v2 = new THREE.Vector2(), el = stage.renderer.domElement; let down = null;
    const pd = (e) => { down = [e.clientX, e.clientY]; };
    const pu = (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5 || !wall) return;
      const b = el.getBoundingClientRect(); v2.set(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1);
      ray.setFromCamera(v2, stage.camera); const hit = ray.intersectObject(wall.mesh)[0];
      if (hit && hit.instanceId !== undefined) { pick = [hit.instanceId % wall.w, Math.floor(hit.instanceId / wall.w)]; stamp = ''; }
    };
    el.addEventListener('pointerdown', pd); el.addEventListener('pointerup', pu);

    let t = 0;
    return {
      update(dt, s) {
        dt = Math.max(0, dt); t += dt;
        if (!inReel()) labStep(5);                      // the recogniser starts training quietly in the background
        const narrow = fitNarrow(stage, [lScene, lCam], -0.12);
        if (s.res !== res) rebuild(s.res);
        // scene model follows the settings
        mango.rotation.y = -s.turn;
        const e = 0.22, cp = new THREE.Vector3(0, MANGO.cy - 0.05 + Math.sin(e) * s.dist, Math.cos(e) * s.dist);
        cam.position.copy(cp); cam.rotation.set(-e, 0, 0);
        const L = LIGHTS[s.tint].rgb; lamp.material.color.setRGB(L[0] * s.light, L[1] * s.light, L[2] * s.light); halo.material.opacity = 0.1 + 0.2 * s.light; halo.scale.setScalar(0.6 + 0.5 * s.light);
        // frustum: four lines from the camera to the corners of its view, 1.2 units past the mango
        const tf = Math.tan(0.31), dd = s.dist + 1.2, fwd = new THREE.Vector3(0, -Math.sin(e), -Math.cos(e)), upv = new THREE.Vector3(0, Math.cos(e), -Math.sin(e)), rt = new THREE.Vector3(1, 0, 0);
        const ar = RES[s.res][0] / RES[s.res][1], pts = [];
        for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) { const c = cp.clone().add(fwd.clone().multiplyScalar(dd)).add(rt.clone().multiplyScalar(a * tf * ar * dd)).add(upv.clone().multiplyScalar(b * tf * dd)); pts.push(cp.clone(), c); }
        fr.geometry.setFromPoints(pts);
        // re-render when something changed (at most about 12 times a second while dragging)
        const key = `${s.res}|${s.turn.toFixed(3)}|${s.dist.toFixed(3)}|${s.light.toFixed(3)}|${s.tint}`;
        if (key !== want) { want = key; }
        if (want !== stamp.split('#')[0] && (t - lastRender > 0.08 || inReel())) { render(s); lastRender = t; stamp = want + '#'; }
        if (im && wall) {
          const k2 = `${stamp}|${s.chan}|${pick}`;
          if (k2 !== this._k2) {
            this._k2 = k2; const n = im.w * im.h;
            wall.set(im, { chan: s.chan, hFn: s.chan >= 0 ? (i) => im.d[s.chan * n + i] : null });
            numbers.redraw(s);
            lWall.element.textContent = `${im.w} × ${im.h} pixels × 3 = ${(im.w * im.h * 3).toLocaleString('en')} numbers`;
          }
          const [X, Y] = wall.pos(pick[0], pick[1]); sel.position.set(X, Y, 0.34);
          const wp = new THREE.Vector3(X + WALL_X + cell / 2, Y + WALL_Y, 0.3), bl = [nb.position.x - 1.0, nb.position.y + 0.9, 0.05], bl2 = [nb.position.x - 1.0, nb.position.y - 0.9, 0.05];
          const lp = linkGeo.attributes.position; lp.setXYZ(0, wp.x, wp.y, wp.z); lp.setXYZ(1, ...bl); lp.setXYZ(2, wp.x, wp.y, wp.z); lp.setXYZ(3, ...bl2); lp.needsUpdate = true;
        }
      },
      readout(s) {
        if (!im) return '';
        const [r, g, b] = px(im, pick[0], pick[1]).map(to255), onM = mask[pick[1] * im.w + pick[0]];
        return `<div class="big">${im.w} × ${im.h} pixels</div>
          <div class="row"><span>Numbers in the picture</span><b>${(im.w * im.h * 3).toLocaleString('en')}</b></div>
          <div class="row"><span>Pixel (${pick[0] + 1}, ${pick[1] + 1})${onM ? ', on the mango' : ''}</span><b>R ${r} · G ${g} · B ${b}</b></div>
          <div class="row"><span>Numbers changed since the first view</span><b>${Math.round(changed * 100)}%</b></div>
          <small>Same mango, different numbers. Click a pixel to read it.</small>`;
      },
      dispose() { el.removeEventListener('pointerdown', pd); el.removeEventListener('pointerup', pu); },
    };
  },
};
