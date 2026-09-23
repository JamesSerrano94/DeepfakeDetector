import { INPUT_SIZE, makePlan, rgbaToTensor } from './preprocess.js';

const CFG = window.SITE_CONFIG || {};
const MODEL_URL = 'model/celeb_model.onnx';
const THRESHOLD = 0.5;          // training counted p > 0.5 as fake
const MIN_FRAMES = 8;
const MAX_FRAMES = 32;          // roughly one frame per second, like training
const MAX_WORK_WIDTH = 1920;    // shrink 4K frames first to keep things quick
const ZERO = { tp: 0, fp: 0, fn: 0, tn: 0 };

const $ = id => document.getElementById(id);

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

ort.env.wasm.wasmPaths = new URL('lib/ort/', location.href).href;
ort.env.wasm.numThreads = 1;

const sessionPromise = ort.InferenceSession.create(MODEL_URL, {
  executionProviders: ['wasm'],
});

sessionPromise
  .then(() => setLamp('ok', 'Model ready'))
  .catch(err => {
    console.error(err);
    setLamp('warn', 'Model failed to load');
    showError('The model file could not be loaded. Refresh the page to try again.');
  });

function setLamp(kind, text) {
  $('lamp').className = 'lamp ' + kind;
  $('lampText').textContent = text;
}

// ---------------------------------------------------------------------------
// Video to frames
// ---------------------------------------------------------------------------

function waitFor(el, okEvent, ms, what) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error(`Timed out ${what}.`));
    }, ms);
    const ok = () => { cleanup(); resolve(); };
    const bad = () => {
      cleanup();
      reject(new Error("Your browser can't play this file. Try an .mp4 (H.264) or .webm video."));
    };
    function cleanup() {
      clearTimeout(timer);
      el.removeEventListener(okEvent, ok);
      el.removeEventListener('error', bad);
    }
    el.addEventListener(okEvent, ok);
    el.addEventListener('error', bad);
  });
}

async function seek(video, t) {
  const done = waitFor(video, 'seeked', 10000, 'reading a frame');
  video.currentTime = t;
  await done;
  await new Promise(r => requestAnimationFrame(() => r()));
}

async function openVideo(file) {
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  const url = URL.createObjectURL(file);
  const loaded = waitFor(video, 'loadeddata', 20000, 'opening the video');
  video.src = url;
  await loaded;

  // Some recorded .webm files report an infinite length until you seek to the end.
  let duration = video.duration;
  if (!Number.isFinite(duration)) {
    await seek(video, 1e9);
    duration = video.duration;
  }
  if (!(duration > 0) || !video.videoWidth) {
    URL.revokeObjectURL(url);
    throw new Error("Couldn't read that video's length or size. Try another file.");
  }
  return { video, url, duration };
}

async function scoreVideo(file) {
  const session = await sessionPromise;
  const started = performance.now();
  const { video, url, duration } = await openVideo(file);

  try {
    const n = Math.min(MAX_FRAMES, Math.max(MIN_FRAMES, Math.ceil(duration)));
    const scale = Math.min(1, MAX_WORK_WIDTH / video.videoWidth);
    const w = Math.round(video.videoWidth * scale);
    const h = Math.round(video.videoHeight * scale);

    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const plan = makePlan(w, h);
    const per = 3 * INPUT_SIZE * INPUT_SIZE;
    const input = new Float32Array(n * per);

    for (let i = 0; i < n; i++) {
      // Spread samples evenly, from the middle of each slice of the clip.
      await seek(video, ((i + 0.5) * duration) / n);
      ctx.drawImage(video, 0, 0, w, h);
      const { data } = ctx.getImageData(0, 0, w, h);
      rgbaToTensor(data, w, h, input, i * per, plan);
    }

    const tensor = new ort.Tensor('float32', input, [n, 3, INPUT_SIZE, INPUT_SIZE]);
    const out = await session.run({ frames: tensor });
    const logits = out.logits.data;
    const frameProbs = Array.from(logits, z => 1 / (1 + Math.exp(-z)));
    const probability = frameProbs.reduce((a, b) => a + b, 0) / frameProbs.length;

    return {
      probability,
      verdict: probability > THRESHOLD ? 'synthetic' : 'authentic',
      frameProbs,
      frames: n,
      duration,
      seconds: (performance.now() - started) / 1000,
    };
  } finally {
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(url);
  }
}

// ---------------------------------------------------------------------------
// Where the counts live: a shared Firestore document, or this browser only.
// ---------------------------------------------------------------------------

function cellFor(verdict, actual) {
  if (verdict === 'synthetic') return actual === 'synthetic' ? 'tp' : 'fp';
  return actual === 'synthetic' ? 'fn' : 'tn';
}

class LocalStore {
  constructor(note) {
    this.key = 'deepfake-detector-matrix-v1';
    this.scope = note || 'counted on this device only';
  }
  read() {
    try {
      return { ...ZERO, ...JSON.parse(localStorage.getItem(this.key) || '{}') };
    } catch {
      return { ...ZERO };
    }
  }
  subscribe(onCounts) {
    this.onCounts = onCounts;
    this.counts = this.read();
    onCounts(this.counts);
  }
  async record(pred) {
    if (!pred.actual) return;
    const c = { ...this.counts };
    c[cellFor(pred.verdict, pred.actual)] += 1;
    this.counts = c;
    try { localStorage.setItem(this.key, JSON.stringify(c)); } catch { /* private mode */ }
    this.onCounts(c);
  }
}

class SharedStore {
  constructor() {
    this.scope = 'shared by everyone who visits';
  }
  async connect(firebaseConfig) {
    const { initializeApp } = await import('./lib/firebase/firebase-app.js');
    const fs = await import('./lib/firebase/firebase-firestore.js');
    this.fs = fs;
    this.db = fs.getFirestore(initializeApp(firebaseConfig));
    this.statsRef = fs.doc(this.db, 'stats', 'confusion');
  }
  subscribe(onCounts, onError) {
    this.fs.onSnapshot(
      this.statsRef,
      snap => onCounts({ ...ZERO, ...(snap.data() || {}) }),
      onError,
    );
  }
  async record(pred) {
    const { fs, db } = this;
    const batch = fs.writeBatch(db);
    if (pred.actual) {
      batch.set(this.statsRef, { [cellFor(pred.verdict, pred.actual)]: fs.increment(1) }, { merge: true });
    }
    batch.set(fs.doc(fs.collection(db, 'predictions')), {
      probability: pred.probability,
      verdict: pred.verdict,
      actual: pred.actual || null,
      frames: pred.frames,
      createdAt: fs.serverTimestamp(),
    });
    await batch.commit();
  }
}

let store = null;

async function startStore() {
  if (CFG.firebase && CFG.firebase.projectId) {
    try {
      const shared = new SharedStore();
      await shared.connect(CFG.firebase);
      store = shared;
      shared.subscribe(renderCounts, err => {
        console.error('Firestore:', err);
        fallBack('shared matrix unavailable, counting on this device only');
      });
      return;
    } catch (err) {
      console.error('Firebase setup failed:', err);
      fallBack('shared matrix unavailable, counting on this device only');
      return;
    }
  }
  fallBack();
}

function fallBack(note) {
  store = new LocalStore(note);
  store.subscribe(renderCounts);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

let shown = null;
const pct = v => (v == null ? '—' : (v * 100).toFixed(1) + '%');

function renderCounts(c) {
  const cells = { tp: 'cTP', fp: 'cFP', fn: 'cFN', tn: 'cTN' };
  for (const [k, id] of Object.entries(cells)) {
    const el = $(id);
    el.querySelector('b').textContent = c[k];
    if (shown && shown[k] !== c[k]) {
      el.classList.remove('bump');
      void el.offsetWidth;
      el.classList.add('bump');
    }
  }
  shown = { ...c };

  const n = c.tp + c.fp + c.fn + c.tn;
  const precision = c.tp + c.fp ? c.tp / (c.tp + c.fp) : null;
  const recall = c.tp + c.fn ? c.tp / (c.tp + c.fn) : null;
  const f1 = precision != null && recall != null && precision + recall > 0
    ? (2 * precision * recall) / (precision + recall)
    : null;

  $('acc').textContent = pct(n ? (c.tp + c.tn) / n : null);
  $('prec').textContent = pct(precision);
  $('rec').textContent = pct(recall);
  $('f1').textContent = f1 == null ? '—' : f1.toFixed(3);

  const count = n === 0 ? 'No labeled videos yet' : `${n} labeled video${n === 1 ? '' : 's'}`;
  $('scope').textContent = `${count}, ${store.scope}`;
}

function renderValidation() {
  const v = CFG.validation;
  if (!v) return;
  const total = v.tp + v.tn + v.fp + v.fn;
  if (!total) return;
  const acc = ((v.tp + v.tn) / total) * 100;
  const el = $('validation');
  el.innerHTML = '';
  el.append(
    'On its own validation split the model scored ',
    Object.assign(document.createElement('b'), { textContent: `${acc.toFixed(1)}% accuracy` }),
    v.auc != null ? ' and an ' : '',
    v.auc != null
      ? Object.assign(document.createElement('b'), { textContent: `AUC of ${Number(v.auc).toFixed(2)}` })
      : '',
    ` across ${total.toLocaleString()} video frames. `,
    v.caveat || '',
  );
  el.hidden = false;
}

function renderCredit() {
  if (!CFG.repoUrl && !CFG.author) return;
  const p = $('credit');
  p.textContent = CFG.author ? `Built by ${CFG.author}. ` : '';
  if (CFG.repoUrl) {
    const a = Object.assign(document.createElement('a'), {
      href: CFG.repoUrl,
      textContent: 'Code, notebook and write-up on GitHub',
    });
    p.append(a, '.');
  }
}

function showResult(r, actual) {
  const box = $('result');
  box.className = 'result ' + r.verdict;
  $('verdict').textContent = r.verdict === 'synthetic' ? 'Likely AI-generated' : 'Likely real footage';

  const chance = Math.round(r.probability * 100);
  let detail = `${chance}% chance it's AI-generated, averaged over ${r.frames} frames ` +
    `of a ${r.duration.toFixed(1)} s clip. Took ${r.seconds.toFixed(1)} s.`;
  if (Math.abs(r.probability - 0.5) < 0.15) detail += ' The model is close to undecided on this one.';
  $('detail').textContent = detail;

  const check = $('check');
  if (actual) {
    const right = r.verdict === actual;
    check.className = 'check ' + (right ? 'right' : 'wrong');
    check.innerHTML = '';
    check.append(
      `You said it was ${actual === 'synthetic' ? 'AI-generated' : 'real footage'}, so the model `,
      Object.assign(document.createElement('b'), { textContent: right ? 'got it right' : 'got it wrong' }),
      '.',
    );
    check.hidden = false;
  } else {
    check.className = 'check';
    check.textContent = 'Not added to the matrix, since you didn’t say what it was.';
    check.hidden = false;
  }
  box.hidden = false;
}

function showError(msg) {
  $('err').textContent = msg;
  $('err').hidden = false;
}

// ---------------------------------------------------------------------------
// Interaction
// ---------------------------------------------------------------------------

const targets = [$('tReal'), $('tFake')];
const unsure = $('tUnknown');
const fileInput = $('file');
let busy = false;
let pending = undefined; // 'authentic' | 'synthetic' | null (unlabeled)

function setBusy(on, label) {
  busy = on;
  for (const t of targets) {
    t.classList.toggle('busy', on && t.dataset.label === label);
    t.classList.toggle('muted', on && t.dataset.label !== label);
    t.setAttribute('aria-disabled', on ? 'true' : 'false');
  }
  unsure.disabled = on;
  unsure.textContent = on && label === null
    ? 'Checking the video…'
    : "Don't know what it is? Check a video without adding it to the matrix";
}

async function handle(file, label) {
  if (busy || !file) return;
  if (file.type && !file.type.startsWith('video/')) {
    showError("That doesn't look like a video file. Try an .mp4 or .webm.");
    return;
  }
  $('err').hidden = true;
  setBusy(true, label);

  try {
    const r = await scoreVideo(file);
    window.__lastResult = r;
    showResult(r, label);
    if (store) {
      store.record({ ...r, actual: label }).catch(err => {
        console.error('Saving failed:', err);
        showError("The result couldn't be saved to the shared matrix, so it wasn't counted.");
      });
    }
  } catch (err) {
    console.error(err);
    showError(err.message || 'Something went wrong reading that video.');
  } finally {
    setBusy(false);
  }
}

function choose(label) {
  if (busy) return;
  pending = label;
  fileInput.click();
}

for (const t of targets) {
  const label = t.dataset.label;
  t.addEventListener('click', () => choose(label));
  t.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(label); }
  });
  t.addEventListener('dragenter', e => { e.preventDefault(); if (!busy) t.classList.add('over'); });
  t.addEventListener('dragover', e => { e.preventDefault(); if (!busy) t.classList.add('over'); });
  t.addEventListener('dragleave', () => t.classList.remove('over'));
  t.addEventListener('drop', e => {
    e.preventDefault();
    t.classList.remove('over');
    handle(e.dataTransfer.files[0], label);
  });
}

unsure.addEventListener('click', () => choose(null));

fileInput.addEventListener('change', e => {
  const f = e.target.files[0];
  e.target.value = '';
  if (f && pending !== undefined) handle(f, pending);
  pending = undefined;
});

// Stop the browser from opening a video dropped outside the two boxes.
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());

renderValidation();
renderCredit();
startStore();
