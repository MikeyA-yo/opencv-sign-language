/**
 * <hand-signs> — drop-in webcam interpreter for a small set of signs.
 *
 *   <script type="module" src="/static/handsigns-widget.js"></script>
 *   <hand-signs speak></hand-signs>
 *
 * Events: `signchange` with detail { sign, label, meaning, confidence }.
 * Camera and MediaPipe run in the browser. Frames never leave the device.
 */
const MP_VERSION = "0.10.21";
const MP_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const CDN_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const LOCAL_MODEL_URL = "/models/hand_landmarker.task";

const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const SIGNS = {
  yes: { label: "YES", meaning: "Yes" },
  no: { label: "NO", meaning: "No" },
  okay: { label: "OKAY", meaning: "Okay / all good" },
  hello: { label: "HELLO", meaning: "Hello / open hand" },
  peace: { label: "PEACE", meaning: "Peace / two" },
  point: { label: "POINT", meaning: "Pointing / one" },
  fist: { label: "FIST", meaning: "Closed hand" },
  ily: { label: "I LOVE YOU", meaning: "I love you (ASL)" },
  call: { label: "CALL ME", meaning: "Call me / phone" },
  later: { label: "LATER", meaning: "Later / letter L" },
  three: { label: "THREE", meaning: "Three" },
  four: { label: "FOUR", meaning: "Four" },
  rock: { label: "ROCK", meaning: "Rock on" },
  promise: { label: "PROMISE", meaning: "Promise / pinky swear" },
  none: { label: "—", meaning: "No recognized sign" },
};

const PINCH_OK = 0.38;
const EXT_ON = 0.42;
const EXT_OFF = 0.38;
const THUMB_EXT_ON = 0.48;
const MIN_CONF = 0.55;

const CSS = `
:host {
  display: block;
  font-family: "Segoe UI", system-ui, sans-serif;
  color: #1a1714;
  --hs-yes: #1f7a4d;
  --hs-no: #b42318;
  --hs-ok: #0b6e99;
  --hs-ink: #1a1714;
  --hs-paper: #f6f1e8;
  --hs-line: #d9d0c3;
}
* { box-sizing: border-box; }
.wrap {
  background: var(--hs-paper);
  border: 1px solid var(--hs-line);
  border-radius: 18px;
  overflow: hidden;
}
.stage {
  position: relative;
  background: #111;
  aspect-ratio: 4 / 3;
}
video, canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1);
}
video { opacity: 0.92; }
.hud {
  position: absolute;
  left: 12px;
  right: 12px;
  bottom: 12px;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
  pointer-events: none;
}
.word {
  font-family: Palatino, "Palatino Linotype", "Iowan Old Style", Georgia, serif;
  font-size: clamp(2.2rem, 8vw, 4.4rem);
  line-height: 0.9;
  letter-spacing: -0.03em;
  color: #fff;
  text-shadow: 0 2px 18px rgba(0,0,0,.55);
}
.word.yes { color: #7dffa8; }
.word.no { color: #ff8d85; }
.word.okay { color: #8fe3ff; }
.word.call { color: #9dffb0; }
.word.later { color: #ffe08a; }
.word.three, .word.four { color: #d7c4ff; }
.word.rock { color: #ffb4e0; }
.word.promise { color: #b8e0ff; }
.meta {
  color: #f0e6d6;
  font-size: 0.92rem;
  text-align: right;
}
.bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 16px;
  border-top: 1px solid var(--hs-line);
}
button {
  font: inherit;
  cursor: pointer;
  border: 1px solid #1a1714;
  background: #1a1714;
  color: #f6f1e8;
  border-radius: 999px;
  padding: 0.45rem 1rem;
}
button.ghost {
  background: transparent;
  color: #1a1714;
}
button:focus-visible {
  outline: 3px solid #0b6e99;
  outline-offset: 2px;
}
button:disabled {
  opacity: 0.55;
  cursor: wait;
}
.status { font-size: 0.9rem; color: #5c564d; }
.gate {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  background: #161310;
  color: #f6f1e8;
  padding: 24px;
  text-align: center;
}
.gate h2 {
  font-family: Palatino, Georgia, serif;
  font-weight: 500;
  margin: 0 0 8px;
}
.gate p { margin: 0 0 12px; color: #d9d0c3; max-width: 38ch; }
.gate-status { color: #c4b8a8; font-size: 0.9rem; min-height: 1.35em; margin: 0 0 14px; }
.hidden { display: none !important; }
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; }
}
`;

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}
function clamp(v, lo = 0, hi = 1) {
  return v < lo ? lo : v > hi ? hi : v;
}
function palmSize(lm) {
  return Math.max(dist(lm[5], lm[17]), dist(lm[0], lm[9]), 1e-6);
}
function fingerExt(lm, tip, pip, mcp) {
  const palm = palmSize(lm);
  const wrist = lm[0];
  const along = (dist(lm[tip], wrist) - dist(lm[pip], wrist)) / (palm * 0.55);
  const reach = (dist(lm[tip], lm[mcp]) - dist(lm[pip], lm[mcp])) / (palm * 0.45);
  const unfold = (dist(lm[tip], wrist) - dist(lm[mcp], wrist)) / (palm * 0.7);
  return clamp(0.45 * along + 0.35 * reach + 0.2 * unfold);
}
function thumbExt(lm) {
  const palm = palmSize(lm);
  const span = dist(lm[4], lm[2]) / (palm * 0.85);
  const away = dist(lm[4], lm[5]) / (palm * 0.9);
  const ip = dist(lm[4], lm[3]) / (palm * 0.4);
  return clamp(0.5 * span + 0.3 * away + 0.2 * ip);
}
function thumbDir(lm) {
  const dx = lm[4].x - lm[2].x;
  const dy = lm[4].y - lm[2].y;
  const len = Math.hypot(dx, dy) || 1e-6;
  const uy = dy / len;
  const vertical = Math.abs(uy);
  return [clamp(-uy * 1.15) * vertical, clamp(uy * 1.15) * vertical];
}
function pinch(lm) {
  return clamp(1 - dist(lm[4], lm[8]) / palmSize(lm) / PINCH_OK);
}
function on(s) {
  return clamp((s - EXT_ON) / (1 - EXT_ON));
}
function off(s) {
  return clamp((EXT_OFF - s) / EXT_OFF);
}
function noneResult(confidence = 0) {
  return { sign: "none", label: "—", meaning: SIGNS.none.meaning, confidence };
}
function classify(lm, minConf = MIN_CONF) {
  const f = {
    thumb: thumbExt(lm),
    index: fingerExt(lm, 8, 6, 5),
    middle: fingerExt(lm, 12, 10, 9),
    ring: fingerExt(lm, 16, 14, 13),
    pinky: fingerExt(lm, 20, 18, 17),
  };
  const p = pinch(lm);
  const [up, down] = thumbDir(lm);
  const thumbOn = clamp((f.thumb - THUMB_EXT_ON) / (1 - THUMB_EXT_ON));
  const idxOn = on(f.index), idxOff = off(f.index);
  const midOn = on(f.middle), midOff = off(f.middle);
  const ringOn = on(f.ring), ringOff = off(f.ring);
  const pinkyOn = on(f.pinky), pinkyOff = off(f.pinky);
  const othersOff = (idxOff + midOff + ringOff + pinkyOff) / 4;
  const threeUp = (midOn + ringOn + pinkyOn) / 3;
  const cands = [];

  const okay = 0.55 * p + 0.45 * threeUp;
  if (p > 0.45 && threeUp > 0.35) cands.push(["okay", okay]);

  const ily = (thumbOn + idxOn + pinkyOn + midOff + ringOff) / 5;
  if (thumbOn > 0.35 && idxOn > 0.4 && pinkyOn > 0.4 && midOff > 0.35 && ringOff > 0.35) {
    cands.push(["ily", ily]);
  }

  const call = (thumbOn + pinkyOn + idxOff + midOff + ringOff) / 5;
  if (thumbOn > 0.35 && pinkyOn > 0.4 && idxOff > 0.35 && midOff > 0.35 && ringOff > 0.35) {
    cands.push(["call", call]);
  }

  if (idxOn > 0.4 && pinkyOn > 0.4 && f.thumb < THUMB_EXT_ON && midOff > 0.35 && ringOff > 0.35) {
    cands.push(["rock", (idxOn + pinkyOn + (1 - f.thumb) + midOff + ringOff) / 5]);
  }

  const later = (thumbOn + idxOn + midOff + ringOff + pinkyOff) / 5;
  if (thumbOn > 0.35 && idxOn > 0.45 && midOff > 0.4 && ringOff > 0.4 && pinkyOff > 0.4 && p < 0.4) {
    cands.push(["later", later]);
  }

  if (pinkyOn > 0.5 && idxOff > 0.4 && midOff > 0.4 && ringOff > 0.4 && f.thumb < THUMB_EXT_ON) {
    cands.push(["promise", (pinkyOn + idxOff + midOff + ringOff + (1 - f.thumb)) / 5]);
  }

  const three = (idxOn + midOn + ringOn + pinkyOff) / 4;
  if (idxOn > 0.45 && midOn > 0.45 && ringOn > 0.4 && pinkyOff > 0.35 && p < 0.4) {
    cands.push(["three", three]);
  }

  const peace = (idxOn + midOn + ringOff + pinkyOff) / 4;
  if (idxOn > 0.45 && midOn > 0.45 && ringOff > 0.35 && pinkyOff > 0.35 && p < 0.4) {
    cands.push(["peace", peace]);
  }

  if (idxOn > 0.5 && midOff > 0.4 && ringOff > 0.4 && pinkyOff > 0.4 && f.thumb < THUMB_EXT_ON && p < 0.4) {
    cands.push(["point", (idxOn + midOff + ringOff + pinkyOff + (1 - f.thumb)) / 5]);
  }

  const four = (idxOn + midOn + ringOn + pinkyOn) / 4;
  if (four > 0.55 && f.thumb < THUMB_EXT_ON && p < 0.35) {
    cands.push(["four", 0.7 * four + 0.3 * (1 - f.thumb)]);
  }

  const yes = 0.4 * thumbOn + 0.35 * othersOff + 0.25 * up;
  if (thumbOn > 0.35 && othersOff > 0.45 && up > 0.45 && p < 0.5) cands.push(["yes", yes]);

  const no = 0.4 * thumbOn + 0.35 * othersOff + 0.25 * down;
  if (thumbOn > 0.35 && othersOff > 0.45 && down > 0.45 && p < 0.5) cands.push(["no", no]);

  const hello = (idxOn + midOn + ringOn + pinkyOn) / 4;
  if (hello > 0.55 && f.thumb >= THUMB_EXT_ON && p < 0.35) {
    cands.push(["hello", 0.7 * hello + 0.3 * Math.max(thumbOn, 0.4)]);
  }

  if (idxOff > 0.4 && midOff > 0.4 && ringOff > 0.4 && pinkyOff > 0.4 && f.thumb < THUMB_EXT_ON) {
    cands.push(["fist", 0.65 * othersOff + 0.35 * (1 - f.thumb)]);
  }

  if (!cands.length) return noneResult(0);
  cands.sort((a, b) => b[1] - a[1]);
  const [sign, confidence] = cands[0];
  if (confidence < minConf) return noneResult(confidence);
  const info = SIGNS[sign];
  return { sign, label: info.label, meaning: info.meaning, confidence };
}

class SignSmoother {
  constructor(window = 6, minHold = 3) {
    this.window = window;
    this.minHold = minHold;
    this.buf = [];
    this.stable = noneResult();
  }
  update(result) {
    this.buf.push(result);
    if (this.buf.length > this.window) this.buf.shift();
    const votes = {};
    for (const item of this.buf) {
      (votes[item.sign] ||= []).push(item);
    }
    let best = "none";
    let count = 0;
    for (const [sign, items] of Object.entries(votes)) {
      if (sign === "none") continue;
      if (items.length > count) {
        best = sign;
        count = items.length;
      }
    }
    if (count >= this.minHold) {
      const chosen = votes[best].reduce((a, b) => (a.confidence > b.confidence ? a : b));
      this.stable = { ...chosen };
      return this.stable;
    }
    if (this.buf.length >= this.window && count === 0) this.stable = noneResult();
    return this.stable;
  }
}

async function fetchModel(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onProgress?.(1);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total) onProgress?.(Math.min(received / total, 0.99));
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  onProgress?.(1);
  return out;
}

const loadListeners = new Set();
let landmarkerPromise = null;
let landmarkerInstance = null;

function emitLoadStatus(message) {
  for (const fn of loadListeners) fn(message);
}

function loadLandmarker() {
  if (landmarkerInstance) return Promise.resolve(landmarkerInstance);
  if (landmarkerPromise) return landmarkerPromise;
  landmarkerPromise = (async () => {
    emitLoadStatus("Downloading interpreter (first visit only)…");
    const visionModPromise = import(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/vision_bundle.mjs`
    );
    const onPct = (pct) => {
      emitLoadStatus(`Downloading hand model… ${Math.round(pct * 100)}%`);
    };
    const modelPromise = fetchModel(LOCAL_MODEL_URL, onPct).catch(() =>
      fetchModel(CDN_MODEL_URL, onPct),
    );
    const { FilesetResolver, HandLandmarker } = await visionModPromise;
    const [buffer, vision] = await Promise.all([
      modelPromise,
      FilesetResolver.forVisionTasks(MP_WASM),
    ]);
    emitLoadStatus("Starting hand tracker…");
    const options = {
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5,
    };
    try {
      landmarkerInstance = await HandLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: { modelAssetBuffer: buffer.slice(), delegate: "GPU" },
      });
    } catch {
      landmarkerInstance = await HandLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: { modelAssetBuffer: buffer, delegate: "CPU" },
      });
    }
    emitLoadStatus("Ready — click Start camera");
    return landmarkerInstance;
  })().catch((err) => {
    landmarkerPromise = null;
    throw err;
  });
  return landmarkerPromise;
}

class HandSignsElement extends HTMLElement {
  constructor() {
    super();
    this._smoother = new SignSmoother();
    this._lastSpoken = "";
    this._running = false;
    this._raf = 0;
    this._lastVideoTime = -1;
    this._startGen = 0;
    this.attachShadow({ mode: "open" });
    this.shadowRoot.innerHTML = `
      <style>${CSS}</style>
      <div class="wrap">
        <div class="stage">
          <video playsinline muted></video>
          <canvas></canvas>
          <div class="hud">
            <div class="word" aria-live="polite">waiting</div>
            <div class="meta"></div>
          </div>
          <div class="gate">
            <div>
              <h2>Start the interpreter</h2>
              <p>This uses your camera on this device only. Hold YES (thumbs up), NO (thumbs down), or OKAY still in frame.</p>
              <p class="gate-status"></p>
              <button type="button" class="start">Start camera</button>
            </div>
          </div>
        </div>
        <div class="bar">
          <span class="status">Camera off</span>
          <div>
            <button type="button" class="ghost speak-btn">Voice off</button>
            <button type="button" class="ghost stop hidden">Stop</button>
          </div>
        </div>
      </div>
    `;
    this._video = this.shadowRoot.querySelector("video");
    this._canvas = this.shadowRoot.querySelector("canvas");
    this._ctx = this._canvas.getContext("2d");
    this._word = this.shadowRoot.querySelector(".word");
    this._meta = this.shadowRoot.querySelector(".meta");
    this._status = this.shadowRoot.querySelector(".status");
    this._gate = this.shadowRoot.querySelector(".gate");
    this._gateStatus = this.shadowRoot.querySelector(".gate-status");
    this._startBtn = this.shadowRoot.querySelector(".start");
    this._stopBtn = this.shadowRoot.querySelector(".stop");
    this._speakBtn = this.shadowRoot.querySelector(".speak-btn");
    this._speak = this.hasAttribute("speak");
    this._syncSpeakBtn();
    this._onLoadStatus = (msg) => {
      if (!this._running) {
        this._status.textContent = msg;
        this._gateStatus.textContent = msg;
      }
    };
  }

  connectedCallback() {
    this._startBtn.addEventListener("click", () => this.start());
    this._stopBtn.addEventListener("click", () => this.stop());
    this._speakBtn.addEventListener("click", () => {
      this._speak = !this._speak;
      this._syncSpeakBtn();
    });
    loadListeners.add(this._onLoadStatus);
    loadLandmarker().catch((err) => {
      this._status.textContent = "Could not load the hand model.";
      this._gateStatus.textContent = "Model download failed. Check the network and refresh.";
      console.error(err);
    });
  }

  disconnectedCallback() {
    loadListeners.delete(this._onLoadStatus);
    this.stop();
  }

  _syncSpeakBtn() {
    this._speakBtn.textContent = this._speak ? "Voice on" : "Voice off";
  }

  async start() {
    if (this._running) return;
    this._running = true;
    const gen = ++this._startGen;
    this._status.textContent = "Asking for camera…";
    this._gateStatus.textContent = "Allow the camera if the browser asks.";
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (gen !== this._startGen) {
        for (const track of stream.getTracks()) track.stop();
        return;
      }
      this._video.srcObject = stream;
      await this._video.play();
    } catch (err) {
      this._running = false;
      this._status.textContent = "Camera permission was denied.";
      this._gateStatus.textContent = "Camera permission was denied.";
      console.error(err);
      return;
    }
    this._gate.classList.add("hidden");
    this._stopBtn.classList.remove("hidden");
    this._status.textContent = landmarkerInstance
      ? "Hold a sign still"
      : "Camera on — finishing model…";
    try {
      this._landmarker = await loadLandmarker();
    } catch (err) {
      console.error(err);
      this.stop();
      this._status.textContent = "Could not load the hand model.";
      return;
    }
    if (gen !== this._startGen) return;
    this._status.textContent = "Hold a sign still";
    this._loop();
  }

  stop() {
    this._startGen += 1;
    this._running = false;
    cancelAnimationFrame(this._raf);
    const stream = this._video.srcObject;
    if (stream) {
      for (const track of stream.getTracks()) track.stop();
    }
    this._video.srcObject = null;
    this._gate.classList.remove("hidden");
    this._stopBtn.classList.add("hidden");
    this._status.textContent = "Camera off";
    this._word.textContent = "waiting";
    this._word.className = "word";
    this._meta.textContent = "";
    this._smoother = new SignSmoother();
  }

  _loop() {
    if (!this._running) return;
    const now = this._video.currentTime;
    if (this._video.readyState >= 2 && now !== this._lastVideoTime) {
      this._lastVideoTime = now;
      const result = this._landmarker.detectForVideo(this._video, performance.now());
      this._draw(result);
      const hands = result.landmarks || [];
      let raw = noneResult();
      for (const lm of hands) {
        const classified = classify(lm);
        if (classified.sign !== "none" && classified.confidence >= raw.confidence) {
          raw = classified;
        }
      }
      const stable = this._smoother.update(raw);
      this._renderSign(stable);
    }
    this._raf = requestAnimationFrame(() => this._loop());
  }

  _draw(result) {
    const w = this._video.videoWidth;
    const h = this._video.videoHeight;
    if (!w || !h) return;
    this._canvas.width = w;
    this._canvas.height = h;
    const ctx = this._ctx;
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(80, 200, 255, 0.9)";
    ctx.fillStyle = "#fff";
    for (const lm of result.landmarks || []) {
      for (const [a, b] of CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(lm[a].x * w, lm[a].y * h);
        ctx.lineTo(lm[b].x * w, lm[b].y * h);
        ctx.stroke();
      }
      for (const p of lm) {
        ctx.beginPath();
        ctx.arc(p.x * w, p.y * h, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  _renderSign(result) {
    this._word.textContent = result.sign === "none" ? "waiting" : result.label;
    this._word.className = `word ${result.sign === "none" ? "" : result.sign}`;
    this._meta.textContent =
      result.sign === "none" ? "" : `${result.meaning} · ${Math.round(result.confidence * 100)}%`;
    if (result.sign !== "none") {
      this._status.textContent = result.meaning;
    }
    if (result.sign !== this._emitted) {
      this._emitted = result.sign;
      this.dispatchEvent(new CustomEvent("signchange", { detail: result, bubbles: true }));
      if (this._speak && result.sign !== "none" && result.sign !== this._lastSpoken) {
        this._lastSpoken = result.sign;
        this._say(result.meaning);
      }
      if (result.sign === "none") this._lastSpoken = "";
    }
  }

  _say(text) {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1;
    window.speechSynthesis.speak(u);
  }
}

if (!customElements.get("hand-signs")) {
  customElements.define("hand-signs", HandSignsElement);
}

export function mount(target, options = {}) {
  const el = typeof target === "string" ? document.querySelector(target) : target;
  const node = document.createElement("hand-signs");
  if (options.speak) node.setAttribute("speak", "");
  node.addEventListener("signchange", (ev) => options.onSign?.(ev.detail));
  el.appendChild(node);
  return node;
}

window.HandSigns = { mount, classify };
