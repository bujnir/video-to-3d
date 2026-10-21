/* Minimal WebGL2 Gaussian splat viewer for the .splat format (32 bytes per splat).
   Camera convention follows COLMAP/OpenCV: x right, y down, z forward.
   SplatViewer(canvas, {splatUrl, view, onStatus}) */
(function () {
  const VS = `#version 300 es
precision highp float; precision highp int;
uniform highp sampler2D uA; uniform highp sampler2D uB; uniform highp sampler2D uC; uniform mediump sampler2D uCol;
uniform mat4 view; uniform mat4 proj; uniform vec2 focal; uniform vec2 viewport;
in vec2 position; in uint index;
out vec4 vColor; out vec2 vPos;
void main () {
  ivec2 uv = ivec2(int(index & 2047u), int(index >> 11));
  vec3 c = texelFetch(uA, uv, 0).xyz;
  vec4 cam = view * vec4(c, 1.0);
  vec4 p2 = proj * cam;
  float clip = 1.2 * p2.w;
  if (cam.z < 0.05 || p2.x < -clip || p2.x > clip || p2.y < -clip || p2.y > clip) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec4 b = texelFetch(uB, uv, 0); vec4 d = texelFetch(uC, uv, 0);
  mat3 Vrk = mat3(b.x, b.y, b.z, b.y, b.w, d.x, b.z, d.x, d.y);
  mat3 J = mat3(focal.x / cam.z, 0.0, -(focal.x * cam.x) / (cam.z * cam.z),
                0.0, -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z),
                0.0, 0.0, 0.0);
  mat3 T = transpose(mat3(view)) * J;
  mat3 cov2d = transpose(T) * Vrk * T;
  cov2d[0][0] += 0.3; cov2d[1][1] += 0.3;
  float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
  float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
  float l1 = mid + radius, l2 = mid - radius;
  if (l2 < 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec2 diag = normalize(vec2(cov2d[0][1], l1 - cov2d[0][0]));
  vec2 major = min(sqrt(2.0 * l1), 1024.0) * diag;
  vec2 minor = min(sqrt(2.0 * l2), 1024.0) * vec2(diag.y, -diag.x);
  vColor = texelFetch(uCol, uv, 0);
  vPos = position;
  vec2 ctr = p2.xy / p2.w;
  gl_Position = vec4(ctr + position.x * major / viewport + position.y * minor / viewport, 0.0, 1.0);
}`;
  const FS = `#version 300 es
precision highp float;
in vec4 vColor; in vec2 vPos; out vec4 fragColor;
void main () {
  float A = -dot(vPos, vPos);
  if (A < -4.0) discard;
  float B = exp(A) * vColor.a;
  fragColor = vec4(B * vColor.rgb, B);
}`;

  // depth sort in a worker: 16-bit counting sort, nearest first (front-to-back compositing)
  const WORKER = `
let pos = null, n = 0;
onmessage = (e) => {
  if (e.data.pos) { pos = e.data.pos; n = pos.length / 3; return; }
  const m = e.data.view; if (!pos) return;
  const depth = new Int32Array(n); let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < n; i++) {
    const z = m[2] * pos[3*i] + m[6] * pos[3*i+1] + m[10] * pos[3*i+2] + m[14];
    const d = (z * 4096) | 0; depth[i] = d; if (d < mn) mn = d; if (d > mx) mx = d;
  }
  const inv = (256 * 256 - 1) / Math.max(1, mx - mn);
  const counts = new Uint32Array(256 * 256);
  for (let i = 0; i < n; i++) { depth[i] = ((depth[i] - mn) * inv) | 0; counts[depth[i]]++; }
  const starts = new Uint32Array(256 * 256);
  for (let i = 1; i < 65536; i++) starts[i] = starts[i - 1] + counts[i - 1];
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[starts[depth[i]]++] = i;
  postMessage({ idx }, [idx.buffer]);
};`;

  function compile(gl, type, src) {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function tex(gl, unit, ifmt, fmt, type, w, h, data) {
    const t = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data);
    return t;
  }

  // small vector helpers
  const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
  const add = (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
  const mul = (a, s) => [a[0]*s, a[1]*s, a[2]*s];
  const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
  const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0]/l, a[1]/l, a[2]/l]; };

  // world->camera (OpenCV) as column-major 4x4 from camera rows x,y,z and position p
  function viewFrom(x, y, z, p) {
    return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, p), -dot(y, p), -dot(z, p), 1];
  }
  function lookAt(pos, target, up) {
    const z = norm(sub(target, pos));
    const x = norm(cross(mul(up, -1), z));
    const y = cross(z, x);
    return viewFrom(x, y, z, pos);
  }
  function viewFromColmap(R, t) { // R row-major 3x3 world->cam
    return [R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, t[0], t[1], t[2], 1];
  }

  window.SplatViewer = function (canvas, opts) {
    const status = opts.onStatus || (() => {});
    const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: true });
    if (!gl) { status('This browser cannot show the 3D splat (WebGL2 is off).'); return null; }
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS)); gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);
    const U = (n) => gl.getUniformLocation(prog, n);
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    const quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(prog, 'position'); gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    const idxBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf);
    const aIdx = gl.getAttribLocation(prog, 'index'); gl.enableVertexAttribArray(aIdx);
    gl.vertexAttribIPointer(aIdx, 1, gl.UNSIGNED_INT, false, 0, 0); gl.vertexAttribDivisor(aIdx, 1);

    const V = opts.view; const fxRef = V.fx / V.w;
    const up = V.up, center = V.center;
    let n = 0, ready = false, sortedCount = 0;
    let state = { yaw: 0, pitch: 0.35, dist: Math.max(2.5, V.radius * 0.9), target: V.cam_center };
    let view = null, dirty = true, mode = 'photo', flyT = 0, photoView = null;
    // build an orbit frame from the up vector
    const ref = Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    const e1 = norm(cross(up, ref)), e2 = cross(up, e1);
    function orbitView() {
      const cp = Math.cos(state.pitch), dir = add(add(mul(e1, Math.cos(state.yaw) * cp), mul(e2, Math.sin(state.yaw) * cp)), mul(up, Math.sin(state.pitch)));
      const pos = add(state.target, mul(dir, state.dist));
      return lookAt(pos, state.target, up);
    }
    function camView(t) {
      const cams = V.cams, i = Math.floor(t) % cams.length, j = (i + 1) % cams.length, f = t - Math.floor(t);
      const a = cams[i], b = cams[j];
      const Ra = a.R, Rb = b.R;
      const Ca = mul([Ra[0]*a.t[0]+Ra[3]*a.t[1]+Ra[6]*a.t[2], Ra[1]*a.t[0]+Ra[4]*a.t[1]+Ra[7]*a.t[2], Ra[2]*a.t[0]+Ra[5]*a.t[1]+Ra[8]*a.t[2]], -1);
      const Cb = mul([Rb[0]*b.t[0]+Rb[3]*b.t[1]+Rb[6]*b.t[2], Rb[1]*b.t[0]+Rb[4]*b.t[1]+Rb[7]*b.t[2], Rb[2]*b.t[0]+Rb[5]*b.t[1]+Rb[8]*b.t[2]], -1);
      const za = [Ra[6], Ra[7], Ra[8]], zb = [Rb[6], Rb[7], Rb[8]];
      const ya = [Ra[3], Ra[4], Ra[5]], yb = [Rb[3], Rb[4], Rb[5]];
      const p = add(mul(Ca, 1 - f), mul(Cb, f));
      const z = norm(add(mul(za, 1 - f), mul(zb, f)));
      const yRaw = add(mul(ya, 1 - f), mul(yb, f));
      const x = norm(cross(yRaw, z)), y = cross(z, x);
      return viewFrom(x, y, z, p);
    }

    const worker = new Worker(URL.createObjectURL(new Blob([WORKER], { type: 'text/javascript' })));
    let sorting = false, pendingView = null;
    worker.onmessage = (e) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, idxBuf); gl.bufferData(gl.ARRAY_BUFFER, e.data.idx, gl.DYNAMIC_DRAW);
      sortedCount = e.data.idx.length; sorting = false; dirty = true;
      if (pendingView) { const v = pendingView; pendingView = null; requestSort(v); }
    };
    function requestSort(v) { if (sorting) { pendingView = v; return; } sorting = true; worker.postMessage({ view: Float32Array.from(v) }); }

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      const w = Math.max(1, Math.round(canvas.clientWidth * dpr)), h = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; dirty = true; }
    }

    function draw() {
      resize();
      const W = canvas.width, H = canvas.height;
      const f = fxRef * W;                      // same horizontal field of view as the training photos
      const znear = 0.05, zfar = 200;
      const proj = [2 * f / W, 0, 0, 0, 0, -2 * f / H, 0, 0, 0, 0, zfar / (zfar - znear), 1, 0, 0, -(zfar * znear) / (zfar - znear), 0];
      gl.viewport(0, 0, W, H);
      gl.uniformMatrix4fv(U('proj'), false, proj);
      gl.uniformMatrix4fv(U('view'), false, view);
      gl.uniform2fv(U('focal'), [f, f]);
      gl.uniform2fv(U('viewport'), [W, H]);
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      if (sortedCount) gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, sortedCount);
    }

    let lastSortView = null;
    function camPos(v) { // camera centre from a world->camera matrix
      return [-(v[0]*v[12] + v[1]*v[13] + v[2]*v[14]), -(v[4]*v[12] + v[5]*v[13] + v[6]*v[14]), -(v[8]*v[12] + v[9]*v[13] + v[10]*v[14])];
    }
    function toOrbit() { // keep the current camera position, switch to orbiting the room centre
      if (mode === 'orbit' || !view) { mode = 'orbit'; return; }
      const d = sub(camPos(view), state.target), dist = Math.hypot(d[0], d[1], d[2]) || 1, n = mul(d, 1 / dist);
      state.dist = dist; state.pitch = Math.asin(Math.max(-1, Math.min(1, dot(n, up)))); state.yaw = Math.atan2(dot(n, e2), dot(n, e1));
      mode = 'orbit'; dirty = true;
    }
    function frame() {
      if (ready) {
        if (mode === 'fly') { flyT += 0.012; view = camView(flyT); dirty = true; }
        else if (mode === 'photo') { if (!photoView) photoView = camView(0); view = photoView; }
        else if (dirty || !view) view = orbitView();
        if (!lastSortView || view.some((v, i) => Math.abs(v - lastSortView[i]) > 0.01)) { lastSortView = view.slice(); requestSort(view); }
        if (dirty) { draw(); dirty = mode === 'fly'; }
      }
      requestAnimationFrame(frame);
    }

    // input: drag to orbit, wheel / pinch to zoom
    let drag = null; const pts = new Map();
    canvas.addEventListener('pointerdown', (e) => { if (mode === 'fly') return; toOrbit(); canvas.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); drag = [e.clientX, e.clientY]; });
    canvas.addEventListener('pointermove', (e) => {
      if (!pts.has(e.pointerId) || mode !== 'orbit') return;
      const prev = pts.get(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]);
      if (pts.size === 2) {
        const [p1, p2] = [...pts.values()];
        const d = Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
        if (drag && drag.pinch) state.dist = Math.min(30, Math.max(0.5, state.dist * drag.pinch / d));
        drag = { pinch: d };
      } else {
        state.yaw -= (e.clientX - prev[0]) * 0.006;
        state.pitch = Math.min(1.45, Math.max(-1.2, state.pitch + (e.clientY - prev[1]) * 0.006));
      }
      dirty = true;
    });
    const end = (e) => { pts.delete(e.pointerId); drag = null; };
    canvas.addEventListener('pointerup', end); canvas.addEventListener('pointercancel', end);
    canvas.addEventListener('wheel', (e) => { e.preventDefault(); if (mode === 'fly') return; toOrbit(); state.dist = Math.min(30, Math.max(0.5, state.dist * Math.exp(e.deltaY * 0.001))); dirty = true; }, { passive: false });

    status('Loading the 3D model…');
    const urls = opts.parts || [opts.splatUrl];
    let loaded = 0;
    Promise.all(urls.map((u) => fetch(u).then((r) => { if (!r.ok) throw new Error(r.status); return r.arrayBuffer(); })
      .then((b) => { loaded++; if (loaded < urls.length) status(`Loading the 3D model… ${loaded} of ${urls.length}`); return b; })))
    .then((bufs) => {
      // decode every part into one list of splats
      const parts = bufs.map((buf) => {
        if (!opts.parts) return { kind: 'splat', buf, n: Math.floor(buf.byteLength / 32) };
        const dv = new DataView(buf), m = dv.getUint32(0, true);
        const lo = [dv.getFloat32(8, true), dv.getFloat32(12, true), dv.getFloat32(16, true)];
        const hi = [dv.getFloat32(20, true), dv.getFloat32(24, true), dv.getFloat32(28, true)];
        const smin = dv.getFloat32(32, true), smax = dv.getFloat32(36, true);
        let o = 48;
        const P = new Uint16Array(buf, o, m * 3); o += m * 6;
        const S = new Uint8Array(buf, o, m * 3); o += m * 3;
        const R = new Uint8Array(buf, o, m * 4); o += m * 4;
        const Cc = new Uint8Array(buf, o, m * 4);
        return { kind: 'q', n: m, lo, hi, smin, smax, P, S, R, Cc };
      });
      n = parts.reduce((a, p) => a + p.n, 0);
      const TW = 2048, TH = Math.ceil(n / TW);
      const A = new Float32Array(TW * TH * 4), B = new Float32Array(TW * TH * 4), C = new Float32Array(TW * TH * 4), col = new Uint8Array(TW * TH * 4);
      const pos = new Float32Array(n * 3);
      let i = 0;
      const put = (px, py, pz, sx, sy, sz, w, x, y, z, r, g, b, a) => {
        pos[3*i] = px; pos[3*i+1] = py; pos[3*i+2] = pz;
        A[4*i] = px; A[4*i+1] = py; A[4*i+2] = pz;
        const ql = Math.hypot(w, x, y, z) || 1; w /= ql; x /= ql; y /= ql; z /= ql;
        const R0 = 1 - 2*(y*y + z*z), R1 = 2*(x*y - w*z), R2 = 2*(x*z + w*y),
              R3 = 2*(x*y + w*z), R4 = 1 - 2*(x*x + z*z), R5 = 2*(y*z - w*x),
              R6 = 2*(x*z - w*y), R7 = 2*(y*z + w*x), R8 = 1 - 2*(x*x + y*y);
        const M0 = R0*sx, M1 = R1*sy, M2 = R2*sz, M3 = R3*sx, M4 = R4*sy, M5 = R5*sz, M6 = R6*sx, M7 = R7*sy, M8 = R8*sz;
        B[4*i] = M0*M0 + M1*M1 + M2*M2; B[4*i+1] = M0*M3 + M1*M4 + M2*M5; B[4*i+2] = M0*M6 + M1*M7 + M2*M8;
        B[4*i+3] = M3*M3 + M4*M4 + M5*M5; C[4*i] = M3*M6 + M4*M7 + M5*M8; C[4*i+1] = M6*M6 + M7*M7 + M8*M8;
        col[4*i] = r; col[4*i+1] = g; col[4*i+2] = b; col[4*i+3] = a;
        i++;
      };
      for (const p of parts) {
        if (p.kind === 'splat') {
          const f32 = new Float32Array(p.buf), u8 = new Uint8Array(p.buf);
          for (let k = 0; k < p.n; k++) put(f32[8*k], f32[8*k+1], f32[8*k+2], f32[8*k+3], f32[8*k+4], f32[8*k+5],
            (u8[32*k+28]-128)/128, (u8[32*k+29]-128)/128, (u8[32*k+30]-128)/128, (u8[32*k+31]-128)/128,
            u8[32*k+24], u8[32*k+25], u8[32*k+26], u8[32*k+27]);
        } else {
          const ex = [(p.hi[0]-p.lo[0])/65535, (p.hi[1]-p.lo[1])/65535, (p.hi[2]-p.lo[2])/65535], ss = (p.smax - p.smin) / 255;
          for (let k = 0; k < p.n; k++) put(p.lo[0] + p.P[3*k]*ex[0], p.lo[1] + p.P[3*k+1]*ex[1], p.lo[2] + p.P[3*k+2]*ex[2],
            Math.exp(p.smin + p.S[3*k]*ss), Math.exp(p.smin + p.S[3*k+1]*ss), Math.exp(p.smin + p.S[3*k+2]*ss),
            (p.R[4*k]-128)/128, (p.R[4*k+1]-128)/128, (p.R[4*k+2]-128)/128, (p.R[4*k+3]-128)/128,
            p.Cc[4*k], p.Cc[4*k+1], p.Cc[4*k+2], p.Cc[4*k+3]);
        }
      }
      tex(gl, 0, gl.RGBA32F, gl.RGBA, gl.FLOAT, TW, TH, A); gl.uniform1i(U('uA'), 0);
      tex(gl, 1, gl.RGBA32F, gl.RGBA, gl.FLOAT, TW, TH, B); gl.uniform1i(U('uB'), 1);
      tex(gl, 2, gl.RGBA32F, gl.RGBA, gl.FLOAT, TW, TH, C); gl.uniform1i(U('uC'), 2);
      tex(gl, 3, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, TW, TH, col); gl.uniform1i(U('uCol'), 3);
      worker.postMessage({ pos }, [pos.buffer]);
      ready = true; dirty = true; status('');
    }).catch(() => status('Could not load the 3D model.'));
    new ResizeObserver(() => { dirty = true; }).observe(canvas);
    requestAnimationFrame(frame);

    return {
      count: () => n,
      fly(on) { mode = on ? 'fly' : 'photo'; if (!on) photoView = view; dirty = true; },
      reset() { mode = 'photo'; photoView = camView(0); dirty = true; },
      photo(k) { mode = 'photo'; photoView = camView(k); dirty = true; },
      cameras: () => V.cams.length,
    };
  };
})();
