// Turns video frames into the exact tensor format the model was trained on.
//
// Training used torchvision: transforms.Resize((112, 112)) on a PIL image,
// then transforms.ToTensor(). PIL's bilinear resize is antialiased: when it
// shrinks an image it averages over a window that widens with the scale
// factor. A browser canvas resize does not do this the same way, so the
// resampling below reproduces Pillow's algorithm (ImagingResample with the
// bilinear/triangle filter, horizontal pass first, 8-bit rounding between
// passes). No normalization, because training used none.

export const INPUT_SIZE = 112;

function triangle(x) {
  x = Math.abs(x);
  return x < 1 ? 1 - x : 0;
}

// Pillow's precompute_coeffs for one axis.
export function resampleCoeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 1.0 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const weights = new Float32Array(outSize * ksize);

  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;

    let total = 0;
    for (let x = 0; x < xmax; x++) {
      const w = triangle((x + xmin - center + 0.5) * ss);
      weights[xx * ksize + x] = w;
      total += w;
    }
    if (total !== 0) {
      for (let x = 0; x < xmax; x++) weights[xx * ksize + x] /= total;
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { bounds, weights, ksize };
}

const clip8 = v => (v <= 0 ? 0 : v >= 255 ? 255 : Math.floor(v + 0.5));

// rgba: Uint8ClampedArray from getImageData, width*height*4.
// Writes 3*out*out floats in CHW order, scaled to [0, 1], into dst at offset.
export function rgbaToTensor(rgba, width, height, dst, offset, plan) {
  const out = INPUT_SIZE;
  const h = plan.h, v = plan.v;

  // Horizontal pass: height x out x 3, rounded to 8 bits like Pillow.
  const mid = new Uint8ClampedArray(height * out * 3);
  for (let y = 0; y < height; y++) {
    const row = y * width * 4;
    for (let xx = 0; xx < out; xx++) {
      const xmin = h.bounds[xx * 2], n = h.bounds[xx * 2 + 1];
      const k = xx * h.ksize;
      let r = 0, g = 0, b = 0;
      for (let x = 0; x < n; x++) {
        const w = h.weights[k + x];
        const p = row + (xmin + x) * 4;
        r += rgba[p] * w;
        g += rgba[p + 1] * w;
        b += rgba[p + 2] * w;
      }
      const m = (y * out + xx) * 3;
      mid[m] = clip8(r);
      mid[m + 1] = clip8(g);
      mid[m + 2] = clip8(b);
    }
  }

  // Vertical pass, then scale to [0, 1] and lay out as channels-first.
  const plane = out * out;
  for (let yy = 0; yy < out; yy++) {
    const ymin = v.bounds[yy * 2], n = v.bounds[yy * 2 + 1];
    const k = yy * v.ksize;
    for (let xx = 0; xx < out; xx++) {
      let r = 0, g = 0, b = 0;
      for (let y = 0; y < n; y++) {
        const w = v.weights[k + y];
        const m = ((ymin + y) * out + xx) * 3;
        r += mid[m] * w;
        g += mid[m + 1] * w;
        b += mid[m + 2] * w;
      }
      const i = offset + yy * out + xx;
      dst[i] = clip8(r) / 255;
      dst[i + plane] = clip8(g) / 255;
      dst[i + 2 * plane] = clip8(b) / 255;
    }
  }
}

export function makePlan(width, height) {
  return {
    h: resampleCoeffs(width, INPUT_SIZE),
    v: resampleCoeffs(height, INPUT_SIZE),
  };
}
