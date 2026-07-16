const scratch = new ArrayBuffer(4);
const scratchFloat = new Float32Array(scratch);
const scratchUint = new Uint32Array(scratch);

export function float32ToFloat16(value) {
  scratchFloat[0] = value;
  const bits = scratchUint[0];
  const sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;

  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0x1000) >>> 13);
  }
  if (exponent >= 31) {
    return sign | (mantissa === 0 ? 0x7c00 : 0x7e00);
  }
  if (mantissa & 0x1000) {
    mantissa += 0x2000;
    if (mantissa & 0x800000) {
      mantissa = 0;
      exponent++;
      if (exponent >= 31) return sign | 0x7c00;
    }
  }
  return sign | (exponent << 10) | (mantissa >>> 13);
}

export function float16ToFloat32(value) {
  const sign = (value & 0x8000) << 16;
  let exponent = (value >>> 10) & 0x1f;
  let mantissa = value & 0x03ff;
  let bits;

  if (exponent === 0) {
    if (mantissa === 0) {
      bits = sign;
    } else {
      exponent = 1;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        exponent--;
      }
      mantissa &= 0x03ff;
      bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
  } else if (exponent === 31) {
    bits = sign | 0x7f800000 | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }

  scratchUint[0] = bits;
  return scratchFloat[0];
}

export function toFloat16Array(values) {
  const output = new Uint16Array(values.length);
  for (let index = 0; index < values.length; index++) {
    output[index] = float32ToFloat16(values[index]);
  }
  return output;
}

export function fromFloat16Array(values) {
  const output = new Float32Array(values.length);
  for (let index = 0; index < values.length; index++) {
    output[index] = float16ToFloat32(values[index]);
  }
  return output;
}
