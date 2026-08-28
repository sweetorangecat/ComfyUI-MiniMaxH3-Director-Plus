function normalizeSlots(values) {
  if (!Array.isArray(values)) throw new TypeError("slots must be an array");
  return values.map((value) => value == null ? "" : String(value));
}

function validateIndex(index, length) {
  if (!Number.isInteger(index) || index < 0 || index >= length) {
    throw new RangeError("slot index out of range");
  }
}

export function clearSlot(values, index) {
  const result = normalizeSlots(values);
  validateIndex(index, result.length);
  result[index] = "";
  return result;
}

export function compactSlots(values, index) {
  const result = normalizeSlots(values);
  validateIndex(index, result.length);
  result.splice(index, 1);
  result.push("");
  return result;
}

export function compactBoundSlots(files, names, index) {
  const normalizedFiles = normalizeSlots(files);
  const normalizedNames = normalizeSlots(names);
  if (normalizedFiles.length !== normalizedNames.length) {
    throw new RangeError("bound slot arrays must have equal length");
  }
  validateIndex(index, normalizedFiles.length);
  normalizedFiles.splice(index, 1);
  normalizedFiles.push("");
  normalizedNames.splice(index, 1);
  normalizedNames.push("");
  return { files: normalizedFiles, names: normalizedNames };
}
