import test from "node:test";
import assert from "node:assert/strict";
import { clearSlot, compactSlots, compactBoundSlots } from "../../js/media_slot_state.mjs";

test("clearing one endpoint leaves the other endpoint unchanged", () => {
  assert.deepEqual(clearSlot(["first.png", "last.png"], 0), ["", "last.png"]);
});

test("removing a middle REF image compacts later pictures", () => {
  assert.deepEqual(compactSlots(["a.png", "b.png", "c.png", ""], 1), ["a.png", "c.png", "", ""]);
});

test("audio filename and role name move as one unit", () => {
  assert.deepEqual(
    compactBoundSlots(["a.wav", "b.wav", "c.wav"], ["S1", "S2", "S3"], 1),
    { files: ["a.wav", "c.wav", ""], names: ["S1", "S3", ""] },
  );
});
