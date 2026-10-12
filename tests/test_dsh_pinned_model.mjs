import test from 'node:test';
import assert from 'node:assert/strict';
import { pinnedModelSelection, verifyModelRoutes } from '../tools/dsh_headless_continuing_runner.mjs';

test('campaign route uses explicit Flash selection and rejects missing configuration', () => {
  const expected = { provider: 'deepseek-official', model: 'deepseek-v4-flash' };
  assert.deepEqual(pinnedModelSelection(expected), expected);
  assert.throws(() => pinnedModelSelection({}), /explicit provider and model/);
  assert.throws(() => pinnedModelSelection({ provider: 'test' }), /explicit provider and model/);
});

test('actual session routes must match the pinned model for every request', () => {
  const flash = { provider: 'deepseek-official', model: 'deepseek-v4-flash' };
  const event = config => ({ type: 'request/header', data: { header: { config } } });
  assert.deepEqual(verifyModelRoutes([event(flash), event(flash)], flash), [flash, flash]);
  assert.throws(() => verifyModelRoutes([], flash), /routing mismatch/);
  assert.throws(() => verifyModelRoutes([event(flash), event({ provider: 'test', model: 'Qwen3.8-27B' })], flash), /routing mismatch/);
});
