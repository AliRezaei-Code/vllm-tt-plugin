// Measures the plugin so the atlas never carries a hand-transcribed number.
//
// Every figure the atlas states about the tree it describes — per-file line
// counts, the plugin total, the host test count, the test-module count, and
// whether the suite passed — is observed here at build time. Hand-maintaining
// them went wrong three times in one session: a stale total, a stale per-file
// table, and a stale test count, each in its own commit. A number that is
// measured cannot silently disagree with the code it describes.
//
// LOCAL DIVERGENCE: this file and the `measure()` call in build.mjs are not
// part of the upstream inkboard/system-atlas skill (MIT). They exist because
// this atlas describes a real repository, and a map of a moving codebase has to
// either measure or go stale.
import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const SRC = 'src/vllm_tt_plugin';
const TESTS = 'tests';

// Modules listed in the size table. The table is a reading aid, not an
// inventory: anything not named here is still counted in the total. Sorted
// largest-first at build time regardless of the order below.
const LISTED = [
  'model_runner.py',
  'platform.py',
  'input_batch.py',
  'scheduler.py',
  'worker.py',
  'async_decode.py',
  'lane_scheduler.py',
  'spec_accept.py',
  'spec_decode.py',
  'config.py',
  'utils/dp_discovery.py',
  'logprobs.py',
  'model_input.py',
  'structured_output.py',
  'loader.py',
];

function walk(dir, ext, out = []) {
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries) {
    const p = join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name === '__pycache__') continue;
      walk(p, ext, out);
    } else if (e.name.endsWith(ext)) {
      out.push(p);
    }
  }
  return out;
}

const lineCount = (p) => readFileSync(p, 'utf8').split('\n').length - 1;

const exists = (p) => {
  try {
    statSync(p);
    return true;
  } catch {
    return false;
  }
};

/**
 * Run the host suite and report what actually happened.
 *
 * The full run, not `--collect-only`. The atlas says whether the tests passed
 * and whether anything was skipped, xfailed or errored, so the build has to be
 * what observed those facts. Collecting alone would let the generated text
 * assert a result nobody checked — the exact failure this module exists to end,
 * and worse for appearing in generated text, which reads as self-maintaining
 * and therefore trustworthy.
 */
function runHostSuite(repoRoot) {
  const venv = join(repoRoot, '.venv/bin/python');
  if (!exists(venv)) return null;
  let out;
  try {
    out = execFileSync(
      venv,
      ['-m', 'pytest', TESTS, '--ignore=tests/tt', '-q'],
      {
        cwd: repoRoot,
        env: { ...process.env, PYTHONPATH: 'ci/host-stubs' },
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'pipe'],
      },
    );
  } catch (e) {
    // pytest exits non-zero when tests fail. The summary is still on stdout,
    // and a failing suite is a fact the atlas should report, not swallow.
    out = `${e.stdout || ''}\n${e.stderr || ''}`;
  }

  // The last non-empty line of a `-q` run is the summary, in any of the forms
  // pytest uses: "730 passed in 14s", "1 failed, 730 passed in 14s",
  // "2 errors, 728 passed, 1 skipped in 14s".
  //
  // It has to be read WHOLE. Matching a substring such as /(\d+) passed/
  // starts part-way into the line and silently drops any failure or error
  // that precedes "passed" on the same line — which is exactly how a suite with
  // a planted failure came to be reported as green.
  const lines = out
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean);
  const line = lines[lines.length - 1] || '';
  if (!/\bpassed\b/.test(line)) return null;

  const n = (re) => {
    const hit = line.match(re);
    return hit ? Number(hit[1]) : 0;
  };
  return {
    passed: n(/(\d+)\s+passed/),
    failed: n(/(\d+)\s+failed/),
    errors: n(/(\d+)\s+error/),
    skipped: n(/(\d+)\s+skipped/),
    xfailed: n(/(\d+)\s+xfailed/),
  };
}

/**
 * Measure the tree.
 *
 * Throws rather than returning a partial result. A build that cannot see the
 * source tree must fail loudly, because the alternative is committing a map
 * that still carries the raw `{{SIZE_TABLE}}` markers from data.mjs — which
 * looks like a rendering bug and silently preserves the stale numbers this
 * module was written to remove.
 */
export function measure(repoRoot) {
  const srcDir = join(repoRoot, SRC);
  if (!exists(srcDir)) {
    throw new Error(
      `measure: cannot find ${SRC} under ${repoRoot}. The atlas is meant to ` +
        'be built from the repository root, so its figures describe this tree. ' +
        'Refusing to emit a map with unmeasured placeholders.',
    );
  }

  const all = walk(srcDir, '.py');
  const total = all.reduce((sum, p) => sum + lineCount(p), 0);

  const sizes = LISTED.map((rel) => {
    const p = join(srcDir, rel);
    return exists(p) ? { name: rel, n: lineCount(p) } : null;
  }).filter(Boolean);
  sizes.sort((a, b) => b.n - a.n);

  // Three per row, aligned on the name column.
  const rows = [];
  for (let i = 0; i < sizes.length; i += 3) {
    rows.push(
      sizes
        .slice(i, i + 3)
        .map(({ name, n }) => name.padEnd(19) + String(n))
        .join('  '),
    );
  }

  // A test module is any test_*.py anywhere under tests/, except tests/tt/
  // which needs a device. Matched on the basename so files in subdirectories
  // such as tests/spec/ are counted.
  const testsDir = join(repoRoot, TESTS);
  const testFiles = walk(testsDir, '.py')
    .map((p) => relative(testsDir, p))
    .filter((f) => !f.startsWith('tt/') && f.split('/').pop().startsWith('test_'));

  const suite = runHostSuite(repoRoot);
  if (suite === null) {
    throw new Error(
      'measure: could not read a pytest summary from the host suite. The atlas ' +
        'states whether the tests passed, so the build needs the real result. ' +
        'Run the build from the repository root with .venv present.',
    );
  }

  return {
    total,
    totalLabel: `${(total / 1000).toFixed(1)}k`,
    sizeTable: rows.join('\n'),
    // Keyed by the path as written in data.mjs, so a marker can name the file
    // it is talking about. The size table is only a reading aid; this is what
    // the prose uses when it says "2,899 lines" about a specific module.
    lines: Object.fromEntries(sizes.map(({ name, n }) => [name, n])),
    suite,
    testModules: testFiles.length,
  };
}
