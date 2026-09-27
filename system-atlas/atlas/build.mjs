// Builds <outDir>/SYSTEM.md and <outDir>/atlas.html from data.mjs (same folder).
// Usage: bun <atlasDir>/build.mjs   — outDir = parent of atlasDir by default, or META.outDir
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { META, DECISIONS, GROUPS, NODES, FLOWS, CH, HOW_HTML } from './data.mjs';
import { measure } from './measure.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const outDir = META.outDir ? join(here, META.outDir) : join(here, '..');
// LOCAL DIVERGENCE from the upstream skill: figures the atlas states about the
// repository are measured, not hand-written. See measure.mjs for why.
const m = measure(join(here, '..', '..'));

// Only what the run observed. No clause is emitted that was not measured, so
// the text cannot claim a passing suite from a command that only collected.
const s = m.suite;
const notGreen = s.failed || s.errors;
const suiteSentence = [
  `${s.passed} passed`,
  notGreen ? `${s.failed} failed` : null,
  s.errors ? `${s.errors} error${s.errors === 1 ? '' : 's'}` : null,
  s.skipped ? `${s.skipped} skipped` : null,
  s.xfailed ? `${s.xfailed} xfailed` : null,
]
  .filter(Boolean)
  .join(', ');

// Fill a {{LINES:path}} marker, or fail. A marker naming a file the
// measurement does not know about is a typo, and silently emitting 0 would
// put a wrong number into prose that reads as measured.
const fill = (str) => {
  if (typeof str !== 'string') return str;
  return str.replace(/\{\{LINES:([\w./]+)\}\}/g, (_, file) => {
    const n = m.lines[file];
    if (n === undefined) {
      throw new Error(
        `build: no measured line count for "${file}". data.mjs names a file ` +
          'the measurement does not know about.',
      );
    }
    // Grouped, so prose reads the same as the hand-written numbers it replaces.
    return n.toLocaleString('en-US');
  });
};

// A recursive walk, not a list of field names. The enumerated version named
// one/what/how/steps/cond/story/lede and silently missed `flow` and META's
// intro/onePara/platformGives/weOwn/filesystem, so a marker dropped into one of
// those tomorrow would keep its hand-written number and nothing would say so.
// Walking every string cannot be incomplete by construction.
const deep = (o) => {
  if (Array.isArray(o)) return o.map(deep);
  if (o && typeof o === 'object') {
    return Object.fromEntries(Object.entries(o).map(([k, v]) => [k, deep(v)]));
  }
  return typeof o === 'string' ? fill(o) : o;
};

// NODES, CH and the cost model carry line counts in their prose too, so the
// markers are filled there too. Fixing only the size table would leave the
// next stale copy of the same number already committed.
const NODES_FILLED = deep(NODES);
const CH_FILLED = deep(CH);
const COST_FILLED = deep(META.costModel);
const META_FILLED = deep(META);
const FLOWS_FILLED = deep(FLOWS);

const HOW = HOW_HTML.replace('{{SIZE_TABLE}}', m.sizeTable).replace(
  '{{TESTS}}',
  `${suiteSentence} across ${m.testModules} test modules under <code>tests/</code> ` +
    `(excluding <code>tests/tt</code>, which needs a live server and TT hardware), ` +
    `measured by running the host suite at build time. ` +
    `<code>ci/host-stubs/ttnn/</code> supplies an import-only <code>ttnn</code> ` +
    `stand-in whose every device-reaching entry point raises, so a test that ` +
    `starts depending on real hardware fails loudly instead of passing against ` +
    `a fake device.`,
);

const STATS = META_FILLED.stats.map((st) =>
  st.k === 'Plugin' ? { ...st, v: `${m.totalLabel} lines` } : st,
);

// Totality check. The property this change exists to guarantee is that no
// figure the atlas states about the tree is hand-written, and a marker that
// survives into the output breaks it silently — a map containing
// "{{LINES:...}}" looks like a rendering bug, not a stale measurement. One
// check over everything the build emits, rather than a grep someone has to
// remember to run.
// Over the FILLED values, which are what reach the output. Checking the raw
// imports here would fire on markers that were resolved correctly.
const emitted = JSON.stringify([NODES_FILLED, CH_FILLED, COST_FILLED, HOW, STATS, DECISIONS, GROUPS, FLOWS_FILLED]);
if (emitted.includes('{{')) {
  const leftover = emitted.match(/[^"\\]{0,40}\{\{[^"\\]{0,40}/g) || [];
  throw new Error(
    `build: unfilled marker(s) reached the output: ${[...new Set(leftover)].join(' | ')}`,
  );
}

// ---------- shared helpers ----------
const Q = (c) => (typeof c === 'string' ? { q: c } : c);
const md = (s) =>
  String(s)
    .replace(/<code>(.*?)<\/code>/g, '`$1`')
    .replace(/<mark>(.*?)<\/mark>/g, '**$1**')
    .replace(/<em>(.*?)<\/em>/g, '_$1_')
    .replace(/<b>(.*?)<\/b>/g, '**$1**')
    .replace(/<\/p>\s*<p>/g, '\n\n')
    .replace(/<[^>]+>/g, '')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .trim();
const groupTitle = Object.fromEntries(GROUPS.map((g) => [g.id, g.title]));
const cnt = { open: 0, res: 0 };
NODES_FILLED.forEach((n) => (n.cond || []).map(Q).forEach((c) => (c.r || c.to ? cnt.res++ : cnt.open++)));

// ---------- SYSTEM.md ----------
function buildSystemMd() {
  const out = [];
  out.push(`# ${META.title} — System Definition`, '');
  out.push(META.intro, '');
  out.push(`_Question status: **${cnt.open} open · ${cnt.res} resolved**._`, '');
  out.push('## One paragraph', '', META.onePara, '');
  out.push('## Decisions locked', '', '| Axis | Decision | ADR |', '|---|---|---|');
  DECISIONS.forEach((d) => out.push(`| ${d.axis} | ${d.decision} | ${d.adr} |`));
  out.push('');
  out.push('## Cost model', '');
  COST_FILLED.forEach((l) => out.push(l));
  if (META.deepDive) out.push('## Deep dives', '', META.deepDive, '');
  out.push('## Reading order (the atlas chapters)', '');
  CH_FILLED.forEach((c, i) => out.push(`${i + 1}. **${c.title}** — ${md(c.lede)}${c.reveal.length ? ` _(adds ${c.reveal.join(', ')})_` : ''}`));
  out.push('');
  out.push('## Structures', '');
  const index = [];
  for (const g of GROUPS) {
    out.push(`### ${g.title}${g.id === 'off' ? ' (designed for, not built)' : ''}`, '');
    for (const n of NODES_FILLED.filter((n) => n.group === g.id)) {
      out.push(`#### ${n.code} · ${n.name}${n.ghost ? ' _(not switched on)_' : ''}`, '');
      out.push(`**In one line.** ${md(n.one)}`, '');
      out.push(`**What it does.** ${md(n.what)}`, '');
      out.push(`**How it's built.** ${md(n.how)}`, '');
      if (n.steps) {
        out.push('**Steps in execution.**', '');
        n.steps.forEach((s, i) => out.push(`${i + 1}. **${s[0]}** — ${s[1]}`));
        out.push('');
      }
      const cs = (n.cond || []).map(Q);
      if (cs.length) {
        out.push('**Questions.**', '');
        cs.forEach((c, i) => {
          const id = `Q-${n.code}${i + 1}`;
          out.push(c.r ? `- ~~**${id}** ${md(c.q)}~~ ✓ ${md(c.r)}` : c.to ? `- **${id}** ${md(c.q)} → _${md(c.to)}_` : `- **${id}** ${md(c.q)}`);
          index.push([id, n.code, c]);
        });
        out.push('');
      }
    }
  }
  out.push('## Flows (representative packets)', '', 'Payload shapes are what the design implies, not measured traffic.', '');
  for (const f of FLOWS_FILLED) {
    out.push(`### ${f.name}`, '', '| # | From → To | Packet | Representative payload |', '|---|---|---|---|');
    f.hops.forEach((h, i) => out.push(`| ${i + 1} | ${h[0]} → ${h[1]} | ${h[2]} | \`${JSON.stringify(h[3]).replace(/\|/g, '\\|')}\` |`));
    out.push('');
  }
  out.push('## Questions — index', '', 'Reference by ID. ✓ resolved (with date) · otherwise open.', '');
  index.forEach(([id, code, c]) => out.push(c.r ? `- ~~**${id}**~~ (${code}) ✓ ${md(c.r)}` : `- **${id}** (${code}) ${md(c.q)}`));
  out.push('');
  if (META.platformGives || META.weOwn) out.push('## What the platform gives vs what we own', '', `**Platform gives:** ${META.platformGives||''}`, '', `**We own:** ${META.weOwn||''}`, '');
  if (META.filesystem) out.push('## Planned filesystem', '', '```', META.filesystem.trimEnd(), '```', '');
  out.push('## How this file is maintained', '', `Generated from \`${META.sourcePath||'atlas/data.mjs'}\` by \`${META.buildCmd||'bun atlas/build.mjs'}\`, which also builds the interactive atlas (\`atlas.html\`${META.artifactUrl?`, published at ${META.artifactUrl}`:''}). Edit the data file, rebuild, republish — never edit this file by hand.`, '');
  return out.join('\n');
}

// ---------- atlas.html ----------
function buildAtlasHtml() {
  const tpl = readFileSync(join(here, 'template.html'), 'utf8');
  const decisionsHtml = DECISIONS.map((d) => `<li><b>${d.axis}.</b> ${md(d.decision).replace(/\*\*(.*?)\*\*/g, '<b>$1</b>').replace(/`(.*?)`/g, '<code>$1</code>').replace(/\[(.*?)\]\((.*?)\)/g, '$1')}</li>`).join('');
  const data = [
    `const GROUPS = ${JSON.stringify(GROUPS)};`,
    `const NODES = ${JSON.stringify(NODES_FILLED)};`,
    `const FLOWS = ${JSON.stringify(FLOWS_FILLED)};`,
    `const CH = ${JSON.stringify(CH_FILLED)};`,
    `const HOW_HTML = ${JSON.stringify(HOW)};`,
    `const DECISIONS_HTML = ${JSON.stringify(decisionsHtml)};`,
  ].join('\n');

  return tpl.replace('__TITLE__', META.title + ' Atlas').replace('/*__DATA__*/', data + `\nconst STATS = ${JSON.stringify(STATS||[])};\nconst TITLE = ${JSON.stringify(META.title||'System')};`);
}

writeFileSync(join(outDir, 'SYSTEM.md'), buildSystemMd());
writeFileSync(join(outDir, 'atlas.html'), buildAtlasHtml());
console.log(`built SYSTEM.md + atlas.html · ${cnt.open} open · ${cnt.res} resolved · ${NODES.length} structures · ${DECISIONS.length} decisions`);
