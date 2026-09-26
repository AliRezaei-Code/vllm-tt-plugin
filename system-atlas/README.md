# system-atlas

Vendored from [inkboard/system-atlas](https://github.com/inkboard/system-atlas)
(see `LICENSE`). The skill instructions and references are unmodified; only
`atlas/data.mjs` is hand-edited.

## Files

| File | Kind | What it is |
|---|---|---|
| `SKILL.md` | vendored | the skill itself |
| `references/design-language.md` | vendored | layout, palette, isometric grammar, shapes by role, chapter recipe |
| `references/process-and-lessons.md` | vendored | why the process is what it is |
| `assets/`, `evals/` | vendored | renderer template, build script, starter data, evals |
| `atlas/build.mjs`, `atlas/template.html` | vendored | copied from `assets/` |
| `atlas/data.mjs` | **edited** | the single source of truth for this repo's atlas |
| `SYSTEM.md` | generated | text twin: decisions, structures, flows, questions by ID |
| `atlas.html` | generated | the interactive self-contained map |
| `CONTEXT.md` | hand-written | the glossary — nouns only, one line each |

## Build

```bash
bun system-atlas/atlas/build.mjs
```

Writes `SYSTEM.md` and `atlas.html` into `system-atlas/`, one level above the
atlas home. Never hand-edit either generated file: change `atlas/data.mjs` and
rebuild.

## Open the map

```bash
open system-atlas/atlas.html
```

`atlas.html` is one self-contained file. `file://` renders it as a static
snapshot in some in-app browsers, so serve the folder if the fonts or the
arrows misbehave:

```bash
cd system-atlas && python3 -m http.server 8000
```

## Baseline

Every claim in the atlas is measured against vLLM **0.26.0**, pinned by
`docs/install-vllm-tt.sh:36` and nested read-only at `third_party/vllm-0.26.0/`
(tag `v0.26.0` = `568afb3a13806beb53bb2e6bd518269357b237c0`). It is not measured
against upstream `main`.
