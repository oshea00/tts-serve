# Engine environments (resolving venvs between tts-serve and engine projects)

Every server in `impl/` runs in a single Python environment that holds two
separate sets of dependencies:

- **the engine**: the TTS package itself (`omnivoice`, `chatterbox-tts`, ...),
  along with torch, torchaudio, CUDA wheels and the engine's own stack;
- **tts-serve**: `tts-engine-common` plus `fastapi`, `uvicorn`, `loguru` and
  `soundfile`.

This document specifies how to build that environment when the engine is a git
checkout next to tts-serve, without global installs and without breaking the
engine project's own venv.

## Current state

Each `impl/server_<name>.md` has an Installation section with the same recipe:
create a venv (`python3 -m venv .venv`), `pip install <engine>` from PyPI, then
`pip install ./tts-engine-common fastapi uvicorn loguru soundfile`. IndexTTS is
different: it runs `uv sync --all-extras` in the engine's clone and pip-installs
the tts-serve dependencies into that venv. LuxTTS is git-only.

This recipe is still valid and stays supported. The problems below only show up
on some machines, or when working from a git checkout.

## The problem

1. **Bare `pip install` is refused.** Ubuntu's system Python and uv-managed
   Pythons are marked externally managed (PEP 668). Unless a venv is active,
   `pip install` fails with `externally-managed-environment`. An active venv
   isn't always enough: venvs created by `uv venv` / `uv sync` don't contain pip,
   so `pip` on the `PATH` is still the system one and fails the same way.
2. **The engine project's venv isn't ours to extend.** Installing the tts-serve
   packages into the engine's own `.venv` works until the next `uv sync` in the
   engine repo. That sync removes every package the engine's lockfile doesn't
   declare, which includes `tts-engine-common`, `uvicorn` and the rest, and the
   server stops starting.
3. **Venvs can't be stitched together.** Pointing one environment at another
   venv's `site-packages` (via `PYTHONPATH` or `.pth` files) breaks as soon as
   the two were built for different interpreters. Compiled extensions are tied
   to one Python ABI (for example cp312 vs cp313).
4. **The right torch build is hardware-specific, and PyPI doesn't encode it.**
   Engine repos often select a CUDA wheel index in `[tool.uv.sources]`. That
   setting is uv-only project metadata, so `pip install <engine>` from PyPI
   ignores it and gets PyPI's torch. On **aarch64 Linux** (NVIDIA GB10 /
   DGX Spark, Jetson Thor, GH200) this is worse: there are no cu128 CUDA wheels
   at all, so CUDA builds start at torch 2.9 on the cu130 index.
5. **Running from a checkout is a use case of its own.** People want to run an
   engine at its latest commit or with local patches, and have a `git pull`
   show up in the server without reinstalling.

## Proposal

tts-serve gets optional **per-engine environment projects** at
`envs/<engine>/`. Each one is a small uv project that describes exactly one
venv for exactly one server, and builds it with `uv sync`. The engine checkout
and its own venv are left alone.

### Layout and files

```
<parent>/
├── tts-serve/
│   └── envs/<engine>/
│       ├── pyproject.toml     committed
│       ├── .python-version    committed
│       ├── .venv/             gitignored (existing `.venv/` rule)
│       └── uv.lock            gitignored (`envs/*/uv.lock`)
└── <EngineRepo>/              engine git checkout, next to tts-serve
```

- `<engine>` is the lowercase engine name matching the server script, for
  example `envs/omnivoice/` for `impl/server_omnivoice.py`.
- Only `pyproject.toml` and `.python-version` are committed.
- **Sibling-checkout convention:** the engine is cloned next to tts-serve, so its
  path from `envs/<engine>/` is `../../../<EngineRepo>`. A user whose checkout is
  somewhere else edits that one path locally.

### `pyproject.toml` rules

- **Virtual project.** Set `[tool.uv] package = false`. The file describes a
  venv, not something installable. `[project]` gets a
  `tts-serve-env-<engine>` name, a version, and `requires-python`.
- **Dependencies.** List the engine, `tts-engine-common`, `fastapi`, `uvicorn`,
  `loguru` and `soundfile`. List **`torch` and `torchaudio` explicitly** too:
  uv only applies `tool.uv.sources` to requirements that are declared directly,
  not to transitive ones.
- **Sources.**
  - The engine is a path source to the sibling checkout with
    `editable = true`. An engine that is only wanted from PyPI can omit the
    source.
  - `tts-engine-common = { path = "../../tts-engine-common", editable = true }`.
  - `torch` and `torchaudio` point at CUDA index(es) through `marker` lists.
- **Indexes.** Every PyTorch index is declared with `explicit = true`, so only
  the packages whose sources name it are fetched from it. Without that, numpy
  and friends would come from the PyTorch index too. **Reuse the index names the
  engine's own pyproject uses** (see "Decisions" for why).
- **Torch pins are `override-dependencies`, not `constraint-dependencies`.**
  Start from the engine's own pins and add an aarch64 Linux entry (cu130,
  torch ≥ 2.9) wherever the engine only knows cu128.
- **Header comment.** Say what the file is for, the two commands to use it, the
  sibling-checkout convention, how to point at a checkout elsewhere, and why the
  torch pins are overrides.

### `.python-version`

Set it to the minor version of the interpreter the engine's own venv uses. uv
then finds the same wheels (torch alone is several GB) in its cache and
hardlinks them instead of downloading them again.

### Usage

From the tts-serve repo root:

```
uv sync --project envs/<engine>                                  # build or update the venv
uv run --project envs/<engine> python impl/server_<name>.py     # run the server
envs/<engine>/.venv/bin/python impl/server_<name>.py            # same, without uv (offline)
```

The server's `<ENGINE>_*` environment variables work unchanged.

### Boundaries

- **Optional.** The pip recipe in each `impl/server_<name>.md` stays the
  default. The env route is documented next to it as an alternative.
- **D7 is untouched.** `tts-engine-common` still has no torch or engine
  dependencies. Torch appears only in `envs/<engine>/pyproject.toml`.
- **Not used by the test suite.** The GPU-free suite keeps running against the
  stubs in `impl/tests/stubs/`, and needs no env.
- **No committed lockfiles.** AGENTS.md says no lockfile is configured, and a
  lock over an editable engine checkout would change with every commit to that
  checkout.

## Implementation (2026-09)

`envs/omnivoice/` was the first environment. It was verified on aarch64
(NVIDIA GB10, uv 0.9.26, CPython 3.13.11):

- `uv sync --project envs/omnivoice` finished in about 4 s from the uv cache.
  It installed `torch 2.9.1+cu130`, and `torch.cuda.is_available()` returned
  True.
- `omnivoice` and `tts-engine-common` both import from their checkouts (editable
  installs).
- The live `GET /capabilities` output was identical to
  `impl/tests/snapshots/omnivoice_capabilities.json`.
- `tools/speak.py` synthesized 24 kHz 16-bit mono WAVs. The first request paid
  the Whisper load (no transcript given); a warm request ran at RTF 0.47.
- Resolution also works against an **unmodified upstream** OmniVoice checkout.
  Its pyproject sends all Linux torch to cu128.

Other changes: `.gitignore` gained `envs/*/uv.lock`,
`impl/server_omnivoice.md` gained an "Alternative: from a local OmniVoice git
checkout (uv)" section, and `README.md` and `AGENTS.md` mention `envs/`.

Decisions taken while implementing:

- **Overrides, not constraints.** uv also reads the `tool.uv.sources` of a path
  dependency. Upstream OmniVoice maps all Linux torch to the cu128 index. With
  constraints, that requirement sits next to ours and resolution fails:
  `Requirements contain conflicting indexes for package torch in split
  platform_machine == 'aarch64' and sys_platform == 'linux'`. An override
  *replaces* every torch requirement in the graph, including the checkout's, so
  our index wins and an unmodified checkout works.
- **Same index names as the engine.** Matching names mean the root's sources and
  the checkout's sources agree wherever the engine's mapping is already correct
  (for example cu128 on x86_64 Linux).
- **Editable engine install.** A `git pull` in the checkout takes effect on the
  next server start. Nothing gets reinstalled, except when the engine's
  dependencies change, and then `uv sync` picks that up.
- **Python pinned to 3.13** to match OmniVoice's venv, so the cp313 aarch64
  torch wheels come from the cache.

Known quirks (harmless, recorded so nobody chases them):

- **Every `uv sync` / `uv run` re-resolves.** `uv lock --check` always reports
  the lock as out of date for this project, whether the pins are overrides or
  constraints. The cause hasn't been pinned down. Resolving from cache takes
  about 100 ms, but once uv's HTTP cache expires it needs index access. Offline,
  run the venv's Python directly.
- **`nvidia-cusparselt-cu13` is reinstalled on every sync.** OmniVoice's own
  venv does the same, so this isn't specific to envs.
- **torch 2.9.1 warns about the GB10** (CUDA capability 12.1 vs a supported
  maximum of 12.0). Synthesis works anyway.

## Adding an environment for another engine

1. Clone the engine next to tts-serve. If it has its own venv, note that venv's
   Python version and torch build.
2. Copy `envs/omnivoice/` to `envs/<engine>/`. Change the project name, the
   engine requirement, and its path source. If the engine's install needs
   extras or git-only dependencies (IndexTTS's `--all-extras`, LuxTTS's
   git-only `linacodec`), express them here: extras on the engine requirement,
   git sources for the rest.
3. Read the engine's own `pyproject.toml`. Copy its torch index names, URLs and
   pins, turn any `constraint-dependencies` into `override-dependencies`, and
   add the aarch64 Linux cu130 entry if the engine lacks one. An engine without
   CUDA torch (for example Qwen3-TTS MLX on Apple Silicon) needs no torch
   sources or indexes.
4. Set `.python-version`.
5. Verify, following the same checks as the OmniVoice implementation above:
   - `uv sync --project envs/<engine>` succeeds;
   - torch imports with the expected version, and CUDA is available where
     expected;
   - the engine imports from the checkout;
   - the live `/capabilities` output matches the committed snapshot (except
     `device` for dots.tts);
   - `tools/speak.py` synthesizes audio;
   - the full test suite still passes.
6. Document it: add an "Alternative: from a local <Engine> git checkout (uv)"
   section to `impl/server_<name>.md`, and list the new directory in the `envs/`
   bullet of `AGENTS.md`.

Candidates, from the current install docs (none done yet):

| Engine | Current install | Why an env would help |
|---|---|---|
| IndexTTS | `uv sync --all-extras` in the engine clone, tts-serve deps pip-installed into that venv | Problems 1 and 2: that uv venv has no pip, so the documented `pip install` reaches the system pip; and even when the deps are installed, the next `uv sync` there removes them |
| LuxTTS | git-only, plus a git-only dependency | No PyPI route exists, so a checkout is the only option |
| Chatterbox, Qwen3-TTS, faster-qwen3-tts, dots.tts | PyPI | Only needed for problem 4 (hardware-specific torch) or to run from a checkout |
| Qwen3-TTS MLX | PyPI (`mlx-audio`) | Low value: no CUDA torch involved |

## Open questions

- **Checkout outside the sibling layout.** Today the user edits the committed
  path locally, which then shows up as a modification in `git status`. If that
  becomes common, a non-committed override mechanism may be worth adding.
- **Shared launcher.** Is a small `envs/run <engine>` wrapper worth having, or
  are `uv run --project` and the direct `.venv/bin/python` command enough?
