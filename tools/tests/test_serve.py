"""GPU-free unit tests for tools/serve.py.

Covers the launcher behavior specified in docs/04-engine-environments.md
('Launcher'): env discovery, reading ``[tool.tts-serve] server``, the
sync-when-needed rules, sync-failure handling, and the final exec.  Every
test runs against a fake repo in tmp_path, with uv, subprocess, and
os.execv stubbed out, so nothing is installed and no server starts.  One
test checks the real committed envs/*/pyproject.toml files.
"""

import os
import types
from pathlib import Path

import pytest

pytest.importorskip("tomllib")  # serve.py needs Python 3.11+

import serve

REAL_REPO = Path(__file__).resolve().parents[2]

# A fixed past instant for input files, so "newer than the stamp" never
# depends on filesystem timestamp granularity.
OLD = 1_700_000_000
NEW = OLD + 1_000


def _write_env(repo: Path, name: str, server: str | None = "impl/server_foo.py", sources: str = "") -> Path:
    env_dir = repo / "envs" / name
    env_dir.mkdir(parents=True)
    tool = f'[tool.tts-serve]\nserver = "{server}"\n' if server is not None else ""
    (env_dir / "pyproject.toml").write_text(
        f'[project]\nname = "env-{name}"\nversion = "0.1.0"\n\n{tool}{sources}',
        encoding="utf-8",
    )
    (env_dir / ".python-version").write_text("3.13\n", encoding="utf-8")
    return env_dir


def _make_venv(env_dir: Path, stamped: bool = True) -> Path:
    python = serve.venv_python(env_dir)
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    if stamped:
        stamp = env_dir / ".venv" / serve.SYNC_STAMP_NAME
        stamp.touch()
        os.utime(stamp, (NEW, NEW))
    return python


def _age(*paths: Path) -> None:
    for path in paths:
        os.utime(path, (OLD, OLD))


@pytest.fixture
def repo(tmp_path):
    """A fake repo: impl/server_foo.py, an engine checkout, and envs/foo/."""
    (tmp_path / "impl").mkdir()
    (tmp_path / "impl" / "server_foo.py").write_text("", encoding="utf-8")
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    env_dir = _write_env(
        tmp_path,
        "foo",
        sources=(
            "[tool.uv.sources]\n"
            'engine = { path = "../../engine", editable = true }\n'
            "torch = [\n"
            '  { index = "cu130", marker = "platform_machine == \'aarch64\'" },\n'
            "]\n"
        ),
    )
    _age(env_dir / "pyproject.toml", env_dir / ".python-version", tmp_path / "engine" / "pyproject.toml")
    # Resolved, as main() does, so direct helper calls agree with it.
    return tmp_path.resolve()


@pytest.fixture
def calls(monkeypatch):
    """Stub uv, subprocess.run and os.execv; record what the launcher asked for."""
    record = types.SimpleNamespace(sync=[], execv=[], sync_returncode=0, uv="/usr/bin/uv")

    def fake_which(name):
        return record.uv if name == "uv" else None

    def fake_run(cmd, env=None):
        record.sync.append((cmd, env))
        if record.sync_returncode == 0:
            # uv sync builds the venv when it doesn't exist yet.
            env_dir = Path(cmd[cmd.index("--project") + 1])
            python = serve.venv_python(env_dir)
            python.parent.mkdir(parents=True, exist_ok=True)
            python.touch()
        return types.SimpleNamespace(returncode=record.sync_returncode)

    def fake_execv(path, argv):
        record.execv.append((Path(path), argv))

    monkeypatch.setattr(serve.shutil, "which", fake_which)
    monkeypatch.setattr(serve.subprocess, "run", fake_run)
    monkeypatch.setattr(serve.os, "execv", fake_execv)
    return record


# ---------------------------------------------------------------------------
# Discovery and env loading
# ---------------------------------------------------------------------------


def test_discover_envs_listsOnlyDirsWithPyproject_sorted(repo):
    # GIVEN envs/ with two real envs, a dir without pyproject.toml, and a stray file
    _write_env(repo, "bar")
    (repo / "envs" / "empty").mkdir()
    (repo / "envs" / "notes.txt").write_text("", encoding="utf-8")
    # WHEN discovering
    # THEN only the envs are listed, sorted
    assert serve.discover_envs(repo) == ["bar", "foo"]


def test_discover_envs_withoutEnvsDir_returnsEmpty(tmp_path):
    assert serve.discover_envs(tmp_path) == []


def test_load_env_resolvesServerAndInputs(repo):
    # GIVEN envs/foo with a path source to the engine and an index-only source
    # WHEN loaded
    env = serve.load_env(repo, "foo")
    # THEN the server resolves against the repo root, and the inputs are the
    # env's own files plus the engine's pyproject.toml (not the index source)
    assert env.server == (repo / "impl" / "server_foo.py").resolve()
    assert env.inputs == [
        repo / "envs" / "foo" / "pyproject.toml",
        repo / "envs" / "foo" / ".python-version",
        (repo / "engine" / "pyproject.toml").resolve(),
    ]


def test_load_env_withListFormPathSource_includesItsPyproject(repo):
    # GIVEN a path source written as a marker-guarded list
    (repo / "other").mkdir()
    (repo / "other" / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    _write_env(
        repo,
        "bar",
        sources=(
            "[tool.uv.sources]\n"
            "other = [\n"
            '  { path = "../../other", marker = "sys_platform == \'linux\'" },\n'
            "]\n"
        ),
    )
    # WHEN loaded
    env = serve.load_env(repo, "bar")
    # THEN that source's pyproject.toml is watched too
    assert (repo / "other" / "pyproject.toml").resolve() in env.inputs


def test_load_env_withMissingPathSource_skipsIt(repo):
    # GIVEN a path source whose checkout doesn't exist (uv sync will report it)
    _write_env(repo, "bar", sources='[tool.uv.sources]\ngone = { path = "../../gone" }\n')
    # WHEN loaded
    env = serve.load_env(repo, "bar")
    # THEN it is simply not watched
    assert len(env.inputs) == 2


def test_load_env_withoutToolTable_raisesServeError(repo):
    _write_env(repo, "bar", server=None)
    with pytest.raises(serve.ServeError, match=r"no \[tool.tts-serve\] server entry"):
        serve.load_env(repo, "bar")


def test_load_env_withMissingServerScript_raisesServeError(repo):
    _write_env(repo, "bar", server="impl/server_missing.py")
    with pytest.raises(serve.ServeError, match="impl/server_missing.py .* does not exist"):
        serve.load_env(repo, "bar")


def test_load_env_withMalformedToml_raisesServeError(repo):
    (repo / "envs" / "foo" / "pyproject.toml").write_text("[project\n", encoding="utf-8")
    with pytest.raises(serve.ServeError, match="Cannot read"):
        serve.load_env(repo, "foo")


def test_venv_python_onWindows_usesScriptsDir(tmp_path):
    assert serve.venv_python(tmp_path, windows=True) == tmp_path / ".venv" / "Scripts" / "python.exe"
    assert serve.venv_python(tmp_path, windows=False) == tmp_path / ".venv" / "bin" / "python"


# ---------------------------------------------------------------------------
# When to sync
# ---------------------------------------------------------------------------


def test_sync_reason_withoutVenv_isNotBuilt(repo):
    env = serve.load_env(repo, "foo")
    assert serve.sync_reason(env, repo) == serve.NOT_BUILT


def test_sync_reason_withUnstampedVenv_needsSync(repo):
    # GIVEN a venv built by a manual uv sync (no launcher stamp)
    _make_venv(repo / "envs" / "foo", stamped=False)
    env = serve.load_env(repo, "foo")
    assert serve.sync_reason(env, repo) == "not yet synced by serve.py"


def test_sync_reason_withFreshStamp_isNone(repo):
    _make_venv(repo / "envs" / "foo")
    env = serve.load_env(repo, "foo")
    assert serve.sync_reason(env, repo) is None


def test_sync_reason_withForce_alwaysSyncs(repo):
    _make_venv(repo / "envs" / "foo")
    env = serve.load_env(repo, "foo")
    assert serve.sync_reason(env, repo, force=True) == "--sync given"


@pytest.mark.parametrize(
    "changed, shown",
    [
        ("envs/foo/pyproject.toml", "envs/foo/pyproject.toml"),
        ("envs/foo/.python-version", "envs/foo/.python-version"),
        ("engine/pyproject.toml", "engine/pyproject.toml"),
    ],
)
def test_sync_reason_withInputNewerThanStamp_namesIt(repo, changed, shown):
    # GIVEN a synced venv, then one of the env's inputs changes
    _make_venv(repo / "envs" / "foo")
    os.utime(repo / changed, (NEW + 1, NEW + 1))
    env = serve.load_env(repo, "foo")
    # THEN the reason names the changed file, relative to the repo root
    assert serve.sync_reason(env, repo) == f"{shown} changed"


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------


def test_main_list_showsEveryStatus(repo, capsys):
    # GIVEN one ready env, one unbuilt, one needing a sync, and one broken
    _make_venv(repo / "envs" / "foo")
    _write_env(repo, "bar")
    _make_venv(_write_env(repo, "baz"), stamped=False)
    _write_env(repo, "qux", server=None)
    # WHEN listing
    assert serve.main(["--list"], repo_root=repo) == 0
    out = capsys.readouterr().out
    # THEN each env shows its server and status
    lines = {line.split()[0]: line for line in out.splitlines()[1:]}
    assert "impl/server_foo.py" in lines["foo"] and lines["foo"].endswith("ready")
    assert lines["bar"].endswith("not built")
    assert lines["baz"].endswith("needs sync (not yet synced by serve.py)")
    assert "error:" in lines["qux"] and "[tool.tts-serve]" in lines["qux"]


def test_main_list_withoutEnvs_saysSo(tmp_path, capsys):
    assert serve.main(["--list"], repo_root=tmp_path) == 0
    assert "No environments found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Usage errors (exit 2)
# ---------------------------------------------------------------------------


def test_main_withoutEngine_exits2(repo, capsys):
    with pytest.raises(SystemExit) as exc:
        serve.main([], repo_root=repo)
    assert exc.value.code == 2
    assert "ENGINE" in capsys.readouterr().err


def test_main_withUnknownEngine_exits2AndListsAvailable(repo, capsys, calls):
    with pytest.raises(SystemExit) as exc:
        serve.main(["nope"], repo_root=repo)
    assert exc.value.code == 2
    assert "unknown environment 'nope' (available: foo)" in capsys.readouterr().err
    assert calls.execv == []


# ---------------------------------------------------------------------------
# Start (sync when needed, then exec)
# ---------------------------------------------------------------------------


def test_main_withReadyEnv_execsServerWithoutUv(repo, calls, capsys):
    # GIVEN a ready venv and no uv on PATH (none is needed)
    python = _make_venv(repo / "envs" / "foo")
    calls.uv = None
    # WHEN starting
    assert serve.main(["foo"], repo_root=repo) == 0
    # THEN no sync ran and the venv python replaced the launcher with the server
    assert calls.sync == []
    assert calls.execv == [(python, [str(python), str((repo / "impl" / "server_foo.py").resolve())])]
    assert "Starting impl/server_foo.py with envs/foo/.venv/bin/python" in capsys.readouterr().out


def test_main_withUnbuiltVenv_syncsStampsAndExecs(repo, calls, capsys, monkeypatch):
    # GIVEN no venv yet, and an unrelated venv active in the shell
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/else/.venv")
    # WHEN starting
    assert serve.main(["foo"], repo_root=repo) == 0
    # THEN uv sync ran for the env without VIRTUAL_ENV, the stamp was written,
    # and the server started
    env_dir = repo / "envs" / "foo"
    (cmd, child_env), = calls.sync
    assert cmd == ["/usr/bin/uv", "sync", "--project", str(env_dir)]
    assert "VIRTUAL_ENV" not in child_env
    assert (env_dir / ".venv" / serve.SYNC_STAMP_NAME).is_file()
    assert len(calls.execv) == 1
    assert "Syncing envs/foo (venv not built yet) ..." in capsys.readouterr().out


def test_main_afterSync_nextStartIsReady(repo, calls):
    # GIVEN a first start that synced
    serve.main(["foo"], repo_root=repo)
    # WHEN starting again with nothing changed
    serve.main(["foo"], repo_root=repo)
    # THEN only the first start synced
    assert len(calls.sync) == 1
    assert len(calls.execv) == 2


def test_main_withSyncFlag_syncsReadyEnv(repo, calls):
    _make_venv(repo / "envs" / "foo")
    assert serve.main(["foo", "--sync"], repo_root=repo) == 0
    assert len(calls.sync) == 1
    assert len(calls.execv) == 1


def test_main_whenAutoSyncFailsWithExistingVenv_warnsAndStarts(repo, calls, capsys):
    # GIVEN a venv whose inputs changed, and a sync that fails (e.g. offline)
    _make_venv(repo / "envs" / "foo")
    os.utime(repo / "envs" / "foo" / "pyproject.toml", (NEW + 1, NEW + 1))
    calls.sync_returncode = 1
    # WHEN starting
    assert serve.main(["foo"], repo_root=repo) == 0
    # THEN it warns, starts anyway, and leaves the stamp so the next start retries
    assert "Warning: uv sync failed (exit code 1)" in capsys.readouterr().out
    assert len(calls.execv) == 1
    env = serve.load_env(repo, "foo")
    assert serve.sync_reason(env, repo) == "envs/foo/pyproject.toml changed"


def test_main_whenForcedSyncFails_exits1(repo, calls, capsys):
    _make_venv(repo / "envs" / "foo")
    calls.sync_returncode = 2
    assert serve.main(["foo", "--sync"], repo_root=repo) == 1
    assert "Error: uv sync failed (exit code 2)." in capsys.readouterr().err
    assert calls.execv == []


def test_main_whenFirstSyncFails_exits1(repo, calls):
    calls.sync_returncode = 1
    assert serve.main(["foo"], repo_root=repo) == 1
    assert calls.execv == []


def test_main_whenSyncNeededWithoutUv_exits1WithInstallHint(repo, calls, capsys):
    calls.uv = None
    assert serve.main(["foo"], repo_root=repo) == 1
    assert serve.UV_INSTALL_URL in capsys.readouterr().err
    assert calls.execv == []


def test_main_withBrokenEnv_exits1(repo, calls, capsys):
    _write_env(repo, "bar", server="impl/server_missing.py")
    assert serve.main(["bar"], repo_root=repo) == 1
    assert "does not exist" in capsys.readouterr().err
    assert calls.sync == [] and calls.execv == []


def test_main_whenExecFails_exits1(repo, calls, capsys, monkeypatch):
    _make_venv(repo / "envs" / "foo")

    def failing_execv(path, argv):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(serve.os, "execv", failing_execv)
    assert serve.main(["foo"], repo_root=repo) == 1
    assert "Cannot start envs/foo/.venv/bin/python" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The committed envs
# ---------------------------------------------------------------------------


def test_committedEnvs_eachNameAnExistingServerScript():
    # GIVEN the envs/ directories committed to this repo
    names = serve.discover_envs(REAL_REPO)
    assert names, "expected at least one env under envs/"
    for name in names:
        # WHEN loaded
        env = serve.load_env(REAL_REPO, name)
        # THEN each names a server script in impl/
        assert env.server.parent == REAL_REPO / "impl"
        assert env.server.name.startswith("server_")
