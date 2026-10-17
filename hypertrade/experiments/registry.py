from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import pandas as pd
import yaml


class ExperimentRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, prefix: str) -> Path:
        prefix = "".join(c if c.isalnum() or c in "_-" else "_" for c in prefix)[:80] or "run"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.root / f"{prefix}_{timestamp}"
        suffix = 0
        candidate = run_dir
        while candidate.exists():
            suffix += 1
            candidate = self.root / f"{prefix}_{timestamp}_{suffix:02d}"
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def save_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2, default=str))

    def save_yaml(self, path: Path, payload: dict) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))

    def save_table(self, path: Path, frame: pd.DataFrame) -> None:
        if frame.empty:
            frame = pd.DataFrame()
        frame.to_parquet(path, index=False)

    def save_study_tables(self, run_dir: Path, trials_df: pd.DataFrame, pareto_df: pd.DataFrame, shortlist_df: pd.DataFrame) -> None:
        self.save_table(run_dir / "trials.parquet", trials_df)
        self.save_table(run_dir / "pareto_trials.parquet", pareto_df)
        self.save_table(run_dir / "candidate_shortlist.parquet", shortlist_df)

    def save_text(self, path: Path, content: str) -> None:
        path.write_text(content)

    def payload_sha256(self, payload: dict) -> str:
        normalized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def file_sha256(self, path: str | Path) -> str | None:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return None
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def git_snapshot(self) -> dict:
        def _run_git(args: list[str]) -> str | None:
            try:
                result = subprocess.run(["git", *args], check=True, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2])
            except Exception:
                return None
            value = result.stdout.strip()
            return value or None

        return {
            "commit": _run_git(["rev-parse", "HEAD"]),
            "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "status_short": _run_git(["status", "--short"]),
        }

    def build_environment_snapshot(self) -> dict:
        return {
            "python_version": sys.version,
            "dependencies": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "git": self.git_snapshot(),
        }

    def artifact_inventory(self, run_dir: str | Path) -> list[dict]:
        root = Path(run_dir)
        rows = []
        for path in sorted(root.glob("*")):
            if not path.is_file():
                continue
            rows.append({"name": path.name, "size_bytes": path.stat().st_size, "sha256": None if path.name == "artifact_index.json" else self.file_sha256(path)})
        return rows

    def validate_artifacts(self, run_dir: str | Path, required_names: list[str] | tuple[str, ...]) -> tuple[bool, list[str]]:
        root = Path(run_dir)
        missing = [name for name in required_names if not (root / name).exists()]
        return not missing, missing


