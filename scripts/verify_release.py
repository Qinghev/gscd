from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


GITHUB_REGULAR_FILE_LIMIT = 100 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest(root: Path) -> Path:
    candidates = (
        root / "results" / "siads_current" / "artifact_manifest.json",
        root / "results" / "sisc_current" / "artifact_manifest.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No artifact manifest found under results/siads_current or "
        "results/sisc_current"
    )


def iter_pt_references(value: object):
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_pt_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_pt_references(child)
    elif isinstance(value, str) and value.endswith(".pt"):
        normalized = value.replace("\\", "/")
        for prefix in ("checkpoints/", "data/"):
            start = normalized.find(prefix)
            if start >= 0:
                yield normalized[start:]
                break


def iter_release_paths(root: Path):
    """Yield artifact paths while excluding Git's internal object store."""
    for path in root.rglob("*"):
        if ".git" not in path.relative_to(root).parts:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--require-artifacts", action="store_true")
    parser.add_argument(
        "--github-ready",
        action="store_true",
        help="also require an initialized Git worktree",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = resolve_manifest(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    required = (
        ".gitattributes",
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
        "README.md",
        "REPRODUCIBILITY.md",
        "environment.yml",
        "requirements.txt",
        "reproducibility/protocol.json",
    )
    required_results = (
        manifest_path.relative_to(root).as_posix(),
        "results/sisc_current/finite_solve_audit_fp12.json",
        "results/sisc_current/solver_control_comparison.json",
        "results/gsympnet_fput_recurrence_5seed.json",
        "results/figures/fput/fput_recurrence_profile_5seed.pdf",
        "results/figures/fput/fput_recurrence_profile_5seed.png",
        "results/multi_ic_toda_5seed.json",
        "results/toda_time_curves_5seed.json",
        "results/toda_invariants_5seed.json",
        "results/gsympnet_multi_ic_5seed.json",
        "results/phi4_time_curves_5seed.json",
        "results/gsympnet_double_pendulum_shadowing_5seed.json",
        "results/double_pendulum_time_curves_5seed.json",
        "results/multi_ic_pendula_wrapped_5seed.json",
        "results/spherical_pendulum_meridian_5seed.json",
        "results/gsympnet_spherical_pendulum_meridian_5seed.json",
        "results/figures/spherical_pendulum/spherical_pendulum_metric_dotplot_with_gsympnet_5seed.pdf",
        "results/figures/spherical_pendulum/spherical_pendulum_metric_dotplot_with_gsympnet_5seed.png",
        "results/structural_diagnostics.json",
    )
    for relative in required:
        if not (root / relative).is_file():
            failures.append(f"missing release file: {relative}")
    for relative in required_results:
        if not (root / relative).is_file():
            failures.append(f"missing primary result: {relative}")

    if args.github_ready:
        if not (root / ".git").exists():
            failures.append(
                "GitHub-ready check requested, but .git is not initialized"
            )
        lfs_check = subprocess.run(
            ["git", "lfs", "version"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if lfs_check.returncode != 0:
            failures.append(
                "Git LFS is not installed; it is required before adding *.pt files"
            )

    for relative in ("paper", "SUPPLEMENTARY_MATERIALS_INDEX.txt"):
        if (root / relative).exists():
            failures.append(f"non-reproducibility material present: {relative}")
    for system in (
        "fput",
        "toda",
        "phi4",
        "double_pendulum",
        "spherical_pendulum",
    ):
        relative = f"experiments/{system}/README.md"
        if not (root / relative).is_file():
            failures.append(f"missing experiment index: {relative}")

    if manifest.get("primary_solver_default") != {
        "implicit_max_iters": 5,
        "implicit_tol": 1e-6,
    }:
        failures.append("primary solver controls are not (5, 1e-6)")
    if manifest.get("strict_fixed_checkpoint_audit") != {
        "implicit_max_iters": 12,
        "implicit_tol": 1e-10,
    }:
        failures.append("strict audit controls are not (12, 1e-10)")
    if len(manifest.get("checkpoints", [])) != 125:
        failures.append("release manifest does not contain 125 checkpoints")

    protocol_path = root / "reproducibility" / "protocol.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        expected_state_error = {
            "periodic_difference_interval": "[-pi, pi)",
            "periodic_indices": {
                "double_pendulum": [0, 1],
                "spherical_pendulum": [1],
            },
        }
        if protocol.get("state_error") != expected_state_error:
            failures.append("protocol state-error convention is inconsistent")
        primary_control = protocol.get("primary_solver_control", {})
        if primary_control.get("initial_iterate") != "explicit_euler_predictor":
            failures.append("protocol does not specify the Euler predictor")
        if primary_control.get("batched_stopping") != "per_state_freeze":
            failures.append("protocol does not specify per-state stopping")
        if (
            protocol.get("training", {})
            .get("activations", {})
            .get("hnn_implicit")
            != "gelu"
        ):
            failures.append("protocol does not record HNN-Implicit GELU")

    pendulum_result_path = (
        root / "results" / "multi_ic_pendula_wrapped_5seed.json"
    )
    if pendulum_result_path.is_file():
        pendulum_result = json.loads(
            pendulum_result_path.read_text(encoding="utf-8")
        )
        expected_result_state_error = {
            "periodic_difference_interval": "[-pi, pi)",
            "periodic_indices": {
                "double_pendulum": [0, 1],
                "spherical_pendulum": [1],
            },
        }
        if (
            pendulum_result.get("state_error_definition")
            != expected_result_state_error
        ):
            failures.append("pendulum result uses the wrong angle convention")

    audit_path = root / "results" / "sisc_current" / "finite_solve_audit_fp12.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("map_defect_normalization") != "none":
            failures.append("finite solve map defects are not formula aligned")

    structural_path = root / "results" / "structural_diagnostics.json"
    if structural_path.is_file():
        structural = json.loads(structural_path.read_text(encoding="utf-8"))
        for system, row in structural.get("cayley", {}).items():
            if "spectral_radius_C_mean" not in row:
                failures.append(
                    f"structural indicator for {system} does not report rho(C_h)"
                )

    spherical_path = root / "results" / "spherical_pendulum_meridian_5seed.json"
    if spherical_path.is_file():
        spherical = json.loads(spherical_path.read_text(encoding="utf-8"))
        if spherical.get("section_definition", {}).get("equation") != (
            "phi - phi_initial = 0 mod 2*pi"
        ):
            failures.append("spherical result has the wrong meridian definition")

    verified = {"code": 0, "data": 0, "checkpoints": 0}
    for group in verified:
        for row in manifest[group]:
            path = root / row["path"]
            if not path.is_file():
                if group == "code" or args.require_artifacts:
                    failures.append(f"missing {group}: {row['path']}")
                continue
            if path.stat().st_size != row["bytes"]:
                failures.append(f"size mismatch: {row['path']}")
                continue
            if sha256(path) != row["sha256"]:
                failures.append(f"SHA-256 mismatch: {row['path']}")
                continue
            verified[group] += 1

    forbidden_names = {
        ".DS_Store",
        "setup_4090.sh",
        "_tmp_main_pdf_text.txt",
    }
    forbidden_directories = {".pytest_cache", "__pycache__"}
    for path in iter_release_paths(root):
        relative = path.relative_to(root)
        if path.is_file() and path.name in forbidden_names:
            failures.append(f"local-only file present: {relative}")
        if (
            path.is_dir()
            and path.name in forbidden_directories
            and any(child.is_file() for child in path.rglob("*"))
        ):
            failures.append(f"cache directory present: {relative}")
        if path.is_file() and relative.match("results/corrected_*"):
            failures.append(f"provisional corrected result present: {relative}")

    attributes_path = root / ".gitattributes"
    attributes = (
        attributes_path.read_text(encoding="utf-8")
        if attributes_path.is_file()
        else ""
    )
    tracks_pt_with_lfs = "*.pt filter=lfs" in attributes
    for path in iter_release_paths(root):
        if not path.is_file() or path.stat().st_size <= GITHUB_REGULAR_FILE_LIMIT:
            continue
        if path.suffix == ".pt" and tracks_pt_with_lfs:
            continue
        failures.append(
            "file exceeds GitHub's 100 MiB regular-file limit without a "
            f"matching LFS rule: {path.relative_to(root)}"
        )

    text_suffixes = {
        ".cff",
        ".json",
        ".md",
        ".py",
        ".tex",
        ".txt",
        ".yaml",
        ".yml",
    }
    unix_user_root = "/" + "Users" + "/"
    unix_home_root = "/" + "home" + "/"
    windows_user_root = "C:" + "\\" + "Users" + "\\"
    absolute_patterns = (
        re.compile(re.escape(unix_user_root) + r"[^/\s\"']+/"),
        re.compile(re.escape(unix_home_root) + r"[^/\s\"']+/"),
        re.compile(
            re.escape(windows_user_root) + r"[^\\\s\"']+\\",
            flags=re.IGNORECASE,
        ),
    )
    for path in iter_release_paths(root):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        for pattern in absolute_patterns:
            match = pattern.search(content)
            if match:
                failures.append(
                    "absolute user path in "
                    f"{path.relative_to(root)}: {match.group(0)}"
                )
                break

    for path in (root / "results").rglob("*.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"invalid result JSON {path.relative_to(root)}: {exc}")
            continue
        for relative in set(iter_pt_references(result)):
            if not (root / relative).is_file():
                failures.append(
                    f"missing artifact referenced by {path.relative_to(root)}: "
                    f"{relative}"
                )

    print(
        "verified "
        + ", ".join(f"{group}={count}" for group, count in verified.items())
    )
    if failures:
        print("FAIL")
        for failure in sorted(set(failures)):
            print(f"- {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
