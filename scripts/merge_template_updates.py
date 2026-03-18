#!/usr/bin/env python3
"""Merge latest template updates into local repos.

Template source:
  git@github.com:RayanZaki/Research_experiment_env.git

Examples:
  python scripts/merge_template_updates.py /path/to/local/repo
  python scripts/merge_template_updates.py /repo1 /repo2 --branch main
  python scripts/merge_template_updates.py /repo --push
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

TEMPLATE_REPO_URL = "git@github.com:RayanZaki/Research_experiment_env.git"
DEFAULT_UPSTREAM_REMOTE = "template-upstream"
DEFAULT_BRANCH = "main"


def run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=check)


def is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def ensure_remote(repo: Path, remote_name: str, remote_url: str) -> None:
    remotes = run(["git", "remote", "-v"], cwd=repo).stdout.strip().splitlines()
    existing = [line for line in remotes if line.startswith(f"{remote_name}\t")]

    if not existing:
        run(["git", "remote", "add", remote_name, remote_url], cwd=repo)
        return

    # Remote exists: make sure URL is correct
    url = existing[0].split()[1]
    if url != remote_url:
        run(["git", "remote", "set-url", remote_name, remote_url], cwd=repo)


def current_branch(repo: Path) -> str:
    return run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).stdout.strip()


def ref_exists(repo: Path, ref: str) -> bool:
    result = run(["git", "rev-parse", "--verify", "--quiet", ref], cwd=repo, check=False)
    return result.returncode == 0


def resolve_merge_target(repo: Path, upstream_remote: str, branch: str) -> tuple[str, str | None]:
    preferred_ref = f"refs/remotes/{upstream_remote}/{branch}"
    if ref_exists(repo, preferred_ref):
        return f"{upstream_remote}/{branch}", None

    # Fall back to remote HEAD branch (e.g. master/main) when requested branch is absent.
    head = run(
        ["git", "symbolic-ref", "--quiet", "--short", f"refs/remotes/{upstream_remote}/HEAD"],
        cwd=repo,
        check=False,
    ).stdout.strip()
    if head:
        fallback_target = head
        fallback_branch = fallback_target.split("/", maxsplit=1)[1] if "/" in fallback_target else fallback_target
        warning = (
            f"Requested branch '{branch}' not found on {upstream_remote}; "
            f"falling back to remote HEAD branch '{fallback_branch}'."
        )
        return fallback_target, warning

    # Last resort: pick any fetched remote branch.
    refs = run(
        ["git", "for-each-ref", "--format=%(refname:short)", f"refs/remotes/{upstream_remote}"],
        cwd=repo,
        check=False,
    ).stdout.strip().splitlines()
    candidates = [r for r in refs if not r.endswith("/HEAD")]
    if candidates:
        warning = (
            f"Requested branch '{branch}' not found on {upstream_remote}; "
            f"falling back to '{candidates[0]}'."
        )
        return candidates[0], warning

    raise RuntimeError(f"No mergeable branches found on remote '{upstream_remote}'.")


def has_uncommitted_changes(repo: Path) -> bool:
    return bool(run(["git", "status", "--porcelain"], cwd=repo).stdout.strip())


def create_backup_branch(repo: Path) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"backup/template-sync-{stamp}"
    run(["git", "branch", name], cwd=repo)
    return name


def stash_push(repo: Path) -> bool:
    if not has_uncommitted_changes(repo):
        return False
    run(["git", "stash", "push", "-u", "-m", "pre-template-sync"], cwd=repo)
    return True


def stash_pop(repo: Path) -> None:
    # If nothing to pop, do nothing.
    stashes = run(["git", "stash", "list"], cwd=repo).stdout.strip()
    if not stashes:
        return
    run(["git", "stash", "pop"], cwd=repo, check=False)


def is_unrelated_histories_error(result: subprocess.CompletedProcess) -> bool:
    combined = f"{result.stdout}\n{result.stderr}".lower()
    return "refusing to merge unrelated histories" in combined


def merge_template(repo: Path, upstream_remote: str, branch: str, push: bool, allow_unrelated_histories: bool) -> int:
    print(f"\n=== Syncing: {repo} ===")

    if not is_git_repo(repo):
        print("[skip] Not a git repo")
        return 1

    try:
        ensure_remote(repo, upstream_remote, TEMPLATE_REPO_URL)

        branch_name = current_branch(repo)
        print(f"[info] Current branch: {branch_name}")
        backup = create_backup_branch(repo)
        print(f"[info] Backup branch created: {backup}")

        stashed = stash_push(repo)
        if stashed:
            print("[info] Working tree stashed")

        run(["git", "fetch", "--prune", upstream_remote], cwd=repo)

        merge_target, warning = resolve_merge_target(repo, upstream_remote, branch)
        if warning:
            print(f"[warn] {warning}")

        merge_cmd = ["git", "merge", "--no-ff", "-m", f"Merge template updates from {merge_target}", merge_target]
        result = run(merge_cmd, cwd=repo, check=False)

        if result.returncode != 0 and allow_unrelated_histories and is_unrelated_histories_error(result):
            print("[warn] Unrelated histories detected; retrying with --allow-unrelated-histories")
            result = run([*merge_cmd, "--allow-unrelated-histories"], cwd=repo, check=False)

        if result.returncode != 0:
            print("[error] Merge conflict or failure")
            print(result.stdout)
            print(result.stderr)
            print(f"[hint] Resolve conflicts, then run: git add -A && git commit")
            if stashed:
                print("[hint] Your local changes are still stashed (run: git stash list)")
            print(f"[hint] Restore previous state from backup branch: {backup}")
            return 2

        print("[ok] Template updates merged")

        if stashed:
            stash_pop(repo)
            print("[info] Stash reapplied (check for conflicts)")

        if push:
            run(["git", "push"], cwd=repo)
            print("[ok] Pushed to origin")

        return 0

    except subprocess.CalledProcessError as e:
        print("[error] Command failed:", " ".join(e.cmd))
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return 3
    except RuntimeError as e:
        print(f"[error] {e}")
        return 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge latest template changes into one or more local repos.")
    parser.add_argument(
        "repos",
        nargs="+",
        help="Path(s) to local repository roots",
    )
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help=f"Template branch to merge (default: {DEFAULT_BRANCH})")
    parser.add_argument(
        "--remote",
        default=DEFAULT_UPSTREAM_REMOTE,
        help=f"Upstream remote name to use/create (default: {DEFAULT_UPSTREAM_REMOTE})",
    )
    parser.add_argument(
        "--allow-unrelated-histories",
        action="store_true",
        default=True,
        help="Allow merging repositories with unrelated histories (default: enabled)",
    )
    parser.add_argument(
        "--no-allow-unrelated-histories",
        dest="allow_unrelated_histories",
        action="store_false",
        help="Disable unrelated-histories merge retry",
    )
    parser.add_argument("--push", action="store_true", help="Push after successful merge")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    codes = []
    for repo in args.repos:
        codes.append(
            merge_template(
                Path(repo).resolve(),
                args.remote,
                args.branch,
                args.push,
                args.allow_unrelated_histories,
            )
        )
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
