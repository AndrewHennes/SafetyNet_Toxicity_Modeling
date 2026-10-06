# Repository Audit Report

Historical record from February 9, 2026. Paths, counts, and pending changes describe that snapshot and are not current setup instructions. See `readme.md` and `docs/development.md` for the maintained workflows.

Date: 2026-02-09
Repo: `/home/zhangyu/SafetyNet`
Branch: `main`

## 1) Is local repo up to date with remote?

- Remote fetch completed successfully: `git fetch --prune --tags origin`
- Upstream tracking: `main` tracks `origin/main`
- Ahead/behind check: `git rev-list --left-right --count origin/main...main` => `0  0`

Conclusion: local `main` is up to date with `origin/main` at commit `f0265cf`.

## 2) Current local diff / uncommitted state

`git status --porcelain=v1 -b` shows:
- `M .gitignore`

No other modified/staged files were detected.

### `.gitignore` diff summary

The local `.gitignore` was substantially rewritten compared to HEAD:
- Added broad OS/editor ignores (`.DS_Store`, `.vscode/`, `.idea/`, swap files)
- Added Python artifact ignores (`__pycache__/`, `*.py[cod]`, caches)
- Added environment ignores under `envs/`
- Keeps `data/`, `models/`, `outputs/` as ignored-by-default with `.gitkeep` exceptions
- Added explicit ignore for `*.pt`, `*.ckpt`, temp scoring outputs

## 3) Large file handling assessment

### 3.1 LFS configuration

`.gitattributes` contains:
- `*.pt filter=lfs diff=lfs merge=lfs -text`
- `*.ckpt filter=lfs diff=lfs merge=lfs -text`

`git lfs ls-files` output is empty.
- No currently tracked Git LFS files.

### 3.2 Current tracked file sizes (HEAD + working tree)

Top tracked file sizes are small; largest is:
- `assets/safetynet_banner.png` (~1.23 MB)

No currently tracked file >= 5 MB was found.

### 3.3 Historical large blobs (in Git history)

Largest historical blob found:
- `data/100K_Toxicity_Data.csv` ~15.93 MB

Other historical data blobs in `data/` are in the ~1.7-6.4 MB range.

Implication: current repo tip is lightweight, but history includes medium-size data artifacts.

## 4) Repository integrity and size

- `git fsck --full`: no integrity errors reported.
- `.git` directory size: ~9.4 MB
- Packed object size: ~9.21 MiB (`git count-objects -vH`)
