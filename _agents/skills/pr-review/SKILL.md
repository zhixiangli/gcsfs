---
name: pr-review
description: Use when reviewing a gcsfs pull request, local branch, or diff, when asked to "review", "review again", "review my review", evaluate another reviewer's comments, or check a PR before opening or merging.
---

# GCSFS PR Review

Review pull requests the way `fsspec/gcsfs` maintainers do: first decide whether the change should exist in this form and layer, then whether it is minimal and preserves `fsspec`/PyPI contracts, then whether it is correct, safe, and provably tested. A correct patch in the wrong layer, a workaround hiding an upstream bug, or a change that silently breaks `fsspec==2026.9.0` or Zonal/HNS/RCU invariants is a failed review if you only check syntax.

## Usage Modes

### No argument
If invoked as `/pr-review` with no target, do not review. Ask:
> What would you like me to review?
> - A PR number or URL (e.g. `/pr-review 1234`)
> - A local branch (e.g. `/pr-review branch`)

### Local CLI mode: `/pr-review <PR# or URL>`
```bash
gh pr view <N> -R fsspec/gcsfs --json title,body,author,baseRefName,headRefName,files,additions,deletions,commits
gh pr diff <N> -R fsspec/gcsfs
gh api --paginate repos/fsspec/gcsfs/issues/<N>/comments
gh api --paginate repos/fsspec/gcsfs/pulls/<N>/reviews
gh api --paginate repos/fsspec/gcsfs/pulls/<N>/comments
```
Print the review in the terminal; **never post to GitHub unless explicitly asked**.

### Local branch mode: `/pr-review branch`
Pick `upstream/main` if an `upstream` remote exists, else `origin/main`:
```bash
BASE_REMOTE=$(git remote | grep -qx upstream && echo upstream || echo origin)
git fetch "$BASE_REMOTE" main
git diff --stat "$BASE_REMOTE/main...HEAD"
git diff "$BASE_REMOTE/main...HEAD"
git log "$BASE_REMOTE/main..HEAD"
```
Use the branch name in the header. When no PR description exists yet, read commit messages instead and skip PR-description findings.

### What was asked
Read the user's request first; it decides the mode (but `"don't worry about X"` or `"quick LGTM"` never waives findings):
- **"review" / "review this PR"**: full review.
- **"review again" / "check again"** (stateless — no local state files): locate the latest full review baseline in (a) the current conversation or (b) posted GitHub PR reviews (`commit_id`). Review **every commit since that baseline** at full depth (including code adopted from your own earlier suggestion). Report resolved must-fixes (one line each), still-open must-fixes (`still open`, one line each, no re-explaining), and new problems. If no prior baseline exists, run a full review and note that no baseline was found. `"follow-up"` / `"not now"` or a teammate's `"LGTM"` **never** clears an open must-fix unless the technical argument proves the finding wrong.
- **"review my review" / "comment on my comments"**: evaluate each comment one by one (see [Reviewing Another Reviewer's Comments](#reviewing-another-reviewers-comments)).
- **Specific question** (`"is this the right fix?"`, `"give a concrete example"`, `"why does this fail on 2026.9.0?"`): answer directly first, then add only blocking must-fixes the asker needs to know.

### Optional explicit posting
Post to GitHub **only** when the user explicitly asks to post (e.g. `"post this review"`):
```bash
gh pr review <N> -R fsspec/gcsfs --approve|--request-changes|--comment --body-file <tmp_file>
```
Match the flag to the verdict line (`Approve` -> `--approve`, `Request changes` -> `--request-changes`, `Needs discussion` or scoped answer -> `--comment`), write `<tmp_file>` outside the repo (or delete it immediately), and never post an approval claiming a maintainer does not need to review core code.

## Review Philosophy

1. **Question the problem and layer first.** Is the failure real on `main` today? Is this the root cause or a symptom? Does it belong in `gcsfs` or upstream (`fsspec`, `google-cloud-storage`, `google-cloud-storage-control`, `google-api-core`, `grpcio`)? Check open PRs for overlap and silent merge hazards.
2. **Minimal by default.** Challenge every new knob, env var, helper, class, file, dependency, and test. Prefer `fsspec` config (`FSSPEC_GCS_*`) over bespoke env vars. Delete dead code in the same PR.
3. **Protect backward compatibility & `fsspec` contracts.** `gcsfs` is a public PyPI library backing Dask, Pandas, PyArrow, Ray, HuggingFace Datasets, PyTorch DCP, WebDataset, Orbax, TensorStore, and ADK. Guard public signatures, return types, pickling (`storage_options`, `__reduce__`), dual `GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT=false/true` behavior, and the `fsspec>=2026.9.0` floor (including CI-on-HEAD skew).
4. **Fail closed on security, auth, CMEK, and layout.** Never silently downgrade auth, TLS, `kms_key_name`, consistency checks, or bucket layout. Enforce [_agents/rules/security.md](../../rules/security.md) inline without alarmist badges or emojis.
5. **Make a human understand it.** Explain the change with a minimal, concrete before/after `Example` (inputs, old behavior, new behavior). Verify PR title and description match HEAD (they become the squash commit message).
6. **Only report problems.** Zero praise, zero empathy filler, zero diff restatement. Every finding names `file:line`, the problem, its concrete consequence, and the smallest valid fix.
7. **Investigate, don't guess.** Read full surrounding functions and callers. Verify upstream behavior in source before claiming how external libraries work. Label anything unverified (or requiring live GCS) under `Questions` or `Not verified`, never `Must fix`.
8. **Skip what CI checks.** Don't report `isort`, `black`, `flake8`, or raw test pass/fail. **Do** report CI/test-wiring gaps (e.g., perf tests not wired into `ci.yml --run-benchmarks-infra`, silent no-op markers, or Codecov 80% patch gaps).
9. **Explicit errors over silent fallbacks.** Raise clear, actionable errors naming what the caller should do instead rather than silently ignoring options or misrouting layouts.
10. **Tests must prove the fix and earn their cost.** Ask whether reverting the PR's core line actually breaks the test (mutation resistance). Do not append generic `"add tests"` wishlists; ask for a test only when a concrete invariant is unguarded after the PR passes the gate.

## Project Context & Reference Routing

Load only the references matching files touched by the diff:

| Touched surface | Load these references |
|---|---|
| **Always** | [review-checklist.md](review-checklist.md), [security.md](../../rules/security.md) |
| `gcsfs/__init__.py`, `core.py`, `retry.py`, `credentials.py`, `checkers.py`, `utils.py` | [gcsfs-invariants.md](references/gcsfs-invariants.md) §1–§3, §9 |
| `gcsfs/_dircache.py` | [gcsfs-invariants.md](references/gcsfs-invariants.md) §4 |
| `gcsfs/extended_gcsfs.py`, `poller.py` | [gcsfs-invariants.md](references/gcsfs-invariants.md) §1, §5–§8 |
| `gcsfs/zb_hns_utils.py`, `zonal_file.py`, `caching.py`, `parallel_upload.py` | [gcsfs-invariants.md](references/gcsfs-invariants.md) §7–§9 |
| `gcsfs/telemetry/**` | [gcsfs-invariants.md](references/gcsfs-invariants.md) §10 |
| `gcsfs/tests/**`, `.github/**`, `cloudbuild/**`, `pyproject.toml`, `environment_gcsfs.yaml` | [gcsfs-invariants.md](references/gcsfs-invariants.md) §11 |
| `fsspec` base overrides, `google-cloud-storage`, `storage-control`, `google-api-core`, `grpc`, dep files | [upstream-contracts.md](references/upstream-contracts.md) (matching library section + §5) |
| Auth/tokens, REST URLs, local writes, TLS/mTLS, `ctypes`, pickling, `.github/`, `cloudbuild/` | [gcsfs_threat_surface.md](../security-audit/references/gcsfs_threat_surface.md) |

## Review Workflow

### Step 1: Context
1. Read PR title, description, linked issues, commits, and full diff stat (files, net +/- LoC). Group hunks: logic, mechanical, tests, benchmarks/CI, docs/changelog.
2. Identify the reference sections from the routing table above and load them.
3. **Human maintainer comments**: identify reviewers with GitHub `author_association in {"OWNER", "MEMBER", "COLLABORATOR"}` excluding `[bot]` accounts. For each comment, verify at HEAD whether it is addressed (fixed in code or answered with no pushback):
   - Unaddressed blocking comment (confirmed valid in Step 5) -> `Must fix` naming `@reviewer`.
   - Non-blocking nit -> `Should consider`.
   - Proven factually mistaken against code at HEAD in Step 5 -> **never** put in `Must fix`; note the discrepancy concisely under `Questions` with `file:line` evidence.
4. **Re-review baseline**: read prior review + author replies. Weigh replies on technical merits only (`"follow-up"` / `"not now"` / teammate `"LGTM"` do not clear must-fixes).
5. **Open-PR overlap check (always)**: run `gh pr list -R fsspec/gcsfs --state open --search "<term>"` for the feature name and each main modified file (also check recent `main` commits for newly introduced helpers). Record search terms and matches in `<details>` (or note `"gh unavailable / offline"` if offline). Flag overlapping rewrites or silent non-conflicting merge hazards.

### Step 2: Necessity & Design Gate
Complete this **before** line-level review. Answer with evidence:
1. **Problem**: what concrete call/input fails on `main` today? If neither the description nor the code shows one, that is the lead finding.
2. **Root cause & layer**: cause or symptom? Should this live in `fsspec`, `google-cloud-storage`, `google-cloud-storage-control`, or `grpcio`? Private/experimental upstream calls require a `TODO` + upstream issue link AND proof an upstream fix isn't cheaper.
3. **Ownership & dependencies**: should `gcsfs` own this? Are new deps justified, upper-bound-free, compliant with the `fsspec` floor policy, and synced to `environment_gcsfs.yaml`?
4. **Existing counterpart**: does `gcsfs` or `fsspec` already have this (`_dircache`, `download_range`, `poll_lro`, `_validate_grpc_write_kms_key_name`, `_classify_layout_error`, `fsspec.asyn.sync_teardown`, `tests/layouts.py`, `tests/lro_fakes.py`)?
5. **Smallest alternative**: write out the minimal design and rough LoC estimate.
6. **Value & perf evidence**: does a new public API/knob have a named user or in-repo consumer? Does a perf PR have a plausible diff mechanism and multi-run numbers beating noise?
7. **One-sided design**: check symmetry across `GCSFileSystem`/`ExtendedGcsFileSystem`, `GCSFile`/`ZonalFile`, Flat/HNS/Zonal/RCU (`rcu_rapid_write` False & True), all write entry points, and sync/async paths.

**Gate outcomes:**
- **PR shouldn't exist** (problem not real on `main`, unjustified new public API with no consumer, wrong owner, superseded, or entire diff reverts safety invariants with no valid mechanism): **stop**. Emit `Verdict: Request changes` (or `Needs discussion` for ownership/direction calls), `What this PR does`, a brief `Example` clarifying why, gate findings + alternative under `Must fix` (or `Questions`), the line `Line-level review skipped because <reason>.`, and `<details>` with open-PR search terms. Do not polish code that should be reverted or rejected.
- **Right problem, wrong approach** (symptom fix, wrong layer, private SDK monkey-patch, knob hiding a bug): lead with the gate finding and smaller alternative, run **only** the Step 3 quick sweep (placement, scope, security, mechanical `(diff check)` traps), and skip deep line-by-line review of code that will be rewritten.
- **Unjustified knob / public API** with no in-repo consumer and no named user: asking for the concrete use case is a `Must fix`, not a `Question`.
- **One-sided design** whose symmetric version removes workaround machinery or closes sibling-path bugs: lead `Must fix` (`Request changes`, or `Needs discussion` if symmetry has a real trade-off). Never downgrade to `Should consider` or a TODO.

### Step 3: Deep Review
1. **Quick sweep + mechanical pass (always)**: inspect every hunk for drive-by edits, wrong file placement, new knobs/env vars, and **actively search added lines for every `(diff check)` item** in [review-checklist.md](review-checklist.md). Check every new/changed function, class, param, and docstring.
2. **Small PR (< ~200 changed lines of logic)**: review directly against the loaded references.
3. **Large PR (>= ~200 changed lines of logic)**: spawn 2–4 read-only subagents in parallel split by **lens** (not directory), passing diff paths, checklist sections, and routed references:
   - **Design, API & Backward Compatibility**: `fsspec` signatures/return types, dual `GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT` modes, pickling, minimalism, config parsing, placement, one-sidedness.
   - **Correctness, Concurrency & Upstream Contracts**: all 7 `BucketLayout`s + `UNKNOWN`, stale `persisted_size`, `MRDPool` health (`is_stream_open`), cancellation/teardown, `fork()`, `dircache`, telemetry, hot-path perf, upstream contracts.
   - **Security, Buffers & Transport**: [security.md](../../rules/security.md), URL quoting, credential redaction, TLS/mTLS/anonymous gRPC, CMEK fail-closed, `ctypes`/`DirectMemmoveBuffer`, local path containment, CI supply chain.
   - **Tests, CI, Docs & Benchmarks**: mutation resistance, silent-marker / real-gRPC traps, `--run-benchmarks-infra`, Cloud Build lockstep, Codecov 80% patch gaps, docs/changelog/PR-description accuracy.
   Subagents return `(file:line, claim, evidence)` only and are strictly forbidden from running `gh pr review` or posting anything to GitHub.

### Step 4: Consolidate
Flatten candidates to `(file:line, one-line claim)` and deduplicate before writing:
- Merge findings sharing a root cause, fix, or `file:line` (unless independent defects).
- Drop lint/formatting CI checks, generic `"add tests"` requests, and praise.
- Rank strictly: **`Must fix`** includes every violated written rule or security guardrail, every mechanical `(diff check)` safety/zero-copy/teardown defect (e.g., missing `**_download_ranges_kwargs`, `if_generation_match=0` drop, `DataCorruption` dead-MRD recycle, `fsspec==2026.9.0` crash), silent wrong results / data loss / leaks / hangs, unaddressed valid maintainer comments, duplicated concepts, gate failures, and misleading PR descriptions. Never soften a must-fix into `Should consider`.

### Step 5: Fact-Check (Findings AND Proposed Fixes)
1. **Verify every consolidated `Must fix`** by re-reading full surrounding functions and callers at HEAD. Drop any invalid finding.
2. **Verify upstream claims using the 3-tier hierarchy**:
   - **Tier 1 (preferred)**: inspect live installed source (`python -c "import inspect, ..."` per [upstream-contracts.md](references/upstream-contracts.md#live-source-verification-commands)) or a local sibling checkout if present.
   - **Tier 2**: cite version-tagged pre-verified entries in [upstream-contracts.md](references/upstream-contracts.md) when `inspect` cannot reach compiled C/C++/Cython internals (`grpcio` C-core) or when auditing floor-vs-HEAD version skew (`fsspec==2026.9.0` vs CI HEAD), provided the version range still applies (re-check HEAD-only notes before flagging).
   - **Tier 3**: anything unconfirmed by Tier 1 or Tier 2 -> mark *unverified* and move to `Questions` or `Not verified`, **never** `Must fix`.
3. **Verify every proposed fix against `gcsfs` guardrails**: check that your suggested fix does not violate any rule (no fixed `total` timeout on `aiohttp` transfers, no runtime dep upper bound, no private SDK monkey-patch without `TODO` + upstream issue link, no mutating shared `ctypes.pythonapi`, no deleting `bind_gcs_sync_methods`, no unquoted REST URL, no bumping `fsspec` lower bound without flagging it as a maintainer decision).

### Step 6: Write (and Post only if explicitly requested)
Render using the Output Format below. Keep the visible body concise (**target <= ~400 words**); move long derivations, gate walkthroughs, open-PR search logs, and alternatives into `<details><summary>Details</summary>`.

## Reviewing Another Reviewer's Comments

For each comment you are asked to evaluate:
1. State **`correct`**, **`partly correct`**, or **`mistaken`** with `file:line` evidence from the code at HEAD (not the PR description).
2. If `correct` or `partly correct`, give the smallest concrete fix and note whether the author's reply actually resolves it.
3. If `mistaken`, state why with code evidence — never rubber-stamp a mistaken comment into `Must fix`.
4. Add only blocking must-fix problems the reviewer missed.

## Output Format

Follow this exact structure. Keep the visible section (everything outside `<details>`) under **~400 words** for a typical PR. Omit empty sections, except `Example` (always include unless trivial typo/1-line config) and `Not verified` (always include when live GCS / external behavior could not be exercised).

```markdown
**Verdict: Approve | Request changes | Needs discussion** -- <one sentence: decisive reason>

**What this PR does.** <2-3 plain sentences.>

**Example.**
<5-15 lines: minimal concrete input/call, old behavior on main vs. new behavior with this PR.>

### Must fix
1. `path/file.py:123` -- <problem>. <consequence>. <smallest valid fix>.
2. ...

### Should consider
- `path/file.py:45` -- <problem and suggestion>.

### Questions
- <questions only the author or a maintainer can answer, including any mistaken maintainer comment with file:line evidence>

<details><summary>Details</summary>

- **Gate & alternatives**: <necessity/layer check, smaller alternative + LoC estimate>
- **Open-PR overlap**: <searched terms and matching PRs, or "none" / "gh unavailable">
- **Supporting analysis**: <deeper code/upstream citations, test mutation notes>

</details>

Not verified: <one concise line naming live GCS layouts (HNS/Zonal/RCU/mTLS/Windows GKE) or unrun checks>.
```

On a re-review (`"review again"`), include one-line status bullets right above or inside `Must fix`:
- `Resolved: <item> (<commit/file>)`
- `Still open: <file:line> -- <one-line reminder>`

### Mechanical Verdict Decision Table

Derive the verdict mechanically from the consolidated findings — never from general vibes or confidence:

| Verdict | When to emit |
|---|---|
| **Approve** | Step 2 gate passed (including symmetry), every commit since the last full review was reviewed, no blocking `Questions` or unverified critical dependency assumptions remain, AND `Must fix` is either empty or contains **only** mechanical edits spelled out verbatim needing no second look (rename, delete, move, add TODO/comment, spelled-out `ValueError`). Routine `Not verified` items covered by Cloud Build E2E (live HNS/Zonal/RCU/mTLS/Windows-GKE) do **not** block `Approve` when code and local mocked/emulator tests are sound — list them under `Not verified`. If mechanical must-fixes remain, start the line with `Approve -- please address the items below before merging`. |
| **Request changes** | Author owes work needing another look: any Step 2 gate failure (problem not real, wrong layer, unjustified API/knob, one-sided workaround), any correctness/security/teardown/BC bug, duplicated concept, non-mechanical rule violation, missing required perf evidence, misleading PR description, or unaddressed valid blocking maintainer comment. |
| **Needs discussion** | What blocks merge is a maintainer decision rather than an author bug: direction/ownership/scope call, BC break, bumping the `fsspec` lower bound, adding a runtime dependency, coordinating with an overlapping open PR, or an unverified correctness-critical assumption. Non-blocking `Questions` (e.g., asking for a number on a small sound change, or noting a factually mistaken maintainer comment) do **not** force `Needs discussion`. |

Only a code change or a technical argument proving a finding wrong clears a must-fix — never `"follow-up"`, `"not now"`, or another reviewer's `"LGTM"`.

## Common Review Pitfalls (Red Flags)

| Temptation / Rationalization | Reality & Required Action |
|---|---|
| *"Let me give helpful line-by-line fixes and a test wishlist for this speculative feature/broken diff."* | Run the Step 2 Gate first. If the PR shouldn't exist (no failing `main` case, no consumer, or reverts safety invariants with no valid mechanism), stop and issue the gate verdict. |
| *"CI is green and release is in an hour — quick LGTM."* | CI tests `fsspec` git HEAD (hiding `2026.9.0` crashes) and may skip mocked/marker tests (`0 failed, 1 skipped`). Verify against `2026.9.0` and run every `(diff check)`. |
| *"Author said 'follow-up PR' and a teammate commented 'LGTM'."* | Neither clears a blocking must-fix (e.g., `if_generation_match=0` silent overwrite). Keep it under `Still open` and emit `Request changes`. |
| *"Missing `**_download_ranges_kwargs` is just a perf nit — put it in Should consider."* | Missing a mandatory `(diff check)` invariant (zero-copy kwargs, `is_stream_open` check after `DataCorruption`, `length == 0` guard) is a `Must fix`, never `Should consider`. |
| *"I'll suggest overwriting private `AsyncGrpcClient._grpc_client` as a quick workaround."* | Step 5 fact-checks proposed fixes too: flag upstream ownership first, and never recommend a private SDK monkey-patch without a `TODO` + upstream issue link and proof upstream isn't cheaper. |
| *"I can't run live HNS/Zonal GCS locally, so I must downgrade a sound PR to `Needs discussion`."* | Routine Cloud Build E2E coverage belongs in `Not verified` and does **not** block `Approve` when code and local tests are sound. |
| *"A `COLLABORATOR` left an unaddressed comment, so it must go in `Must fix`."* | Verify every reviewer comment against code at HEAD in Step 5. A factually mistaken comment goes under `Questions` with `file:line` evidence, never `Must fix`. |
| *"Let me include every code block and derivation in the main review body."* | Keep visible output <= ~400 words with a concrete `Example`; put deep walkthroughs, open-PR search logs, and alternatives inside `<details>`. |
