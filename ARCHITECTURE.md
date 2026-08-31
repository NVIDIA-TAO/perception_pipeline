# Architecture

How the pipeline is put together and why. For installing, configuring and running it, see
[README.md](README.md). These two files are the whole of the documentation; everything else is a
comment beside the code it explains.

**Contents:** [Goal](#goal) · [Stage flow](#stage-flow) · [Layout](#layout) · [Environments](#the-environments) ·
[Key definitions](#where-the-important-definitions-live) · [Detection metrics](#detection-metrics) · [Pose metrics](#pose-error-metrics) ·
[Refinement](#optional-cad-box-prompt-refinement) · [Reranking](#optional-reranking) ·
[Outputs](#outputs) · [Defaults](#effective-defaults)

---

## Goal

Measure how accurately an off-the-shelf perception stack localizes known rigid parts on a work
surface, to millimetre tolerances, on real captures.

Nothing here is trained. SAM3, FoundationStereo and FoundationPose are obtained separately and
run as published; this repository composes the three into one fixed stage order, and scores what 
comes back. The contribution is the composition and the measurement, not the models.

Inference and scoring are [two passes](#inference-and-evaluation-are-two-passes-not-one-loop). 
A finished run can be re-scored at a different threshold, visibility band or rerank cutoff without 
loading a model.

---

## Stage flow

Per scene the pipeline predicts depth from the stereo pair, builds a SAM3 scene state at that
depth map's resolution, runs text-prompted segmentation per target, registers each proposal with
FoundationPose, optionally refines and reranks the proposal set, and scores everything against
occlusion-aware ground truth.

```text
RGB + stereo pair ──► depth (predicted)
                          │
                          ▼
        SAM3 scene embed (sized from the depth map)
                          │
                          ▼
        SAM3 text proposals ──► FoundationPose
                                        ──► [optional CAD box-prompt SAM3 refinement]
                                        ──► FoundationPose again on refined masks
                                        ──► [optional object-agnostic rerank]
                                        ──► (separately, evaluate.py) detection + pose metrics,
                                            CSV/JSON/Markdown report
```

SAM3's scene embed is sized from the depth map: `infer.py` derives `depth_image_size` from
`depth_m.shape` and passes it to `pose_estimator.begin_scene(image, depth_image_size)`. Depth
must therefore exist before SAM3 runs. SAM3 sees depth only through its **shape**, never its
values, which is why swapping depth engines moves pose numbers but leaves detection counts
untouched.

Both bracketed stages are **on by default**: refinement defaults to `replace_mid_nms06` and
proposal selection to `soft_global_v1`, which is the configuration every measured value in
`config/defaults.yaml` was tuned against. To measure the unrefined, unreranked path, disable them
with `--sam3-refinement-policy none --proposal-selection-policy all`.

### Inference and evaluation are two passes, not one loop

The flow above is cut in half, and the cut is the most important structural decision after the
environments:

```text
script/infer.py       images + intrinsics ──► depth/, predictions.jsonl, inference_config.json
                      no ground truth read, no metric computed
script/evaluate.py    predictions.jsonl   ──► detection/pose/depth summaries, report.md,
                                              evaluations.jsonl
script/run_pipeline.py   runs both, in one process, against one output directory
```

Three things this buys, each of which the old interleaved loop made impossible:

- **A capture with no annotations can be processed at all.** The one thing inference legitimately
  needs that usually comes from ground truth is *which objects to look for* — a task
  specification, not a measurement — so `infer.py --objects` supplies it directly. Without that
  flag it falls back to reading the class names out of `scene_gt.json`, which is a convenience on
  an annotated capture: the per-instance poses and counts there are never consulted, so no answer
  can leak into inference.
- **Re-scoring is free.** `predictions.jsonl` keeps *every* proposal with its scores, not only the
  survivors, so a different IoU threshold, visibility band or rerank cutoff is arithmetic over a
  file rather than hours of SAM3 and FoundationPose.
- **The runtime figure is honest.** Inference completes before scoring starts, so SAM3 and
  FoundationPose are released before ground-truth rasterization begins and the "production path"
  number in `report.md` contains nothing that would not exist at deployment.

The boundary is structural, and checked: `perception_pipeline.evaluation` is where everything
that touches ground truth lives, and nothing in `perception_pipeline.inference` may reach it —
directly or transitively. A static import-graph check fails the build if anything does; see
[Layout](#layout).

---

## Layout

```text
pipeline/
  config/                    all YAML config — README.md section 5
    <name>/object_names.json   obj_id -> object name, per dataset (the adapter reads it)
    defaults.yaml            algorithm defaults (thresholds, rerank, refinement, depth)
    <dataset>.yaml           dataset profile: paths, prompts, regression datasets
    example_bop.yaml         annotated template to copy

  src/perception_pipeline/   the importable package
    config.py                YAML loading, profile resolution, typed settings
    geometry.py              PLY I/O, both rasterizers, IoU, matching
    dataset.py               targets and prompts — no ground truth is rendered here, see
                             evaluation/gt.py, because inference/ imports this module
    pose.py                  FoundationPose estimator lifecycle + CAD render
    runtime.py               torch/device helpers
    visualize.py             overlay drawing

    io/                      the ONLY code that knows what a BOP directory looks like
      bop.py                   a scene directory -> SceneInput / ObjectSpec
      files.py                 the small JSON files a dataset ships, incl. load_dataset_map

    extensions.py            loads whatever registers extra backends/sources — see below

    inference/               THE DEPLOYABLE PATH — must not import evaluation/
      types.py                 what inference consumes and produces (StagePredictions, …)
      config.py                per-stage configuration
      engine.py                one target: proposals -> poses -> refinement -> selection
      detect.py                text-prompted proposals from SAM3
      pose.py                  FoundationPose over a set of proposal masks
      refine.py                CAD box-prompt replacement of mid-quality proposals
      select.py                scoring and the rerank decision
      depth.py                 depth-BACKEND registry + the engine backend
      source.py                depth-SOURCE registry + the predicted source
      stereo/                  FoundationStereo depth — TAO Deploy, in this venv
        tao.py                   engine load, ImageNet preprocessing, disparity
        rectify.py               pair selection + rectification
        build.py                 ONNX -> TensorRT engine, cache key + sidecar
        depth.py                 scene -> metric depth in the base camera's frame

    evaluation/              EVERYTHING THAT NEEDS GROUND TRUTH — never runs at deployment
      gt.py                    GT rasterization and the per-scene z-buffer cache —
                               GroundTruthRenderer, render_gt_entries, precompute_dataset
      detection.py             IoU matrices, detection accumulation, PoseMetricRegistry
      pose_error.py            pose error for matched predictions
      depth_error.py           predicted depth vs the collected map
      report.py                CSV + Markdown writers, single- and multi-dataset

  script/                    entry points
    infer.py                 inference only — no ground truth read, no metric computed
    evaluate.py              scoring only — no model loaded
    run_pipeline.py          one dataset, end to end (orchestrates the two above)
    run_batch_eval.py        every dataset + aggregate report
    build_gt_cache.py        precompute the GT z-buffer cache

  test/                      the check that runs a model: depth on one scene, through the engine
  tools/                     standalone CLIs: engine building, install verification, the rerank
                             sweep, and the BOP dataset adapters under bop_adapt/
```

Dependencies run one way, with no cycles:

```text
                          ┌──►  inference/  ──┐
geometry / io  ──►  dataset                    ├──►  script/
                          └──►  evaluation/  ──┘
```

`geometry` imports nothing from the package and `io.files` imports nothing at all — they own the
primitives every layer above needs. `load_dataset_map` lives in `io.files` rather than in
`dataset` precisely so that `dataset` and `evaluation.detection` can both use it without either
importing the other.

The two middle branches are **siblings, not a chain**. `inference` reaching `evaluation` is the
one edge that must never appear — **directly or through any number of hops** — and a static
check enforces it: it walks the package's import graph from the inference roots
(every `inference/` module, plus `script/infer.py`) and fails, printing the offending chain, if
any path leads into `evaluation`. Transitive coverage is the part that matters. A direct import
inside `inference/` is easy to catch in review; the one that actually happens is a hop away —
`inference/` uses a helper from `dataset.py`, and months later someone adds a GT import to
`dataset.py`. Neither site looks wrong on its own, and the boundary is gone.

The same check also fails the build on any file that does not compile, any undefined name, and
any unsanctioned cross-module duplicate definition.

---

## The environments

Everything the pipeline runs lives in one venv.

### One environment

| | pipeline venv |
|---|---|
| Python | 3.12 |
| numpy | 1.26 (`sam3` pins `<2`) |
| opencv | 4.11 |
| torch | 2.10 + cu128 |
| Declared in | `pyproject.toml` / `uv.lock` |

---

## Where the important definitions live

- **Ground truth** — `evaluation.gt.render_gt_entries`. Decides which instances are scored and which
  pixels each owns. GT masks are occlusion-aware (all scene objects rasterized into one shared
  z-buffer); instances below `ground_truth.min_visible_fraction` (default 0.1, the BOP
  convention) are excluded from both detection and pose scoring. Every metric traces back here.
- **Pose error** — `evaluation.detection.PoseMetricRegistry.max_vertex_error_mm`. Symmetry-aware
  maximum-vertex distance (BOP's MSSD), computed in 3D over full CAD vertex sets. It never reads
  a mask, so occlusion does not affect it.
- **Two rasterizers, different jobs** — `geometry.render_mask` draws one object with no depth
  buffer (*amodal*, includes hidden pixels); `geometry.rasterize_mesh` writes into a shared
  z-buffer (*modal*, visible region only). GT must use the second.

The z-buffer cache in `evaluation.gt` exists because `render_gt_entries` rebuilds that shared
z-buffer per target at ~12–14 s each. Precomputing it once per scene — `script/build_gt_cache.py`
— is the difference between a sweep being feasible and not.

---

## Detection metrics

Whether an instance was found at all, scored before the pose metrics and reported independently
of them. Every stage writes `<stage>_detection_summary.{json,csv}` holding `tp`, `fp`, `fn`,
`precision` and `recall`, broken out by mask IoU and box IoU at each configured threshold
(`0.5` and `0.75` by default), `overall` and `by_object`.

Three distinct stages are scored, written under four filenames:

| Stage | What it contains |
|---|---|
| `raw` | Every SAM3 text proposal, before anything filters it |
| `pose_input` | After a first FoundationPose pass and the optional CAD box-prompt refinement — the proposal set the *second* pose pass scores |
| `filtered` | The subset the optional rerank kept: the final prediction set |
| `selected` | A copy of `filtered` under a second name, not a further stage. Always byte-identical to it |

Read them as a funnel. Refinement replaces proposals and the rerank only removes them, so recall
can fall from `raw` to `filtered` while precision rises; that trade is what the rerank cutoff
buys, and both numbers are needed to see it. `raw` → `pose_input` is what refinement did;
`pose_input` → `filtered` is what selection cost.

The two IoU thresholds answer different questions: `0.5` asks whether the right instance was
found, `0.75` whether its extent is tight enough to trust downstream.

Where the pose metrics below are computed only over *matched* predictions — those that produced a
pose and hit a ground-truth instance at `pose_match_threshold` selected-mask IoU — these cover the
whole prediction set, including the ones no pose was scored for. A run can look excellent on pose
and poor on recall; the two are answering different questions and neither substitutes for the
other.

---

## Pose error metrics

There is no single number for "a good pose" — the tolerance a part needs is a property of the
application, not of this pipeline. So every matched prediction is scored several ways and none of
them is gated here. Read the ones your task cares about and set your own bar.

**Max-vertex error** is the strictest of them, and the one the rest of this document quotes. For
each matched prediction it is the largest distance between any object vertex under the
ground-truth pose and the same vertex under the predicted pose; for symmetric objects, the
worst-case nearest-vertex gap in either direction. It is deliberately pessimistic: a rotation
about an object's centre moves the centroid not at all and its corners a long way, so it catches
errors a translation-only measure cannot see.

All of the following land in `pose_summary.{json,csv}` and the pose table in `report.md`:

| Field | What it measures |
|---|---|
| `translation_error_mm_{mean,median}` | Distance between ground-truth and predicted object centres |
| `rotation_error_deg_{mean,median}` | Geodesic angle between ground-truth and predicted orientation |
| `add_or_adds_mm_{mean,median}` | Average distance over corresponding model points — ADD, or ADD-S for symmetric objects |
| `add_or_adds_diameter_frac_{mean,median}` | The same as a fraction of object diameter, so objects of different size compare directly |
| `pose_success_0p1d_rate` | Fraction of matches with ADD/ADD-S below 0.1 × diameter — the conventional BOP success criterion |
| `translation_error_le_3mm_rate`, `add_or_adds_le_3mm_rate` | Fraction within 3 mm on those two metrics |
| `max_vertex_error_mm_{mean,median,p90,p99}` | Max-vertex error, summarized four ways |
| `max_vertex_error_within_threshold_rate` | Fraction at or below `pose.max_vertex_error_threshold_mm` |
| `max_vertex_error_meets_required_rate` | Whether that fraction reached `pose.max_vertex_error_required_rate` |

**The last two are configuration, not a verdict.** `pose.max_vertex_error_threshold_mm` (default
5.0 mm) and `pose.max_vertex_error_required_rate` (default 0.90) in `config/defaults.yaml` say
what *you* want counted; the defaults are a starting point, not a claim about what is good enough.
Both the field names and the report's column labels are derived from those values, so changing
either is reflected everywhere it is reported.

For a multi-dataset sweep, `run_batch_eval.py` pools the raw matches across datasets rather than
averaging the per-dataset summaries, and `summary.json` carries the pooled figures as
`pose_by_visibility.<band>.overall.*` beside `selected_detection_summary.mask.<iou>.{precision,recall}`.
Individual datasets vary widely, so read the pooled set and the per-dataset table together.

**Always read P90 beside P99.** P99 is decided by the worst ~1% of matches, so a handful of gross
failures moves it by tens of millimetres while the bulk of the distribution does not move at all.
A healthy P90 with a large P99 means a thin catastrophic tail — usually a discrete disambiguation
failure such as a ~180° flip — not a broadly degraded run. The two failure modes need opposite
fixes, which is why both are reported.

**Visibility banding, not filtering.** `pose.min_visible_fractions` decides which matched
instances get a *pose measurement*, deliberately not which instances are ground truth. Raising
`ground_truth.min_visible_fraction` instead would delete occluded instances from GT, turning
every detection of one into a false positive — measured on an occluded capture, raising it to
0.9 more than halved precision with the prediction count unchanged. Banding leaves
detection accounting untouched, and every band comes from a single run, so adding one is free.

---

## Optional CAD box-prompt refinement

`--sam3-refinement-policy replace_mid_nms06` runs an initial FoundationPose pass, uses the
rendered CAD box as a geometric prompt back into SAM3, replaces proposals whose initial
render-mask IoU falls in `[--refinement-low-miou, --refinement-high-miou]`, applies mask NMS at
`--refinement-nms-threshold`, and reruns FoundationPose on the refined set.

The band is the point: proposals that already overlap well need no help, and ones that overlap
badly are usually the wrong object entirely, so replacing them adds noise.

**On by default.** Disable with `--sam3-refinement-policy none`.

---

## Optional reranking

`--proposal-selection-policy soft_global_v1` keeps a proposal when

```text
R  =  2 * sam_score  +  4 * render_mask_iou  +  1 * render_box_iou  -  fp_score / 100
R >= 4.19446449798917        # --rerank-cutoff
```

- `sam_score` — SAM3 proposal confidence.
- `render_mask_iou` / `render_box_iou` — overlap between the FoundationPose-rendered CAD
  silhouette and the proposal mask/box at pose resolution.
- `fp_score` — FoundationPose registration score.

The rule is object-agnostic: the same formula and cutoff apply to every proposal regardless of
object id. **The cutoff was tuned on one dataset's result set**, so treat it as an experiment
configuration rather than a robust cross-dataset default. Sweep it offline against a finished run
with `tools/sweep_rerank_cutoff.py` — no GPU, no re-running SAM3 or FoundationPose.

**On by default.** Disable with `--proposal-selection-policy all` (keep every proposal).

---

### Symmetric objects: ADD-S, and what selects it

A pose metric has to know whether an object is symmetric, because for one that is, a rotation onto
itself is not an error. ADD is used for asymmetric objects and ADD-S for symmetric ones, and the
same distinction sets the max-vertex error: nearest-vertex and bidirectional when symmetric,
per-vertex when not. The gap between them is roughly one object diameter for a 180-degree flip, so
the choice is not bookkeeping.

**The declaration selects it, not the geometry.** `models_info.json` is read and its NON-IDENTITY
transformations counted — an identity says nothing about the shape, and whether one is listed is a
property of the writer rather than of the object. That reads both conventions in circulation the
same way: formats that store only the real transformations, and formats that store the identity
first. An object declaring nothing is scored asymmetric, which is the declaration taken at its
word rather than a measurement of the mesh.

## Outputs

Written under the profile's `output.root`, one subfolder per dataset.

| Artifact | Contents |
|---|---|
| `depth/<scene>/` | In `foundationstereo` mode, `depth_m.npy` (base-camera frame, full resolution) plus `depth_rectified_m.npy`, `disparity_px.npy`, and a `metadata.json` recording the stereo pair, model and normalisation. In `gt` mode, `depth_m.npy` only — **no `metadata.json`**, because the input was not predicted and there is nothing to record about how it was made. |
| `predictions/<scene>/` | Per-target JSON and three `.npz` mask sets — raw masks, pose-input masks, kept masks. No images: every overlay is written to `overlays/<scene>/`. `pose_input` is the set entering the final pose/rerank pass. |
| `overlays/<scene>/` | Both scopes live here. Per target, `obj<id>_raw.png` and `obj<id>_pose.png`; per scene, `scene_raw_all.png` (every proposal) and `scene_kept_all.png` (those that survived selection). Suppressed by `--no-overlays`. |
| `foundationpose_engine_cache/` | FoundationPose's TensorRT engines, built on first use. A cache, not a result — safe to delete, and the reason scene 0 runs several times longer than the rest. |
| `predictions.jsonl` | One record per target, **from inference only** — every proposal with its boxes, masks, poses and scores, and nothing ground truth touched. This is what `evaluate.py` re-scores. |
| `evaluations.jsonl` | The evaluation-side mirror: one record per target, metrics only, joined on `target_key`. |
| `inference_config.json` | What inference actually ran with — model path, max width, policies, cutoff, plus `runtime_sec_total` and the per-scene `runtime_by_scene` rows that `evaluate.py` turns into `runtime_summary.*`. The answer to "which depth model produced these numbers". |
| `raw` / `pose_input` / `filtered` / `selected_detection_summary.{json,csv}` | Detection metrics per stage. |
| `pose_summary.{json,csv}` | Max-vertex pose error, `max_vertex_error_within_threshold_rate`, P90/P99, and the `max_vertex_error_meets_required_rate` flag. `by_visibility` repeats these per band. |
| `runtime_summary.{json,csv}` | Per-scene production-path runtime. Prefer the median on short runs — FoundationPose builds TensorRT engines during the first scene. |
| `depth_summary.{json,csv}`, `report.md` | Depth quality and the human-readable report. Every field in `depth_summary` compares predicted depth against **collected** depth, so with `--no-depth-metrics`, or with no collected tree configured, the file is written but empty and the report's depth table reads `n/a`. The run says which of those applied. |

The three detection stages are worth distinguishing: `raw` is what SAM3 proposed, `pose_input` is
what actually entered the final FoundationPose pass (identical to `raw` unless refinement is on),
and `filtered`/`selected` is what survived reranking. Only `raw` is bit-reproducible — everything
after FoundationPose inherits its non-determinism, so any regression check should compare `raw`
exactly and the later stages within a tolerance.

---

## Effective defaults

All from `config/defaults.yaml`; every one is overridable per dataset via a profile's
`overrides:` block and per run via the matching CLI flag.

| Setting | Default |
|---|---|
| Frame used per scene | `rgb/000000.png` |
| SAM3 confidence threshold | `0.2` |
| Depth source | `foundationstereo` |
| GT min visible fraction | `0.1` (BOP convention) |
| Pose visibility bands | `[0.0, 0.9]` |
| Pose error threshold / rate | `5.0` mm, `0.90` (what `max_vertex_error_meets_required_rate` reports against) |
| SAM3 refinement policy | `replace_mid_nms06` (on) |
| Proposal selection policy | `soft_global_v1` (on) |
| Depth model | the TensorRT engine named by `depth.engine` (required) |
| FoundationStereo max width | `800` px |
| Working volume | `0.55`–`1.20` m (drives the disparity pre-shift) |
| CLAHE clip / detail boost | `3.0` / `1.5` |

Each value's rationale is a comment beside it in `config/defaults.yaml`. Several are
measurements; read the comment before changing the number.
