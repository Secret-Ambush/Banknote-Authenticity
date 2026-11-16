# Banknote PSO Lab

Interactive Plotly dashboard for exploring the Banknote Authentication dataset,
comparing both contributors' selected models on identical test rows, and
inspecting PSO training history.

The app has three tabs: "Data explorer", "Shared-test model comparison", and
"PSO deep dive". The comparison contains architecture diagrams, paired metrics,
confusion matrices, ROC curves and a sample inspector/download. The PSO deep dive
shows each implementation's velocity and position equations, saved coefficients,
inertia schedules, and expected random attraction multipliers. Maazuddin's saved
configuration can be selected to inspect its personal, informant, and global
weights. These coefficient comparisons are not measured movement contributions;
the saved results do not contain per-particle term histories.

## Informant-enabled notebook PSO

The notebook now adds an informant-best attraction to its personal-best and
global-best terms. Set `c_informant` (strength) and `informant_count` (number,
including self) in `BASE_PSO` or `REFINEMENT_PSO`. Each particle resamples
distinct informants every iteration and follows their lowest-loss personal best.
With attraction enabled the count must be between 1 and the swarm size.
Screening uses strength 0.5 and count 3; refinement profiles use counts 3, 5 and 7
alongside other setting changes, so they do not isolate the effect of count.
Set `c_informant=0` to reproduce the original optimizer and random stream.

Saved results from before this change are historical, not informant-enabled
measurements. Run all notebook cells in order and refresh the named result
bundles before claiming new results. Both hyperparameters are retained in each
saved configuration. The dashboard reads those saved settings rather than
assuming the new defaults were used.

Run the focused checks from the workspace root:

```bash
python -m unittest discover -s Biologically-Inspired-Computation-Coursework/Riddhi/Scripts/tests -p 'test_notebook_pso.py' -v
```

To audit the saved scores, run `test_result_integrity.py` with the same command's
`-p` option. It independently evaluates the saved weights, checks source labels,
split isolation and training-only scaling, reproduces two complete CV candidates,
and runs a shuffled-training-label negative control. It does not write artifacts
or select a new model using test performance.

The notebook prints class counts, proportions and majority/minority ratios for
the deduplicated data and every split before training, plus training-only
majority baselines for each validation fold. Its final "Is this class imbalance
or confident separation?" section adds interactive class-count, per-class recall
and probability plots, with balanced accuracy, macro F1 and Brier score tables.
These checks do not retrain or rebalance data. Weighting or resampling, if later
justified by development validation, must stay inside the training folds.

The shared comparison fixes both thresholds at 0.5 and shows exact error counts.
Perfect scores
on this split do not establish perfect generalisation: CV selection is optimistic,
the holdout has already been inspected, and unknown specimen relationships or
distribution shifts are not ruled out by exact-duplicate checks.

From the workspace root, install the dashboard dependencies and launch it:

```bash
python -m pip install -r Biologically-Inspired-Computation-Coursework/Riddhi/Scripts/requirements.txt
python Biologically-Inspired-Computation-Coursework/Riddhi/Scripts/save_person_results.py
streamlit run Biologically-Inspired-Computation-Coursework/dashboard.py
```

Notebook Plotly output requires `ipython` and `nbformat>=4.2.0`, included in the
requirements above. If you install them while a notebook is running, restart its
kernel before rerunning the plotting cell: Plotly can cache a missing dependency
when it is first imported.

The app reads the dataset from
`Biologically-Inspired-Computation-Coursework/outline/data_banknote_authentication.txt`
and experiment artifacts from paths anchored to the project, not the current
working directory. Riddhi's model, predictions, and experiment are read from
`Biologically-Inspired-Computation-Coursework/Riddhi/ann_results/`.
Maazuddin's experiment groups are read from
`Biologically-Inspired-Computation-Coursework/experiments/results/`, and the
named comparison bundles are read from `Riddhi/ann_results/` inside the project,
with a legacy fallback to the workspace-root `ann_results/` for Maazuddin's
bundle only. Riddhi's bundle, model and predictions must all be in
`Riddhi/ann_results/`; the dashboard never substitutes an older workspace model.
All paths are resolved relative to the dashboard file, so launching from a
different working directory does not change which results are loaded.
Run `Riddhi/Scripts/save_person_results.py` from the project root after changing
either person's source results to refresh those bundles.

## Generating the shared comparison

From the project root, after running Riddhi's notebook and refreshing its bundle:

```bash
python -m src.evaluation.shared_comparison
python -m pytest tests/test_shared_comparison.py -q
```

This selects Maazuddin's configuration across the 13 profiles in
`src/experiments/configurations.py` using Riddhi's development-only stratified
folds and optimizer seeds. Each fold fits its own training-only scaler. Ranking
uses mean validation accuracy, then fewer parameters, then configuration order.
The winning configuration is refitted with seeds 101/202/303 on the full shared
development set; the restart with best training accuracy wins (first seed on a
tie). Riddhi's existing CV-selected model is not retrained.

The command saves `maazuddin_shared_model.npz` and `shared_comparison.json` in
`Riddhi/ann_results/`, separate from historical Maazuddin experiments. It never
uses test predictions to select a configuration or restart. The dashboard runs
inference with both saved models on the same 270 deduplicated test rows. Source
hashes prevent stale or mixed runs from silently appearing as a valid comparison;
rerun the command whenever Riddhi's model or experiment changes.

This is not a reconstruction of Maazuddin's original fixed-70/30 model: those
experiments did not retain trained weights. It is also not a controlled ablation:
the two implementations have different objectives, evaluation budgets and search
spaces. The shared holdout has already been inspected, so its scores must not be
presented as fresh external validation. Architecture/seed selection optimism,
unknown specimen dependence and distribution shift remain limitations.

## Riddhi's nested generalisation study

From the project root:

```bash
python -m src.evaluation.riddhi_nested_cv
python -m pytest tests/test_riddhi_nested_cv.py -q
```

The runner extracts the actual notebook ANN/PSO definitions and search settings
without executing notebook experiments. It runs five outer folds on the 1,078
development rows, leaving the existing 270-row holdout excluded. Within each
outer training partition it repeats all 14 architecture candidates and nine
shortlisted PSO refinements using three inner folds and two seeds. Scalers are
training-only at both levels. Three outer-training restarts are selected by
training objective, never by outer-validation accuracy. The threshold stays 0.5.

There are 690 inner fits and 15 outer-training restarts. Results are saved
separately under `Riddhi/ann_results/nested_cv/`: all inner scores, selected
configs, split identities, five fitted fold models and one out-of-fold prediction
per development row. A completed study is not overwritten automatically.
The notebook's final section displays per-fold accuracy, balanced accuracy, F1,
BCE, majority baselines, pooled confusion and mistakes. These do not replace the
dashboard's final models. Fold variation is descriptive, not a confidence
interval. Prior dataset/search-space exploration and unknown specimen dependence
still limit claims of external generalisation.

The completed five-fold study (outer seed 271828) produced 1 error among 1,078
out-of-fold predictions: pooled accuracy 99.9072%, balanced accuracy 99.9153%,
and Class 1 F1 99.8976%. Outer-fold accuracies were 99.5370%, 100%, 100%, 100%,
100%; their sample SD was 0.2070 percentage points. The confusion matrix was
`[[589, 1], [0, 488]]` (actual rows, predicted columns). The one error was source
row index 349 (zero-based), actual Class 0, predicted Class 1 with probability
0.54427. This supports strong within-dataset generalisation of the search
procedure, not perfection or a fresh external validation claim.
