# Banknote PSO Lab

Interactive Plotly dashboard for exploring the Banknote Authentication dataset,
comparing the ANN architectures from repeated cross-validation, and inspecting
the selected model's locked-test performance and PSO training history.

The "Riddhi's ANN architectures" tab contains the notebook's architecture
experiments; "People comparison" includes both contributors. The PSO deep dive
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

From the workspace root, install the dashboard dependencies and launch it:

```bash
python -m pip install -r Biologically-Inspired-Computation-Coursework/Riddhi/Scripts/requirements.txt
python Biologically-Inspired-Computation-Coursework/Riddhi/Scripts/save_person_results.py
streamlit run Biologically-Inspired-Computation-Coursework/Riddhi/Scripts/dashboard.py
```

The app reads the dataset from
`Biologically-Inspired-Computation-Coursework/outline/data_banknote_authentication.txt`
and experiment artifacts from paths anchored to the project, not the current
working directory. Riddhi's model, predictions, and experiment are read from
`Biologically-Inspired-Computation-Coursework/Riddhi/ann_results/`.
Maazuddin's experiment groups are read from
`Biologically-Inspired-Computation-Coursework/experiments/results/`, and the
named comparison bundles are read from `Riddhi/ann_results/` inside the project,
with a fallback to `ann_results/` at the workspace root for older bundles.
Run `Riddhi/Scripts/save_person_results.py` from the project root after changing
either person's source results to refresh those bundles.

The notebook results are labeled Riddhi; the other saved experiment groups are
labeled Maazuddin. Their metrics use different evaluation protocols: the
notebook architecture comparison uses repeated stratified 3-fold validation,
while the other experiment groups report repeated runs on a fixed stratified
70/30 test split. The dashboard displays this distinction rather than treating
the results as a controlled head-to-head. Exact duplicates are removed for EDA
to match the notebook's preprocessing. The locked test set is reported only for
the selected notebook model.
