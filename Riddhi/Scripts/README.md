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

From the workspace root, install the dashboard dependencies and launch it:

```bash
python -m pip install -r Biologically-Inspired-Computation-Coursework/Scripts/Riddhi/requirements.txt
python save_person_results.py
streamlit run Biologically-Inspired-Computation-Coursework/Scripts/Riddhi/dashboard.py
```

The app reads the dataset from
`Biologically-Inspired-Computation-Coursework/outline/data_banknote_authentication.txt`
and experiment artifacts from paths anchored to the project, not the current
working directory. Riddhi's model, predictions, and experiment are read from
`Biologically-Inspired-Computation-Coursework/Scripts/ann_results/`.
Maazuddin's experiment groups are read from
`Biologically-Inspired-Computation-Coursework/experiments/results/`, and the
named comparison bundles are read from `ann_results/` at the workspace root.
Run the workspace-root `save_person_results.py` again after changing either
person's source results to refresh those bundles.

The notebook results are labeled Riddhi; the other saved experiment groups are
labeled Maazuddin. Their metrics use different evaluation protocols: the
notebook architecture comparison uses repeated stratified 3-fold validation,
while the other experiment groups report repeated runs on a fixed stratified
70/30 test split. The dashboard displays this distinction rather than treating
the results as a controlled head-to-head. Exact duplicates are removed for EDA
to match the notebook's preprocessing. The locked test set is reported only for
the selected notebook model.
