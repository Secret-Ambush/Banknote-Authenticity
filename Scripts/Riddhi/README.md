# Banknote PSO Lab

Interactive Plotly dashboard for exploring the Banknote Authentication dataset,
comparing the ANN architectures from repeated cross-validation, and inspecting
the selected model's locked-test performance and PSO training history.

Install the dashboard dependencies and launch it from this directory:

```bash
python -m pip install -r requirements.txt
.venv/bin/python save_person_results.py
.venv/bin/streamlit run dashboard.py
```

The app reads the dataset from
`Biologically-Inspired-Computation-Coursework/outline/data_banknote_authentication.txt`
and experiment sources from `ann_results/` and
`Biologically-Inspired-Computation-Coursework/experiments/results/`.
`save_person_results.py` collects those sources into
`ann_results/riddhi_results.json` and `ann_results/maazuddin_results.json`.
Run it again after changing either source to refresh the dashboard's named
results. The notebook also writes `riddhi_results.json` next to its standard
experiment artifacts when its final save cell is run.

The notebook results are labeled Riddhi; the other saved experiment groups are
labeled Maazuddin. Their metrics use different evaluation protocols: the
notebook architecture comparison uses repeated stratified 3-fold validation,
while the other experiment groups report repeated runs on a fixed stratified
70/30 test split. The dashboard displays this distinction rather than treating
the results as a controlled head-to-head. Exact duplicates are removed for EDA
to match the notebook's preprocessing. The locked test set is reported only for
the selected notebook model.
