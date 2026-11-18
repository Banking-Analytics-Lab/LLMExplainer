# Data

No loan data is stored in this repository. The `.gitignore` excludes `data/`, apart from this file.

## Freddie Mac Single-Family Loan-Level Dataset

The credit risk models use loans originated in 2015 and 2016 from the Freddie Mac Single-Family
Loan-Level Dataset: <https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset>.

Access: register and sign in on the Freddie Mac page above, accept its terms and conditions, and
download the quarterly files for 2015 and 2016. Freddie Mac allows non-commercial and research use
under those terms, and requires a licensing agreement for commercial redistribution. For that
reason the data is not included here.

### Where the code expects the files

All paths are relative to the repository root. The notebooks and scripts reach them through
`../data`, so run each one from its own folder.

| Path | Written by | Read by |
|---|---|---|
| `data/raw_data/<year>/*.zip` | you (the download) | `data_preprocessing/data-preperation.ipynb` |
| `data/extracted_data/<year>/origdata <year>-Q<n>.csv`, `svcgdata <year>-Q<n>.csv` | `data-preperation.ipynb` (unzips the archives) | `data-preperation.ipynb` |
| `data/prepared_data/prepared_<year>_full_year_data.csv` | `data-preperation.ipynb` | `data_preprocessing/data-cleaning.ipynb` |
| `data/cleaned_data/df_origination_{train,test}_graph_scaled.csv` | `data-cleaning.ipynb` | `graph_constructions/gat-construction.ipynb` |
| `data/cleaned_data/df_origination_{train,test}_explanation_scaled.csv` | `data-cleaning.ipynb` | `gat-construction.ipynb` (the test file) |
| `data/cleaned_data/df_origination_{train,test}_scaled.csv` | Not documented | `models/xgb-train.ipynb`, the scripts in `models/`, `finetuned_llms/synthetic-dataset-generation.py` |
| `data/graph_data/train_graph.pt`, `test_graph.pt`, `july_explanation_graph.pt` | `gat-construction.ipynb` | `models/gat-train.ipynb`, the GAT and hybrid scripts in `models/` |

`<year>` is 2015 or 2016 and `<n>` is the quarter, 1 to 4. The quarterly CSV files must have a header
row with the column names that `data-preperation.ipynb` uses, for example `id_loan`, `fico`,
`svcg_cycle`, `delq_sts` and `cd_zero_bal`. The step that turns the Freddie Mac download into files
with these names and columns is not in the repository.

## Survey data

The human evaluation data is in `human_evaluations/`, not here:

- `recorded-data-final.csv`: the Qualtrics export of the survey responses (184 response rows, below
  the three Qualtrics header rows). It holds the Likert ratings of the explanations from Gemma 3, DeepSeek R1
  and Gemini 2.5 on eight metrics, the answer to whether the participant has experience in credit
  risk (the cohort), optional demographic questions (age group, gender identity, education, region,
  economic sector) and two free-text questions.
- `recorded-data-final.qsf`: the Qualtrics survey definition, with the explanation texts shown to
  participants. `text-analysis.ipynb` reads it.

The paper states that the survey data is released with the code. No separate licence for the survey
data is stated.
