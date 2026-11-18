# LLMExplainer

Code for the paper:

> Sahab Zandi, Noah Kostesku, Christophe Mues, María Óskarsdóttir, Cristián Bravo (2026). Communicating Credit Risk with Large Language Models: Evaluation of Explanations from Standard and Alternative Data-Based Models. *arXiv preprint arXiv:2608.17715*. <https://doi.org/10.48550/arXiv.2608.17715>

This is the code for Zandi et al. (2026) on LLM-based explainability of credit risk systems. It
builds three default-prediction pipelines on Freddie Mac single-family loan-level data: a tabular
pipeline (XGBoost with SHAP), a network pipeline (a graph neural network on a graph of loans that
share an area, a provider or both, with GNNExplainer), and a bimodal pipeline that combines the two
models' scores. Large language models turn the attributions of each pipeline into narratives for
the borrower. Three configurations write the narratives: Gemma 3 4B and DeepSeek-R1-Distill-Llama-70B,
both fine-tuned with LoRA on synthetic explanations written by GPT-4o, and Gemini 2.5 Flash without
fine-tuning. The code then scores the narratives (perplexity under GPT-Neo 125M and an LLM judge),
and analyses a survey in which credit risk professionals and non-professionals rated the bimodal
narratives on eight metrics. The survey data is included.

## Requirements

Python 3 and, for one script, R. The Python version and the package versions used for the paper
were not recorded.

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the packages imported by the code, without versions.
`human_evaluations/lmm_lrt.R` needs the R packages `lme4` and `lmerTest`.

Hardware: not documented. The fine-tuning and generation scripts load Gemma 3 4B and
DeepSeek-R1-Distill-Llama-70B in 8-bit with `bitsandbytes` and `device_map="auto"`, so they need
GPUs.

### Configuration

The code reads these environment variables. The Python scripts in `finetuned_llms/` and `models/`
call `load_dotenv()`, so a `.env` file in the folder you run them from also works (`.gitignore`
excludes `.env` and `*.env`).

| Variable | Used by | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `finetuned_llms/synthetic-dataset-generation.py`, the scripts in `models/` | GPT-4o writes the fine-tuning data and grades each explanation |
| `GEMINI_API_KEY` | `models/gemini-xgb.py`, `models/gemini-gat.py`, `models/gemini-hybrid.py` | Gemini 2.5 Flash API |
| `ANTHROPIC_API_KEY` | `judge/llm_judge.ipynb` | read by `anthropic.Anthropic()` for the Claude Sonnet 4.6 judge |
| `OPENAI_GRADER_MODEL` | the three `*-hybrid.py` scripts | grader model, default `gpt-4o` |
| `HTTP_PROXY`, `HTTPS_PROXY` | the two fine-tuning scripts, `models/gemini-*.py` | optional proxy; with neither set, the fine-tuning scripts put Hugging Face in offline mode |

Local model paths are placeholders in the code. Before running, replace `<my-path>` with the folder
that holds the models:

- `model_id` in `finetuned_llms/all-gemma3-models-finetune.py` and
  `finetuned_llms/all-deepseek-models-finetune.py` (the base models `gemma-3-4b-it` and
  `DeepSeek-R1-Distill-Llama-70B`);
- `BASE_MODEL_PATH` and `GEMMA_MODEL_PATH` or `DEEPSEEK_MODEL_PATH` in the `Config` of the
  `models/gemma3-*.py` and `models/deepseek-r1-*.py` scripts (the base model and the LoRA adapter).

The perplexity function loads GPT-Neo 125M from `local_models/gpt-neo-125M`, relative to the folder
you run from.

All paths in the code are relative. Run each script or notebook from its own folder.

## Data

The loan data is the Freddie Mac Single-Family Loan-Level Dataset, 2015 and 2016 originations. It
needs a free registration with Freddie Mac and is not redistributed here. `data/README.md` says how
to obtain it and where each notebook expects the files.

The survey data is in `human_evaluations/recorded-data-final.csv` (responses) and
`human_evaluations/recorded-data-final.qsf` (the Qualtrics survey with the explanation texts shown
to participants).

## Training

Run these in order, each from its own folder.

| Step | File | What it does | Output |
|---|---|---|---|
| 1 | `data_preprocessing/data-preperation.ipynb` | Unzips the quarterly files, flags defaults (delinquency status 3 or more or `R`, or zero-balance codes 3, 6 and 9, observed up to December 2016 for 2015 loans and December 2017 for 2016 loans), keeps loans with at least six months of performance records and samples 15,000 loans per quarter with a target default rate of 6% | `data/prepared_data/prepared_<year>_full_year_data.csv` |
| 2 | `data_preprocessing/data-cleaning.ipynb` | Imputes, engineers features, splits by origination month (graph sets: January to June 2015 and 2016, the 2016 set sampled to 5,000 loans; explanation sets: July 2015 and July 2016) and min-max scales with the scaler fitted on January to June 2015 | `data/cleaned_data/df_origination_*_scaled.csv` |
| 3 | `graph_constructions/gat-construction.ipynb` | Builds PyTorch Geometric graphs. Nodes are loans. Within each group of loans that share an area, a provider, or both, up to 20 sampled pairs are joined by edges; edge features are the edge type and the inverse group size | `data/graph_data/train_graph.pt`, `test_graph.pt`, `july_explanation_graph.pt` |
| 4 | `models/xgb-train.ipynb` | XGBoost with a 5-fold grid search on ROC AUC, an F1-optimal threshold, a bootstrap AUC interval and SHAP plots | `models/trained_xgb_model.pkl`, plots in `evaluations/xgb_base_model_performance/` |
| 5 | `models/gat-train.ipynb` | Graph model with two `TransformerConv` layers (the class is named `GAT`), focal loss and `NeighborLoader` mini-batches | `models/trained_gat_model.pt` |
| 6 | `finetuned_llms/synthetic-dataset-generation.py` | For 1,000 training loans, builds SHAP, GNNExplainer and combined prompts and has GPT-4o write the explanations; explanations that match a list of counter-intuitive patterns are rewritten by `gpt-3.5-turbo` | `finetuned_llms/samples/shap_test.jsonl`, `gnn_test.jsonl`, `combined_test.jsonl` |
| 7 | `finetuned_llms/all-gemma3-models-finetune.py`, `finetuned_llms/all-deepseek-models-finetune.py` | LoRA fine-tuning (rank 16, one epoch) of the 8-bit base model on one sample file. Set `dataset_path` and `new_model_name` for each of the three adapters; the commented lines give the alternatives | a LoRA adapter folder named `new_model_name` |

```bash
cd finetuned_llms
python synthetic-dataset-generation.py
python all-gemma3-models-finetune.py
python all-deepseek-models-finetune.py
```

Not documented:

- how `data/cleaned_data/df_origination_train_scaled.csv` and `df_origination_test_scaled.csv`,
  which steps 4 and 6 and the scripts in `models/` read, are made from the files of step 2;
- how the output of step 6 (`samples/*_test.jsonl`) becomes the input of step 7
  (`samples/combined_finetune.jsonl`, `shap_finetune.jsonl` or `gat_finetune.jsonl`).

Known issue: `models/xgb-train.ipynb` calls `shap` without importing it.

## Evaluation

### Explanations

Each script in `models/` explains test borrowers with one pipeline and one LLM, writes the
narratives to `models/explanations/<llm>_<pipeline>_explanations.jsonl`, computes their perplexity
under GPT-Neo 125M and grades them with GPT-4o on four criteria (individual feature coverage and
consistency, network coverage and consistency). Run `gemini-xgb.py` first. It samples 100 test
borrowers with seed 42 and saves them to `evaluations/gemini_xgb/selected_borrower_indices.json`,
which the GAT and hybrid scripts read. `gemma3-xgb.py` and `deepseek-r1-xgb.py` draw their own
sample of 100 with the same seed. The hybrid scripts combine the two models' scores with
weights 0.6 (graph model) and 0.4 (XGBoost), set in `Config`.

```bash
cd models
python gemini-xgb.py
python gemini-gat.py
python gemini-hybrid.py
python gemma3-xgb.py
python gemma3-gat.py
python gemma3-hybrid.py
python deepseek-r1-xgb.py
python deepseek-r1-gat.py
python deepseek-r1-hybrid.py
```

Known issues in the committed code:

- `models/deepseek-r1-hybrid.py`, `models/gemma3-gat.py`, `models/gemma3-hybrid.py` and
  `models/gemma3-xgb.py` do not parse (a split import line and indentation errors).
- `models/gemini-xgb.py`, `models/gemma3-xgb.py` and `models/deepseek-r1-xgb.py` use `target_col`
  without defining it when they load the test data.

### LLM judge

`judge/llm_judge.ipynb` grades the explanations of the nine pipeline and LLM configurations with
Claude Sonnet 4.6 on the same four criteria and prints 95% confidence intervals. It reads from
`models/explanations/` and `models/explanations_test/` and writes
`models/grades/<llm>_<pipeline>_grades.jsonl`. Each record it grades needs an `insight` (or
`shap_insight`) field with the attributions; the scripts in `models/` write only `node_index` and
`explanation`, so the step that adds this field is not documented.

### Statistics

```bash
cd ratings
python extract_data.py
python anova_perp_judge.py
```

- `ratings/extract_data.py` collects perplexities and judge grades into `ratings/perplexity_data.csv`
  and `ratings/llm_judge_data.csv`. It reads `models/explanations/<llm>-<pipeline>-explanations.jsonl`
  files with `perplexity`, `model_type` and `llm_name` fields, which the scripts in `models/` do not
  write. That step is not documented.
- `ratings/anova_perp_judge.py` fits two-way ANOVAs (pipeline by LLM, with eta squared) for perplexity
  and for each judge criterion and writes `ratings/anova_simulated_outputs/anova_*.csv`.

```bash
cd human_evaluations
Rscript lmm_lrt.R
```

The notebooks in `human_evaluations/` read the survey from the same folder:

- `paired_bh_correction.ipynb` turns the survey into one rating per participant, LLM and metric, and
  compares DeepSeek R1 with Gemini 2.5 by Wilcoxon signed-rank tests per cohort and metric, with Holm
  and Benjamini-Hochberg adjustments.
- `variance-decomposition.ipynb` fits a random-intercepts model with participant and explanation
  (LLM by metric) effects.
- `lmm_lrt.R` fits linear mixed models with cohort, LLM and metric, compares them with
  likelihood-ratio tests and writes `stats_out/lrt_results.csv`, `stats_out/lmm_type3_anova.csv`
  and `stats_out/table9_descriptive_stats.csv`.
- `text-analysis.ipynb` computes readability, sentiment and count features of the explanation texts
  and compares top-quartile with bottom-quartile rated explanations per metric (Mann-Whitney tests,
  Cohen's d).
  It writes `heatmap.png`, `text_analysis_results.csv` and `text_analysis_summary_stats.csv`.

`variance-decomposition.ipynb` and `lmm_lrt.R` read `human_evaluations/stats_out/survey_long.csv`.
The code that writes this file is not documented; `paired_bh_correction.ipynb` builds the same long
table in memory but does not save it.

## Pretrained models

Not released. The trained XGBoost and graph models and the LoRA adapters are not in the repository,
and `.gitignore` excludes `models/trained_xgb_model.pkl` and `models/trained_gat_model.pt`.

## Results

Table and figure numbers follow arXiv version 1 of the paper.

| Table or figure in the paper | Command | Output |
|---|---|---|
| Table 4, LLM-as-a-judge scores | `judge/llm_judge.ipynb` | printed 95% intervals |
| Table 5, two-way ANOVA of judge scores | `ratings/extract_data.py`, then `ratings/anova_perp_judge.py` | `ratings/anova_simulated_outputs/anova_*.csv` |
| Tables 6 and 7, mean ratings by cohort | Not documented | |
| Table 8, participant-level means by cohort, LLM and metric | `human_evaluations/lmm_lrt.R` | `stats_out/table9_descriptive_stats.csv` |
| Table 9, paired differences, DeepSeek R1 and Gemini 2.5 | `human_evaluations/paired_bh_correction.ipynb` | printed table |
| Table 10, variance decomposition | `human_evaluations/variance-decomposition.ipynb` | printed variance components |
| Figure 3, between-cohort comparisons | Not documented | |
| Figure 4, linguistic features of high- and low-rated explanations | `human_evaluations/text-analysis.ipynb` | `heatmap.png` |
| Table D.1, perplexity | the scripts in `models/` | printed per run |
| Table D.2, two-way ANOVA of perplexity | `ratings/anova_perp_judge.py` | `ratings/anova_simulated_outputs/anova_perplexity.csv` |
| Tables D.3 to D.6, simulated ratings | Not documented | |
| Figure F.1, participant characteristics | Not documented | |

## Citation

```bibtex
@article{zandi2026communicating,
  title   = {Communicating Credit Risk with Large Language Models: Evaluation of Explanations from Standard and Alternative Data-Based Models},
  author  = {Zandi, Sahab and Kostesku, Noah and Mues, Christophe and {\'O}skarsd{\'o}ttir, Mar{\'i}a and Bravo, Cristi{\'a}n},
  year    = {2026},
  journal = {arXiv preprint arXiv:2608.17715},
  doi     = {10.48550/arXiv.2608.17715},
}
```

`CITATION.cff` holds the same reference for GitHub's "Cite this repository" button.

## Licence

Apache License 2.0. See `LICENSE`.

## Acknowledgements

The first author acknowledges the support of the Natural Sciences and Engineering Research Council
(NSERC) of Canada through the Canada Graduate Scholarships – Doctoral (CGS D) program. The third
author acknowledges the support of the Economic and Social Research Council (ESRC) [grant number
ES/P000673/1]. The fourth author acknowledges the support of the Icelandic Research Fund (IRF)
[grant number 228511-051]. The last author acknowledges the support of the NSERC [discovery grant
RGPIN-2020-07114]. This research was undertaken, in part, thanks to funding from the Canada Research
Chairs program [CRC-2024-00192]. This work was enabled in part by support provided by Compute
Ontario, Calcul Québec and the Digital Research Alliance of Canada.
