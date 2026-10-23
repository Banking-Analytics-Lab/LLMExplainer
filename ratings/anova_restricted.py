"""Refit of the judge-score ANOVAs (Table 5) for the EJOR revision (R1.4, R3.2).

Run from ratings/:  python anova_restricted.py [--root ..] [--dedupe]

Reads the grade files written by judge/llm_judge.ipynb and the explanation files they
graded, so each score keeps the node_index of its loan. Steps:
  0. data checks: rows per file, duplicates, overlap of node_index between configurations
  1. original model on all 3 pipelines (reproduces Table 5 if the data are the same)
  2. restricted 2x3 model on comparable pipelines, OLS Type II, classical and HC3 F
  3. same restricted model with a random intercept per loan (node_index), ML fits and
     likelihood-ratio tests in Type II order
  4. id_loan check: is row i of the tabular test file the same loan as node i of the
     July explanation graph? (needs the data files under data/)
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm

# same list as judge/llm_judge.ipynb (paths relative to the repository root)
SPECS = [
    ('models/explanations/gemma_xgb_explanations.jsonl', 'tabular', 'gemma'),
    ('models/explanations/gemma_gnn_explanations.jsonl', 'network', 'gemma'),
    ('models/explanations/gemma_hybrid_explanations.jsonl', 'bimodal', 'gemma'),
    ('models/explanations_test/deepseek_xgb_explanations.jsonl', 'tabular', 'deepseek'),
    ('models/explanations/deepseek_gnn_explanations.jsonl', 'network', 'deepseek'),
    ('models/explanations_test/deepseek_hybrid_explanations.jsonl', 'bimodal', 'deepseek'),
    ('models/explanations_test/gemini_xgb_explanations.jsonl', 'tabular', 'gemini'),
    ('models/explanations/gemini_gnn_explanations.jsonl', 'network', 'gemini'),
    ('models/explanations_test/gemini_hybrid_explanations.jsonl', 'bimodal', 'gemini'),
]
PIPELINE = {'tabular': 'Tabular-based', 'network': 'Network-based', 'bimodal': 'Bimodal'}
LLM = {'gemma': 'Gemma 3 (ft)', 'deepseek': 'DeepSeek R1 (ft)', 'gemini': 'Gemini 2.5'}
METRICS = {'TFC': 'Individual Feature Coverage', 'TDC': 'Individual Feature Consistency',
           'NFC': 'Network Coverage', 'NDC': 'Network Consistency'}
COMPARABLE = {'TFC': ['Tabular-based', 'Bimodal'], 'TDC': ['Tabular-based', 'Bimodal'],
              'NFC': ['Network-based', 'Bimodal'], 'NDC': ['Network-based', 'Bimodal']}
# Table 5 as submitted: eta^2 for pipeline, llm, interaction
TABLE5 = {'TFC': (0.391, 0.014, 0.024), 'TDC': (0.352, 0.031, 0.052),
          'NFC': (0.377, 0.067, 0.046), 'NDC': (0.069, 0.051, 0.041)}
FORMULA = 'score ~ C(pipeline) + C(llm) + C(pipeline):C(llm)'


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def load_scores(root):
    rows, checks = [], []
    for expl_path, model_type, llm_key in SPECS:
        expl_file = root / expl_path
        grade_file = root / 'models/grades' / f'{llm_key}_{model_type}_grades.jsonl'
        if not grade_file.exists() and model_type == 'bimodal':
            grade_file = root / 'models/grades' / f'{llm_key}_hybrid_grades.jsonl'
        if not expl_file.exists() or not grade_file.exists():
            print(f'MISSING: {expl_file if not expl_file.exists() else grade_file}')
            continue
        expl = load_jsonl(expl_file)
        grades = load_jsonl(grade_file)
        node = [r.get('node_index') for r in expl]
        n_out_of_range = n_zero = 0
        for g in grades:
            i = g['record_index']
            for code, key in METRICS.items():
                val = g.get(key)
                if val in (0, None):  # dropped, as in extract_data.py
                    n_zero += 1
                    continue
                if not 1 <= val <= 5:
                    n_out_of_range += 1
                    continue
                rows.append({'pipeline': PIPELINE[model_type], 'llm': LLM[llm_key], 'metric': code,
                             'score': float(val), 'record_index': i,
                             'node_index': node[i] if i < len(node) else None})
        rec_idx = [g['record_index'] for g in grades]
        checks.append({'config': f'{llm_key}/{model_type}', 'explanation_file': expl_path,
                       'explanations': len(expl), 'graded': len(grades),
                       'dup_record_index': len(rec_idx) - len(set(rec_idx)),
                       'dup_node_index': len(node) - len(set(node)),
                       'missing_node_index': sum(v is None for v in node),
                       'scores_0_or_missing': n_zero, 'scores_outside_1_5': n_out_of_range})
    return pd.DataFrame(rows), pd.DataFrame(checks)


def eta_table(d, typ=2, robust=None):
    model = smf.ols(FORMULA, data=d).fit()
    aov = anova_lm(model, typ=typ)
    aov['eta_sq'] = aov['sum_sq'] / aov['sum_sq'].sum()
    if robust:
        rob = anova_lm(model, typ=typ, robust=robust)
        aov[f'F_{robust}'] = rob['F']
        aov[f'p_{robust}'] = rob['PR(>F)']
    return aov


def lrt(d, group):
    """Type II order: interaction vs additive model, then each main effect vs additive."""
    fits = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for name, f in [('full', FORMULA), ('add', 'score ~ C(pipeline) + C(llm)'),
                        ('no_pipe', 'score ~ C(llm)'), ('no_llm', 'score ~ C(pipeline)')]:
            fits[name] = smf.mixedlm(f, d, groups=d[group]).fit(reml=False)
    out = []
    for term, big, small in [('pipeline:llm', 'full', 'add'), ('pipeline', 'add', 'no_pipe'), ('llm', 'add', 'no_llm')]:
        chi2 = 2 * (fits[big].llf - fits[small].llf)
        df = len(fits[big].fe_params) - len(fits[small].fe_params)
        out.append({'term': term, 'chi2': chi2, 'df': df, 'p': stats.chi2.sf(chi2, df)})
    var_loan = float(fits['full'].cov_re.iloc[0, 0])
    icc = var_loan / (var_loan + fits['full'].scale)
    notes = sorted({str(w.message)[:80] for w in caught})
    return pd.DataFrame(out), var_loan, icc, notes


def id_loan_check(root):
    data = root / 'data/cleaned_data'
    tab_path, expl_path = data / 'df_origination_test_scaled.csv', data / 'df_origination_test_explanation_scaled.csv'
    if not tab_path.exists() or not expl_path.exists():
        print('  data files not found; skipped')
        return
    lists = {}
    for llm_key in ['gemma3', 'deepseek', 'gemini']:
        p = root / f'evaluations/{llm_key}_xgb/selected_borrower_indices.json'
        if p.exists():
            lists[llm_key] = json.loads(p.read_text())
    if 'gemini' not in lists:
        print('  evaluations/gemini_xgb/selected_borrower_indices.json not found; skipped')
        return
    ref = lists['gemini']
    for k, v in lists.items():
        print(f'  selected indices {k}: {len(v)}, same list as gemini_xgb: {v == ref}')

    tab = pd.read_csv(tab_path)
    july = pd.read_csv(expl_path)
    july = july[(pd.to_numeric(july['year'], errors='coerce') == 2016) &
                (pd.to_numeric(july['month'], errors='coerce') == 7)].copy()
    if 'd_timer' in july.columns:  # same filter as create_loan_graph (threshold 0)
        july['d_timer'] = pd.to_numeric(july['d_timer'], errors='coerce')
        july['d_timer'] = july['d_timer'].fillna(july['d_timer'].median())
        july = july[july['d_timer'] >= 0]
    july = july.reset_index(drop=True)
    print(f'  tabular test rows {len(tab)}, rebuilt July graph nodes {len(july)}')
    graph = root / 'data/graph_data/july_explanation_graph.pt'
    if graph.exists():
        try:
            import torch
            g = torch.load(graph, map_location='cpu', weights_only=False)
            print(f'  july_explanation_graph.pt num_nodes {g.num_nodes} (must equal {len(july)})')
        except Exception as e:
            print(f'  could not load graph: {e}')
    key = 'id_loan' if 'id_loan' in tab.columns and 'id_loan' in july.columns else 'id'
    in_graph = [i for i in ref if i < len(july)]
    same = [i for i in in_graph if tab.loc[i, key] == july.loc[i, key]]
    print(f'  indices beyond the July graph (dropped by the GAT scripts): {len(ref) - len(in_graph)}')
    print(f'  {key} equal for row i (tabular) and node i (July graph): {len(same)} of {len(in_graph)}')
    pd.DataFrame({'index': in_graph, f'{key}_tabular_row': [tab.loc[i, key] for i in in_graph],
                  f'{key}_july_node': [july.loc[i, key] for i in in_graph]}).to_csv(OUT_DIR / 'id_loan_check.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='..', help='repository root')
    ap.add_argument('--dedupe', action='store_true', help='keep the first score per configuration and node_index')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    OUT_DIR = Path('./anova_restricted_outputs')
    OUT_DIR.mkdir(exist_ok=True)

    print('STEP 0. Data checks')
    scores, checks = load_scores(root)
    print(checks.to_string(index=False))
    checks.to_csv(OUT_DIR / 'data_checks.csv', index=False)
    if scores.empty:
        raise SystemExit('No scores loaded.')
    if args.dedupe:
        before = len(scores)
        scores = scores.drop_duplicates(['pipeline', 'llm', 'metric', 'node_index'])
        print(f'--dedupe: {before - len(scores)} duplicate scores removed')
    sets = scores.groupby(['pipeline', 'llm'])['node_index'].apply(lambda s: frozenset(s.dropna()))
    for pipe in PIPELINE.values():
        if pipe in sets.index.get_level_values(0):
            s = sets[pipe]
            print(f'  {pipe}: same node_index set for all LLMs: {len(set(s.values)) == 1}')
    scores.to_csv(OUT_DIR / 'judge_scores_with_node_index.csv', index=False)

    print('\nSTEP 1. Original model, all pipelines (compare with Table 5)')
    step1 = []
    for code in METRICS:
        d = scores[scores.metric == code]
        aov = eta_table(d)
        print(f'\n{code}: rows {len(d)}')
        print(d.groupby(['pipeline', 'llm']).size().unstack(fill_value=0).to_string())
        print(aov.round(4).to_string())
        e = aov['eta_sq']
        step1.append({'metric': code, 'eta_pipe': e['C(pipeline)'], 'eta_llm': e['C(llm)'],
                      'eta_int': e['C(pipeline):C(llm)'], 'table5_pipe': TABLE5[code][0],
                      'table5_llm': TABLE5[code][1], 'table5_int': TABLE5[code][2]})
    step1 = pd.DataFrame(step1)
    print('\n' + step1.round(3).to_string(index=False))
    step1.to_csv(OUT_DIR / 'step1_original_vs_table5.csv', index=False)

    print('\nSTEP 2. Restricted 2x3 model on comparable pipelines, OLS Type II (classical and HC3)')
    step2 = []
    for code in METRICS:
        d = scores[(scores.metric == code) & (scores.pipeline.isin(COMPARABLE[code]))]
        aov = eta_table(d, robust='hc3')
        print(f'\n{code} on {" + ".join(COMPARABLE[code])}: rows {len(d)}')
        print(aov.round(4).to_string())
        for term in ['C(pipeline)', 'C(llm)', 'C(pipeline):C(llm)']:
            r = aov.loc[term]
            step2.append({'metric': code, 'term': term, 'df': r['df'], 'F': r['F'], 'p': r['PR(>F)'],
                          'F_hc3': r['F_hc3'], 'p_hc3': r['p_hc3'], 'eta_sq': r['eta_sq']})
        aov.to_csv(OUT_DIR / f'step2_restricted_{code}.csv')
    step2 = pd.DataFrame(step2)
    step2.to_csv(OUT_DIR / 'step2_restricted_summary.csv', index=False)
    wide = step2.pivot(index='metric', columns='term', values='eta_sq')
    wide['pipe_over_llm'] = wide['C(pipeline)'] / wide['C(llm)']
    print('\nRestricted eta^2:\n' + wide.round(3).to_string())

    print('\nSTEP 3. Restricted model with a random intercept per loan (node_index), ML, LR tests')
    step3 = []
    for code in METRICS:
        d = scores[(scores.metric == code) & (scores.pipeline.isin(COMPARABLE[code]))].dropna(subset=['node_index'])
        tests, var_loan, icc, notes = lrt(d, 'node_index')
        print(f'\n{code}: rows {len(d)}, loans {d.node_index.nunique()}, loan variance {var_loan:.3f}, ICC {icc:.3f}')
        print(tests.round(4).to_string(index=False))
        for n in notes:
            print(f'  warning: {n}')
        step3.append(tests.assign(metric=code, loan_var=var_loan, icc=icc))
    pd.concat(step3).to_csv(OUT_DIR / 'step3_mixed_lrt.csv', index=False)

    print('\nSTEP 4. id_loan check (tabular row i vs July graph node i)')
    id_loan_check(root)
    print(f'\nOutputs in {OUT_DIR.resolve()}')
