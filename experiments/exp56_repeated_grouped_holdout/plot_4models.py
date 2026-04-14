from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _normalize_model_names(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        'cnn': '1D-CNN',
        'resnet': 'ResNet1D',
        'concat': 'Concat-CNN',
        'hda': 'HDA-CNN',
        'CNN_1D': '1D-CNN',
        'Concat_CNN': 'Concat-CNN',
        'HDA_CNN': 'HDA-CNN',
    }
    df = df.copy()
    if 'model' in df.columns:
        df['model'] = df['model'].replace(mapping)
    return df


def _find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f'Missing required column. Tried: {candidates}. Available: {list(df.columns)}')


def _split_key(df: pd.DataFrame) -> str:
    if 'split_dir' in df.columns:
        return 'split_dir'
    if 'split_seed' in df.columns:
        return 'split_seed'
    if 'split_id' in df.columns:
        return 'split_id'
    raise ValueError('Expected split_dir, split_seed, or split_id in input csv.')


def parse_model_map(text: str | None) -> Dict[str, float]:
    if not text:
        return {}
    out: Dict[str, float] = {}
    for part in text.split(','):
        if not part.strip():
            continue
        k, v = part.split('=')
        out[k.strip()] = float(v)
    return out


def build_split_means(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_model_names(df)
    split_col = _split_key(df)

    if 'accuracy_mean' in df.columns and 'macro_f1_mean' in df.columns:
        keep = [c for c in df.columns if c.endswith('_mean') or c.endswith('_sd') or c in ['model', split_col, 'split_seed', 'split_id']]
        out = df[keep].copy()
        if split_col != 'split_dir' and 'split_dir' not in out.columns:
            out['split_dir'] = out[split_col].astype(str)
        return out

    acc_col = _find_col(df, ['accuracy', 'test_accuracy', 'acc'])
    f1_col = _find_col(df, ['macro_f1', 'macro avg_f1', 'f1'])
    prec_col = next((c for c in ['macro_precision', 'macro avg_precision', 'precision'] if c in df.columns), None)
    rec_col = next((c for c in ['macro_recall', 'macro avg_recall', 'recall'] if c in df.columns), None)
    wf1_col = next((c for c in ['weighted_f1', 'weighted avg_f1'] if c in df.columns), None)

    group_cols = ['model', split_col]
    agg = {acc_col: ['mean', 'std'], f1_col: ['mean', 'std']}
    if prec_col:
        agg[prec_col] = ['mean', 'std']
    if rec_col:
        agg[rec_col] = ['mean', 'std']
    if wf1_col:
        agg[wf1_col] = ['mean', 'std']

    out = df.groupby(group_cols).agg(agg)
    out.columns = ['_'.join([c[0], c[1]]) for c in out.columns]
    out = out.reset_index()
    rename = {
        f'{acc_col}_mean': 'accuracy_mean',
        f'{acc_col}_std': 'accuracy_sd',
        f'{f1_col}_mean': 'macro_f1_mean',
        f'{f1_col}_std': 'macro_f1_sd',
    }
    if prec_col:
        rename[f'{prec_col}_mean'] = 'macro_precision_mean'
        rename[f'{prec_col}_std'] = 'macro_precision_sd'
    if rec_col:
        rename[f'{rec_col}_mean'] = 'macro_recall_mean'
        rename[f'{rec_col}_std'] = 'macro_recall_sd'
    if wf1_col:
        rename[f'{wf1_col}_mean'] = 'weighted_f1_mean'
        rename[f'{wf1_col}_std'] = 'weighted_f1_sd'
    out = out.rename(columns=rename)
    if split_col != 'split_dir' and 'split_dir' not in out.columns:
        out['split_dir'] = out[split_col].astype(str)
    return out


def summarize_over_splits(split_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in split_df.columns if c.endswith('_mean')]
    parts = []
    for model, g in split_df.groupby('model'):
        row = {'model': model}
        for c in cols:
            row[c] = g[c].mean()
            row[c.replace('_mean', '_across_split_sd')] = g[c].std(ddof=1)
        parts.append(row)
    order = ['1D-CNN', 'ResNet1D', 'Concat-CNN', 'HDA-CNN']
    out = pd.DataFrame(parts)
    out['order'] = out['model'].apply(lambda x: order.index(x) if x in order else 99)
    return out.sort_values('order').drop(columns='order')


def paired_diff(split_df: pd.DataFrame, focus: str, baselines: List[str]) -> pd.DataFrame:
    split_col = _split_key(split_df)
    wide_f1 = split_df.pivot(index=split_col, columns='model', values='macro_f1_mean')
    wide_acc = split_df.pivot(index=split_col, columns='model', values='accuracy_mean')
    rows = []
    for base in baselines:
        if focus not in wide_f1.columns or base not in wide_f1.columns:
            continue
        common = wide_f1[[focus, base]].dropna().index
        for idx in common:
            rows.append({
                split_col: idx,
                'focus_model': focus,
                'baseline_model': base,
                'macro_f1_diff': float(wide_f1.loc[idx, focus] - wide_f1.loc[idx, base]),
                'accuracy_diff': float(wide_acc.loc[idx, focus] - wide_acc.loc[idx, base]),
            })
    return pd.DataFrame(rows)


def rank_and_winrate(split_df: pd.DataFrame, focus: str, baselines: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_col = _split_key(split_df)
    ranks = []
    for metric in ['accuracy_mean', 'macro_f1_mean']:
        wide = split_df.pivot(index=split_col, columns='model', values=metric)
        for split_name, row in wide.iterrows():
            ranked = row.rank(ascending=False, method='average')
            for model, rank in ranked.items():
                ranks.append({split_col: split_name, 'metric': metric, 'model': model, 'rank': float(rank)})
    rank_df = pd.DataFrame(ranks)

    paired = paired_diff(split_df, focus, baselines)
    win_rows = []
    for base in baselines:
        g = paired[paired['baseline_model'] == base]
        if len(g) == 0:
            continue
        win_rows.append({
            'focus_model': focus,
            'baseline_model': base,
            'metric': 'macro_f1',
            'mean_diff': g['macro_f1_diff'].mean(),
            'median_diff': g['macro_f1_diff'].median(),
            'win_rate': (g['macro_f1_diff'] > 0).mean(),
        })
        win_rows.append({
            'focus_model': focus,
            'baseline_model': base,
            'metric': 'accuracy',
            'mean_diff': g['accuracy_diff'].mean(),
            'median_diff': g['accuracy_diff'].median(),
            'win_rate': (g['accuracy_diff'] > 0).mean(),
        })
    return rank_df, pd.DataFrame(win_rows)


def retention_ratio(split_summary: pd.DataFrame, formal_df: pd.DataFrame) -> pd.DataFrame:
    formal = _normalize_model_names(formal_df.copy())
    rows = []
    for _, r in split_summary.iterrows():
        m = r['model']
        hit = formal[formal['model'] == m]
        if hit.empty:
            continue
        fa = float(hit.iloc[0]['formal_accuracy'])
        ff = float(hit.iloc[0]['formal_macro_f1'])
        rows.append({
            'model': m,
            'repeated_accuracy_mean': r['accuracy_mean'],
            'formal_accuracy': fa,
            'retention_accuracy': r['accuracy_mean'] / fa if fa else np.nan,
            'repeated_macro_f1_mean': r['macro_f1_mean'],
            'formal_macro_f1': ff,
            'retention_macro_f1': r['macro_f1_mean'] / ff if ff else np.nan,
        })
    return pd.DataFrame(rows)


def save_boxplot(split_df: pd.DataFrame, metric_col: str, ylab: str, out_path: Path):
    order = ['1D-CNN', 'ResNet1D', 'Concat-CNN', 'HDA-CNN']
    data = [split_df.loc[split_df['model'] == m, metric_col].dropna().tolist() for m in order]
    plt.figure(figsize=(8.5, 5.5))
    plt.boxplot(data, labels=order)
    plt.xlabel('Model')
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_paired_plot(paired_df: pd.DataFrame, diff_col: str, ylab: str, out_path: Path, split_label: str):
    plt.figure(figsize=(9, 5.5))
    for base in paired_df['baseline_model'].dropna().unique().tolist():
        g = paired_df[paired_df['baseline_model'] == base].sort_values(split_label)
        plt.plot(g[split_label].astype(str).tolist(), g[diff_col].tolist(), marker='o', label=f'HDA-CNN - {base}')
    plt.axhline(0.0, linestyle='--')
    plt.xlabel(split_label)
    plt.ylabel(ylab)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_rank_bar(rank_df: pd.DataFrame, out_path: Path):
    use = rank_df[rank_df['metric'] == 'macro_f1_mean'].groupby('model', as_index=False)['rank'].mean()
    order = ['1D-CNN', 'ResNet1D', 'Concat-CNN', 'HDA-CNN']
    use['order'] = use['model'].apply(lambda x: order.index(x) if x in order else 99)
    use = use.sort_values('order')
    plt.figure(figsize=(8, 5))
    plt.bar(use['model'], use['rank'])
    plt.xlabel('Model')
    plt.ylabel('Average rank (Macro F1)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_winrate_bar(win_df: pd.DataFrame, out_path: Path):
    use = win_df[(win_df['focus_model'] == 'HDA-CNN') & (win_df['metric'] == 'macro_f1')].copy()
    plt.figure(figsize=(7, 5))
    plt.bar(use['baseline_model'], use['win_rate'])
    plt.ylim(0, 1)
    plt.xlabel('Baseline model')
    plt.ylabel('HDA-CNN win rate (Macro F1)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_retention_bar(ret_df: pd.DataFrame, out_path: Path):
    order = ['1D-CNN', 'ResNet1D', 'Concat-CNN', 'HDA-CNN']
    ret_df = ret_df.copy()
    ret_df['order'] = ret_df['model'].apply(lambda x: order.index(x) if x in order else 99)
    ret_df = ret_df.sort_values('order')
    x = np.arange(len(ret_df))
    w = 0.35
    plt.figure(figsize=(8.5, 5.5))
    plt.bar(x - w/2, ret_df['retention_accuracy'], width=w, label='Accuracy retention')
    plt.bar(x + w/2, ret_df['retention_macro_f1'], width=w, label='Macro F1 retention')
    plt.xticks(x, ret_df['model'])
    plt.ylabel('Retention ratio')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', required=True, help='seed_metrics.csv or split_mean_metrics.csv')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--focus-model', default='HDA-CNN')
    parser.add_argument('--compare-models', default='ResNet1D,Concat-CNN,1D-CNN')
    parser.add_argument('--formal-csv', default='')
    parser.add_argument('--formal-acc', default='')
    parser.add_argument('--formal-f1', default='')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input_csv)
    split_df = build_split_means(raw)
    split_df.to_csv(out_dir / 'split_mean_metrics_recomputed.csv', index=False)

    split_summary = summarize_over_splits(split_df)
    split_summary.to_csv(out_dir / 'summary_over_splits.csv', index=False)

    baselines = [x.strip() for x in args.compare_models.split(',') if x.strip()]
    paired = paired_diff(split_df, args.focus_model, baselines)
    paired.to_csv(out_dir / 'paired_differences.csv', index=False)

    rank_df, win_df = rank_and_winrate(split_df, args.focus_model, baselines)
    rank_df.to_csv(out_dir / 'split_level_ranks.csv', index=False)
    win_df.to_csv(out_dir / 'rank_winrate_summary.csv', index=False)

    formal_df = pd.DataFrame()
    if args.formal_csv:
        formal_df = pd.read_csv(args.formal_csv)
    else:
        acc_map = parse_model_map(args.formal_acc)
        f1_map = parse_model_map(args.formal_f1)
        if acc_map and f1_map:
            models = sorted(set(acc_map) & set(f1_map))
            formal_df = pd.DataFrame({
                'model': models,
                'formal_accuracy': [acc_map[m] for m in models],
                'formal_macro_f1': [f1_map[m] for m in models],
            })
    if not formal_df.empty:
        formal_df = _normalize_model_names(formal_df)
        if 'formal_accuracy' not in formal_df.columns:
            formal_df = formal_df.rename(columns={'accuracy': 'formal_accuracy', 'macro_f1': 'formal_macro_f1'})
        ret = retention_ratio(split_summary, formal_df)
        ret.to_csv(out_dir / 'retention_ratio.csv', index=False)
        save_retention_bar(ret, out_dir / 'fig_retention_ratio.png')

    split_label = _split_key(split_df)
    save_boxplot(split_df, 'macro_f1_mean', 'Macro F1 (split-level mean)', out_dir / 'fig_box_macro_f1_splitmean.png')
    save_boxplot(split_df, 'accuracy_mean', 'Accuracy (split-level mean)', out_dir / 'fig_box_accuracy_splitmean.png')
    save_paired_plot(paired, 'macro_f1_diff', 'Macro F1 difference', out_dir / 'fig_paired_macro_f1_splitmean.png', split_label)
    save_paired_plot(paired, 'accuracy_diff', 'Accuracy difference', out_dir / 'fig_paired_accuracy_splitmean.png', split_label)
    save_rank_bar(rank_df, out_dir / 'fig_average_rank_macro_f1.png')
    save_winrate_bar(win_df, out_dir / 'fig_hda_winrate_macro_f1.png')

    print('Saved outputs under:', out_dir)


if __name__ == '__main__':
    main()
