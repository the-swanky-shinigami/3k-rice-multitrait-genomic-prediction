import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch

base_dir = Path(__file__).resolve().parent
artifacts_dir = base_dir / "artifacts"
figures_dir = base_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

# Set global publication plot style
plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# -----------------------------------------------------------------------------
# FIGURE 1: Phenotype Architecture, Missingness & Correlation
# -----------------------------------------------------------------------------
def plot_figure_1():
    print("Generating Figure 1: Phenotype Architecture...")
    ph_path = artifacts_dir / "canonical_phenotypes.csv"
    if not ph_path.exists():
        ph_path = base_dir / "data" / "phenotype_matrix.csv"
    ph = pd.read_csv(ph_path)
    
    stats_path = artifacts_dir / "trait_coverage_summary.csv"
    if not stats_path.exists():
        trait_cols = [c for c in ph.columns if c not in ['FID', 'IID']]
        rows = []
        for c in trait_cols:
            n_obs = ph[c].notna().sum()
            pct = (n_obs / len(ph)) * 100.0
            t_type = 'Helper (Dense)' if c.startswith('H') else 'Target (Sparse)'
            code = c.split('_')[0]
            name = '_'.join(c.split('_')[1:])
            rows.append({'Trait_Code': code, 'Trait_Name': name, 'Type': t_type, 'N_Observed': n_obs, 'Coverage_Pct': pct})
        stats = pd.DataFrame(rows)
    else:
        stats = pd.read_csv(stats_path)
    
    fig = plt.figure(figsize=(18, 6), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.2, 0.9], wspace=0.35)
    
    # Panel A: Coverage Bar Chart
    ax1 = fig.add_subplot(gs[0])
    colors = ['#1f77b4' if t == 'Helper (Dense)' else '#e24a33' for t in stats['Type']]
    bars = ax1.barh(range(len(stats))[::-1], stats['Coverage_Pct'], color=colors, edgecolor='black', alpha=0.88, height=0.65)
    ax1.set_yticks(range(len(stats))[::-1])
    ax1.set_yticklabels([f"{r.Trait_Code}: {r.Trait_Name.replace('_', ' ')}" for _, r in stats.iterrows()], fontsize=9, fontweight='bold')
    ax1.set_xlabel('Cohort Coverage Percentage (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Natural Phenotype Missingness Architecture', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlim(0, 100)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    
    for i, bar in enumerate(bars):
        pct = stats['Coverage_Pct'].iloc[i]
        n_obs = stats['N_Observed'].iloc[i]
        ax1.text(pct + 1.5, len(stats) - 1 - i, f"{pct:.1f}% (N={n_obs:,})", va='center', fontsize=8, fontweight='bold')
        
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='Dense Helper Traits (~80%)'),
        Patch(facecolor='#e24a33', edgecolor='black', label='Sparse Target Traits (~18-20%)')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=8)
    
    # Panel B: Pairwise Correlation Heatmap
    ax2 = fig.add_subplot(gs[1])
    trait_cols = [c for c in ph.columns if c not in ['FID', 'IID']]
    corr_mat = ph[trait_cols].corr()
    corr_labels = [c.split('_')[0] for c in trait_cols]
    
    sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1.0, vmax=1.0,
                xticklabels=corr_labels, yticklabels=corr_labels, ax=ax2, cbar_kws={'label': 'Pearson Correlation (r)', 'shrink': 0.8},
                annot_kws={'size': 8, 'weight': 'bold'}, linewidths=0.5)
    ax2.set_title('(B) Cross-Domain Pairwise Phenotypic Correlation', fontsize=12, fontweight='bold', pad=10)
    
    # Panel C: Target Trait Standardized Density Distributions (z-scores)
    ax3 = fig.add_subplot(gs[2])
    target_cols = [c for c in trait_cols if c.startswith('T')]
    for t_col in target_cols:
        vals = ph[t_col].dropna()
        # Strictly standardize to z-scores so the axis label is scientifically accurate
        z_vals = (vals - vals.mean()) / vals.std()
        sns.kdeplot(z_vals, ax=ax3, label=t_col.split('_')[0], linewidth=1.8)
    ax3.set_xlabel('Standardized Phenotypic Value ($z$-score)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax3.set_title('(C) Standardized Target Stress Trait Distributions', fontsize=12, fontweight='bold', pad=10)
    ax3.legend(title='Target', frameon=True, fontsize=8)
    ax3.grid(axis='both', linestyle='--', alpha=0.5)
    
    fig.subplots_adjust(top=0.90, bottom=0.15, left=0.08, right=0.98, wspace=0.35)
    plt.savefig(figures_dir / "fig1_phenotype_architecture.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig1_phenotype_architecture.pdf", bbox_inches='tight')
    plt.close()
    print("Saved Figure 1.")

# -----------------------------------------------------------------------------
# FIGURE 2: Classical Quantitative Genetics & Linear Baselines
# -----------------------------------------------------------------------------
def plot_figure_2():
    print("Generating Figure 2: Classical Baselines...")
    master_df = pd.read_csv(artifacts_dir / "master_leaderboard_table.csv")
    per_trait_df = pd.read_csv(artifacts_dir / "per_trait_native_metrics.csv")
    
    # Explicit ordered models and palette: Helper Stacking, RidgeCV, GBLUP, Training Mean
    ordered_base_models = ['Helper_Stacking_Ridge_OOF', 'Single_Trait_RidgeCV', 'Single_Trait_GBLUP_Full', 'Training_Mean']
    palette_base = ['#17becf', '#1f77b4', '#2ca02c', '#7f7f7f']
    df_base = master_df.set_index('Model').loc[ordered_base_models].reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # Panel A: Fold-Level Macro R2 (Mean ± Fold SD)
    bars = ax1.bar(range(len(df_base)), df_base['Fold_Macro_R2_Mean'],
                   yerr=df_base['Fold_Macro_R2_Std'], capsize=5,
                   color=palette_base, edgecolor='black', alpha=0.88)
    ax1.set_xticks(range(len(df_base)))
    ax1.set_xticklabels([m.replace('_', ' ') for m in df_base['Model']], rotation=20, ha='right', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Held-Out Macro $R^2$ (Fold Mean ± SD across 15 Folds)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Predictive Accuracy Across Classical Baselines (15 Folds, Target-Only)', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.axhline(0, color='black', linewidth=1)
    
    for bar, (_, r) in zip(bars, df_base.iterrows()):
        yval = bar.get_height()
        va_pos = 'bottom' if yval >= 0 else 'top'
        y_pos = yval + 0.005 if yval >= 0 else yval - 0.015
        p_val = r['Primary_Repeat_Pooled_Macro_R2']
        ax1.text(bar.get_x() + bar.get_width()/2.0, y_pos, f"Mean: {yval:.4f}\nPooled: {p_val:.4f}", ha='center', va=va_pos, fontsize=8, fontweight='bold')
        
    # Panel B: Trait Breakdown with explicit Patch legend
    pivot_trait = per_trait_df[per_trait_df['Model'].isin(ordered_base_models)].pivot(index='Trait', columns='Model', values='R2')
    pivot_trait = pivot_trait[ordered_base_models]
    pivot_trait.index = [t.split('_')[0] + ' ' + ' '.join(t.split('_')[1:3]) for t in pivot_trait.index]
    pivot_trait.plot(kind='bar', ax=ax2, color=palette_base, edgecolor='black', alpha=0.88, width=0.8)
    ax2.set_ylabel('Held-Out $R^2$ (Repeat-Pooled)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Target Stress Traits', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Trait-Specific Accuracy Across Classical Baselines', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=20, ha='right', fontsize=9, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.axhline(0, color='black', linewidth=1)
    
    legend_elements_base = [
        Patch(facecolor='#17becf', edgecolor='black', label='Helper Stacking Ridge OOF'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='Single-Trait RidgeCV'),
        Patch(facecolor='#2ca02c', edgecolor='black', label='Single-Trait GBLUP Full'),
        Patch(facecolor='#7f7f7f', edgecolor='black', label='Training Mean')
    ]
    ax2.legend(handles=legend_elements_base, title='Model', frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "fig2_classical_baselines.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig2_classical_baselines.pdf", bbox_inches='tight')
    plt.close()
    print("Saved Figure 2.")

# -----------------------------------------------------------------------------
# FIGURE 3: Neural Multi-Task Learning Benchmark
# -----------------------------------------------------------------------------
def plot_figure_3():
    print("Generating Figure 3: Neural MTL Benchmark...")
    master_df = pd.read_csv(artifacts_dir / "master_leaderboard_table.csv")
    per_trait_df = pd.read_csv(artifacts_dir / "per_trait_native_metrics.csv")
    
    compare_models = ['Single_Trait_GBLUP_Full', 'Single_Trait_RidgeCV', 'Standardized_MTL', 'Uncertainty_Weighted_MTL', 'Naive_MTL']
    df_cmp = master_df[master_df['Model'].isin(compare_models)].copy()
    df_cmp['Model'] = pd.Categorical(df_cmp['Model'], categories=compare_models, ordered=True)
    df_cmp = df_cmp.sort_values('Model')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    colors3 = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#e377c2']
    
    # Panel A: Fold-Level Macro R2 Comparison
    bars = ax1.bar(range(len(df_cmp)), df_cmp['Fold_Macro_R2_Mean'],
                   yerr=df_cmp['Fold_Macro_R2_Std'], capsize=5,
                   color=colors3, edgecolor='black', alpha=0.88)
    ax1.set_xticks(range(len(df_cmp)))
    ax1.set_xticklabels([m.replace('_', ' ') for m in df_cmp['Model']], rotation=20, ha='right', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Held-Out Macro $R^2$ (Fold Mean ± SD across 15 Folds)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Neural MTL vs Classical Baselines (15 Folds, Target-Only Masking)', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.axhline(0, color='black', linewidth=1)
    
    for bar, (_, r) in zip(bars, df_cmp.iterrows()):
        yval = bar.get_height()
        va_pos = 'bottom' if yval >= 0 else 'top'
        y_pos = yval + 0.005 if yval >= 0 else yval - 0.015
        p_val = r['Primary_Repeat_Pooled_Macro_R2']
        ax1.text(bar.get_x() + bar.get_width()/2.0, y_pos, f"Mean: {yval:.4f}\nPooled: {p_val:.4f}", ha='center', va=va_pos, fontsize=8, fontweight='bold')
        
    # Panel B: Neural Trait-Specific Breakdown with explicit Patch legend
    neu_plot_models = ['Naive_MTL', 'Standardized_MTL', 'Uncertainty_Weighted_MTL']
    neu_colors = ['#e377c2', '#ff7f0e', '#d62728'] # pink, orange, red
    pivot_neu = per_trait_df[per_trait_df['Model'].isin(neu_plot_models)].pivot(index='Trait', columns='Model', values='R2')
    pivot_neu = pivot_neu[neu_plot_models]
    pivot_neu.index = [t.split('_')[0] + ' ' + ' '.join(t.split('_')[1:3]) for t in pivot_neu.index]
    pivot_neu.plot(kind='bar', ax=ax2, color=neu_colors, edgecolor='black', alpha=0.88, width=0.75)
    ax2.set_ylabel('Held-Out $R^2$ (Repeat-Pooled)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Target Stress Traits', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Stabilization of Sparse Predictions Across Neural Variants', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=20, ha='right', fontsize=9, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.axhline(0, color='black', linewidth=1)
    
    legend_elements_neu = [
        Patch(facecolor='#e377c2', edgecolor='black', label='Naive MTL'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Standardized MTL'),
        Patch(facecolor='#d62728', edgecolor='black', label='Uncertainty-Weighted MTL')
    ]
    ax2.legend(handles=legend_elements_neu, title='Neural Variant', frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "fig3_neural_mtl_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig3_neural_mtl_comparison.pdf", bbox_inches='tight')
    plt.close()
    print("Saved Figure 3.")

# -----------------------------------------------------------------------------
# FIGURE 4: Genuinely Independent 2^3 Factorial Component Ablation
# -----------------------------------------------------------------------------
def plot_figure_4():
    print("Generating Figure 4: Factorial Ablation...")
    df_abl = pd.read_csv(artifacts_dir / "factorial_ablation_summary.csv")
    df_eff = pd.read_csv(artifacts_dir / "factorial_ablation_effects.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6), dpi=300)
    
    # Panel A: Leaderboard across 8 orthogonal conditions
    bars = ax1.barh(range(len(df_abl))[::-1], df_abl['Mean'], xerr=df_abl['Std'], capsize=4,
                    color='#3b528b', edgecolor='black', alpha=0.88, height=0.65)
    ax1.set_yticks(range(len(df_abl))[::-1])
    ax1.set_yticklabels(df_abl['Condition'], fontsize=9, fontweight='bold')
    ax1.set_xlabel('Held-Out Macro $R^2$ (Fold Mean ± SD across 15 Folds)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) 2^3 Factorial Ablation: All 8 Orthogonal Configurations', fontsize=12, fontweight='bold', pad=10)
    ax1.axvline(0, color='black', linewidth=1)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    
    ax1.set_xlim(-0.09, 0.17)
    for i, bar in enumerate(bars):
        xval = df_abl['Mean'].iloc[i]
        sd_val = df_abl['Std'].iloc[i]
        if xval >= 0:
            x_pos = xval + sd_val + 0.003
            ha = 'left'
        else:
            x_pos = xval - sd_val - 0.003
            ha = 'right'
        ax1.text(x_pos, len(df_abl) - 1 - i, f"{xval:.4f}", va='center', ha=ha, fontsize=8, fontweight='bold')
        
    # Panel B: Orthogonal Yates Contrast Analysis (Main Effects & Interactions with 95% CIs)
    y_pos = range(len(df_eff))[::-1]
    eff_mid = df_eff['Estimate']
    xerr_low = np.maximum(0, eff_mid - df_eff['CI95_Lower'])
    xerr_high = np.maximum(0, df_eff['CI95_Upper'] - eff_mid)
    
    # Use Holm-adjusted significance
    has_holm_col = 'Significant_Holm_0.05' in df_eff.columns
    sig_col = 'Significant_Holm_0.05' if has_holm_col else 'Significant_0.05'
    colors_eff = ['#2ca02c' if sig else '#7f7f7f' for sig in df_eff[sig_col]]
    
    ax2.errorbar(eff_mid, y_pos, xerr=[xerr_low, xerr_high], fmt='o',
                 color='#1f77b4', ecolor='#d62728', elinewidth=2.0, capsize=5, markersize=7)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_eff['Effect'], fontsize=8, fontweight='bold')
    ax2.set_xlabel(r'Estimated Factor Effect on Macro $R^2$ (95% CI across 15 Folds)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Main Effects & Interactions (Unadjusted 95% CIs; Holm-Adjusted p-values)', fontsize=12, fontweight='bold', pad=10)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1.2)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    
    for i, r in df_eff.iterrows():
        y_idx = len(df_eff) - 1 - i
        is_sig = r[sig_col]
        p_h = r.get('Holm_Adjusted_p_val', r.get('Permutation_Exact_p_val', np.nan))
        sig_flag = f" (p_Holm={p_h:.4f}*)" if is_sig else f" (p_Holm={p_h:.3f}, NS)"
        ax2.text(r['CI95_Upper'] + 0.002, y_idx, f"[{r['CI95_Lower']:+.3f}, {r['CI95_Upper']:+.3f}]{sig_flag}",
                 va='center', fontsize=8, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(figures_dir / "fig4_factorial_ablation.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig4_factorial_ablation.pdf", bbox_inches='tight')
    plt.close()
    print("Saved Figure 4.")

# -----------------------------------------------------------------------------
# FIGURE 5: Subgroup Robustness Across Inductive Clusters
# -----------------------------------------------------------------------------
def plot_figure_5():
    print("Generating Figure 5: Subgroup Robustness...")
    master_df = pd.read_csv(artifacts_dir / "master_leaderboard_table.csv")
    subgroup_df = pd.read_csv(artifacts_dir / "subgroup_robustness_all_models.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    # Panel A: Worst-Group NRMSE
    df_sorted = master_df.sort_values('Worst_Group_NRMSE_Mean')
    colors = ['#2ca02c' if 'GBLUP' in m else '#1f77b4' if 'Ridge' in m else '#9467bd' if 'DRO' in m else '#ff7f0e' for m in df_sorted['Model']]
    bars = ax1.barh(range(len(df_sorted))[::-1], df_sorted['Worst_Group_NRMSE_Mean'],
                    color=colors, edgecolor='black', alpha=0.88, height=0.65)
    ax1.set_yticks(range(len(df_sorted))[::-1])
    ax1.set_yticklabels([m.replace('_', ' ') for m in df_sorted['Model']], fontsize=9, fontweight='bold')
    ax1.set_xlabel('Worst-Group Normalized RMSE (Lower is Better)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Worst-Subgroup Error Across 4 Aligned Clusters', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    
    for i, bar in enumerate(bars):
        xval = df_sorted['Worst_Group_NRMSE_Mean'].iloc[i]
        ax1.text(xval + 0.01, len(df_sorted) - 1 - i, f"{xval:.4f}", va='center', fontsize=8, fontweight='bold')
        
    # Panel B: Heatmap of NRMSE across Clusters
    pivot_sub = subgroup_df.pivot(index='Model', columns='Cluster', values='Trait_NRMSE_Mean')
    pivot_sub = pivot_sub.loc[df_sorted['Model']]
    pivot_sub.columns = [f"Cluster {c}" for c in pivot_sub.columns]
    
    sns.heatmap(pivot_sub, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Normalized RMSE'},
                annot_kws={'size': 8, 'weight': 'bold'}, linewidths=0.5)
    ax2.set_yticklabels([m.replace('_', ' ') for m in pivot_sub.index], fontsize=8, fontweight='bold')
    ax2.set_title('(B) Subgroup Predictive Error (NRMSE) by Hungarian-Aligned Cluster', fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "fig5_subgroup_robustness.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig5_subgroup_robustness.pdf", bbox_inches='tight')
    plt.close()
    print("Saved Figure 5.")

# -----------------------------------------------------------------------------
# FIGURE 6: Phenotype Sparsity Decay Curves (Secondary Exploratory)
# -----------------------------------------------------------------------------
def plot_figure_6():
    print("Generating Figure 6: Sparsity Sensitivity...")
    df_sens = pd.read_csv(artifacts_dir / "sparsity_sensitivity_summary.csv")
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    palette = {'Single_Trait_GBLUP_Full': '#2ca02c', 'Single_Trait_RidgeCV': '#1f77b4', 'Uncertainty_Weighted_MTL': '#d62728'}
    
    for m in df_sens['Model'].unique():
        sub = df_sens[df_sens['Model'] == m].sort_values('Retention_Ratio')
        ax.plot(sub['Retention_Ratio'], sub['Mean'], marker='o', linewidth=2.5, markersize=8,
                label=m.replace('_', ' '), color=palette.get(m, '#333'))
        ax.fill_between(sub['Retention_Ratio'], sub['Mean'] - sub['Std'], sub['Mean'] + sub['Std'],
                        alpha=0.15, color=palette.get(m, '#333'))
        
    ax.set_xlabel('Target Trait Label Retention Ratio (1.0 = 100%, 0.2 = 20%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Held-Out Macro $R^2$ (Mean ± Fold SD)', fontsize=11, fontweight='bold')
    ax.set_title('Phenotype Sparsity Decay Curves (Secondary Exploratory Sensitivity, 5 Folds)', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.legend(frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "fig6_sparsity_decay_curves.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig6_sparsity_decay_curves.pdf", bbox_inches='tight')
    plt.close()
    print("Saved Figure 6.")

# -----------------------------------------------------------------------------
# FIGURE 7: Leave-One-Genotype-Cluster-Out (LOGCO) Generalization
# -----------------------------------------------------------------------------
def plot_figure_7():
    print("Generating Figure 7: LOGCO Generalization...")
    df_lopo = pd.read_csv(artifacts_dir / "lopo_results_per_cluster.csv")
    master_df = pd.read_csv(artifacts_dir / "master_leaderboard_table.csv")
    
    models = ['Single_Trait_RidgeCV', 'Single_Trait_GBLUP_Full', 'Uncertainty_Weighted_MTL']
    df_lopo_sub = df_lopo[df_lopo['Model'].isin(models)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    # Panel A: Within-Cohort CV vs LOGCO Out-of-Distribution Comparison
    within_r2 = [master_df[master_df['Model'] == m]['Primary_Repeat_Pooled_Macro_R2'].values[0] for m in models]
    lopo_r2 = [df_lopo_sub[df_lopo_sub['Model'] == m]['LOPO_Macro_R2'].mean() for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width/2, within_r2, width, label='Standard 5-Fold CV (Within-Cohort)', color='#1f77b4', edgecolor='black', alpha=0.88)
    ax1.bar(x + width/2, lopo_r2, width, label='LOGCO (Out-of-Distribution)', color='#d62728', edgecolor='black', alpha=0.88)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ') for m in models], rotation=15, ha='right', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Macro $R^2$', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Within-Cohort CV vs LOGCO Accuracy Collapse', fontsize=12, fontweight='bold', pad=10)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(frameon=True, fontsize=9)
    
    # Panel B: LOGCO Performance Breakdown by Cluster with explicit Patch legend
    logco_models = ['Single_Trait_RidgeCV', 'Single_Trait_GBLUP_Full', 'Uncertainty_Weighted_MTL']
    logco_colors = ['#1f77b4', '#2ca02c', '#d62728'] # blue, green, red
    pivot_lopo = df_lopo_sub.pivot(index='Holdout_Cluster', columns='Model', values='LOPO_Macro_R2')
    pivot_lopo.index = [f"Cluster {c}" for c in pivot_lopo.index]
    pivot_lopo = pivot_lopo[logco_models]
    pivot_lopo.plot(kind='bar', ax=ax2, color=logco_colors, edgecolor='black', alpha=0.88)
    ax2.set_ylabel('LOGCO Held-Out Macro $R^2$', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Holdout Genotype-Derived Cluster (Defined on Full Genotypes)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) LOGCO Generalization Accuracy by Holdout Cluster', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, fontsize=9, fontweight='bold')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    legend_elements_logco = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='Single-Trait RidgeCV'),
        Patch(facecolor='#2ca02c', edgecolor='black', label='Single-Trait GBLUP Full'),
        Patch(facecolor='#d62728', edgecolor='black', label='Uncertainty-Weighted MTL')
    ]
    ax2.legend(handles=legend_elements_logco, title='Model', frameon=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "fig7_population_generalization.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig7_population_generalization.pdf", bbox_inches='tight')
    plt.close()
    print("Saved Figure 7.")

# -----------------------------------------------------------------------------
# FIGURE 8: Master Leaderboard & Forest Plot of Contrasts
# -----------------------------------------------------------------------------
def plot_figure_8():
    print("Generating Figure 8: Master Leaderboard & Contrasts...")
    master_df = pd.read_csv(artifacts_dir / "master_leaderboard_table.csv")
    df_contrasts = pd.read_csv(artifacts_dir / "statistical_contrasts.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
    
    cat_palette = {
        'Genomic Kernel (404k SNPs)': '#2ca02c',
        'Linear Regularized (120 PCs)': '#1f77b4',
        'Linear Transfer (OOF Stacked)': '#17becf',
        'Neural Multi-Task (Standardized)': '#ff7f0e',
        'Neural Multi-Task (Uncertainty-Weighted)': '#d62728',
        'Robust Multi-Task (Group-DRO)': '#9467bd',
        'Robust Multi-Task (CVaR Tail-Risk)': '#8c564b',
        'Sanity Baseline (Sample Mean)': '#7f7f7f',
        'Neural Multi-Task (Naive Unweighted)': '#e377c2'
    }
    
    # Panel A: Fold-Level Macro R2 Mean ± Fold SD across 15 folds (with Primary Repeat-Pooled Point annotated)
    colors_lead = [cat_palette.get(c, '#333') for c in master_df['Category']]
    bars = ax1.barh(range(len(master_df))[::-1], master_df['Fold_Macro_R2_Mean'],
                    xerr=master_df['Fold_Macro_R2_Std'], capsize=4,
                    color=colors_lead, edgecolor='black', alpha=0.88, height=0.65)
    
    ax1.set_yticks(range(len(master_df))[::-1])
    ax1.set_yticklabels([m.replace('_', ' ') for m in master_df['Model']], fontsize=9, fontweight='bold')
    ax1.set_xlabel('Held-Out Macro $R^2$ (Fold Mean ± SD across 15 Folds)', fontsize=11, fontweight='bold')
    ax1.set_title('(A) Master Predictive Accuracy Leaderboard (15 Folds, Target-Only)', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='black', linewidth=1)
    
    for i, (bar, (_, r)) in enumerate(zip(bars, master_df.iterrows())):
        xval = bar.get_width()
        xerr = r['Fold_Macro_R2_Std']
        whisker_right = xval + xerr
        # Move annotation cleanly beyond the error-bar whiskers
        x_pos = whisker_right + 0.005 if xval >= 0 else max(0.006, whisker_right + 0.005)
        ha = 'left'
        p_val = r['Primary_Repeat_Pooled_Macro_R2']
        ax1.text(x_pos, len(master_df) - 1 - i, f"Mean: {xval:.4f} (Pooled: {p_val:.4f})", va='center', ha=ha, fontsize=8, fontweight='bold')
        
    ax1.set_xlim(-0.06, 0.285)
    cat_legend = [Patch(facecolor=color, edgecolor='black', label=cat) for cat, color in cat_palette.items() if cat in master_df['Category'].values]
    ax1.legend(handles=cat_legend, loc='lower right', frameon=True, fontsize=7.2, framealpha=0.92)
    
    # Panel B: Forest Plot of Contrasts (Accession-Level Bootstrap 95% CIs)
    y_pos = range(len(df_contrasts))[::-1]
    boot_mid = df_contrasts['Delta_Macro_R2_Primary']
    xerr_low = np.maximum(0, boot_mid - df_contrasts['CI95_Lower'])
    xerr_high = np.maximum(0, df_contrasts['CI95_Upper'] - boot_mid)
    
    ax2.errorbar(boot_mid, y_pos, 
                 xerr=[xerr_low, xerr_high],
                 fmt='o', color='#1f77b4', ecolor='#d62728', elinewidth=2.5, capsize=6, markersize=8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(df_contrasts['Contrast'], fontsize=9, fontweight='bold')
    ax2.set_xlabel(r'Primary $\Delta \mathrm{Macro } R^2$ (Unadjusted 95% Paired Accession-Bootstrap CI, $B=10,000$ Resamples)', fontsize=11, fontweight='bold')
    ax2.set_title('(B) Primary Contrasts: Paired Accession Bootstrap CIs; Holm-Adjusted p-values', fontsize=12, fontweight='bold', pad=10)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1.2)
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    
    has_prim_sig = 'Primary_Significant_0.05' in df_contrasts.columns
    sig_col_c = 'Primary_Significant_0.05' if has_prim_sig else 'Significant_Holm_0.05'
    holm_col_c = 'Primary_Holm_p_val' if 'Primary_Holm_p_val' in df_contrasts.columns else 'Holm_Bonferroni_p_val'
    
    for i, r in df_contrasts.iterrows():
        y_idx = len(df_contrasts) - 1 - i
        is_sig = r[sig_col_c]
        p_val = r[holm_col_c]
        sig_str = f" (p_Holm={p_val:.4f}*)" if is_sig else " (NS)"
        ax2.text(r['CI95_Upper'] + 0.002, y_idx, f"[{r['CI95_Lower']:+.3f}, {r['CI95_Upper']:+.3f}]{sig_str}", va='center', fontsize=8, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(figures_dir / "fig8_master_leaderboard_contrasts.png", dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / "fig8_master_leaderboard_contrasts.pdf", bbox_inches='tight')
    plt.close()
    print("Saved Figure 8.")

def generate_all_figures():
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()
    plot_figure_7()
    plot_figure_8()
    print("All 8 publication figures generated successfully in PNG and PDF.")

if __name__ == '__main__':
    generate_all_figures()
