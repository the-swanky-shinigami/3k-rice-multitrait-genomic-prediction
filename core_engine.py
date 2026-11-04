import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from scipy.optimize import linear_sum_assignment
from pandas_plink import read_plink1_bin

# Device configuration
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# -----------------------------------------------------------------------------
# 0. LEARNED PER-MARKER MEAN IMPUTER (CRITIQUE SECTION 5)
# -----------------------------------------------------------------------------
class MarkerMeanImputer:
    """
    Strictly in-fold per-marker mean imputer.
    Computes column means strictly on training partition genotypes (ignoring NaNs).
    If a marker is entirely NaN in the training set, falls back to fallback_val (1.0).
    Imputes missing values in training, validation, and test partitions using only
    the fitted training column means, eliminating constant-imputation leakage.
    """
    def __init__(self, fallback_val=1.0):
        self.fallback_val = float(fallback_val)
        self.means_ = None

    def fit(self, X_train):
        means = np.nanmean(X_train, axis=0)
        means = np.where(np.isnan(means), self.fallback_val, means)
        self.means_ = means.astype(np.float32)
        return self

    def transform(self, X):
        if self.means_ is None:
            raise ValueError("MarkerMeanImputer must be fitted before transform.")
        X_out = np.copy(X)
        nan_mask = np.isnan(X_out)
        if np.any(nan_mask):
            X_out[nan_mask] = np.take(self.means_, np.where(nan_mask)[1])
        return X_out

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)

# -----------------------------------------------------------------------------
# 1. DATA AUDIT & PHENOTYPE CONSOLIDATION (CRITIQUE SECTION 1 & 4)
# -----------------------------------------------------------------------------
def load_and_audit_data(data_dir):
    """
    Loads genotype and phenotype matrices with rigorous cross-coordinate audits:
    1. Verifies (FID, IID) 1-to-1 matching between FAM and phenotype CSV.
    2. Explicitly verifies PLINK genotype array sample coordinates (G.fid, G.iid).
    3. Resolves MMC4 duplicate column and computes true biological ratio H4 (GL / GW).
    """
    fam_path = data_dir / "base_filtered_v0.7_0.8_10kb_1_0.8_50_1.fam"
    pheno_path = data_dir / "base_filtered_v0.7_0.8_10kb_1_0.8_50_1_consolidated_phenotypes.csv"
    bed_path = data_dir / "base_filtered_v0.7_0.8_10kb_1_0.8_50_1.bed"
    
    fam = pd.read_csv(fam_path, sep=r'\s+', header=None, names=['FID','IID','PID','MID','SEX','PHENO'])
    ph = pd.read_csv(pheno_path)
    
    fam['FID'] = fam['FID'].astype(str)
    fam['IID'] = fam['IID'].astype(str)
    ph['FID'] = ph['FID'].astype(str)
    ph['IID'] = ph['IID'].astype(str)
    
    # Assertions on sample integrity
    assert len(fam) == len(ph) == 3024, f"Sample count mismatch: FAM={len(fam)}, Phenotypes={len(ph)}"
    assert not fam['IID'].duplicated().any(), "Duplicate IID found in FAM file!"
    assert not ph['IID'].duplicated().any(), "Duplicate IID found in Phenotypes CSV!"
    assert (fam['FID'].values == ph['FID'].values).all(), "FID order mismatch between FAM and Phenotypes!"
    assert (fam['IID'].values == ph['IID'].values).all(), "IID order mismatch between FAM and Phenotypes!"
    
    # Trait column names in consolidated table
    col_tgw = '1_s2_0_S2590346225001312_mmc3__Thousand_grain_weight__Thousand_grain_weight'
    col_gl  = '1_s2_0_S2590346225001312_mmc4__Grain_length__Grain_length'
    col_gw  = '1_s2_0_S2590346225001312_mmc4__Grain_width__Grain_width'
    col_dup_ratio = '1_s2_0_S2590346225001312_mmc4__Ratio_of_grain_length_to_grain__Ratio_of_grain_length_to_grain_width'
    
    col_drought = '1_s2_0_S2590346225001312_mmc5__Drought_tolerance__Drought_resistance_grade'
    col_heat_surv = '1_s2_0_S2590346225001312_mmc5__Heat_tolerance__Survival_rate'
    col_heat_leaf = '1_s2_0_S2590346225001312_mmc5__Heat_tolerance__Leaf_score_of_heat_tolerance'
    col_sb_lh = '1_s2_0_S2590346225001312_mmc6__Sheath_blight__Lesion_height'
    col_sb_rlh = '1_s2_0_S2590346225001312_mmc6__Sheath_blight__Relative_lesion_height'
    
    # Verify MMC4 duplicate column
    max_dup_diff = float((ph[col_gw] - ph[col_dup_ratio]).abs().max())
    assert max_dup_diff == 0.0, f"MMC4 duplicate ratio does not match Grain Width: diff={max_dup_diff}"
    
    # Compute true ratio GL / GW
    assert (ph[col_gw] == 0).sum() == 0, "Zero values found in Grain Width!"
    true_ratio = ph[col_gl] / ph[col_gw]
    
    canonical_df = pd.DataFrame({
        'FID': ph['FID'],
        'IID': ph['IID'],
        'H1_Thousand_Grain_Weight': ph[col_tgw],
        'H2_Grain_Length': ph[col_gl],
        'H3_Grain_Width': ph[col_gw],
        'H4_Grain_Length_Width_Ratio': true_ratio,
        'T1_Drought_Resistance_Grade': ph[col_drought],
        'T2_Heat_Survival_Rate': ph[col_heat_surv],
        'T3_Heat_Leaf_Score': ph[col_heat_leaf],
        'T4_Sheath_Blight_Lesion_Height': ph[col_sb_lh],
        'T5_Sheath_Blight_Relative_Lesion_Height': ph[col_sb_rlh],
    })
    
    # Genotypes
    G = read_plink1_bin(str(bed_path), ref="a0", verbose=False)
    
    # CRITIQUE SECTION 4: Explicitly audit genotype array coordinate ordering
    g_fid = G.fid.values.astype(str)
    g_iid = G.iid.values.astype(str)
    assert len(g_iid) == 3024, f"Genotype sample count mismatch: {len(g_iid)} vs 3024"
    assert (g_fid == fam['FID'].values).all(), "Genotype array FID coordinate mismatch with FAM file!"
    assert (g_iid == fam['IID'].values).all(), "Genotype array IID coordinate mismatch with FAM file!"
    assert (g_fid == ph['FID'].values).all(), "Genotype array FID coordinate mismatch with Phenotypes table!"
    assert (g_iid == ph['IID'].values).all(), "Genotype array IID coordinate mismatch with Phenotypes table!"
    
    # METHODOLOGICAL DISCLOSURE:
    # The 10,000-marker panel is selected by taking evenly spaced indices across the full 404,388 SNPs.
    # It represents a deterministic, uniform physical grid used for the 120-PC feature representation
    # in RidgeCV, Helper Stacking, and Neural MTL models.
    # By contrast, Single-Trait GBLUP Full uses the complete 404,388-marker VanRaden relationship matrix.
    idx_10k = np.linspace(0, G.shape[1] - 1, 10000, dtype=int)
    X_10k = G.isel(variant=idx_10k).values.astype(np.float32)
    X_full = G.values.astype(np.float32)
    
    return canonical_df, X_10k, X_full, max_dup_diff

# -----------------------------------------------------------------------------
# 2. TARGET-ONLY MASKING FUNCTION (CRITIQUE SECTION 2)
# -----------------------------------------------------------------------------
def generate_target_only_masks(Y_train, obs_ratio, seed, helper_indices=[0, 1, 2, 3], target_indices=[4, 5, 6, 7, 8]):
    """
    Primary Experiment:
    Preserves 100% of naturally observed helper labels (H1-H4) in the training fold.
    Artificially retains only obs_ratio of naturally observed target labels (T1-T5),
    masking the remaining (1 - obs_ratio).
    """
    rng = np.random.RandomState(seed)
    Y_m = Y_train.copy()
    M_orig = (~np.isnan(Y_train)).astype(np.float32)
    M_m = M_orig.copy()
    
    mask_stats = []
    # Helpers: no artificial masking
    for h_idx in helper_indices:
        valid_idx = np.where(~np.isnan(Y_train[:, h_idx]))[0]
        mask_stats.append({
            'trait_idx': h_idx,
            'is_target': False,
            'n_orig_train': len(valid_idx),
            'n_kept_train': len(valid_idx),
            'n_masked_train': 0
        })
        
    # Targets: retain obs_ratio fraction of observed labels
    for t_idx in target_indices:
        valid_idx = np.where(~np.isnan(Y_train[:, t_idx]))[0]
        n_valid = len(valid_idx)
        n_keep = int(np.round(n_valid * obs_ratio))
        if n_keep > 0 and obs_ratio < 1.0:
            kept_idx = rng.choice(valid_idx, size=n_keep, replace=False)
            masked_idx = np.setdiff1d(valid_idx, kept_idx)
            Y_m[masked_idx, t_idx] = np.nan
            M_m[masked_idx, t_idx] = 0.0
        else:
            masked_idx = np.array([], dtype=int)
            n_keep = n_valid
            
        mask_stats.append({
            'trait_idx': t_idx,
            'is_target': True,
            'n_orig_train': n_valid,
            'n_kept_train': n_keep,
            'n_masked_train': len(masked_idx)
        })
        
    return Y_m, M_m, M_orig, mask_stats

def generate_all_traits_masks(Y_train, obs_ratio, seed):
    """Secondary Sensitivity Analysis: Artificially masks ALL 9 traits."""
    rng = np.random.RandomState(seed)
    Y_m = Y_train.copy()
    M_orig = (~np.isnan(Y_train)).astype(np.float32)
    M_m = M_orig.copy()
    for j in range(Y_train.shape[1]):
        valid_idx = np.where(~np.isnan(Y_train[:, j]))[0]
        n_valid = len(valid_idx)
        n_keep = int(np.round(n_valid * obs_ratio))
        if n_keep > 0 and obs_ratio < 1.0:
            kept_idx = rng.choice(valid_idx, size=n_keep, replace=False)
            masked_idx = np.setdiff1d(valid_idx, kept_idx)
            Y_m[masked_idx, j] = np.nan
            M_m[masked_idx, j] = 0.0
    return Y_m, M_m, M_orig

# -----------------------------------------------------------------------------
# 3. COHORT-REFERENCE ALIGNMENT OF GENOTYPE-DERIVED CLUSTERS (CRITIQUE SECTION 8)
# -----------------------------------------------------------------------------
def compute_inductive_clusters(X_train_raw, X_test_raw, seed, n_clusters=4, ref_centroids=None):
    """
    Cohort-reference alignment of genotype-derived clusters:
    1. Local training step: Fits MarkerMeanImputer, PCA(20), and KMeans(4) on outer-training genotypes ONLY.
    2. Local prediction: Transforms outer-test genotypes using the outer-training pipeline.
    3. Hungarian alignment: When ref_centroids is provided (unsupervised cohort reference centroids derived
       from all cohort genotypes in genotype space without phenotype labels), local cluster centroids are
       aligned to the fixed cohort reference via the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
    *METHODOLOGICAL CLARIFICATION*:
       Hungarian alignment resolves arbitrary cluster label switching / permutation across folds.
       It does NOT establish that accession cluster memberships are identical across folds or that clusters
       represent validated biological lineages; it provides consistent coordinate labeling across cross-validation splits.
    """
    imputer = MarkerMeanImputer()
    X_tr_imp = imputer.fit_transform(X_train_raw)
    X_te_imp = imputer.transform(X_test_raw)
    
    pca = PCA(n_components=20, random_state=seed)
    pcs_tr = pca.fit_transform(X_tr_imp)
    pcs_te = pca.transform(X_te_imp)
    
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    tr_clusters_raw = km.fit_predict(pcs_tr)
    te_clusters_raw = km.predict(pcs_te)
    
    if ref_centroids is not None:
        if ref_centroids.shape[1] == pcs_tr.shape[1]:
            local_centroids = km.cluster_centers_
        else:
            local_centroids = np.array([X_tr_imp[tr_clusters_raw == c].mean(axis=0) for c in range(n_clusters)])
        cost_matrix = np.linalg.norm(local_centroids[:, None, :] - ref_centroids[None, :, :], axis=2)
        _, col_ind = linear_sum_assignment(cost_matrix)
        mapping = {old_c: new_c for old_c, new_c in enumerate(col_ind)}
        tr_clusters = np.array([mapping[c] for c in tr_clusters_raw], dtype=int)
        te_clusters = np.array([mapping[c] for c in te_clusters_raw], dtype=int)
    else:
        tr_clusters = tr_clusters_raw
        te_clusters = te_clusters_raw
        
    return tr_clusters, te_clusters

# -----------------------------------------------------------------------------
# 4. NESTED PREPROCESSING (CRITIQUE SECTION 4 & 5)
# -----------------------------------------------------------------------------
def fit_outer_preprocessing(X_10k_train, X_10k_test, seed, n_components=120):
    """Fits MarkerMeanImputer, StandardScaler, and PCA strictly on outer training genotypes."""
    imputer = MarkerMeanImputer()
    X_tr_imp = imputer.fit_transform(X_10k_train)
    X_te_imp = imputer.transform(X_10k_test)
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_imp)
    X_te_scaled = scaler.transform(X_te_imp)
    
    pca = PCA(n_components=n_components, random_state=seed)
    X_tr_pca = pca.fit_transform(X_tr_scaled)
    X_te_pca = pca.transform(X_te_scaled)
    return X_tr_pca, X_te_pca

def compute_phenotype_scalers(Y_train_masked):
    """Computes mean and std strictly on visible training labels."""
    mu = np.zeros(Y_train_masked.shape[1], dtype=np.float32)
    sd = np.ones(Y_train_masked.shape[1], dtype=np.float32)
    for j in range(Y_train_masked.shape[1]):
        col = Y_train_masked[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            mu[j] = np.mean(valid)
            s = np.std(valid)
            sd[j] = s if s > 1e-4 else 1.0
    return mu, sd

# -----------------------------------------------------------------------------
# 5. CLASSICAL BASELINES IMPLEMENTATIONS (CRITIQUE SECTION 6 & 7)
# -----------------------------------------------------------------------------
def fit_predict_gblup(X_full_train, X_full_test, Y_train_masked, M_train_masked, target_indices=[4, 5, 6, 7, 8]):
    """
    Exact Henderson/VanRaden GBLUP on all 404,388 SNPs.
    Training allele frequencies p_j computed strictly on training accessions (ignoring NaNs).
    Missing genotypes imputed by training marker means (giving deviation = 0 in centered Z).
    Spectral variance tuning (delta) via 5-fold inner CV with strict inner-train outcome centering.
    """
    n_tr = len(X_full_train)
    n_te = len(X_full_test)
    
    # 1. In-fold allele frequencies (ignoring NaNs)
    p = np.nanmean(X_full_train, axis=0) / 2.0
    p = np.clip(p, 0.005, 0.995)
    
    # Center non-NaN genotypes by 2*p, and impute missing entries by training mean (giving deviation 0.0)
    Z_tr = np.nan_to_num(X_full_train - 2.0 * p, nan=0.0)
    Z_te = np.nan_to_num(X_full_test - 2.0 * p, nan=0.0)
    
    denom = 2.0 * np.sum(p * (1.0 - p))
    if denom <= 0:
        denom = 1.0
        
    K_tr = np.dot(Z_tr, Z_tr.T) / denom + np.eye(n_tr) * 1e-4
    K_te_tr = np.dot(Z_te, Z_tr.T) / denom
    
    preds_test = np.zeros((n_te, len(target_indices)), dtype=np.float32)
    
    for idx_out, t_idx in enumerate(target_indices):
        y_col = Y_train_masked[:, t_idx]
        mask_col = M_train_masked[:, t_idx] > 0
        y_valid = y_col[mask_col]
        
        if len(y_valid) < 5:
            preds_test[:, idx_out] = np.mean(y_valid) if len(y_valid) > 0 else 0.0
            continue
            
        # Sub-matrix for observed training accessions
        K_obs = K_tr[np.ix_(mask_col, mask_col)]
        obs_indices = np.where(mask_col)[0]
        
        # Inner CV to tune delta with strict inner-train centering (zero inner-val leakage)
        kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
        delta_candidates = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
        best_delta = 1.0
        best_mse = float('inf')
        
        for delta in delta_candidates:
            cv_mses = []
            for in_tr, in_va in kf_inner.split(obs_indices):
                y_in_tr_raw = y_valid[in_tr]
                y_in_va_raw = y_valid[in_va]
                
                # Strict inner-train outcome centering
                in_tr_mean = float(np.mean(y_in_tr_raw))
                y_in_tr_c = y_in_tr_raw - in_tr_mean
                
                K_in_tr = K_obs[np.ix_(in_tr, in_tr)]
                K_in_va_tr = K_obs[np.ix_(in_va, in_tr)]
                
                alpha = np.linalg.solve(K_in_tr + delta * np.eye(len(in_tr)), y_in_tr_c)
                p_in_va = np.dot(K_in_va_tr, alpha) + in_tr_mean # Restore inner-train mean
                cv_mses.append(float(np.mean((y_in_va_raw - p_in_va) ** 2)))
                
            mean_mse = float(np.mean(cv_mses))
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_delta = delta
                
        # Fit on full observed training accessions with full training mean
        y_full_mean = float(np.mean(y_valid))
        y_full_c = y_valid - y_full_mean
        alpha_full = np.linalg.solve(K_obs + best_delta * np.eye(len(obs_indices)), y_full_c)
        K_te_obs = K_te_tr[:, obs_indices]
        p_te = np.dot(K_te_obs, alpha_full) + y_full_mean
        preds_test[:, idx_out] = p_te
        
    return preds_test

def fit_predict_ridgecv(X_pca_train, X_pca_test, Y_train_masked, M_train_masked, target_indices=[4, 5, 6, 7, 8]):
    """Independent RidgeCV regression per target on 120 training PCs."""
    n_te = len(X_pca_test)
    preds_test = np.zeros((n_te, len(target_indices)), dtype=np.float32)
    alphas = np.logspace(-3, 4, 15)
    
    for idx_out, t_idx in enumerate(target_indices):
        mask_col = M_train_masked[:, t_idx] > 0
        y_valid = Y_train_masked[mask_col, t_idx]
        X_valid = X_pca_train[mask_col]
        
        if len(y_valid) < 5:
            preds_test[:, idx_out] = np.mean(y_valid) if len(y_valid) > 0 else 0.0
            continue
            
        model = RidgeCV(alphas=alphas)
        model.fit(X_valid, y_valid)
        preds_test[:, idx_out] = model.predict(X_pca_test)
        
    return preds_test

def fit_predict_helper_stacking(X_10k_train, X_10k_test, Y_train_masked, M_train_masked, seed,
                                helper_indices=[0, 1, 2, 3], target_indices=[4, 5, 6, 7, 8],
                                exclude_h4=False):
    """
    Two-stage linear stacking:
    1. Inner 5-fold CV to generate leak-free out-of-fold helper trait predictions for training accessions.
    2. Outer helper models to predict outer test accessions.
    Target RidgeCV models trained on [120 PCs, OOF Helper Predictions].
    """
    n_tr = len(X_10k_train)
    n_te = len(X_10k_test)
    alphas = np.logspace(-3, 4, 15)
    
    active_helpers = [h for h in helper_indices if not (exclude_h4 and h == 3)]
    n_act_helpers = len(active_helpers)
    
    # 1. Inner 5-Fold CV to generate OOF helper predictions for training accessions
    kf_inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    H_train_oof = np.zeros((n_tr, n_act_helpers), dtype=np.float32)
    
    for in_tr_idx, in_va_idx in kf_inner.split(X_10k_train):
        # Fit inner preprocessing strictly on inner-train
        imp_in = MarkerMeanImputer()
        X_in_tr_imp = imp_in.fit_transform(X_10k_train[in_tr_idx])
        X_in_va_imp = imp_in.transform(X_10k_train[in_va_idx])
        
        scaler_in = StandardScaler()
        pca_in = PCA(n_components=120, random_state=seed)
        
        X_in_tr_pca = pca_in.fit_transform(scaler_in.fit_transform(X_in_tr_imp))
        X_in_va_pca = pca_in.transform(scaler_in.transform(X_in_va_imp))
        
        for h_pos, h_idx in enumerate(active_helpers):
            h_mask = M_train_masked[in_tr_idx, h_idx] > 0
            y_h = Y_train_masked[in_tr_idx[h_mask], h_idx]
            X_h = X_in_tr_pca[h_mask]
            
            if len(y_h) >= 5:
                mod_h = RidgeCV(alphas=alphas)
                mod_h.fit(X_h, y_h)
                H_train_oof[in_va_idx, h_pos] = mod_h.predict(X_in_va_pca)
            else:
                H_train_oof[in_va_idx, h_pos] = np.mean(y_h) if len(y_h) > 0 else 0.0
                
    # 2. Outer helper models to predict outer test accessions
    imp_out = MarkerMeanImputer()
    X_tr_imp = imp_out.fit_transform(X_10k_train)
    X_te_imp = imp_out.transform(X_10k_test)
    
    scaler_out = StandardScaler()
    pca_out = PCA(n_components=120, random_state=seed)
    X_tr_pca = pca_out.fit_transform(scaler_out.fit_transform(X_tr_imp))
    X_te_pca = pca_out.transform(scaler_out.transform(X_te_imp))
    
    H_test_pred = np.zeros((n_te, n_act_helpers), dtype=np.float32)
    for h_pos, h_idx in enumerate(active_helpers):
        h_mask = M_train_masked[:, h_idx] > 0
        y_h = Y_train_masked[h_mask, h_idx]
        X_h = X_tr_pca[h_mask]
        if len(y_h) >= 5:
            mod_h = RidgeCV(alphas=alphas)
            mod_h.fit(X_h, y_h)
            H_test_pred[:, h_pos] = mod_h.predict(X_te_pca)
        else:
            H_test_pred[:, h_pos] = np.mean(y_h) if len(y_h) > 0 else 0.0
            
    # 3. Stack features: [120 PCs, Helper Predictions]
    X_stacked_tr = np.hstack([X_tr_pca, H_train_oof])
    X_stacked_te = np.hstack([X_te_pca, H_test_pred])
    
    preds_test = np.zeros((n_te, len(target_indices)), dtype=np.float32)
    for idx_out, t_idx in enumerate(target_indices):
        mask_col = M_train_masked[:, t_idx] > 0
        y_valid = Y_train_masked[mask_col, t_idx]
        X_valid = X_stacked_tr[mask_col]
        
        if len(y_valid) < 5:
            preds_test[:, idx_out] = np.mean(y_valid) if len(y_valid) > 0 else 0.0
            continue
            
        model = RidgeCV(alphas=alphas)
        model.fit(X_valid, y_valid)
        preds_test[:, idx_out] = model.predict(X_stacked_te)
        
    return preds_test

def fit_predict_training_mean(Y_train_masked, M_train_masked, n_test, target_indices=[4, 5, 6, 7, 8]):
    """Sanity baseline predicting visible training mean."""
    preds_test = np.zeros((n_test, len(target_indices)), dtype=np.float32)
    for idx_out, t_idx in enumerate(target_indices):
        mask_col = M_train_masked[:, t_idx] > 0
        y_valid = Y_train_masked[mask_col, t_idx]
        m_val = np.mean(y_valid) if len(y_valid) > 0 else 0.0
        preds_test[:, idx_out] = m_val
    return preds_test

# -----------------------------------------------------------------------------
# 6. NEURAL MULTI-TASK LEARNING ARCHITECTURES & LOSS FUNCTIONS
# -----------------------------------------------------------------------------
class SharedBackboneMTL(nn.Module):
    def __init__(self, in_features=120, hidden_dim=160, n_tasks=9, dropout=0.20):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head = nn.Linear(hidden_dim, n_tasks)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks, dtype=torch.float32))

    def forward(self, x):
        return self.head(self.backbone(x))

def calc_task_normalized_losses(pred, target, mask):
    """
    Returns array of per-task MSEs, normalized strictly by each task's observed count:
    MSE_t = sum_i M_it * (y_it - yhat_it)^2 / max(1, sum_i M_it)
    """
    n_tasks = pred.shape[1]
    mses = []
    active = []
    for t in range(n_tasks):
        m_t = mask[:, t]
        cnt = m_t.sum()
        if cnt > 0:
            diff2 = (pred[:, t] - target[:, t]) ** 2
            mse_t = (diff2 * m_t).sum() / cnt
            mses.append(mse_t)
            active.append(t)
        else:
            mses.append(torch.tensor(0.0, device=pred.device))
    return mses, active

def loss_naive_mtl(pred, target, mask):
    """Unstandardized global masked MSE."""
    diff2 = (pred - target) ** 2
    cnt = torch.clamp(mask.sum(), min=1.0)
    return (diff2 * mask).sum() / cnt + (pred * 0.0).sum()

def loss_standardized_mtl(pred, target, mask, w_aux=0.5):
    """
    Task-standardized loss with explicit per-task normalization:
    L = w_aux * Mean_Aux(MSE_h) + Mean_Tgt(MSE_t)
    """
    mses, active = calc_task_normalized_losses(pred, target, mask)
    aux_active = [mses[h] for h in range(4) if h in active]
    tgt_active = [mses[4 + j] for j in range(5) if (4 + j) in active]
    
    loss_aux = torch.stack(aux_active).mean() if len(aux_active) > 0 else (pred * 0.0).sum()
    loss_tgt = torch.stack(tgt_active).mean() if len(tgt_active) > 0 else (pred * 0.0).sum()
    return w_aux * loss_aux + loss_tgt

def loss_uncertainty_mtl(pred, target, mask, log_vars):
    """
    Homoscedastic uncertainty loss (Kendall & Gal) with per-task normalization.
    Missing tasks do NOT contribute their uncertainty regularizers to the loss.
    """
    mses, active = calc_task_normalized_losses(pred, target, mask)
    aux_terms = []
    for h in range(4):
        if h in active:
            s_h = log_vars[h]
            aux_terms.append(0.5 * torch.exp(-s_h) * mses[h] + 0.5 * s_h)
            
    tgt_terms = []
    for j in range(5):
        t_idx = 4 + j
        if t_idx in active:
            s_t = log_vars[t_idx]
            tgt_terms.append(0.5 * torch.exp(-s_t) * mses[t_idx] + 0.5 * s_t)
            
    loss_aux = torch.stack(aux_terms).mean() if len(aux_terms) > 0 else (pred * 0.0).sum()
    loss_tgt = torch.stack(tgt_terms).mean() if len(tgt_terms) > 0 else (pred * 0.0).sum()
    return loss_aux + loss_tgt

def loss_group_dro_mtl(pred, target, mask, group_ids, n_groups, log_vars, alpha=0.40):
    """
    Group-DRO with shared uncertainty base loss.
    Each group target loss is normalized strictly by the number of ACTIVE target tasks in that group.
    """
    mses, active = calc_task_normalized_losses(pred, target, mask)
    aux_terms = [0.5 * torch.exp(-log_vars[h]) * mses[h] + 0.5 * log_vars[h] for h in range(4) if h in active]
    loss_aux = torch.stack(aux_terms).mean() if len(aux_terms) > 0 else (pred * 0.0).sum()
    
    group_losses = []
    for g in range(n_groups):
        g_mask = (group_ids == g)
        if g_mask.sum() == 0:
            continue
        pred_g = pred[g_mask]
        targ_g = target[g_mask]
        mask_g = mask[g_mask]
        
        g_tgt_terms = []
        for j in range(5):
            t_idx = 4 + j
            m_gt = mask_g[:, t_idx]
            cnt_gt = m_gt.sum()
            if cnt_gt > 0:
                diff2_gt = (pred_g[:, t_idx] - targ_g[:, t_idx]) ** 2
                mse_gt = (diff2_gt * m_gt).sum() / cnt_gt
                s_t = log_vars[t_idx]
                g_tgt_terms.append(0.5 * torch.exp(-s_t) * mse_gt + 0.5 * s_t)
                
        if len(g_tgt_terms) > 0:
            group_losses.append(torch.stack(g_tgt_terms).mean())
            
    if len(group_losses) == 0:
        return loss_aux + (pred * 0.0).sum()
        
    g_stack = torch.stack(group_losses)
    loss_dro = (1.0 - alpha) * g_stack.mean() + alpha * g_stack.max()
    return loss_aux + loss_dro

def loss_cvar_mtl(pred, target, mask, log_vars, tail_q=0.30):
    """
    CVaR tail risk objective with shared uncertainty base loss.
    Computes per-sample target uncertainty loss, then averages the worst tail_q fraction.
    """
    mses, active = calc_task_normalized_losses(pred, target, mask)
    aux_terms = [0.5 * torch.exp(-log_vars[h]) * mses[h] + 0.5 * log_vars[h] for h in range(4) if h in active]
    loss_aux = torch.stack(aux_terms).mean() if len(aux_terms) > 0 else (pred * 0.0).sum()
    
    # Target loss per sample
    target_mask = mask[:, 4:]
    has_target = (target_mask.sum(dim=1) > 0)
    if has_target.sum() == 0:
        return loss_aux + (pred * 0.0).sum()
        
    diff2_tgt = (pred[:, 4:] - target[:, 4:]) ** 2
    inv_var_t = 0.5 * torch.exp(-log_vars[4:]).view(1, -1)
    reg_t = 0.5 * log_vars[4:].view(1, -1)
    sample_weighted = (diff2_tgt * inv_var_t + reg_t) * target_mask
    sample_losses = sample_weighted[has_target].sum(dim=1) / torch.clamp(target_mask[has_target].sum(dim=1), min=1.0)
    
    k = max(1, int(np.ceil(len(sample_losses) * tail_q)))
    worst_k_losses, _ = torch.topk(sample_losses, k=k)
    return loss_aux + worst_k_losses.mean()

# -----------------------------------------------------------------------------
# 7. STRICTLY NESTED NEURAL MTL TRAINING ENGINE
# -----------------------------------------------------------------------------
def train_neural_mtl_nested(model_type, X_10k_train, X_10k_test, Y_train_masked, M_train_masked,
                            tr_clusters, te_clusters, seed, device, max_epochs=80, batch_size=64):
    """
    Strictly nested training:
    1. Inner 85/15 validation split inside outer training set.
    2. Inner MarkerMeanImputer, PCA, and inner phenotype scalers fitted strictly on inner-train.
    3. Early stopping on inner-val target loss to identify best epoch E*.
    4. Refit model on FULL outer training set with outer MarkerMeanImputer, PCA, and scalers for E* epochs.
    5. Predict outer test once and unstandardize.
    """
    n_tr = len(X_10k_train)
    n_te = len(X_10k_test)
    
    # 1. Inner validation partition
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_tr)
    n_in_tr = int(np.round(n_tr * 0.85))
    in_tr_idx = perm[:n_in_tr]
    in_va_idx = perm[n_in_tr:]
    
    # Inner preprocessing strictly on inner-train
    imp_in = MarkerMeanImputer()
    X_in_tr_imp = imp_in.fit_transform(X_10k_train[in_tr_idx])
    X_in_va_imp = imp_in.transform(X_10k_train[in_va_idx])
    
    scaler_in = StandardScaler()
    pca_in = PCA(n_components=120, random_state=seed)
    X_in_tr_pca = pca_in.fit_transform(scaler_in.fit_transform(X_in_tr_imp))
    X_in_va_pca = pca_in.transform(scaler_in.transform(X_in_va_imp))
    
    mu_in, sd_in = compute_phenotype_scalers(Y_train_masked[in_tr_idx])
    
    if model_type == 'Naive_MTL':
        Y_in_tr_norm = np.nan_to_num(Y_train_masked[in_tr_idx], nan=0.0)
        Y_in_va_norm = np.nan_to_num(Y_train_masked[in_va_idx], nan=0.0)
    else:
        Y_in_tr_norm = np.nan_to_num((Y_train_masked[in_tr_idx] - mu_in) / sd_in, nan=0.0)
        Y_in_va_norm = np.nan_to_num((Y_train_masked[in_va_idx] - mu_in) / sd_in, nan=0.0)
        
    M_in_tr = M_train_masked[in_tr_idx]
    M_in_va = M_train_masked[in_va_idx]
    G_in_tr = tr_clusters[in_tr_idx]
    G_in_va = tr_clusters[in_va_idx]
    
    # Inner Tensors
    t_X_in_tr = torch.tensor(X_in_tr_pca, dtype=torch.float32)
    t_Y_in_tr = torch.tensor(Y_in_tr_norm, dtype=torch.float32)
    t_M_in_tr = torch.tensor(M_in_tr, dtype=torch.float32)
    t_G_in_tr = torch.tensor(G_in_tr, dtype=torch.long)
    
    t_X_in_va = torch.tensor(X_in_va_pca, dtype=torch.float32).to(device)
    t_Y_in_va = torch.tensor(Y_in_va_norm, dtype=torch.float32).to(device)
    t_M_in_va = torch.tensor(M_in_va, dtype=torch.float32).to(device)
    t_G_in_va = torch.tensor(G_in_va, dtype=torch.long).to(device)
    
    dataset_in = TensorDataset(t_X_in_tr, t_Y_in_tr, t_M_in_tr, t_G_in_tr)
    loader_in = DataLoader(dataset_in, batch_size=batch_size, shuffle=True)
    
    # Hyperparameter search for robust variants
    if model_type == 'Group_DRO_MTL':
        cand_params = [0.20, 0.40, 0.60] # alpha candidates
    elif model_type == 'CVaR_MTL':
        cand_params = [0.20, 0.30, 0.50] # tail_q candidates
    else:
        cand_params = [None]
        
    best_cand = cand_params[0]
    best_overall_val_loss = float('inf')
    best_overall_epoch = 40
    
    for cand in cand_params:
        torch.manual_seed(seed)
        net_tune = SharedBackboneMTL().to(device)
        optimizer = optim.AdamW(net_tune.parameters(), lr=1e-3, weight_decay=1e-3)
        
        best_val = float('inf')
        best_ep = 1
        bad_count = 0
        
        for ep in range(1, max_epochs + 1):
            net_tune.train()
            for bx, by, bm, bg in loader_in:
                bx, by, bm, bg = bx.to(device), by.to(device), bm.to(device), bg.to(device)
                optimizer.zero_grad()
                pred = net_tune(bx)
                
                if model_type == 'Naive_MTL':
                    loss = loss_naive_mtl(pred, by, bm)
                elif model_type == 'Standardized_MTL':
                    loss = loss_standardized_mtl(pred, by, bm)
                elif model_type == 'Uncertainty_Weighted_MTL':
                    loss = loss_uncertainty_mtl(pred, by, bm, net_tune.log_vars)
                elif model_type == 'Group_DRO_MTL':
                    loss = loss_group_dro_mtl(pred, by, bm, bg, n_groups=4, log_vars=net_tune.log_vars, alpha=cand)
                elif model_type == 'CVaR_MTL':
                    loss = loss_cvar_mtl(pred, by, bm, log_vars=net_tune.log_vars, tail_q=cand)
                    
                if loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                
            # Evaluate target validation loss
            net_tune.eval()
            with torch.no_grad():
                val_pred = net_tune(t_X_in_va)
                diff2_val = (val_pred[:, 4:] - t_Y_in_va[:, 4:]) ** 2
                val_tgt_loss = float(((diff2_val * t_M_in_va[:, 4:]).sum() / torch.clamp(t_M_in_va[:, 4:].sum(), min=1.0)).item())
                
            if val_tgt_loss < best_val - 1e-4:
                best_val = val_tgt_loss
                best_ep = ep
                bad_count = 0
            else:
                bad_count += 1
                if bad_count >= 15:
                    break
                    
        if best_val < best_overall_val_loss:
            best_overall_val_loss = best_val
            best_overall_epoch = best_ep
            best_cand = cand
            
    # 2. Refit on FULL outer training set using optimal epoch E* and best_cand
    imp_out = MarkerMeanImputer()
    X_out_tr_imp = imp_out.fit_transform(X_10k_train)
    X_out_te_imp = imp_out.transform(X_10k_test)
    
    scaler_out = StandardScaler()
    pca_out = PCA(n_components=120, random_state=seed)
    X_out_tr_pca = pca_out.fit_transform(scaler_out.fit_transform(X_out_tr_imp))
    X_out_te_pca = pca_out.transform(scaler_out.transform(X_out_te_imp))
    
    mu_out, sd_out = compute_phenotype_scalers(Y_train_masked)
    if model_type == 'Naive_MTL':
        Y_out_tr_norm = np.nan_to_num(Y_train_masked, nan=0.0)
    else:
        Y_out_tr_norm = np.nan_to_num((Y_train_masked - mu_out) / sd_out, nan=0.0)
        
    t_X_out_tr = torch.tensor(X_out_tr_pca, dtype=torch.float32)
    t_Y_out_tr = torch.tensor(Y_out_tr_norm, dtype=torch.float32)
    t_M_out_tr = torch.tensor(M_train_masked, dtype=torch.float32)
    t_G_out_tr = torch.tensor(tr_clusters, dtype=torch.long)
    
    dataset_out = TensorDataset(t_X_out_tr, t_Y_out_tr, t_M_out_tr, t_G_out_tr)
    loader_out = DataLoader(dataset_out, batch_size=batch_size, shuffle=True)
    
    torch.manual_seed(seed + 999)
    final_net = SharedBackboneMTL().to(device)
    optimizer_final = optim.AdamW(final_net.parameters(), lr=1e-3, weight_decay=1e-3)
    
    final_net.train()
    for ep in range(best_overall_epoch):
        for bx, by, bm, bg in loader_out:
            bx, by, bm, bg = bx.to(device), by.to(device), bm.to(device), bg.to(device)
            optimizer_final.zero_grad()
            pred = final_net(bx)
            if model_type == 'Naive_MTL':
                loss = loss_naive_mtl(pred, by, bm)
            elif model_type == 'Standardized_MTL':
                loss = loss_standardized_mtl(pred, by, bm)
            elif model_type == 'Uncertainty_Weighted_MTL':
                loss = loss_uncertainty_mtl(pred, by, bm, final_net.log_vars)
            elif model_type == 'Group_DRO_MTL':
                loss = loss_group_dro_mtl(pred, by, bm, bg, n_groups=4, log_vars=final_net.log_vars, alpha=best_cand)
            elif model_type == 'CVaR_MTL':
                loss = loss_cvar_mtl(pred, by, bm, log_vars=final_net.log_vars, tail_q=best_cand)
            if loss.requires_grad:
                loss.backward()
                optimizer_final.step()
            
    # 3. Predict outer test
    final_net.eval()
    t_X_out_te = torch.tensor(X_out_te_pca, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_raw = final_net(t_X_out_te).cpu().numpy()[:, 4:] # target columns
        
    if model_type == 'Naive_MTL':
        preds_test = preds_raw
    else:
        preds_test = preds_raw * sd_out[4:].reshape(1, -1) + mu_out[4:].reshape(1, -1)
        
    return preds_test, best_overall_epoch, best_cand

# -----------------------------------------------------------------------------
# 8. GENUINELY INDEPENDENT 2^3 FACTORIAL ABLATION ENGINE (CRITIQUE SECTION 3)
# -----------------------------------------------------------------------------
def train_factorial_ablation_model(X_10k_train, X_10k_test, Y_train_masked, M_train_masked,
                                   aux_supervision: bool,    # Factor A: True=Helpers Active, False=Helpers Masked
                                   standardize_pheno: bool,  # Factor B: True=Standardized Phenotypes, False=Raw Native
                                   uncertainty_weight: bool, # Factor C: True=Homoscedastic Loss, False=Equal Task Weights
                                   seed: int, device, max_epochs=60, batch_size=64):
    """
    Executes an orthogonal factorial condition with genuinely independent switches:
    - Factor A (Auxiliary Supervision): Controls whether helper tasks (0..3) are present in training mask M.
    - Factor B (Phenotype Standardization): Controls whether phenotypes are standardized by training (mu, sd).
    - Factor C (Uncertainty Weighting): Controls whether active task losses are combined with equal weights (1/T)
      or with Kendall & Gal learned homoscedastic uncertainty regularizers.
    All other factors (task-count normalization, architecture, optimizer, early stopping) are strictly held fixed.
    """
    n_tr = len(X_10k_train)
    n_te = len(X_10k_test)
    
    # Apply Factor A switch to training mask
    M_tr_cfg = M_train_masked.copy()
    if not aux_supervision:
        M_tr_cfg[:, :4] = 0.0 # helper tasks completely disabled
        
    # 1. Inner validation partition
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_tr)
    n_in_tr = int(np.round(n_tr * 0.85))
    in_tr_idx = perm[:n_in_tr]
    in_va_idx = perm[n_in_tr:]
    
    # Inner preprocessing
    imp_in = MarkerMeanImputer()
    X_in_tr_imp = imp_in.fit_transform(X_10k_train[in_tr_idx])
    X_in_va_imp = imp_in.transform(X_10k_train[in_va_idx])
    
    scaler_in = StandardScaler()
    pca_in = PCA(n_components=120, random_state=seed)
    X_in_tr_pca = pca_in.fit_transform(scaler_in.fit_transform(X_in_tr_imp))
    X_in_va_pca = pca_in.transform(scaler_in.transform(X_in_va_imp))
    
    # Apply Factor B switch to inner targets
    mu_in, sd_in = compute_phenotype_scalers(Y_train_masked[in_tr_idx])
    if standardize_pheno:
        Y_in_tr_target = np.nan_to_num((Y_train_masked[in_tr_idx] - mu_in) / sd_in, nan=0.0)
        Y_in_va_target = np.nan_to_num((Y_train_masked[in_va_idx] - mu_in) / sd_in, nan=0.0)
    else:
        Y_in_tr_target = np.nan_to_num(Y_train_masked[in_tr_idx], nan=0.0)
        Y_in_va_target = np.nan_to_num(Y_train_masked[in_va_idx], nan=0.0)
        
    M_in_tr = M_tr_cfg[in_tr_idx]
    M_in_va = M_tr_cfg[in_va_idx]
    
    t_X_in_tr = torch.tensor(X_in_tr_pca, dtype=torch.float32)
    t_Y_in_tr = torch.tensor(Y_in_tr_target, dtype=torch.float32)
    t_M_in_tr = torch.tensor(M_in_tr, dtype=torch.float32)
    
    t_X_in_va = torch.tensor(X_in_va_pca, dtype=torch.float32).to(device)
    t_Y_in_va = torch.tensor(Y_in_va_target, dtype=torch.float32).to(device)
    t_M_in_va = torch.tensor(M_in_va, dtype=torch.float32).to(device)
    
    dataset_in = TensorDataset(t_X_in_tr, t_Y_in_tr, t_M_in_tr)
    loader_in = DataLoader(dataset_in, batch_size=batch_size, shuffle=True)
    
    torch.manual_seed(seed)
    net_tune = SharedBackboneMTL().to(device)
    optimizer = optim.AdamW(net_tune.parameters(), lr=1e-3, weight_decay=1e-3)
    
    best_val = float('inf')
    best_ep = 1
    bad_count = 0
    
    for ep in range(1, max_epochs + 1):
        net_tune.train()
        for bx, by, bm in loader_in:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            optimizer.zero_grad()
            pred = net_tune(bx)
            
            # Per-task MSE normalized by task observation count
            mses, active_tasks = calc_task_normalized_losses(pred, by, bm)
            
            active_tgt = [t for t in active_tasks if t >= 4]
            active_aux = [h for h in active_tasks if h < 4]
            
            if len(active_tgt) == 0:
                loss_tgt = (pred * 0.0).sum()
            else:
                if uncertainty_weight:
                    tgt_terms = [0.5 * torch.exp(-net_tune.log_vars[t]) * mses[t] + 0.5 * net_tune.log_vars[t] for t in active_tgt]
                else:
                    tgt_terms = [mses[t] for t in active_tgt]
                loss_tgt = torch.stack(tgt_terms).mean()
                
            if (not aux_supervision) or len(active_aux) == 0:
                loss_aux = (pred * 0.0).sum()
            else:
                if uncertainty_weight:
                    aux_terms = [0.5 * torch.exp(-net_tune.log_vars[h]) * mses[h] + 0.5 * net_tune.log_vars[h] for h in active_aux]
                else:
                    aux_terms = [mses[h] for h in active_aux]
                loss_aux = torch.stack(aux_terms).mean()
                
            loss = loss_tgt + loss_aux
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                
        # Evaluate target validation loss
        net_tune.eval()
        with torch.no_grad():
            val_pred = net_tune(t_X_in_va)
            diff2_val = (val_pred[:, 4:] - t_Y_in_va[:, 4:]) ** 2
            val_tgt_loss = float(((diff2_val * t_M_in_va[:, 4:]).sum() / torch.clamp(t_M_in_va[:, 4:].sum(), min=1.0)).item())
            
        if val_tgt_loss < best_val - 1e-4:
            best_val = val_tgt_loss
            best_ep = ep
            bad_count = 0
        else:
            bad_count += 1
            if bad_count >= 15:
                break
                
    # 2. Refit on FULL outer training set using optimal epoch E*
    imp_out = MarkerMeanImputer()
    X_out_tr_imp = imp_out.fit_transform(X_10k_train)
    X_out_te_imp = imp_out.transform(X_10k_test)
    
    scaler_out = StandardScaler()
    pca_out = PCA(n_components=120, random_state=seed)
    X_out_tr_pca = pca_out.fit_transform(scaler_out.fit_transform(X_out_tr_imp))
    X_out_te_pca = pca_out.transform(scaler_out.transform(X_out_te_imp))
    
    mu_out, sd_out = compute_phenotype_scalers(Y_train_masked)
    if standardize_pheno:
        Y_out_tr_target = np.nan_to_num((Y_train_masked - mu_out) / sd_out, nan=0.0)
    else:
        Y_out_tr_target = np.nan_to_num(Y_train_masked, nan=0.0)
        
    t_X_out_tr = torch.tensor(X_out_tr_pca, dtype=torch.float32)
    t_Y_out_tr = torch.tensor(Y_out_tr_target, dtype=torch.float32)
    t_M_out_tr = torch.tensor(M_tr_cfg, dtype=torch.float32)
    
    dataset_out = TensorDataset(t_X_out_tr, t_Y_out_tr, t_M_out_tr)
    loader_out = DataLoader(dataset_out, batch_size=batch_size, shuffle=True)
    
    torch.manual_seed(seed + 999)
    final_net = SharedBackboneMTL().to(device)
    optimizer_final = optim.AdamW(final_net.parameters(), lr=1e-3, weight_decay=1e-3)
    
    final_net.train()
    for ep in range(best_ep):
        for bx, by, bm in loader_out:
            bx, by, bm = bx.to(device), by.to(device), bm.to(device)
            optimizer_final.zero_grad()
            pred = final_net(bx)
            
            mses, active_tasks = calc_task_normalized_losses(pred, by, bm)
            active_tgt = [t for t in active_tasks if t >= 4]
            active_aux = [h for h in active_tasks if h < 4]
            
            if len(active_tgt) == 0:
                loss_tgt = (pred * 0.0).sum()
            else:
                if uncertainty_weight:
                    tgt_terms = [0.5 * torch.exp(-final_net.log_vars[t]) * mses[t] + 0.5 * final_net.log_vars[t] for t in active_tgt]
                else:
                    tgt_terms = [mses[t] for t in active_tgt]
                loss_tgt = torch.stack(tgt_terms).mean()
                
            if (not aux_supervision) or len(active_aux) == 0:
                loss_aux = (pred * 0.0).sum()
            else:
                if uncertainty_weight:
                    aux_terms = [0.5 * torch.exp(-final_net.log_vars[h]) * mses[h] + 0.5 * final_net.log_vars[h] for h in active_aux]
                else:
                    aux_terms = [mses[h] for h in active_aux]
                loss_aux = torch.stack(aux_terms).mean()
                
            loss = loss_tgt + loss_aux
            if loss.requires_grad:
                loss.backward()
                optimizer_final.step()
                
    # 3. Predict outer test
    final_net.eval()
    t_X_out_te = torch.tensor(X_out_te_pca, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_raw = final_net(t_X_out_te).cpu().numpy()[:, 4:] # target columns
        
    if standardize_pheno:
        preds_test = preds_raw * sd_out[4:].reshape(1, -1) + mu_out[4:].reshape(1, -1)
    else:
        preds_test = preds_raw
        
    return preds_test, best_ep

# -----------------------------------------------------------------------------
# 9. STATISTICAL CONTRASTS & EXACT PAIRED PERMUTATION TEST (CRITIQUE SECTION 10 & 11)
# -----------------------------------------------------------------------------
def calculate_primary_repeat_pooled_macro_r2(df_oof_observed, model_name, target_cols):
    """
    Primary Estimand:
    1. Pool predictions across outer folds within each repeat.
    2. Compute R2 for each target trait using only naturally observed held-out labels.
    3. Compute Macro-R2 per repeat: unweighted mean of 5 trait R2s.
    4. Primary estimand is average of the 3 repeat Macro-R2 values.
    """
    repeat_macros = []
    for rep in [1, 2, 3]:
        df_r = df_oof_observed[df_oof_observed['Repeat'] == rep]
        t_r2s = []
        for t_col in target_cols:
            df_rt = df_r[df_r['Trait'] == t_col]
            if len(df_rt) >= 5:
                y_tr = df_rt['Y_True'].values
                y_pr = df_rt[f"Pred_{model_name}"].values
                t_r2s.append(r2_score(y_tr, y_pr))
        repeat_macros.append(np.mean(t_r2s))
    return float(np.mean(repeat_macros)), repeat_macros

def compute_exact_paired_sign_flip_p_val(fold_diffs):
    """
    Exact two-sided paired sign-flip permutation calculation across N fold differences.
    It enumerates all 2^N exact sign configurations (for N=15 folds, exactly 2^15 = 32,768 permutations)
    under the sharp null reflection assumption d_k ~ -d_k.

    IMPORTANT METHODOLOGICAL LIMITATION:
    While this calculation enumerates all exact sign configurations, repeated cross-validation fold
    differences are statistically dependent (sharing 80% of training accessions across folds and
    identical evaluation sets across repeats). Because the repeated-CV fold differences are dependent,
    the resulting fold-based inference is exploratory rather than guaranteed to be a calibrated
    finite-sample population test.
    """
    d = np.array(fold_diffs, dtype=np.float64)
    n = len(d)
    t_obs = float(np.abs(np.mean(d)))
    if t_obs == 0.0:
        return 1.0
    signs = np.array(np.meshgrid(*[[-1, 1]] * n)).T.reshape(-1, n)
    perm_means = np.abs(np.dot(signs, d) / n)
    p_exact = float(np.mean(perm_means >= t_obs - 1e-12))
    return p_exact

def adjust_p_values_holm(p_values):
    """
    Step-down Holm-Bonferroni multiplicity adjustment.
    Guarantees monotonicity, caps values at 1.0, correctly handles ties,
    and returns adjusted p-values in original input order.
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    if p_arr.size == 0:
        return np.array([], dtype=np.float64)
    if p_arr.size == 1:
        return np.clip(p_arr, 0.0, 1.0)
    
    m = len(p_arr)
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    
    adj_sorted = np.zeros(m, dtype=np.float64)
    cum_max = 0.0
    for k in range(m):
        raw_val = sorted_p[k]
        multiplier = m - k
        val = min(1.0, max(0.0, raw_val * multiplier))
        cum_max = max(cum_max, val)
        adj_sorted[k] = cum_max
        
    adj = np.zeros(m, dtype=np.float64)
    adj[order] = adj_sorted
    return adj

def compute_factorial_ablation_effects(df_abl_folds, ablation_configs):
    """
    Computes orthogonal Yates contrast effects across 15 outer folds for 2^3 design.
    Returns DataFrame matching authoritative schema with Holm-adjusted p-values:
    ['Effect', 'Estimate', 'SE', 'CI95_Lower', 'CI95_Upper', 'Raw_p_val', 'Holm_Adjusted_p_val', 'Significant_Holm_0.05']
    """
    import scipy.stats as stats
    factor_labels = [
        ('Main Effect: Auxiliary Supervision (A)', [ -1,  1, -1, -1,  1,  1, -1,  1 ]),
        ('Main Effect: Phenotype Standardization (B)', [ -1, -1,  1, -1,  1, -1,  1,  1 ]),
        ('Main Effect: Uncertainty Weighting (C)',    [ -1, -1, -1,  1, -1,  1,  1,  1 ]),
        ('Two-Way Interaction: Aux x Std (A:B)',      [  1, -1, -1,  1,  1, -1, -1,  1 ]),
        ('Two-Way Interaction: Aux x Uncert (A:C)',   [  1, -1,  1, -1, -1,  1, -1,  1 ]),
        ('Two-Way Interaction: Std x Uncert (B:C)',   [  1,  1, -1, -1, -1, -1,  1,  1 ]),
        ('Three-Way Interaction: Aux x Std x Uncert', [ -1,  1,  1,  1, -1, -1, -1,  1 ])
    ]
    
    cond_order = [cfg['name'] for cfg in ablation_configs]
    df_pivot = df_abl_folds.pivot(index=['Repeat', 'Fold'], columns='Condition', values='Macro_R2')[cond_order]
    t_crit_14 = stats.t.ppf(0.975, df=14)
    
    effect_rows = []
    for eff_name, c_vec in factor_labels:
        c_arr = np.array(c_vec, dtype=np.float64)
        fold_effects = np.dot(df_pivot.values, c_arr) / 4.0
        eff_mean = float(np.mean(fold_effects))
        eff_se = float(np.std(fold_effects, ddof=1) / np.sqrt(15))
        eff_ci_lower = eff_mean - t_crit_14 * eff_se
        eff_ci_upper = eff_mean + t_crit_14 * eff_se
        p_val_eff = compute_exact_paired_sign_flip_p_val(fold_effects)
        
        effect_rows.append({
            'Effect': eff_name,
            'Estimate': eff_mean,
            'SE': eff_se,
            'CI95_Lower': eff_ci_lower,
            'CI95_Upper': eff_ci_upper,
            'Raw_p_val': p_val_eff
        })
        
    df_effects = pd.DataFrame(effect_rows)
    df_effects['Holm_Adjusted_p_val'] = adjust_p_values_holm(df_effects['Raw_p_val'].values)
    df_effects['Significant_Holm_0.05'] = df_effects['Holm_Adjusted_p_val'] < 0.05
    return df_effects

def compute_accession_level_bootstrap_contrasts(df_oof_observed, df_fold_metrics,
                                               contrasts_list, target_cols,
                                               n_boot=10000, seed=42):
    """
    Dual Inference Protocol (Primary Point Estimates & Exploratory Uncertainty):
    1. Primary Estimand: Repeat-Pooled Macro-R2 Difference (Delta_Macro_R2_Primary)
       evaluated conditionally on the naturally observed target outcomes across
       the 1,442 accessions with at least one observed target trait.
    2. Primary Uncertainty (Evaluation Sample Uncertainty):
       Paired accession-level bootstrap (10,000 resamples over the 1,442 unique accessions)
       yielding Bootstrap_SE, 95% percentile CIs, and exploratory uncentered tail p-values.
       *CRITICAL METHODOLOGICAL LIMITATION*:
       These bootstrap intervals condition on the saved out-of-fold prediction vectors
       and capture evaluation test sample variability only; they DO NOT capture model
       refitting or training-partition variance.
    3. Exploratory Fold Dispersion:
       Fold_Diff_Mean and Fold_Diff_Std across the 15 outer folds (3 repeats x 5 folds).
       Exploratory_Fold_Permutation_p_val from exact sign-flip enumeration.
       *NOTE ON ASSUMPTIONS*:
       Repeated-CV folds share 80% training set overlap and identical test sets across
       repeats, violating independent exchangeability. Fold permutation tests are strictly
       exploratory summaries of cross-validation fold dispersion, NOT confirmatory population tests.
    """
    rng = np.random.RandomState(seed)
    unique_accessions = df_oof_observed['Accession_IID'].unique()
    n_acc = len(unique_accessions)
    
    rep_dfs = {rep: df_oof_observed[df_oof_observed['Repeat'] == rep] for rep in [1, 2, 3]}
    
    contrast_results = []
    
    for c_name, m1, m2 in contrasts_list:
        # 1. Exact primary point estimate on repeat-pooled predictions
        macro_1, _ = calculate_primary_repeat_pooled_macro_r2(df_oof_observed, m1, target_cols)
        macro_2, _ = calculate_primary_repeat_pooled_macro_r2(df_oof_observed, m2, target_cols)
        delta_point = float(macro_1 - macro_2)
        
        # 2. Exploratory fold-level differences across 15 outer folds
        f_sub1 = df_fold_metrics[df_fold_metrics['Model'] == m1].sort_values(['Repeat', 'Fold'])
        f_sub2 = df_fold_metrics[df_fold_metrics['Model'] == m2].sort_values(['Repeat', 'Fold'])
        fold_diffs = (f_sub1['Macro_R2'].values - f_sub2['Macro_R2'].values).astype(np.float64)
        p_fold_exact = compute_exact_paired_sign_flip_p_val(fold_diffs)
        
        # 3. Paired accession-level bootstrap for primary estimand inference
        boot_deltas = np.zeros(n_boot, dtype=np.float64)
        for b in range(n_boot):
            sampled_iids = rng.choice(unique_accessions, size=n_acc, replace=True)
            counts = pd.Series(sampled_iids).value_counts()
            
            rep_diffs = []
            for rep in [1, 2, 3]:
                df_r = rep_dfs[rep]
                df_sub = df_r[df_r['Accession_IID'].isin(counts.index)]
                
                t_r2_1 = []
                t_r2_2 = []
                for t_col in target_cols:
                    df_rt = df_sub[df_sub['Trait'] == t_col]
                    if len(df_rt) >= 5:
                        y = df_rt['Y_True'].values
                        w = df_rt['Accession_IID'].map(counts).values
                        p1 = df_rt[f"Pred_{m1}"].values
                        p2 = df_rt[f"Pred_{m2}"].values
                        
                        mean_y = np.average(y, weights=w)
                        ss_tot = np.sum(w * (y - mean_y) ** 2)
                        if ss_tot <= 0.0:
                            raise ValueError(f"Degenerate bootstrap sample for trait {t_col}: zero total variance.")
                        ss_res1 = np.sum(w * (y - p1) ** 2)
                        ss_res2 = np.sum(w * (y - p2) ** 2)
                        
                        r2_1 = 1.0 - (ss_res1 / ss_tot)
                        r2_2 = 1.0 - (ss_res2 / ss_tot)
                        t_r2_1.append(r2_1)
                        t_r2_2.append(r2_2)
                rep_diffs.append(np.mean(t_r2_1) - np.mean(t_r2_2))
            boot_deltas[b] = np.mean(rep_diffs)
            
        boot_mean = float(np.mean(boot_deltas))
        boot_se = float(np.std(boot_deltas, ddof=1))
        ci_lower = float(np.percentile(boot_deltas, 2.5))
        ci_upper = float(np.percentile(boot_deltas, 97.5))
        
        # Empirical two-sided bootstrap p-value testing H0: Delta_primary = 0 (Exploratory tail proxy)
        p_boot_raw = float(2.0 * min(np.mean(boot_deltas <= 0.0), np.mean(boot_deltas >= 0.0)))
        p_boot = max(1.0 / n_boot, p_boot_raw) if p_boot_raw == 0.0 else p_boot_raw
        
        contrast_results.append({
            'Contrast': c_name,
            'Model_A': m1,
            'Model_B': m2,
            # Primary repeat-pooled inference
            'Delta_Macro_R2_Primary': delta_point,
            'Bootstrap_Mean': boot_mean,
            'Bootstrap_SE': boot_se,
            'CI95_Lower': ci_lower,
            'CI95_Upper': ci_upper,
            'Primary_Bootstrap_p_val': p_boot,
            # Exploratory fold-level CV metrics
            'Fold_Diff_Mean': float(np.mean(fold_diffs)),
            'Fold_Diff_Std': float(np.std(fold_diffs, ddof=1)),
            'Exploratory_Fold_Permutation_p_val': p_fold_exact
        })
        
    df_contrasts = pd.DataFrame(contrast_results)
    df_contrasts['Primary_Holm_p_val'] = adjust_p_values_holm(df_contrasts['Primary_Bootstrap_p_val'].values)
    df_contrasts['Primary_Significant_0.05'] = df_contrasts['Primary_Holm_p_val'] < 0.05
    return df_contrasts

def compute_secondary_sensitivity_accession_contrasts(df_sec_acc, n_boot=10000, seed=42):
    """
    Computes repeat-pooled Delta_Macro_R2 point estimates and paired accession-level
    bootstrap 95% confidence intervals directly from stored held-out prediction records
    (secondary_sensitivity_accession_predictions.csv).
    Resamples across the 1,442 unique accessions with observed target outcomes.
    Explicitly guards against degenerate bootstrap samples (zero total sum of squares).
    """
    rng = np.random.RandomState(seed)
    unique_acc = df_sec_acc['Accession_IID'].unique()
    n_acc = len(unique_acc)
    rep_dfs = {r: df_sec_acc[df_sec_acc['Repeat'] == r] for r in [1, 2, 3]}
    traits = df_sec_acc['Trait'].unique()
    
    comparisons = [
        ('Target-Only MTL vs All-Nine-Masked MTL', 'Pred_Target_Only_MTL', 'Pred_All_Nine_Masked_MTL'),
        ('Stacking 4 Helpers vs Stacking 3 Helpers (No H4)', 'Pred_Stacking_4_Helpers', 'Pred_Stacking_3_Helpers_NoH4')
    ]
    
    results = []
    for comp_name, col_a, col_b in comparisons:
        # 1. Exact repeat-pooled point estimate
        rep_deltas = []
        for r in [1, 2, 3]:
            df_r = rep_dfs[r]
            t_r2_a, t_r2_b = [], []
            for t in traits:
                df_rt = df_r[df_r['Trait'] == t]
                y = df_rt['Y_True'].values
                pa = df_rt[col_a].values
                pb = df_rt[col_b].values
                t_r2_a.append(r2_score(y, pa))
                t_r2_b.append(r2_score(y, pb))
            rep_deltas.append(np.mean(t_r2_a) - np.mean(t_r2_b))
        delta_primary = float(np.mean(rep_deltas))
        
        # 2. Paired accession-level bootstrap (conditioning on saved predictions)
        boot_d = np.zeros(n_boot, dtype=np.float64)
        for b in range(n_boot):
            sampled_iids = rng.choice(unique_acc, size=n_acc, replace=True)
            counts = pd.Series(sampled_iids).value_counts()
            
            b_rep_d = []
            for r in [1, 2, 3]:
                df_r = rep_dfs[r]
                df_sub = df_r[df_r['Accession_IID'].isin(counts.index)]
                t_r2_a, t_r2_b = [], []
                for t in traits:
                    df_rt = df_sub[df_sub['Trait'] == t]
                    if len(df_rt) >= 5:
                        y = df_rt['Y_True'].values
                        w = df_rt['Accession_IID'].map(counts).values
                        pa = df_rt[col_a].values
                        pb = df_rt[col_b].values
                        
                        y_bar = np.average(y, weights=w)
                        ss_tot = np.sum(w * (y - y_bar)**2)
                        if ss_tot <= 0.0:
                            raise ValueError(f"Degenerate bootstrap sample for trait {t}: zero total variance.")
                        ss_res_a = np.sum(w * (y - pa)**2)
                        ss_res_b = np.sum(w * (y - pb)**2)
                        t_r2_a.append(1.0 - (ss_res_a / ss_tot))
                        t_r2_b.append(1.0 - (ss_res_b / ss_tot))
                b_rep_d.append(np.mean(t_r2_a) - np.mean(t_r2_b))
            boot_d[b] = np.mean(b_rep_d)
            
        boot_mean = float(np.mean(boot_d))
        boot_se = float(np.std(boot_d, ddof=1))
        ci_lower = float(np.percentile(boot_d, 2.5))
        ci_upper = float(np.percentile(boot_d, 97.5))
        p_boot_raw = float(2.0 * min(np.mean(boot_d <= 0.0), np.mean(boot_d >= 0.0)))
        p_boot = max(1.0 / n_boot, p_boot_raw) if p_boot_raw == 0.0 else p_boot_raw
        
        results.append({
            'Comparison': comp_name,
            'Delta_Macro_R2_Primary': delta_primary,
            'Bootstrap_Mean': boot_mean,
            'Bootstrap_SE': boot_se,
            'CI95_Lower': ci_lower,
            'CI95_Upper': ci_upper,
            'Exploratory_Tail_p_val': p_boot
        })
        
    return pd.DataFrame(results)
