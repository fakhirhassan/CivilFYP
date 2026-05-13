"""Build slump_modeling_full.ipynb — Phase 0-6 on the slump dataset."""
import json, uuid

NB = '/Users/fakhirhassan/Desktop/CivilFYP/slump_modeling_full.ipynb'

def md(src):
    return {'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},'source':src.splitlines(keepends=True)}
def code(src):
    return {'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{},'execution_count':None,'outputs':[],'source':src.splitlines(keepends=True)}

cells = []

cells.append(md("""# Slump — full pipeline (Phase 0 → Phase 6)

Same methodology as `cs_modeling_full.ipynb`, applied to slump.

**Dataset:** `slump data complete s.xlsx` — 84 rows, 15 features, target = `Slump` (mm).
**Target range:** 32–650 mm, mean 194.6, std 130.0 (much wider relative spread than CS).

Pipeline:
- Phase 0: load + audit + group IDs
- Phase 1: honest baseline (random KFold + GroupKFold) on the 7 original models
- Phase 2: feature engineering + VIF check
- Phase 3: Optuna tuning (100 trials × 6 boosting models × 2 feature sets)
- Phase 4: small-data models (Ridge/Lasso/SVR/KNN/ExtraTrees/HGB/GP/PLS — 11 models × 50 trials)
- Phase 5: stacking ensemble
- Phase 6: relaxed grouping + SMOGN

The first run takes ~25–35 min on this machine. Re-runs of single cells are much faster.
"""))

cells.append(md("""## Phase 0 — Load and audit"""))

cells.append(code("""import sys, time, importlib, json, math
sys.path.insert(0, '/Users/fakhirhassan/Desktop/CivilFYP')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

import civil_utils as cu; importlib.reload(cu)
pd.set_option('display.float_format', lambda x: f'{x:,.4f}')
sns.set_theme(style='whitegrid', context='notebook')

df = cu.slump_dataset()
TARGET = cu.SLUMP_TARGET
print(f'Shape: {df.shape}')
print('Target stats:')
print(df[TARGET].describe())
df.head()
"""))

cells.append(code("""# Group IDs and EDA
X_raw_df = df.drop(columns=[TARGET])
y = df[TARGET].values
groups = cu.assign_group_ids(X_raw_df)
n_groups = len(np.unique(groups))
gs = pd.Series(groups).value_counts()
print(f'Unique mix-design groups: {n_groups}')
print(f'Group sizes: median={gs.median():.0f}, max={gs.max()}, mean={gs.mean():.2f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(y, bins=20, color='steelblue', edgecolor='white')
axes[0].set_title(f'Slump distribution (n={len(y)})')
axes[0].set_xlabel('Slump (mm)')
axes[1].hist(gs.values, bins=range(1, gs.max()+2), color='coral', edgecolor='white')
axes[1].set_title(f'Group sizes ({n_groups} unique groups)')
axes[1].set_xlabel('rows per group')
plt.tight_layout(); plt.show()

# Feature correlations with target
corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#d62728' if v < 0 else '#2ca02c' for v in corr.values]
ax.barh(corr.index, corr.values, color=colors)
ax.set_title('Feature correlation with Slump')
ax.set_xlabel('Pearson correlation'); ax.axvline(0, color='k', lw=0.8)
plt.tight_layout(); plt.show()
"""))

cells.append(md("""## Phase 1 — Honest baseline"""))

cells.append(code("""importlib.reload(cu)
zoo = cu.default_model_zoo()

random_rows, group_rows = [], []
for name, factory in zoo.items():
    print(f'  evaluating {name} ...', flush=True)
    fs_rand = cu.cv_evaluate(factory, X_raw_df, y, groups=None,
                             n_splits=5, n_repeats=5, random_state=0)
    random_rows.append(fs_rand.summary_row(name))
    fs_grp = cu.cv_evaluate(factory, X_raw_df, y, groups=groups,
                            n_splits=5, n_repeats=1, random_state=0)
    group_rows.append(fs_grp.summary_row(name))

cv_random = pd.DataFrame(random_rows)
cv_group = pd.DataFrame(group_rows)
show = ['model','train_r2_mean','val_r2_mean','test_r2_mean','test_r2_std','overfit_gap']
print('\\n=== Random KFold ===')
print(cv_random[show].sort_values('test_r2_mean', ascending=False).round(3).to_string(index=False))
print('\\n=== GroupKFold ===')
print(cv_group[show].sort_values('test_r2_mean', ascending=False).round(3).to_string(index=False))
"""))

cells.append(md("""## Phase 2 — Feature engineering + VIF"""))

cells.append(code("""importlib.reload(cu)
feature_sets = {}
for mode in ('raw', 'engineered', 'both'):
    fdf = cu.engineer_features(df, mode=mode, target=TARGET)
    feature_sets[mode] = fdf.drop(columns=[TARGET])
    print(f'{mode:>11s}: {feature_sets[mode].shape[1]} features')

vif_summary = {}
for mode in ('raw','engineered','both'):
    v = cu.vif_table(feature_sets[mode])
    vif_summary[mode] = v
    n_high = (v['VIF'] > 10).sum()
    median = v['VIF'].replace(np.inf, np.nan).median()
    print(f'  {mode}: median VIF={median:.2f}  high(>10)={n_high}')
"""))

cells.append(code("""# Phase 2 evaluation
phase2_rows = []
for mode, Xm in feature_sets.items():
    for name, factory in zoo.items():
        fs_rand = cu.cv_evaluate(factory, Xm, y, groups=None,
                                 n_splits=5, n_repeats=5, random_state=0)
        fs_grp  = cu.cv_evaluate(factory, Xm, y, groups=groups,
                                 n_splits=5, n_repeats=1, random_state=0)
        rr = fs_rand.summary_row(name); rr['feature_set']=mode; rr['cv']='random_5x5'
        gr = fs_grp.summary_row(name);  gr['feature_set']=mode; gr['cv']='GroupKFold'
        phase2_rows.append(rr); phase2_rows.append(gr)
phase2 = pd.DataFrame(phase2_rows)

g = phase2[phase2.cv=='GroupKFold']
pivot = g.pivot(index='model', columns='feature_set', values='test_r2_mean')[['raw','engineered','both']]
gap_pivot = g.pivot(index='model', columns='feature_set', values='overfit_gap')[['raw','engineered','both']]
print('GroupKFold test R^2 by feature set:'); print(pivot.round(3).to_string())
print('\\nGroupKFold gap by feature set:'); print(gap_pivot.round(3).to_string())
"""))

cells.append(md("""## Phase 3 — Optuna tuning (Bayesian, GroupKFold val R² objective)"""))

cells.append(code("""# Permutation importance to define engineered_9
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold as _GKF

X_eng12 = feature_sets['engineered']
best_eng = g[g.feature_set=='engineered'].sort_values('test_r2_mean', ascending=False)['model'].iloc[0]
print(f'Best on engineered+GroupKFold: {best_eng}')
imp_acc = np.zeros(X_eng12.shape[1])
for trainval_idx, test_idx in _GKF(n_splits=5).split(X_eng12, y, groups):
    model = zoo[best_eng]()
    model.fit(X_eng12.iloc[trainval_idx].values, y[trainval_idx])
    pi = permutation_importance(model, X_eng12.iloc[test_idx].values, y[test_idx],
                                n_repeats=10, random_state=0, scoring='r2')
    imp_acc += pi.importances_mean
imp_acc /= 5
imp_df = pd.DataFrame({'feature': X_eng12.columns, 'importance': imp_acc}).sort_values('importance', ascending=False)
print('\\nPermutation importance:'); print(imp_df.to_string(index=False))

neg_features = imp_df.tail(3)['feature'].tolist()
X_eng9 = X_eng12.drop(columns=[c for c in neg_features if c in X_eng12.columns])
candidate_sets = {'engineered_12': X_eng12, 'engineered_9': X_eng9}
print(f'\\nDropping {neg_features} -> engineered_9 ({X_eng9.shape[1]} features)')
"""))

cells.append(code("""N_TRIALS = 100
tuned_rows = []
best_params_store = {}

for fset_name, X_fs in candidate_sets.items():
    print(f'\\n========== feature set: {fset_name} ==========')
    for model_name in ['Random Forest','Gradient Boosting','AdaBoost','CatBoost','LightGBM','XGBoost']:
        t0 = time.time()
        best_params, best_val, _ = cu.tune_model(
            model_name, X_fs, y, groups,
            n_trials=N_TRIALS, n_splits=5, random_state=0)
        elapsed = time.time() - t0
        best_params_store[(fset_name, model_name)] = best_params

        factory = cu.make_tuned_factory(model_name, best_params)
        fs_rand = cu.cv_evaluate(factory, X_fs, y, groups=None,
                                 n_splits=5, n_repeats=5, random_state=0)
        fs_grp = cu.cv_evaluate(factory, X_fs, y, groups=groups,
                                n_splits=5, n_repeats=1, random_state=0)
        rr = fs_rand.summary_row(model_name); rr['feature_set']=fset_name; rr['cv']='random_5x5'
        gr = fs_grp.summary_row(model_name);  gr['feature_set']=fset_name; gr['cv']='GroupKFold'
        rr['best_optuna_val']=best_val; gr['best_optuna_val']=best_val
        rr['n_trials']=N_TRIALS; gr['n_trials']=N_TRIALS
        rr['tune_seconds']=elapsed; gr['tune_seconds']=elapsed
        tuned_rows.append(rr); tuned_rows.append(gr)
        print(f'  {model_name:<22s} group test={gr["test_r2_mean"]:+.3f} gap={gr["overfit_gap"]:.3f} ({elapsed:.0f}s)')

tuned = pd.DataFrame(tuned_rows)
"""))

cells.append(md("""## Phase 4 — Small-data-friendly models"""))

cells.append(code("""N_TRIALS_SMALL = 50
p4_rows = []
p4_best_params = {}

for fset_name, X_fs in candidate_sets.items():
    print(f'\\n========== feature set: {fset_name} ==========')
    for model_name in cu.SMALL_DATA_SPECS.keys():
        t0 = time.time()
        try:
            best_params, best_val, _ = cu.tune_model(
                model_name, X_fs, y, groups,
                n_trials=N_TRIALS_SMALL, n_splits=5, random_state=0)
        except Exception as e:
            print(f'  {model_name:<22s} TUNE-FAILED: {e}')
            continue
        elapsed = time.time() - t0
        p4_best_params[(fset_name, model_name)] = best_params
        factory = cu.make_tuned_factory(model_name, best_params)
        fs_rand = cu.cv_evaluate(factory, X_fs, y, groups=None,
                                 n_splits=5, n_repeats=5, random_state=0)
        fs_grp = cu.cv_evaluate(factory, X_fs, y, groups=groups,
                                n_splits=5, n_repeats=1, random_state=0)
        rr = fs_rand.summary_row(model_name); rr['feature_set']=fset_name; rr['cv']='random_5x5'
        gr = fs_grp.summary_row(model_name);  gr['feature_set']=fset_name; gr['cv']='GroupKFold'
        rr['best_optuna_val']=best_val; gr['best_optuna_val']=best_val
        p4_rows.append(rr); p4_rows.append(gr)
        print(f'  {model_name:<22s} group test={gr["test_r2_mean"]:+.3f} gap={gr["overfit_gap"]:.3f} ({elapsed:.0f}s)')

phase4 = pd.DataFrame(p4_rows)
"""))

cells.append(code("""# Combined leaderboard
all_tuned = pd.concat([tuned, phase4], ignore_index=True)
g = all_tuned[all_tuned.cv=='GroupKFold'].copy()
best_per = (g.sort_values('test_r2_mean', ascending=False)
              .groupby('model', as_index=False).head(1)
              .sort_values('test_r2_mean', ascending=False))
leaderboard = best_per[['model','feature_set','test_r2_mean','test_r2_std',
                        'val_r2_mean','train_r2_mean','overfit_gap']].reset_index(drop=True)
print('=== Leaderboard (GroupKFold, both phases combined) ===')
print(leaderboard.round(3).to_string(index=False))
"""))

cells.append(md("""## Phase 5 — Stacking ensemble"""))

cells.append(code("""# Top 4 diverse base models — pick from leaderboard
diverse_picks = []
seen_families = set()
for _, row in leaderboard.iterrows():
    m = row['model']
    fam = ('linear' if m in ('Ridge','Lasso','ElasticNet','Ridge+Poly','BayesianRidge','Linear Regression','PLS')
           else 'kernel' if m in ('SVR_RBF','GaussianProcess')
           else 'tree_ens' if m in ('Random Forest','ExtraTrees','AdaBoost')
           else 'boost' if m in ('XGBoost','LightGBM','CatBoost','Gradient Boosting','HistGradientBoosting')
           else 'instance' if m=='KNN' else 'other')
    if fam in seen_families and m != 'SVR_RBF': continue  # skip dups
    if m in ('Linear Regression','Ridge','Lasso','ElasticNet','BayesianRidge','PLS','Ridge+Poly'):
        continue  # exclude unstable linears
    diverse_picks.append(m); seen_families.add(fam)
    if len(diverse_picks) >= 4: break

print('Stacking bases:', diverse_picks)
stack_config = {}
for nm in diverse_picks:
    found = None
    for fset in ('engineered_12','engineered_9'):
        if (fset, nm) in best_params_store: found = (fset, best_params_store[(fset,nm)]); break
        if (fset, nm) in p4_best_params:    found = (fset, p4_best_params[(fset,nm)]); break
    stack_config[nm] = found
    print(f'  {nm:<22s} from {found[0]}')

from collections import Counter
chosen_fset = Counter(c[0] for c in stack_config.values()).most_common(1)[0][0]
X_stack = candidate_sets[chosen_fset]
print(f'Stack feature set: {chosen_fset}')

base_factories = {nm: cu.make_tuned_factory(nm, params) for nm,(_,params) in stack_config.items()}
from sklearn.linear_model import Ridge as _Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
def meta_factory():
    return Pipeline([('sc', StandardScaler()),('m', _Ridge(alpha=1.0, random_state=42))])

fs_rand_stack = cu.stacking_cv_evaluate(base_factories, meta_factory, X_stack, y, groups=None,
                                        n_splits=5, n_repeats=5, random_state=0, inner_splits=5)
fs_grp_stack = cu.stacking_cv_evaluate(base_factories, meta_factory, X_stack, y, groups=groups,
                                       n_splits=5, n_repeats=1, random_state=0, inner_splits=5)
stack_label = 'Stacked (' + '+'.join(diverse_picks) + ')'
stack_rand = fs_rand_stack.summary_row(stack_label); stack_rand['feature_set']=chosen_fset; stack_rand['cv']='random_5x5'
stack_grp = fs_grp_stack.summary_row(stack_label); stack_grp['feature_set']=chosen_fset; stack_grp['cv']='GroupKFold'
print(f'\\nStack random: test R^2 = {stack_rand["test_r2_mean"]:+.3f} +/- {stack_rand["test_r2_std"]:.3f} gap={stack_rand["overfit_gap"]:.3f}')
print(f'Stack group:  test R^2 = {stack_grp["test_r2_mean"]:+.3f} +/- {stack_grp["test_r2_std"]:.3f} gap={stack_grp["overfit_gap"]:.3f}')
"""))

cells.append(md("""## Phase 6 — Relaxed grouping + SMOGN"""))

cells.append(code("""groups_relaxed = cu.assign_group_ids_relaxed(X_raw_df)
print(f'Relaxed groups: {len(np.unique(groups_relaxed))} (strict was {len(np.unique(groups))})')

# SMOGN on the diverse bases
smogn_models = diverse_picks
fset_pick = chosen_fset
p6_smogn_rows = []
for nm in smogn_models:
    params = best_params_store.get((fset_pick, nm)) or p4_best_params.get((fset_pick, nm))
    if params is None: continue
    factory = cu.make_tuned_factory(nm, params)
    X_fs = candidate_sets[fset_pick]
    print(f'\\n=== SMOGN: {nm} ===')
    fs_rand = cu.cv_evaluate_with_smogn(factory, X_fs, y, groups=None,
                                        n_splits=5, n_repeats=3, random_state=0)
    fs_grp = cu.cv_evaluate_with_smogn(factory, X_fs, y, groups=groups,
                                       n_splits=5, n_repeats=1, random_state=0)
    rr = fs_rand.summary_row(nm); rr['feature_set']=fset_pick; rr['cv']='SMOGN_random'
    gr = fs_grp.summary_row(nm); gr['feature_set']=fset_pick; gr['cv']='SMOGN_GroupKFold'
    p6_smogn_rows.extend([rr, gr])
    print(f'  random: test R^2={rr["test_r2_mean"]:+.3f} gap={rr["overfit_gap"]:.3f}')
    print(f'  group:  test R^2={gr["test_r2_mean"]:+.3f} gap={gr["overfit_gap"]:.3f}')

p6_smogn = pd.DataFrame(p6_smogn_rows)
"""))

cells.append(md("""## Final dual-regime leaderboard"""))

cells.append(code("""all_with_stack = pd.concat([all_tuned, pd.DataFrame([stack_rand, stack_grp])], ignore_index=True)

def _best(df, cv):
    sub = df[df.cv==cv]
    return (sub.sort_values('test_r2_mean', ascending=False)
               .groupby('model', as_index=False).head(1))

slim_cols = ['model','feature_set','test_r2_mean','test_r2_std','overfit_gap']
parts = []
for cv_label, src in [('random_KF', _best(all_with_stack,'random_5x5')),
                      ('group_strict', _best(all_with_stack,'GroupKFold')),
                      ('SMOGN_random', _best(p6_smogn,'SMOGN_random')),
                      ('SMOGN_group', _best(p6_smogn,'SMOGN_GroupKFold'))]:
    parts.append(src[slim_cols].assign(regime=cv_label))
dual = pd.concat(parts, ignore_index=True)

dual_test = dual.pivot_table(index='model', columns='regime', values='test_r2_mean', aggfunc='max')
dual_gap = dual.pivot_table(index='model', columns='regime', values='overfit_gap', aggfunc='min')
order_cols = [c for c in ['random_KF','SMOGN_random','group_strict','SMOGN_group'] if c in dual_test.columns]
print('=== Test R^2 by model and regime ===')
print(dual_test[order_cols].round(3).to_string())
print('\\n=== Gap by model and regime ===')
print(dual_gap[order_cols].round(3).to_string())
"""))

cells.append(md("""## Save the phase-by-phase log"""))

cells.append(code("""out_path = '/Users/fakhirhassan/Desktop/CivilFYP/slump_full_phase_results.xlsx'

def _params_to_df(store):
    rows = []
    for (fset, model), p in store.items():
        row = {'feature_set':fset, 'model':model}
        row.update({k: round(v,6) if isinstance(v, float) else v for k,v in p.items()})
        rows.append(row)
    return pd.DataFrame(rows)

with pd.ExcelWriter(out_path, engine='openpyxl') as w:
    cv_random.to_excel(w, sheet_name='Phase1_random', index=False)
    cv_group.to_excel(w, sheet_name='Phase1_group', index=False)
    phase2.to_excel(w, sheet_name='Phase2_full', index=False)
    pivot.to_excel(w, sheet_name='Phase2_GroupKFold_R2')
    gap_pivot.to_excel(w, sheet_name='Phase2_GroupKFold_gap')
    for mode, vdf in vif_summary.items():
        vdf.to_excel(w, sheet_name=f'Phase2_VIF_{mode}', index=False)
    imp_df.to_excel(w, sheet_name='Phase2_perm_importance', index=False)
    tuned.to_excel(w, sheet_name='Phase3_tuned', index=False)
    _params_to_df(best_params_store).to_excel(w, sheet_name='Phase3_best_params', index=False)
    phase4.to_excel(w, sheet_name='Phase4_full', index=False)
    leaderboard.to_excel(w, sheet_name='Phase4_leaderboard', index=False)
    _params_to_df(p4_best_params).to_excel(w, sheet_name='Phase4_best_params', index=False)
    pd.DataFrame([stack_rand, stack_grp]).to_excel(w, sheet_name='Phase5_stack', index=False)
    p6_smogn.to_excel(w, sheet_name='Phase6_SMOGN', index=False)
    dual_test.to_excel(w, sheet_name='Final_test_R2_by_regime')
    dual_gap.to_excel(w, sheet_name='Final_gap_by_regime')

print(f'Saved -> {out_path}')
print(f'\\nTop 5 by GroupKFold test R^2:')
print(leaderboard.head(5).round(3).to_string(index=False))
"""))

nb = {'cells': cells, 'metadata': {
    'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
    'language_info': {'name':'python','version':'3.11'},
}, 'nbformat': 4, 'nbformat_minor': 5}

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)
print(f'Wrote {NB} with {len(cells)} cells')
