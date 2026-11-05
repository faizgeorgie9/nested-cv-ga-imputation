"""
Eksperimen Sensitivity Analysis Hyperparameter Genetic Algorithm (GA-KNN)
Berdasarkan Protokol P1.9 (Nested Cross-Validation, Zero-Leakage, Inductive Deployment).

Tujuan:
1. Menguji sensitivitas parameter GA: pop_size, generations, crossover_rate, mutation_rate.
2. Membuktikan apakah performa konfigurasi baseline saat ini (pop=15, gen=15, cx=0.8, mut=0.2)
   optimal, stabil, atau hanya 'kebetulan'.
3. Menganalisis trade-off antara discriminative performance (ROC-AUC) dan computational cost (runtime).

TIDAK MERUBAH file p1_9.ipynb maupun hasil eksperimen sebelumnya.
Output disimpan terpisah ke: d:/skripsi/topic1/ga_sensitivity_results.csv
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.pipeline import Pipeline

RNG_SEED = 42
np.random.seed(RNG_SEED)

print("="*80)
print("SENSITIVITY ANALYSIS HYPERPARAMETER GA (GA-KNN IMPUTER)")
print("="*80)

# 1. Load Dataset BankChurners
data_path = "data/BankChurners.csv"
if not os.path.exists(data_path):
    # Cek alternatif path jika dijalankan dari root
    data_path = os.path.join(os.path.dirname(__file__), "data", "BankChurners.csv")

df = pd.read_csv(data_path)
df = df.drop(columns=[
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"
])
df["Attrition_Flag"] = df["Attrition_Flag"].map({"Existing Customer": 0, "Attrited Customer": 1})

categorical_cols = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
continuous_cols = [col for col in df.columns if col not in categorical_cols + ["Attrition_Flag"]]

X_all = df.drop(columns=["Attrition_Flag"])
y_all = df["Attrition_Flag"]
all_features = X_all.columns.tolist()

print(f"Dataset Loaded: {X_all.shape}, Target Mean: {y_all.mean():.4f}")

# 2. Fungsi Missingness MCAR Terkendali (Identik dengan P1.9)
def introduce_mcar_missing(data, missing_rate=0.10, columns=None, random_state=42):
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1).")
    columns = list(data.columns if columns is None else columns)
    rng = np.random.RandomState(random_state)
    n_rows = len(data)
    n_cols = len(columns)
    mask = np.zeros((n_rows, n_cols), dtype=bool)

    max_feasible_rate = 1.0 - (1.0 / n_cols)
    if missing_rate > max_feasible_rate:
        mask[:, :] = True
        observed_cols = np.resize(rng.permutation(n_cols), n_rows)
        mask[np.arange(n_rows), observed_cols] = False
    else:
        n_missing = min(int(n_rows * missing_rate), max(0, n_rows - 1))
        for col_pos in range(n_cols):
            selected = rng.choice(n_rows, size=n_missing, replace=False)
            mask[selected, col_pos] = True

        all_missing_rows = np.flatnonzero(mask.all(axis=1))
        for row_pos in all_missing_rows:
            col_pos = int(rng.choice(np.flatnonzero(mask[row_pos])))
            mask[row_pos, col_pos] = False

    for col_pos in range(n_cols):
        if mask[:, col_pos].all():
            candidates = np.flatnonzero(mask.sum(axis=1) >= 2)
            if len(candidates):
                mask[int(rng.choice(candidates)), col_pos] = False
            else:
                mask[int(rng.randint(0, n_rows)), col_pos] = False

    data_missing = data.copy()
    data_missing_values = data_missing[columns].values.astype(float)
    data_missing_values[mask] = np.nan
    data_missing[columns] = data_missing_values

    missing_mask_df = pd.DataFrame(mask, columns=columns, index=data.index)
    original_values = {col: data.loc[missing_mask_df[col], col].copy() for col in columns}
    return data_missing, missing_mask_df, original_values

def calculate_imputation_metrics(original_values, missing_mask, imputed_data, cont_cols, cat_cols):
    rmse_scores = []
    pfc_scores = []
    for col in original_values.keys():
        true_val = original_values[col]
        pred_val = imputed_data.loc[missing_mask[col], col]
        if len(true_val) == 0:
            continue
        if col in cont_cols:
            rmse = np.sqrt(mean_squared_error(true_val, pred_val))
            rmse_scores.append(rmse)
        elif col in cat_cols:
            pred_round = pred_val.round().astype(int)
            pfc = np.mean(true_val != pred_round)
            pfc_scores.append(pfc)
            
    mean_rmse = float(np.mean(rmse_scores)) if len(rmse_scores) > 0 else 0.0
    mean_pfc = float(np.mean(pfc_scores)) if len(pfc_scores) > 0 else 0.0
    return mean_rmse, mean_pfc

# 3. Kelas GAKNNImputer dengan Logging Konvergensi & Early Stopping Detail
class GAKNNSensitivityImputer:
    def __init__(self, pop_size=15, generations=15, mutation_rate=0.2, crossover_rate=0.8,
                 early_stopping_patience=3, fitness_subset_size=3000, random_state=42):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.early_stopping_patience = early_stopping_patience
        self.fitness_subset_size = fitness_subset_size
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.inductive_imputer = None
        self.col_limits = {}
        self.actual_generations_run = 0
        self.best_fitness_val = -1.0

    def fit_transform(self, X_missing, y_train, baseline_imputations=None, cat_cols=None):
        missing_rows, missing_cols = np.where(X_missing.isna())
        missing_coords = list(zip(missing_rows, missing_cols))
        cat_cols = cat_cols or []
        for col in cat_cols:
            obs = X_missing[col].dropna()
            self.col_limits[col] = (obs.min(), obs.max())

        if len(missing_coords) == 0:
            return X_missing.copy(), []

        col_stats = {}
        for col_idx, col_name in enumerate(X_missing.columns):
            observed = X_missing[col_name].dropna()
            col_stats[col_idx] = {
                "mean": observed.mean(),
                "std": observed.std() if observed.std() > 0 else 1.0,
                "min": observed.min(),
                "max": observed.max(),
            }

        gene_stds = np.array([col_stats[c_idx]["std"] for c_idx in missing_cols])
        gene_mins = np.array([col_stats[c_idx]["min"] for c_idx in missing_cols])
        gene_maxs = np.array([col_stats[c_idx]["max"] for c_idx in missing_cols])

        # Subsample inner CV strictly inside outer_train
        if len(X_missing) > self.fitness_subset_size:
            X_sub, _, y_sub, _ = train_test_split(
                X_missing, y_train, train_size=self.fitness_subset_size, random_state=self.random_state, stratify=y_train
            )
            sub_indices = X_sub.index
            y_eval_final = y_sub
        else:
            sub_indices = X_missing.index
            y_eval_final = y_train

        categorical_positions = {}
        for col in cat_cols:
            if col in X_missing.columns:
                col_idx = X_missing.columns.get_loc(col)
                observed_categories = np.sort(X_missing[col].dropna().unique().astype(float))
                if len(observed_categories):
                    categorical_positions[col_idx] = observed_categories

        knn_chromosome = baseline_imputations["KNN"].values[missing_rows, missing_cols].astype(float)

        def repair_chromosome(chromosome):
            values = np.asarray(chromosome, dtype=float).copy()
            values = np.clip(values, gene_mins, gene_maxs)
            for gene_pos, col_idx in enumerate(missing_cols):
                if col_idx in categorical_positions:
                    categories = categorical_positions[col_idx]
                    values[gene_pos] = categories[np.argmin(np.abs(categories - values[gene_pos]))]
            return values

        knn_chromosome = repair_chromosome(knn_chromosome)
        X_missing_values = X_missing.values.copy()

        def evaluate_fitness(chromosome):
            chromosome = repair_chromosome(chromosome)
            X_filled_vals = X_missing_values.copy()
            X_filled_vals[missing_rows, missing_cols] = chromosome
            X_filled_df = pd.DataFrame(X_filled_vals, columns=X_missing.columns, index=X_missing.index)
            X_eval = X_filled_df.loc[sub_indices]
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("rf", RandomForestClassifier(n_estimators=50, max_depth=6, random_state=self.random_state, n_jobs=-1))
            ])
            scores = cross_val_score(pipeline, X_eval, y_eval_final, cv=3, scoring="roc_auc", n_jobs=-1)
            return scores.mean()

        # Hybrid seeding
        population = [knn_chromosome.copy()]
        for name, X_base in (baseline_imputations or {}).items():
            if name == "KNN":
                continue
            population.append(repair_chromosome(X_base.values[missing_rows, missing_cols]))
            if len(population) >= self.pop_size:
                break

        while len(population) < self.pop_size:
            candidate = knn_chromosome + self.rng.normal(0, gene_stds * 0.1)
            population.append(repair_chromosome(candidate))

        fitness_scores = [evaluate_fitness(ind) for ind in population]
        best_fitness_history = []
        best_fitness = -np.inf
        best_chromosome = knn_chromosome.copy()
        no_improvement_count = 0

        gens_run = 0
        for gen in range(self.generations):
            gens_run += 1
            best_idx = int(np.argmax(fitness_scores))
            current_best_fitness = fitness_scores[best_idx]
            best_fitness_history.append(current_best_fitness)

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_chromosome = population[best_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.early_stopping_patience:
                break

            def tournament_select():
                i, j = self.rng.randint(0, len(population), size=2)
                return population[i] if fitness_scores[i] > fitness_scores[j] else population[j]

            new_population = [best_chromosome.copy()]
            if len(new_population) < self.pop_size:
                new_population.append(knn_chromosome.copy())

            while len(new_population) < self.pop_size:
                p1 = tournament_select()
                p2 = tournament_select()
                if self.rng.rand() < self.crossover_rate:
                    crossover_mask = self.rng.rand(len(missing_coords)) < 0.5
                    child = np.where(crossover_mask, p1, p2)
                else:
                    child = p1.copy()

                mutation_mask = self.rng.rand(len(missing_coords)) < self.mutation_rate
                for gene_pos in np.flatnonzero(mutation_mask):
                    col_idx = missing_cols[gene_pos]
                    if col_idx in categorical_positions:
                        categories = categorical_positions[col_idx]
                        if self.rng.rand() < 0.7:
                            child[gene_pos] = knn_chromosome[gene_pos]
                        else:
                            child[gene_pos] = self.rng.choice(categories)
                    else:
                        child[gene_pos] = knn_chromosome[gene_pos] + self.rng.normal(0, gene_stds[gene_pos] * 0.1)
                new_population.append(repair_chromosome(child))

            population = new_population
            fitness_scores = [evaluate_fitness(ind) for ind in population]

        final_best_idx = int(np.argmax(fitness_scores))
        if fitness_scores[final_best_idx] > best_fitness:
            best_fitness = fitness_scores[final_best_idx]
            best_chromosome = population[final_best_idx].copy()

        self.actual_generations_run = gens_run
        self.best_fitness_val = float(best_fitness)

        X_imputed_vals = X_missing_values.copy()
        X_imputed_vals[missing_rows, missing_cols] = repair_chromosome(best_chromosome)
        X_imputed = pd.DataFrame(X_imputed_vals, columns=X_missing.columns, index=X_missing.index)
        for col in cat_cols:
            if col in X_imputed.columns:
                min_v, max_v = self.col_limits[col]
                X_imputed[col] = X_imputed[col].round().clip(lower=min_v, upper=max_v).astype(int)

        # Fit Inductive Reference Map
        self.inductive_imputer = KNNImputer(n_neighbors=5)
        self.inductive_imputer.fit(X_imputed)

        return X_imputed, best_fitness_history

    def transform(self, X_val_missing):
        imputed_vals = self.inductive_imputer.transform(X_val_missing)
        imputed_df = pd.DataFrame(imputed_vals, columns=X_val_missing.columns, index=X_val_missing.index)
        for col, (min_v, max_v) in self.col_limits.items():
            if col in imputed_df.columns:
                imputed_df[col] = imputed_df[col].round().clip(lower=min_v, upper=max_v).astype(int)
        return imputed_df


# 4. DEFINISI KONFIGURASI SENSITIVITAS HYPERPARAMETER GA (OFAT Grid)
param_configs = [
    {
        "Config_Name": "Baseline_Default",
        "Category": "Proposed",
        "pop_size": 15, "generations": 15, "crossover_rate": 0.8, "mutation_rate": 0.2,
        "Description": "Konfigurasi Utama Paper (Pop=15, Gen=15, Cx=0.8, Mut=0.2)"
    },
    # Variasi Pop Size
    {
        "Config_Name": "PopSize_10_Small",
        "Category": "Population_Size",
        "pop_size": 10, "generations": 15, "crossover_rate": 0.8, "mutation_rate": 0.2,
        "Description": "Populasi Kecil (Pop=10): Budget rendah, risiko premature convergence"
    },
    {
        "Config_Name": "PopSize_25_Large",
        "Category": "Population_Size",
        "pop_size": 25, "generations": 15, "crossover_rate": 0.8, "mutation_rate": 0.2,
        "Description": "Populasi Besar (Pop=25): Eksplorasi luas, komputasi ~1.7x lebih berat"
    },
    # Variasi Generations
    {
        "Config_Name": "Gen_8_Low",
        "Category": "Generations",
        "pop_size": 15, "generations": 8, "crossover_rate": 0.8, "mutation_rate": 0.2,
        "Description": "Generasi Rendah (Gen=8): Siklus evolusi terbatas"
    },
    {
        "Config_Name": "Gen_25_High",
        "Category": "Generations",
        "pop_size": 15, "generations": 25, "crossover_rate": 0.8, "mutation_rate": 0.2,
        "Description": "Generasi Tinggi (Gen=25): Menguji efek over-searching / early stopping"
    },
    # Variasi Crossover Rate
    {
        "Config_Name": "Crossover_0.60_Low",
        "Category": "Crossover_Rate",
        "pop_size": 15, "generations": 15, "crossover_rate": 0.6, "mutation_rate": 0.2,
        "Description": "Crossover Rendah (Cx=0.60): Rekombinasi konservatif"
    },
    {
        "Config_Name": "Crossover_0.95_High",
        "Category": "Crossover_Rate",
        "pop_size": 15, "generations": 15, "crossover_rate": 0.95, "mutation_rate": 0.2,
        "Description": "Crossover Tinggi (Cx=0.95): Rekombinasi sangat agresif"
    },
    # Variasi Mutation Rate
    {
        "Config_Name": "Mutation_0.05_Low",
        "Category": "Mutation_Rate",
        "pop_size": 15, "generations": 15, "crossover_rate": 0.8, "mutation_rate": 0.05,
        "Description": "Mutasi Rendah (Mut=0.05): Eksploitasi murni, risiko local minima"
    },
    {
        "Config_Name": "Mutation_0.40_High",
        "Category": "Mutation_Rate",
        "pop_size": 15, "generations": 15, "crossover_rate": 0.8, "mutation_rate": 0.40,
        "Description": "Mutasi Tinggi (Mut=0.40): Stochastisitas tinggi, risiko schemata disruption"
    }
]

# Missing rates representatif spektrum (Rendah, Sedang, Tinggi)
# 15% (Low), 45% (Medium), 75% (Severe - keunggulan GA-KNN)
test_missing_rates = [0.15, 0.45, 0.75]

print(f"\nJumlah Konfigurasi Parameter: {len(param_configs)}")
print(f"Missing Rates yang Diuji: {[f'{int(r*100)}%' for r in test_missing_rates]}")
print(f"Total Eksperimen Sensitivitas: {len(param_configs)} configs x {len(test_missing_rates)} rates = {len(param_configs)*len(test_missing_rates)} evaluasi")

# Setup Outer Split (Outer Fold 1 dari Stratified 5-Fold)
skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
train_idx, val_idx = next(skf_outer.split(X_all, y_all))

X_outer_train_raw = X_all.iloc[train_idx].copy().reset_index(drop=True)
y_outer_train = y_all.iloc[train_idx].copy().reset_index(drop=True)
X_outer_val_raw = X_all.iloc[val_idx].copy().reset_index(drop=True)
y_outer_val = y_all.iloc[val_idx].copy().reset_index(drop=True)

# Encoding Kategorikal (Fit strictly on outer_train)
cat_encoder = {}
X_outer_train_clean = X_outer_train_raw.copy()
X_outer_val_clean = X_outer_val_raw.copy()

for col in categorical_cols:
    train_cats = {val: idx for idx, val in enumerate(X_outer_train_raw[col].unique())}
    cat_encoder[col] = train_cats
    mode_cat = X_outer_train_raw[col].mode()[0]
    fallback_idx = train_cats[mode_cat]
    X_outer_train_clean[col] = X_outer_train_raw[col].map(train_cats)
    X_outer_val_clean[col] = X_outer_val_raw[col].map(lambda x: train_cats.get(x, fallback_idx))

results = []

exp_idx = 0
total_exps = len(param_configs) * len(test_missing_rates)

start_all_time = time.time()

for r in test_missing_rates:
    rate_str = f"{int(r*100)}%"
    print(f"\n=======================================================")
    print(f"MENJALANKAN EXPERIMEN PADA MISSING RATE: {rate_str}")
    print(f"=======================================================")

    # Inject missingness pada train dan val secara independen
    X_train_missing, train_mask, train_orig = introduce_mcar_missing(
        X_outer_train_clean, missing_rate=r, columns=all_features, random_state=RNG_SEED
    )
    X_val_missing, val_mask, val_orig = introduce_mcar_missing(
        X_outer_val_clean, missing_rate=r, columns=all_features, random_state=RNG_SEED + 999
    )

    # Siapkan Baseline Seeding (Mean, Median, KNN) satu kali per missing rate
    mean_imp = SimpleImputer(strategy="mean")
    X_train_mean_vals = mean_imp.fit_transform(X_train_missing)
    X_train_mean = pd.DataFrame(X_train_mean_vals, columns=all_features)

    med_imp = SimpleImputer(strategy="median")
    X_train_med_vals = med_imp.fit_transform(X_train_missing)
    X_train_med = pd.DataFrame(X_train_med_vals, columns=all_features)

    knn_imp = KNNImputer(n_neighbors=5)
    X_train_knn_vals = knn_imp.fit_transform(X_train_missing)
    X_train_knn = pd.DataFrame(X_train_knn_vals, columns=all_features)

    baseline_imputations = {
        "Mean": X_train_mean,
        "Median": X_train_med,
        "KNN": X_train_knn
    }

    for cfg in param_configs:
        exp_idx += 1
        cfg_name = cfg["Config_Name"]
        t_start = time.time()

        # Inisialisasi GAKNNImputer dengan konfigurasi uji
        imputer = GAKNNSensitivityImputer(
            pop_size=cfg["pop_size"],
            generations=cfg["generations"],
            mutation_rate=cfg["mutation_rate"],
            crossover_rate=cfg["crossover_rate"],
            early_stopping_patience=3,
            fitness_subset_size=3000,
            random_state=RNG_SEED
        )

        # 1. Fit & Transform pada Outer Train
        t_fit_start = time.time()
        X_train_imputed, fitness_hist = imputer.fit_transform(
            X_train_missing, y_outer_train,
            baseline_imputations=baseline_imputations,
            cat_cols=categorical_cols
        )
        t_fit_end = time.time()
        train_impute_time = t_fit_end - t_fit_start

        # 2. Inductive Transform pada Outer Val (Deployment)
        t_infer_start = time.time()
        X_val_incomp_imputed = imputer.transform(X_val_missing)
        t_infer_end = time.time()
        inference_time = t_infer_end - t_infer_start

        # 3. Hitung Metrik Rekonstruksi (Val RMSE & PFC)
        val_rmse, val_pfc = calculate_imputation_metrics(
            val_orig, val_mask, X_val_incomp_imputed, continuous_cols, categorical_cols
        )

        # 4. Standardisasi dan Latih Downstream Random Forest pada X_train_imputed
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train_imputed)
        X_val_clean_scaled = scaler.transform(X_outer_val_clean)
        X_val_incomp_scaled = scaler.transform(X_val_incomp_imputed)

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RNG_SEED, n_jobs=-1)
        rf.fit(X_tr_scaled, y_outer_train)

        # Evaluasi Clean Val (Kontrol)
        clean_probs = rf.predict_proba(X_val_clean_scaled)[:, 1]
        clean_preds = rf.predict(X_val_clean_scaled)
        clean_auc = roc_auc_score(y_outer_val, clean_probs)
        clean_f1 = f1_score(y_outer_val, clean_preds)
        clean_acc = accuracy_score(y_outer_val, clean_preds)

        # Evaluasi Incomplete Val (Deployment Realism)
        incomp_probs = rf.predict_proba(X_val_incomp_scaled)[:, 1]
        incomp_preds = rf.predict(X_val_incomp_scaled)
        incomp_auc = roc_auc_score(y_outer_val, incomp_probs)
        incomp_f1 = f1_score(y_outer_val, incomp_preds)
        incomp_acc = accuracy_score(y_outer_val, incomp_preds)

        t_total = time.time() - t_start

        record = {
            "Config_Name": cfg_name,
            "Category": cfg["Category"],
            "pop_size": cfg["pop_size"],
            "generations": cfg["generations"],
            "crossover_rate": cfg["crossover_rate"],
            "mutation_rate": cfg["mutation_rate"],
            "Missing_Rate": r,
            "Actual_Generations": imputer.actual_generations_run,
            "Best_Inner_Fitness": imputer.best_fitness_val,
            "Clean_Val_ROC_AUC": clean_auc,
            "Clean_Val_F1": clean_f1,
            "Clean_Val_Accuracy": clean_acc,
            "Incomp_Val_ROC_AUC": incomp_auc,
            "Incomp_Val_F1": incomp_f1,
            "Incomp_Val_Accuracy": incomp_acc,
            "Val_RMSE": val_rmse,
            "Val_PFC": val_pfc,
            "Train_Impute_Runtime": train_impute_time,
            "Inference_Runtime": inference_time,
            "Total_Runtime": t_total
        }
        results.append(record)

        print(f"[{exp_idx}/{total_exps}] {cfg_name:22s} | Rate: {rate_str} | "
              f"Clean AUC: {clean_auc:.4f} | Incomp AUC: {incomp_auc:.4f} | "
              f"Gens: {imputer.actual_generations_run}/{cfg['generations']} | Time: {train_impute_time:.1f}s")

# Simpan ke CSV
res_df = pd.DataFrame(results)
output_csv = "d:/skripsi/topic1/ga_sensitivity_results.csv"
res_df.to_csv(output_csv, index=False)
print(f"\n[OK] Hasil Sensitivity Analysis Berhasil Disimpan: {output_csv}")
print(f"Total Waktu Eksekusi: {time.time() - start_all_time:.1f} detik")
