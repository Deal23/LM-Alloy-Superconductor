import os
import numpy as np
import pandas as pd
import time
import itertools
import json
from joblib import load, Parallel, delayed
from datetime import datetime
import math
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)
from utils import parse_formula, generate_random_formula, METAL_ELEMENTS


def _read_task_params():
    raw = os.environ.get("LMSC_TASK_PARAMS", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


TASK_PARAMS = _read_task_params()

num_elements = int(TASK_PARAMS.get("num_elements", 2))
num_formulas = int(TASK_PARAMS.get("num_formulas", 1000))
num_workers = int(TASK_PARAMS.get("num_workers", 4))
model_key = str(TASK_PARAMS.get("model_key", "ET_web"))
elements_param = TASK_PARAMS.get("elements", [])
stability_check = bool(TASK_PARAMS.get("stability_check", False))


def _resolve_model_files(key):
    candidates = [key]
    if key in {"ET", "RF"}:
        candidates = [f"{key}_web", key]
    for cand in candidates:
        model_path = os.path.join(PROJECT_ROOT, "ml", "models", f"{cand}.joblib")
        feature_path = os.path.join(
            PROJECT_ROOT, "ml", "models", f"{cand}_feature.joblib"
        )
        if os.path.exists(model_path) and os.path.exists(feature_path):
            return cand, model_path, feature_path
    return None, None, None


resolved_model_key, model_path, feature_path = _resolve_model_files(model_key)
if not resolved_model_key:
    raise FileNotFoundError(f"Model files not found for model_key={model_key}")

CONFIG = {
    "model_path": model_path,
    "feature_names_path": feature_path,
    "log_file": os.path.join(
        PROJECT_ROOT,
        "results",
        "predictions",
        f"alloy_log_{datetime.now().strftime('%m%d%H%M')}.txt",
    ),
    "epoch": math.ceil(math.comb(66, num_elements) * 2 / 3600),
    "output_csv": os.path.join(
        PROJECT_ROOT,
        "results",
        "predictions",
        f"{num_elements}Alloy_{resolved_model_key.lower()}.csv",
    ),
}

model = load(CONFIG["model_path"])
feature_names = load(CONFIG["feature_names_path"])


def predict_transition_temp(model, feature_names, elements_ratios):
    feature_row = {
        element: elements_ratios.get(element, 0) for element in feature_names
    }
    input_features = pd.DataFrame([feature_row], columns=feature_names)
    predicted_temp = model.predict(input_features)
    return predicted_temp[0]


def log_message(message):
    try:
        with open(CONFIG["log_file"], "a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")
    except Exception:
        pass


def f_pre(formula):
    try:
        return predict_transition_temp(model, feature_names, parse_formula(formula))
    except Exception as e:
        log_message(f"Error processing formula {formula}: {e}")
        return -1


metal_elements = METAL_ELEMENTS

if elements_param:
    metal_elements = [e for e in METAL_ELEMENTS if e in elements_param]
    if len(metal_elements) < num_elements:
        metal_elements = METAL_ELEMENTS

elements_combination = []
for ec in itertools.combinations(metal_elements, num_elements):
    elements_combination.append("-".join(ec))
df = pd.DataFrame(elements_combination, columns=["elements"])
df["calculated"] = "no"


def ensure_result_columns(frame):
    """Initialize columns before scalar assignment, including resumed CSVs."""
    defaults = {
        "Predicted_Tc": np.nan,
        "Elements": pd.Series(None, index=frame.index, dtype="object"),
        "stability_variance": np.nan,
        "stability_log": pd.Series(None, index=frame.index, dtype="object"),
    }
    for name, default in defaults.items():
        if name not in frame.columns:
            frame[name] = default
    # Older CSVs may infer all-empty string columns as float.
    for name in ("Elements", "stability_log"):
        frame[name] = frame[name].astype(object)
    return frame


def update_cal(df, num_formulas, epoch=10):
    ensure_result_columns(df)
    start_time = time.time()
    c_epoch = 0
    last_processed = None

    # 稳定性分析梯度
    sampling_gradients = (
        [int(num_formulas * 0.1), int(num_formulas * 0.5), num_formulas]
        if stability_check
        else [num_formulas]
    )
    # 确保最小值
    sampling_gradients = [max(10, g) for g in sampling_gradients]

    for index, row in df.iterrows():
        if row.get("calculated", "no") == "no":
            alloy_time = time.time()
            elements = str(row["elements"]).split("-")

            best_tcs = []
            best_formulas = []

            for g_size in sampling_gradients:
                formulas = generate_random_formula(elements, g_size)
                predict = pd.DataFrame(formulas, columns=["formula"])

                formulas_series = predict["formula"]
                formulas_list = [
                    str(x)
                    for x in (formulas_series if formulas_series is not None else [])
                ]

                results_raw = Parallel(
                    n_jobs=num_workers, prefer="processes", backend="loky"
                )(delayed(f_pre)(f) for f in formulas_list)

                results = [r if r is not None else -1 for r in (results_raw or [])]
                
                # Create a proper Series to avoid typing issues with fillna and idxmax
                tc_series = pd.Series(results, dtype="float64")
                tc_series = tc_series.fillna(-1.0)
                predict["Predicted_Tc"] = tc_series
                
                if not predict.empty and "Predicted_Tc" in predict.columns:
                    # Ignore typing issues and just get idxmax
                    col_vals = predict["Predicted_Tc"]
                    max_index = pd.Series(list(col_vals) if col_vals is not None else [], dtype=float).idxmax()
                    max_predicted_tc = predict.at[max_index, "Predicted_Tc"]
                    
                    best_tcs.append(max_predicted_tc)
                    best_formulas.append(predict.at[max_index, "formula"])
                else:
                    best_tcs.append(0)
                    best_formulas.append(elements[0])

            # 使用最大采样的结果作为最终结果
            df.at[index, "Predicted_Tc"] = best_tcs[-1]
            df.at[index, "Elements"] = best_formulas[-1]
            df.at[index, "calculated"] = "yes"

            if stability_check:
                df.at[index, "stability_variance"] = np.var(best_tcs)
                df.at[index, "stability_log"] = (
                    f"[{', '.join([f'{t:.2f}' for t in best_tcs])}]"
                )

            alloy_time = time.time() - alloy_time
            last_processed = (str(row["elements"]), alloy_time)

            log_message(
                f"[{datetime.now().strftime('%H:%M:%S')}] {row['elements']} - 采样 {num_formulas} 次 -> 最佳Tc: {best_tcs[-1]:.2f}K ({best_formulas[-1]})"
            )

            c_epoch += 1
            print("LMSC_PROGRESS " + json.dumps({"completed": int((df["calculated"] == "yes").sum()), "total": len(df)}), flush=True)
            if c_epoch > int(len(df) / epoch):
                break

    elapsed_time = time.time() - start_time
    if last_processed:
        log_message(
            f"\n{last_processed[0]} finished in {last_processed[1]:.2f} seconds"
        )
    log_message(f"\nElapsed time: {elapsed_time:.2f} seconds")

    return df


def main():
    global df
    os.makedirs(os.path.dirname(CONFIG["output_csv"]), exist_ok=True)
    if os.path.exists(CONFIG["output_csv"]):
        epoch = CONFIG["epoch"]
        log_message(f"File {CONFIG['output_csv']} exists. Resuming calculation.")
        log_message(f"Total {epoch} epochs")

        try:
            loaded_df = pd.read_csv(CONFIG["output_csv"])
            if isinstance(loaded_df, pd.DataFrame):
                df = loaded_df
        except Exception:
            pass

        for i in range(epoch):
            update_cal(df, num_formulas, epoch)
            df.to_csv(CONFIG["output_csv"], index=False)
            log_message(
                f"\n-----------------------Epoch {i + 1} finished-----------------------------"
            )
    else:
        log_message(
            f"File {CONFIG['output_csv']} does not exist. Starting new calculation."
        )
        epoch = CONFIG["epoch"]
        log_message(f"Total {epoch} epochs")
        for i in range(epoch):
            update_cal(df, num_formulas, epoch)
            df.to_csv(CONFIG["output_csv"], index=False)
            log_message(
                f"\n-----------------------Epoch {i + 1} finished-----------------------------"
            )


if __name__ == "__main__":
    main()
