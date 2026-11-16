import glob
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from evaluation.post_process import *


def _reform_data_from_dict(data, flatten=True):
    if isinstance(data, dict):
        sort_data = sorted(data.items(), key=lambda x: x[0])
        sort_data = [i[1] for i in sort_data]
        sort_data = torch.cat(sort_data, dim=0)
    else:
        sort_data = data

    if flatten:
        if isinstance(sort_data, torch.Tensor):
            sort_data = np.reshape(sort_data.cpu(), (-1))
        else:
            sort_data = np.reshape(np.array(sort_data), (-1))
    else:
        sort_data = np.array(sort_data.cpu())
    return sort_data


def get_video_paths(data_path):
    all_items = glob.glob(data_path + os.sep + "*")
    data_dirs = [d for d in all_items if os.path.isdir(d)]
    return sorted(data_dirs)


def aggregate_predictions(predictions):
    """
    Reagrupa chunks fragmentados associando-os ao vídeo correto,
    preservando o alinhamento temporal. Buracos viram NaN.
    """
    full_signals = {}

    for key, value in predictions.items():
        key_str = str(key)
        if "_" in key_str:
            parts = key_str.split("_")
            video_idx = int(parts[0])
            chunk_idx = int(parts[1])
        else:
            video_idx = int(key_str)
            chunk_idx = 0

        chunk_signal = _reform_data_from_dict(value)

        if video_idx not in full_signals:
            full_signals[video_idx] = {}

        full_signals[video_idx][chunk_idx] = chunk_signal

    aggregated_predictions = {}
    for v_idx, chunks_dict in full_signals.items():
        if not chunks_dict:
            continue

        c_len = len(next(iter(chunks_dict.values())))
        max_idx = max(chunks_dict.keys())

        total_length = (max_idx + 1) * c_len
        continuous_signal = np.full(total_length, np.nan)

        for c_idx, sig in chunks_dict.items():
            start = c_idx * c_len
            end = start + len(sig)
            continuous_signal[start:end] = sig

        aggregated_predictions[str(v_idx)] = continuous_signal

    return aggregated_predictions


def _safe_getattr(obj, attr, default=None):
    try:
        return getattr(obj, attr)
    except Exception:
        return default


def _get_nested_attr(obj, attr_path, default=None):
    current = obj
    for attr in attr_path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(attr, default)
        else:
            current = _safe_getattr(current, attr, default)
    return current


def _get_experiment_context(config, data_root):
    """
    Obtém a identidade experimental preferencialmente das variáveis de ambiente
    exportadas pelo script de execução. Isso evita modificar YAMLs e main.py.
    """
    model_from_config = _get_nested_attr(config, "MODEL.NAME", "unknown_model")
    model_path = _get_nested_attr(config, "INFERENCE.MODEL_PATH", "")
    checkpoint_name = os.path.basename(model_path) if model_path else ""

    experiment_id = os.environ.get("TCC_EXPERIMENT_ID")
    model_name = os.environ.get("TCC_MODEL_NAME")
    pretrain_dataset = os.environ.get("TCC_PRETRAIN_DATASET")
    output_dir = os.environ.get("TCC_OUTPUT_DIR")
    config_file = os.environ.get("TCC_CONFIG_FILE", "")

    if not model_name:
        model_name = str(model_from_config)

    if not pretrain_dataset:
        pretrain_dataset = "unknown_pretrain"

    if not experiment_id:
        if checkpoint_name:
            experiment_id = f"{model_name}_{os.path.splitext(checkpoint_name)[0]}"
        else:
            experiment_id = f"{model_name}_{pretrain_dataset}"

    if not output_dir:
        output_dir = os.path.join(data_root, "evaluation_outputs", experiment_id)

    return {
        "experiment_id": str(experiment_id),
        "model_name": str(model_name),
        "pretrain_dataset": str(pretrain_dataset),
        "checkpoint_name": str(checkpoint_name),
        "model_path": str(model_path),
        "config_file": str(config_file),
        "output_dir": str(output_dir),
    }


def _read_hr_events(csv_path):
    """
    Lê o arquivo hr_sync.csv do dataset atual.

    Formato esperado:
        timestamp_brt,heart_rate_bpm,relative_video_time
        2026-04-19 18:31:08,106,1.48

    A coluna timestamp_brt é ignorada. As colunas relevantes são:
        - heart_rate_bpm
        - relative_video_time
    """
    df = pd.read_csv(csv_path)

    required_cols = {"heart_rate_bpm", "relative_video_time"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"CSV de HR inválido: faltam colunas {sorted(missing)} em {csv_path}"
        )

    bpm = pd.to_numeric(df["heart_rate_bpm"], errors="coerce")
    rel_time = pd.to_numeric(df["relative_video_time"], errors="coerce")

    out = pd.DataFrame({"gt_bpm": bpm, "timestamp": rel_time})
    out = out.dropna(subset=["gt_bpm", "timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    return out


def _read_failure_intervals(annotation_path):
    """
    Lê intervalos de falha no formato atual:
        FAIL 74.174 75.042

    Também aceita, por segurança, linhas antigas no formato:
        74.174,75.042
    """
    intervals = []

    if not os.path.exists(annotation_path):
        return intervals

    with open(annotation_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            raw_line = line.strip()
            if not raw_line or raw_line.startswith("#"):
                continue

            try:
                parts = raw_line.replace(",", " ").split()

                if parts[0].upper() == "FAIL" and len(parts) >= 3:
                    start_sec = float(parts[1])
                    end_sec = float(parts[2])
                elif len(parts) >= 2:
                    start_sec = float(parts[0])
                    end_sec = float(parts[1])
                else:
                    raise ValueError("linha sem início/fim")

                if end_sec > start_sec:
                    intervals.append((start_sec, end_sec))
            except Exception as e:
                print(
                    f"[AVISO] Não foi possível ler anotação '{raw_line}' "
                    f"em {annotation_path}:{line_num}. Erro: {e}"
                )

    return intervals


def _window_overlaps_intervals(start_time, end_time, intervals):
    for fail_start, fail_end in intervals:
        if start_time < fail_end and end_time > fail_start:
            return True
    return False


def _mask_intervals(signal, intervals, fs):
    masked = signal.copy()
    for start_sec, end_sec in intervals:
        start_f = max(0, int(start_sec * fs))
        end_f = min(len(masked), int(end_sec * fs))
        if start_f < len(masked) and start_f < end_f:
            masked[start_f:end_f] = np.nan
    return masked


def _base_event_record(
    context,
    video_idx,
    subject_id,
    subject_dir,
    primary_video_path,
    hr_csv_path,
    annotation_path,
    timestamp,
    gt_bpm,
    start_frame,
    end_frame,
    fs,
    window_seconds,
):
    return {
        "experiment_id": context["experiment_id"],
        "model_name": context["model_name"],
        "pretrain_dataset": context["pretrain_dataset"],
        "checkpoint_name": context["checkpoint_name"],
        "model_path": context["model_path"],
        "config_file": context["config_file"],
        "video_idx": video_idx,
        "subject_id": subject_id,
        "video_name": subject_id,
        "video_dir": subject_dir,
        "primary_video_path": primary_video_path,
        "hr_csv_path": hr_csv_path,
        "annotation_path": annotation_path,
        "timestamp": float(timestamp),
        "start_time": float(start_frame / fs) if start_frame is not None else np.nan,
        "end_time": float(end_frame / fs) if end_frame is not None else np.nan,
        "start_frame": int(start_frame) if start_frame is not None else np.nan,
        "end_frame_exclusive": int(end_frame) if end_frame is not None else np.nan,
        "gt_bpm": float(gt_bpm),
        "pred_bpm": np.nan,
        "abs_error": np.nan,
        "squared_error": np.nan,
        "percentage_error": np.nan,
        "valid": False,
        "discard_reason": "",
        "window_seconds": float(window_seconds),
        "fs": float(fs),
        "hr_method": "FFT",
    }


def _compute_metrics_from_arrays(pred, gt):
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)

    valid = ~np.isnan(pred) & ~np.isnan(gt)
    pred = pred[valid]
    gt = gt[valid]

    n = len(gt)
    if n == 0:
        return {
            "n_valid_events": 0,
            "MAE": np.nan,
            "MAE_std_err": np.nan,
            "RMSE": np.nan,
            "RMSE_std_err": np.nan,
            "MAPE": np.nan,
            "MAPE_std_err": np.nan,
            "Pearson": np.nan,
            "Pearson_std_err": np.nan,
            "median_abs_error": np.nan,
            "std_abs_error": np.nan,
        }

    errors = pred - gt
    abs_errors = np.abs(errors)
    sq_errors = np.square(errors)

    safe_gt = gt.copy()
    safe_gt[safe_gt == 0] = np.nan
    perc_errors = np.abs(errors / safe_gt) * 100

    mae = np.nanmean(abs_errors)
    rmse = np.sqrt(np.nanmean(sq_errors))
    mape = np.nanmean(perc_errors)

    if n > 1 and np.nanstd(pred) >= 1e-5 and np.nanstd(gt) >= 1e-5:
        pearson = np.corrcoef(pred, gt)[0][1]
        pearson_std_err = np.sqrt((1 - pearson**2) / (n - 2)) if n > 2 else np.nan
    else:
        pearson = np.nan
        pearson_std_err = np.nan

    return {
        "n_valid_events": int(n),
        "MAE": float(mae),
        "MAE_std_err": float(np.nanstd(abs_errors) / np.sqrt(n)) if n > 0 else np.nan,
        "RMSE": float(rmse),
        "RMSE_std_err": float(np.nanstd(np.sqrt(sq_errors)) / np.sqrt(n)) if n > 0 else np.nan,
        "MAPE": float(mape) if not np.isnan(mape) else np.nan,
        "MAPE_std_err": float(np.nanstd(perc_errors) / np.sqrt(n)) if n > 0 else np.nan,
        "Pearson": float(pearson) if not np.isnan(pearson) else np.nan,
        "Pearson_std_err": float(pearson_std_err) if not np.isnan(pearson_std_err) else np.nan,
        "median_abs_error": float(np.nanmedian(abs_errors)),
        "std_abs_error": float(np.nanstd(abs_errors)),
    }


def _summarize_by_subject(events_df):
    rows = []

    if events_df.empty:
        return pd.DataFrame(rows)

    for subject_id, group in events_df.groupby("subject_id", dropna=False):
        total = len(group)
        valid_group = group[group["valid"] == True]
        metrics = _compute_metrics_from_arrays(valid_group["pred_bpm"], valid_group["gt_bpm"])

        row = {
            "experiment_id": group["experiment_id"].iloc[0],
            "model_name": group["model_name"].iloc[0],
            "pretrain_dataset": group["pretrain_dataset"].iloc[0],
            "checkpoint_name": group["checkpoint_name"].iloc[0],
            "subject_id": subject_id,
            "video_name": group["video_name"].iloc[0],
            "video_dir": group["video_dir"].iloc[0],
            "n_total_events": int(total),
            "n_valid_events": int(len(valid_group)),
            "n_invalid_events": int(total - len(valid_group)),
            "coverage_rate": float(len(valid_group) / total) if total > 0 else np.nan,
        }
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def _build_group_metrics(events_df):
    """
    Gera uma tabela inicial de métricas por grupos disponíveis sem metadados externos.
    Futuramente, event_predictions.csv pode ser combinado com session_metadata.csv para
    gerar grupos como dia/noite, oclusão, fala etc.
    """
    rows = []

    if events_df.empty:
        return pd.DataFrame(rows)

    group_specs = [
        ("global", None),
        ("subject_id", "subject_id"),
        ("discard_reason", "discard_reason"),
    ]

    for group_name, col in group_specs:
        if col is None:
            groups = [("all", events_df)]
        else:
            groups = list(events_df.groupby(col, dropna=False))

        for group_value, group in groups:
            total = len(group)
            valid_group = group[group["valid"] == True]
            metrics = _compute_metrics_from_arrays(valid_group["pred_bpm"], valid_group["gt_bpm"])

            row = {
                "experiment_id": group["experiment_id"].iloc[0],
                "model_name": group["model_name"].iloc[0],
                "pretrain_dataset": group["pretrain_dataset"].iloc[0],
                "checkpoint_name": group["checkpoint_name"].iloc[0],
                "group_by": group_name,
                "group_value": str(group_value),
                "n_total_events": int(total),
                "n_valid_events": int(len(valid_group)),
                "n_invalid_events": int(total - len(valid_group)),
                "coverage_rate": float(len(valid_group) / total) if total > 0 else np.nan,
            }
            row.update(metrics)
            rows.append(row)

    return pd.DataFrame(rows)


def _write_summary(output_dir, context, events_df, subject_metrics_df, group_metrics_df):
    total_events = int(len(events_df))
    valid_events = int(events_df["valid"].sum()) if not events_df.empty else 0
    coverage_rate = float(valid_events / total_events) if total_events > 0 else np.nan

    valid_df = events_df[events_df["valid"] == True] if not events_df.empty else pd.DataFrame()
    metrics = _compute_metrics_from_arrays(valid_df.get("pred_bpm", []), valid_df.get("gt_bpm", []))

    summary = {
        "experiment_id": context["experiment_id"],
        "model_name": context["model_name"],
        "pretrain_dataset": context["pretrain_dataset"],
        "checkpoint_name": context["checkpoint_name"],
        "model_path": context["model_path"],
        "config_file": context["config_file"],
        "status": "completed",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "n_subjects": int(events_df["subject_id"].nunique()) if not events_df.empty else 0,
        "n_total_events": total_events,
        "n_valid_events": valid_events,
        "n_invalid_events": int(total_events - valid_events),
        "coverage_rate": coverage_rate,
        "global_metrics": metrics,
        "outputs": {
            "event_predictions": os.path.join(output_dir, "event_predictions.csv"),
            "subject_metrics": os.path.join(output_dir, "subject_metrics.csv"),
            "group_metrics": os.path.join(output_dir, "group_metrics.csv"),
        },
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    completed_path = os.path.join(output_dir, "COMPLETED")
    with open(completed_path, "w", encoding="utf-8") as f:
        f.write(f"completed_at={summary['completed_at']}\n")
        f.write(f"experiment_id={context['experiment_id']}\n")
        f.write(f"model_name={context['model_name']}\n")
        f.write(f"pretrain_dataset={context['pretrain_dataset']}\n")
        f.write(f"n_total_events={total_events}\n")
        f.write(f"n_valid_events={valid_events}\n")

    return summary


def calculate_metrics(predictions, labels, config):
    fs = config.TEST.DATA.FS
    WINDOW_SECONDS = 10.0
    window_frame_size = int(WINDOW_SECONDS * fs)

    data_root = config.TEST.DATA.DATA_PATH
    video_folders = get_video_paths(data_root)

    context = _get_experiment_context(config, data_root)
    output_dir = context["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Reagrupa chunks mantendo a linha do tempo contínua
    predictions = aggregate_predictions(predictions)

    event_records = []
    predict_hr_all = []
    gt_hr_all = []

    for index in predictions.keys():
        try:
            video_idx = int(index)
            if video_idx >= len(video_folders):
                print(f"[AVISO] video_idx {video_idx} fora do intervalo de pastas do dataset.")
                continue

            subject_dir = video_folders[video_idx]
            subject_id = os.path.basename(subject_dir)
            primary_video_path = os.path.join(subject_dir, "primary_video.mp4")
            hr_csv_path = os.path.join(subject_dir, "hr_sync.csv")
            annotation_path = os.path.join(subject_dir, "primary_video_auto_anotacoes.txt")

            if not os.path.exists(hr_csv_path):
                print(f"[AVISO] hr_sync.csv não encontrado para {subject_id}: {hr_csv_path}")
                continue

            try:
                hr_events = _read_hr_events(hr_csv_path)
            except Exception as e:
                print(f"[AVISO] Não foi possível ler HR para {subject_id}: {e}")
                continue

            if hr_events.empty:
                print(f"[AVISO] Nenhum evento de HR válido para {subject_id}.")
                continue

            failure_intervals = _read_failure_intervals(annotation_path)
            prediction_signal = _mask_intervals(predictions[index], failure_intervals, fs)

            for _, row in hr_events.iterrows():
                gt_bpm = float(row["gt_bpm"])
                timestamp = float(row["timestamp"])

                end_frame = int(timestamp * fs)
                start_frame = end_frame - window_frame_size

                base_record = _base_event_record(
                    context=context,
                    video_idx=video_idx,
                    subject_id=subject_id,
                    subject_dir=subject_dir,
                    primary_video_path=primary_video_path,
                    hr_csv_path=hr_csv_path,
                    annotation_path=annotation_path,
                    timestamp=timestamp,
                    gt_bpm=gt_bpm,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    fs=fs,
                    window_seconds=WINDOW_SECONDS,
                )

                if timestamp < WINDOW_SECONDS:
                    base_record["discard_reason"] = "before_window_seconds"
                    event_records.append(base_record)
                    continue

                if start_frame < 0 or end_frame > len(prediction_signal):
                    base_record["discard_reason"] = "window_out_of_prediction_bounds"
                    event_records.append(base_record)
                    continue

                start_time = start_frame / fs
                end_time = end_frame / fs

                if _window_overlaps_intervals(start_time, end_time, failure_intervals):
                    base_record["discard_reason"] = "annotated_failure_interval"
                    event_records.append(base_record)
                    continue

                pred_window = prediction_signal[start_frame:end_frame]

                if len(pred_window) != window_frame_size or np.isnan(pred_window).any():
                    base_record["discard_reason"] = "missing_or_nan_prediction"
                    event_records.append(base_record)
                    continue

                diff_flag = config.TEST.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized"
                dummy_lbl = np.zeros_like(pred_window)

                _, pred_bpm, _, _ = calculate_metric_per_video(
                    pred_window,
                    dummy_lbl,
                    diff_flag=diff_flag,
                    fs=fs,
                    hr_method="FFT",
                )

                abs_error = abs(pred_bpm - gt_bpm)
                squared_error = (pred_bpm - gt_bpm) ** 2
                percentage_error = abs((pred_bpm - gt_bpm) / gt_bpm) * 100 if gt_bpm != 0 else np.nan

                base_record.update(
                    {
                        "pred_bpm": float(pred_bpm),
                        "abs_error": float(abs_error),
                        "squared_error": float(squared_error),
                        "percentage_error": float(percentage_error) if not np.isnan(percentage_error) else np.nan,
                        "valid": True,
                        "discard_reason": "",
                    }
                )

                event_records.append(base_record)
                predict_hr_all.append(pred_bpm)
                gt_hr_all.append(gt_bpm)

        except Exception as e:
            print(f"Erro ao processar vídeo {index}: {e}")
            continue

    events_df = pd.DataFrame(event_records)

    if events_df.empty:
        # Garante schema mínimo mesmo sem eventos, para facilitar automação.
        events_df = pd.DataFrame(
            columns=[
                "experiment_id", "model_name", "pretrain_dataset", "checkpoint_name",
                "model_path", "config_file", "video_idx", "subject_id", "video_name",
                "video_dir", "primary_video_path", "hr_csv_path", "annotation_path",
                "timestamp", "start_time", "end_time", "start_frame",
                "end_frame_exclusive", "gt_bpm", "pred_bpm", "abs_error",
                "squared_error", "percentage_error", "valid", "discard_reason",
                "window_seconds", "fs", "hr_method",
            ]
        )

    subject_metrics_df = _summarize_by_subject(events_df)
    group_metrics_df = _build_group_metrics(events_df)

    events_path = os.path.join(output_dir, "event_predictions.csv")
    subject_path = os.path.join(output_dir, "subject_metrics.csv")
    group_path = os.path.join(output_dir, "group_metrics.csv")

    events_df.to_csv(events_path, index=False)
    subject_metrics_df.to_csv(subject_path, index=False)
    group_metrics_df.to_csv(group_path, index=False)

    summary = _write_summary(output_dir, context, events_df, subject_metrics_df, group_metrics_df)

    # === RELATÓRIO FINAL DETALHADO ===
    predict_hr_all = np.array(predict_hr_all)
    gt_hr_all = np.array(gt_hr_all)
    num = len(gt_hr_all)

    print("-" * 70)
    print(
        f"RELATÓRIO DE BPM (EVENT-BASED) | "
        f"Experimento: {context['experiment_id']} | Amostras Válidas: {num}"
    )
    print("-" * 70)

    if num > 0:
        metrics = summary["global_metrics"]
        print(f"MAE:     {metrics['MAE']:.4f} +/- {metrics['MAE_std_err']:.4f} BPM")
        print(f"RMSE:    {metrics['RMSE']:.4f} +/- {metrics['RMSE_std_err']:.4f} BPM")
        print(f"MAPE:    {metrics['MAPE']:.4f} +/- {metrics['MAPE_std_err']:.4f} %")

        if np.isnan(metrics["Pearson"]):
            print("Pearson: NaN (Falta de variância nos dados ou amostras insuficientes)")
        else:
            if np.isnan(metrics["Pearson_std_err"]):
                print(f"Pearson: {metrics['Pearson']:.4f}")
            else:
                print(f"Pearson: {metrics['Pearson']:.4f} +/- {metrics['Pearson_std_err']:.4f}")

        print(f"Cobertura: {summary['coverage_rate']:.4f} ({summary['n_valid_events']}/{summary['n_total_events']})")
    else:
        print("SEM DADOS VÁLIDOS PARA AVALIAÇÃO.")
        print(f"Eventos totais registrados: {summary['n_total_events']}")

    print("-" * 70)
    print(f"CSV de eventos:  {events_path}")
    print(f"CSV por subject: {subject_path}")
    print(f"CSV de grupos:   {group_path}")
    print(f"Resumo JSON:     {os.path.join(output_dir, 'summary.json')}")
    print(f"Marcador:        {os.path.join(output_dir, 'COMPLETED')}")
    print("-" * 70 + "\n")
