# ===============================
# routers/standard.py （融合简化版）
# ===============================
import os
import torch
import pickle
import numpy as np
import pandas as pd
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse

# ===== 导入模型类 =====
from models.LSTM import LSTMModel
from models.BiLSTM import BiLSTMModel
from models.GRU import GRUModel
from models.BiGRU import BiGRUModel
from models.Transformer import TransformerModel
from models.LSTM_Attention import LSTMAttention
from models.GRU_Attention import GRUAttention

router = APIRouter()

# ===== 路径配置 =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
BUILTIN_DIR = os.path.join(DATA_DIR, "builtin")
Softmax_DIR = os.path.join(DATA_DIR, "fusion")
WEIGHT_DIR = os.path.join(BASE_DIR, "weights")
RESULT_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)


# ===== 工具函数 =====
def ensure_exists(path: str, hint_dir: str = None):
    if not os.path.exists(path):
        msg = f"文件不存在：{path}"
        if hint_dir and os.path.isdir(hint_dir):
            msg += f"；请检查目录 {hint_dir} 下的文件：{os.listdir(hint_dir)}"
        raise FileNotFoundError(msg)


def inverse_scale_latlon(pred, scaler):
    """反归一化预测结果 -> 经纬度"""
    pred = np.asarray(pred, dtype=np.float32).copy()
    lat_min, lat_max = scaler["lat"]["min"], scaler["lat"]["max"]
    lon_min, lon_max = scaler["lon"]["min"], scaler["lon"]["max"]
    pred[:, 0] = pred[:, 0] * (lat_max - lat_min) + lat_min
    pred[:, 1] = pred[:, 1] * (lon_max - lon_min) + lon_min
    return pred


def load_model(model_name: str, step: int):
    """按名称加载单模型权重"""
    model_map = {
        "LSTM": LSTMModel,
        "BiLSTM": BiLSTMModel,
        "GRU": GRUModel,
        "BiGRU": BiGRUModel,
        "Transformer": TransformerModel,
        "LSTM-Attention": LSTMAttention,
        "GRU-Attention": GRUAttention,
    }
    model_class = model_map[model_name]
    model = model_class(pred_steps=step)
    weight_path = os.path.join(WEIGHT_DIR, f"{model_name}_weights_{model_name}_S{step}_E200_all.pth")
    ensure_exists(weight_path, WEIGHT_DIR)
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ===== 融合预测（标准模式专用） =====
@router.post("/predict_fusion")
async def predict_fusion(data: dict = Body(...)):
    """
    标准模式：直接融合预测（7个模型固定）
    支持多个样本编号
    """
    try:
        step = int(data.get("step", 15))
        samples = str(data.get("samples", "0"))
        sample_indices = [int(i) for i in samples.split(",") if i.strip().isdigit()]

        models_selected = [
            "LSTM", "BiLSTM", "GRU", "BiGRU",
            "Transformer", "LSTM-Attention", "GRU-Attention"
        ]

        # ==== 路径检查 ====
        scaler_path = os.path.join(BUILTIN_DIR, f"global_scaling_params_{step}_all.pkl")
        X_path = os.path.join(BUILTIN_DIR, f"X_test_{step}_all.pkl")
        softmax_path = os.path.join(Softmax_DIR, f"softmax_weights_{step}_all.npy")
        ensure_exists(scaler_path, BUILTIN_DIR)
        ensure_exists(X_path, BUILTIN_DIR)
        ensure_exists(softmax_path, Softmax_DIR)

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_test = pickle.load(open(X_path, "rb"))
        n_samples = len(X_test)

        valid_indices = [i for i in sample_indices if 0 <= i < n_samples]
        skipped = [i for i in sample_indices if i not in valid_indices]
        if not valid_indices:
            return JSONResponse({"ok": False, "error": f"无效样本编号 {sample_indices}，范围应为 0~{n_samples-1}。"})

        # ==== 加载 softmax 权重 ====
        weights = np.load(softmax_path)
        weights = np.squeeze(weights)
        if weights.ndim == 4:
            weights = np.squeeze(weights).T
        if weights.shape[0] != 7:
            weights = weights.T
        weights = np.exp(weights) / np.sum(np.exp(weights), axis=0, keepdims=True)

        # ==== 预测与融合 ====
        fusion_results, input_tracks = [], []

        for idx in valid_indices:
            X_sample = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0)
            preds = []

            for name in models_selected:
                model = load_model(name, step)
                with torch.no_grad():
                    y_pred = model(X_sample).squeeze(0).cpu().numpy()
                preds.append(y_pred)

            preds = np.stack(preds, axis=0)  # (7, step, 2)

            # softmax加权融合
            fusion_pred = np.zeros((step, 2), dtype=np.float32)
            for t in range(step):
                fusion_pred[t, :] = np.sum(preds[:, t, :] * weights[:, t][:, None], axis=0)

            # 反归一化
            fusion_pred = inverse_scale_latlon(fusion_pred, scaler)
            input_seq = X_sample.squeeze(0).numpy()[:, :2]
            input_seq = inverse_scale_latlon(input_seq.copy(), scaler)

            fusion_results.append({
                "name": f"Fusion_sample{idx}",
                "color": "#FFFFFF",
                "coords": fusion_pred.tolist(),
            })
            input_tracks.append(input_seq.tolist())

        # ==== 导出 CSV ====
        all_rows = []
        for fr in fusion_results:
            name = fr["name"]
            sid = name.split("sample")[-1]
            for i, (lat, lon) in enumerate(fr["coords"]):
                all_rows.append({"sample_id": sid, "idx": i, "lat": lat, "lon": lon})
        df = pd.DataFrame(all_rows, columns=["sample_id", "idx", "lat", "lon"])
        csv_path = os.path.join(RESULT_DIR, f"fusion_prediction_{step}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        return JSONResponse({
            "ok": True,
            "input_tracks": input_tracks,
            "tracks": fusion_results,
            "csv_file": f"/data/results/fusion_prediction_{step}.csv",
            "skipped": skipped,
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"融合预测失败：{e}"})


# ===== 导出 CSV（供手动调用） =====
@router.post("/export_csv")
async def export_csv(data: dict = Body(...)):
    try:
        step = str(data.get("step", ""))
        tracks = data.get("tracks", []) or []
        if not tracks:
            return JSONResponse({"ok": False, "error": "没有可导出的轨迹，请先运行预测。"})

        rows = []
        for tr in tracks:
            name = tr.get("name", "")
            sample_id = ""
            if "_sample" in name:
                sample_id = name.split("_sample")[-1]
            coords = tr.get("coords", []) or []
            for i, pt in enumerate(coords):
                if len(pt) < 2:
                    continue
                rows.append({
                    "sample_id": sample_id,
                    "point_idx": i,
                    "lat": pt[0],
                    "lon": pt[1],
                })

        df = pd.DataFrame(rows)
        csv_path = os.path.join(RESULT_DIR, f"fusion_export_{step}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return JSONResponse({"ok": True, "file": f"/data/results/fusion_export_{step}.csv"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"CSV 导出失败：{e}"})
