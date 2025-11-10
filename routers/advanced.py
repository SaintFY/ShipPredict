# ===============================
# routers/advanced.py （方案A - 可视化修正版）
# ===============================
import os
import torch
import pickle
import numpy as np
import pandas as pd
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

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
BUILTIN_DIR = os.path.join(DATA_DIR, "builtin")  # ✅ 你的 pkl 都在这里
Softmax_DIR = os.path.join(DATA_DIR, "fusion")
WEIGHT_DIR = os.path.join(BASE_DIR, "weights")
RESULT_DIR = os.path.join(DATA_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ===== 工具函数 =====
def ensure_exists(path: str, hint_dir: str = None):
    if not os.path.exists(path):
        msg = f"文件不存在：{path}"
        if hint_dir and os.path.isdir(hint_dir):
            msg += f"；请检查目录 {hint_dir} 下文件: {os.listdir(hint_dir)}"
        raise FileNotFoundError(msg)

def inverse_scale_latlon(pred, scaler):
    """反归一化预测结果 -> 经纬度"""
    pred = np.asarray(pred, dtype=np.float32).copy()
    lat_min, lat_max = scaler["lat"]["min"], scaler["lat"]["max"]
    lon_min, lon_max = scaler["lon"]["min"], scaler["lon"]["max"]
    pred[:, 0] = pred[:, 0] * (lat_max - lat_min) + lat_min
    pred[:, 1] = pred[:, 1] * (lon_max - lon_min) + lon_min
    return pred

# ===== 模型加载函数（动态步长） =====
def load_model(model_name: str, step: int):
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

# ===== 颜色定义（图例） =====
MODEL_COLORS = {
    "Input": "#00BFFF",
    "LSTM": "#4169E1",
    "BiLSTM": "#00FFFF",
    "GRU": "#00FF7F",
    "BiGRU": "#FF69B4",
    "Transformer": "#FF8C00",
    "LSTM-Attention": "#FFD700",
    "GRU-Attention": "#FF00FF",
    "Fusion": "#FFFFFF"
}

# ===== 单模型预测 =====
@router.post("/predict_single")
async def predict_single(data: dict = Body(...)):
    try:
        step = int(data["step"])
        models_selected = data["models"]
        sample_indices = [int(i) for i in str(data.get("samples", "0")).split(",")]

        scaler_path = os.path.join(BUILTIN_DIR, f"global_scaling_params_{step}_all.pkl")
        X_path = os.path.join(BUILTIN_DIR, f"X_test_{step}_all.pkl")
        ensure_exists(scaler_path, BUILTIN_DIR)
        ensure_exists(X_path, BUILTIN_DIR)

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_test = pickle.load(open(X_path, "rb"))

        results = []
        input_tracks = []
        for idx in sample_indices:
            X_sample = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0)

            # 每个模型预测
            for name in models_selected:
                model = load_model(name, step)
                with torch.no_grad():
                    y_pred = model(X_sample).squeeze(0).cpu().numpy()
                y_pred = inverse_scale_latlon(y_pred, scaler)

                results.append({
                    "name": f"{name}_sample{idx}",
                    "color": MODEL_COLORS[name],
                    "coords": y_pred.tolist()
                })

            # 保存输入轨迹
            input_seq = X_sample.squeeze(0).numpy()[:, :2]
            input_seq = inverse_scale_latlon(input_seq.copy(), scaler)
            input_tracks.append(input_seq.tolist())

        return JSONResponse({
            "ok": True,
            "input_tracks": input_tracks,
            "tracks": results
        })

    except Exception as e:
        return JSONResponse({"ok": False, "error": f"预测失败：{e}"})



# ===== 融合预测 =====
@router.post("/predict_fusion")
async def predict_fusion(data: dict = Body(...)):
    try:
        step = int(data.get("step", 15))
        models_selected = data.get("models", [
            "LSTM", "BiLSTM", "GRU", "BiGRU",
            "Transformer", "LSTM-Attention", "GRU-Attention"
        ])

        if len(models_selected) != 7:
            return JSONResponse({"ok": False, "error": "融合预测需要选择全部7个模型"})

        scaler_path = os.path.join(BUILTIN_DIR, f"global_scaling_params_{step}_all.pkl")
        X_path = os.path.join(BUILTIN_DIR, f"X_test_{step}_all.pkl")
        softmax_path = os.path.join(Softmax_DIR, f"softmax_weights_{step}_all.npy")
        ensure_exists(scaler_path, BUILTIN_DIR)
        ensure_exists(X_path, BUILTIN_DIR)
        ensure_exists(softmax_path, BUILTIN_DIR)

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        X_test = pickle.load(open(X_path, "rb"))
        X_sample = torch.tensor(X_test[0], dtype=torch.float32).unsqueeze(0)

        # ===== 获取7个模型预测 =====
        preds = []
        for name in models_selected:
            model = load_model(name, step)
            with torch.no_grad():
                y_pred = model(X_sample).squeeze(0).cpu().numpy()
            preds.append(y_pred)
        preds = np.stack(preds, axis=0)  # (7, step, 2)

        # ===== 加载 softmax 动态权重 =====
        weights = np.load(softmax_path)
        weights = np.squeeze(weights)

        # 自动调整维度
        if weights.ndim == 1:
            weights = np.expand_dims(weights, axis=1)  # (7,1)
        elif weights.ndim == 4:  # 常见为 (step,1,1,7)
            weights = np.squeeze(weights).T  # 变成 (7,step)

        # 确保方向一致
        if weights.shape[0] != 7:
            weights = weights.T

        # ===== Softmax归一化 + 时步融合 =====
        weights = np.exp(weights) / np.sum(np.exp(weights), axis=0, keepdims=True)  # (7, step)
        fusion_pred = np.zeros((step, 2))
        for t in range(step):
            fusion_pred[t, :] = np.sum(preds[:, t, :] * weights[:, t][:, None], axis=0)

        fusion_pred = inverse_scale_latlon(fusion_pred, scaler)

        # ===== 输入轨迹 =====
        input_seq = X_sample.squeeze(0).numpy()[:, :2]
        input_seq = inverse_scale_latlon(input_seq.copy(), scaler)

        return JSONResponse({
            "ok": True,
            "input_track": input_seq.tolist(),
            "tracks": [{
                "name": "Fusion",
                "color": MODEL_COLORS["Fusion"],
                "coords": fusion_pred.tolist()
            }]
        })

    except Exception as e:
        return JSONResponse({"ok": False, "error": f"融合预测失败：{e}"})



# ===== 导出 CSV =====
@router.post("/export_csv")
async def export_csv(data: dict = Body(...)):
    try:
        df = pd.DataFrame(data["data"])
        csv_path = os.path.join(RESULT_DIR, f"prediction_{data['step']}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return JSONResponse({"ok": True, "file": f"/data/results/prediction_{data['step']}.csv"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"CSV 导出失败：{e}"})


# ===== 导出 PDF =====
@router.post("/export_pdf")
async def export_pdf(data: dict = Body(...)):
    try:
        pdf_path = os.path.join(RESULT_DIR, f"report_{data['step']}.pdf")
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("<b>ShipTrack 预测报告</b>", styles["Title"]),
            Spacer(1, 12),
            Paragraph(f"预测步长：{data['step']} 步", styles["Normal"]),
            Paragraph(f"模型：{', '.join(data['models'])}", styles["Normal"]),
        ]
        doc.build(story)
        return JSONResponse({"ok": True, "file": f"/data/results/report_{data['step']}.pdf"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"PDF 导出失败：{e}"})
