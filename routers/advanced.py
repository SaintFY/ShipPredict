# ===============================
# routers/advanced.py  (前端配套：去PDF + CSV导出当前轨迹 + 模板下载/上传)
# ===============================
import os
import io
import torch
import pickle
import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse

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
UPLOAD_DIR = os.path.join(DATA_DIR, "uploaded")
RESOURCE_DIR = os.path.join(BASE_DIR, "resources")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESOURCE_DIR, exist_ok=True)

# ===== 工具函数 =====
def ensure_exists(path: str, hint_dir: str = None):
    if not os.path.exists(path):
        msg = f"文件不存在：{path}"
        if hint_dir and os.path.isdir(hint_dir):
            msg += f"；目录{hint_dir}包含：{os.listdir(hint_dir)}"
        raise FileNotFoundError(msg)

def inverse_scale_latlon(pred, scaler):
    pred = np.asarray(pred, dtype=np.float32).copy()
    lat_min, lat_max = scaler["lat"]["min"], scaler["lat"]["max"]
    lon_min, lon_max = scaler["lon"]["min"], scaler["lon"]["max"]
    pred[:, 0] = pred[:, 0] * (lat_max - lat_min) + lat_min
    pred[:, 1] = pred[:, 1] * (lon_max - lon_min) + lon_min
    return pred

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

# ===== 颜色定义（仅用于前端渲染，不影响服务端） =====
MODEL_COLORS = {
    "Input": "#00BFFF",
    "LSTM": "#4169E1",
    "BiLSTM": "#00FFFF",
    "GRU": "#00FF7F",
    "BiGRU": "#FF69B4",
    "Transformer": "#FF8C00",
    "LSTM-Attention": "#FFD700",
    "GRU-Attention": "#FF00FF",
    "Fusion": "#FFFFFF",
}

# ===== 单模型预测 =====
@router.post("/predict_single")
async def predict_single(data: dict = Body(...)):
    try:
        step = int(data["step"])
        models_selected = data["models"]
        sample_indices = [int(i) for i in str(data.get("samples", "0")).split(",") if i.strip().isdigit()]

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
                    y_pred = model(X_sample).squeeze(0).cpu().numpy()   # (step,2) 归一化
                y_pred = inverse_scale_latlon(y_pred, scaler)
                results.append({
                    "name": f"{name}_sample{idx}",
                    "color": MODEL_COLORS.get(name, "#ffffff"),
                    "coords": y_pred.tolist()
                })
            # 输入轨迹（反归一化）
            input_seq = X_sample.squeeze(0).numpy()[:, :2]
            input_seq = inverse_scale_latlon(input_seq.copy(), scaler)
            input_tracks.append(input_seq.tolist())

        return JSONResponse({"ok": True, "input_tracks": input_tracks, "tracks": results})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"预测失败：{e}"})


# ===== 融合预测（必须 7 模型全选） =====
@router.post("/predict_fusion")
async def predict_fusion(data: dict = Body(...)):
    try:
        step = int(data.get("step", 15))
        models_selected = data.get("models", [
            "LSTM", "BiLSTM", "GRU", "BiGRU",
            "Transformer", "LSTM-Attention", "GRU-Attention"
        ])
        samples = str(data.get("samples", "0"))
        sample_indices = [int(i) for i in samples.split(",") if i.strip().isdigit()]

        if len(models_selected) != 7:
            return JSONResponse({"ok": False, "error": "融合预测需要选择全部7个模型"})

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
            return JSONResponse({"ok": False, "error": f"输入样本编号 {sample_indices} 均无效；可用范围 0~{n_samples-1}。"})

        # softmax 权重 (7, step)
        weights = np.load(softmax_path)
        weights = np.squeeze(weights)
        if weights.ndim == 4:
            weights = np.squeeze(weights).T
        if weights.shape[0] != 7:
            weights = weights.T
        weights = np.exp(weights) / np.sum(np.exp(weights), axis=0, keepdims=True)

        fusion_results, input_tracks = [], []
        palette = ["#FFFFFF", "#FFDAB9", "#98FB98", "#B0E0E6", "#E6E6FA", "#FFE4E1", "#E0FFFF"]

        for k, idx in enumerate(valid_indices):
            X_sample = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0)

            preds = []
            for name in models_selected:
                model = load_model(name, step)
                with torch.no_grad():
                    y_pred = model(X_sample).squeeze(0).cpu().numpy()  # (step,2)
                preds.append(y_pred)
            preds = np.stack(preds, axis=0)  # (7, step, 2)

            # 时步融合
            fusion_pred = np.zeros((step, 2), dtype=np.float32)
            for t in range(step):
                fusion_pred[t, :] = np.sum(preds[:, t, :] * weights[:, t][:, None], axis=0)

            # 反归一化
            fusion_pred = inverse_scale_latlon(fusion_pred, scaler)
            input_seq = X_sample.squeeze(0).numpy()[:, :2]
            input_seq = inverse_scale_latlon(input_seq.copy(), scaler)

            fusion_results.append({
                "name": f"Fusion_sample{idx}",
                "color": palette[k % len(palette)],
                "coords": fusion_pred.tolist()
            })
            input_tracks.append(input_seq.tolist())

        return JSONResponse({
            "ok": True,
            "input_tracks": input_tracks,
            "tracks": fusion_results,
            "skipped": skipped,
        })
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"融合预测失败：{e}"})


# ===== 导出 CSV（导出“当前前端可视化轨迹”） =====
@router.post("/export_csv")
async def export_csv(data: dict = Body(...)):
    """
    前端会把 window.lastPredictionTracks 传进来：
      data = {
        "step": "15",
        "tracks": [
          {"name":"LSTM_sample0","color":"#4169E1","coords":[[lat,lon], ...]},
          ...
        ]
      }
    我们导出列：sample_id, model, point_idx, lat, lon
    """
    try:
        step = str(data.get("step", ""))
        tracks = data.get("tracks", []) or []
        if not tracks:
            return JSONResponse({"ok": False, "error": "没有可导出的轨迹，请先运行预测/融合。"})

        rows = []
        for tr in tracks:
            name = tr.get("name", "")
            # 解析样本号（尽量兼容 LSTM_sample0 / Fusion_sample3）
            sample_id = ""
            if "_sample" in name:
                sample_id = name.split("_sample")[-1]
            coords = tr.get("coords", []) or []
            for i, pt in enumerate(coords):
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    continue
                lat, lon = float(pt[0]), float(pt[1])
                rows.append({
                    "sample_id": sample_id,
                    "model": name,
                    "point_idx": i,
                    "lat": lat,
                    "lon": lon
                })

        df = pd.DataFrame(rows, columns=["sample_id", "model", "point_idx", "lat", "lon"])
        csv_path = os.path.join(RESULT_DIR, f"prediction_{step}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return JSONResponse({"ok": True, "file": f"/data/results/prediction_{step}.csv", "count": len(rows)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"CSV 导出失败：{e}"})


# ===== 下载“示例文件” =====
@router.get("/download_template")
async def download_template():
    """
    提供一个可下载的示例 .pkl 文件（如果不存在则临时生成）。
    格式：DataFrame，列 ['lat','lon','speed','aspect']，10 行示例数据。
    """
    template_path = os.path.join(RESOURCE_DIR, "sample_template.pkl")
    if not os.path.exists(template_path):
        lats = np.linspace(30.0, 30.09, 10)
        lons = np.linspace(121.0, 121.09, 10)
        df = pd.DataFrame({
            "lat": lats,
            "lon": lons,
            "speed": np.full(10, 12.0),
            "aspect": np.full(10, 90.0),
        })
        with open(template_path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

    return FileResponse(template_path, filename="sample_template.pkl", media_type="application/octet-stream")


# ===== 上传用户数据（简单落盘） =====
@router.post("/upload_data")
async def upload_data(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".pkl"):
            return JSONResponse({"ok": False, "error": "仅支持 .pkl 文件"})
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as f:
            f.write(await file.read())
        return JSONResponse({"ok": True, "message": f"文件已上传：{file.filename}"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"上传失败：{e}"})

# 只贴出新增/修改的部分；你可以把这段加到 routers/advanced.py 里其他路由之后

from fastapi.responses import StreamingResponse
import io
import csv

@router.get("/example_csv")
async def download_example_csv(step: int = 15):
    """
    动态返回一个 CSV 示例，包含两个样本，字段：
    sample_id, idx(点序), lat, lon, speed, aspect
    为了演示简单，轨迹是两条短线段。
    """
    # 造两条小轨迹（你也可以调整基点/步数与 step 对齐）
    base1 = (31.2, 121.5)  # 上海附近
    base2 = (30.2, 122.0)

    rows = [("sample_id","idx","lat","lon","speed","aspect")]
    # 样本0
    for t in range(min(step, 10)):
      lat = base1[0] + 0.01*t
      lon = base1[1] + 0.015*t
      rows.append((0, t, f"{lat:.6f}", f"{lon:.6f}", 12.3, 45.0))
    # 样本1
    for t in range(min(step, 10)):
      lat = base2[0] + 0.008*t
      lon = base2[1] + 0.012*t
      rows.append((1, t, f"{lat:.6f}", f"{lon:.6f}", 10.1, 30.0))

    # 写入内存
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    buf.seek(0)

    filename = f"user_example_step{step}.csv"
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8-sig")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
