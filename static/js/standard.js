// ================================
// standard.js (标准模式前端逻辑)
// ================================
window.addEventListener("DOMContentLoaded", () => {
  console.log("✅ 标准模式已加载");

  // ===== 元素获取 =====
  const fusionBtn = document.getElementById("fusion-btn");
  const exportBtn = document.getElementById("export-csv");
  const stepSelect = document.getElementById("step");
  const sampleInput = document.getElementById("sample-index");
  const dataSource = document.getElementById("data-source");
  const downloadExample = document.getElementById("download-example");
  const uploadLabel = document.getElementById("upload-data-label");
  const userFile = document.getElementById("user-file");

  // ===== 地图初始化 =====
  const map = L.map("map", { zoomControl: true }).setView([31.2, 121.5], 7);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  const layers = {};

  // ===== 坐标清洗函数 =====
  function isFiniteNumber(x) {
    return typeof x === "number" && isFinite(x);
  }

  function cleanCoords(arr) {
    const cleaned = (arr || []).filter(
      (pt) =>
        Array.isArray(pt) &&
        pt.length >= 2 &&
        isFiniteNumber(pt[0]) &&
        isFiniteNumber(pt[1])
    );
    return cleaned;
  }

  function safeFitBounds(coords) {
    const c = cleanCoords(coords);
    if (c.length > 0) {
      try {
        map.fitBounds(c, { padding: [20, 20] });
      } catch (e) {
        console.warn("⚠️ fitBounds 失败：", e);
      }
    }
  }

  // ===== 动态绘制轨迹 =====
  async function animateTrack(coords, color, layerName, delay = 80) {
    const seq = cleanCoords(coords);
    if (seq.length === 0) return;
    let drawn = [];
    for (let i = 0; i < seq.length; i++) {
      drawn.push(seq[i]);
      if (layers[layerName]) map.removeLayer(layers[layerName]);
      layers[layerName] = L.polyline(drawn, { color, weight: 4 }).addTo(map);
      await new Promise((res) => setTimeout(res, delay));
    }
    const end = seq[seq.length - 1];
    L.circleMarker(end, {
      color,
      radius: 5,
      fillColor: color,
      fillOpacity: 0.9,
    }).addTo(map);
  }

  // ===== 数据来源控制（与高级模式同步） =====
  function toggleUploadState() {
    if (dataSource.value === "user") {
      downloadExample.classList.remove("disabled");
      uploadLabel.classList.remove("disabled");
    } else {
      downloadExample.classList.add("disabled");
      uploadLabel.classList.add("disabled");
    }
  }
  toggleUploadState();
  dataSource.addEventListener("change", toggleUploadState);

  // ===== 下载示例文件 =====
  downloadExample.onclick = () => {
    if (downloadExample.classList.contains("disabled")) return;
    const link = document.createElement("a");
    link.href = "/data/example_user_upload.csv"; // 你可放在 data 根目录
    link.download = "example_user_upload.csv";
    link.click();
  };

  // ===== 上传文件（仅预览名） =====
  userFile.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      uploadLabel.textContent = `已选择：${e.target.files[0].name}`;
    } else {
      uploadLabel.textContent = "⬆️ 上传数据";
    }
  });

  // ===== 融合预测 =====
  fusionBtn.onclick = async () => {
    const step = stepSelect.value;
    const samples = sampleInput.value || "0";

    try {
      fusionBtn.disabled = true;
      fusionBtn.textContent = "预测中...";

      const res = await fetch("/standard/predict_fusion", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ step, samples }),
      });

      const data = await res.json();
      fusionBtn.disabled = false;
      fusionBtn.textContent = "融合预测";

      if (!data.ok) {
        alert(`❌ 融合预测失败：${data.error}`);
        return;
      }

      console.log("✅ 收到融合结果：", data);

      // 清空旧图层
      Object.values(layers).forEach((l) => map.removeLayer(l));

      // 绘制输入轨迹
      (data.input_tracks || []).forEach((t) => {
        const coords = cleanCoords(t);
        if (coords.length === 0) return;
        L.polyline(coords, { color: "#00BFFF", weight: 3, dashArray: "5,5" }).addTo(map);
        L.circleMarker(coords[0], { color: "blue", radius: 5 }).addTo(map);
        L.circleMarker(coords[coords.length - 1], { color: "red", radius: 5 }).addTo(map);
      });

      // 绘制融合结果轨迹
      for (const tr of data.tracks || []) {
        if (!tr || !Array.isArray(tr.coords)) continue;
        await animateTrack(tr.coords, tr.color || "#FFFFFF", tr.name || "fusion", 60);
      }

      // 自适应视图
      const allCoords = (data.tracks || []).flatMap((t) =>
        Array.isArray(t.coords) ? t.coords : []
      );
      safeFitBounds(allCoords);

      if (data.csv_file) {
        console.log("✅ 融合结果CSV文件：", data.csv_file);
      }

    } catch (err) {
      console.error("❌ 预测异常：", err);
      alert(`预测失败：${err}`);
      fusionBtn.disabled = false;
      fusionBtn.textContent = "融合预测";
    }
  };

  // ===== 导出 CSV =====
  exportBtn.onclick = async () => {
    const step = stepSelect.value;
    try {
      const res = await fetch("/standard/export_csv", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ step }),
      });
      const data = await res.json();
      if (!data.ok) {
        alert(`CSV 导出失败: ${data.error}`);
        return;
      }
      const link = document.createElement("a");
      link.href = data.file;
      link.download = `fusion_export_${step}.csv`;
      link.click();
    } catch (err) {
      console.error("❌ CSV 导出异常:", err);
      alert(`导出失败: ${err}`);
    }
  };

  console.log("✅ 标准模式系统已就绪");
});
