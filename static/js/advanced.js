// ================================
// advanced.js（带数据源高亮 + 示例CSV下载 + 上传按钮着色）
// ================================
window.addEventListener("DOMContentLoaded", () => {
  const runBtn = document.getElementById("run-btn");
  const fusionBtn = document.getElementById("fusion-btn");
  const exportCSVBtn = document.getElementById("export-csv");
  const stepSelect = document.getElementById("step");
  const sampleInput = document.getElementById("sample-index");
  const modelCheckboxes = document.querySelectorAll(".model-checkbox");

  const dataSource = document.getElementById("data-source");
  const downloadBtn = document.getElementById("download-example");
  const uploadLabel = document.getElementById("upload-data-label");
  const userFile = document.getElementById("user-file");

  // ========== 地图 ==========
  const map = L.map("map", { zoomControl: true }).setView([31.2, 121.5], 7);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 18,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  const layers = {};
  function cleanCoords(arr) {
    return (arr || []).filter(
      (pt) => Array.isArray(pt) && pt.length >= 2 && isFinite(pt[0]) && isFinite(pt[1])
    );
  }
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
    L.circleMarker(seq[seq.length - 1], { color, radius: 5, fillColor: color, fillOpacity: 0.9 }).addTo(map);
  }

  // ========== 数据源联动着色 ==========
  function refreshSourceUI() {
    const isUser = dataSource.value === "user";
    // 下拉框高亮
    dataSource.classList.toggle("user-accent", isUser);

    // 下载/上传按钮可用 + 着色
    if (isUser) {
      downloadBtn.classList.remove("gray");
      downloadBtn.classList.add("active");
      uploadLabel.classList.remove("gray");
      uploadLabel.classList.add("active");
      downloadBtn.disabled = false;
      userFile.disabled = false;
    } else {
      downloadBtn.classList.add("gray");
      downloadBtn.classList.remove("active");
      uploadLabel.classList.add("gray");
      uploadLabel.classList.remove("active");
      downloadBtn.disabled = true;
      userFile.disabled = true;
    }
  }
  dataSource.addEventListener("change", refreshSourceUI);
  refreshSourceUI();

  // ========== 下载示例 CSV ==========
  downloadBtn.addEventListener("click", async () => {
    if (downloadBtn.disabled) return;
    const step = stepSelect.value;
    // 直接下载
    window.location.href = `/advanced/example_csv?step=${encodeURIComponent(step)}`;
  });

  // ========== 读取用户上传（这里只演示读取内容，不改变你的现有后端逻辑）==========
  userFile.addEventListener("change", async (e) => {
    if (!e.target.files || !e.target.files[0]) return;
    // 这里可选：本地预检查/提示
    alert(`已选择文件：${e.target.files[0].name}，上传并解析逻辑按你的后端需求接上即可。`);
  });

  // ========== 单模型预测 ==========
  runBtn.onclick = async () => {
    const models = Array.from(modelCheckboxes).filter(cb => cb.checked).map(cb => cb.value);
    if (models.length === 0) { alert("⚠️ 请至少选择一个模型！"); return; }

    const step = stepSelect.value;
    const samples = sampleInput.value || "0";
    try {
      runBtn.disabled = true; runBtn.textContent = "预测中...";
      const res = await fetch("/advanced/predict_single", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ models, step, samples })
      });
      const data = await res.json();
      runBtn.disabled = false; runBtn.textContent = "运行预测";
      if (!data.ok) { alert(`❌ 预测失败：${data.error}`); return; }

      Object.values(layers).forEach(l => map.removeLayer(l));
      (data.input_tracks || []).forEach(track => {
        const t = cleanCoords(track);
        if (!t.length) return;
        L.polyline(t, { color: "#00BFFF", weight: 3, dashArray: "5,5" }).addTo(map);
        L.circleMarker(t[0], { color: "blue", radius: 5 }).addTo(map);
        L.circleMarker(t[t.length-1], { color: "red", radius: 5 }).addTo(map);
      });
      for (const tr of data.tracks || []) {
        if (!tr || !Array.isArray(tr.coords)) continue;
        await animateTrack(tr.coords, tr.color || "#fff", tr.name || "track", 80);
      }
    } catch (e) {
      runBtn.disabled = false; runBtn.textContent = "运行预测";
      alert(`预测失败：${e}`);
    }
  };

  // ========== 融合预测 ==========
  fusionBtn.onclick = async () => {
    const models = Array.from(modelCheckboxes).filter(cb=>cb.checked).map(cb=>cb.value);
    if (models.length !== 7) { alert("⚠️ 融合预测需要选择全部 7 个模型！"); return; }
    const step = stepSelect.value;
    const samples = sampleInput.value || "0";
    try {
      fusionBtn.disabled = true; fusionBtn.textContent = "融合中...";
      const res = await fetch("/advanced/predict_fusion", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ models, step, samples })
      });
      const data = await res.json();
      fusionBtn.disabled = false; fusionBtn.textContent = "融合预测";
      if (!data.ok) { alert(`❌ 融合预测失败：${data.error}`); return; }

      Object.values(layers).forEach(l=>map.removeLayer(l));
      (data.input_tracks || []).forEach(tk=>{
        const t = cleanCoords(tk);
        if (!t.length) return;
        L.polyline(t, { color:"#00BFFF", weight:3, dashArray:"5,5" }).addTo(map);
        L.circleMarker(t[0], { color:"blue", radius:5 }).addTo(map);
        L.circleMarker(t[t.length-1], { color:"red", radius:5 }).addTo(map);
      });
      for (const tr of data.tracks || []) {
        if (!tr || !Array.isArray(tr.coords)) continue;
        await animateTrack(tr.coords, tr.color || "#fff", tr.name || "fusion", 60);
      }
    } catch (e) {
      fusionBtn.disabled = false; fusionBtn.textContent = "融合预测";
      alert(`融合预测失败：${e}`);
    }
  };

  // ========== 导出 CSV ==========
  exportCSVBtn.onclick = async () => {
    const step = stepSelect.value;
    // 这里沿用你现有的导出逻辑（后端会从最近一次预测缓存导出）
    try{
      const res = await fetch("/advanced/export_csv", {
        method:"POST", headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({ step })
      });
      const data = await res.json();
      if (!data.ok){ alert(`CSV 导出失败：${data.error}`); return; }
      const a = document.createElement("a");
      a.href = data.file; a.download = `prediction_${step}.csv`; a.click();
    }catch(e){ alert(`导出失败：${e}`); }
  };
});
