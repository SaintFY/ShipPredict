// ================================
// advanced.js (带动态动画播放版本)
// ================================

window.addEventListener("DOMContentLoaded", () => {
    console.log("✅ 页面加载完成，准备绑定事件...");

    const runBtn = document.getElementById("run-btn");
    const fusionBtn = document.getElementById("fusion-btn");
    const exportCSVBtn = document.getElementById("export-csv");
    const exportPDFBtn = document.getElementById("export-pdf");
    const stepSelect = document.getElementById("step");
    const sampleInput = document.getElementById("sample-index");
    const modelCheckboxes = document.querySelectorAll(".model-checkbox");

    // 初始化地图
    const map = L.map("map", { zoomControl: true }).setView([31.2, 121.5], 7);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { maxZoom: 18 }).addTo(map);
    const layers = {};

    // =======================
    // 动态绘制函数
    // =======================
    async function animateTrack(coords, color, layerName, delay = 80) {
        if (!coords || coords.length === 0) return;
        let drawn = [];
        for (let i = 0; i < coords.length; i++) {
            drawn.push(coords[i]);
            if (layers[layerName]) map.removeLayer(layers[layerName]);
            layers[layerName] = L.polyline(drawn, { color, weight: 4 }).addTo(map);
            await new Promise(res => setTimeout(res, delay)); // 延时
        }
        // 终点标记
        const end = coords[coords.length - 1];
        L.circleMarker(end, { color, radius: 5, fillColor: color, fillOpacity: 0.9 }).addTo(map);
    }

    // =======================
    // 运行预测
    // =======================
    runBtn.onclick = async () => {
        const models = Array.from(modelCheckboxes)
            .filter(cb => cb.checked)
            .map(cb => cb.value);

        if (models.length === 0) {
            alert("⚠️ 请至少选择一个模型！");
            return;
        }

        const step = stepSelect.value;
        const samples = sampleInput.value || "0";
        const payload = { models, step, samples };

        try {
            runBtn.disabled = true;
            runBtn.textContent = "预测中...";

            const res = await fetch("/advanced/predict_single", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            runBtn.disabled = false;
            runBtn.textContent = "运行预测";

            if (!data.ok) {
                alert(`❌ 预测失败: ${data.error}`);
                return;
            }

            // 清空旧轨迹
            Object.values(layers).forEach(l => map.removeLayer(l));

            // 绘制输入轨迹（虚线 + 起终点）
            data.input_tracks.forEach(track => {
                L.polyline(track, { color: "#00BFFF", weight: 3, dashArray: "5,5" }).addTo(map);
                L.circleMarker(track[0], { color: "blue", radius: 5 }).addTo(map);
                L.circleMarker(track[track.length - 1], { color: "red", radius: 5 }).addTo(map);
            });

            // 动态绘制预测轨迹
            for (const tr of data.tracks) {
                animateTrack(tr.coords, tr.color, tr.name, 80); // 每 80ms 绘制一个点
            }

            map.fitBounds(data.tracks[0].coords);

        } catch (err) {
            console.error("❌ 运行预测异常:", err);
            alert(`预测失败: ${err}`);
            runBtn.disabled = false;
            runBtn.textContent = "运行预测";
        }
    };

    // =======================
    // 融合预测（带动画）
    // =======================
    fusionBtn.onclick = async () => {
        const models = Array.from(modelCheckboxes)
            .filter(cb => cb.checked)
            .map(cb => cb.value);

        if (models.length !== 7) {
            alert("⚠️ 融合预测需要选择全部 7 个模型！");
            return;
        }

        const step = stepSelect.value;

        try {
            fusionBtn.disabled = true;
            fusionBtn.textContent = "融合中...";

            const res = await fetch("/advanced/predict_fusion", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ models, step })
            });

            const data = await res.json();
            fusionBtn.disabled = false;
            fusionBtn.textContent = "融合预测";

            if (!data.ok) {
                alert(`❌ 融合预测失败: ${data.error}`);
                return;
            }

            // 清除旧轨迹
            Object.values(layers).forEach(l => map.removeLayer(l));

            // 输入轨迹
            const inputTrack = data.input_track;
            L.polyline(inputTrack, { color: "#00BFFF", weight: 3, dashArray: "5,5" }).addTo(map);
            L.circleMarker(inputTrack[0], { color: "blue", radius: 5 }).addTo(map);
            L.circleMarker(inputTrack[inputTrack.length - 1], { color: "red", radius: 5 }).addTo(map);

            // 动态播放融合结果
            const fusionTrack = data.tracks[0].coords;
            await animateTrack(fusionTrack, "#FFFFFF", "fusion", 60);

            map.fitBounds(fusionTrack);

        } catch (err) {
            console.error("❌ 融合预测异常:", err);
            alert(`融合预测失败: ${err}`);
            fusionBtn.disabled = false;
            fusionBtn.textContent = "融合预测";
        }
    };

    // =======================
    // 导出 CSV
    // =======================
    exportCSVBtn.onclick = async () => {
        const step = stepSelect.value;
        try {
            const res = await fetch("/advanced/export_csv", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ step, data: [] })
            });
            const data = await res.json();
            if (!data.ok) return alert(`CSV 导出失败: ${data.error}`);
            const link = document.createElement("a");
            link.href = data.file;
            link.download = `prediction_${step}.csv`;
            link.click();
        } catch (err) {
            alert(`导出失败: ${err}`);
        }
    };

    // =======================
    // 导出 PDF
    // =======================
    exportPDFBtn.onclick = async () => {
        const step = stepSelect.value;
        const models = Array.from(modelCheckboxes)
            .filter(cb => cb.checked)
            .map(cb => cb.value);
        try {
            const res = await fetch("/advanced/export_pdf", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ step, models })
            });
            const data = await res.json();
            if (!data.ok) return alert(`PDF 导出失败: ${data.error}`);
            const link = document.createElement("a");
            link.href = data.file;
            link.download = `report_${step}.pdf`;
            link.click();
        } catch (err) {
            alert(`导出失败: ${err}`);
        }
    };

    console.log("✅ 按钮绑定完成，系统就绪。");
});
