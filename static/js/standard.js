window.onload = () => {
  const map = L.map('map').setView([31.2, 121.5], 8);
  L.tileLayer('https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);

  document.getElementById('runPredict').addEventListener('click', () => {
    const step = document.getElementById('stepSelect').value;
    fetch('/standard/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ step })
    })
    .then(res => res.json())
    .then(result => {
      map.eachLayer(l => { if (l instanceof L.Polyline) map.removeLayer(l); });
      result.tracks.forEach(track => {
        L.polyline(track.coords, { color: track.color, weight: 3 }).addTo(map);
      });
    });
  });

  document.getElementById('exportCSV').addEventListener('click', () => {
    fetch('/standard/export/csv').then(() => alert('✅ 结果已导出'));
  });
};
