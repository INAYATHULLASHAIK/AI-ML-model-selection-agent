let currentResults = null;

const fileInput = document.getElementById("file");
const message = document.getElementById("message");
const datasetSection = document.getElementById("datasetSection");
const resultsSection = document.getElementById("resultsSection");

function showMessage(text, error=false) {
  message.innerHTML = `<div class="${error ? "error" : "success"}">${text}</div>`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"
  }[c]));
}

document.getElementById("analyzeBtn").onclick = async () => {
  const file = fileInput.files[0];
  if (!file) return showMessage("Please select a CSV file.", true);

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/analyze", {method:"POST", body:form});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    showMessage("Dataset analyzed successfully.");
    datasetSection.classList.remove("hidden");
    resultsSection.classList.add("hidden");

    document.getElementById("datasetStats").innerHTML = `
      <div class="stat"><b>Rows</b><br>${data.rows}</div>
      <div class="stat"><b>Columns</b><br>${data.columns}</div>
      <div class="stat"><b>Duplicates</b><br>${data.duplicates}</div>
      <div class="stat"><b>Missing Cells</b><br>${Object.values(data.missing_values).reduce((a,b)=>a+b,0)}</div>`;

    const target = document.getElementById("target");
    target.innerHTML = data.column_names.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");

    const preview = document.getElementById("preview");
    const rows = data.preview;
    if (rows.length) {
      const cols = Object.keys(rows[0]);
      preview.innerHTML = `<thead><tr>${cols.map(c=>`<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>
        <tbody>${rows.map(r=>`<tr>${cols.map(c=>`<td>${escapeHtml(r[c])}</td>`).join("")}</tr>`).join("")}</tbody>`;
    }
  } catch (e) {
    showMessage(e.message, true);
  }
};

document.getElementById("trainBtn").onclick = async () => {
  const file = fileInput.files[0];
  const target = document.getElementById("target").value;
  if (!file) return showMessage("Please upload the CSV file.", true);

  const form = new FormData();
  form.append("file", file);
  form.append("target", target);

  showMessage("Training and comparing models. Please wait...");
  try {
    const res = await fetch("/train", {method:"POST", body:form});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    currentResults = data;
    renderResults(data);
  } catch (e) {
    showMessage(e.message, true);
  }
};

function renderResults(data) {
  resultsSection.classList.remove("hidden");

  document.getElementById("recommendation").innerHTML = `
    <div class="success">
      <b>Problem Type:</b> ${escapeHtml(data.problem_type)}<br>
      <b>Target:</b> ${escapeHtml(data.target_column)}<br>
      <b>Recommended Model:</b> ${escapeHtml(data.best_model)}<br>
      <b>Score:</b> ${data.best_score}<br>
      <b>Reason:</b> ${escapeHtml(data.reason)}
    </div>`;

  const results = data.results;
  if (!results.length) return;

  const cols = Object.keys(results[0]);
  document.getElementById("resultsTable").innerHTML =
    `<thead><tr>${cols.map(c=>`<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>
     <tbody>${results.map(r=>`<tr>${cols.map(c=>`<td>${escapeHtml(r[c])}</td>`).join("")}</tr>`).join("")}</tbody>`;

  Plotly.newPlot("chart1", [{
    x:data.chart_labels, y:data.chart_primary, type:"bar"
  }], {
    title:data.problem_type === "Classification" ? "Model vs F1 Score" : "Model vs R²",
    xaxis:{title:"Model"}, yaxis:{title:data.problem_type === "Classification" ? "F1 Score" : "R²"}
  }, {responsive:true});

  Plotly.newPlot("chart2", [{
    x:data.chart_labels, y:data.chart_secondary, type:"bar"
  }], {
    title:data.problem_type === "Classification" ? "Model vs Accuracy" : "Model vs RMSE",
    xaxis:{title:"Model"}, yaxis:{title:data.problem_type === "Classification" ? "Accuracy" : "RMSE"}
  }, {responsive:true});

  if (data.problem_type === "Classification") {
    Plotly.newPlot("chart3", [{
      z:data.confusion_matrix, x:data.confusion_labels, y:data.confusion_labels,
      type:"heatmap"
    }], {title:"Confusion Matrix"}, {responsive:true});
  } else {
    Plotly.newPlot("chart3", [{
      x:data.actual, y:data.predicted, mode:"markers", type:"scatter"
    }], {
      title:"Actual vs Predicted",
      xaxis:{title:"Actual"}, yaxis:{title:"Predicted"}
    }, {responsive:true});
  }
}

document.getElementById("downloadBtn").onclick = async () => {
  if (!currentResults) return;
  const res = await fetch("/download", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(currentResults)
  });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "model_selection_report.txt";
  a.click();
  URL.revokeObjectURL(url);
};
