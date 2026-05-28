const API_BASE_URL = 'https://risklens-erez.onrender.com';

const form = document.getElementById('predict-form');
const resultDiv = document.getElementById('result');
const batchFile = document.getElementById('batch-file');
const batchResult = document.getElementById('batch-result');
const backendStatus = document.getElementById('backend-status');
const backendStatusDetail = document.getElementById('backend-status-detail');
const healthValue = document.getElementById('health-value');
const healthDetail = document.getElementById('health-detail');
const modelName = document.getElementById('model-name');
const modelDetail = document.getElementById('model-detail');
const thresholdValue = document.getElementById('threshold-value');
const batchRows = document.getElementById('batch-rows');
const batchPositiveRate = document.getElementById('batch-positive-rate');
const batchTopRisk = document.getElementById('batch-top-risk');
const fillExample = document.getElementById('fill-example');
const runBatch = document.getElementById('run-batch');
const downloadSample = document.getElementById('download-sample');
const refreshHealth = document.getElementById('refresh-health');
const refreshModel = document.getElementById('refresh-model');
const tabs = Array.from(document.querySelectorAll('.tab'));
const panels = Array.from(document.querySelectorAll('.tab-panel'));
const fileDropLabel = document.querySelector('.file-drop span');

const numericFields = [
  'age',
  'driving_license',
  'region_code',
  'previously_insured',
  'annual_premium',
  'policy_sales_channel',
  'vintage'
];

const sampleCsv = `id,gender,age,driving_license,region_code,previously_insured,vehicle_age,vehicle_damage,annual_premium,policy_sales_channel,vintage
CUST_001,Male,35,1,28,0,1-2 Year,No,40000,26,200
CUST_002,Female,48,1,8,1,< 1 Year,Yes,26000,124,90
CUST_003,Male,22,1,19,0,> 2 Years,No,18000,152,35`;

function setActiveTab(tabName) {
  tabs.forEach((tab) => tab.classList.toggle('active', tab.dataset.tab === tabName));
  panels.forEach((panel) => panel.classList.toggle('active', panel.id === `${tabName}-panel`));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function parseCsvRows(csvText) {
  const lines = csvText.trim().split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) {
    throw new Error('CSV needs a header row and at least one data row');
  }

  const headers = lines[0].split(',').map((header) => header.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(',');
    return headers.reduce((row, header, index) => {
      row[header] = (values[index] ?? '').trim();
      return row;
    }, {});
  });
}

function getFormPayload() {
  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  numericFields.forEach((field) => {
    payload[field] = Number(payload[field]);
  });

  return payload;
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};

  if (!response.ok) {
    throw new Error(data.error || data.detail || response.statusText);
  }

  return data;
}

function renderSingleResult(data) {
  const probability = Number(data.probability);
  const percent = (probability * 100).toFixed(2);
  const classLabel = data.prediction === 1 ? 'High risk' : 'Low risk';
  const classTone = data.prediction === 1 ? 'warn' : 'good';

  thresholdValue.textContent = Number(data.threshold).toFixed(4);
  resultDiv.innerHTML = `
    <div class="result-header">
      <span class="badge ${classTone}">${classLabel}</span>
      <strong>${escapeHtml(data.id)}</strong>
    </div>
    <div class="result-grid">
      <div class="metric"><span>Prediction</span><strong>${data.prediction}</strong></div>
      <div class="metric"><span>Probability</span><strong>${percent}%</strong></div>
      <div class="metric"><span>Threshold</span><strong>${Number(data.threshold).toFixed(4)}</strong></div>
      <div class="metric"><span>Model version</span><strong>${escapeHtml(data.model_version)}</strong></div>
    </div>
    <div class="confidence-bar" aria-label="Prediction confidence"><div style="width:${Math.min(Math.max(probability * 100, 4), 100)}%"></div></div>
  `;
  resultDiv.classList.remove('hidden');
}

function renderBatchResult(payload) {
  const predictions = payload.predictions || [];
  const positiveCount = predictions.filter((row) => row.prediction === 1).length;
  const topRisk = predictions.length
    ? predictions.reduce((best, row) => (row.probability > best.probability ? row : best), predictions[0])
    : null;

  batchRows.textContent = String(predictions.length);
  batchPositiveRate.textContent = predictions.length ? `${((positiveCount / predictions.length) * 100).toFixed(1)}%` : '0%';
  batchTopRisk.textContent = topRisk ? `${topRisk.id} (${(topRisk.probability * 100).toFixed(1)}%)` : '-';

  const rows = predictions.map((row) => {
    const badgeClass = row.prediction === 1 ? 'warn' : 'good';
    const badgeLabel = row.prediction === 1 ? 'High risk' : 'Low risk';
    return `
      <tr>
        <td>${escapeHtml(row.id)}</td>
        <td><span class="badge ${badgeClass}">${badgeLabel}</span></td>
        <td>${(row.probability * 100).toFixed(2)}%</td>
        <td>${Number(row.threshold).toFixed(4)}</td>
      </tr>
    `;
  }).join('');

  batchResult.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Customer</th>
          <th>Class</th>
          <th>Probability</th>
          <th>Threshold</th>
        </tr>
      </thead>
      <tbody>${rows || '<tr><td colspan="4">No rows returned</td></tr>'}</tbody>
    </table>
  `;
  batchResult.classList.remove('hidden');
}

async function refreshHealthStatus() {
  backendStatus.textContent = 'Checking...';
  backendStatusDetail.textContent = 'Pinging /health';

  try {
    const data = await fetchJson('/health');
    const badge = data.model_loaded ? 'good' : 'warn';
    backendStatus.innerHTML = `<span class="badge ${badge}">${data.status}</span>`;
    backendStatusDetail.textContent = `${data.model_version} · ${new Date(data.timestamp).toLocaleString()}`;
    healthValue.innerHTML = `<span class="badge ${badge}">${data.status}</span>`;
    healthDetail.textContent = data.model_loaded ? 'Model loaded and ready' : 'Model not loaded';
  } catch (error) {
    backendStatus.innerHTML = '<span class="badge bad">offline</span>';
    backendStatusDetail.textContent = error.message;
    healthValue.innerHTML = '<span class="badge bad">error</span>';
    healthDetail.textContent = error.message;
  }
}

async function refreshModelInfo() {
  modelName.textContent = 'Loading...';
  modelDetail.textContent = 'Fetching /model-info';

  try {
    const data = await fetchJson('/model-info');
    modelName.textContent = data.model_name;
    modelDetail.textContent = `${data.model_version} · threshold ${Number(data.decision_threshold).toFixed(4)}`;
    thresholdValue.textContent = Number(data.decision_threshold).toFixed(4);
  } catch (error) {
    modelName.textContent = 'Unavailable';
    modelDetail.textContent = error.message;
  }
}

async function submitSinglePrediction(event) {
  event.preventDefault();
  resultDiv.classList.add('hidden');
  const payload = getFormPayload();

  try {
    const data = await fetchJson('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    renderSingleResult(data);
  } catch (error) {
    resultDiv.innerHTML = `<div class="badge bad">Prediction failed</div><p>${escapeHtml(error.message)}</p>`;
    resultDiv.classList.remove('hidden');
  }
}

async function submitBatchPrediction() {
  if (!batchFile.files || !batchFile.files.length) {
    batchResult.innerHTML = '<div class="badge bad">Choose a CSV first</div>';
    batchResult.classList.remove('hidden');
    return;
  }

  const formData = new FormData();
  formData.append('file', batchFile.files[0]);

  batchResult.classList.remove('hidden');
  batchResult.innerHTML = '<p>Running batch prediction...</p>';

  try {
    const data = await fetchJson('/predict-batch', {
      method: 'POST',
      body: formData
    });

    renderBatchResult(data);
  } catch (error) {
    batchResult.innerHTML = `<div class="badge bad">Batch prediction failed</div><p>${escapeHtml(error.message)}</p>`;
    batchRows.textContent = '0';
    batchPositiveRate.textContent = '0%';
    batchTopRisk.textContent = '-';
  }
}

function downloadSampleCsv() {
  const blob = new Blob([sampleCsv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'risklens-sample-batch.csv';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function fillExampleForm() {
  form.reset();
  form.elements.id.value = 'CUST_001';
  form.elements.gender.value = 'Male';
  form.elements.age.value = 35;
  form.elements.driving_license.value = '1';
  form.elements.region_code.value = 28;
  form.elements.previously_insured.value = '0';
  form.elements.vehicle_age.value = '1-2 Year';
  form.elements.vehicle_damage.value = 'No';
  form.elements.annual_premium.value = 40000;
  form.elements.policy_sales_channel.value = 26;
  form.elements.vintage.value = 200;
}

tabs.forEach((tab) => {
  tab.addEventListener('click', () => setActiveTab(tab.dataset.tab));
});

form.addEventListener('submit', submitSinglePrediction);
fillExample.addEventListener('click', fillExampleForm);
runBatch.addEventListener('click', submitBatchPrediction);
downloadSample.addEventListener('click', downloadSampleCsv);
refreshHealth.addEventListener('click', refreshHealthStatus);
refreshModel.addEventListener('click', refreshModelInfo);

batchFile.addEventListener('change', () => {
  const file = batchFile.files && batchFile.files[0];
  if (!file) {
    return;
  }

  fileDropLabel.textContent = file.name;

  const reader = new FileReader();
  reader.onload = () => {
    try {
      const rows = parseCsvRows(String(reader.result || ''));
      batchRows.textContent = String(rows.length);
      batchPositiveRate.textContent = 'Ready';
      batchTopRisk.textContent = rows[0]?.id || '-';
    } catch (error) {
      batchRows.textContent = '0';
      batchPositiveRate.textContent = 'Invalid CSV';
      batchTopRisk.textContent = error.message;
    }
  };
  reader.readAsText(file);
});

setActiveTab('single');
refreshHealthStatus();
refreshModelInfo();
