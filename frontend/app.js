const form = document.getElementById('predict-form');
const resultDiv = document.getElementById('result');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  resultDiv.classList.add('hidden');
  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  // convert numeric fields
  payload.age = Number(payload.age);
  payload.driving_license = Number(payload.driving_license);
  payload.region_code = Number(payload.region_code);
  payload.previously_insured = Number(payload.previously_insured);
  payload.annual_premium = Number(payload.annual_premium);
  payload.policy_sales_channel = Number(payload.policy_sales_channel);
  payload.vintage = Number(payload.vintage);

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      const err = await resp.json();
      resultDiv.textContent = `Error: ${err.detail || resp.statusText}`;
      resultDiv.classList.remove('hidden');
      return;
    }

    const data = await resp.json();
    resultDiv.innerHTML = `<strong>Prediction:</strong> ${data.prediction} (<em>${(data.probability*100).toFixed(2)}%</em>)<br><small>Threshold: ${data.threshold}</small>`;
    resultDiv.classList.remove('hidden');
  } catch (err) {
    resultDiv.textContent = `Network error: ${err.message}`;
    resultDiv.classList.remove('hidden');
  }
});
