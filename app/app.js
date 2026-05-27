const API = 'https://underline-constable-quill.ngrok-free.dev';
const TEAMS = ["AmaZulu FC","Cape Town City FC","Chippa United","Durban City","Golden Arrows","Kaizer Chiefs","Magesi FC","Mamelodi Sundowns","Marumo Gallants","Orbit College","Orlando Pirates","Polokwane City","Richards Bay","Royal AM","Sekhukhune United","Siwelele FC","Stellenbosch FC","SuperSport United","TS Galaxy"];

let predictChart = null;

// ── NAVIGATION ──────────────────────────────────────────────
function navigate(pageId) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  document.getElementById('page-' + pageId).classList.add('active');
  document.querySelector(`[data-page="${pageId}"]`).classList.add('active');
}

document.querySelectorAll('.nav-link').forEach(link => {
  link.addEventListener('click', e => {
    e.preventDefault();
    navigate(link.dataset.page);
  });
});

// ── TEAMS POPULATION ────────────────────────────────────────
async function loadTeams() {
  let teams = TEAMS;
  try {
    const r = await fetch(API + '/teams', { headers: { 'ngrok-skip-browser-warning': '1' } });
    if (r.ok) { const d = await r.json(); if (d.teams?.length) teams = d.teams; }
  } catch(e) {}
  ['home-team','away-team','ins-home','ins-away'].forEach(id => {
    const sel = document.getElementById(id);
    teams.forEach(t => { const o = document.createElement('option'); o.value = o.textContent = t; sel.appendChild(o); });
  });
}

// ── DASHBOARD: TRAIN ────────────────────────────────────────
async function trainModel() {
  const btn = document.getElementById('train-btn');
  const status = document.getElementById('train-status');
  btn.disabled = true;
  document.getElementById('train-icon').textContent = '⏳';
  status.style.color = '#93c5fd';
  status.textContent = 'Training in progress...';
  try {
    const r = await fetch(API + '/train', { method: 'POST', headers: { 'ngrok-skip-browser-warning': '1' } });
    const d = await r.json();
    status.style.color = '#10b981';
    status.textContent = '✓ ' + (d.message || 'Model trained successfully!');
  } catch(e) {
    status.style.color = '#f87171';
    status.textContent = '✗ Error: ' + e.message;
  } finally {
    btn.disabled = false;
    document.getElementById('train-icon').textContent = '▶';
  }
}

// ── MATCH PREDICTOR ──────────────────────────────────────────
async function predict() {
  const home = document.getElementById('home-team').value;
  const away = document.getElementById('away-team').value;
  const errEl = document.getElementById('predict-error');
  errEl.style.display = 'none';
  if (!home || !away) { errEl.textContent = 'Please select both teams.'; errEl.style.display = 'block'; return; }

  try {
    const r = await fetch(API + '/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
      body: JSON.stringify({ home_team: home, away_team: away })
    });
    if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Prediction failed.'); }
    const d = await r.json();
    showPredictResult(d);
  } catch(e) {
    errEl.textContent = e.message;
    errEl.style.display = 'block';
  }
}

function showPredictResult(d) {
  document.getElementById('predict-placeholder').style.display = 'none';
  const res = document.getElementById('predict-result');
  res.style.display = 'block';
  document.getElementById('result-prediction').textContent = d.prediction;

  const labels = ['Home Win', 'Draw', 'Away Win'];
  const values = [d.home_win_prob, d.draw_prob, d.away_win_prob];
  const colors = ['#3b82f6', '#8b5cf6', '#ef4444'];

  if (predictChart) predictChart.destroy();
  predictChart = new Chart(document.getElementById('predict-chart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: colors, borderRadius: 8 }]
    },
    options: {
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` ${ctx.raw}%` } },
        datalabels: { display: false }
      },
      scales: {
        x: { ticks: { color: '#8892a4' }, grid: { display: false } },
        y: { display: false, max: 100 }
      },
      animation: { duration: 600 }
    }
  });
}

// ── INSURANCE ────────────────────────────────────────────────
async function calcInsurance() {
  const home = document.getElementById('ins-home').value;
  const away = document.getElementById('ins-away').value;
  const prize = parseFloat(document.getElementById('ins-prize').value);
  const margin = parseFloat(document.getElementById('ins-margin').value);
  const cond = document.getElementById('ins-condition').value;
  const errEl = document.getElementById('ins-error');
  errEl.style.display = 'none';

  if (!home || !away) { errEl.textContent = 'Please select both teams.'; errEl.style.display = 'block'; return; }

  try {
    const r = await fetch(API + '/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': '1' },
      body: JSON.stringify({ home_team: home, away_team: away })
    });
    if (!r.ok) throw new Error('Prediction failed');
    const d = await r.json();
    const prob = cond === 'home' ? d.home_win_prob : cond === 'draw' ? d.draw_prob : d.away_win_prob;
    const probDec = prob / 100;
    const expectedLoss = probDec * prize;
    const premium = expectedLoss * (1 + margin / 100);
    const profit = premium - expectedLoss;
    showInsurance({ prob, prize, expectedLoss, premium, profit, margin });
  } catch(e) {
    errEl.textContent = e.message;
    errEl.style.display = 'block';
  }
}

function fmt(n) { return 'R ' + Math.round(n).toLocaleString('en-ZA'); }

function showInsurance({ prob, prize, expectedLoss, premium, profit, margin }) {
  document.getElementById('ins-placeholder').style.display = 'none';
  const res = document.getElementById('ins-result');
  res.style.display = 'block';
  document.getElementById('ins-premium-val').textContent = fmt(premium);
  document.getElementById('ins-prob').textContent = prob.toFixed(2) + '%';
  document.getElementById('ins-prize-val').textContent = fmt(prize);
  document.getElementById('ins-loss').textContent = fmt(expectedLoss);
  document.getElementById('ins-margin-val').textContent = margin + '%';
  document.getElementById('ins-profit').textContent = '+' + fmt(profit);

  const interpretation = `The XGBoost algorithm determined a **${prob.toFixed(2)}%** probability for this event. 
  To cover a risk of **${fmt(prize)}**, the actuarial "pure" cost is **${fmt(expectedLoss)}**. 
  By applying a **${margin}%** margin, the premium of **${fmt(premium)}** generates a statistical profit of **${fmt(profit)}**. 
  This pricing secures the financial risk by relying on objective historical data rather than sporting intuition.`;
  
  const textEl = document.getElementById('ins-interpretation-text');
  if (textEl) {
    textEl.innerHTML = interpretation.replace(/\*\*(.*?)\*\*/g, '<strong style="color:var(--text)">$1</strong>');
    document.getElementById('ins-interpretation').style.display = 'block';
  }
}


// ── BATCH PREDICT ────────────────────────────────────────────
function fileSelected(input) {
  document.getElementById('dropzone-label').textContent = input.files[0]?.name || 'Drag and drop or click to select CSV';
}

async function batchPredict(e) {
  e.preventDefault();
  const file = document.getElementById('batch-file').files[0];
  if (!file) return alert('Please select a CSV file.');
  const formData = new FormData();
  formData.append('file', file);
  try {
    const r = await fetch(API + '/batch', { method: 'POST', headers: { 'ngrok-skip-browser-warning': '1' }, body: formData });
    const d = await r.json();
    renderBatchTable(d.results || []);
  } catch(e) {
    alert('Error: ' + e.message);
  }
}

function renderBatchTable(rows) {
  const tbody = document.getElementById('batch-tbody');
  tbody.innerHTML = '';
  rows.forEach(r => {
    const tr = document.createElement('tr');
    if (r.Error) {
      tr.innerHTML = `<td>${r.HomeTeam}</td><td>${r.AwayTeam}</td><td colspan="4" style="color:#f87171">${r.Error}</td>`;
    } else {
      tr.innerHTML = `<td>${r.HomeTeam}</td><td>${r.AwayTeam}</td><td>${r['Home Win Prob']}%</td><td>${r['Draw Prob']}%</td><td>${r['Away Win Prob']}%</td><td><strong>${r.Prediction}</strong></td>`;
    }
    tbody.appendChild(tr);
  });
  document.getElementById('batch-results').style.display = 'block';
}

// ── INIT ─────────────────────────────────────────────────────
loadTeams();
