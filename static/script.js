// Chart.js registrations
if (window.Chart) {
  const { TimeSeriesScale, TimeScale, LinearScale, CategoryScale, Tooltip, Legend } = Chart;
  // Register core pieces; plugin controllers (candlestick/ohlc) will be used only when chart is requested
  Chart.register(TimeSeriesScale, TimeScale, LinearScale, CategoryScale, Tooltip, Legend);
}

function escapeHtml(v){
  return String(v).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// Tabs removed: single page

const autopilotHistory = [];
let autopilotChart = null;
let autopilotBusy = false;

function isTabVisible(id){
  const el = document.getElementById(id);
  return !!(el && !el.classList.contains('hidden'));
}

function ensureAutopilotChart(){ return null; }

async function apiGet(path, params = {}) {
  const url = new URL(`/api/${path}`, location.origin);
  Object.entries(params).forEach(([k,v]) => url.searchParams.append(k, v));
  const r = await fetch(url);
  if (!r.ok) throw new Error(`API ${r.status}: ${await r.text()}`);
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(`/api/${path}`, { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`API ${r.status}: ${await r.text()}`);
  return r.json();
}

// Chart tab
document.getElementById('load-chart')?.addEventListener('click', async () => {
  try {
    const sym = document.getElementById('chart-symbol').value || 'BTC/USD';
    const tf = document.getElementById('chart-timeframe').value || '5Min';
    const data = await apiGet('bars', { symbols: sym, timeframe: tf, limit: 300, sort: 'asc' });
    const cont = data.bars || {};
    const series = Array.isArray(cont) ? cont : (cont[sym] || Object.values(cont)[0] || []);
    const candles = series.map(b => ({ x: new Date(b.t || b.timestamp), o: b.o, h: b.h, l: b.l, c: b.c }));
    const ctx = document.getElementById('candles-canvas').getContext('2d');
    if (window.candleChart) window.candleChart.destroy();
    // Try candlestick; if plugin missing, fall back to line chart
    const canUseCandle = !!(Chart.CandlestickController || (Chart.registry && Chart.registry.getController && Chart.registry.getController('candlestick')));
    if (canUseCandle) {
    window.candleChart = new Chart(ctx, { type: 'candlestick', data: { datasets: [{ label: `${sym} ${tf}`, data: candles }] }, options: { responsive: true } });
    } else {
      const line = candles.map(c => ({ x: c.x, y: c.c }));
      window.candleChart = new Chart(ctx, { type: 'line', data: { datasets: [{ label: `${sym} ${tf}`, data: line, borderColor: '#7aa2ff', pointRadius: 0 }] }, options: { responsive: true, parsing: false } });
      alert('Chart plugin not available; showing line chart instead.');
    }
  } catch(e) {
    alert('Chart load error: '+(e?.message || e));
  }
});

// Formatting helpers
function fmt(n, d=2){ n = Number(n || 0); return n.toLocaleString(undefined,{maximumFractionDigits:d}); }
function setText(id, v){ const el=document.getElementById(id); if(el) el.textContent = v; }
function setClass(el, c){ if(el){ el.classList.remove('positive','negative'); if(c) el.classList.add(c); } }

// Autopilot
const autopilotForm = document.getElementById('autopilot-form');
autopilotForm?.addEventListener('submit', async e => {
  e.preventDefault();
  if (autopilotBusy) return;
  if (!localStorage.getItem('crypto_disclosure_ok')) {
    const ok = confirm('Crypto disclosure: Crypto trading via Alpaca Crypto LLC is not FDIC/SIPC insured, is non‑marginable, not shortable, and runs 24/7. Supported order types: Market, Limit, Stop Limit. Proceed?');
    if (!ok) return;
    localStorage.setItem('crypto_disclosure_ok','1');
  }
  autopilotBusy = true;
  const startBtn = document.getElementById('start-autopilot');
  const stopBtn = document.getElementById('stop-autopilot');
  if (startBtn) startBtn.disabled = true;
  const fd = new FormData(autopilotForm);
  const payload = {
    symbol: String(fd.get('symbol') || '').trim(),
    fast_window: Number(fd.get('fast_window')),
    slow_window: Number(fd.get('slow_window')),
    rsi_window: Number(fd.get('rsi_window')),
    overbought: Number(fd.get('overbought')),
    oversold: Number(fd.get('oversold')),
    base_interval: Number(fd.get('base_interval')),
    base_steps: Number(fd.get('base_steps')),
    rung_notional: Number(fd.get('rung_notional')),
    max_notional: Number(fd.get('max_notional')),
    volatility_lookback: Number(fd.get('volatility_lookback')),
    risk_multiplier: Number(fd.get('risk_multiplier')),
    poll_seconds: Number(fd.get('poll_seconds')),
  };
  try { await apiPost('start-autopilot', payload); await refreshStatus(); }
  catch (err) { alert('Autopilot start error: ' + err.message); }
  finally { autopilotBusy = false; if (startBtn) startBtn.disabled = false; if (stopBtn) stopBtn.disabled = false; }
});
document.getElementById('stop-autopilot')?.addEventListener('click', async () => {
  if (autopilotBusy) return;
  autopilotBusy = true;
  const startBtn = document.getElementById('start-autopilot');
  const stopBtn = document.getElementById('stop-autopilot');
  if (stopBtn) stopBtn.disabled = true;
  try { await apiPost('stop-autopilot', {}); await refreshStatus(); }
  catch (err) { alert('Autopilot stop error: ' + err.message); }
  finally { autopilotBusy = false; if (startBtn) startBtn.disabled = false; if (stopBtn) stopBtn.disabled = false; }
});

// Manual ladder UI removed; presets and preview logic deleted

// Rendering helpers
function renderOrders(id, orders){
  const tb = document.querySelector(`#${id} tbody`);
  if (!tb) return;
  tb.innerHTML = orders.map(o => `
    <tr>
      <td>${o.side?.toUpperCase()||''}</td>
      <td>${o.qty ?? ''}</td>
      <td>${o.limit_price ?? o.filled_avg_price ?? ''}</td>
      <td>${o.status ?? ''}</td>
      <td title="${o.id}">${String(o.id).slice(0,8)}…</td>
    </tr>
  `).join('');
}
function renderFilledOrders(id, orders){
  const tb = document.querySelector(`#${id} tbody`);
  if (!tb) return;
  tb.innerHTML = (orders||[]).map(o => {
    const t = o.filled_at || o.updated_at || o.created_at || '';
    const tt = t ? new Date(t).toLocaleTimeString() : '';
    const price = o.filled_avg_price ?? o.limit_price ?? '';
    return `
      <tr>
        <td>${o.side?.toUpperCase()||''}</td>
        <td>${o.filled_qty ?? o.qty ?? ''}</td>
        <td>${price}</td>
        <td>${escapeHtml(tt)}</td>
        <td title="${o.id}">${String(o.id).slice(0,8)}…</td>
      </tr>`;
  }).join('');
}
function renderActivity(lines){
  const tb = document.querySelector('#activity tbody');
  if (!tb) return;
  const events = [];
  for (const line of lines.slice(-50)) {
    if (line.includes('Placed')) events.push({ t: line.slice(0,19), e: 'Placed', d: line.split('Placed ')[1] });
    else if (line.includes('Cancelled stray')) events.push({ t: line.slice(0,19), e: 'Cancelled', d: line.split('Cancelled ')[1] });
    else if (line.includes('Cancelled ')) events.push({ t: line.slice(0,19), e: 'Cancelled', d: line.split('Cancelled ')[1] });
    else if (line.includes('Processed fill')) events.push({ t: line.slice(0,19), e: 'Filled', d: line.split('Processed fill ')[1] });
    else if (line.includes('price')) events.push({ t: line.slice(0,19), e: 'Price', d: line.split('INFO ladder ')[1] || line });
  }
  tb.innerHTML = events.map(ev => `<tr><td>${ev.t}</td><td>${ev.e}</td><td>${ev.d||''}</td></tr>`).join('');
}

function renderAutopilotHistory(auto){
  const body = document.getElementById('auto-history-body');
  if (!body) return;
  const rows = (auto?.history || []).slice().reverse();
  if (!rows.length) { body.innerHTML = '<tr><td colspan="7">Waiting for the next autopilot run…</td></tr>'; return; }
  const pct = v => (v==null?'—':`${Number(v).toFixed(2)}%`);
  const num = (v, d=2) => (v==null?'—':Number(v).toFixed(d));
  const price = v => (v==null?'—':`$${Number(v).toLocaleString(undefined,{maximumFractionDigits:2})}`);
  body.innerHTML = rows.map(e => {
    const t = e.ts ? new Date(e.ts).toLocaleTimeString() : '—';
    return `<tr>
      <td>${escapeHtml(t)}</td>
      <td>${escapeHtml(e.action||'—')}</td>
      <td>${escapeHtml(e.note||'—')}</td>
      <td>${escapeHtml(pct(e.trend_pct))}</td>
      <td>${escapeHtml(pct(e.volatility_pct))}</td>
      <td>${escapeHtml(num(e.rsi))}</td>
      <td>${escapeHtml(price(e.price))}</td>
    </tr>`;
  }).join('');
}

function updateAutopilotPanel(auto){
  const state = document.getElementById('auto-state');
  if (!state) return;
  const running = !!(auto && auto.running);
  state.textContent = running ? 'Running' : 'Idle';
  state.className = running ? 'badge badge-on' : 'badge badge-off';
  const startBtn = document.getElementById('start-autopilot');
  const stopBtn = document.getElementById('stop-autopilot');
  if (startBtn) startBtn.disabled = running;
  if (stopBtn) stopBtn.disabled = !running;

  const set = (id, v)=>{ const el=document.getElementById(id); if (el) el.textContent = v; };
  let sig = 'Idle';
  if (running) sig = (auto?.last_signal && auto.last_signal!=='stopped') ? auto.last_signal.toUpperCase() : 'Watching';
  set('auto-last-signal', sig);
  set('auto-last-reason', running ? (auto?.last_reason || 'Waiting for EMA / RSI alignment') : '—');

  const ps = auto?.config?.poll_seconds;
  if (ps) {
    if (running && auto?.last_run) {
      const last = new Date(auto.last_run);
      const next = new Date(last.getTime() + ps*1000);
      set('auto-next-poll', `${ps}s cadence · next ~${next.toLocaleTimeString()}`);
    } else {
      set('auto-next-poll', `${ps}s cadence`);
    }
  } else set('auto-next-poll','—');

  const d = auto?.last_decision || {};
  set('auto-trend', d.trend_pct!=null ? `${d.trend_pct.toFixed(2)}%` : '—');
  set('auto-vol',   d.volatility_pct!=null ? `${d.volatility_pct.toFixed(2)}%` : '—');
  set('auto-rsi',   d.rsi!=null ? d.rsi.toFixed(2) : '—');
  set('auto-price', d.price!=null ? fmt(d.price,2) : '—');

  const applied = auto?.applied_ladder;
  set('auto-applied', applied ? `${applied.direction} · ${applied.steps} steps @ $${Number(applied.interval).toFixed(2)} (size ${Number(applied.size).toFixed(6)})` : '—');
  set('auto-last-run', auto?.last_run ? new Date(auto.last_run).toLocaleTimeString() : '—');
  const errEl = document.getElementById('auto-error');
  if (errEl) { const has = !!auto?.last_error; errEl.textContent = has ? auto.last_error : '—'; errEl.classList.toggle('negative', has); }
  const cfgEl = document.getElementById('auto-config');
  if (cfgEl) cfgEl.textContent = auto?.config ? JSON.stringify(auto.config, null, 2) : '{}';

  renderAutopilotHistory(auto);
}

let lastPriceCache = 0;
async function refreshStatus(){
  try {
    const st = await apiGet('status');
    document.getElementById('status-box').textContent = JSON.stringify(st, null, 2);

    // badges and buttons
    const running = !!st.running;
    const badge = document.getElementById('bot-state');
    badge.textContent = running ? 'Running' : 'Stopped';
    badge.className = running ? 'badge badge-on' : 'badge badge-off';
    // Manual start/stop controls removed

    // metrics
    setText('m-position', fmt(st.position_qty,6));
    setText('m-avgprice', fmt(st.avg_price,2));
    setText('m-pnl', fmt(st.realized_pnl,2));
    const pnlEl = document.getElementById('m-pnl'); if (pnlEl) { pnlEl.className = 'value ' + ((st.realized_pnl||0)>=0 ? 'money-pos' : 'money-neg'); }
    setText('m-openorders', st.open_orders.length);

    lastPriceCache = st.last_price; // backend must supply; no fallback
    setText('m-last', fmt(lastPriceCache,2));

    const upnlVal = st.unrealized_pnl_usd ?? ((st.position_qty || 0) * ((lastPriceCache || 0) - (st.avg_price || 0)));
    const upnlEl = document.getElementById('m-upnl'); setText('m-upnl', fmt(upnlVal,2)); setClass(upnlEl, upnlVal>=0?'positive':'negative');

    const steps = st.config?.steps || 0;
    const size = st.config?.size || 0;
    const maxNotional = (steps*size) * (lastPriceCache || 0);
    const capitalUsed = (st.position_qty||0) * (lastPriceCache||0);
    const remaining = Math.max(0, maxNotional - capitalUsed);
    setText('m-capused', fmt(capitalUsed,2));
    const cuEl = document.getElementById('m-capused'); if (cuEl) { cuEl.className = 'value ' + ((capitalUsed||0)>0 ? 'used-active' : ''); }
    setText('m-maxnotional', fmt(maxNotional,2));
    setText('m-remaining', fmt(remaining,2));
    setText('m-working', fmt(st.open_orders_notional_usd,2));
    setText('m-posnotional', fmt(st.position_notional_usd,2));
    setText('m-target', fmt(st.deployed_ladder_notional_usd,2));
    setText('m-net', fmt(st.net_deployed_usd,2));

    const realized = Number.isFinite(st.realized_pnl) ? st.realized_pnl : 0;
    const netPnl = realized + (Number.isFinite(upnlVal) ? upnlVal : 0);
    autopilotHistory.push({ t:new Date(), capital:Number.isFinite(capitalUsed)?capitalUsed:0, pnl:Number.isFinite(netPnl)?netPnl:0 });
    if (autopilotHistory.length > 240) autopilotHistory.shift();
  // chart removed

    if (st.position_qty >= (st.config?.max_exposure || 0) * 0.95) document.getElementById('m-position').classList.add('negative');
    else document.getElementById('m-position').classList.remove('negative');

    // risk banner (UI only)
    const lossEnabled = document.getElementById('risk-enable-loss')?.checked;
    const cap = Number(document.getElementById('risk-losscap')?.value || 0);
    const volEnabled = document.getElementById('risk-enable-vol')?.checked;
    const vol = Number(document.getElementById('risk-vol')?.value || 0);
    const dailyOK = !lossEnabled || (st.realized_pnl + upnlVal >= -Math.abs(cap));
    const percMove = st.avg_price ? Math.abs((lastPriceCache - st.avg_price)/st.avg_price)*100 : 0;
    const volOK = !volEnabled || percMove <= Math.abs(vol);
    const banner = document.getElementById('last-action');
    if (banner) banner.textContent = (dailyOK && volOK)
      ? `OK: position ${fmt(st.position_qty,6)} BTC, ${st.open_orders.length} open rungs`
      : `PAUSED: ${!dailyOK?`Daily loss cap hit (${fmt(st.realized_pnl+upnlVal,2)})`:''} ${!volOK?`Vol ${percMove.toFixed(2)}% > ${vol}%`:''}`.trim();

    // side panel
    setText('sp-open', st.open_orders.length);
    setText('sp-position', fmt(st.position_qty,6));
    setText('sp-avg', fmt(st.avg_price,2));
    setText('sp-last', fmt(lastPriceCache,2));
    setText('sp-upnl', fmt(upnlVal,2));
    const supnl = document.getElementById('sp-upnl'); if (supnl) { supnl.className = 'mono ' + (upnlVal>=0 ? 'money-pos' : 'money-neg'); }
    setText('sp-rpnl', fmt(st.realized_pnl,2));
    const srpnl = document.getElementById('sp-rpnl'); if (srpnl) { srpnl.className = 'mono ' + ((st.realized_pnl||0)>=0 ? 'money-pos' : 'money-neg'); }
    setText('sp-used', fmt(capitalUsed,2));
    setText('sp-remaining', fmt(remaining,2));
    document.getElementById('sp-action').textContent = st.last_action || '—';
    const dir = st.config?.direction || 'BUY';
    const rungs = (()=>{
      const arr = []; const s = st.config?.steps||0; const itv = st.config?.interval||0; const last = lastPriceCache||0;
      for (let i=0;i<s;i++){ const sign = dir==='BUY' ? -1 : 1; arr.push((last + sign*i*itv).toFixed(2)); }
      return arr;
    })();
    document.getElementById('sp-rungs').innerHTML = rungs.map(p=>`<li>$${p}</li>`).join('');
    setText('strip-pos', fmt(st.position_qty,6));
    setText('strip-avg', fmt(st.avg_price,2));
    setText('strip-upnl', fmt(upnlVal,2));
    setText('strip-upct', fmt(st.unrealized_pnl_pct,2));
    setText('strip-day', fmt(st.day_realized_pnl_usd,2));
    setText('strip-used', fmt(capitalUsed,2));
    setText('strip-cap', fmt(remaining,2));
    setText('sp-range', rungs.length? `$${rungs[0]} → $${rungs[rungs.length-1]}`:'-');

    try {
      const acct = await apiGet('account');
      setText('strip-cash', fmt(acct.cash ?? 0, 2));
      setText('strip-equity', fmt(acct.equity ?? acct.portfolio_value ?? 0, 2));
      setText('strip-bp', fmt(acct.buying_power ?? 0, 2));
    } catch {}

    renderOrders('open-orders', st.open_orders);
    renderFilledOrders('filled-orders', st.filled_orders);

    // chips
    const chipBot = document.getElementById('chip-bot');
    const chipFlat = document.getElementById('chip-flat');
    const chipOrders = document.getElementById('chip-orders');
    const chipRisk = document.getElementById('chip-risk');
    if (chipBot) { chipBot.textContent = running? 'Bot: Running' : 'Bot: Stopped'; chipBot.className = 'chip ' + (running?'ok':'warn'); }
    const isFlat = Math.abs(st.position_qty || 0) < 1e-8;
    if (chipFlat) { chipFlat.textContent = isFlat? 'Position: Flat' : `Position: ${fmt(st.position_qty,6)} BTC`; chipFlat.className = 'chip '+(isFlat?'ok':'warn'); }
    const openCount = st.open_order_count ?? st.open_orders.length;
    if (chipOrders) { chipOrders.textContent = openCount? `Open orders: ${openCount}` : 'Open orders: 0'; chipOrders.className = 'chip ' + (openCount? 'warn':'ok'); }
    const riskOK = document.getElementById('last-action')?.textContent.startsWith('OK:');
    if (chipRisk) { chipRisk.textContent = riskOK? 'Risk: OK' : 'Risk: Paused'; chipRisk.className = 'chip ' + (riskOK? 'ok':'err'); }

    updateAutopilotPanel(st.autopilot);
  } catch (e) {
    document.getElementById('status-box').textContent = 'Error: '+e;
  }
}
async function refreshLogs(){
  try {
    const L = await apiGet('logs', {tail: 200});
    const lines = L.logs || [];
    const box = document.getElementById('log-box');
    if (box) {
      const prevTop = box.scrollTop;
      const atBottom = (box.scrollTop + box.clientHeight) >= (box.scrollHeight - 5);
      box.textContent = lines.join('\n');
      if (atBottom) box.scrollTop = box.scrollHeight;
      else box.scrollTop = prevTop;
    }
    renderActivity(lines);
  } catch (e) {
    document.getElementById('log-box').textContent = 'Log error: '+e;
  }
}
setInterval(refreshStatus, 5000);
setInterval(refreshLogs, 7000);
refreshStatus();
refreshLogs();

// Health
async function refreshHealth(){
  try {
    const h = await apiGet('health');
    const ok = v => v ? 'OK' : 'Fail';
    const setBadge = (id, state)=>{
      const el = document.getElementById(id);
      if (!el) return;
      el.textContent = state ? 'OK' : 'Fail';
      el.className = 'badge ' + (state ? 'badge-on' : 'badge-off');
    };
    setBadge('h-trading', h.trading_ok);
    setBadge('h-data', h.data_ok);
    const age = (v)=> (v==null? '—' : `${Math.round(v)}s`);
    setText('h-price-age', age(h.last_price_age_s));
    const ar = document.getElementById('h-auto-running'); if (ar) { ar.textContent = h.autopilot_running ? 'Yes' : 'No'; ar.className = 'badge ' + (h.autopilot_running ? 'badge-on' : 'badge-off'); }
    setText('h-auto-age', age(h.autopilot_last_run_age_s));
    const hc = document.getElementById('h-crypto'); if (hc) { hc.textContent = h.crypto_status || '—'; hc.className = 'badge ' + (h.crypto_status==='ACTIVE' ? 'badge-on' : 'badge-warn'); }
    const hb = document.getElementById('h-blocked'); if (hb) { hb.textContent = h.account_blocked ? 'Yes' : 'No'; hb.className = 'badge ' + (h.account_blocked ? 'badge-off' : 'badge-on'); }
    const hn = document.getElementById('h-nmbp'); if (hn) { hn.textContent = (h.non_marginable_buying_power!=null) ? `$${Number(h.non_marginable_buying_power).toLocaleString(undefined,{maximumFractionDigits:2})}` : '—'; }

    // Summary banner
    const badge = document.getElementById('h-summary-badge');
    const txt = document.getElementById('h-summary-text');
    if (badge) {
      const cls = h.severity === 'error' ? 'badge-off' : (h.severity === 'warn' ? 'badge-warn' : 'badge-on');
      badge.className = `badge ${cls}`;
      badge.textContent = h.severity === 'error' ? 'Error' : (h.severity === 'warn' ? 'Warning' : 'OK');
    }
    if (txt) {
      txt.textContent = h.summary || '—';
      if (h.issues && h.issues.length) {
        txt.title = h.issues.join(' \n');
      }
    }
  } catch (e) {
    const err = id=>{ const el=document.getElementById(id); if(el){ el.textContent='Error'; el.className='badge badge-off'; }};
    err('h-trading'); err('h-data');
    const badge = document.getElementById('h-summary-badge'); if (badge) { badge.className='badge badge-off'; badge.textContent='Error'; }
    const txt = document.getElementById('h-summary-text'); if (txt) { txt.textContent='Health check failed'; }
  }
}
setInterval(refreshHealth, 8000);
refreshHealth();

// Self-test
document.getElementById('run-selftest')?.addEventListener('click', async ()=>{
  const out = document.getElementById('selftest-output'); if (out) out.textContent = 'Running self-test…';
  try {
    const r = await apiPost('self-test', {});
    if (out) out.textContent = r.report || JSON.stringify(r, null, 2);
  } catch (e) {
    if (out) out.textContent = 'Self-test error: ' + (e?.message || e);
  }
});

// Demo controls removed
document.getElementById('panic-close')?.addEventListener('click', async ()=>{
  if (!confirm('Cancel all open orders and flatten any positions (paper)?')) return;
  try {
    await fetch('/api/panic-close?symbol=ALL', { method: 'POST' });
    await refreshStatus();
  } catch (e) { alert('Panic close error: '+(e?.message||e)); }
});

// Header controls reuse the same handlers
document.getElementById('header-start')?.addEventListener('click', ()=>{
  document.getElementById('start-autopilot')?.click();
});
document.getElementById('header-stop')?.addEventListener('click', async ()=>{
  try { await apiPost('stop-autopilot', {}); await refreshStatus(); }
  catch (err) { alert('Autopilot stop error: ' + err.message); }
});
document.getElementById('header-panic')?.addEventListener('click', async ()=>{
  if (!confirm('Cancel all open orders and flatten any positions (paper)?')) return;
  try { await fetch('/api/panic-close?symbol=ALL', { method: 'POST' }); await refreshStatus(); }
  catch (e) { alert('Panic close error: '+(e?.message||e)); }
});
document.getElementById('header-selftest')?.addEventListener('click', ()=>{
  document.getElementById('run-selftest')?.click();
});
