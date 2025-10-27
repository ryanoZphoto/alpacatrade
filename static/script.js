// Ensure Chart.js has required registrations for financial charts
if (window.Chart) {
    const { TimeSeriesScale, TimeScale, LinearScale, CategoryScale, Tooltip, Legend } = Chart;
    if (TimeSeriesScale) Chart.register(TimeSeriesScale);
    if (TimeScale) Chart.register(TimeScale);
    if (LinearScale) Chart.register(LinearScale);
    if (CategoryScale) Chart.register(CategoryScale);
    if (Tooltip) Chart.register(Tooltip);
    if (Legend) Chart.register(Legend);
    // Financial plugin registration (if present)
    if (Chart.FinancialController) Chart.register(Chart.FinancialController);
    if (Chart.CandlestickController) Chart.register(Chart.CandlestickController);
    if (Chart.OhlcController) Chart.register(Chart.OhlcController);
    if (Chart.CandlestickElement) Chart.register(Chart.CandlestickElement);
    if (Chart.OhlcElement) Chart.register(Chart.OhlcElement);
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// Tabs (Overview | Manual | Autopilot | Market)
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const tab = btn.getAttribute('data-tab');
        document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
        const el = document.getElementById(tab);
        if (el) el.classList.remove('hidden');
        // Refresh status when switching back to Status tab
        if (tab === 'overview-tab') { refreshStatus(); refreshLogs(); }
        if (tab === 'autopilot-tab') {
            const chart = ensureAutopilotChart();
            if (chart) {
                chart.data.datasets[0].data = autopilotHistory.map(d => ({ x: d.t, y: d.capital }));
                chart.data.datasets[1].data = autopilotHistory.map(d => ({ x: d.t, y: d.pnl }));
                chart.update();
                chart.resize();
            }
        }
    });
});

const autopilotHistory = [];
let autopilotChart = null;

function ensureAutopilotChart() {
    if (autopilotChart) return autopilotChart;
    const canvas = document.getElementById('autopilot-usage-canvas');
    if (!canvas || !window.Chart) return null;
    if (!canvas.offsetWidth) return null;
    const ctx = canvas.getContext('2d');
    autopilotChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: 'Capital Used (USD)',
                    data: [],
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: false,
                    tension: 0.2,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'Net P/L (USD)',
                    data: [],
                    borderColor: '#16a34a',
                    backgroundColor: 'rgba(22, 163, 74, 0.1)',
                    fill: false,
                    tension: 0.2,
                    pointRadius: 0,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            parsing: false,
            scales: {
                x: { type: 'time', time: { unit: 'minute' }, ticks: { autoSkip: true } },
                y: {
                    position: 'left',
                    ticks: {
                        callback: v => Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })
                    },
                    title: { display: true, text: 'Capital Used (USD)' }
                },
                y1: {
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: {
                        callback: v => Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 })
                    },
                    title: { display: true, text: 'Net P/L (USD)' }
                }
            },
            plugins: {
                legend: { position: 'bottom' },
                tooltip: {
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: $${Number(ctx.parsed.y).toLocaleString(undefined, { maximumFractionDigits: 2 })}`
                    }
                }
            }
        }
    });
    return autopilotChart;
}

async function apiGet(path, params = {}) {
    const url = new URL(`/api/${path}`, location.origin);
    Object.entries(params).forEach(([k, v]) => url.searchParams.append(k, v));
    const resp = await fetch(url);
    if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`API ${resp.status}: ${txt}`);
    }
    return resp.json();
}
async function apiPost(path, body) {
    const resp = await fetch(`/api/${path}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
    });
    if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`API ${resp.status}: ${txt}`);
    }
    return resp.json();
}

/* ---------- Chart (optional) ---------- */
document.getElementById('load-chart').addEventListener('click', async () => {
    const symbol = document.getElementById('chart-symbol').value.trim();
    const tf = document.getElementById('chart-timeframe').value;
    try {
        const data = await apiGet(
            'bars',
            { symbols: symbol, timeframe: tf, limit: 500 }
        );
        const container = data.bars || [];
        const series = Array.isArray(container)
            ? container
            : (container[symbol] || container[symbol.replace('-', '/')] ||
               container[Object.keys(container)[0]] || []);
        const candles = series.map(b => ({
            x: new Date(b.t || b.timestamp),
            o: b.o ?? b.open,
            h: b.h ?? b.high,
            l: b.l ?? b.low,
            c: b.c ?? b.close
        }));
        const ctx = document.getElementById('candles-canvas').getContext('2d');
        const existing = Chart.getChart('candles-canvas');
        if (existing) existing.destroy();
        const unit = tf.includes('Day') ? 'day' : (tf.includes('Hour') ? 'hour' : 'minute');
        // If candlestick controller not available, render a simple line as fallback
        const hasCandle = !!(Chart.registry && Chart.registry.controllers.get && Chart.registry.controllers.get('candlestick'))
            || !!Chart.CandlestickController;
        const chartType = hasCandle ? 'candlestick' : 'line';
        const chartData = hasCandle
            ? { datasets: [{ label: String(symbol), data: candles }] }
            : { datasets: [{ label: String(symbol), data: candles.map(c => ({ x: c.x, y: c.c })) }] };
        window.candleChart = new Chart(ctx, {
            type: chartType,
            data: chartData,
            options: {
                responsive: true,
                parsing: false,
                scales: {
                    x: { type: 'timeseries', time: { unit } },
                    y: { type: 'linear', ticks: { callback: v => Number(v).toLocaleString() } }
                }
            }
        });
    } catch (e) { alert('Chart error: '+e.message); }
});

/* ---------- Ladder form ---------- */
document.getElementById('ladder-form').addEventListener('submit', async e => {
    e.preventDefault();
    const f = e.target;
    // Convert USD inputs to BTC using the latest price
    const last = lastPriceCache || Number(document.getElementById('m-last')?.textContent.replace(/,/g,'')) || 0;
    const usdPerRung = Number(f.size.value);
    const usdMaxExposure = Number(f.max_exposure.value);
    const sizeBtc = last ? (usdPerRung / last) : 0;
    const maxBtc = last ? (usdMaxExposure / last) : 0;
    const cfg = {
        symbol: f.symbol.value.trim(),
        direction: f.direction.value,
        steps: Number(f.steps.value),
        interval: Number(f.interval.value),
        size: Number(sizeBtc.toFixed(8)),
        max_exposure: Number(maxBtc.toFixed(8))
    };
    try { await apiPost('start-ladder', cfg); }
    catch (err) { alert('Start error: '+err.message); }
});
document.getElementById('stop-btn').addEventListener('click', async () => {
    try { await apiPost('stop-ladder', {}); }
    catch (err) { alert('Stop error: '+err.message); }
});

// Nudge / Recenter
const btnUp = document.getElementById('nudge-up');
if (btnUp) btnUp.addEventListener('click', async ()=>{ try { await apiPost('nudge', { direction:'up' }); await refreshStatus(); } catch(e){ alert(e); } });
const btnDn = document.getElementById('nudge-down');
if (btnDn) btnDn.addEventListener('click', async ()=>{ try { await apiPost('nudge', { direction:'down' }); await refreshStatus(); } catch(e){ alert(e); } });
const btnRc = document.getElementById('recenter-now');
if (btnRc) btnRc.addEventListener('click', async ()=>{ try { await apiPost('recenter', {}); await refreshStatus(); } catch(e){ alert(e); } });

const btnClose = document.getElementById('close-all');
if (btnClose) btnClose.addEventListener('click', async ()=>{
    if (!confirm('Close all open orders and flatten position at market?')) return;
    try { await apiPost('close-all', {}); await refreshStatus(); }
    catch (e) { alert('Close-all error: '+e); }
});

// Cancel open rungs only
const btnCancel = document.getElementById('cancel-open');
if (btnCancel) btnCancel.addEventListener('click', async ()=>{
    try { await apiPost('cancel-open', {}); await refreshStatus(); }
    catch (e) { alert('Cancel-open error: '+e.message); }
});

/* ---------- Autopilot ---------- */
const autopilotForm = document.getElementById('autopilot-form');
if (autopilotForm) {
    autopilotForm.addEventListener('submit', async e => {
        e.preventDefault();
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
        try {
            await apiPost('start-autopilot', payload);
            await refreshStatus();
        } catch (err) {
            alert('Autopilot start error: ' + err.message);
        }
    });
}
const stopAutoBtn = document.getElementById('stop-autopilot');
if (stopAutoBtn) {
    stopAutoBtn.addEventListener('click', async () => {
        try {
            await apiPost('stop-autopilot', {});
            await refreshStatus();
        } catch (err) {
            alert('Autopilot stop error: ' + err.message);
        }
    });
}

// Presets
const presets = {
    cons: { steps: 5, interval: 300, size: 0.005 },
    neut: { steps: 7, interval: 200, size: 0.01 },
    aggr: { steps: 10, interval: 100, size: 0.015 },
};
[{id:'preset-cons',k:'cons'},{id:'preset-neut',k:'neut'},{id:'preset-aggr',k:'aggr'}].forEach(({id,k}) => {
    const btn = document.getElementById(id);
    if (btn) btn.addEventListener('click', () => {
        const f = document.getElementById('ladder-form');
        f.steps.value = presets[k].steps;
        f.interval.value = presets[k].interval;
        f.size.value = presets[k].size;
        computePreview();
    });
});

function computeRungs(last, direction, steps, interval) {
    const dir = String(direction).toUpperCase() === 'BUY' ? -1 : 1;
    const r = [];
    for (let i=0;i<steps;i++) r.push(Number((last + dir*i*interval).toFixed(2)));
    return r.sort((a,b)=>a-b);
}
async function computePreview() {
    const f = document.getElementById('ladder-form');
    const symbol = f.symbol.value.trim();
    let last = Number(document.getElementById('m-last')?.textContent.replace(/,/g,'')) || 0;
    if (!last) {
        try {
            const lp = await apiGet('bars', { symbols: symbol, timeframe: '1Min', limit: 1 });
            const cont = lp.bars || {};
            const arr = Array.isArray(cont) ? cont : (cont[symbol] || Object.values(cont)[0] || []);
            last = arr?.[0]?.c || last;
        } catch {}
    }
    const steps = Number(f.steps.value), interval = Number(f.interval.value), size = Number(f.size.value);
    const rungs = computeRungs(last, f.direction.value, steps, interval);
    const totalBtc = steps * size; const totalNotional = totalBtc * last;
    const set = (id,v)=>{ const el=document.getElementById(id); if (el) el.textContent = (typeof v==='number')? v.toLocaleString(undefined,{maximumFractionDigits:2}):v; };
    set('preview-total-btc', totalBtc); set('preview-total-notional', totalNotional);
    set('preview-min', rungs[0]||0); set('preview-max', rungs[rungs.length-1]||0);
    const list=document.getElementById('preview-list'); if (list) list.innerHTML = rungs.map(p=>`<li>$${p.toLocaleString()}</li>`).join('');
    // preview usage vs capacity
    const posBtc = Number((document.getElementById('m-position')?.textContent || '0').replace(/,/g,''));
    const used = posBtc * last; set('preview-used', used); set('preview-remaining', Math.max(0, totalNotional - used));
    // side panel mini rungs
    const spR = document.getElementById('sp-rungs'); if (spR) spR.innerHTML = rungs.slice(0,8).map(p=>`<li>$${p.toLocaleString()}</li>`).join('');
}
['symbol','direction','steps','interval','size'].forEach(n=>{ const el=document.querySelector(`#ladder-form [name="${n}"]`); if(el) el.addEventListener('input', computePreview); });
setTimeout(computePreview, 300);


/* ---------- Auto‑refresh status & logs ---------- */
function setText(id, value) { document.getElementById(id).textContent = value; }
function setClass(el, cls) { el.classList.remove('positive','negative'); if (cls) el.classList.add(cls); }
function fmt(n, d=2) { return Number(n).toLocaleString(undefined, {maximumFractionDigits: d}); }
function renderOrders(tableId, orders) {
    const tb = document.querySelector(`#${tableId} tbody`);
    tb.innerHTML = orders.map(o => `
        <tr>
            <td>${o.side}</td>
            <td>${o.qty ?? o.filled_qty}</td>
            <td>${o.limit_price ?? o.filled_avg_price ?? ''}</td>
            <td>${o.status ?? ''}</td>
            <td title="${o.id}">${String(o.id).slice(0,8)}…</td>
        </tr>
    `).join('');
}
function renderActivity(logLines) {
    const tb = document.querySelector('#activity tbody');
    const events = [];
    for (const line of logLines.slice(-50)) {
        if (line.includes('Placed')) events.push({ t: line.slice(0,19), e: 'Placed', d: line.split('Placed ')[1] });
        else if (line.includes('Cancelled stray')) events.push({ t: line.slice(0,19), e: 'Cancelled', d: line.split('Cancelled ')[1] });
        else if (line.includes('Cancelled ')) events.push({ t: line.slice(0,19), e: 'Cancelled', d: line.split('Cancelled ')[1] });
        else if (line.includes('Processed fill')) events.push({ t: line.slice(0,19), e: 'Filled', d: line.split('Processed fill ')[1] });
        else if (line.includes('price')) events.push({ t: line.slice(0,19), e: 'Price', d: line.split('INFO ladder ')[1] || line });
    }
    tb.innerHTML = events.map(ev => `
        <tr>
            <td>${ev.t}</td>
            <td>${ev.e}</td>
            <td>${ev.d || ''}</td>
        </tr>
    `).join('');
}

function renderAutopilotHistory(auto) {
    const body = document.getElementById('auto-history-body');
    if (!body) return;
    const rows = (auto?.history || []).slice().reverse();
    if (!rows.length) {
        body.innerHTML = '<tr><td colspan="7">Waiting for the next autopilot run…</td></tr>';
        return;
    }
    const fmtPct = v => (v == null ? '—' : `${Number(v).toFixed(2)}%`);
    const fmtRsi = v => (v == null ? '—' : Number(v).toFixed(2));
    const fmtPrice = v => (v == null ? '—' : `$${Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })}`);
    body.innerHTML = rows.map(entry => {
        const time = entry.ts ? new Date(entry.ts).toLocaleTimeString() : '—';
        const action = escapeHtml(entry.action || '—');
        const note = escapeHtml(entry.note || '—');
        const trend = fmtPct(entry.trend_pct);
        const vol = fmtPct(entry.volatility_pct);
        const rsi = fmtRsi(entry.rsi);
        const price = fmtPrice(entry.price);
        return `
            <tr>
                <td>${escapeHtml(time)}</td>
                <td>${action}</td>
                <td>${note}</td>
                <td>${escapeHtml(trend)}</td>
                <td>${escapeHtml(vol)}</td>
                <td>${escapeHtml(rsi)}</td>
                <td>${escapeHtml(price)}</td>
            </tr>
        `;
    }).join('');
}

function updateAutopilotPanel(auto) {
    const stateBadge = document.getElementById('auto-state');
    if (!stateBadge) return;
    const running = !!(auto && auto.running);
    stateBadge.textContent = running ? 'Running' : 'Idle';
    stateBadge.className = running ? 'badge badge-on' : 'badge badge-off';
    const startBtn = document.getElementById('start-autopilot');
    const stopBtn = document.getElementById('stop-autopilot');
    if (startBtn) startBtn.disabled = running;
    if (stopBtn) stopBtn.disabled = !running;
    const set = (id, val) => {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
    };
    let signalLabel = 'Idle';
    if (running) {
        if (auto?.last_signal && auto.last_signal !== 'stopped') {
            signalLabel = auto.last_signal.toUpperCase();
        } else {
            signalLabel = 'Watching';
        }
    }
    set('auto-last-signal', signalLabel);
    const note = running
        ? (auto?.last_reason || 'Waiting for EMA / RSI alignment')
        : '—';
    set('auto-last-reason', note);
    const pollSeconds = auto?.config?.poll_seconds;
    if (running && pollSeconds) {
        const cadence = `${pollSeconds}s cadence`;
        if (auto?.last_run) {
            const last = new Date(auto.last_run);
            if (!Number.isNaN(last.getTime())) {
                const next = new Date(last.getTime() + pollSeconds * 1000);
                set('auto-next-poll', `${cadence} · next ~${next.toLocaleTimeString()}`);
            } else {
                set('auto-next-poll', cadence);
            }
        } else {
            set('auto-next-poll', cadence);
        }
    } else if (pollSeconds) {
        set('auto-next-poll', `${pollSeconds}s cadence (configured)`);
    } else {
        set('auto-next-poll', '—');
    }
    const decision = auto?.last_decision || {};
    set('auto-trend', decision.trend_pct != null ? `${decision.trend_pct.toFixed(2)}%` : '—');
    set('auto-vol', decision.volatility_pct != null ? `${decision.volatility_pct.toFixed(2)}%` : '—');
    set('auto-rsi', decision.rsi != null ? decision.rsi.toFixed(2) : '—');
    set('auto-price', decision.price != null ? fmt(decision.price, 2) : '—');
    const applied = auto?.applied_ladder;
    if (applied) {
        const summary = `${applied.direction} · ${applied.steps} steps @ $${Number(applied.interval).toFixed(2)} (size ${Number(applied.size).toFixed(6)})`;
        set('auto-applied', summary);
    } else {
        set('auto-applied', '—');
    }
    set('auto-last-run', auto?.last_run ? new Date(auto.last_run).toLocaleTimeString() : '—');
    const errEl = document.getElementById('auto-error');
    if (errEl) {
        const hasError = !!auto?.last_error;
        errEl.textContent = hasError ? auto.last_error : '—';
        errEl.classList.toggle('negative', hasError);
    }
    const cfgEl = document.getElementById('auto-config');
    if (cfgEl) {
        cfgEl.textContent = auto?.config ? JSON.stringify(auto.config, null, 2) : '{}';
    }
    renderAutopilotHistory(auto);
}

let lastPriceCache = 0;
async function refreshStatus() {
    try {
        const st = await apiGet('status');
        document.getElementById('status-box').textContent =
            JSON.stringify(st, null, 2);

        // Badge and buttons
        const running = !!st.running;
        const badge = document.getElementById('bot-state');
        badge.textContent = running ? 'Running' : 'Stopped';
        badge.className = running ? 'badge badge-on' : 'badge badge-off';
        document.getElementById('start-btn').disabled = running;
        document.getElementById('stop-btn').disabled = !running;

        // Metrics
        setText('m-position', fmt(st.position_qty, 6));
        setText('m-avgprice', fmt(st.avg_price, 2));
        setText('m-pnl', fmt(st.realized_pnl, 2));
        setText('m-openorders', st.open_orders.length);

        // Last price from status (backend) or fallback via bars
        if (st.last_price) { lastPriceCache = st.last_price; }
        else {
            try {
                const lp = await apiGet('bars', { symbols: st.config?.symbol || 'BTC/USD', timeframe: '1Min', limit: 1 });
                const barCont = lp.bars || {};
                const arr = Array.isArray(barCont) ? barCont : (barCont[st.config?.symbol || 'BTC/USD'] || Object.values(barCont)[0] || []);
                lastPriceCache = arr?.[0]?.c ?? lastPriceCache;
            } catch {}
        }
        setText('m-last', fmt(lastPriceCache, 2));
        const upnlVal = st.unrealized_pnl_usd ?? ((st.position_qty || 0) * ((lastPriceCache || 0) - (st.avg_price || 0)));
        const upnlEl = document.getElementById('m-upnl');
        setText('m-upnl', fmt(upnlVal, 2));
        setClass(upnlEl, upnlVal >= 0 ? 'positive' : 'negative');

        // Capital used and max notional estimates
        const steps = st.config?.steps || 0;
        const size = st.config?.size || 0;
        const maxExposureBtc = st.config?.max_exposure || 0;
        const maxNotional = (steps * size) * (lastPriceCache || 0);
        const capitalUsed = (st.position_qty || 0) * (lastPriceCache || 0);
        const remaining = Math.max(0, maxNotional - capitalUsed);
        setText('m-capused', fmt(capitalUsed, 2));
        setText('m-maxnotional', fmt(maxNotional, 2));
        setText('m-remaining', fmt(remaining, 2));
        const realized = Number.isFinite(st.realized_pnl) ? st.realized_pnl : 0;
        const netPnl = realized + (Number.isFinite(upnlVal) ? upnlVal : 0);
        const point = {
            t: new Date(),
            capital: Number.isFinite(capitalUsed) ? capitalUsed : 0,
            pnl: Number.isFinite(netPnl) ? netPnl : 0
        };
        autopilotHistory.push(point);
        if (autopilotHistory.length > 240) autopilotHistory.shift();
        const chart = ensureAutopilotChart();
        if (chart) {
            chart.data.datasets[0].data = autopilotHistory.map(d => ({ x: d.t, y: d.capital }));
            chart.data.datasets[1].data = autopilotHistory.map(d => ({ x: d.t, y: d.pnl }));
            chart.update('none');
        }
        // highlight if near exposure cap
        if (st.position_qty >= maxExposureBtc * 0.95) {
            document.getElementById('m-position').classList.add('negative');
        } else {
            document.getElementById('m-position').classList.remove('negative');
        }

        // Risk banner
        const lossEnabled = document.getElementById('risk-enable-loss')?.checked;
        const cap = Number(document.getElementById('risk-losscap')?.value || 0);
        const volEnabled = document.getElementById('risk-enable-vol')?.checked;
        const vol = Number(document.getElementById('risk-vol')?.value || 0);
        const dailyOK = !lossEnabled || (st.realized_pnl + upnlVal >= -Math.abs(cap));
        const percMove = st.avg_price ? Math.abs((lastPriceCache - st.avg_price)/st.avg_price)*100 : 0;
        const volOK = !volEnabled || percMove <= Math.abs(vol);
        const banner = document.getElementById('last-action');
        if (banner) {
            banner.textContent = (dailyOK && volOK)
                ? `OK: position ${fmt(st.position_qty,6)} BTC, ${st.open_orders.length} open rungs`
                : `PAUSED: ${!dailyOK?`Daily loss cap hit (${fmt(st.realized_pnl+upnlVal,2)})`:''} ${!volOK?`Vol ${percMove.toFixed(2)}% > ${vol}%`:''}`.trim();
        }

        // Side panel snapshot
        setText('sp-open', st.open_orders.length);
        setText('sp-position', fmt(st.position_qty,6));
        setText('sp-avg', fmt(st.avg_price,2));
        setText('sp-last', fmt(lastPriceCache,2));
        const spUpnl = document.getElementById('sp-upnl');
        if (spUpnl) { spUpnl.textContent = fmt(upnlVal,2); setClass(spUpnl, upnlVal>=0?'positive':'negative'); }
        setText('sp-rpnl', fmt(st.realized_pnl,2));
        setText('sp-used', fmt(capitalUsed,2));
        setText('sp-remaining', fmt(remaining,2));
        const rungMin = document.getElementById('preview-min')?.textContent || '-';
        const rungMax = document.getElementById('preview-max')?.textContent || '-';
        setText('sp-range', `$${rungMin} – $${rungMax}`);
        setText('sp-action', st.last_action || '-');

        // Top PnL & Funds Strip
        setText('strip-pos', fmt(st.position_qty, 6));
        setText('strip-avg', fmt(st.avg_price, 2));
        const upEl = document.getElementById('strip-upnl');
        if (upEl) { upEl.textContent = fmt(upnlVal, 2); setClass(upEl, upnlVal>=0?'positive':'negative'); }
        const pct = (st.unrealized_pnl_pct != null) ? st.unrealized_pnl_pct : ((st.avg_price? ((lastPriceCache - st.avg_price)/st.avg_price*100):0));
        setText('strip-upct', fmt(pct, 2));
        const day = (st.day_realized_pnl_usd != null) ? st.day_realized_pnl_usd : 0;
        const dayEl = document.getElementById('strip-day'); if (dayEl) { dayEl.textContent = fmt(day,2); setClass(dayEl, day>=0?'positive':'negative'); }
        setText('strip-used', fmt(st.capital_used ?? capitalUsed, 2));
        setText('strip-cap', fmt(st.capacity_remaining ?? remaining, 2));
        setText('strip-open', st.open_order_count ?? st.open_orders.length);

        // Account funds (cash/equity/buying power)
        try {
            const acct = await apiGet('account');
            setText('strip-cash', fmt(acct.cash ?? 0, 2));
            setText('strip-equity', fmt(acct.equity ?? acct.portfolio_value ?? 0, 2));
            setText('strip-bp', fmt(acct.buying_power ?? 0, 2));
        } catch {}

        // Tables
        renderOrders('open-orders', st.open_orders);
        renderOrders('filled-orders', st.filled_orders);

        // Status chips: bot state, flat/position, orders present, risk
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
async function refreshLogs() {
    try {
        const L = await apiGet('logs', {tail: 200});
        const lines = L.logs || [];
        document.getElementById('log-box').textContent = lines.join('\n');
        renderActivity(lines);
    } catch (e) {
        document.getElementById('log-box').textContent = 'Log error: '+e;
    }
}
setInterval(refreshStatus, 5000);
setInterval(refreshLogs, 7000);
refreshStatus();
refreshLogs();
ensureAutopilotChart();
