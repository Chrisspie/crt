// CRT Scanner – Web (client-only)
// Ports core logic from the Python Streamlit app to JS.

(function () {
  // ----- Constants & Config -----
  const WIG20_HC_TICKERS = [
    "PKN","KGH","PKO","PEO","PZU","PGE","CDR","ALE","DNP","LPP","OPL","CPS","ALR","ING","MBK","TPE","JSW","CCC","KTY"
  ];
  const MWIG40_HC_TICKERS = [
    "XTB","PLW","TEN","KRU","GPW","BDX","BHW","LWB","AMC","ASB","11B","CIG","AFR","MLG","STP","PKP","MAB","NEU","OPN","VRG","WPL","WRT","DOM","MRC","PHN","PEK","IPF","TIM","MFO","PBX","BRS","FTE","BUD","APS","DVR","TOR","CIGAMES","LIVE","AMX","DIN"
  ];
  const SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","BRK-B","AVGO","TSLA","UNH","LLY","XOM","JPM","V","JNJ","PG","MA","HD","COST","MRK","ABBV","PEP","KO","TMO","BAC","WMT","ADBE","NFLX","CRM","CSCO","INTC","QCOM","NKE","LIN","ACN","MCD","ORCL","TXN","AMD"
  ];
  const GPW_ALIASES = { DIN: "DNP", BUD: "BDX", AMX: "AMC", CIGAMES: "CIG", LIVE: "LVC", PEK: "PBX" };

  const UI = id => document.getElementById(id);

  // Simple in-memory caches for fetched data (reset by Clear cache)
  const cache = {
    weekly: new Map(), // key: ticker, value: array of {t, o, h, l, c}
    htf: new Map(), // key: `${ticker}|${interval}`, value: array of {t, o, h, l, c}
    sp500: null, // cached list of {company, yahoo_ticker, group}
    swig80: null,
  };

  // ----- Utilities -----
  function fmtDate(d) {
    if (!d) return "";
    const dt = (d instanceof Date) ? d : new Date(d);
    if (Number.isNaN(dt.getTime())) return "";
    return dt.toISOString().slice(0, 10);
  }
  function toFixedOrNa(v, n = 2) {
    return (v == null || Number.isNaN(v)) ? "" : Number(v).toFixed(n);
  }
  function csvEscape(v) {
    if (v == null) return "";
    const s = String(v);
    if (s.includes("\n") || s.includes(";") || s.includes('"')) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  }
  function downloadCSV(rows, filename) {
    if (!rows || !rows.length) return;
    const cols = Object.keys(rows[0]);
    const lines = [cols.join(";")].concat(
      rows.map(r => cols.map(c => csvEscape(r[c])).join(";"))
    );
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = UI("btn_download_csv");
    a.href = url;
    a.download = filename;
    a.style.display = "inline-block";
  }

  function limitConcurrency(tasks, limit = 6) {
    return new Promise((resolve) => {
      let i = 0; let active = 0; const results = new Array(tasks.length);
      const runNext = () => {
        if (i >= tasks.length && active === 0) { resolve(results); return; }
        while (active < limit && i < tasks.length) {
          const idx = i++; const fn = tasks[idx]; active++;
          Promise.resolve().then(fn).then(res => { results[idx] = res; })
            .catch(err => { results[idx] = { error: String(err) }; })
            .finally(() => { active--; runNext(); });
        }
      };
      runNext();
    });
  }

  // ----- Universe building -----
  function _fixGPWSymbol(sym) {
    let s = String(sym || "").trim().toUpperCase();
    s = s.replace(/\s+/g, "");
    s = s.replace(/[^A-Z0-9\.\^]/g, "");
    if (GPW_ALIASES[s]) s = GPW_ALIASES[s];
    return s;
  }
  function parseUSTickers(raw) {
    if (!raw) return [];
    const parts = String(raw).replace(/;/g, ',').split(',').map(p => p.trim().toUpperCase()).filter(Boolean);
    return parts.map(sym => ({ company: sym, yahoo_ticker: sym, group: 'US (manual)', Active: true }));
  }
  function parseGPWTickers(raw) {
    if (!raw) return [];
    const parts = String(raw).replace(/;/g, ',').split(',').map(p => p.trim()).filter(Boolean).map(_fixGPWSymbol).filter(Boolean);
    const toYF = (s) => s.startsWith('^') || s.includes('.') ? s : `${s}.WA`;
    return parts.map(sym => ({ company: sym, yahoo_ticker: toYF(sym), group: 'GPW (manual)', Active: true }));
  }
  function uniqBy(arr, keyFn) {
    const seen = new Set(); const out = [];
    for (const x of arr) { const k = keyFn(x); if (!seen.has(k)) { seen.add(k); out.push(x); } }
    return out;
  }
  function hardcodedWIG20() {
    return WIG20_HC_TICKERS.slice().sort().map(t => ({ company: t, yahoo_ticker: `${t}.WA`, group: 'WIG20', Active: true }));
  }
  function hardcodedMWIG40() {
    return MWIG40_HC_TICKERS.slice().sort().map(t => ({ company: t, yahoo_ticker: `${t}.WA`, group: 'mWIG40', Active: true }));
  }
  async function fetchSP500Companies() {
    if (cache.sp500) return cache.sp500;
    try {
      const res = await fetch('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', { cache: 'force-cache' });
      const html = await res.text();
      const doc = new DOMParser().parseFromString(html, 'text/html');
      const table = doc.querySelector('table.wikitable');
      const rows = table ? Array.from(table.querySelectorAll('tbody tr')) : [];
      const out = [];
      for (const tr of rows) {
        const tds = tr.querySelectorAll('td');
        if (tds.length < 2) continue;
        const sym = tds[0].textContent.trim().toUpperCase().replace(/\./g, '-');
        const name = tds[1].textContent.trim();
        if (sym && /^[A-Z0-9\-]+$/.test(sym)) out.push({ company: name, yahoo_ticker: sym, group: 'S&P500', Active: true });
      }
      cache.sp500 = out.length ? out : SP500_FALLBACK.map(s => ({ company: s, yahoo_ticker: s, group: 'S&P500', Active: true }));
      return cache.sp500;
    } catch (e) {
      cache.sp500 = SP500_FALLBACK.map(s => ({ company: s, yahoo_ticker: s, group: 'S&P500', Active: true }));
      return cache.sp500;
    }
  }
  async function fetchSWIG80() {
    if (cache.swig80) return cache.swig80;
    const urls = ['https://pl.wikipedia.org/wiki/SWIG80', 'https://en.wikipedia.org/wiki/SWIG80'];
    for (const url of urls) {
      try {
        const res = await fetch(url, { cache: 'force-cache' });
        const html = await res.text();
        const doc = new DOMParser().parseFromString(html, 'text/html');
        const tables = Array.from(doc.querySelectorAll('table')); let table = null;
        for (const t of tables) {
          const head = t.querySelectorAll('th');
          const cols = Array.from(head).map(th => th.textContent.trim().toLowerCase());
          if (cols.some(c => /ticker|symbol|kod/.test(c))) { table = t; break; }
        }
        if (!table) continue;
        const out = [];
        for (const tr of table.querySelectorAll('tbody tr')) {
          const tds = tr.querySelectorAll('td');
          if (tds.length < 2) continue;
          const name = tds[0].textContent.trim();
          let sym = tds[1].textContent.trim().toUpperCase();
          sym = sym.replace(/\s+/g, '').replace(/[^A-Z0-9]/g, '');
          if (!sym) continue;
          out.push({ company: name, yahoo_ticker: `${sym}.WA`, group: 'sWIG80', Active: true });
        }
        cache.swig80 = out; return out;
      } catch (e) { /* try next */ }
    }
    cache.swig80 = [];
    return cache.swig80;
  }
  async function getWIGAll() {
    const parts = [hardcodedWIG20(), hardcodedMWIG40(), await fetchSWIG80()];
    const tickers = parts.flat().map(x => ({ company: x.company || x.yahoo_ticker.replace('.WA',''), yahoo_ticker: x.yahoo_ticker, group: 'WIG', Active: true }));
    return uniqBy(tickers, x => x.yahoo_ticker);
  }

  async function buildUniverse(opts) {
    const frames = [];
    if (opts.use_wig_all) frames.push(...await getWIGAll());
    if (opts.use_wig20) frames.push(...hardcodedWIG20());
    if (opts.use_mwig40) frames.push(...hardcodedMWIG40());
    const gpw = parseGPWTickers(opts.gpw_raw);
    if (gpw.length) frames.push(...gpw);
    if (opts.use_sp500) frames.push(...await fetchSP500Companies());
    const us = parseUSTickers(opts.raw_us);
    if (us.length) frames.push(...us);
    let uni = uniqBy(frames, x => x.yahoo_ticker);
    for (const u of uni) {
      if (!u.company) u.company = u.yahoo_ticker.replace('.WA','');
      if (typeof u.Active !== 'boolean') u.Active = true;
    }
    uni.sort((a,b) => (a.group.localeCompare(b.group) || a.yahoo_ticker.localeCompare(b.yahoo_ticker)));
    return uni;
  }

  // ----- Yahoo Finance fetch -----
  // Returns array of {t: Date, o,h,l,c}
  async function fetchYahooOHLC(ticker, { interval = '1wk', range = '5y', start = null } = {}) {
    const key = start ? `${ticker}|${interval}|${start}` : `${ticker}|${interval}|${range}`;
    const targetCache = interval === '1mo' || interval === '3mo' ? cache.htf : cache.weekly;
    if (targetCache.has(key)) return targetCache.get(key);
    const url = new URL(`https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}`);
    if (start) {
      const period1 = Math.floor(new Date(start).getTime() / 1000);
      url.searchParams.set('period1', String(period1));
      url.searchParams.set('period2', String(Math.floor(Date.now()/1000)));
    } else {
      url.searchParams.set('range', range);
    }
    url.searchParams.set('interval', interval);
    url.searchParams.set('includePrePost', 'false');
    url.searchParams.set('events', '');
    const res = await fetch(url.toString());
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const r = data && data.chart && data.chart.result && data.chart.result[0];
    if (!r || !r.timestamp || !r.indicators || !r.indicators.quote || !r.indicators.quote[0]) return [];
    const ts = r.timestamp.map(t => new Date(t * 1000));
    const q = r.indicators.quote[0];
    const out = [];
    for (let i = 0; i < ts.length; i++) {
      const o = q.open[i], h = q.high[i], l = q.low[i], c = q.close[i];
      if ([o,h,l,c].every(v => typeof v === 'number' && !Number.isNaN(v))) out.push({ t: ts[i], o, h, l, c });
    }
    targetCache.set(key, out);
    return out;
  }

  // ----- CRT core (port from Python) -----
  function midline(low, high) { return (Number(low) + Number(high)) / 2.0; }
  function crtScan(dfBars, { lookback_bars = 30, require_midline = false, strict_vs_c1open = false, confirm_within = 0, confirm_method = 'high', directions = ['bullish','bearish'] } = {}) {
    const out = [];
    const d = dfBars.filter(b => [b.o,b.h,b.l,b.c].every(v => typeof v === 'number' && !Number.isNaN(v)));
    if (d.length < 5) return out;
    const n = d.length; const startIdx = Math.max(1, n - lookback_bars);
    const findC3 = (i_c2, c1_low, c1_high, method, dirTag) => {
      let c3_any_idx = null;
      for (let j = i_c2 + 1; j < n; j++) {
        const H = d[j].h, L = d[j].l, C = d[j].c;
        const cond_any = dirTag === 'BULL' ? (method === 'high' ? H > c1_high : C > c1_high) : (method === 'high' ? L < c1_low : C < c1_low);
        if (cond_any) { c3_any_idx = j; break; }
      }
      return [null, c3_any_idx];
    };
    for (let i = startIdx; i < n; i++) {
      const C1 = d[i-1], C2 = d[i];
      const C1L = C1.l, C1H = C1.h; const C2L = C2.l, C2H = C2.h, C2C = C2.c, C1O = C1.o;
      const C1_mid = midline(C1L, C1H); const close_in = (C1L <= C2C && C2C <= C1H);
      const record = (direction, swept_side) => {
        const dirTag = direction === 'bullish' ? 'BULL' : 'BEAR';
        let [c3_within_idx, c3_any_idx] = findC3(i, C1L, C1H, confirm_method, dirTag);
        let confirmed_within = false;
        if (confirm_within && (i + 1) < n) {
          const end_j = Math.min(n - 1, i + confirm_within);
          for (let j = i + 1; j <= end_j; j++) {
            const H = d[j].h, L = d[j].l, C = d[j].c;
            const cond = dirTag === 'BULL' ? (confirm_method === 'high' ? H > C1H : C > C1H) : (confirm_method === 'high' ? L < C1L : C < C1L);
            if (cond) { confirmed_within = true; c3_within_idx = j; break; }
          }
        }
        let confirm_rule = 'no confirm';
        if (confirm_within) {
          const rule = (confirm_method === 'high') ? (dirTag === 'BULL' ? 'high>C1H' : 'low<C1L') : (dirTag === 'BULL' ? 'close>C1H' : 'close<C1L');
          confirm_rule = `${rule} in ${confirm_within}`;
        }
        out.push({
          direction: dirTag, C1_date: C1.t, C2_date: C2.t,
          C3_date_within: c3_within_idx != null ? d[c3_within_idx].t : null,
          C3_date_any: c3_any_idx != null ? d[c3_any_idx].t : null,
          confirmed: !!confirmed_within, c3_happened: !!(c3_any_idx != null), confirm_rule,
          C1_low: C1L, C1_high: C1H, C1_mid, C1_open: C1O,
          C2_low: C2L, C2_high: C2H, C2_close: C2C,
          C2_position_in_range: (C1H > C1L) ? ((C2C - C1L) / (C1H - C1L)) : NaN,
          swept_side,
        });
      };
      if (directions.includes('bullish')) {
        let cond = (C2L < C1L) && close_in;
        if (require_midline) cond = cond && (C2C >= C1_mid);
        if (strict_vs_c1open) cond = cond && (C2C >= C1O);
        if (cond) record('bullish', 'LOW');
      }
      if (directions.includes('bearish')) {
        let cond = (C2H > C1H) && close_in;
        if (require_midline) cond = cond && (C2C <= C1_mid);
        if (strict_vs_c1open) cond = cond && (C2C <= C1O);
        if (cond) record('bearish', 'HIGH');
      }
    }
    out.sort((a,b) => new Date(b.C2_date) - new Date(a.C2_date));
    return out;
  }

  function getKeyLevelAndConfluence(htfBars, c2_ts, direction, c1_low, c1_high, c2_low, c2_high, key_window_months, key_interact, key_strict, htf_interval) {
    if (!htfBars || !htfBars.length || !c2_ts) return ['-', NaN, null, false];
    const barsPer = (htf_interval === '1mo') ? 1 : 3;
    const nBars = Math.max(1, Math.ceil(key_window_months / barsPer));
    const hist = htfBars.filter(b => b.t <= c2_ts);
    if (!hist.length) return ['-', NaN, null, false];
    const win = hist.slice(Math.max(0, hist.length - nBars));
    if (direction === 'BULL') {
      let keyLevelVal = Math.min(...win.map(b => b.l));
      let keyIdx = win.reduce((acc, b, i) => (b.l < win[acc].l ? i : acc), 0);
      let keyDate = win[keyIdx].t;
      const value = (key_interact === 'Tylko C1') ? Number(c1_low) : (key_interact === 'Tylko C2' ? Number(c2_low) : Math.min(Number(c1_low), Number(c2_low)));
      const confluence = key_strict.startsWith('strict') ? (value < keyLevelVal) : (value <= keyLevelVal);
      return [(htf_interval === '1mo' ? '1M' : '3M'), keyLevelVal, keyDate, confluence];
    } else {
      let keyLevelVal = Math.max(...win.map(b => b.h));
      let keyIdx = win.reduce((acc, b, i) => (b.h > win[acc].h ? i : acc), 0);
      let keyDate = win[keyIdx].t;
      const value = (key_interact === 'Tylko C1') ? Number(c1_high) : (key_interact === 'Tylko C2' ? Number(c2_high) : Math.max(Number(c1_high), Number(c2_high)));
      const confluence = key_strict.startsWith('strict') ? (value > keyLevelVal) : (value >= keyLevelVal);
      return [(htf_interval === '1mo' ? '1M' : '3M'), keyLevelVal, keyDate, confluence];
    }
  }

  // ----- Rendering helpers -----
  function renderUniverseTable(universe) {
    const mount = UI('universe_table'); mount.innerHTML = '';
    if (!universe.length) { mount.textContent = 'Brak spółek do skanowania.'; return; }
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    thead.innerHTML = `<tr><th>Aktywna</th><th>Spółka</th><th>Ticker</th><th>Grupa</th></tr>`;
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    universe.forEach((u, idx) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><input type="checkbox" data-idx="${idx}" ${u.Active ? 'checked' : ''}></td>
        <td>${u.company}</td>
        <td>${u.yahoo_ticker}</td>
        <td>${u.group}</td>
      `;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    mount.appendChild(table);
    tbody.addEventListener('change', (e) => {
      const t = e.target; if (t && t.matches('input[type=checkbox][data-idx]')) {
        const i = Number(t.getAttribute('data-idx')); universe[i].Active = t.checked; updateActiveCount(universe);
      }
    });
    updateActiveCount(universe);
  }
  function updateActiveCount(universe) {
    const cnt = universe.filter(u => u.Active).length;
    UI('active_count').textContent = `Aktywnych tickerów: ${cnt} / ${universe.length}`;
  }
  function renderResultsTable(rows) {
    const mount = UI('results_table'); mount.innerHTML = '';
    if (!rows.length) { mount.textContent = 'Brak wyników dla bieżących ustawień.'; UI('btn_download_csv').style.display = 'none'; return; }
    const cols = [
      'Ticker','Spółka','Grupa','Kierunek','C1','C2','C3_any','Potwierdzenie_w_N','C3_happened','Zasada potwierdzenia',
      'C1L','C1H','Mid(50%)','C1O','C2L','C2H','C2C','C2 pos w C1%','Sweep','Trigger','Stop','TP1','TP2','R:TP1','R:TP2','KeyTF','KeyLevel','KeyDate','Confluence'
    ];
    const table = document.createElement('table');
    const thead = document.createElement('thead');
    thead.innerHTML = `<tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr>`;
    table.appendChild(thead);
    const tbody = document.createElement('tbody');
    for (const r of rows) {
      const tr = document.createElement('tr');
      tr.innerHTML = cols.map(c => `<td>${r[c] ?? ''}</td>`).join('');
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    mount.appendChild(table);
    const filename = `crt_scan_${fmtDate(new Date())}.csv`;
    downloadCSV(rows, filename);
  }

  function setProgress(text) { UI('scan_progress').textContent = text || ''; }
  function setFailedInfo(text) { UI('failed_info').textContent = text || ''; }

  function setChartOptions(tickers) {
    const sel = UI('chart_ticker');
    sel.innerHTML = '';
    tickers.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t; opt.textContent = t; sel.appendChild(opt);
    });
  }

  async function drawChart(ticker, rec) {
    const weekly = await fetchYahooOHLC(ticker, { interval: '1wk', range: 'max' });
    if (!weekly.length) {
      Plotly.purge('chart'); UI('chart').innerHTML = '<div class="muted">Brak danych tygodniowych do rysowania wykresu.</div>'; return;
    }
    const x = weekly.map(b => b.t);
    const fig = [{
      type: 'candlestick', x,
      open: weekly.map(b => b.o), high: weekly.map(b => b.h), low: weekly.map(b => b.l), close: weekly.map(b => b.c), name: ticker
    }];
    const shapes = []; const annotations = [];
    function addH(y, text, dash) {
      if (y == null || Number.isNaN(y)) return;
      shapes.push({ type: 'line', xref: 'x', yref: 'y', x0: x[0], x1: x[x.length-1], y0: y, y1: y, line: { width: 1, dash: dash || 'dot', color: '#9aa3b2' } });
      annotations.push({ x: x[0], y, xref: 'x', yref: 'y', text, showarrow: false, xanchor: 'left', yanchor: 'bottom', font: { size: 10, color: '#9aa3b2' } });
    }
    addH(rec.Trigger, 'Trigger', 'solid');
    addH(rec.Stop, 'SL', 'dot');
    addH(rec.TP1, 'TP1', 'dash');
    addH(rec.TP2, 'TP2', 'dash');
    addH(rec.KeyLevel, 'Key', 'dot');
    // Direction arrow
    const c2Date = rec.C2 ? new Date(rec.C2) : null;
    const last = weekly[weekly.length-1];
    const yBase = c2Date && weekly.some(b => +b.t === +c2Date) ? weekly.find(b => +b.t === +c2Date).c : last.c;
    const xPoint = c2Date && weekly.some(b => +b.t === +c2Date) ? c2Date : last.t;
    if (rec.Kierunek === 'BULL') {
      annotations.push({ x: xPoint, y: yBase, xref: 'x', yref: 'y', text: '↑ BULL', showarrow: true, arrowhead: 2, ax: 0, ay: -60, font: { color: '#22c55e' } });
    } else if (rec.Kierunek === 'BEAR') {
      annotations.push({ x: xPoint, y: yBase, xref: 'x', yref: 'y', text: '↓ BEAR', showarrow: true, arrowhead: 2, ax: 0, ay: 60, font: { color: '#ef4444' } });
    }
    const layout = { margin: { l: 0, r: 0, t: 20, b: 0 }, xaxis: { rangeslider: { visible: false } }, height: 520, shapes, annotations, paper_bgcolor: '#171a2b', plot_bgcolor: '#171a2b', font: { color: '#e4e7f1' } };
    Plotly.newPlot('chart', fig, layout, { displayModeBar: false, responsive: true });
  }

  // ----- Main handlers -----
  let universe = [];
  let lastRows = [];

  async function handleBuildUniverse() {
    const opts = {
      use_wig_all: UI('use_wig_all').checked,
      use_wig20: UI('use_wig20').checked,
      use_mwig40: UI('use_mwig40').checked,
      use_sp500: UI('use_sp500').checked,
      gpw_raw: UI('gpw_raw').value,
      raw_us: UI('raw_us').value,
    };
    UI('btn_build_universe').disabled = true;
    universe = await buildUniverse(opts);
    UI('btn_build_universe').disabled = false;
    renderUniverseTable(universe);
  }

  function requireActiveTickers() {
    const active = universe.filter(u => u.Active).map(u => u.yahoo_ticker);
    if (!active.length) throw new Error('Zaznacz przynajmniej jedną spółkę.');
    return active;
  }

  async function scan() {
    try {
      const activeTickers = requireActiveTickers();
      const opportunity_mode = UI('opportunity_mode').checked;
      const confirm_on = UI('confirm_on').checked;
      const confirm_within = Number(UI('confirm_within').value);
      const confirm_method = UI('confirm_method').value;
      const period = UI('period').value;
      const dirSel = UI('directions').value;
      const directions = dirSel === 'both' ? ['bullish','bearish'] : [dirSel];
      const lookback_bars = Number(UI('lookback_bars').value);
      const require_midline = UI('require_midline').checked;
      const strict_vs_c1open = UI('strict_vs_c1open').checked;
      const key_on = UI('key_on').checked;
      const key_tf_label = UI('key_tf_label').value; // '1mo' | '3mo'
      const key_window_months = Number(UI('key_window_months').value);
      const key_interact = UI('key_interact').value;
      const key_strict = UI('key_strict').value; // 'strict' | 'touch'
      const key_require = UI('key_require').checked;

      setFailedInfo(''); setProgress('Analiza...'); UI('btn_scan').disabled = true;
      let effective_lookback = lookback_bars; let start = null;
      if (opportunity_mode) {
        const effectiveWeeks = 2 + (confirm_on ? confirm_within : 0) + 6;
        const startDate = new Date(); startDate.setDate(startDate.getDate() - 7 * effectiveWeeks); start = fmtDate(startDate);
        effective_lookback = 3;
      }

      const rows = []; const failed = [];
      let done = 0; const total = activeTickers.length;
      const tasks = activeTickers.map(yt => async () => {
        setProgress(`Skanowanie: ${yt} (${++done}/${total})`);
        try {
          const weekly = await fetchYahooOHLC(yt, start ? { interval: '1wk', start } : { interval: '1wk', range: period });
          if (!weekly || weekly.length < 5) { failed.push(yt); return; }
          const setups = crtScan(weekly, { lookback_bars: effective_lookback, require_midline, strict_vs_c1open, confirm_within: (confirm_on ? confirm_within : 0), confirm_method, directions });
          const lastTwoDates = weekly.slice(-2).map(b => +b.t);
          let htfBars = [];
          if (key_on) {
            const htfInterval = key_tf_label; // '1mo' or '3mo'
            htfBars = await fetchYahooOHLC(yt, { interval: htfInterval, range: 'max' });
          }
          for (const rec of setups) {
            const c2Ts = rec.C2_date ? new Date(rec.C2_date) : null;
            const c2Millis = c2Ts ? +c2Ts : null;
            if (opportunity_mode) {
              const c3h = !!rec.c3_happened;
              if (!(c2Millis && lastTwoDates.includes(c2Millis)) || c3h) continue;
            }
            const C1L = rec.C1_low, C1H = rec.C1_high, C2L = rec.C2_low, C2H = rec.C2_high, C2C = rec.C2_close;
            const rng = (isFinite(C1H) && isFinite(C1L)) ? (C1H - C1L) : NaN;
            let trigger, stop, tp1, tp2, risk, r_tp1, r_tp2;
            if (rec.direction === 'BULL') {
              trigger = C1H; stop = C2L;
              tp1 = isFinite(rng) ? C1H + 0.5*rng : NaN;
              tp2 = isFinite(rng) ? C1H + 1.0*rng : NaN;
              risk = (isFinite(trigger) && isFinite(stop)) ? (trigger - stop) : NaN;
              r_tp1 = (isFinite(risk) && risk>0 && isFinite(tp1)) ? ((tp1 - trigger)/risk) : NaN;
              r_tp2 = (isFinite(risk) && risk>0 && isFinite(tp2)) ? ((tp2 - trigger)/risk) : NaN;
            } else {
              trigger = C1L; stop = C2H;
              tp1 = isFinite(rng) ? C1L - 0.5*rng : NaN;
              tp2 = isFinite(rng) ? C1L - 1.0*rng : NaN;
              risk = (isFinite(trigger) && isFinite(stop)) ? (stop - trigger) : NaN;
              r_tp1 = (isFinite(risk) && risk>0 && isFinite(tp1)) ? ((trigger - tp1)/risk) : NaN;
              r_tp2 = (isFinite(risk) && risk>0 && isFinite(tp2)) ? ((trigger - tp2)/risk) : NaN;
            }
            let keyTF = '-', keyLevelVal = NaN, keyDate = null, confluence = false;
            if (key_on && htfBars && htfBars.length) {
              const [tf, lvl, kdate, conf] = getKeyLevelAndConfluence(htfBars, c2Ts, rec.direction, C1L, C1H, C2L, C2H, key_window_months, key_interact, key_strict, key_tf_label);
              keyTF = tf; keyLevelVal = lvl; keyDate = kdate; confluence = !!conf;
            }
            if (key_on && key_require && !confluence) continue;
            const meta = universe.find(u => u.yahoo_ticker === yt) || { company: yt.replace('.WA',''), group: '' };
            rows.push({
              'Ticker': yt,
              'Spółka': meta.company,
              'Grupa': meta.group,
              'Kierunek': rec.direction,
              'C1': fmtDate(rec.C1_date),
              'C2': fmtDate(rec.C2_date),
              'C3_any': rec.C3_date_any ? fmtDate(rec.C3_date_any) : '',
              'Potwierdzenie_w_N': rec.confirmed ? 'TAK' : 'NIE',
              'C3_happened': rec.c3_happened ? 'TAK' : 'NIE',
              'Zasada potwierdzenia': rec.confirm_rule,
              'C1L': toFixedOrNa(C1L), 'C1H': toFixedOrNa(C1H),
              'Mid(50%)': toFixedOrNa(rec.C1_mid), 'C1O': toFixedOrNa(rec.C1_open),
              'C2L': toFixedOrNa(C2L), 'C2H': toFixedOrNa(C2H), 'C2C': toFixedOrNa(C2C),
              'C2 pos w C1%': isFinite(rec.C2_position_in_range) ? (100*rec.C2_position_in_range).toFixed(1) : '',
              'Sweep': rec.swept_side,
              'Trigger': isFinite(trigger) ? toFixedOrNa(trigger) : '',
              'Stop': isFinite(stop) ? toFixedOrNa(stop) : '',
              'TP1': isFinite(tp1) ? toFixedOrNa(tp1) : '',
              'TP2': isFinite(tp2) ? toFixedOrNa(tp2) : '',
              'R:TP1': isFinite(r_tp1) ? toFixedOrNa(r_tp1) : '',
              'R:TP2': isFinite(r_tp2) ? toFixedOrNa(r_tp2) : '',
              'KeyTF': keyTF,
              'KeyLevel': isFinite(keyLevelVal) ? toFixedOrNa(keyLevelVal) : '',
              'KeyDate': keyDate ? fmtDate(keyDate) : '',
              'Confluence': key_on ? (confluence ? 'TAK' : 'NIE') : '-',
              __sortC2: c2Ts ? +c2Ts : 0,
            });
          }
        } catch (err) {
          failed.push(yt);
        }
      });
      await limitConcurrency(tasks, 6);
      setProgress('');
      if (failed.length) setFailedInfo(`Brak danych dla: ${failed.slice(0,40).join(', ')}${failed.length>40?'…':''}`);
      // Sort rows by C2 desc, Group asc, Ticker asc
      rows.sort((a,b) => (b.__sortC2 - a.__sortC2) || (a['Grupa'].localeCompare(b['Grupa'])) || (a['Ticker'].localeCompare(b['Ticker'])));
      rows.forEach(r => delete r.__sortC2);
      lastRows = rows;
      renderResultsTable(rows);
      // Populate chart select with valid tickers
      const valid = Array.from(new Set(rows.filter(r => r['Kierunek']==='BULL' || r['Kierunek']==='BEAR').map(r => r['Ticker'])));
      const chartList = valid.length ? valid : Array.from(new Set(rows.map(r => r['Ticker'])));
      setChartOptions(chartList);
      if (chartList.length) {
        const sel = UI('chart_ticker');
        const selTicker = sel.value || chartList[0];
        const topRec = rows.filter(r => r['Ticker'] === selTicker).sort((a,b) => (new Date(b['C2']) - new Date(a['C2'])))[0];
        if (topRec) await drawChart(selTicker, topRec);
      }
    } catch (e) {
      setProgress('');
      alert(e.message || String(e));
    } finally {
      UI('btn_scan').disabled = false;
    }
  }

  function clearCache() {
    cache.weekly.clear(); cache.htf.clear(); setFailedInfo(''); setProgress('');
    UI('btn_download_csv').style.display = 'none';
  }

  // ----- Wire up UI -----
  function init() {
    UI('btn_build_universe').addEventListener('click', handleBuildUniverse);
    UI('btn_clear_cache').addEventListener('click', clearCache);
    UI('btn_select_all').addEventListener('click', () => { for (const u of universe) u.Active = true; renderUniverseTable(universe); });
    UI('btn_unselect_all').addEventListener('click', () => { for (const u of universe) u.Active = false; renderUniverseTable(universe); });
    UI('btn_scan').addEventListener('click', scan);
    UI('chart_ticker').addEventListener('change', async (e) => {
      const ticker = e.target.value;
      const rec = lastRows.filter(r => r['Ticker'] === ticker).sort((a,b) => (new Date(b['C2']) - new Date(a['C2'])))[0];
      if (rec) await drawChart(ticker, rec);
    });
    // Initial universe build (defaults)
    handleBuildUniverse();
  }

  document.addEventListener('DOMContentLoaded', init);
})();

