/**
 * LMSC Lightweight SPA
 */

const API = {
  health: '/api/health',
  status: '/api/model_status',
  options: '/api/model_options',
  loadModel: '/api/load_model',
  predict: '/api/predict',
  batchPredict: '/api/batch_predict',
  train: '/api/train',
  trainingJobs: '/api/training_jobs',
  trainingJob: id => `/api/training_jobs/${encodeURIComponent(id)}`,
  datasets: '/api/dataset/list',
  uploadDataset: '/api/dataset/upload',
  deleteDataset: id => `/api/dataset/${id}`,
  previewDataset: id => `/api/dataset/${id}/preview`,
  runTask: '/api/run_task',
  taskStatus: id => `/api/task_status/${id}`,
  stopTask: id => `/api/stop_task/${id}`,
  taskResult: name => `/api/task_result/${name}`,
  taskDownload: name => `/api/task_download/${name}`,
  reliability: '/api/reliability_explanation',
  visualize: id => `/api/visualize/${id}`,
  models: '/api/models',
  loadModelByKey: key => `/api/models/${encodeURIComponent(key)}/load`,
  deleteModel: key => `/api/models/${encodeURIComponent(key)}`,
  renameModel: key => `/api/models/${encodeURIComponent(key)}/rename`,
  batchDeleteModels: '/api/models/batch_delete',
};

const state = {
  currentTab: 'dashboard',
  model: { loaded: false, name: '未加载' },
  datasets: [],
  task: { id: null, timer: null, name: null },
  models: [],
  modelsSort: { field: 'created_at', order: 'desc' },
  modelsSelected: new Set(),
  predictions: [],
  batchResults: [],
};

// ------------------------------------------------------------------
// Utils
// ------------------------------------------------------------------

const $ = id => document.getElementById(id);
const $$ = (sel, root = document) => root.querySelectorAll(sel);
const sleep = ms => new Promise(r => setTimeout(r, ms));
let availableUpdate = null;

async function checkUpdate() {
  const button = $('btn-check-update');
  setLoading(button, true, '检查中');
  $('btn-apply-update').hidden = true;
  availableUpdate = null;
  try {
    const info = await fetchJson('/api/updates');
    if (!info.latest) $('update-status').textContent = info.notes;
    else if (!info.newer) $('update-status').textContent = `已是最新版本（v${info.current}）`;
    else if (!info.ready) $('update-status').textContent = `发现 v${info.latest}，暂时无法更新`;
    else {
      availableUpdate = info.latest;
      $('update-status').textContent = `发现 v${info.latest}。更新会覆盖程序文件并删除远端不存在的程序文件，保留用户数据。`;
      $('btn-apply-update').hidden = false;
    }
  } catch (error) { $('update-status').textContent = error.message; }
  finally { setLoading(button, false); }
}

async function applyUpdate() {
  if (!availableUpdate || !confirm(`确定更新到 v${availableUpdate}？程序文件将被覆盖或删除；请先另行保存需要保留的本地代码修改。`)) return;
  const button = $('btn-apply-update');
  setLoading(button, true, '更新中');
  try {
    const result = await fetchJson('/api/updates/apply', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({version: availableUpdate}), timeout: 120000,
    });
    $('update-status').textContent = result.restarting ? `已更新至 v${result.version}，服务正在重启；若未恢复请手动启动。依赖变化时请重新运行安装脚本。` : `已更新至 v${result.version}，请手动重启服务；依赖变化时请重新运行安装脚本。`;
    availableUpdate = null;
    button.hidden = true;
    $('btn-check-update').disabled = true;
  } catch (error) { $('update-status').textContent = error.message; }
  finally { setLoading(button, false); }
}
let machineResources = null;
function duration(seconds) {
  if (!Number.isFinite(seconds)) return '估算中';
  seconds = Math.max(0, Math.ceil(seconds));
  return seconds >= 60 ? `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒` : `${seconds} 秒`;
}
function etaText(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return '预计剩余：估算中';
  return `预计剩余：${duration(seconds)}${seconds > 600 ? ' · 预计超过 10 分钟' : ''}`;
}
async function loadResourceInfo() {
  try {
    machineResources = await fetchJson('/api/resources');
    const cpus = machineResources.logical_cpus;
    const memory = machineResources.memory_bytes ? `${(machineResources.memory_bytes / 1073741824).toFixed(1)} GB 内存` : '内存信息不可用';
    const host = `本机 ${cpus} 逻辑核 · ${memory}`;
    $('train-resource').textContent = `${host} · 训练最多使用 ${cpus} 核，内存随数据及树规模变化 · 用时：训练后估算`;
    $('dash-train-resource').textContent = `${host} · 最多使用 ${cpus} 核 · 用时：训练中估算`;
    $('batch-resource').textContent = `${host} · 批量预测逐批运行 · 用时：首批完成后估算`;
    $('workflow-resource').textContent = `${host} · 筛选最多使用 ${Math.min(4, cpus)} 个工作进程，内存随模型和任务变化 · 用时：运行中估算`;
    $('viz-resource').textContent = `${host} · 图表逐张生成 · 用时：首张完成后估算`;
  } catch (_) {
    ['train-resource', 'dash-train-resource', 'batch-resource', 'workflow-resource', 'viz-resource'].forEach(id => { $(id).textContent = '本机资源信息不可用 · 用时将在运行中估算'; });
  }
}

function formatMetric(value, digits = 3) {
  if (value === null || value === undefined || value === '' || Number.isNaN(Number(value))) return '-';
  const n = Number(value);
  if (!isFinite(n)) return '-';
  if (Math.abs(n) >= 1e6) return n.toExponential(Math.max(1, digits - 1));
  const fixed = n.toFixed(digits);
  const trimmed = fixed.includes('.') ? fixed.replace(/\.?0+$/, '') : fixed;
  return trimmed;
}

let toastTimer;
function toast(message, type = 'info') {
  const el = $('toast');
  el.textContent = message;
  el.className = `toast ${type}`;
  requestAnimationFrame(() => el.classList.add('show'));
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), type === 'error' ? 6000 : 3500);
}

async function fetchJson(url, options = {}) {
  const controller = new AbortController();
  const timeout = options.timeout ?? 30000;
  const timer = setTimeout(() => controller.abort(), timeout);
  const { timeout: _timeout, signal, ...requestOptions } = options;
  try {
    const res = await fetch(url, { ...requestOptions, signal: signal || controller.signal });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      let message = `请求失败（HTTP ${res.status}）`;
      try { message = JSON.parse(text).error || message; } catch (_) { /* plain text */ }
      throw new Error(message || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('请求超时，请稍后重试。训练或复杂图表可能需要更长时间。');
    if (e instanceof TypeError) throw new Error('无法连接本地服务，请确认服务仍在运行。');
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

function setLoading(btn, loading = true, text = '') {
  if (!btn) return;
  btn.disabled = loading;
  btn.setAttribute('aria-busy', String(loading));
  if (loading) {
    btn.dataset.originalText = btn.textContent;
    btn.innerHTML = `<span class="spinner"></span> ${text || '处理中...'}`;
  } else {
    btn.textContent = btn.dataset.originalText || btn.textContent;
  }
}

// ------------------------------------------------------------------
// Navigation
// ------------------------------------------------------------------

function switchTab(name) {
  if (!$(`section-${name}`)) name = 'dashboard';
  state.currentTab = name;

  $$('.section').forEach(s => s.classList.remove('active'));
  const section = $(`section-${name}`);
  if (section) section.classList.add('active');

  $$('.nav-item').forEach(n => { n.classList.remove('active'); n.removeAttribute('aria-current'); });
  const nav = $(`nav-${name === 'visualization' ? 'data' : name}`);
  if (nav) { nav.classList.add('active'); nav.setAttribute('aria-current', 'page'); }

  const titleText = name === 'visualization' ? '数据集 / 可视化分析' : (nav?.dataset.title || nav?.textContent?.trim() || '');
  $('page-title').textContent = titleText;
  $('page-title').title = titleText;

  // Close mobile sidebar
  setSidebarOpen(false);
  if (window.location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);

  // Refresh data when entering certain tabs
  if (name === 'data') loadDatasets();
  if (name === 'dashboard') refreshStatus();
  if (name === 'models') loadModels();
  if (name === 'visualization') { populateAnalysisDatasets(); loadDatasetAnalysis(); renderAnalysisBatch(); }
}

function setSidebarOpen(open) {
  $('sidebar').classList.toggle('open', open);
  $('sidebar-backdrop').hidden = !open;
  $('menu-toggle').setAttribute('aria-expanded', String(open));
  $('menu-toggle').setAttribute('aria-controls', 'sidebar');
}

// ------------------------------------------------------------------
// Model status
// ------------------------------------------------------------------

async function refreshStatus() {
  try {
    const status = await fetchJson(API.status);
    state.model = status;
    renderStatus();
  } catch (e) {
    $('model-pill').querySelector('span:not(.dot)').textContent = '服务未连接';
    $('model-pill').title = e.message;
  }
}

function renderStatus() {
  const pill = $('model-pill');
  const badge = pill.querySelector('.dot');
  const text = pill.querySelector('span:not(.dot)');

  if (state.model.loaded) {
    pill.classList.remove('loading');
    badge.style.background = 'var(--success)';
    const label = `${state.model.name} · R² ${formatMetric(state.model.metrics?.R2, 3)}`;
    text.textContent = label;
    pill.title = label;
  } else {
    pill.classList.add('loading');
    badge.style.background = 'var(--warning)';
    text.textContent = state.model.name || '未加载';
    pill.title = text.textContent;
  }

  // Dashboard card
  const cardName = $('dash-model-name');
  const cardR2 = $('dash-model-r2');
  const cardRmse = $('dash-model-rmse');
  const cardFeat = $('dash-model-features');
  const cardStatus = $('dash-model-status');
  if (cardName) { cardName.textContent = state.model.name || '-'; cardName.title = cardName.textContent; }
  if (cardR2) { cardR2.textContent = formatMetric(state.model.metrics?.R2, 3); cardR2.title = cardR2.textContent; }
  if (cardRmse) { cardRmse.textContent = formatMetric(state.model.metrics?.RMSE, 3); cardRmse.title = cardRmse.textContent; }
  if (cardFeat) { cardFeat.textContent = state.model.features_count ?? '-'; cardFeat.title = cardFeat.textContent; }
  if (cardStatus) cardStatus.textContent = state.model.loaded ? '已加载' : '等待加载';
  $('model-notice').hidden = Boolean(state.model.loaded);
}

async function loadModelOptions(selectId, selected, availableOptions) {
  try {
    const options = availableOptions || (await fetchJson(API.options)).options;
    const select = $(selectId);
    if (!select) return options;
    select.innerHTML = '';
    select.disabled = !options.length;
    if (!options.length) {
      select.add(new Option('暂无模型，请先训练', ''));
      return options;
    }
    options.forEach(opt => {
      const el = document.createElement('option');
      el.value = opt.key;
      el.textContent = opt.name;
      el.title = opt.key;
      select.appendChild(el);
    });
    if (selected && options.some(o => o.key === selected)) {
      select.value = selected;
    } else if (state.model?.key && options.some(o => o.key === state.model.key)) {
      select.value = state.model.key;
    }
    return options;
  } catch (e) {
    console.error('Failed to load model options', e);
    return [];
  }
}

async function loadModelByKey(key, silent = false) {
  if (!key) return false;
  try {
    const data = await fetchJson(API.loadModelByKey(key), { method: 'POST' });
    if (data.success) {
      state.model = data.model_info;
      renderStatus();
      if (!silent) toast(`模型已加载: ${data.model_info.name}`, 'success');
      syncModelSelects();
      return true;
    }
    throw new Error(data.error || '加载失败');
  } catch (e) {
    if (!silent) toast(e.message, 'error');
    return false;
  }
}

function syncModelSelects() {
  const key = state.model?.key;
  if (!key) return;
  ['select-model', 'predict-model', 'batch-model', 'workflow-model', 'phase-model'].forEach(id => {
    const sel = $(id);
    if (sel && [...sel.options].some(o => o.value === key)) sel.value = key;
  });
}

async function refreshAllModelSelects() {
  try {
    const { options } = await fetchJson(API.options);
    await Promise.all(['select-model', 'predict-model', 'batch-model', 'workflow-model', 'phase-model']
      .map(id => loadModelOptions(id, state.model?.key, options)));
    $('btn-load-model').disabled = !options.length;
    $('dash-model-empty').hidden = Boolean(options.length);
  } catch (e) {
    toast(e.message, 'error');
  }
}

async function handleLoadModel() {
  const key = $('select-model')?.value || 'ET';
  const btn = $('btn-load-model');
  setLoading(btn, true, '加载中');
  const ok = await loadModelByKey(key);
  setLoading(btn, false);
  if (ok) loadModels();
}

// ------------------------------------------------------------------
// Local model management
// ------------------------------------------------------------------

async function loadModels() {
  const tbody = $('models-body');
  if (!tbody) return;
  try {
    const data = await fetchJson(API.models);
    state.models = data.models || [];
    // Drop selections that no longer exist
    const keys = new Set(state.models.map(m => m.key));
    state.modelsSelected = new Set([...state.modelsSelected].filter(k => keys.has(k)));
    renderModels();
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty"><div class="empty-icon">⚠️</div>${escapeHtml(e.message)}</td></tr>`;
  }
}

function formatBytes(bytes) {
  if (!bytes || bytes <= 0) return '-';
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

function sortModels(models, field, order) {
  const dir = order === 'asc' ? 1 : -1;
  const val = m => {
    switch (field) {
      case 'name': return (m.name || '').toLowerCase();
      case 'type': return (m.type || '').toLowerCase();
      case 'created_at': return m.created_at || '';
      case 'r2': return Number(m.metrics?.R2 ?? -Infinity);
      case 'rmse': return Number(m.metrics?.RMSE ?? Infinity);
      case 'dataset': return (m.dataset?.name || '').toLowerCase();
      default: return m.created_at || '';
    }
  };
  return [...models].sort((a, b) => {
    const va = val(a), vb = val(b);
    if (va < vb) return -1 * dir;
    if (va > vb) return 1 * dir;
    return a.key.localeCompare(b.key);
  });
}

function renderSortIndicators() {
  $$('#models-table th.sortable').forEach(th => {
    const indicator = th.querySelector('.sort-indicator');
    if (!indicator) return;
    if (th.dataset.sort === state.modelsSort.field) {
      indicator.textContent = state.modelsSort.order === 'asc' ? '▲' : '▼';
      th.classList.add('sort-active');
      th.setAttribute('aria-sort', state.modelsSort.order === 'asc' ? 'ascending' : 'descending');
    } else {
      indicator.textContent = '';
      th.classList.remove('sort-active');
      th.removeAttribute('aria-sort');
    }
  });
}

function handleSortModels(field) {
  if (state.modelsSort.field === field) {
    state.modelsSort.order = state.modelsSort.order === 'asc' ? 'desc' : 'asc';
  } else {
    state.modelsSort = { field, order: field === 'name' || field === 'type' || field === 'dataset' ? 'asc' : 'desc' };
  }
  renderModels();
}

function updateBatchBar() {
  const bar = $('models-batch-bar');
  const countEl = $('models-selected-count');
  if (!bar) return;
  const n = state.modelsSelected.size;
  bar.style.display = n > 0 ? 'flex' : 'none';
  if (countEl) countEl.textContent = `已选 ${n} 个模型`;
  const selectAll = $('models-select-all');
  if (selectAll) {
    selectAll.checked = state.models.length > 0 && n === state.models.length;
    selectAll.indeterminate = n > 0 && n < state.models.length;
  }
}

function renderModels() {
  const tbody = $('models-body');
  const models = state.models;
  if (!models.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty">暂无模型，请先在「训练模型」页训练</td></tr>`;
    renderSortIndicators();
    updateBatchBar();
    return;
  }
  const sorted = sortModels(models, state.modelsSort.field, state.modelsSort.order);
  const activeKey = state.model?.key;
  tbody.innerHTML = sorted.map(m => {
    const created = m.created_at ? m.created_at.replace('T', ' ').slice(0, 16) : '-';
    const dataset = m.dataset?.name || (m.dataset?.path ? m.dataset.path.split('/').pop() : '-');
    const hp = m.hyperparameters || {};
    const paramsText = Object.entries(hp).filter(([, v]) => v !== null && v !== undefined && v !== '')
      .map(([k, v]) => `${k}=${v}`).join(', ');
    const title = `Key: ${m.key}\n参数: ${paramsText || '-'}\n特征数: ${m.feature_count}\n文件大小: ${formatBytes(m.file_size)}\n训练样本: ${m.train_info?.n_samples_train ?? '-'}\n测试样本: ${m.train_info?.n_samples_test ?? '-'}`;
    const checked = state.modelsSelected.has(m.key) ? 'checked' : '';
    return `
    <tr style="${m.key === activeKey ? 'outline:2px solid var(--primary);outline-offset:-2px' : ''}">
      <td><input type="checkbox" class="model-check" data-key="${escapeHtml(m.key)}" ${checked}></td>
      <td title="${escapeHtml(title)}">${escapeHtml(m.name)}${m.key === activeKey ? ' <span class="badge badge-success">当前</span>' : ''}</td>
      <td>${escapeHtml(m.type)}</td>
      <td>${created}</td>
      <td>${formatMetric(m.metrics?.R2, 3)}</td>
      <td>${formatMetric(m.metrics?.RMSE, 3)}</td>
      <td title="${escapeHtml(m.dataset?.path || '')}">${escapeHtml(dataset)}</td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="loadModelByKey('${escapeHtml(m.key)}')">加载</button>
        <button class="btn btn-secondary btn-sm" style="margin-left:6px" onclick="showModelDetails('${escapeHtml(m.key)}')">详情</button>
        <button class="btn btn-secondary btn-sm" style="margin-left:6px" onclick="renameModel('${escapeHtml(m.key)}')">重命名</button>
        <button class="btn btn-secondary btn-sm" style="margin-left:6px" onclick="handleDeleteModel('${escapeHtml(m.key)}')">删除</button>
      </td>
    </tr>`;
  }).join('');

  // Bind checkbox events
  $$('#models-body .model-check').forEach(cb => {
    cb.addEventListener('change', e => {
      const key = e.target.dataset.key;
      if (e.target.checked) state.modelsSelected.add(key);
      else state.modelsSelected.delete(key);
      updateBatchBar();
    });
  });

  renderSortIndicators();
  updateBatchBar();
}

async function renameModel(key) {
  const model = state.models.find(m => m.key === key);
  const name = prompt('输入模型显示名称（最多 80 字）：', model?.name || '');
  if (name === null) return;
  if (!name.trim() || name.trim().length > 80) return toast('名称不能为空且最多 80 字', 'error');
  try {
    await fetchJson(API.renameModel(key), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name.trim() }) });
    await refreshStatus();
    await Promise.all([loadModels(), refreshAllModelSelects()]);
    toast('模型名称已更新', 'success');
  } catch (e) { toast(e.message, 'error'); }
}

function handleSelectAllModels(checked) {
  if (checked) state.models.forEach(m => state.modelsSelected.add(m.key));
  else state.modelsSelected.clear();
  renderModels();
}

function clearModelSelection() {
  state.modelsSelected.clear();
  renderModels();
}

async function handleBatchDeleteModels() {
  const keys = [...state.modelsSelected];
  if (!keys.length) return toast('请先勾选要删除的模型', 'error');
  const activeKey = state.model?.key;
  const includesActive = keys.includes(activeKey);
  const msg = `删除选中的 ${keys.length} 个模型及其文件？` +
    (includesActive ? '\n其中包含当前模型。' : '');
  if (!confirm(msg)) return;

  const btn = $('btn-models-batch-delete');
  setLoading(btn, true, '删除中');
  try {
    const data = await fetchJson(API.batchDeleteModels, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keys }),
    });
    const deleted = data.deleted || [];
    const failed = data.failed || [];
    if (data.model_info) state.model = data.model_info;
    renderStatus();
    state.modelsSelected.clear();
    await Promise.all([loadModels(), refreshAllModelSelects()]);

    if (failed.length) {
      toast(`已删除 ${deleted.length} 个，失败 ${failed.length} 个: ${failed.map(f => f.key).join(', ')}`, 'error');
    } else {
      toast(`已批量删除 ${deleted.length} 个模型`, 'success');
    }
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    setLoading(btn, false);
  }
}

function showModelDetails(key) {
  fetchJson(API.models)
    .then(data => {
      const m = (data.models || []).find(x => x.key === key);
      if (!m) return toast('未找到该模型', 'error');
      const hp = m.hyperparameters || {};
      const hpRows = Object.entries(hp).map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(String(v))}</td></tr>`).join('');
      const modal = document.createElement('div');
      modal.className = 'panel';
      modal.style.cssText = 'position:fixed;inset:8%;z-index:200;overflow:auto;background:var(--surface);max-width:720px;margin:auto;padding:24px;border-radius:12px;box-shadow:0 20px 60px rgba(0,0,0,.5)';
      modal.innerHTML = `
        <div class="panel-title" style="display:flex;justify-content:space-between;align-items:center">
          <span>模型详情</span>
          <button class="btn btn-secondary" style="padding:4px 10px;font-size:0.8rem" onclick="this.closest('.panel').remove()">关闭</button>
        </div>
        <p style="color:var(--muted);margin:10px 0 14px">${escapeHtml(m.name)}</p>
        <div class="metric-row" style="margin-bottom:14px">
          <div class="metric-box"><div class="label">R²</div><div class="num">${formatMetric(m.metrics?.R2, 3)}</div></div>
          <div class="metric-box"><div class="label">RMSE</div><div class="num">${formatMetric(m.metrics?.RMSE, 3)}</div></div>
          <div class="metric-box"><div class="label">特征数</div><div class="num">${m.feature_count}</div></div>
          <div class="metric-box"><div class="label">文件大小</div><div class="num">${formatBytes(m.file_size)}</div></div>
        </div>
        <p style="color:var(--muted);font-size:0.85rem">Key: <code>${escapeHtml(m.key)}</code></p>
        <p style="color:var(--muted);font-size:0.85rem">创建时间: ${m.created_at ? m.created_at.replace('T', ' ').slice(0, 19) : '-'} · 数据集: ${escapeHtml(m.dataset?.name || '-')} (${escapeHtml(m.dataset?.path || '-')})</p>
        <p style="color:var(--muted);font-size:0.85rem">训练样本: ${m.train_info?.n_samples_train ?? '-'} · 测试样本: ${m.train_info?.n_samples_test ?? '-'} · 耗时: ${m.train_info?.train_time ?? '-'}s</p>
        ${hpRows ? `<table class="data-table" style="margin-top:10px"><thead><tr><th>超参数</th><th>值</th></tr></thead><tbody>${hpRows}</tbody></table>` : '<p style="color:var(--muted);font-size:0.85rem;margin-top:10px">该模型无元数据（可能是预置模型），仅显示指标。</p>'}
      `;
      document.body.appendChild(modal);
      modal.addEventListener('click', e => { if (e.target === modal) modal.remove(); });
    })
    .catch(e => toast(e.message, 'error'));
}

async function handleDeleteModel(key) {
  if (!confirm(`删除模型 ${key} 及其文件？`)) return;
  try {
    const data = await fetchJson(API.deleteModel(key), { method: 'DELETE' });
    if (data.success) {
      toast('模型已删除', 'success');
      if (data.model_info) state.model = data.model_info;
      renderStatus();
      state.modelsSelected.delete(key);
      await Promise.all([loadModels(), refreshAllModelSelects()]);
    } else {
      throw new Error(data.error || '删除失败');
    }
  } catch (e) {
    toast(e.message, 'error');
  }
}

// ------------------------------------------------------------------
// Prediction
// ------------------------------------------------------------------

async function handlePredict() {
  if ($('btn-predict').disabled) return;
  const input = $('predict-input');
  const formula = input.value.trim();
  if (!formula) return toast('请输入化学式', 'error');

  const btn = $('btn-predict');
  setLoading(btn, true, '预测中');
  try {
    const modelKey = $('predict-model')?.value || undefined;
    const payload = { formula };
    if (modelKey) payload.model_key = modelKey;
    const result = await fetchJson(API.predict, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    renderPredictionResult(result);
    if (!result.error) {
      state.predictions = [result, ...state.predictions.filter(r => r.formula !== result.formula)].slice(0, 8);
      renderPredictionHistory();
      refreshStatus();
    }
  } catch (e) {
    $('predict-result').style.display = 'none';
    toast(e.message, 'error');
  } finally {
    setLoading(btn, false);
  }
}

function renderPredictionResult(res) {
  const container = $('predict-result');
  if (res.error) {
    container.innerHTML = `<div class="result" role="alert"><p style="color:var(--danger)">${escapeHtml(res.error)}</p></div>`;
    container.style.display = 'block';
    return;
  }

  const r = res.reliability || {};
  const levelClass = r.level === '高' ? 'badge-success' : r.level === '中' ? 'badge-warning' : 'badge-danger';

  container.innerHTML = `
    <div class="result">
      <div class="result-header">
        <div>
          <div class="result-formula" title="${escapeHtml(res.formula)}">${escapeHtml(res.formula)}</div>
          <div style="margin-top:6px;color:var(--muted);font-size:0.85rem">模型: ${escapeHtml(res.model)}</div>
        </div>
        <div style="text-align:right">
          <div class="result-tc" title="${escapeHtml(res.predicted_tc)} K">${formatMetric(res.predicted_tc, 2)} <span style="font-size:0.9rem;color:var(--muted)">K</span></div>
          <span class="badge ${levelClass}" style="margin-top:6px">可靠度: ${escapeHtml(r.level || '未知')} · ${formatMetric(r.score, 1)}</span>
        </div>
      </div>
      <p style="color:var(--muted);font-size:0.92rem">${escapeHtml(res.interpretation)}</p>
      ${r.proximity_note ? `<p class="research-note">${escapeHtml(r.proximity_note)}</p>` : ''}
      <div class="reliability-meter" role="meter" aria-label="启发式可靠度评分" aria-valuenow="${Math.max(0, Math.min(100, Number(r.score) || 0))}" aria-valuemin="0" aria-valuemax="100"><div style="width:${Math.max(0, Math.min(100, Number(r.score) || 0))}%"></div></div>
      <div class="metric-row">
        <div class="metric-box"><div class="label">模型估计区间</div><div class="num" title="${escapeHtml(r.confidence_interval || '-')}">${escapeHtml(r.confidence_interval || '-')}</div></div>
        <div class="metric-box"><div class="label">树间标准差</div><div class="num">${formatMetric(r.details?.ensemble_std, 2)}</div></div>
        <div class="metric-box"><div class="label">训练邻近度</div><div class="num">${r.details?.training_proximity == null ? '未计算' : `${formatMetric(r.details.training_proximity, 1)}%`}</div></div>
        <div class="metric-box"><div class="label">模型 R²</div><div class="num" title="${formatMetric(r.details?.model_r2, 3)}">${formatMetric(r.details?.model_r2, 3)}</div></div>
        <div class="metric-box"><div class="label">RMSE</div><div class="num" title="${formatMetric(r.details?.rmse, 3)}">${formatMetric(r.details?.rmse, 3)}</div></div>
      </div>
    </div>
  `;
  container.style.display = 'block';
}

function renderPredictionHistory() {
  $('prediction-history').hidden = !state.predictions.length;
  $('history-list').innerHTML = state.predictions.map(r => `<button type="button" class="history-item" data-formula="${escapeHtml(r.formula)}" data-target="predict-input">${escapeHtml(r.formula)}<span>${formatMetric(r.predicted_tc, 2)} K</span></button>`).join('');
}

function parseBatchInput(raw) {
  return raw.split(/[\n,，;；]+/).map(s => s.trim()).filter(Boolean);
}

function updateBatchCount() {
  const count = parseBatchInput($('batch-input').value).length;
  $('batch-count').textContent = count ? `已输入 ${count} 个化学式` : '每行一个化学式';
}

async function handleBatchPredict() {
  if ($('btn-batch-predict').disabled) return;
  const area = $('batch-input');
  const raw = area.value.trim();
  if (!raw) return toast('请输入化学式列表', 'error');

  const formulas = parseBatchInput(raw);
  if (!formulas.length) return toast('未解析到有效化学式', 'error');
  if (formulas.length > 1000) return toast('一次最多预测 1000 个化学式，请分批提交。', 'error');

  const btn = $('btn-batch-predict');
  setLoading(btn, true, '批量预测中');
  $('batch-progress').hidden = false;
  $('batch-progress-bar').value = 0;
  $('batch-progress-text').textContent = '准备预测 · 预计剩余：估算中';
  const started = performance.now();
  try {
    const modelKey = $('batch-model')?.value || undefined;
    const results = [];
    for (let offset = 0; offset < formulas.length; offset += 10) {
      const batch = formulas.slice(offset, offset + 10);
      const payload = { formulas: batch };
      if (modelKey) payload.model_key = modelKey;
      const data = await fetchJson(API.batchPredict, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), timeout: 300000 });
      if (data.error) throw new Error(data.error);
      results.push(...data.results.map((result, index) => ({ formula: batch[index], ...result })));
      const elapsed = (performance.now() - started) / 1000;
      const remaining = elapsed / results.length * (formulas.length - results.length);
      $('batch-progress-bar').value = results.length / formulas.length * 100;
      $('batch-progress-text').textContent = `${results.length}/${formulas.length} · 已用 ${duration(elapsed)} · ${etaText(remaining)}`;
    }
    renderBatchResults(results);
    refreshStatus();
  } catch (e) {
    $('batch-result').style.display = 'none';
    $('batch-progress-text').textContent = `预测中断：${e.message}`;
    toast(e.message, 'error');
  } finally {
    setLoading(btn, false);
  }
}

function renderBatchResults(results) {
  state.batchResults = results;
  const container = $('batch-result');
  const valid = results.filter(r => !r.error && Number.isFinite(Number(r.predicted_tc)));
  const top = [...valid].sort((a, b) => b.predicted_tc - a.predicted_tc).slice(0, 10);
  const max = Math.max(...top.map(r => Number(r.predicted_tc)), 1);
  const chart = top.map(r => `<div class="comparison-row"><span class="comparison-name" title="${escapeHtml(r.formula)}">${escapeHtml(r.formula)}</span><div class="comparison-track"><div class="comparison-bar" style="width:${Math.max(0, Number(r.predicted_tc)) / max * 100}%"></div></div><span class="comparison-value">${formatMetric(r.predicted_tc, 2)} K</span></div>`).join('');
  const rows = results.map((r, i) => {
    if (r.error) {
      return `<tr><td>${i + 1}</td><td>${escapeHtml(r.formula)}</td><td colspan="3" style="color:var(--danger)">${escapeHtml(r.error)}</td></tr>`;
    }
    const levelClass = r.reliability?.level === '高' ? 'badge-success' : r.reliability?.level === '中' ? 'badge-warning' : 'badge-danger';
    return `    <tr>
      <td>${i + 1}</td>
      <td title="${escapeHtml(r.formula)}">${escapeHtml(r.formula)}</td>
      <td><strong>${formatMetric(r.predicted_tc, 2)}</strong> K</td>
      <td><span class="badge ${levelClass}">${escapeHtml(r.reliability?.level || '-')}</span></td>
      <td>${formatMetric(r.reliability?.score, 1)}</td>
    </tr>`;
  }).join('');

  container.innerHTML = `
    <div class="batch-toolbar"><div class="batch-summary"><strong>${valid.length}</strong> 条成功 · ${results.length - valid.length} 条失败${top.length ? ` · 最高 Tc <strong>${formatMetric(top[0].predicted_tc, 2)} K</strong>` : ''}</div><button type="button" class="btn btn-secondary btn-sm" id="btn-export-batch">导出 CSV ↓</button></div>
    ${chart ? `<div class="chart-wrap"><div class="chart-title">预测 Tc 排名 · Top ${top.length}（K）</div>${chart}</div>` : ''}
    <div class="table-scroll"><table class="data-table">
      <thead><tr><th>#</th><th>化学式</th><th>Tc (K)</th><th>可靠度</th><th>评分</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  `;
  $('btn-export-batch').addEventListener('click', exportBatchCsv);
  container.style.display = 'block';
}

function csvCell(value) {
  const text = String(value ?? '');
  // Prevent spreadsheet formula execution when exported user input is opened.
  const safe = /^[=+\-@\t\r]/.test(text) ? `'${text}` : text;
  return `"${safe.replace(/"/g, '""')}"`;
}

function exportBatchCsv() {
  const rows = [['formula', 'predicted_tc_K', 'reliability_level', 'reliability_score', 'error'],
    ...state.batchResults.map(r => [r.formula, r.predicted_tc, r.reliability?.level, r.reliability?.score, r.error])];
  const blob = new Blob(['\uFEFF' + rows.map(row => row.map(csvCell).join(',')).join('\r\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `LMSC_predictions_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ------------------------------------------------------------------
// Training
// ------------------------------------------------------------------

async function handleTrain(quick = false) {
  if ($('btn-train').disabled || $('btn-dash-train').disabled) return;
  const payload = quick ? {
    model_type: 'ET', n_estimators: 300, min_samples_split: 2, min_samples_leaf: 1,
    dataset_id: null, display_name: $('dash-train-name').value.trim(),
  } : {
    model_type: $('train-algo').value,
    n_estimators: parseInt($('train-estimators').value) || 300,
    max_depth: $('train-max-depth').value,
    min_samples_split: parseInt($('train-min-split').value) || 2,
    min_samples_leaf: parseInt($('train-min-leaf').value) || 1,
    dataset_id: $('train-dataset').value || null,
    display_name: $('train-name').value.trim(),
  };

  const btn = $('btn-train');
  setLoading(btn, true, '训练中');
  setLoading($('btn-dash-train'), true, '训练中');
  $('train-progress').hidden = false;
  $('dash-train-progress').hidden = false;
  $('dash-train-result').textContent = '';
  $('train-stage').textContent = '提交训练任务';
  $('train-percent').textContent = '0%';
  $('train-progress-bar').value = 0;
  $('train-result').style.display = 'none';
  try {
    const data = await fetchJson(API.trainingJobs, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    let job;
    do {
      await sleep(1000);
      job = await fetchJson(API.trainingJob(data.job_id));
      $('train-stage').textContent = job.stage;
      $('train-percent').textContent = `${job.progress}%`;
      $('train-progress-bar').value = job.progress;
      const detail = job.elapsed_seconds != null ? ` · 已用 ${duration(job.elapsed_seconds)} · ${etaText(job.eta_seconds)}` : '';
      $('train-stage').textContent = job.stage + detail;
      $('dash-train-stage').textContent = job.stage + detail;
      $('dash-train-percent').textContent = `${job.progress}%`;
      $('dash-train-bar').value = job.progress;
    } while (job.status === 'running');
    if (job.status === 'completed') {
      const info = job.result;
      state.model = { ...info, loaded: true, key: info.model_key };
      renderStatus();
      renderTrainResult(info);
      $('dash-train-result').textContent = `${info.name} 已加载 · R² ${formatMetric(info.metrics.R2, 3)} · RMSE ${formatMetric(info.metrics.RMSE, 3)} K`;
      await Promise.all([refreshStatus(), refreshAllModelSelects(), loadModels()]);
      toast(`训练完成: ${info.name}`, 'success');
    } else {
      throw new Error(job.error || '训练失败');
    }
  } catch (e) {
    $('train-stage').textContent = `训练状态：${e.message}`;
    $('dash-train-stage').textContent = `训练状态：${e.message}`;
    toast(e.message, 'error');
  } finally {
    setLoading(btn, false);
    setLoading($('btn-dash-train'), false);
  }
}

function renderTrainResult(info) {
  $('train-result').innerHTML = `
    <div class="result">
      <div class="result-header">
        <div>
          <div class="result-formula" title="${escapeHtml(info.name)}">${escapeHtml(info.name)}</div>
          <div style="margin-top:6px;color:var(--muted);font-size:0.85rem">Key: ${info.model_key}</div>
        </div>
      </div>
      <div class="metric-row">
        <div class="metric-box"><div class="label">R²</div><div class="num" title="${formatMetric(info.metrics.R2, 3)}">${formatMetric(info.metrics.R2, 3)}</div></div>
        <div class="metric-box"><div class="label">RMSE</div><div class="num" title="${formatMetric(info.metrics.RMSE, 3)}">${formatMetric(info.metrics.RMSE, 3)}</div></div>
        <div class="metric-box"><div class="label">训练样本</div><div class="num" title="${info.n_samples_train}">${info.n_samples_train}</div></div>
        <div class="metric-box"><div class="label">测试样本</div><div class="num" title="${info.n_samples_test}">${info.n_samples_test}</div></div>
        <div class="metric-box"><div class="label">耗时</div><div class="num" title="${info.train_time}s">${info.train_time}s</div></div>
      </div>
    </div>
  `;
  $('train-result').style.display = 'block';
}

// ------------------------------------------------------------------
// Datasets
// ------------------------------------------------------------------

async function loadDatasets() {
  try {
    const data = await fetchJson(API.datasets);
    state.datasets = data.datasets || [];
    const builtin = state.datasets.find(ds => ds.builtin);
    $('dashboard-dataset-info').textContent = builtin && builtin.rows != null
      ? `默认数据：LM-based-SC · ${builtin.rows.toLocaleString()} 条记录`
      : '默认数据：LM-based-SC · 数据不可用';
    renderDatasets();
    populateDatasetSelects();
    populateAnalysisDatasets();
  } catch (e) {
    $('dashboard-dataset-info').textContent = '默认数据：LM-based-SC · 数据不可用';
    console.error(e);
  }
}

function populateAnalysisDatasets() {
  const select = $('analysis-dataset');
  if (!select) return;
  const current = select.value || 'builtin';
  select.innerHTML = '';
  (state.datasets.length ? state.datasets : [{ id: 'builtin', name: 'LM-based 基础数据集' }]).forEach(ds => {
    const option = document.createElement('option');
    option.value = ds.id;
    option.textContent = ds.name;
    select.appendChild(option);
  });
  select.value = [...select.options].some(o => o.value === current) ? current : 'builtin';
}

const analysisCharts = {};
function renderAnalysisBars(id, rows, unit = '') {
  const container = $(id);
  analysisCharts[id] = { rows, unit };
  const button = document.querySelector?.(`.analysis-download[data-chart="${id}"]`);
  if (button) button.disabled = !rows.length;
  if (!rows.length) { container.textContent = '当前条件下没有数据。'; return; }
  const max = Math.max(...rows.map(row => Number(row.count)), 1);
  container.innerHTML = rows.map(row => `<div class="comparison-row"><span class="comparison-name" title="${escapeHtml(row.label)}">${escapeHtml(row.label)}</span><div class="comparison-track"><div class="comparison-bar" style="width:${Math.max(0, Number(row.count)) / max * 100}%"></div></div><span class="comparison-value">${formatMetric(row.count, 0)}${unit}</span></div>`).join('');
}

function downloadAnalysisChart(id) {
  const chart = analysisCharts[id];
  if (!chart?.rows.length) return;
  const titles = { 'analysis-tc': 'Tc 分布（K）', 'analysis-elements': '元素出现频次', 'analysis-components': '化学式元素数量', 'analysis-batch': '批量预测 Tc（K）' };
  const rows = chart.rows, max = Math.max(...rows.map(r => Number(r.count)), 1);
  const height = 105 + rows.length * 32;
  const subtitle = id === 'analysis-batch' ? '当前会话' : $('analysis-status').textContent;
  const bars = rows.map((r, i) => {
    const y = 80 + i * 32, width = Math.max(0, Number(r.count)) / max * 430;
    return `<text x="160" y="${y + 12}" text-anchor="end" font-size="13" fill="#40546a">${escapeHtml(r.label)}</text><rect x="175" y="${y}" width="430" height="16" rx="8" fill="#edf1f5"/><rect x="175" y="${y}" width="${width}" height="16" rx="8" fill="#078b91"/><text x="615" y="${y + 12}" font-size="13" fill="#40546a">${escapeHtml(formatMetric(r.count, 2) + chart.unit)}</text>`;
  }).join('');
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="760" height="${height}" viewBox="0 0 760 ${height}"><rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif"><text x="30" y="30" font-size="19" font-weight="bold" fill="#1d2b40">${escapeHtml(titles[id])}</text><text x="30" y="54" font-size="12" fill="#68788e">${escapeHtml(subtitle)}</text>${bars}</g></svg>`;
  const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url; link.download = `LMSC_${id}.svg`; document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

let analysisRequest = 0;
async function loadDatasetAnalysis() {
  invalidateDatasetPlots();
  const id = $('analysis-dataset').value || 'builtin';
  const threshold = $('analysis-threshold').value;
  if (threshold === '' || !Number.isFinite(Number(threshold)) || Number(threshold) < 0) return toast('请输入非负 Tc 阈值', 'error');
  const request = ++analysisRequest;
  $('analysis-status').textContent = '正在分析数据集…';
  ['analysis-tc', 'analysis-elements', 'analysis-components'].forEach(id => { delete analysisCharts[id]; const button = document.querySelector(`.analysis-download[data-chart="${id}"]`); if (button) button.disabled = true; });
  try {
    const data = await fetchJson(`/api/dataset/${encodeURIComponent(id)}/analysis?threshold=${encodeURIComponent(threshold)}`, { timeout: 120000 });
    if (request !== analysisRequest) return;
    $('analysis-status').textContent = data.count ? `${data.name} · ${data.count} 条记录 · Tc ${formatMetric(data.min, 2)}–${formatMetric(data.max, 2)} K · 平均 ${formatMetric(data.mean, 2)} K` : `${data.name} · 没有符合阈值的记录。`;
    renderAnalysisBars('analysis-tc', data.histogram);
    renderAnalysisBars('analysis-elements', data.elements);
    renderAnalysisBars('analysis-components', data.components);
  } catch (e) {
    if (request !== analysisRequest) return;
    $('analysis-status').textContent = e.message;
    ['analysis-tc', 'analysis-elements', 'analysis-components'].forEach(id => { $(id).textContent = '分析不可用'; });
  }
}

function renderAnalysisBatch() {
  const valid = state.batchResults.filter(r => !r.error && r.predicted_tc !== null && r.predicted_tc !== undefined && Number.isFinite(Number(r.predicted_tc)));
  if (!valid.length) {
    delete analysisCharts['analysis-batch'];
    const button = document.querySelector('.analysis-download[data-chart="analysis-batch"]');
    if (button) button.disabled = true;
    $('analysis-batch').innerHTML = '暂无批量结果。<button class="text-button" type="button" data-open-tab="batch">前往批量预测 →</button>';
    return;
  }
  const rows = [...valid].sort((a, b) => Number(b.predicted_tc) - Number(a.predicted_tc)).slice(0, 10)
    .map(r => ({ label: r.formula, count: Number(r.predicted_tc) }));
  renderAnalysisBars('analysis-batch', rows, ' K');
}

function renderDatasets() {
  const tbody = $('datasets-body');
  if (!state.datasets.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty">暂无数据集</td></tr>`;
    return;
  }
  tbody.innerHTML = state.datasets.map(ds => `
    <tr>
      <td title="${escapeHtml(ds.name)}">${escapeHtml(ds.name)} ${ds.builtin ? '<span class="badge badge-success">基础数据</span>' : ''}${ds.available === false ? ' <span class="badge badge-danger">文件缺失</span>' : ''}</td>
      <td>${ds.rows ?? '-'}</td>
      <td>${ds.valid_rows ?? '-'}</td>
      <td>${ds.created_at?.slice(0, 10) || '-'}</td>
      <td>
        <button class="btn btn-secondary btn-sm dataset-detail-button" data-id="${escapeHtml(ds.id)}">详情</button>
        ${ds.builtin ? '' : `<button class="btn btn-secondary btn-sm dataset-delete-button" data-id="${escapeHtml(ds.id)}">删除</button>`}
      </td>
    </tr>
  `).join('');
}

function populateDatasetSelects() {
  const selects = ['train-dataset', 'workflow-dataset'];
  selects.forEach(id => {
    const sel = $(id);
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = '<option value="">LM 基础数据</option>';
    state.datasets.filter(ds => !ds.builtin).forEach(ds => {
      const opt = document.createElement('option');
      opt.value = ds.id;
      opt.textContent = `${ds.name} (${ds.rows} 行)`;
      sel.appendChild(opt);
    });
    sel.value = current;
  });
}

function initUpload() {
  const zone = $('dropzone');
  const input = $('dataset-file');
  if (!zone || !input) return;

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  });
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) input.files = e.dataTransfer.files;
    updateDropzoneText();
  });
  input.addEventListener('change', updateDropzoneText);
}

function updateDropzoneText() {
  const input = $('dataset-file');
  const zone = $('dropzone-text');
  if (input.files && input.files[0]) {
    zone.innerHTML = `<strong>${escapeHtml(input.files[0].name)}</strong><br><span style="font-size:0.85rem">${formatBytes(input.files[0].size)} · 点击或拖拽替换文件</span>`;
    $('upload-status').textContent = '文件已选择';
  }
}

async function handleUpload() {
  const input = $('dataset-file');
  const nameInput = $('dataset-name');
  if (!input.files || !input.files[0]) return toast('请选择文件', 'error');

  const form = new FormData();
  form.append('file', input.files[0]);
  form.append('name', nameInput.value.trim() || input.files[0].name);

  const btn = $('btn-upload');
  setLoading(btn, true, '校验中');
  $('upload-status').textContent = '正在读取并逐行校验…';
  try {
    const data = await fetchJson(API.uploadDataset, {
      method: 'POST',
      body: form,
      timeout: 120000,
    });
    if (data.success) {
      toast('数据集上传成功', 'success');
      input.value = '';
      $('dropzone-text').innerHTML = '点击或拖拽 CSV / Excel 文件至此';
      nameInput.value = '';
      $('upload-status').textContent = `${data.dataset.name} · ${data.dataset.rows} 行 · 已导入`;
      await loadDatasets();
    } else {
      throw new Error(data.error || '上传失败');
    }
  } catch (e) {
    $('upload-status').textContent = e.message;
    toast(e.message, 'error');
  } finally {
    setLoading(btn, false);
  }
}

let datasetDetail = { id: null, page: 1, total: 0, pageSize: 20 };
function renderDescriptionMarkdown(markdown) {
  // The local README uses a small Markdown subset. Escape before formatting;
  // links are shown as labels, never interpreted as untrusted HTML or URLs.
  const inline = value => escapeHtml(value)
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  let inList = false;
  const output = [];
  for (const raw of String(markdown || '').split('\n')) {
    const line = raw.trim();
    const item = line.match(/^-\s+(.+)/);
    if (item) {
      if (!inList) { output.push('<ul>'); inList = true; }
      output.push(`<li>${inline(item[1])}</li>`);
      continue;
    }
    if (inList) { output.push('</ul>'); inList = false; }
    const heading = line.match(/^#{1,3}\s+(.+)/);
    if (heading) output.push(`<h4>${inline(heading[1])}</h4>`);
    else if (line) output.push(`<p>${inline(line)}</p>`);
  }
  if (inList) output.push('</ul>');
  return output.join('');
}
async function previewDataset(id, page = 1) {
  try {
    const data = await fetchJson(`${API.previewDataset(encodeURIComponent(id))}?page=${page}`);
    if (!data.success) throw new Error(data.error);
    const rows = (data.preview || []).map(row => `<tr>${Object.values(row).map(v => `<td>${escapeHtml(v ?? '')}</td>`).join('')}</tr>`).join('');
    const cols = Object.keys(data.preview[0] || {}).map(c => `<th>${escapeHtml(c)}</th>`).join('');
    const stats = data.stats || {};
    datasetDetail = { id, page: data.page, total: data.total, pageSize: data.page_size };
    $('dataset-list-view').hidden = true;
    $('dataset-detail-view').hidden = false;
    $('dataset-detail-title').textContent = data.info.name;
    $('dataset-detail-path').textContent = data.info.file_path || '';
    $('dataset-detail-summary').textContent = `${stats.count} 行 · Tc ${formatMetric(stats.min, 2)}–${formatMetric(stats.max, 2)} K · 平均 ${formatMetric(stats.mean, 2)} K`;
    $('dataset-detail-description').innerHTML = renderDescriptionMarkdown(data.description);
    $('dataset-detail-head').innerHTML = `<tr>${cols}</tr>`;
    $('dataset-detail-body').innerHTML = rows || '<tr><td class="empty">没有数据</td></tr>';
    $('dataset-page').textContent = `第 ${data.page} / ${Math.max(1, Math.ceil(data.total / data.page_size))} 页`;
    $('dataset-prev').disabled = data.page <= 1;
    $('dataset-next').disabled = data.page * data.page_size >= data.total;
    window.scrollTo(0, 0);
  } catch (e) {
    toast(e.message, 'error');
  }
}

async function deleteDataset(id) {
  if (!confirm('删除该数据集及其本地文件？')) return;
  try {
    const data = await fetchJson(API.deleteDataset(id), { method: 'DELETE' });
    if (data.success) {
      toast('已删除', 'success');
      loadDatasets();
    } else {
      throw new Error(data.error);
    }
  } catch (e) {
    toast(e.message, 'error');
  }
}

// ------------------------------------------------------------------
// Workflow / screening
// ------------------------------------------------------------------

async function handleRunTask(taskName) {
  if (state.task.timer) {
    clearInterval(state.task.timer);
    state.task.timer = null;
  }

  const params = {};
  let body = {};

  if (taskName === 'alloy_predict') {
    const elements = $('workflow-elements').value.trim();
    if (!elements) return toast('请输入目标元素', 'error');
    params.elements = elements.split(/[,\s]+/).filter(Boolean);
    params.num_elements = parseInt($('workflow-num-elements').value) || 3;
    params.num_formulas = parseInt($('workflow-num-formulas').value) || 500;
    params.num_workers = Math.min(4, machineResources?.logical_cpus || 1);
    params.stability_check = $('workflow-stability').checked;
    params.model_key = $('workflow-model').value || undefined;
    params.dataset_id = $('workflow-dataset').value || undefined;
    body = { task_name: 'alloy_predict', params };
  } else {
    body = { task_name: taskName, params: {} };
  }

  const btn = $(`btn-${taskName}`);
  setLoading(btn, true, '启动中');
  $('workflow-log').textContent = '等待启动...\n';
  $('workflow-progress').textContent = '正在启动 · 预计剩余：估算中';
  $('workflow-result').style.display = 'none';

  try {
    const data = await fetchJson(API.runTask, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    state.task.id = data.task_id;
    state.task.name = taskName;
    state.task.started = performance.now();
    toast(`任务已启动: ${data.task_id.slice(0, 8)}`, 'success');
    startPolling();
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    setLoading(btn, false);
  }
}

function startPolling() {
  const log = $('workflow-log');
  log.textContent = '运行中，请稍候...\n';
  state.task.timer = setInterval(async () => {
    try {
      const data = await fetchJson(API.taskStatus(state.task.id));
      const p = data.progress || {};
      const amount = p.total != null ? `${p.completed}/${p.total} 组 · ` : '';
      $('workflow-progress').textContent = `${amount}已用 ${duration((performance.now() - state.task.started) / 1000)} · ${etaText(p.eta_seconds)}`;
      const rawStatus = data.status || 'running';
      if (rawStatus.startsWith('completed')) log.textContent = '筛选完成';
      else if (rawStatus.startsWith('failed') || rawStatus.startsWith('timeout')) log.textContent = rawStatus;
      else if (rawStatus.startsWith('stopped')) log.textContent = '任务已停止';
      else log.textContent = '筛选进行中…';
      log.scrollTop = log.scrollHeight;

      if (data.status && (data.status.startsWith('completed') || data.status.startsWith('failed') || data.status.startsWith('timeout'))) {
        clearInterval(state.task.timer);
        state.task.timer = null;
        if (data.status.startsWith('completed')) {
          toast('任务完成', 'success');
          loadTaskResult(state.task.name);
        } else {
          toast('任务失败/超时', 'error');
        }
      }
    } catch (e) {
      log.textContent += `\n查询失败: ${e.message}`;
    }
  }, 2000);
}

async function loadTaskResult(taskName) {
  try {
    const data = await fetchJson(API.taskResult(taskName));
    if (!data.ready) {
      $('workflow-result').innerHTML = `<p style="color:var(--danger)">${data.error}</p>`;
      $('workflow-result').style.display = 'block';
      return;
    }
    const rows = (data.top10 || []).map((r, i) => `<tr>
      <td>${i + 1}</td>
      <td title="${escapeHtml(r.Elements || r.elements || r.formula || '-')}">${escapeHtml(r.Elements || r.elements || r.formula || '-')}</td>
      <td><strong>${r.Predicted_Tc?.toFixed?.(2) ?? r.Predicted_Tc}</strong></td>
      <td>${r.stability_variance?.toFixed?.(2) ?? '-'}</td>
    </tr>`).join('');

    $('workflow-result').innerHTML = `
      <div class="panel-title">Top 10 候选 <a class="btn btn-secondary" style="padding:6px 12px;font-size:0.8rem" href="${API.taskDownload(taskName)}">下载 CSV</a></div>
      <table class="data-table">
        <thead><tr><th>#</th><th>Formula</th><th>Predicted Tc</th><th>Stability</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p style="color:var(--muted);font-size:0.82rem;margin-top:8px">共 ${data.total} 条结果 · ${data.file}</p>
    `;
    $('workflow-result').style.display = 'block';
  } catch (e) {
    console.error(e);
  }
}

function stopTask() {
  if (!state.task.id) return;
  fetch(API.stopTask(state.task.id), { method: 'POST' });
  if (state.task.timer) clearInterval(state.task.timer);
  toast('已发送停止命令', 'info');
}

// ------------------------------------------------------------------
// Visualization
// ------------------------------------------------------------------

let phaseResult = null;
function updatePhaseEstimate() {
  const elements = $('phase-elements').value.split(/[,，\s]+/).filter(Boolean);
  const steps = Number($('phase-steps').value);
  if (!Number.isInteger(steps) || steps < 4 || elements.length < 2 || elements.length > 10 || steps > (elements.length === 2 ? 1000 : 10000)) { $('phase-estimate').textContent = '请输入有效元素和等分数。'; return; }
  const points = elements.length === 2 ? steps + 1 : (steps + 1) * (steps + 2) / 2;
  const rough = points * (elements.length + 1) * 20 / 1073741824;
  $('phase-estimate').textContent = `${points.toLocaleString()} 个点 · CSV 大约 ${rough >= 1 ? `${rough.toFixed(1)} GB` : `${(rough * 1024).toFixed(1)} MB`}（粗估）${points > 100000 ? ' · 耗时可能很长' : ''}`;
}
async function generatePhase() {
  const elements = $('phase-elements').value.split(/[,，\s]+/).filter(Boolean);
  const steps = Number($('phase-steps').value), dpi = Number($('phase-dpi').value);
  const fixed = $('phase-fixed').value.trim() ? $('phase-fixed').value.split(/[,，\s]+/).filter(Boolean).map(Number) : [];
  const key = $('phase-model').value;
  if (!key) return toast('请先训练或加载模型', 'error');
  if (elements.length < 2 || elements.length > 10 || new Set(elements).size !== elements.length || elements.some(e => !/^[A-Z][a-z]?$/.test(e))) return toast('请输入 2–10 个不同的元素符号', 'error');
  if (!Number.isInteger(steps) || steps < 4 || steps > (elements.length === 2 ? 1000 : 10000) || !Number.isInteger(dpi) || dpi < 72 || dpi > 600 || fixed.length !== Math.max(0, elements.length - 3) || fixed.some(x => !Number.isFinite(x) || x < 0) || fixed.reduce((a,b) => a+b, 0) >= 100) return toast('请检查网格、DPI 和固定组分', 'error');
  const btn = $('btn-phase');
  setLoading(btn, true, '计算中');
  $('btn-phase-png').disabled = true;
  $('btn-phase-csv').disabled = true;
  $('phase-image-wrap').hidden = true;
  phaseResult = null;
  const count = elements.length === 2 ? steps + 1 : (steps + 1) * (steps + 2) / 2;
  if (count > 100000 && !confirm(`将计算 ${count.toLocaleString()} 个点并保存完整 CSV，可能耗时很长且占用大量磁盘。继续？`)) { setLoading(btn, false); return; }
  ['phase-elements', 'phase-model', 'phase-steps', 'phase-fixed', 'phase-dpi'].forEach(id => { $(id).disabled = true; });
  $('phase-progress').hidden = false;
  $('phase-progress').value = 0;
  $('phase-status').textContent = `正在预测 ${count} 个成分点…`;
  try {
    const started = await fetchJson('/api/visualize/phase/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ elements, model_key: key, steps, fixed_percentages: fixed, dpi }) });
    let data;
    do {
      await sleep(1000);
      data = await fetchJson(`/api/visualize/phase/jobs/${started.job_id}`);
      $('phase-progress').value = data.completed / data.total * 100;
      $('phase-status').textContent = `${data.stage || '计算中'} · ${data.completed.toLocaleString()}/${data.total.toLocaleString()} · ${etaText(data.eta_seconds)}`;
    } while (data.status === 'running');
    if (data.status !== 'completed') throw new Error(data.error || '计算失败');
    data.job_id = started.job_id;
    phaseResult = data;
    $('phase-image').src = `data:image/png;base64,${data.image}`;
    $('phase-image-wrap').hidden = false;
    $('btn-phase-png').disabled = false;
    $('btn-phase-csv').disabled = false;
    $('phase-status').textContent = `${data.model} · ${data.total.toLocaleString()} 点 · Tc ${formatMetric(data.min_tc, 2)}–${formatMetric(data.max_tc, 2)} K · 图片显示 ${Math.min(steps, 1000)} 等分`;
  } catch (e) { $('phase-status').textContent = e.message; toast(e.message, 'error'); }
  finally { setLoading(btn, false); ['phase-elements', 'phase-model', 'phase-steps', 'phase-fixed', 'phase-dpi'].forEach(id => { $(id).disabled = false; }); }
}

function downloadPhase(type) {
  if (!phaseResult) return;
  const link = document.createElement('a');
  if (type === 'png') link.href = `data:image/png;base64,${phaseResult.image}`;
  else link.href = `/api/visualize/phase/jobs/${phaseResult.job_id}/download`;
  link.download = `LMSC_${phaseResult.elements.join('-')}.${type}`;
  document.body.appendChild(link); link.click(); link.remove();
}

const vizCache = {};
const vizInFlight = new Set();
let vizGeneratingAll = false;
function invalidateDatasetPlots() {
  vizOrder.forEach(plotId => {
    delete vizCache[plotId];
    const card = document.querySelector(`.plot-card[data-plot="${plotId}"]`);
    if (!card) return;
    card.querySelector('.plot-image').style.display = 'none';
    card.querySelector('.plot-placeholder').textContent = '选择生成图表';
    card.querySelector('.plot-placeholder').style.display = 'block';
    card.querySelector('.btn-viz-download').disabled = true;
  });
  $('viz-status').textContent = '';
}
const vizOrder = ['fig2a', 'fig2b', 'fig2c'];
const vizNames = {
  fig2a: 'Tc 分布直方图',
  fig2b: '元素数量统计',
  fig2c: '元素周期表热图',
  fig4b: '数据收敛曲线',
  fig4c: '二元合金 Tc 热图',
  fig5: '预测 vs 实验',
  fig5d: 'RF / ET / 实验对比',
  ternary: 'Ga-In-Sn 三元相图',
};

function getVizParams() {
  return {
    model_type: $('viz-model')?.value || 'ET',
    tc_threshold: Number($('analysis-threshold')?.value),
    dataset_id: $('analysis-dataset').value || 'builtin',
    dpi: Number($('viz-dpi')?.value),
    elements: $('viz-elements').value.split(/[,，\s]+/).map(s => s.trim()).filter(Boolean),
  };
}

function setVizLoading(plotId, loading) {
  const card = document.querySelector(`.plot-card[data-plot="${plotId}"]`);
  if (!card) return;
  card.querySelector('.plot-loading').style.display = loading ? 'flex' : 'none';
  card.setAttribute('aria-busy', String(loading));
  const placeholder = card.querySelector('.plot-placeholder');
  if (placeholder) placeholder.style.display = loading ? 'none' : placeholder.style.display;
}

function showVizError(plotId, message) {
  const card = document.querySelector(`.plot-card[data-plot="${plotId}"]`);
  if (!card) return;
  setVizLoading(plotId, false);
  const img = card.querySelector('.plot-image');
  img.style.display = 'none';
  const placeholder = card.querySelector('.plot-placeholder');
  placeholder.className = 'plot-placeholder plot-error';
  placeholder.textContent = message;
  placeholder.style.display = 'block';
  card.querySelector('.btn-viz-download').disabled = true;
}

function displayVizImage(plotId, base64, info = '') {
  const card = document.querySelector(`.plot-card[data-plot="${plotId}"]`);
  if (!card) return;
  setVizLoading(plotId, false);
  const img = card.querySelector('.plot-image');
  img.src = `data:image/png;base64,${base64}`;
  img.alt = vizNames[plotId];
  img.title = '点击放大查看';
  img.tabIndex = 0;
  img.setAttribute('role', 'button');
  img.setAttribute('aria-label', `放大查看：${vizNames[plotId]}`);
  img.style.display = 'block';
  const placeholder = card.querySelector('.plot-placeholder');
  placeholder.style.display = 'none';
  placeholder.className = 'plot-placeholder';
  placeholder.textContent = '点击“生成所有图表”渲染';
  vizCache[plotId] = base64;
  card.querySelector('.btn-viz-download').disabled = false;
}

function syncVizButtons() {
  const busy = vizGeneratingAll || vizInFlight.size > 0;
  $$('.btn-viz-single').forEach(btn => { btn.disabled = busy; });
  $('btn-viz-generate').disabled = busy;
  $('btn-viz-reset').disabled = busy;
  ['viz-model', 'viz-threshold', 'viz-dpi', 'viz-elements', 'analysis-dataset', 'analysis-threshold', 'btn-analysis'].forEach(id => { $(id).disabled = busy; });
}

async function generateVizPlot(plotId) {
  const card = document.querySelector(`.plot-card[data-plot="${plotId}"]`);
  if (!card || vizInFlight.has(plotId)) return false;
  const params = getVizParams();
  if (!Number.isInteger(params.dpi) || params.dpi < 72 || params.dpi > 600 || !Number.isFinite(params.tc_threshold) || params.tc_threshold < 0) {
    toast('请填写有效参数：DPI 为 72–600 的整数，Tc 阈值不小于 0。', 'error');
    return false;
  }
  if (plotId === 'ternary' && (params.elements.length !== 3 || new Set(params.elements).size !== 3 || params.elements.some(e => !/^[A-Z][a-z]?$/.test(e)))) {
    showVizError(plotId, '请输入三个不同的元素符号，例如 Ga,In,Sn');
    return false;
  }
  vizInFlight.add(plotId);
  syncVizButtons();
  setVizLoading(plotId, true);
  let url = `/api/visualize/dataset/${encodeURIComponent(plotId)}?dpi=${params.dpi}&threshold=${encodeURIComponent(params.tc_threshold)}&dataset_id=${encodeURIComponent(params.dataset_id)}`;
  if (['fig4c', 'fig5', 'ternary'].includes(plotId)) url += `&model_type=${params.model_type}`;
  if (plotId === 'ternary') url += `&elements=${encodeURIComponent(params.elements.join(','))}`;

  try {
    const data = await fetchJson(url, { timeout: 300000 });
    if (data.success) {
      if (plotId === 'ternary') {
        vizNames.ternary = `${params.elements.join('-')} 三元相图`;
        card.querySelector('.plot-title').textContent = vizNames.ternary;
      }
      displayVizImage(plotId, data.image, data.data_info ? JSON.stringify(data.data_info) : '');
      return true;
    } else {
      showVizError(plotId, data.error || '生成失败');
    }
  } catch (e) {
    const msg = e.message?.includes('404') || e.message?.includes('not loaded')
      ? '图表模块未启用，请重新安装项目依赖后重启服务。'
      : `生成失败: ${e.message}`;
    showVizError(plotId, msg);
  } finally {
    vizInFlight.delete(plotId);
    syncVizButtons();
  }
  return false;
}

async function handleGenerateAllViz(btn) {
  if (vizGeneratingAll || vizInFlight.size) return;
  vizGeneratingAll = true;
  syncVizButtons();
  const status = $('viz-status');
  status.style.color = '';
  status.textContent = '正在生成图表，请稍候...';
  const started = performance.now();
  setLoading(btn, true, '生成中');
  try {
    let succeeded = 0;
    for (const plotId of vizOrder) {
      const done = vizOrder.indexOf(plotId);
      const elapsed = (performance.now() - started) / 1000;
      status.textContent = `生成 ${done + 1}/${vizOrder.length}：${vizNames[plotId]} · ${etaText(done ? elapsed / done * (vizOrder.length - done) : null)}`;
      if (await generateVizPlot(plotId)) succeeded++;
    }
    status.style.color = succeeded === vizOrder.length ? 'var(--success)' : 'var(--warning)';
    status.textContent = `生成结束：${succeeded}/${vizOrder.length} 张成功${succeeded < vizOrder.length ? '，失败图表可单独重试。' : '。点击图片放大，或保存 PNG。'}`;
  } catch (e) {
    status.style.color = 'var(--danger)';
    status.textContent = `生成中断: ${e.message}`;
  } finally {
    setLoading(btn, false);
    vizGeneratingAll = false;
    syncVizButtons();
  }
}

function handleDownloadViz(plotId) {
  const base64 = vizCache[plotId];
  if (!base64) return toast('图片未生成', 'error');
  const link = document.createElement('a');
  link.href = `data:image/png;base64,${base64}`;
  link.download = `LMSC_${plotId}_${new Date().toISOString().slice(0,10)}.png`;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function handleResetViz() {
  if (vizGeneratingAll || vizInFlight.size) return;
  $('viz-model').value = 'ET';
  $('viz-threshold').value = '5';
  $('viz-dpi').value = '300';
  $('viz-elements').value = 'Ga,In,Sn';
  vizNames.ternary = 'Ga-In-Sn 三元相图';
  document.querySelector('.plot-card[data-plot="ternary"] .plot-title').textContent = vizNames.ternary;
  const status = $('viz-status');
  status.textContent = '';
  status.style.color = '';
  document.querySelectorAll('.plot-card').forEach(card => {
    const plotId = card.dataset.plot;
    delete vizCache[plotId];
    card.querySelector('.plot-image').style.display = 'none';
    const placeholder = card.querySelector('.plot-placeholder');
    placeholder.textContent = '点击“生成所有图表”渲染';
    placeholder.className = 'plot-placeholder';
    placeholder.style.display = 'block';
    card.querySelector('.btn-viz-download').disabled = true;
  });
}

async function checkVisualizationAvailable() {
  const status = $('viz-status');
  const allBtn = $('btn-viz-generate');
  const singleBtns = document.querySelectorAll('.btn-viz-single');
  try {
    await fetchJson(API.visualize('list'));
    if (vizGeneratingAll || vizInFlight.size) return;
    if (Object.keys(vizCache).length) return;
    status.style.color = '';
    status.textContent = '可视化模块已就绪，点击“生成所有图表”开始渲染。';
    allBtn.disabled = false;
    singleBtns.forEach(b => b.disabled = false);
  } catch (e) {
    const msg = '图表模块未启用，请重新运行安装脚本后重启服务。';
    status.textContent = msg;
    status.style.color = 'var(--danger)';
    allBtn.disabled = true;
    singleBtns.forEach(b => b.disabled = true);
  }
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

function escapeHtml(text) {
  return String(text ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

// ------------------------------------------------------------------
// Init
// ------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
  $('btn-check-update').addEventListener('click', checkUpdate);
  $('btn-apply-update').addEventListener('click', applyUpdate);
  // Give all controls programmatic labels without duplicating template copy.
  $$('.form-group').forEach(group => {
    const label = group.querySelector('label');
    const control = group.querySelector('input,select,textarea');
    if (label && control?.id) label.htmlFor = control.id;
  });
  $$('.nav-icon').forEach(el => el.setAttribute('aria-hidden', 'true'));
  $('predict-result').setAttribute('aria-live', 'polite');
  $('batch-result').setAttribute('aria-live', 'polite');
  switchTab(window.location.hash.slice(1) || 'dashboard');
  loadDatasets();
  loadResourceInfo();
  initUpload();

  $('menu-toggle').addEventListener('click', () => setSidebarOpen(!$('sidebar').classList.contains('open')));
  $('sidebar-backdrop').addEventListener('click', () => setSidebarOpen(false));
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && $('sidebar').classList.contains('open')) {
      setSidebarOpen(false);
      $('menu-toggle').focus();
    }
  });
  window.addEventListener('hashchange', () => switchTab(window.location.hash.slice(1)));
  const themeBtn = $('theme-toggle');
  const updateThemeLabel = () => {
    const dark = document.documentElement.dataset.theme === 'dark';
    themeBtn.textContent = dark ? '☾' : '☼';
    themeBtn.setAttribute('aria-label', dark ? '切换到浅色主题' : '切换到深色主题');
    themeBtn.title = themeBtn.getAttribute('aria-label');
  };
  updateThemeLabel();
  themeBtn.addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('lmsc-theme', theme); } catch (_) { /* storage is optional */ }
    updateThemeLabel();
  });
  const quickPredict = () => {
    const formula = $('dash-formula').value.trim();
    if (!formula) { $('dash-formula').focus(); return toast('请输入化学式，或选择下方示例。'); }
    $('predict-input').value = formula;
    switchTab('predict');
    $('predict-input').focus();
    handlePredict();
  };
  $('btn-dash-predict').addEventListener('click', quickPredict);
  $('dash-formula').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.isComposing) quickPredict(); });
  document.addEventListener('click', e => {
    const chip = e.target.closest('[data-formula]');
    if (chip) {
      const input = $(chip.dataset.target || 'dash-formula');
      input.value = chip.dataset.formula;
      input.focus();
    }
    const link = e.target.closest('[data-open-tab]');
    if (link) switchTab(link.dataset.openTab);
  });
  $('btn-clear-predict').addEventListener('click', () => {
    $('predict-input').value = '';
    $('predict-result').style.display = 'none';
    $('predict-input').focus();
  });
  $('batch-input').addEventListener('input', updateBatchCount);
  $('btn-clear-batch').addEventListener('click', () => {
    $('batch-input').value = '';
    $('batch-result').style.display = 'none';
    state.batchResults = [];
    updateBatchCount();
    $('batch-input').focus();
  });

  // Bind buttons
  $('btn-load-model')?.addEventListener('click', handleLoadModel);
  $('btn-predict')?.addEventListener('click', handlePredict);
  $('predict-input')?.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.isComposing) handlePredict(); });
  $('btn-batch-predict')?.addEventListener('click', handleBatchPredict);
  $('btn-train')?.addEventListener('click', () => handleTrain(false));
  $('btn-dash-train')?.addEventListener('click', () => handleTrain(true));
  $('btn-upload')?.addEventListener('click', handleUpload);
  $('datasets-body').addEventListener('click', e => {
    const detail = e.target.closest('.dataset-detail-button');
    const remove = e.target.closest('.dataset-delete-button');
    if (detail) previewDataset(detail.dataset.id);
    if (remove) deleteDataset(remove.dataset.id);
  });
  $('btn-dataset-back').addEventListener('click', () => { $('dataset-detail-view').hidden = true; $('dataset-list-view').hidden = false; });
  $('dataset-prev').addEventListener('click', () => previewDataset(datasetDetail.id, datasetDetail.page - 1));
  $('dataset-next').addEventListener('click', () => previewDataset(datasetDetail.id, datasetDetail.page + 1));
  $('btn-analyze-dataset').addEventListener('click', () => { populateAnalysisDatasets(); $('analysis-dataset').value = datasetDetail.id; switchTab('visualization'); });
  $('btn-open-analysis').addEventListener('click', () => switchTab('visualization'));
  $('btn-analysis-back').addEventListener('click', () => switchTab('data'));
  $$('.analysis-download').forEach(button => button.addEventListener('click', () => downloadAnalysisChart(button.dataset.chart)));
  $('btn-analysis').addEventListener('click', loadDatasetAnalysis);
  $('btn-phase').addEventListener('click', generatePhase);
  $('btn-phase-png').addEventListener('click', () => downloadPhase('png'));
  $('btn-phase-csv').addEventListener('click', () => downloadPhase('csv'));
  ['phase-elements', 'phase-model', 'phase-steps', 'phase-fixed', 'phase-dpi'].forEach(id => {
    $(id).addEventListener('input', () => {
      updatePhaseEstimate();
      if (!phaseResult) return;
      phaseResult = null;
      $('btn-phase-png').disabled = true;
      $('btn-phase-csv').disabled = true;
      $('phase-image-wrap').hidden = true;
      $('phase-status').textContent = '参数已更改，请重新生成。';
    });
  });
  updatePhaseEstimate();
  $('analysis-dataset').addEventListener('change', loadDatasetAnalysis);
  $('analysis-threshold').addEventListener('keydown', e => { if (e.key === 'Enter') loadDatasetAnalysis(); });
  ['analysis-dataset', 'analysis-threshold'].forEach(id => $(id).addEventListener('input', invalidateDatasetPlots));
  $('btn-alloy_predict')?.addEventListener('click', () => handleRunTask('alloy_predict'));
  $('btn-stop-task')?.addEventListener('click', stopTask);

  // Auto-load model when a prediction model select changes
  ['predict-model', 'batch-model', 'workflow-model'].forEach(id => {
    $(id)?.addEventListener('change', e => {
      const key = e.target.value;
      if (key) loadModelByKey(key);
    });
  });

  // Local models: sorting + batch operations
  $$('#models-table th.sortable').forEach(th => {
    th.addEventListener('click', () => handleSortModels(th.dataset.sort));
    th.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        handleSortModels(th.dataset.sort);
      }
    });
  });
  $('models-select-all')?.addEventListener('change', e => handleSelectAllModels(e.target.checked));
  $('btn-models-clear-selection')?.addEventListener('click', clearModelSelection);
  $('btn-models-batch-delete')?.addEventListener('click', handleBatchDeleteModels);

  // Nav
  $$('.nav-item').forEach(el => {
    if (el.dataset.tab) {
      el.addEventListener('click', () => switchTab(el.dataset.tab));
    }
  });

  // Visualization bindings
  $('btn-viz-generate')?.addEventListener('click', () => handleGenerateAllViz($('btn-viz-generate')));
  $('btn-viz-reset')?.addEventListener('click', handleResetViz);
  document.querySelectorAll('.btn-viz-single').forEach(btn => {
    btn.addEventListener('click', () => generateVizPlot(btn.dataset.plot));
  });
  document.querySelectorAll('.btn-viz-download').forEach(btn => {
    btn.addEventListener('click', () => handleDownloadViz(btn.dataset.plot));
  });
  const openImage = img => {
    if (!img.getAttribute('src')) return;
    $('image-dialog-title').textContent = img.alt;
    $('image-dialog-img').src = img.src;
    $('image-dialog-img').alt = img.alt;
    $('image-dialog').showModal();
  };
  $$('.plot-image').forEach(img => {
    img.addEventListener('click', () => openImage(img));
    img.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openImage(img); }
    });
  });
  $('btn-close-image').addEventListener('click', () => $('image-dialog').close());
  $('image-dialog').addEventListener('click', e => { if (e.target === $('image-dialog')) $('image-dialog').close(); });
  // Status must settle before choosing the model in every select.
  await refreshStatus();
  await refreshAllModelSelects();
});
