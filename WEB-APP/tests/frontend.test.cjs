const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function createContext() {
  const elements = new Map();
  const document = {
    addEventListener() {},
    querySelectorAll() { return []; },
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, {
        innerHTML: '', textContent: '', hidden: true, style: {},
        addEventListener() {},
      });
      return elements.get(id);
    },
  };
  const context = vm.createContext({ document, console, setTimeout, clearTimeout });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../app/static/app.js'), 'utf8'), context);
  return { run: code => vm.runInContext(code, context), elements };
}

test('metrics retain integer zeros and handle unavailable values', () => {
  const { run } = createContext();
  assert.equal(run('formatMetric(120, 0)'), '120');
  assert.equal(run('formatMetric(0, 0)'), '0');
  assert.equal(run('formatMetric(1.230)'), '1.23');
  assert.equal(run('formatMetric(null)'), '-');
  assert.equal(run('formatMetric(Infinity)'), '-');
});

test('chemical formula input accepts common separators without deduplicating', () => {
  const { run } = createContext();
  assert.equal(run('JSON.stringify(parseBatchInput("MgB2，YBa2Cu3O7;\\nMgB2；Ba0.5In0.5"))'),
    '["MgB2","YBa2Cu3O7","MgB2","Ba0.5In0.5"]');
  assert.equal(run('parseBatchInput(" , ;； ").length'), 0);
});

test('HTML escaping protects text and quoted attributes', () => {
  const { run } = createContext();
  assert.equal(run('escapeHtml(null)'), '');
  assert.equal(run('escapeHtml(`<img src="x" onerror=\'alert(1)\'>&`)'),
    '&lt;img src=&quot;x&quot; onerror=&#39;alert(1)&#39;&gt;&amp;');
});

test('CSV exports quote data and neutralize spreadsheet formulas', () => {
  const { run } = createContext();
  assert.equal(run('csvCell("MgB2")'), '"MgB2"');
  assert.equal(run('csvCell("=1+1")'), '"\'=1+1"');
  assert.equal(run('csvCell(`a,"b"`)'), '"a,""b"""');
});

test('batch view reports partial failures and escapes error content', () => {
  const { run, elements } = createContext();
  run(`renderBatchResults([
    { formula: 'MgB2', predicted_tc: 38.5, reliability: { level: '高', score: 80 } },
    { formula: '<bad>', error: '<img src=x onerror=alert(1)>' }
  ])`);
  const html = elements.get('batch-result').innerHTML;
  assert.match(html, /1<\/strong> 条成功 · 1 条失败/);
  assert.match(html, /38.5 K/);
  assert.match(html, /Top 1/);
  assert.match(html, /&lt;img/);
  assert.doesNotMatch(html, /<img/);
});

test('batch view handles an entirely invalid batch without a chart', () => {
  const { run, elements } = createContext();
  run(`renderBatchResults([{ formula: 'bad', error: 'Invalid formula' }])`);
  assert.match(elements.get('batch-result').innerHTML, /0<\/strong> 条成功 · 1 条失败/);
  assert.doesNotMatch(elements.get('batch-result').innerHTML, /class="chart-wrap"/);
});

test('prediction errors are rendered as text rather than executable markup', () => {
  const { run, elements } = createContext();
  run(`renderPredictionResult({ error: '<script>alert(1)</script>' })`);
  assert.match(elements.get('predict-result').innerHTML, /&lt;script&gt;/);
  assert.doesNotMatch(elements.get('predict-result').innerHTML, /<script>/);
});

test('dataset description renders markdown safely', () => {
  const { run } = createContext();
  const html = run('renderDescriptionMarkdown("## 数据说明\\n- **目标**：`Tc_K`\\n- <img src=x onerror=alert(1)>")');
  assert.match(html, /<h4>数据说明<\/h4>/);
  assert.match(html, /<strong>目标<\/strong>/);
  assert.match(html, /<code>Tc_K<\/code>/);
  assert.match(html, /&lt;img/);
  assert.doesNotMatch(html, /<img/);
});
