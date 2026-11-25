/* Poll the run JSON and refresh the visible numbers while training is active. */
(function () {
  const root = document.getElementById('run');
  if (!root || !window.RUN_POLL_URL) return;

  const el = (id) => document.getElementById(id);
  const fmt = (v, digits) => {
    if (v === null || v === undefined) return '—';
    const n = Number(v);
    if (!Number.isFinite(n)) return String(v);
    return n.toPrecision(digits || 10).replace(/(\.\d*?)0+(e|$)/, '$1$2');
  };
  let lastStatus = root.dataset.status;
  let lastUpdates = null;

  function rowHtml(u) {
    return `<tr>
      <td>${u.idx + 1}</td><td>${u.samples}</td>
      <td class="mono">${fmt(u.grad_w, 6)} / ${fmt(u.grad_b, 6)}</td>
      <td class="mono">${fmt(u.w_after)}</td><td class="mono">${fmt(u.b_after)}</td>
      <td class="mono">${fmt(u.v_w, 6)}</td><td class="mono">${fmt(u.v_b, 6)}</td>
      <td class="mono">${fmt(u.loss_mean, 6)}</td><td>${fmt(u.group_ms, 4)}</td>
    </tr>`;
  }

  async function poll() {
    let data;
    try {
      const res = await fetch(window.RUN_POLL_URL, { cache: 'no-store' });
      if (!res.ok) return;
      data = await res.json();
    } catch (err) {
      return;
    }
    const run = data.run;
    el('m-updates').textContent = run.updates_done;
    el('m-samples').textContent = run.samples_done;
    const part = el('m-part');
    if (part) part.textContent = run.part_rows;
    el('m-w').textContent = fmt(run.w);
    el('m-b').textContent = fmt(run.b);
    el('m-vw').textContent = fmt(run.v_w);
    el('m-vb').textContent = fmt(run.v_b);
    el('message').textContent = run.message;
    const precise = el('precise-values');
    if (precise) {
      const g = (v) => Number(v).toPrecision(17);
      precise.textContent =
        `w = ${g(run.w)}\nb = ${g(run.b)}\nv_w = ${g(run.v_w)}\nv_b = ${g(run.v_b)}`;
    }
    const bar = el('bar');
    if (bar && run.total_rows) {
      const done = Number(run.samples_done) + Number(run.part_rows || 0);
      bar.style.width = (100 * done / run.total_rows).toFixed(2) + '%';
    }
    const body = el('updates-body');
    if (body && data.updates && data.updates.length &&
        (lastUpdates === null || data.updates[0].idx !== lastUpdates)) {
      lastUpdates = data.updates[0].idx;
      body.innerHTML = data.updates.map(rowHtml).join('');
    }
    if (run.status !== lastStatus) {
      // Status changes also change which buttons are enabled: reload once.
      lastStatus = run.status;
      window.location.reload();
      return;
    }
    const active = run.status === 'running' || data.worker_running;
    if (active) {
      setTimeout(poll, 700);
    } else {
      // One extra deferred poll catches a worker that finished in between.
      setTimeout(() => { if (run.status !== 'running') poll(); }, 1500);
    }
  }

  if (lastStatus === 'running' || root.dataset.workerRunning === '1') {
    setTimeout(poll, 400);
  }
})();
