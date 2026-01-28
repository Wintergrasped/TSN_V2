document.addEventListener("DOMContentLoaded", () => {
  const refreshers = document.querySelectorAll("[data-refresh]");
  refreshers.forEach((button) => {
    button.addEventListener("click", async () => {
      const target = button.getAttribute("data-refresh");
      if (!target) return;
      button.disabled = true;
      try {
        const res = await fetch(target);
        if (!res.ok) throw new Error("Refresh failed");
        const payload = await res.json();
        console.info("Refresh payload", payload);
      } catch (err) {
        console.error(err);
        alert("Unable to refresh; check logs");
      } finally {
        button.disabled = false;
      }
    });
  });

  const feedButton = document.getElementById("net-feed-refresh");
  const feedTable = document.querySelector("#net-control-feed tbody");
  const feedUrl = feedButton?.dataset.feedUrl;

  async function refreshFeed() {
    if (!feedUrl || !feedTable) return;
    feedButton.disabled = true;
    try {
      const res = await fetch(feedUrl);
      if (!res.ok) throw new Error("feed failed");
      const rows = await res.json();
      feedTable.innerHTML = rows
        .map(
          (row) => `
            <tr>
              <td>${row.detected_at}</td>
              <td>${row.callsign_id}</td>
              <td>${row.confidence ?? "—"}</td>
              <td>${(row.context || row.transcript || "").slice(0, 240)}</td>
            </tr>`
        )
        .join("");
    } catch (error) {
      console.error("feed_refresh_error", error);
    } finally {
      feedButton.disabled = false;
    }
  }

  if (feedButton && feedTable) {
    feedButton.addEventListener("click", (event) => {
      event.preventDefault();
      refreshFeed();
    });
    setInterval(refreshFeed, 15000);
  }

  const htmlEscapes = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => htmlEscapes[char] || char);
  }

  function formatDate(value) {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return String(value);
    }
    return date.toLocaleString();
  }

  function formatNumber(value) {
    const numeric = Number(value ?? 0);
    if (Number.isNaN(numeric)) {
      return "0";
    }
    return numeric.toLocaleString();
  }

  function renderCallsignRows(rows, options) {
    if (!rows.length) {
      return `<tr><td colspan="${options.columns}" class="muted">No callsigns available yet.</td></tr>`;
    }
    return rows
      .map((row) => {
        const callsign = row.callsign || "Unknown";
        const method = (row.validation_method || "Unverified").toUpperCase();
        const mentions = formatNumber(row.seen_count);
        const firstSeen = formatDate(row.first_seen);
        const lastSeen = formatDate(row.last_seen);
        const searchText = `${callsign} ${method} ${mentions}`;
        return `
          <tr data-search-text="${escapeHtml(searchText)}">
            <td><a href="/callsigns/${encodeURIComponent(callsign)}">${escapeHtml(callsign)}</a></td>
            <td>${escapeHtml(firstSeen)}</td>
            <td>${escapeHtml(lastSeen)}</td>
            <td>${escapeHtml(mentions)}</td>
            <td>${escapeHtml(method)}</td>
          </tr>`;
      })
      .join("");
  }

  function renderRowsForType(type, rows, options) {
    if (type === "callsign") {
      return renderCallsignRows(rows, options);
    }
    if (!rows.length) {
      return `<tr><td colspan="${options.columns}" class="muted">No data available.</td></tr>`;
    }
    return rows
      .map((row) => `<tr><td colspan="${options.columns}">${escapeHtml(JSON.stringify(row))}</td></tr>`)
      .join("");
  }

  async function refreshAsyncTable(tbody) {
    const endpoint = tbody.dataset.endpoint;
    if (!endpoint) return;
    const emptySelector = tbody.dataset.empty;
    const emptyEl = emptySelector ? document.querySelector(emptySelector) : null;
    const columns = Number.parseInt(tbody.dataset.columns || "1", 10);
    if (emptyEl) emptyEl.hidden = true;
    tbody.innerHTML = `<tr><td colspan="${columns}" class="muted">Loading…</td></tr>`;
    try {
      const response = await fetch(endpoint);
      if (!response.ok) throw new Error(`Request failed: ${response.status}`);
      const payload = await response.json();
      const rows = Array.isArray(payload) ? payload : [];
      const html = renderRowsForType(tbody.dataset.rowType || "json", rows, { columns });
      tbody.innerHTML = html;
      if (emptyEl) emptyEl.hidden = rows.length !== 0;
    } catch (error) {
      console.error("async_table_error", error);
      tbody.innerHTML = `<tr><td colspan="${columns}" class="muted">Unable to load data.</td></tr>`;
      if (emptyEl) emptyEl.hidden = false;
    } finally {
      dispatchSearchUpdates();
    }
  }

  function initAsyncTables() {
    document.querySelectorAll("[data-async-table]").forEach((tbody) => {
      refreshAsyncTable(tbody);
    });
  }

  function initAsyncReloads() {
    document.querySelectorAll("[data-async-reload]").forEach((button) => {
      const selector = button.dataset.asyncReload;
      if (!selector) return;
      button.addEventListener("click", async () => {
        const target = document.querySelector(selector);
        if (!target) return;
        button.disabled = true;
        try {
          await refreshAsyncTable(target);
        } finally {
          button.disabled = false;
        }
      });
    });
  }

  function applySearchFilter(input) {
    const selector = input.dataset.searchTarget;
    if (!selector) return;
    const emptySelector = input.dataset.searchEmpty;
    const emptyEl = emptySelector ? document.querySelector(emptySelector) : null;
    const query = input.value.trim().toLowerCase();
    let matches = 0;
    document.querySelectorAll(selector).forEach((node) => {
      const text = (node.dataset.searchText || node.textContent || "").toLowerCase();
      const visible = !query || text.includes(query);
      node.style.display = visible ? "" : "none";
      if (visible) matches += 1;
    });
    if (emptyEl) {
      emptyEl.hidden = matches !== 0;
    }
  }

  function initSearchFilters() {
    document.querySelectorAll("[data-search-target]").forEach((input) => {
      input.addEventListener("input", () => applySearchFilter(input));
      applySearchFilter(input);
    });
  }

  function dispatchSearchUpdates() {
    document.querySelectorAll("[data-search-target]").forEach((input) => applySearchFilter(input));
  }

  initSearchFilters();
  initAsyncTables();
  initAsyncReloads();
});
