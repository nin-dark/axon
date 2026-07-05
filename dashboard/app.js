const API_BASE = "http://localhost:8000";
const WS_BASE  = "ws://localhost:8000";

let ws = null;
let totalSources = 0;

function escapeHtml(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}


// ── QUERY SUBMISSION ──────────────────────────────────────────────────────────

document.getElementById("submit-btn").addEventListener("click", () => {
    const prompt = document.getElementById("prompt-input").value.trim();
    if (!prompt) return;
    runQuery(prompt);
});

document.getElementById("prompt-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") document.getElementById("submit-btn").click();
});


const API_KEY = "my-super-secret-local-key";
const USER_ID = "user_sales_ny";

async function runQuery(prompt) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    await resetUI();
    window.queryResults = []; // Store raw data
    setStatus("Connecting...");
    document.getElementById("submit-btn").disabled = true;

    // Pass the required security parameters as query parameters
    ws = new WebSocket(`${WS_BASE}/ws/query?api_key=${API_KEY}&user_id=${USER_ID}`);

    ws.onopen = () => {
        ws.send(JSON.stringify({ prompt }));
        setStatus("Generating SQL...");
    };
    ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
    ws.onerror = () => {
        setStatus("Connection failed — is the server running?");
        document.getElementById("submit-btn").disabled = false;
    };
    ws.onclose = () => {
        document.getElementById("submit-btn").disabled = false;
    };
}


function handleMessage(data) {

    if (data.type === "error") {
        setStatus(`Error: ${data.message}`);
        return;
    }

    if (data.type === "timeout") {
        setStatus("Approval timed out — query cancelled");
        return;
    }

    if (data.type === "destructive_warning") {
        document.getElementById("destructive-warning").classList.remove("hidden");
        document.getElementById("sql-panel").classList.add("hidden");
        hideApprovalButtons();
        setStatus("Waiting for confirmation on destructive command...");
        return;
    }

    // ── SUGGESTION FROM TFIDF ─────────────────────────────────────
    if (data.type === "suggestion") {
        showSuggestionPanel(data.suggestion, data.intent_key);
        return;
    }

    // ── SQL READY — SHOW APPROVAL UI ─────────────────────────────
    if (data.type === "approval_required") {
        window.originalSql = data.sql; // Store original for Active Learning comparison
        document.getElementById("sql-display").textContent = data.sql;
        document.getElementById("sql-panel").classList.remove("hidden");
        document.getElementById("cost-display").textContent =
            data.from_cache
                ? "Cost: $0.000000 (cache hit)"
                : `Est. cost: $${(data.cost_usd || 0).toFixed(8)}`;

        showApprovalButtons(data.sql);
        setStatus("Review the generated SQL and approve or reject");
        return;
    }

    if (data.type === "rejected") {
        setStatus("Query rejected");
        hideApprovalButtons();
        return;
    }

    if (data.type === "db_result") {
        updateSourceCell(data.database, data.status);
        document.getElementById("grid-counter").textContent =
            `${data.completed} / ${data.total}`;
        setStatus(`${data.completed} / ${data.total} (${data.progress_pct}%)`);
        
        if (data.status === "success" && data.data && data.data.length > 0) {
            window.queryResults.push({
                database: data.database,
                rows: data.data.length,
                data: data.data
            });
        }
        return;
    }

    if (data.type === "complete") {
        renderInsightReport(data.insight_report, data.ml_analysis);
        renderRawData();
        setStatus(`Complete — ${data.total_completed} sources queried`);
        loadMetrics();
        loadVault();
        return;
    }
}


// ── APPROVAL UI ───────────────────────────────────────────────────────────────

function showApprovalButtons(sql) {
    let panel = document.getElementById("approval-buttons");
    if (!panel) {
        panel = document.createElement("div");
        panel.id = "approval-buttons";
        panel.className = "approval-panel";
        document.getElementById("sql-panel").after(panel);
    }
    
    // Default Read-Only Mode Buttons
    panel.innerHTML = `
        <button onclick="sendApproval(true)" class="approval-btn approve">
            ✓ Approve & Execute
        </button>
        <button onclick="enableEditMode()" class="approval-btn reject" style="background:#21262d; color:white; border: 1px solid #30363d;">
            ✎ Reject & Edit
        </button>
    `;
    panel.style.display = "flex";
}

function enableEditMode() {
    const sqlDisplay = document.getElementById("sql-display");
    sqlDisplay.contentEditable = "true";
    sqlDisplay.focus();
    document.getElementById("edit-indicator").style.display = "block";
    
    let panel = document.getElementById("approval-buttons");
    panel.innerHTML = `
        <button onclick="sendApproval(true)" class="approval-btn" style="background:var(--accent); color:#fff; border:none;">
            🚀 Run Custom Query
        </button>
        <button onclick="cancelEditMode()" class="approval-btn" style="background:var(--bg-input); color:var(--text-primary); border: 1px solid var(--border-color);">
            Cancel
        </button>
    `;
}

function cancelEditMode() {
    // Revert edits and go back to original SQL
    document.getElementById("sql-display").textContent = window.originalSql;
    document.getElementById("sql-display").contentEditable = "false";
    document.getElementById("edit-indicator").style.display = "none";
    showApprovalButtons(window.originalSql);
}

function hideApprovalButtons() {
    const panel = document.getElementById("approval-buttons");
    if (panel) panel.style.display = "none";
}

function sendApproval(approved, forceDestructive = false) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        let payload = { approved, force_destructive: forceDestructive };
        
        if (approved) {
            const currentSql = document.getElementById("sql-display").textContent.trim();
            if (window.originalSql && currentSql !== window.originalSql.trim()) {
                payload.corrected_sql = currentSql;
            }
            setStatus(payload.corrected_sql ? "Validating & running custom query..." : "Approved — querying sources...");
        }

        ws.send(JSON.stringify(payload));
        hideApprovalButtons();
        
        // Reset edit mode visuals for next time
        document.getElementById("sql-display").contentEditable = "false";
        document.getElementById("edit-indicator").style.display = "none";
    }
}

function forceDestructive() {
    document.getElementById("destructive-warning").classList.add("hidden");
    document.getElementById("sql-panel").classList.remove("hidden");
    sendApproval(true, true);
}

function cancelDestructive() {
    document.getElementById("destructive-warning").classList.add("hidden");
    setStatus("Query cancelled by user.");
}


// ── SUGGESTION PANEL ──────────────────────────────────────────────────────────

function showSuggestionPanel(suggestion, intentKey) {
    let panel = document.getElementById("suggestion-panel");
    if (!panel) {
        panel = document.createElement("div");
        panel.id = "suggestion-panel";
        panel.className = "suggestion-panel";
        document.querySelector(".query-panel").after(panel);
    }
    const matchPct = (suggestion.similarity * 100).toFixed(1);
    panel.innerHTML = `
        <div style="color:#d29922;font-size:13px;font-weight:600;margin-bottom:8px;letter-spacing:1px;">
            SIMILAR QUERY FOUND IN MEMPALACE (${matchPct}% match)
        </div>
        <div style="color:var(--text-secondary);margin-bottom:8px;">Previous intent: ${escapeHtml(suggestion.intent_key)}</div>
        <pre style="color:#a5d6ff;font-family:'Fira Code', monospace;margin-bottom:20px;background:rgba(0,0,0,0.2);padding:16px;border-radius:8px;">${escapeHtml(suggestion.sql)}</pre>
        <div style="display:flex;gap:12px;">
            <button id="accept-suggestion-btn" class="approval-btn" style="background:#d29922;color:#161b22;">
                Use this cached query
            </button>
            <button id="decline-suggestion-btn" class="approval-btn" style="background:rgba(255,255,255,0.05);color:var(--text-primary);border:1px solid var(--border-color);">
                Call AI instead
            </button>
        </div>
    `;
    document.getElementById("accept-suggestion-btn").addEventListener("click", () => {
        acceptSuggestion(intentKey, suggestion.sql);
    });
    document.getElementById("decline-suggestion-btn").addEventListener("click", () => {
        declineSuggestion(intentKey);
    });
}

function acceptSuggestion(intentKey, suggestionSql) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            use_suggestion: true,
            suggestion_sql: suggestionSql,
            intent_key: intentKey
        }));
        document.getElementById("suggestion-panel")?.remove();
        setStatus("Using cached suggestion...");
    }
}

function declineSuggestion(intentKey) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            use_suggestion: false,
            intent_key: intentKey
        }));
        document.getElementById("suggestion-panel")?.remove();
        setStatus("Calling AI for fresh SQL...");
    }
}


// ── UI HELPERS ────────────────────────────────────────────────────────────────

async function resetUI() {
    document.getElementById("sql-panel").classList.add("hidden");
    document.getElementById("insight-panel").classList.add("hidden");
    document.getElementById("raw-data-panel").classList.add("hidden");
    const toggleBtn = document.getElementById("toggle-raw-data-btn");
    if (toggleBtn) toggleBtn.innerHTML = "📊 Show Raw JSON Data";
    document.getElementById("destructive-warning").classList.add("hidden");
    document.getElementById("sql-display").contentEditable = "false";
    document.getElementById("edit-indicator").style.display = "none";
    document.getElementById("cost-display").textContent = "";
    document.getElementById("insight-text").textContent = "";
    document.getElementById("grid-counter").textContent = "0 / 0";
    document.getElementById("suggestion-panel")?.remove();
    hideApprovalButtons();
    await initSourceGrid();
}

function setStatus(text) {
    document.getElementById("progress-text").textContent = text;
}

async function initSourceGrid() {
    const grid = document.getElementById("source-grid");
    grid.innerHTML = "";
    try {
        const response = await fetch(`${API_BASE}/sources`, {
            headers: { "X-API-Key": API_KEY }
        });
        const sources = await response.json();
        let validSources = 0;
        sources.forEach(source => {
            if (!source.source_name || source.source_name === "null") return;
            
            validSources++;
            const cell = document.createElement("div");
            cell.className = "source-cell pending";
            cell.id = `cell-${source.source_name}`;
            cell.setAttribute("data-name", source.source_name);
            grid.appendChild(cell);
        });
        totalSources = validSources;
    } catch (e) {
        console.error("Failed to load sources for grid:", e);
    }
}

function updateSourceCell(dbName, status) {
    const cell = document.getElementById(`cell-${dbName}`);
    if (cell) {
        cell.className = `source-cell ${status}`;
    }
}

function renderInsightReport(report, mlAnalysis) {
    document.getElementById("insight-text").textContent = report || "No report generated";

    const statsEl = document.getElementById("ml-stats");
    statsEl.innerHTML = "";
    
    if (mlAnalysis && !mlAnalysis.error) {
        const stats = [
            { label: "Sources Analyzed", value: mlAnalysis.total_sources },
            { label: "Anomalies", value: mlAnalysis.anomaly_count },
            { label: "Trend", value: mlAnalysis.trend_direction },
            { label: "Clusters", value: mlAnalysis.cluster_count },
            { label: "Mean Value", value: mlAnalysis.mean_value },
            { label: "Std Dev", value: mlAnalysis.std_value },
        ];

        stats.forEach(s => {
            const div = document.createElement("div");
            div.className = "ml-stat";
            div.innerHTML = `${s.label}: <span>${s.value}</span>`;
            statsEl.appendChild(div);
        });
    }

    document.getElementById("insight-panel").classList.remove("hidden");
}

function renderRawData() {
    const display = document.getElementById("raw-data-display");
    
    if (window.queryResults && window.queryResults.length > 0) {
        try {
            // 1. Sort by database name ascending
            window.queryResults.sort((a, b) => {
                if (!a.database || !b.database) return 0;
                return String(a.database).localeCompare(String(b.database));
            });
            
            // 2. Sort rows by primary key
            window.queryResults.forEach(dbResult => {
                if (dbResult.data && Array.isArray(dbResult.data) && dbResult.data.length > 0) {
                    const firstRow = dbResult.data[0];
                    if (typeof firstRow === 'object' && firstRow !== null) {
                        const keys = Object.keys(firstRow);
                        const pk = keys.find(k => k.toLowerCase() === 'id') || 
                                   keys.find(k => k.toLowerCase().endsWith('_id')) || 
                                   keys[0];
                        
                        if (pk) {
                            dbResult.data.sort((rowA, rowB) => {
                                const valA = rowA[pk];
                                const valB = rowB[pk];
                                
                                if (typeof valA === 'number' && typeof valB === 'number') {
                                    return valA - valB;
                                }
                                return String(valA || "").localeCompare(String(valB || ""));
                            });
                        }
                    }
                }
            });
        } catch (e) {
            console.error("Error sorting raw data:", e);
        }

        display.textContent = JSON.stringify(window.queryResults, null, 2);
    } else {
        display.textContent = "No data returned.";
    }
}

function toggleRawData() {
    const panel = document.getElementById("raw-data-panel");
    const btn = document.getElementById("toggle-raw-data-btn");
    if (panel.classList.contains("hidden")) {
        panel.classList.remove("hidden");
        btn.innerHTML = "📊 Hide Raw JSON Data";
    } else {
        panel.classList.add("hidden");
        btn.innerHTML = "📊 Show Raw JSON Data";
    }
}


// ── METRICS PANEL ─────────────────────────────────────────────────────────────

async function loadMetrics() {
    try {
        const response = await fetch(`${API_BASE}/metrics`, {
            headers: { "X-API-Key": API_KEY }
        });
        const data = await response.json();

        if (data.message || data.error) return;

        const display = document.getElementById("metrics-display");
        const metrics = [
            { label: "Total Queries",     value: data.total_queries,           cls: "" },
            { label: "Cache Hit Rate",    value: `${data.cache_hit_rate_pct}%`, cls: "green" },
            { label: "Avg Latency",       value: `${data.avg_latency_ms}ms`,   cls: "" },
            { label: "P90 Latency",       value: `${data.p90_latency_ms}ms`,   cls: "" },
            { label: "AI Spend",          value: `$${data.total_ai_spend_usd}`, cls: "yellow" },
            { label: "Cache Savings",     value: `$${data.cache_savings_usd}`, cls: "green" },
            { label: "Vault Size",        value: data.vault_size,              cls: "" },
            { label: "Error Rate",        value: `${data.error_rate_pct}%`,    cls: data.error_rate_pct > 5 ? "yellow" : "green" },
        ];

        display.innerHTML = metrics.map(m => `
            <div class="metric-card">
                <div class="metric-label">${m.label}</div>
                <div class="metric-value ${m.cls}">${m.value}</div>
            </div>
        `).join("");

    } catch (e) {
        console.error("Metrics load failed:", e);
    }
}


// ── VAULT PANEL ───────────────────────────────────────────────────────────────

async function loadVault() {
    try {
        const response = await fetch(`${API_BASE}/vault`, {
            headers: { "X-API-Key": API_KEY }
        });
        const entries = await response.json();

        const tbody = document.getElementById("vault-body");
        tbody.innerHTML = "";

        entries.forEach(entry => {
            const row = document.createElement("tr");
            const lastUsed = entry.last_used
                ? new Date(entry.last_used).toLocaleString()
                : "—";
            row.innerHTML = `
                <td title="${escapeHtml(entry.intent_key)}">${escapeHtml(entry.intent_key)}</td>
                <td title="${escapeHtml(entry.approved_sql)}">${escapeHtml(entry.approved_sql)}</td>
                <td>${entry.hit_count}</td>
                <td>${lastUsed}</td>
            `;
            tbody.appendChild(row);
        });

        if (entries.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="color:#484f58;padding:16px">No cached queries yet</td></tr>';
        }

    } catch (e) {
        console.error("Vault load failed:", e);
    }
}


// ── INITIAL LOAD ──────────────────────────────────────────────────────────────

initSourceGrid().then(() => {
    loadMetrics();
    loadVault();
});

// Refresh metrics every 30 seconds
setInterval(loadMetrics, 30000);