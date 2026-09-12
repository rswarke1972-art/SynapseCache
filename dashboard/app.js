/**
 * SynapseCache (KV-Optic) - Interactive Telemetry Application Logic
 */

(function() {
  // State
  let budget = 128;
  let kSink = 4;
  let wLocal = 32;
  let kAnchor = 64;

  let isStreaming = false;
  let streamInterval = null;
  let totalTokensSeen = 0;
  let totalEvictions = 0;

  // Active token structures: id -> { id, tier, mass, entropy, salience, headWeights, isNeedle }
  const activeTokens = new Map();
  const evictedTokenIds = new Set();
  const localQueue = [];
  const anchorSet = new Set();
  const sinkSet = new Set();

  let needleId = -1;

  // DOM Elements
  const btnStream = document.getElementById("btn-stream");
  const btnInjectNeedle = document.getElementById("btn-inject-needle");
  const btnReset = document.getElementById("btn-reset");
  const sliderBudget = document.getElementById("slider-budget");
  const valBudget = document.getElementById("val-budget");
  const statBudget = document.getElementById("stat-budget");
  const valTokensSeen = document.getElementById("val-tokens-seen");
  const valRetained = document.getElementById("val-retained");
  const valEvictions = document.getElementById("val-evictions");
  const tokenGrid = document.getElementById("token-grid");

  const countSink = document.getElementById("count-sink");
  const countAnchor = document.getElementById("count-anchor");
  const countLocal = document.getElementById("count-local");

  const barSink = document.getElementById("bar-sink");
  const barAnchor = document.getElementById("bar-anchor");
  const barLocal = document.getElementById("bar-local");

  const inspectTokenName = document.getElementById("inspect-token-name");
  const inspectTierBadge = document.getElementById("inspect-tier-badge");
  const inspectMass = document.getElementById("inspect-mass");
  const inspectEntropy = document.getElementById("inspect-entropy");
  const inspectSalience = document.getElementById("inspect-salience");
  const headBarsContainer = document.getElementById("head-bars");

  // Initialize 8 Head Bars in Inspector
  function initInspectorBars() {
    headBarsContainer.innerHTML = "";
    for (let h = 0; h < 8; h++) {
      const row = document.createElement("div");
      row.className = "head-bar-row";
      row.innerHTML = `
        <span class="head-bar-label">Head ${h + 1}</span>
        <div class="head-bar-outer">
          <div class="head-bar-fill" id="head-fill-${h}" style="width: 0%"></div>
        </div>
        <span class="head-bar-val" id="head-val-${h}">0.00</span>
      `;
      headBarsContainer.appendChild(row);
    }
  }

  // Update Inspector Card
  function inspectToken(token) {
    if (!token) {
      inspectTokenName.textContent = "Token #--";
      inspectTierBadge.textContent = "Inactive";
      inspectTierBadge.className = "badge badge-slate";
      inspectMass.textContent = "0.000";
      inspectEntropy.textContent = "0.000";
      inspectSalience.textContent = "0.000";
      for (let h = 0; h < 8; h++) {
        document.getElementById(`head-fill-${h}`).style.width = "0%";
        document.getElementById(`head-val-${h}`).textContent = "0.00";
      }
      return;
    }

    inspectTokenName.textContent = token.isNeedle ? `Token #${token.id} (🎯 Needle)` : `Token #${token.id}`;
    inspectMass.textContent = token.mass.toFixed(4);
    inspectEntropy.textContent = token.entropy.toFixed(4);
    inspectSalience.textContent = token.salience.toFixed(4);

    let badgeClass = "badge-slate";
    if (token.tier === "sink") badgeClass = "badge-purple";
    else if (token.tier === "anchor") badgeClass = "badge-cyan";
    else if (token.tier === "local") badgeClass = "badge-emerald";

    inspectTierBadge.className = `badge ${badgeClass}`;
    inspectTierBadge.textContent = token.tier.toUpperCase();

    // Fill head weights
    for (let h = 0; h < 8; h++) {
      const w = token.headWeights[h] || 0.0;
      const pct = Math.min(100, Math.round(w * 100));
      document.getElementById(`head-fill-${h}`).style.width = `${pct}%`;
      document.getElementById(`head-val-${h}`).textContent = w.toFixed(2);
    }
  }

  // Generate Synthetic Multi-Head Weights
  function generateHeadWeights(isAnchor = false) {
    const raw = [];
    for (let h = 0; h < 8; h++) {
      if (isAnchor) {
        // High cross-head agreement
        raw.push(0.15 + Math.random() * 0.15);
      } else {
        // Skewed / single-head attention
        raw.push(h === 0 ? 0.6 + Math.random() * 0.2 : Math.random() * 0.04);
      }
    }
    const sum = raw.reduce((a, b) => a + b, 0);
    const normalized = raw.map(v => v / (sum + 1e-9));

    // Calculate Shannon entropy: - sum p log2(p) / log2(8)
    let hEntropy = 0;
    for (let p of normalized) {
      if (p > 0) {
        hEntropy -= p * Math.log2(p + 1e-9);
      }
    }
    const normEntropy = Math.max(0, Math.min(1, hEntropy / Math.log2(8)));
    const meanMass = sum / 8;
    const compositeSalience = (0.6 * meanMass) + (0.4 * normEntropy);

    return {
      weights: normalized,
      mass: meanMass,
      entropy: normEntropy,
      salience: compositeSalience
    };
  }

  // Process 1 Token Step
  function stepToken(forceNeedle = false) {
    const tid = totalTokensSeen++;
    const isSink = tid < kSink;
    const isNeedle = forceNeedle;
    const isAnchor = isNeedle || (Math.random() < 0.08 && !isSink);

    const synth = generateHeadWeights(isAnchor);
    const tier = isSink ? "sink" : "local";

    const tokenObj = {
      id: tid,
      tier: tier,
      mass: synth.mass,
      entropy: synth.entropy,
      salience: isNeedle ? 99.0 : synth.salience,
      headWeights: synth.weights,
      isNeedle: isNeedle
    };

    activeTokens.set(tid, tokenObj);

    if (isSink) {
      sinkSet.add(tid);
    } else {
      localQueue.push(tid);
    }

    if (isNeedle) {
      needleId = tid;
      promoteAnchor(tid);
    } else if (isAnchor && !isSink) {
      promoteAnchor(tid);
    }

    // Slide local window if exceeded
    if (localQueue.length > wLocal) {
      const oldestLocal = localQueue.shift();
      if (activeTokens.has(oldestLocal) && !anchorSet.has(oldestLocal) && !sinkSet.has(oldestLocal)) {
        activeTokens.get(oldestLocal).tier = "candidate";
      }
    }

    // Invariant Enforcement: |C_t| <= B
    while (activeTokens.size > budget) {
      evictOne();
    }

    updateUI();
  }

  function promoteAnchor(tid) {
    if (!activeTokens.has(tid) || sinkSet.has(tid)) return;
    const t = activeTokens.get(tid);
    t.tier = "anchor";
    anchorSet.add(tid);

    const lIdx = localQueue.indexOf(tid);
    if (lIdx !== -1) localQueue.splice(lIdx, 1);

    // If anchors exceed capacity, demote lowest
    if (anchorSet.size > kAnchor) {
      let lowestId = null;
      let minScore = Infinity;
      for (let aid of anchorSet) {
        const score = activeTokens.get(aid).salience;
        if (score < minScore) {
          minScore = score;
          lowestId = aid;
        }
      }
      if (lowestId !== null) {
        anchorSet.delete(lowestId);
        activeTokens.get(lowestId).tier = "candidate";
      }
    }
  }

  function evictOne() {
    let targetId = null;
    let minScore = Infinity;

    // First try candidates
    for (let [tid, tok] of activeTokens.entries()) {
      if (tok.tier === "candidate" && tok.salience < minScore) {
        minScore = tok.salience;
        targetId = tid;
      }
    }

    // If no candidates, try lowest anchor (except needle)
    if (targetId === null) {
      for (let aid of anchorSet) {
        const tok = activeTokens.get(aid);
        if (!tok.isNeedle && tok.salience < minScore) {
          minScore = tok.salience;
          targetId = aid;
        }
      }
      if (targetId !== null) anchorSet.delete(targetId);
    }

    // Fallback: evict oldest local
    if (targetId === null && localQueue.length > 0) {
      targetId = localQueue.shift();
    }

    if (targetId !== null) {
      activeTokens.delete(targetId);
      evictedTokenIds.add(targetId);
      totalEvictions++;
    }
  }

  function updateUI() {
    valTokensSeen.textContent = totalTokensSeen;
    valRetained.textContent = `${activeTokens.size} / ${budget}`;
    valEvictions.textContent = totalEvictions;

    // Update Counts
    let sCount = sinkSet.size;
    let aCount = anchorSet.size;
    let lCount = 0;
    for (let tok of activeTokens.values()) {
      if (tok.tier === "local" || tok.tier === "candidate") lCount++;
    }

    countSink.textContent = sCount;
    countAnchor.textContent = aCount;
    countLocal.textContent = lCount;

    // Update Progress Bar
    const total = Math.max(1, activeTokens.size);
    barSink.style.width = `${(sCount / total) * 100}%`;
    barAnchor.style.width = `${(aCount / total) * 100}%`;
    barLocal.style.width = `${(lCount / total) * 100}%`;

    // Render Token Grid (show most recent 160 tokens)
    tokenGrid.innerHTML = "";
    const startId = Math.max(0, totalTokensSeen - 160);

    for (let id = startId; id < totalTokensSeen; id++) {
      const pill = document.createElement("div");
      pill.className = "token-pill";
      pill.textContent = id;

      if (activeTokens.has(id)) {
        const tok = activeTokens.get(id);
        if (tok.isNeedle) {
          pill.classList.add("pill-needle");
          pill.textContent = "🎯";
        } else if (tok.tier === "sink") {
          pill.classList.add("pill-sink");
        } else if (tok.tier === "anchor") {
          pill.classList.add("pill-anchor");
        } else {
          pill.classList.add("pill-local");
        }
        pill.addEventListener("mouseenter", () => inspectToken(tok));
      } else {
        pill.classList.add("pill-evicted");
        pill.addEventListener("mouseenter", () => inspectToken({
          id: id,
          tier: "evicted",
          mass: 0.0,
          entropy: 0.0,
          salience: 0.0,
          headWeights: [0, 0, 0, 0, 0, 0, 0, 0],
          isNeedle: false
        }));
      }

      tokenGrid.appendChild(pill);
    }
  }

  // Event Listeners
  btnStream.addEventListener("click", () => {
    isStreaming = !isStreaming;
    if (isStreaming) {
      btnStream.textContent = "⏸ Pause Stream";
      btnStream.classList.replace("btn-primary", "btn-secondary");
      streamInterval = setInterval(() => stepToken(), 100);
    } else {
      btnStream.textContent = "▶ Run Stream";
      btnStream.classList.replace("btn-secondary", "btn-primary");
      clearInterval(streamInterval);
    }
  });

  btnInjectNeedle.addEventListener("click", () => {
    stepToken(true);
  });

  btnReset.addEventListener("click", () => {
    if (streamInterval) clearInterval(streamInterval);
    isStreaming = false;
    btnStream.textContent = "▶ Run Stream";
    btnStream.classList.replace("btn-secondary", "btn-primary");

    totalTokensSeen = 0;
    totalEvictions = 0;
    activeTokens.clear();
    evictedTokenIds.clear();
    localQueue.length = 0;
    anchorSet.clear();
    sinkSet.clear();
    needleId = -1;

    updateUI();
    inspectToken(null);
  });

  sliderBudget.addEventListener("input", (e) => {
    budget = parseInt(e.target.value);
    valBudget.textContent = budget;
    statBudget.innerHTML = `${budget} <span class="stat-unit">tokens</span>`;
    wLocal = Math.max(8, Math.floor(budget / 4));
    kAnchor = Math.max(8, Math.floor(budget / 2));
    while (activeTokens.size > budget) {
      evictOne();
    }
    updateUI();
  });

  // Init
  initInspectorBars();
  updateUI();
})();
