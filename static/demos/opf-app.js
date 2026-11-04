/* OPF PT-BR PII — frontend controller.
 *
 * The design mock shipped an offline mock (`demos/pii-data.js`) whose
 * entities were hand-annotated.  Here every entity comes from the real OPF
 * runtime: the page asks `/api/meta` for the taxonomy and the example
 * catalogue and `/api/analyze` for the spans.
 */
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const esc = (s) =>
    String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");

  let CATALOG = null;
  let EX = [];
  let ALL = [];
  let PTBR24 = [];
  let CATS = {};
  let state = {
    ex: null,
    scope: "24",
    labels: new Set(),
    view: "annotate",
    ran: false,
    result: null,
    busy: false,
  };

  /* ---------------------------------------------------------------- meta */

  async function boot() {
    try {
      const res = await fetch("/api/meta", { headers: { accept: "application/json" } });
      if (!res.ok) throw new Error("HTTP " + res.status);
      CATALOG = await res.json();
    } catch (err) {
      $("#examples").innerHTML =
        '<p class="note">Não foi possível carregar o catálogo de rótulos (' +
        esc(err.message) +
        "). A análise continua disponível, mas os chips de rótulo ficam indisponíveis.</p>";
      return;
    }
    PTBR24 = CATALOG.canonical || [];
    ALL = CATALOG.labels || [];
    CATS = {};
    (CATALOG.categories || []).forEach((c) => (CATS[c.key] = c));
    CATS.custom = CATALOG.custom_category || { key: "custom", label: "Label customizado (zero-shot)", labels: [] };
    EX = CATALOG.examples || [];
    state.labels = new Set(PTBR24);

    drawExamples();
    drawLabels();
    syncScopeButtons();
    const total = $("#labtotal");
    if (total) total.textContent = ALL.length + " rótulos";
    const scopeAll = $('#scope button[data-v="all"]');
    if (scopeAll) scopeAll.textContent = "Todos os " + ALL.length;
    const canonical = $('#scope button[data-v="24"]');
    if (canonical) canonical.textContent = PTBR24.length + " canônicos PT-BR";

    const q = new URLSearchParams(location.search);
    if (q.get("ex")) {
      loadEx(q.get("ex"));
      run();
    }
  }

  function drawExamples() {
    const groups = {};
    EX.forEach((e) => {
      (groups[e.genre] = groups[e.genre] || []).push(e);
    });
    $("#examples").innerHTML = Object.keys(groups)
      .map(
        (g) =>
          '<div class="ex-row"><span class="g">' +
          esc(g) +
          '</span><div class="chips">' +
          groups[g]
            .map(
              (e) =>
                '<button type="button" class="chip" data-ex="' +
                esc(e.id) +
                '" aria-pressed="false" title="' +
                esc(e.title) +
                '">' +
                esc(e.title) +
                "</button>"
            )
            .join("") +
          "</div></div>"
      )
      .join("");
  }

  function scopeLabels(key) {
    const labels = (CATS[key] && CATS[key].labels) || [];
    return labels.filter((l) => state.scope === "all" || PTBR24.includes(l));
  }

  function drawLabels() {
    const order = ["doc", "fin", "contact", "name", "dob", "loc", "sens", "cred", "prof"];
    $("#labels").innerHTML = order
      .map((c) => {
        const ls = scopeLabels(c);
        if (!ls.length) return "";
        return (
          '<div class="labgrp ' +
          c +
          '"><div class="gh"><span class="sq"></span>' +
          esc(CATS[c].label) +
          '<button type="button" data-all="' +
          c +
          '" aria-label="Alternar todos os rótulos de ' +
          esc(CATS[c].label) +
          '">todos / nenhum</button></div><div class="chips">' +
          ls
            .map(
              (l) =>
                '<button type="button" class="chip ' +
                (PTBR24.includes(l) ? "" : "x") +
                '" data-l="' +
                esc(l) +
                '" aria-pressed="' +
                state.labels.has(l) +
                '">' +
                esc(l.replace(/_/g, " ").replace("personal description of ", "")) +
                "</button>"
            )
            .join("") +
          "</div></div>"
        );
      })
      .join("");
    const total = state.scope === "all" ? ALL.length : PTBR24.length;
    $("#labcount").textContent = state.labels.size + " / " + total + " ativos";
  }

  function syncScopeButtons() {
    $$("#scope button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === state.scope)));
    $$("#view button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === state.view)));
  }

  /* -------------------------------------------------------------- actions */

  function loadEx(id) {
    const ex = EX.find((e) => e.id === id);
    if (!ex) return false;
    state.ex = ex;
    $("#txt").value = ex.text;
    $$("[data-ex]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.ex === id)));
    if (ex.labels && ex.labels.length) {
      state.scope = "all";
      state.labels = new Set(ex.labels);
      drawLabels();
      syncScopeButtons();
    }
    count();
    return true;
  }

  function count() {
    $("#chars").textContent = $("#txt").value.length.toLocaleString("pt-BR") + " caracteres";
  }

  function idle() {
    $("#anno").innerHTML =
      '<div class="empty"><span class="mono">Aguardando texto</span>Escolha um exemplo ou cole um documento e clique em Analisar.</div>';
    $("#summary").innerHTML = "";
    $("#summeta").textContent = "";
  }

  function toast(m) {
    const t = $("#toast");
    t.textContent = m;
    t.classList.add("show");
    clearTimeout(t._t);
    t._t = setTimeout(() => t.classList.remove("show"), 1800);
  }

  function setBusy(busy) {
    state.busy = busy;
    $("#run").disabled = busy;
    $("#run").setAttribute("aria-busy", String(busy));
    $("#anno").setAttribute("aria-busy", String(busy));
    $("#run").textContent = busy ? "Analisando…" : "Analisar texto →";
  }

  function errorBox(detail) {
    $("#anno").innerHTML =
      '<div class="empty"><span class="mono">Falha na análise</span>' +
      esc(detail || "O runtime OPF não respondeu.") +
      "</div>";
    $("#summary").innerHTML = "";
    $("#summeta").textContent = "";
  }

  async function run() {
    if (state.busy) return;
    const text = $("#txt").value;
    if (!text.trim()) {
      toast("Cole um texto primeiro");
      $("#txt").focus();
      return;
    }
    setBusy(true);
    try {
      const res = await fetch("/gradio_api/call/analyze", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({data: [text, Array.from(state.labels), $("#overlap").checked]}),
      });
      if (!res.ok) throw new Error("HTTP " + res.status);
      const {event_id} = await res.json();
      const data = await new Promise((resolve, reject) => {
        const events = new EventSource("/gradio_api/call/analyze/" + encodeURIComponent(event_id));
        events.addEventListener("complete", (event) => {
          events.close();
          try { resolve(JSON.parse(event.data)[0]); } catch (error) { reject(error); }
        });
        events.addEventListener("error", (event) => { events.close(); reject(new Error(event.data || "Falha na conexão")); });
      });
      // Python reports Unicode code-point offsets; JavaScript slices UTF-16.
      const points = Array.from(text);
      data.spans = data.spans.map(span => ({...span,
        start: points.slice(0, span.start).join("").length,
        end: points.slice(0, span.end).join("").length
      }));
      if (text !== $("#txt").value) { toast("Texto alterado; analise novamente"); return; }
      state.result = data;
      state.ran = true;
      paint(data);
      const r = document.getElementById("out").getBoundingClientRect();
      if (r.top > window.innerHeight * 0.7) window.scrollBy({ top: r.top - 80, behavior: "smooth" });
    } catch (err) {
      errorBox(String(err));
      toast("Falha de rede na análise");
    } finally {
      setBusy(false);
    }
  }

  /* ------------------------------------------------------------ rendering */

  function renderSpans(text, spans, mode) {
    let out = "";
    let cur = 0;
    spans.forEach((e) => {
      if (e.start < cur) return;
      out += esc(text.slice(cur, e.start));
      const lb = esc(e.label.replace(/_/g, " "));
      if (mode === "redact") {
        out += '<span class="ent red cat-' + esc(e.category) + '" title="' + lb + '">' + esc(e.placeholder) + "</span>";
      } else {
        out +=
          '<span class="ent cat-' +
          esc(e.category) +
          '" title="' +
          lb +
          '"><span class="t">' +
          esc(text.slice(e.start, e.end)) +
          '</span><span class="lb">' +
          lb +
          "</span></span>";
      }
      cur = e.end;
    });
    out += esc(text.slice(cur));
    return out;
  }

  function paint(data) {
    const spans = data.spans || [];
    const text = data.text || "";
    $("#anno").innerHTML = spans.length
      ? renderSpans(text, spans, state.view)
      : '<div class="empty"><span class="mono">Nenhuma entidade</span>Nada dentro dos rótulos selecionados.</div>';

    const bits = [spans.length + " entidade(s)", state.labels.size ? "filtro: " + state.labels.size + " rótulo(s)" : "sem filtro · todos os rótulos"];
    if ($("#overlap").checked) bits.push("sobreposições descartadas");
    if (data.hidden) bits.push(data.hidden + " sobreposição(ões) omitida(s) da anotação");
    if (data.elapsed_ms != null) bits.push(data.elapsed_ms + " ms");
    $("#runmeta").textContent = bits.join(" · ");
    summary(data.all_spans || spans);
  }

  function summary(spans) {
    const g = {};
    spans.forEach((e) => (g[e.category] = g[e.category] || []).push(e));
    const sens = (g.sens || []).length;
    const docs = (g.doc || []).length;
    const fin = (g.fin || []).length;
    const cred = (g.cred || []).length;
    $("#summeta").textContent = spans.length + " ocorrência(s)";
    if (!spans.length) {
      $("#summary").innerHTML = "";
      return;
    }
    let h =
      '<div class="risk"><div class="' +
      (sens ? "hot" : "") +
      '"><b>' +
      sens +
      '</b><small>Dados sensíveis · art. 5º, II</small></div><div><b>' +
      docs +
      '</b><small>Documentos oficiais</small></div><div><b>' +
      fin +
      '</b><small>Dados financeiros</small></div><div><b>' +
      cred +
      '</b><small>Credenciais / identificadores</small></div></div>';
    if (sens)
      h +=
        '<p class="note"><b>Ponto de atenção.</b> ' +
        sens +
        " ocorrência(s) de dado pessoal sensível. Tratamento condicionado às hipóteses do art. 11 da LGPD.</p>";
    if (cred) h += '<p class="note"><b>Credenciais.</b> Revisar retenção e criptografia em trânsito e em repouso.</p>';
    ["sens", "doc", "fin", "contact", "name", "dob", "loc", "cred", "prof", "custom"].forEach((c) => {
      const rows = g[c];
      if (!rows) return;
      h +=
        '<div class="cath ' +
        c +
        '"><span class="sq"></span>' +
        esc((CATS[c] && CATS[c].label) || c) +
        '<span class="n">' +
        rows.length +
        '</span></div><table class="tbl"><thead><tr><th style="width:200px">Label</th><th>Trecho</th><th style="width:150px">Placeholder</th></tr></thead><tbody>' +
        rows
          .map(
            (e) =>
              '<tr><td><span class="code">' +
              esc(e.label) +
              '</span></td><td style="font-weight:500">' +
              esc(e.text) +
              '</td><td class="num">' +
              esc(e.placeholder || "") +
              "</td></tr>"
          )
          .join("") +
        "</tbody></table>";
    });
    $("#summary").innerHTML = h;
  }

  /* --------------------------------------------------------------- wiring */

  document.addEventListener("DOMContentLoaded", () => {
    $("#labels").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      if (b.dataset.l) {
        if (state.labels.has(b.dataset.l)) state.labels.delete(b.dataset.l);
        else state.labels.add(b.dataset.l);
      }
      if (b.dataset.all) {
        const ls = scopeLabels(b.dataset.all);
        const allOn = ls.every((l) => state.labels.has(l));
        ls.forEach((l) => (allOn ? state.labels.delete(l) : state.labels.add(l)));
      }
      drawLabels();
      if (state.ran) run();
    });

    $("#scope").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      state.scope = b.dataset.v;
      state.labels = new Set(state.scope === "all" ? ALL : PTBR24);
      drawLabels();
      syncScopeButtons();
      if (state.ran) run();
    });

    $("#view").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      state.view = b.dataset.v;
      syncScopeButtons();
      if (state.ran) run();
    });

    $("#overlap").addEventListener("change", () => {
      if (state.ran) run();
    });

    $("#examples").addEventListener("click", (e) => {
      const b = e.target.closest("[data-ex]");
      if (!b) return;
      if (loadEx(b.dataset.ex)) run();
    });

    $("#txt").addEventListener("input", () => {
      count();
      if (state.ex && $("#txt").value !== state.ex.text) {
        state.ex = null;
        $$("[data-ex]").forEach((b) => b.setAttribute("aria-pressed", "false"));
      }
    });

    $("#clear").addEventListener("click", () => {
      $("#txt").value = "";
      state.ex = null;
      state.ran = false;
      state.result = null;
      $$("[data-ex]").forEach((b) => b.setAttribute("aria-pressed", "false"));
      count();
      idle();
      $("#runmeta").textContent = "Exemplos construídos manualmente · fora do conjunto de treino";
      $("#txt").focus();
    });

    $("#run").addEventListener("click", run);

    $("#copy").addEventListener("click", () => {
      if (!state.result || !(state.result.spans || []).length) {
        toast("Analise um texto primeiro");
        return;
      }
      const text = state.result.redacted_text || "";
      if (navigator.clipboard) navigator.clipboard.writeText(text);
      toast("Texto redigido copiado");
    });

    idle();
    count();
    boot();
  });
})();
