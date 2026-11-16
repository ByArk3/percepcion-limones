// Página web del servicio. Habla con el backend por /api/... (nginx lo redirige al contenedor backend).
const API = "/api";
const CLASES = ["verde", "pinton", "maduro"];
const NOMBRE = { verde: "Verde", pinton: "Pintón", maduro: "Maduro", error: "Error" };
let ultimoZip = null;

const $ = (id) => document.getElementById(id);
const pct = (p) => (p == null ? "-" : (100 * p).toFixed(1) + " %");
const prob4 = (p) => (p == null ? "" : p.toFixed(4));
const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------------------------------------------------------------- pestañas
document.querySelectorAll(".pestana").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".pestana").forEach((x) => x.classList.toggle("activa", x === b));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("activo", p.id === b.dataset.panel));
    if (b.dataset.panel === "historial") cargarHistorial();
    if (b.dataset.panel === "lotes") cargarLotes();
  })
);

// ---------------------------------------------------------------- estado y datos del modelo
async function cargarEstado() {
  try {
    const r = await fetch(`${API}/health`);
    if (!r.ok) throw new Error(r.status);
    const s = await r.json();
    $("estado").innerHTML =
      `<span class="ok">Servicio activo</span> · modelo ${esc(s.version_modelo)} · SHA-256 ${esc(s.sha256.slice(0, 16))}... ` +
      (s.sha256_coincide_con_info ? "(verificado)" : '<span class="mal">(no coincide con modelo_info.json)</span>');
  } catch (e) {
    $("estado").innerHTML = '<span class="mal">El servicio no responde.</span> Revisa docker compose.';
  }
}

// Las cifras salen de modelo/modelo_info.json (a través del backend), no están escritas en la página.
async function cargarModelo() {
  try {
    const r = await fetch(`${API}/modelo`);
    const m = await r.json();
    const mp = m.metricas_prueba || {};
    $("m-exactitud").textContent = mp.exactitud == null ? "-" : (100 * mp.exactitud).toFixed(1) + " %";
    $("m-f1").textContent = mp.f1_macro == null ? "-" : mp.f1_macro.toFixed(3);
    $("m-recall").textContent = mp.recall_maduro == null ? "-" : mp.recall_maduro.toFixed(3);
    $("m-fotos").textContent = m.n_train ?? "-";
    $("nota-metricas").textContent =
      `Métricas del modelo desplegado (semilla ${m.semilla}) sobre ${m.n_test} fotos de prueba. ` +
      `Fuente: modelo_info.json (${m.fecha}).`;
    $("sobre-pre").textContent = `${m.modelo}. Entrada: ${m.entrada}. Preprocesamiento: ${m.preprocesamiento}.`;
    const filas = [
      ["Versión", m.version_modelo], ["SHA-256", m.sha256], ["Formato", m.formato],
      ["Clases", (m.nombres || m.clases || []).join(", ")], ["Épocas", m.epocas], ["Semilla", m.semilla],
      ["Selección", m.seleccion], ["Fotos (entrenamiento / validación / prueba)", `${m.n_train} / ${m.n_val} / ${m.n_test}`],
      ["Umbral de tono amarillo", `H < ${m.umbral_h} (OpenCV 0-179)`], ["Fecha de exportación", m.fecha],
      ...Object.entries(mp).map(([k, v]) => [`Prueba: ${k}`, typeof v === "number" ? v.toFixed(4) : v]),
    ];
    $("tabla-modelo").innerHTML = filas.map(([k, v]) => `<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`).join("");
  } catch (e) {
    $("nota-metricas").textContent = "No se pudieron leer los datos del modelo.";
  }
}

// ---------------------------------------------------------------- analizar
let elegido = null;
const zona = $("zona");
["dragenter", "dragover"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.add("encima"); }));
["dragleave", "drop"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.remove("encima"); }));
zona.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) elegir(e.dataTransfer.files[0]); });
$("entrada").addEventListener("change", (e) => { if (e.target.files[0]) elegir(e.target.files[0]); e.target.value = ""; });
$("analizar-btn").addEventListener("click", () => elegido && enviar(elegido));
$("limpiar-btn").addEventListener("click", limpiar);

function elegir(archivo) {
  elegido = archivo;
  const esZip = archivo.name.toLowerCase().endsWith(".zip");
  $("previa").classList.toggle("oculto", esZip);
  if (!esZip) $("previa").src = URL.createObjectURL(archivo);
  $("zona-texto").innerHTML = esZip ? `<strong>${esc(archivo.name)}</strong><br>archivo .zip con fotos` : "";
  $("contador").textContent = `${esc(archivo.name)} · ${(archivo.size / 1024 / 1024).toFixed(2)} MB`;
  $("analizar-btn").disabled = false;
  $("resultado-foto").classList.add("oculto");
  $("resultado-zip").classList.add("oculto");
  mensaje("");
}

function limpiar() {
  elegido = null;
  $("previa").classList.add("oculto");
  $("zona-texto").innerHTML = "<strong>Arrastra aquí una foto o un .zip</strong><br>o haz clic para elegir (.jpg, .jpeg, .png o .zip)";
  $("contador").textContent = "Ningún archivo elegido";
  $("analizar-btn").disabled = true;
  $("resultado-foto").classList.add("oculto");
  $("resultado-zip").classList.add("oculto");
  mensaje("");
}

function mensaje(texto, mal = false, id = "mensaje") {
  $(id).textContent = texto;
  $(id).classList.toggle("mal", mal);
}

async function enviar(archivo) {
  const esZip = archivo.name.toLowerCase().endsWith(".zip");
  $("resultado-foto").classList.add("oculto");
  $("resultado-zip").classList.add("oculto");
  $("analizar-btn").disabled = true;
  mensaje(esZip ? `Clasificando las fotos de ${archivo.name}...` : "Clasificando...");
  const datos = new FormData();
  datos.append("archivo", archivo);
  const t0 = performance.now();
  try {
    const r = await fetch(`${API}/${esZip ? "predict_lote" : "predict"}`, { method: "POST", body: datos });
    const cuerpo = await r.json();
    if (!r.ok) throw new Error(cuerpo.detail || `Error ${r.status}`);
    const ms = performance.now() - t0;
    mensaje("");
    esZip ? mostrarZip(cuerpo) : mostrarFoto(cuerpo, ms);
  } catch (e) {
    mensaje(`No se pudo clasificar: ${e.message}`, true);
  } finally {
    $("analizar-btn").disabled = !elegido;
  }
}

function barras(prob) {
  return CLASES.map((c) => `
    <div class="barra">
      <span>${NOMBRE[c]}</span>
      <div class="riel"><div class="relleno f-${c}" style="width:${(100 * (prob?.[c] ?? 0)).toFixed(1)}%"></div></div>
      <span class="valor">${pct(prob?.[c])}</span>
    </div>`).join("");
}

function mostrarFoto(r, msTotal) {
  $("clase").textContent = NOMBRE[r.clase];
  $("clase").className = `clase c-${r.clase}`;
  $("barras").innerHTML = barras(r.probabilidades);
  $("detalle").textContent =
    `Modelo ${r.version_modelo} · inferencia ${r.latencia_ms.toFixed(1)} ms en el servidor · ` +
    `${msTotal.toFixed(0)} ms ida y vuelta desde el navegador`;
  $("resultado-foto").classList.remove("oculto");
}

function cifras(conteo, total, etiqueta = "fotos") {
  const partes = [`<div class="cifra"><b>${total}</b><span>${etiqueta}</span></div>`];
  for (const c of [...CLASES, "error"]) {
    if (c === "error" && !conteo[c]) continue;
    partes.push(`<div class="cifra"><b class="c-${c}">${conteo[c] || 0}</b><span>${NOMBRE[c]}</span></div>`);
  }
  return partes.join("");
}

function mostrarZip(r) {
  ultimoZip = r;
  $("resumen-zip").innerHTML = cifras(r.conteo, r.fotos);
  $("detalle-zip").textContent = `${r.fotos} fotos en ${r.tiempo_total_s} s · modelo ${r.version_modelo}`;
  $("tabla-zip").innerHTML =
    `<tr><th>Archivo</th><th>Clase</th>${CLASES.map((c) => `<th class="num">${NOMBRE[c]}</th>`).join("")}</tr>` +
    r.resultados.map((x) => `<tr><td>${esc(x.archivo)}</td><td class="c-${x.clase}"><b>${NOMBRE[x.clase]}</b></td>` +
      CLASES.map((c) => `<td class="num">${pct(x.probabilidades?.[c])}</td>`).join("") + "</tr>").join("");
  $("resultado-zip").classList.remove("oculto");
}

$("descargar").addEventListener("click", () => {
  if (!ultimoZip) return;
  const filas = [["archivo", "clase", ...CLASES.map((c) => `prob_${c}`)]];
  ultimoZip.resultados.forEach((x) => filas.push([x.archivo, x.clase, ...CLASES.map((c) => prob4(x.probabilidades?.[c]))]));
  const csv = filas.map((f) => f.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob(["﻿" + csv], { type: "text/csv" }));
  a.download = `clasificacion_${ultimoZip.id_envio}.csv`;
  a.click();
});

// ---------------------------------------------------------------- historial
async function cargarHistorial() {
  try {
    const r = await fetch(`${API}/historial?limite=100`);
    const h = await r.json();
    $("resumen-historial").innerHTML = cifras(h.por_clase, h.total, "predicciones guardadas");
    $("tabla-historial").innerHTML =
      "<tr><th>Fecha</th><th>Origen</th><th>Archivo</th><th>Clase</th>" +
      CLASES.map((c) => `<th class="num">${NOMBRE[c]}</th>`).join("") +
      '<th class="num">ms</th><th>Modelo (SHA-256)</th></tr>' +
      h.ultimas.map((x) => `<tr><td>${esc(x.fecha.replace("T", " ").slice(0, 19))}</td><td>${esc(x.origen)}</td>` +
        `<td>${esc(x.archivo)}</td><td class="c-${x.clase}"><b>${NOMBRE[x.clase] || esc(x.clase)}</b></td>` +
        `<td class="num">${pct(x.prob_verde)}</td><td class="num">${pct(x.prob_pinton)}</td><td class="num">${pct(x.prob_maduro)}</td>` +
        `<td class="num">${x.latencia_ms == null ? "-" : x.latencia_ms.toFixed(1)}</td>` +
        `<td>${esc(x.version_modelo)} · ${esc(x.sha256_modelo.slice(0, 12))}</td></tr>`).join("");
  } catch (e) {
    $("resumen-historial").textContent = "No se pudo leer el historial.";
  }
}
$("recargar-historial").addEventListener("click", cargarHistorial);

// ---------------------------------------------------------------- lotes de Spark
async function cargarLotes() {
  try {
    const r = await fetch(`${API}/lotes`);
    const d = await r.json();
    if (!d.lotes.length) {
      mensaje("Todavía no hay lotes cargados. Corre el job de Spark y luego pulsa el botón.", false, "mensaje-lotes");
      $("tabla-lotes").innerHTML = ""; $("tabla-lotes-detalle").innerHTML = "";
      return;
    }
    mensaje("", false, "mensaje-lotes");
    $("tabla-lotes").innerHTML = '<tr><th>Lote</th><th class="num">Fotos</th><th>Cargado</th></tr>' +
      d.lotes.map((l) => `<tr><td>${esc(l.lote)}</td><td class="num">${l.fotos}</td><td>${esc(l.cargado.replace("T", " ").slice(0, 19))}</td></tr>`).join("");
    // tabla dinámica: lote y celular en filas, predicción en columnas
    const claves = {};
    d.detalle.forEach((x) => {
      const k = `${x.lote}|${x.celular}`;
      claves[k] = claves[k] || { lote: x.lote, celular: x.celular, verde: 0, pinton: 0, maduro: 0, error: 0 };
      claves[k][x.prediccion] = x.fotos;
    });
    $("tabla-lotes-detalle").innerHTML =
      "<tr><th>Lote</th><th>Celular</th>" + CLASES.map((c) => `<th class="num">${NOMBRE[c]}</th>`).join("") + '<th class="num">Total</th></tr>' +
      Object.values(claves).map((f) => `<tr><td>${esc(f.lote)}</td><td>${esc(f.celular)}</td>` +
        CLASES.map((c) => `<td class="num c-${c}">${f[c]}</td>`).join("") +
        `<td class="num">${CLASES.reduce((s, c) => s + f[c], 0) + f.error}</td></tr>`).join("");
  } catch (e) {
    mensaje("No se pudieron leer los lotes.", true, "mensaje-lotes");
  }
}

$("sincronizar").addEventListener("click", async () => {
  const b = $("sincronizar");
  b.disabled = true;
  mensaje("Cargando el Parquet de Spark en PostgreSQL...", false, "mensaje-lotes");
  try {
    const r = await fetch(`${API}/lotes/sincronizar`, { method: "POST" });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.status);
    await cargarLotes();
    mensaje("Cargado: " + d.cargados.map((c) => `lote ${c.lote} (${c.fotos} fotos)`).join(", "), false, "mensaje-lotes");
  } catch (e) {
    mensaje(`No se pudo cargar: ${e.message}`, true, "mensaje-lotes");
  } finally {
    b.disabled = false;
  }
});

cargarEstado();
cargarModelo();
