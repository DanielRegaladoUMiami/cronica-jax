/* CFB Picks: sitio estático que lee data/picks.json y data/backtest.json.
   Sin dependencias. Todo el texto que viene de los datos entra con textContent. */
(function () {
  'use strict';

  // ------------------------------------------------------------ utilidades
  const SVGNS = 'http://www.w3.org/2000/svg';
  const MINUS = '−';

  function put(el, attrs, kids) {
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null || v === false) continue;
        if (k === 'class') el.setAttribute('class', v);
        else if (k.slice(0, 2) === 'on') el.addEventListener(k.slice(2), v);
        else el.setAttribute(k, v === true ? '' : String(v));
      }
    }
    for (const kid of kids.flat(Infinity)) {
      if (kid == null || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }
  const h = (tag, attrs, ...kids) => put(document.createElement(tag), attrs, kids);
  const s = (tag, attrs, ...kids) => put(document.createElementNS(SVGNS, tag), attrs, kids);

  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const trim = (x, d = 1) => Number(x).toFixed(d).replace(/\.0+$/, '');
  // Nunca se muestra 100% ni 0%: el tope visible es 99%.
  const pct = (p, d = 0) => (p == null ? '–' : clamp(p * 100, 1, 99).toFixed(d) + '%');
  const rate = (p, d = 1) => (p == null ? '–' : (p * 100).toFixed(d) + '%');
  function signed(x, d = 1, unit = '') {
    if (x == null) return '–';
    const txt = Math.abs(x).toFixed(d);
    if (Number(txt) === 0) return txt + unit;
    return (x > 0 ? '+' : MINUS) + txt + unit;
  }
  const edgeFmt = (e) => (e == null ? '–' : signed(e * 100, 1, '%'));
  const oddsFmt = (o) => (o == null ? 'sin momio' : (o > 0 ? '+' : MINUS) + Math.abs(o));
  const lineFmt = (x) => (x === 0 ? 'PK' : (x > 0 ? '+' : MINUS) + trim(Math.abs(x)));
  const toDecimal = (o) => (o > 0 ? 1 + o / 100 : 1 + 100 / Math.abs(o));
  function decimalToAmerican(dec) {
    if (!(dec > 1)) return null;
    return dec >= 2 ? Math.round((dec - 1) * 100) : -Math.round(100 / (dec - 1));
  }
  const money = (x) => '$' + (Math.abs(x) >= 100 ? Math.round(x).toLocaleString('en-US') : x.toFixed(2));

  let fmtDay, fmtHour, fmtStamp;
  try {
    fmtDay = new Intl.DateTimeFormat('es-US', { timeZone: 'America/New_York', weekday: 'short', day: 'numeric', month: 'short' });
    fmtHour = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit' });
    fmtStamp = new Intl.DateTimeFormat('es-US', { timeZone: 'America/New_York', weekday: 'short', day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit' });
  } catch (err) { /* sin Intl: se muestra la fecha ISO */ }
  const when = (iso) => (fmtDay ? fmtDay.format(new Date(iso)) + ' · ' + fmtHour.format(new Date(iso)) + ' ET' : iso);
  const stamp = (iso) => (fmtStamp ? fmtStamp.format(new Date(iso)) + ' ET' : iso);

  const store = {
    get(key, fallback) {
      try {
        const raw = localStorage.getItem('cfb:' + key);
        return raw == null ? fallback : JSON.parse(raw);
      } catch (err) { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem('cfb:' + key, JSON.stringify(value)); } catch (err) { /* sin almacenamiento */ }
    },
  };

  // ----------------------------------------------------------------- estado
  const state = {
    picks: null,
    backtest: null,
    error: null,
    scope: store.get('scope', 'mine'),
    tipo: store.get('tipo', 'todos'),
    solo: store.get('solo', 'todas'),
    orden: store.get('orden', 'hora'),
    modo: store.get('modo', 'tarjetas'),
    bankroll: store.get('bankroll', 1000),
    stake: store.get('stake', 10),
    legs: store.get('legs', {}),
    parlayTipo: 'ml',
    slipOpen: false,
  };
  const view = document.getElementById('view');
  const tip = document.getElementById('tooltip');
  let refocus = null;

  function setState(key, value, focusId) {
    state[key] = value;
    store.set(key, value);
    refocus = focusId || null;
    render();
  }

  // ------------------------------------------------------- acceso a los datos
  const TYPE_LABEL = { ml: 'ML', spread: 'Spread', total: 'Total' };
  const allGames = () => state.picks.juegos.filter((g) => !g.sin_datos);
  const mineCount = () => allGames().filter((g) => g.en_mi_lista).length;
  function scopedGames() {
    const games = allGames();
    return state.scope === 'mine' && mineCount() ? games.filter((g) => g.en_mi_lista) : games;
  }
  const visibleTypes = () => (state.tipo === 'todos' ? ['ml', 'spread', 'total'] : [state.tipo]);
  const sideKeys = (tipo) => (tipo === 'total' ? ['over', 'under'] : ['visita', 'local']);

  function sideLabel(g, tipo, lado) {
    if (tipo === 'total') return (lado === 'over' ? 'Over ' : 'Under ') + trim(g.total.linea);
    const team = g[lado].nombre;
    if (tipo === 'ml') return team + ' ML';
    const line = lado === 'local' ? g.spread.linea_local : -g.spread.linea_local;
    return team + ' ' + lineFmt(line);
  }
  function pickOf(g, tipo) {
    const market = g[tipo];
    const lado = market && market.pick;
    const side = lado ? market[lado] : null;
    return side ? { g, tipo, lado, side } : null;
  }
  function sourceLabel(g, fuente) {
    if (fuente === 'espn') return (g.mercado.espn && g.mercado.espn.proveedor) || 'ESPN';
    if (fuente === 'asumido') return 'precio asumido';
    return fuente || '';
  }
  function pill(label) {
    const cls = { JUGADA: 'jugada', EVITAR: 'evitar', PASAR: 'pasar' }[label] || 'sin';
    return h('span', { class: 'pill ' + cls }, label);
  }
  function stakeText(side) {
    if (!side.stake) return null;
    const dollars = state.bankroll > 0 ? ' (' + money(side.stake * state.bankroll) + ')' : '';
    return (side.stake * 100).toFixed(1) + '% del bankroll' + dollars;
  }
  const matchup = (g) => g.visita.abbr + ' @ ' + g.local.abbr;

  // ------------------------------------------------------------- controles
  function seg(name, options, value, onPick) {
    const box = h('div', { class: 'seg', role: 'group', 'aria-label': name });
    for (const [val, text] of options) {
      const id = 'seg-' + name.replace(/\W+/g, '') + '-' + val;
      box.append(h('button', { type: 'button', id, 'aria-pressed': String(val === value), onclick: () => onPick(val, id) }, text));
    }
    return box;
  }

  function showTip(target, nodes, point) {
    tip.replaceChildren(...nodes);
    tip.hidden = false;
    const box = target.getBoundingClientRect();
    const x = point ? point.x : box.left + box.width / 2;
    const y = point ? point.y : box.top;
    const w = tip.offsetWidth, hgt = tip.offsetHeight;
    tip.style.left = clamp(x - w / 2, 8, window.innerWidth - w - 8) + 'px';
    tip.style.top = (y - hgt - 12 < 8 ? y + 16 : y - hgt - 12) + 'px';
  }
  const hideTip = () => { tip.hidden = true; };

  // ================================================================== PICKS
  function renderPicks() {
    const P = state.picks;
    const out = [evidencePanel()];
    if (P.avisos && P.avisos.length) {
      out.push(h('section', { class: 'panel notice' }, h('h2', { class: 'h3' }, 'Avisos de los datos'),
        h('ul', { class: 'alerts' }, P.avisos.map((a) => h('li', null, a)))));
    }
    out.push(filtersRow());

    const games = scopedGames();
    if (!games.length) {
      out.push(h('p', { class: 'empty' }, 'No hay juegos pendientes con proyección en esta semana.'));
    } else if (state.modo === 'lista') {
      out.push(playsList(games));
    } else {
      out.push(cardsGrid(games));
    }
    const missing = P.juegos.filter((g) => g.sin_datos && (state.scope !== 'mine' || g.en_mi_lista || !mineCount()));
    if (missing.length) {
      out.push(h('section', { class: 'panel flat' }, h('h2', { class: 'h3' }, 'Sin datos'),
        h('ul', { class: 'alerts' }, missing.map((g) => h('li', { class: 'cmp' }, g.visita.nombre + ' @ ' + g.local.nombre + ': ' + (g.motivo_sin_datos || 'sin datos'))))));
    }
    return out;
  }

  function evidencePanel() {
    const P = state.picks, m = P.modelo, ev = P.evidencia || {};
    const all = ev.todas || {};
    const mkt = state.backtest && state.backtest.mercado;
    const box = h('section', { class: 'panel notice' }, h('h2', { class: 'h3' }, 'Lo que dice el backtest antes de apostar'));
    if (!m.n_backtest) {
      box.append(h('p', null, 'Todavía no hay juegos de ' + P.temporada + ' para medir al modelo: las proyecciones salen del rating del año anterior y son menos fiables.'));
    } else box.append(h('p', null,
      'El modelo acierta ', h('b', null, rate(m.acierto, 0) + ' de los ganadores'), ' (' + m.n_backtest + ' juegos de ' + P.temporada +
      ' que no había visto) con error típico de ±' + trim(m.mae_margen) + ' pts' + (mkt ? '; el mercado, ±' + trim(mkt.mae_margen) + '.' : '.')));
    if (all.JUGADA && all.JUGADA.n) {
      const j = all.JUGADA, e = all.EVITAR || {};
      box.append(h('p', null,
        'La regla JUGADA contra la línea de cierre: ',
        h('b', null, j.ganadas + '–' + j.perdidas + ' (' + rate(j.pct, 0) + '), ROI ' + signed(j.roi * 100, 1, '%')),
        ' ± ' + trim(j.roi_error * 100) + '%' + (e.n ? '. EVITAR: ROI ' + signed(e.roi * 100, 1, '%') : '') +
        '. No ha sido rentable: léela como “aquí mi número difiere del mercado”.'));
      const det = h('details', null, h('summary', { class: 'small' }, 'Ver por tipo de apuesta (temporadas ' + (ev.temporadas || []).join(', ') + ')'));
      det.append(h('div', { class: 'scroll' }, recordTable(ev)));
      box.append(det);
    }
    return box;
  }

  function recordTable(block) {
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['Apuesta', 'G–P', '% gan.', 'ROI', 'Unid.'].map((t) => h('th', { class: t === '% gan.' ? 'opt' : null }, t)))));
    const body = h('tbody');
    for (const tipo of ['ml', 'spread', 'total', 'todas']) {
      for (const label of ['JUGADA', 'EVITAR']) {
        const r = block[tipo] && block[tipo][label];
        if (!r || !r.n) continue;
        body.append(h('tr', { class: tipo === 'todas' ? 'hl' : null },
          h('td', null, (tipo === 'todas' ? 'Todas' : TYPE_LABEL[tipo]) + ' · ' + (label === 'JUGADA' ? 'jugada' : 'evitar')),
          h('td', null, r.ganadas + '–' + r.perdidas), h('td', { class: 'opt' }, rate(r.pct, 0)),
          h('td', { class: r.roi >= 0 ? 'pos' : 'neg' }, signed(r.roi * 100, 1, '%'), h('span', { class: 'pm' }, ' ±' + trim(r.roi_error * 100, 0))),
          h('td', { class: r.unidades >= 0 ? 'pos' : 'neg' }, signed(r.unidades, 0, 'u'))));
      }
    }
    table.append(body);
    return table;
  }

  function filtersRow() {
    const row = h('div', { class: 'filters' });
    const mine = mineCount(), total = allGames().length;
    if (mine) {
      row.append(seg('Juegos', [['mine', 'Mis juegos (' + mine + ')'], ['all', 'Todos (' + total + ')']],
        state.scope === 'mine' ? 'mine' : 'all', (v, id) => setState('scope', v, id)));
    }
    row.append(seg('Tipo de apuesta', [['todos', 'Todo'], ['ml', 'ML'], ['spread', 'Spread'], ['total', 'Total']], state.tipo, (v, id) => setState('tipo', v, id)));
    row.append(seg('Etiqueta', [['todas', 'Todas'], ['jugada', 'Solo jugadas']], state.solo, (v, id) => setState('solo', v, id)));
    row.append(seg('Orden', [['hora', 'Por hora'], ['edge', 'Por edge']], state.orden, (v, id) => setState('orden', v, id)));
    row.append(seg('Vista', [['tarjetas', 'Tarjetas'], ['lista', 'Lista']], state.modo, (v, id) => setState('modo', v, id)));
    row.append(h('label', { class: 'field-input', for: 'bankroll' }, 'Bankroll $',
      h('input', { id: 'bankroll', type: 'number', min: '0', step: '50', inputmode: 'decimal', value: String(state.bankroll),
        onchange: (e) => setState('bankroll', Math.max(0, Number(e.target.value) || 0), 'bankroll') })));
    return row;
  }

  function bestEdge(g, label) {
    let best = -Infinity;
    for (const tipo of visibleTypes()) {
      const p = pickOf(g, tipo);
      if (p && p.side.edge != null && (!label || p.side.etiqueta === label)) best = Math.max(best, p.side.edge);
    }
    return best;
  }

  function cardsGrid(games) {
    let list = games.slice();
    if (state.solo === 'jugada') list = list.filter((g) => bestEdge(g, 'JUGADA') > -Infinity);
    if (state.orden === 'edge') {
      list.sort((a, b) => (bestEdge(b, 'JUGADA') - bestEdge(a, 'JUGADA')) || (bestEdge(b) - bestEdge(a)));
    } else {
      list.sort((a, b) => a.fecha.localeCompare(b.fecha));
    }
    if (!list.length) return h('p', { class: 'empty' }, 'Ningún juego tiene una JUGADA con estos filtros.');
    return h('div', { class: 'cards' }, list.map(gameCard));
  }

  function gameCard(g) {
    const pr = g.proyeccion, model = state.picks.modelo, espn = g.mercado.espn;
    const favLocal = pr.margen >= 0;
    const card = h('article', { class: 'panel game' });
    const tags = h('div', { class: 'tags' });
    if (g.en_mi_lista) tags.append(h('span', { class: 'tag' }, 'Tus líneas'));
    if (g.neutral) tags.append(h('span', { class: 'tag' }, 'Neutral'));
    if (g.fcs) tags.append(h('span', { class: 'tag' }, 'vs FCS'));
    card.append(h('div', { class: 'game-top' }, h('span', { class: 'kick' }, when(g.fecha)), tags));

    const teams = h('div', { class: 'teams' });
    for (const key of ['visita', 'local']) {
      const t = g[key];
      const isFav = (key === 'local') === favLocal;
      const meta = [t.record, t.rank ? 'AP #' + t.rank : null, t.rating_rank ? 'rating #' + t.rating_rank : null,
        key === 'local' && !g.neutral ? 'local' : null].filter(Boolean).join(' · ');
      teams.append(
        h('div', { class: 'team' + (isFav ? ' fav' : '') },
          h('span', { class: 'chip', style: 'background:#' + (/^[0-9a-f]{6}$/i.test(t.color || '') ? t.color : '888888') }),
          h('div', { class: 'team-name' }, h('b', null, t.nombre), h('span', null, meta))),
        h('div', { class: 'proj' }, pr[key] == null ? '–' : String(Math.round(pr[key]))),
        h('div', { class: 'wp' }, pct(key === 'local' ? pr.p_local : pr.p_visita)));
    }
    card.append(teams);
    card.append(h('div', { class: 'wpbar', role: 'img', 'aria-label': 'Probabilidad de ganar: ' + g.visita.nombre + ' ' + pct(pr.p_visita) + ', ' + g.local.nombre + ' ' + pct(pr.p_local) },
      h('i', { class: favLocal ? null : 'fav', style: 'flex:' + Math.max(pr.p_visita, 0.02) }),
      h('i', { class: favLocal ? 'fav' : null, style: 'flex:' + Math.max(pr.p_local, 0.02) })));

    const fav = favLocal ? g.local : g.visita;
    const lines = h('dl', { class: 'lines' });
    lines.append(h('dt', null, 'Modelo'), h('dd', null,
      fav.nombre + ' por ' + trim(Math.abs(pr.margen)) + (pr.total != null ? ' · total ' + trim(pr.total) : '') +
      (model.mae_margen != null ? ' · error típico ±' + trim(model.mae_margen) + ' pts' : '')));
    if (g.mercado.margen != null) {
      const mFav = g.mercado.margen >= 0 ? g.local : g.visita;
      let text = Math.abs(g.mercado.margen) < 0.01 ? 'Parejo' : mFav.nombre + ' por ' + trim(Math.abs(g.mercado.margen));
      if (espn && espn.total != null) text += ' · total ' + trim(espn.total);
      if (espn && espn.spread_apertura != null && (Math.abs(g.mercado.mov_spread || 0) >= 1 || Math.abs(g.mercado.mov_total || 0) >= 1)) {
        text += ' · abrió ' + (espn.spread_apertura === 0 ? 'PK' : g.local.abbr + ' ' + lineFmt(espn.spread_apertura)) +
          (espn.total_apertura != null ? ' / ' + trim(espn.total_apertura) : '');
      }
      lines.append(h('dt', null, 'Mercado'), h('dd', null, text));
    }
    if (pr.ajuste_margen_local || pr.ajuste_total) {
      lines.append(h('dt', null, 'Ajuste'), h('dd', null, 'manual: ' + signed(pr.ajuste_margen_local, 1) + ' pts al local, ' + signed(pr.ajuste_total, 1) + ' al total'));
    }
    card.append(lines);

    const markets = h('div', { class: 'markets' });
    for (const tipo of visibleTypes()) {
      const row = marketRow(g, tipo);
      if (row) markets.append(row);
    }
    if (markets.children.length) card.append(markets);

    const notes = [...(g.alertas || []).map((a) => h('li', null, a)), ...(g.comparacion || []).map((c) => h('li', { class: 'cmp' }, c))];
    if (notes.length) card.append(h('ul', { class: 'alerts' }, notes));
    if (g.notas) card.append(h('p', { class: 'note' }, h('b', null, 'Tus notas: '), g.notas));
    return card;
  }

  function marketRow(g, tipo) {
    const p = pickOf(g, tipo);
    if (!p) return null;
    const side = p.side, market = g[tipo];
    const summary = h('summary', null,
      h('span', { class: 'mkt-type' }, TYPE_LABEL[tipo]),
      h('span', { class: 'mkt-pick' }, sideLabel(g, tipo, p.lado), ' ', h('span', { class: 'price' }, '(' + oddsFmt(side.momio) + ')')),
      pill(side.etiqueta));
    const sub = h('span', { class: 'mkt-sub' }, h('span', null, 'modelo ', h('b', null, pct(side.p_modelo))));
    if (side.p_mercado != null) {
      sub.append(h('span', null, 'mercado ', h('b', null, pct(side.p_mercado))), h('span', null, 'edge ', h('b', null, edgeFmt(side.edge))));
    }
    const stake = stakeText(side);
    if (stake) sub.append(h('span', null, 'sugerido ', h('b', null, stake)));
    if (side.etiqueta === 'EVITAR' && side.motivos.length) sub.append(h('span', null, side.motivos[0]));
    summary.append(sub);

    const body = h('div', { class: 'mkt-body' });
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null, ['Lado', 'Momio', 'Modelo', 'Mercado', 'Edge'].map((t) => h('th', null, t)))));
    const tbody = h('tbody');
    for (const key of sideKeys(tipo)) {
      const sd = market[key];
      if (!sd) continue;
      tbody.append(h('tr', { class: key === p.lado ? 'hl' : null },
        h('td', { class: 'wrapcell' }, sideLabel(g, tipo, key)), h('td', null, oddsFmt(sd.momio)), h('td', null, pct(sd.p_modelo, 1)),
        h('td', null, sd.p_mercado == null ? '–' : pct(sd.p_mercado, 1)), h('td', null, edgeFmt(sd.edge))));
    }
    table.append(tbody);
    body.append(h('div', { class: 'scroll' }, table));
    const facts = [];
    if (side.fuente === 'asumido') facts.push('Precio asumido (' + oddsFmt(side.momio) + '): confirma el momio real en tu casa.');
    else if (side.momio != null) facts.push('Momio de ' + sourceLabel(g, side.fuente) + '.');
    if (market.sin_vig) facts.push('Mercado sin vig: ' + market.sin_vig.replace('espn', sourceLabel(g, 'espn')) + '.');
    if (side.ev != null) facts.push('Valor esperado al momio real: ' + signed(side.ev * 100, 1, '%') + ' por unidad.');
    if (side.estable === true) facts.push('Estable: el pick se mantiene al quitar los últimos 2 partidos de cada equipo.');
    if (side.kelly) facts.push('Kelly completo ' + (side.kelly * 100).toFixed(1) + '%; se sugiere 1/4 con tope de 2%.');
    if (tipo !== 'total' && g.proyeccion.margen_sin_ultimos2 != null) {
      facts.push('Sin esos partidos el margen del local sería ' + signed(g.proyeccion.margen_sin_ultimos2, 1) + ' (ahora ' + signed(g.proyeccion.margen, 1) + ').');
    }
    if (tipo === 'total' && g.proyeccion.total_sin_ultimos2 != null) {
      facts.push('Sin los últimos 2 partidos el total sería ' + trim(g.proyeccion.total_sin_ultimos2) + ' (ahora ' + trim(g.proyeccion.total) + ').');
    }
    body.append(h('p', null, facts.join(' ')));
    if (side.motivos.length) body.append(h('ul', null, side.motivos.map((mv) => h('li', null, mv))));
    return h('details', { class: 'mkt' }, summary, body);
  }

  function playsList(games) {
    const rows = [];
    for (const g of games) {
      for (const tipo of visibleTypes()) {
        const p = pickOf(g, tipo);
        if (p && p.side.edge != null) rows.push(p);
      }
    }
    rows.sort((a, b) => b.side.edge - a.side.edge);
    const plays = rows.filter((r) => r.side.etiqueta === 'JUGADA');
    const avoid = rows.filter((r) => r.side.etiqueta === 'EVITAR');
    const out = h('div', { class: 'stack' });

    const exposure = plays.reduce((acc, r) => acc + r.side.stake, 0);
    const head = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Jugadas por edge'));
    if (plays.length) {
      head.append(h('p', { class: 'small' }, plays.length + ' jugadas · exposición total ' + (exposure * 100).toFixed(1) + '% del bankroll' +
        (state.bankroll > 0 ? ' (' + money(exposure * state.bankroll) + ')' : '') +
        '. ML y spread del mismo equipo son casi la misma apuesta: elige una.'));
      head.append(h('div', { class: 'plays' }, plays.map((r, i) => playRow(r, i + 1))));
    } else {
      head.append(h('p', { class: 'empty' }, 'Ninguna jugada supera el 5% de edge con estos filtros.'));
    }
    out.append(head);
    if (avoid.length && state.solo !== 'jugada') {
      out.append(h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Evitar'),
        h('p', { class: 'small' }, 'Tienen edge en papel, pero el pick es inestable o el mercado se movió en contra.'),
        h('div', { class: 'plays' }, avoid.map((r) => playRow(r, null)))));
    }
    return out;
  }

  function playRow(r, rank) {
    const { g, tipo, lado, side } = r;
    const stake = stakeText(side);
    return h('div', { class: 'play' },
      rank ? h('span', { class: 'rankno' }, rank + '.') : pill('EVITAR'),
      h('span', { class: 'what' }, sideLabel(g, tipo, lado), ' ', h('span', { class: 'price' }, '(' + oddsFmt(side.momio) + ')')),
      h('span', { class: 'edge' }, edgeFmt(side.edge)),
      h('span', { class: 'sub' }, TYPE_LABEL[tipo] + ' · ' + matchup(g) + ' · ' + when(g.fecha) + ' · modelo ' + pct(side.p_modelo) + ' vs mercado ' + pct(side.p_mercado) +
        (stake ? ' · ' + stake : '') + (side.motivos.length ? ' · ' + side.motivos[0] : '')));
  }

  // ================================================================= PARLAY
  function legOf(g, tipo, lado) {
    const side = g[tipo] && g[tipo][lado];
    return side ? { g, tipo, lado, side, label: sideLabel(g, tipo, lado) } : null;
  }
  function selectedLegs() {
    const byId = new Map(allGames().map((g) => [g.id, g]));
    const legs = [];
    for (const [id, sel] of Object.entries(state.legs || {})) {
      const g = byId.get(id);
      const leg = g ? legOf(g, sel.tipo, sel.lado) : null;
      if (leg) legs.push(leg);
    }
    return legs.sort((a, b) => a.g.fecha.localeCompare(b.g.fecha));
  }
  function parlayMath(legs) {
    const out = { n: legs.length, pModel: null, pMarket: null, decimal: null };
    if (!legs.length) return out;
    out.pModel = legs.reduce((acc, l) => acc * l.side.p_modelo, 1);
    if (legs.every((l) => l.side.p_mercado != null)) out.pMarket = legs.reduce((acc, l) => acc * l.side.p_mercado, 1);
    if (legs.every((l) => l.side.momio != null)) out.decimal = legs.reduce((acc, l) => acc * toDecimal(l.side.momio), 1);
    return out;
  }
  function toggleLeg(g, tipo, lado) {
    const legs = Object.assign({}, state.legs);
    const cur = legs[g.id];
    if (cur && cur.tipo === tipo && cur.lado === lado) delete legs[g.id];
    else legs[g.id] = { tipo, lado };   // una pierna por juego: dos del mismo juego no son independientes
    setState('legs', legs, 'leg-' + g.id + '-' + tipo + '-' + lado);
  }

  function renderParlay() {
    const out = [];
    for (const saved of state.picks.parlays || []) out.push(savedParlay(saved));

    const builder = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Arma tu parlay'),
      h('p', { class: 'small' }, 'Una pierna por juego. La probabilidad multiplica las del modelo y supone que los juegos son independientes.'));
    const row = h('div', { class: 'filters' });
    if (mineCount()) {
      row.append(seg('Juegos', [['mine', 'Mis juegos'], ['all', 'Todos']], state.scope === 'mine' ? 'mine' : 'all', (v, id) => setState('scope', v, id)));
    }
    row.append(seg('Mercado', [['ml', 'ML'], ['spread', 'Spread'], ['total', 'Total']], state.parlayTipo, (v, id) => { state.parlayTipo = v; refocus = id; render(); }));
    builder.append(row);
    const list = h('div');
    const tipo = state.parlayTipo;
    for (const g of scopedGames().slice().sort((a, b) => a.fecha.localeCompare(b.fecha))) {
      const opts = h('div', { class: 'legopts' });
      let any = false;
      for (const lado of sideKeys(tipo)) {
        const leg = legOf(g, tipo, lado);
        if (!leg) continue;
        any = true;
        const cur = state.legs[g.id];
        const on = !!cur && cur.tipo === tipo && cur.lado === lado;
        opts.append(h('button', { type: 'button', class: 'leg', id: 'leg-' + g.id + '-' + tipo + '-' + lado, 'aria-pressed': String(on), onclick: () => toggleLeg(g, tipo, lado) },
          h('b', null, leg.label), h('span', { class: 'o' }, oddsFmt(leg.side.momio)),
          h('span', { class: 'p' }, 'modelo ' + pct(leg.side.p_modelo) + (leg.side.p_mercado != null ? ' · mercado ' + pct(leg.side.p_mercado) : ''))));
      }
      if (any) list.append(h('div', { class: 'legrow' }, h('span', { class: 'when' }, matchup(g) + ' · ' + when(g.fecha)), opts));
    }
    builder.append(list.children.length ? list : h('p', { class: 'empty' }, 'No hay líneas de este mercado para los juegos elegidos.'));
    out.push(builder);
    out.push(slip());
    return out;
  }

  function savedParlay(saved) {
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Tu parlay: ' + saved.nombre));
    const legs = (saved.piernas || []).filter((l) => !l.sin_datos);
    if (saved.p_modelo == null) {
      box.append(h('p', { class: 'empty' }, 'No encontré juegos pendientes para estas piernas.'));
      return box;
    }
    const kv = h('div', { class: 'kv' });
    kv.append(h('div', null, h('div', { class: 'k' }, 'Según el modelo'), h('div', { class: 'v' }, pct(saved.p_modelo, 1)), h('div', { class: 's' }, legs.length + ' piernas multiplicadas')));
    if (saved.p_mercado != null) kv.append(h('div', null, h('div', { class: 'k' }, 'Mercado sin vig'), h('div', { class: 'v' }, pct(saved.p_mercado, 1)), h('div', { class: 's' }, 'lo que el propio mercado estima')));
    if (saved.prob_casa != null) {
      const am = decimalToAmerican(saved.decimal);
      kv.append(h('div', null, h('div', { class: 'k' }, 'La casa paga como'), h('div', { class: 'v' }, pct(saved.prob_casa, 1)), h('div', { class: 's' }, 'momio ' + oddsFmt(am) + ' (x' + saved.decimal.toFixed(2) + ')')));
      kv.append(h('div', null, h('div', { class: 'k' }, 'Valor esperado'), h('div', { class: 'v neg' }, signed(saved.ev_modelo * 100, 0, '%')),
        h('div', { class: 's' }, 'por unidad, según el modelo' + (saved.ev_mercado != null ? '; ' + signed(saved.ev_mercado * 100, 0, '%') + ' según el mercado' : ''))));
    }
    box.append(kv);
    box.append(h('p', { class: 'small' }, 'Cada pierna se paga con vig, y en un parlay el vig se multiplica: por eso la casa paga como si fuera más probable de lo que el propio mercado cree. ' +
      'Pierna más débil: ' + saved.mas_debil + (saved.peor_valor && saved.peor_valor !== saved.mas_debil ? '; peor valor: ' + saved.peor_valor : '') + '.'));
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null, ['Pierna', 'Modelo', 'Mercado', 'Momio', 'Edge'].map((t) => h('th', null, t)))));
    const body = h('tbody');
    for (const leg of legs.slice().sort((a, b) => a.p_modelo - b.p_modelo)) {
      body.append(h('tr', { class: leg.equipo === saved.mas_debil ? 'hl weak' : null },
        h('td', { class: 'cell2' }, leg.equipo, h('small', null, (leg.sede === 'local' ? 'vs ' : '@ ') + leg.rival)), h('td', null, pct(leg.p_modelo, 1)),
        h('td', null, leg.p_mercado == null ? '–' : pct(leg.p_mercado, 1)), h('td', null, oddsFmt(leg.momio)),
        h('td', { class: leg.edge == null ? null : leg.edge >= 0 ? 'pos' : 'neg' }, edgeFmt(leg.edge))));
    }
    table.append(body);
    box.append(h('div', { class: 'scroll' }, table));
    for (const leg of (saved.piernas || []).filter((l) => l.sin_datos)) box.append(h('p', { class: 'small' }, 'Sin datos para ' + leg.equipo + '.'));
    box.append(h('div', null, h('button', { type: 'button', class: 'btn', id: 'load-' + saved.nombre.replace(/\W+/g, ''), onclick: () => {
      const next = {};
      for (const leg of legs) next[leg.juego_id] = { tipo: 'ml', lado: leg.lado };
      state.parlayTipo = 'ml';
      state.slipOpen = true;
      setState('legs', next, 'slip');
    } }, 'Cargar en el builder')));
    return box;
  }

  function slip() {
    const legs = selectedLegs();
    const math = parlayMath(legs);
    const american = math.decimal != null ? decimalToAmerican(math.decimal) : null;
    const headline = !legs.length ? 'vacío' : pct(math.pModel, 1) + ' · ' + (american != null ? 'paga ' + oddsFmt(american) : 'sin momio');
    const box = h('details', { class: 'slip', id: 'slip', open: state.slipOpen === true, ontoggle: (e) => { state.slipOpen = e.target.open; } });
    box.append(h('summary', null, h('span', { class: 'h3' }, 'Boleto · ' + legs.length + (legs.length === 1 ? ' pierna' : ' piernas')), h('span', { class: 'slip-sum' }, headline)));
    const body = h('div', { class: 'slip-body' });
    box.append(body);
    if (!legs.length) {
      body.append(h('p', { class: 'small' }, 'Toca un equipo arriba para agregarlo.'));
      return box;
    }
    const grid = h('div', { class: 'slip-grid' });
    grid.append(h('div', null, h('div', { class: 'k' }, 'Prob. modelo'), h('div', { class: 'v' }, pct(math.pModel, 1))));
    if (math.pMarket != null) grid.append(h('div', null, h('div', { class: 'k' }, 'Mercado sin vig'), h('div', { class: 'v' }, pct(math.pMarket, 1))));
    if (math.decimal != null) {
      const ev = math.pModel * math.decimal - 1;
      grid.append(h('div', null, h('div', { class: 'k' }, 'Paga'), h('div', { class: 'v' }, oddsFmt(american))));
      grid.append(h('div', null, h('div', { class: 'k' }, 'Casa implica'), h('div', { class: 'v' }, pct(1 / math.decimal, 1))));
      grid.append(h('div', null, h('div', { class: 'k' }, 'Valor esperado'), h('div', { class: 'v ' + (ev >= 0 ? 'pos' : 'neg') }, signed(ev * 100, 0, '%'))));
    } else {
      grid.append(h('div', null, h('div', { class: 'k' }, 'Paga'), h('div', { class: 'v' }, '–'), h('div', { class: 'small' }, 'falta el momio de alguna pierna')));
    }
    body.append(grid);
    if (math.decimal != null) {
      body.append(h('label', { class: 'field-input', for: 'slip-stake' }, 'Apuesta $',
        h('input', { id: 'slip-stake', type: 'number', min: '0', step: '1', inputmode: 'decimal', value: String(state.stake),
          onchange: (e) => setState('stake', Math.max(0, Number(e.target.value) || 0), 'slip-stake') }),
        h('span', null, 'cobras ' + money(state.stake * math.decimal) + ' si entran todas')));
    }
    const weakest = legs.reduce((a, b) => (a.side.p_modelo <= b.side.p_modelo ? a : b));
    body.append(h('p', { class: 'small' }, 'Pierna más débil: ' + weakest.label + ' (' + pct(weakest.side.p_modelo) + '). Toca una pierna para quitarla.'));
    body.append(h('div', { class: 'slip-legs' }, legs.map((l) => h('button', { type: 'button', 'aria-label': 'Quitar ' + l.label, onclick: () => toggleLeg(l.g, l.tipo, l.lado) },
      l.label + ' ' + oddsFmt(l.side.momio) + ' ×'))));
    body.append(h('div', null, h('button', { type: 'button', class: 'btn ghost', onclick: () => setState('legs', {}) }, 'Limpiar boleto')));
    return box;
  }

  // =========================================================== TRACK RECORD
  function renderRecord() {
    const B = state.backtest, out = [];
    const sel = B.comparacion.find((c) => c.seleccionado) || B.comparacion[0];
    const cur = sel.actual, mkt = B.mercado;
    if (!cur || !cur.n) {
      return [h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Track record'),
        h('p', { class: 'empty' }, 'Todavía no hay juegos jugados en ' + B.temporada + '. El track record aparece después de la primera semana.'))];
    }

    const kpi = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Cómo le ha ido al modelo en ' + B.temporada),
      h('p', { class: 'small' }, sel.label + '. Cada semana se predijo solo con lo jugado antes (walk-forward), ' + cur.n + ' juegos FBS vs FBS.'));
    const kv = h('div', { class: 'kv' });
    kv.append(h('div', null, h('div', { class: 'k' }, 'Ganador acertado'), h('div', { class: 'v' }, rate(cur.acierto)), h('div', { class: 's' }, mkt ? 'el mercado: ' + rate(mkt.acierto) : '')));
    kv.append(h('div', null, h('div', { class: 'k' }, 'Con 14+ pts de ventaja'), h('div', { class: 'v' }, rate(cur.acierto_14)), h('div', { class: 's' }, cur.n_14 + ' juegos')));
    kv.append(h('div', null, h('div', { class: 'k' }, 'Error del margen'), h('div', { class: 'v' }, cur.mae_margen.toFixed(1) + ' pts'), h('div', { class: 's' }, mkt ? 'el mercado: ' + trim(mkt.mae_margen, 1) : '')));
    kv.append(h('div', null, h('div', { class: 'k' }, 'Error del total'), h('div', { class: 'v' }, cur.mae_total.toFixed(1) + ' pts'), h('div', { class: 's' }, mkt ? 'el mercado: ' + trim(mkt.mae_total, 1) : '')));
    kpi.append(kv);
    out.push(kpi);

    out.push(comparisonPanel(B, sel));
    out.push(roiPanel(B));
    out.push(weeksPanel(B));
    out.push(calibrationPanel(B));
    return out;
  }

  function comparisonPanel(B, sel) {
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Modelos comparados'),
      h('p', { class: 'small' }, 'Error absoluto medio del margen en ' + B.temporada + ' (menos es mejor). Se eligió el de menor error; la línea de cierre del mercado va como referencia.'));
    const rows = B.comparacion.map((c) => ({ name: c.label.replace(/\s*\(.*\)$/, ''), value: c.actual.mae_margen, hl: c.seleccionado }));
    if (B.mercado) rows.push({ name: 'Mercado (cierre)', value: B.mercado.mae_margen, hl: false });
    rows.sort((a, b) => b.value - a.value);
    const max = Math.max(...rows.map((r) => r.value));
    const bars = h('div', { class: 'bars' });
    for (const r of rows) {
      bars.append(h('span', { class: 'name' + (r.hl ? ' hl' : '') }, r.name),
        h('span', null, h('span', { class: 'bar' + (r.hl ? ' hl' : ''), style: 'display:block;width:' + ((r.value / max) * 100).toFixed(1) + '%' })),
        h('span', { class: 'num' }, r.value.toFixed(2)));
    }
    box.append(bars);

    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['Modelo', 'Acierto', '14+ pts', 'Error margen', 'Error total', 'Brier', B.calibrado_con + ' error'].map((t) => h('th', null, t)))));
    const body = h('tbody');
    for (const c of B.comparacion) {
      const a = c.actual, hist = c.historia || {};
      body.append(h('tr', { class: c.seleccionado ? 'hl' : null },
        h('td', { class: 'wrapcell' }, c.label), h('td', null, rate(a.acierto)), h('td', null, rate(a.acierto_14)), h('td', null, a.mae_margen.toFixed(2)),
        h('td', null, a.mae_total == null ? '–' : a.mae_total.toFixed(2)), h('td', null, a.brier.toFixed(3)), h('td', null, hist.mae_margen == null ? '–' : hist.mae_margen.toFixed(2))));
    }
    const ref = B.referencia_prototipo;
    body.append(h('tr', null, h('td', { class: 'wrapcell' }, 'Prototipo original (referencia)'), h('td', null, rate(ref.acierto, 0)), h('td', null, rate(ref.acierto_14, 0)),
      h('td', null, '~' + ref.mae_margen), h('td', null, '~' + ref.mae_total), h('td', null, '–'), h('td', null, '–')));
    table.append(body);
    const det = h('details', null, h('summary', { class: 'small' }, 'Ver tabla completa'), h('div', { class: 'scroll' }, table),
      h('p', { class: 'small' }, 'Con juegos contra FCS incluidos (como el prototipo), el modelo elegido acierta ' + rate(sel.actual_con_fcs.acierto) +
        ' y ' + rate(sel.actual_con_fcs.acierto_14) + ' con 14+ puntos de ventaja.'));
    box.append(det);
    return box;
  }

  function roiPanel(B) {
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'ROI simulado de las jugadas'));
    const cur = (B.apuestas.actual || {}).todas, hist = (B.apuestas.historia || {}).todas;
    if (!cur || !cur.JUGADA) {
      box.append(h('p', { class: 'empty' }, 'Todavía no hay líneas de cierre para simular.'));
      return box;
    }
    const j = cur.JUGADA;
    box.append(h('p', { class: 'small' }, 'Una unidad por cada JUGADA, contra la línea de cierre. ' + B.temporada + ': ' + j.ganadas + '–' + j.perdidas + ', ' +
      signed(j.unidades, 1, ' unidades') + ', ROI ' + signed(j.roi * 100, 1, '%') + ' ± ' + trim(j.roi_error * 100) + '%.' +
      (hist && hist.JUGADA ? ' Con todas las temporadas que tienen líneas (' + B.apuestas.temporadas_con_lineas.join(', ') + '): ROI ' + signed(hist.JUGADA.roi * 100, 1, '%') + '.' : '')));
    const chart = h('div', { class: 'chart', id: 'chart-roi' });
    box.append(h('h3', { class: 'h3' }, 'Unidades acumuladas por semana'), chart);
    const weeks = B.semanas.filter((w) => w.jugadas > 0);
    queueChart(() => drawUnits(chart, weeks));
    const two = h('div', { class: 'two' });
    two.append(h('div', null, h('h3', { class: 'h3' }, B.temporada + ' por tipo'), h('div', { class: 'scroll' }, recordTable(B.apuestas.actual))));
    two.append(h('div', null, h('h3', { class: 'h3' }, 'Temporadas ' + B.apuestas.temporadas_con_lineas.join(', ')), h('div', { class: 'scroll' }, recordTable(B.apuestas.historia))));
    box.append(two);
    return box;
  }

  function weeksPanel(B) {
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Semana a semana'),
      h('p', { class: 'small' }, 'Margen y Total son el error medio en puntos; Jugadas, el récord de las JUGADA de esa semana contra la línea de cierre.'));
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['Sem.', 'Acierto', 'Margen', 'Total', 'Jugadas', 'Unid.'].map((t) => h('th', { class: t === 'Total' ? 'opt' : null }, t)))));
    const body = h('tbody');
    for (const w of B.semanas) {
      body.append(h('tr', null, h('td', null, String(w.semana)),
        h('td', null, h('span', { class: 'meter', 'aria-hidden': 'true' }, h('i', { style: 'width:' + (w.acierto * 100).toFixed(0) + '%' })), rate(w.acierto, 0), h('span', { class: 'pm opt' }, ' de ' + w.n)),
        h('td', null, w.mae_margen.toFixed(1)), h('td', { class: 'opt' }, w.mae_total == null ? '–' : w.mae_total.toFixed(1)),
        h('td', null, w.jugadas ? w.ganadas + '–' + w.perdidas : '–'),
        h('td', { class: w.jugadas ? (w.unidades >= 0 ? 'pos' : 'neg') : null }, w.jugadas ? signed(w.unidades, 1, 'u') : '–')));
    }
    table.append(body);
    box.append(h('div', { class: 'scroll' }, table));
    box.append(h('p', { class: 'small' }, 'Las primeras semanas no tienen jugadas: con menos de ' + B.reglas.min_games + ' partidos por equipo el rating todavía no es fiable.'));
    return box;
  }

  function calibrationPanel(B) {
    const cal = B.calibracion;
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Calibración'),
      h('p', { class: 'small' }, 'Cuando el modelo da X% a su favorito, ¿gana X% de las veces? Sobre la diagonal está bien calibrado. Desvío medio histórico: ' +
        (cal.error_medio == null ? '–' : trim(cal.error_medio * 100, 1) + ' puntos porcentuales') + '.'));
    const hist = (B.temporadas_historia || []);
    const histName = hist.length ? hist[0] + '–' + hist[hist.length - 1] : 'Historia';
    box.append(h('div', { class: 'legend' },
      h('span', null, h('i', { class: 'dot', style: 'background:var(--series-1)' }), String(B.temporada)),
      h('span', null, h('i', { class: 'dot', style: 'background:var(--series-2)' }), histName),
      h('span', null, h('i', { style: 'background:var(--track);height:1px' }), 'Calibración perfecta')));
    const chart = h('div', { class: 'chart', id: 'chart-cal' });
    box.append(chart);
    queueChart(() => drawCalibration(chart, cal.actual, cal.historia, String(B.temporada), histName));
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['El modelo dijo', B.temporada + ' ganó', 'Juegos', histName + ' ganó', 'Juegos'].map((t) => h('th', null, t)))));
    const body = h('tbody');
    const byBin = new Map(cal.historia.map((b) => [b.desde, b]));
    const bins = Array.from(new Set([...cal.actual.map((b) => b.desde), ...cal.historia.map((b) => b.desde)])).sort((a, b) => a - b);
    const curBy = new Map(cal.actual.map((b) => [b.desde, b]));
    for (const lo of bins) {
      const c = curBy.get(lo), hb = byBin.get(lo);
      body.append(h('tr', null, h('td', null, Math.round(lo * 100) + '–' + Math.round((lo + 0.05) * 100) + '%'),
        h('td', null, c ? rate(c.observado, 0) : '–'), h('td', null, c ? String(c.n) : '–'),
        h('td', null, hb ? rate(hb.observado, 0) : '–'), h('td', null, hb ? String(hb.n) : '–')));
    }
    table.append(body);
    box.append(h('details', null, h('summary', { class: 'small' }, 'Ver como tabla'), h('div', { class: 'scroll' }, table)));
    return box;
  }

  // --------------------------------------------------------------- gráficas
  let pendingCharts = [];
  const queueChart = (fn) => pendingCharts.push(fn);
  function flushCharts() {
    const jobs = pendingCharts;
    pendingCharts = [];
    for (const job of jobs) job();
  }
  function niceStep(span, target) {
    const raw = span / Math.max(target, 1);
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    const base = raw / pow;
    return (base <= 1 ? 1 : base <= 2 ? 2 : base <= 5 ? 5 : 10) * pow;
  }

  function drawUnits(el, weeks) {
    el.replaceChildren();
    if (!weeks.length) { el.append(h('p', { class: 'empty' }, 'Sin jugadas todavía.')); return; }
    const W = Math.max(el.clientWidth, 260), H = 210, m = { l: 36, r: 54, t: 12, b: 30 };
    const values = weeks.map((w) => w.acumulado);
    const lo = Math.min(0, ...values), hi = Math.max(0, ...values);
    const step = niceStep((hi - lo) || 1, 4);
    const y0 = Math.floor(lo / step) * step, y1 = Math.ceil(hi / step) * step || step;
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const x = (i) => m.l + (weeks.length === 1 ? iw / 2 : (i / (weeks.length - 1)) * iw);
    const y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * ih;
    const svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'aria-label': 'Unidades acumuladas de las jugadas por semana' });
    for (let v = y0; v <= y1 + 1e-9; v += step) {
      svg.append(s('line', { class: Math.abs(v) < 1e-9 ? 'axis' : 'grid', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }),
        s('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, signed(v, 0)));
    }
    const every = Math.ceil(weeks.length / Math.max(Math.floor(iw / 30), 1));
    weeks.forEach((w, i) => {
      if (i % every === 0 || i === weeks.length - 1) svg.append(s('text', { x: x(i), y: H - 10, 'text-anchor': 'middle' }, 'S' + w.semana));
    });
    const pts = weeks.map((w, i) => [x(i), y(w.acumulado)]);
    const path = pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(' ');
    svg.append(s('path', { d: path + ' L' + pts[pts.length - 1][0].toFixed(1) + ' ' + y(0) + ' L' + pts[0][0].toFixed(1) + ' ' + y(0) + ' Z', fill: 'var(--series-1)', 'fill-opacity': '0.1', stroke: 'none' }));
    svg.append(s('path', { d: path, fill: 'none', stroke: 'var(--series-1)', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    const last = pts[pts.length - 1];
    svg.append(s('circle', { cx: last[0], cy: last[1], r: 5, fill: 'var(--series-1)', stroke: 'var(--surface)', 'stroke-width': 2 }));
    svg.append(s('text', { class: 'lbl', x: last[0] + 9, y: last[1] + 4 }, signed(values[values.length - 1], 1, 'u')));
    const cross = s('line', { class: 'axis', y1: m.t, y2: H - m.b, x1: 0, x2: 0, visibility: 'hidden' });
    const dot = s('circle', { r: 5, fill: 'var(--series-1)', stroke: 'var(--surface)', 'stroke-width': 2, visibility: 'hidden' });
    svg.append(cross, dot);
    const band = weeks.length > 1 ? iw / (weeks.length - 1) : iw;
    weeks.forEach((w, i) => {
      const hit = s('rect', { x: x(i) - band / 2, y: m.t, width: band, height: ih, fill: 'transparent', tabindex: 0,
        'aria-label': 'Semana ' + w.semana + ': acumulado ' + signed(w.acumulado, 1) + ' unidades' });
      const on = (evt) => {
        cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
        dot.setAttribute('cx', x(i)); dot.setAttribute('cy', y(w.acumulado)); dot.setAttribute('visibility', 'visible');
        const box = svg.getBoundingClientRect();
        showTip(hit, [h('b', null, signed(w.acumulado, 1, 'u acumuladas')), h('span', null, 'Semana ' + w.semana + ': ' + w.ganadas + '–' + w.perdidas + ', ' + signed(w.unidades, 1, 'u'))],
          { x: box.left + (x(i) / W) * box.width, y: evt && evt.clientY ? evt.clientY : box.top + (y(w.acumulado) / H) * box.height });
      };
      const off = () => { cross.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); hideTip(); };
      hit.addEventListener('pointerenter', on);
      hit.addEventListener('pointermove', on);
      hit.addEventListener('pointerleave', off);
      hit.addEventListener('focus', on);
      hit.addEventListener('blur', off);
      svg.append(hit);
    });
    el.append(svg);
  }

  function drawCalibration(el, cur, hist, curName, histName) {
    el.replaceChildren();
    const W = Math.max(el.clientWidth, 260), H = Math.round(clamp(W * 0.72, 230, 360)), m = { l: 44, r: 14, t: 10, b: 42 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const observed = [...cur, ...hist].map((b) => b.observado);
    const yMin = Math.min(0.5, Math.floor(Math.min(...observed, 0.5) * 10) / 10);
    const x = (p) => m.l + ((p - 0.5) / 0.5) * iw;
    const y = (p) => m.t + (1 - (p - yMin) / (1 - yMin)) * ih;
    const svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'aria-label': 'Calibración: probabilidad del modelo contra frecuencia real' });
    for (let v = yMin; v <= 1 + 1e-9; v += 0.1) {
      svg.append(s('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }), s('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, Math.round(v * 100) + '%'));
    }
    for (let v = 0.5; v <= 1 + 1e-9; v += 0.1) svg.append(s('text', { x: x(v), y: H - m.b + 16, 'text-anchor': 'middle' }, Math.round(v * 100) + '%'));
    svg.append(s('text', { x: m.l + iw / 2, y: H - 6, 'text-anchor': 'middle' }, 'Probabilidad que dio el modelo'));
    svg.append(s('text', { x: 12, y: m.t + ih / 2, 'text-anchor': 'middle', transform: 'rotate(-90 12 ' + (m.t + ih / 2) + ')' }, 'Frecuencia real'));
    svg.append(s('line', { class: 'axis', x1: x(0.5), y1: y(0.5), x2: x(1), y2: y(1) }));
    const series = [[hist, 'var(--series-2)', histName], [cur, 'var(--series-1)', curName]];
    for (const [bins, color] of series) {
      if (!bins.length) continue;
      const d = bins.map((b, i) => (i ? 'L' : 'M') + x(b.predicho).toFixed(1) + ' ' + y(b.observado).toFixed(1)).join(' ');
      svg.append(s('path', { d, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    }
    for (const [bins, color, name] of series) {
      for (const b of bins) {
        const cx = x(b.predicho), cy = y(b.observado);
        svg.append(s('circle', { cx, cy, r: 4.5, fill: color, stroke: 'var(--surface)', 'stroke-width': 2 }));
        const hit = s('circle', { cx, cy, r: 13, fill: 'transparent', tabindex: 0,
          'aria-label': name + ': el modelo dijo ' + rate(b.predicho, 0) + ' y ganó ' + rate(b.observado, 0) + ' en ' + b.n + ' juegos' });
        const on = () => showTip(hit, [h('b', null, 'Ganó ' + rate(b.observado, 0)), h('span', null, name + ' · el modelo dijo ' + rate(b.predicho, 0) + ' · ' + b.n + ' juegos (±' + trim(b.error_est * 100, 0) + ')')]);
        hit.addEventListener('pointerenter', on);
        hit.addEventListener('focus', on);
        hit.addEventListener('pointerleave', hideTip);
        hit.addEventListener('blur', hideTip);
        svg.append(hit);
      }
    }
    el.append(svg);
  }

  // ========================================================== CÓMO FUNCIONA
  function renderHow() {
    const P = state.picks, B = state.backtest, r = P.reglas, m = P.modelo;
    const sec = (title, ...kids) => h('section', { class: 'panel' }, h('h2', { class: 'h2' }, title), h('div', { class: 'prose' }, kids));
    const p = (...kids) => h('p', null, kids);
    return [
      sec('El modelo en un párrafo',
        p('Cada equipo tiene un rating: su margen de puntos promedio, ajustado por la calidad de los rivales que enfrentó (un SRS). ' +
          'El rating de un equipo depende del de sus rivales y viceversa, así que se resuelven todos a la vez con ' + m.juegos_entrenamiento + ' juegos de la temporada. ' +
          'La proyección de un juego es la diferencia de ratings más la ventaja de local; un segundo sistema igual, sobre los totales, da el marcador.'),
        p('Con pocos juegos el rating se encoge hacia el del año anterior, y las palizas tienen tope para que un 63–0 no pese de más.')),
      sec('De la proyección a la probabilidad',
        p('El margen real se desvía del proyectado con una desviación estándar de ' + trim(m.sigma_margen) + ' puntos (' + trim(m.sigma_total) + ' para el total), medida en el backtest. ' +
          'Con esa campana se calcula la probabilidad de ganar, de cubrir el spread y de que el total supere la línea.'),
        p('El edge es la probabilidad del modelo menos la del mercado sin vig. El vig se quita con los dos lados de una misma casa; si solo hay un lado, se estima.')),
      sec('Las etiquetas',
        h('ul', null,
          h('li', null, h('b', null, 'JUGADA: '), 'edge de ' + (r.edge_min * 100).toFixed(0) + '% o más, valor esperado positivo al momio real y ninguna alerta.'),
          h('li', null, h('b', null, 'EVITAR: '), 'hay edge en papel, pero el pick cambia de lado al quitar los últimos 2 partidos de cada equipo, el modelo difiere ' +
            trim(r.gap_margin) + '+ puntos del spread (' + trim(r.gap_total) + '+ del total), o la línea se movió ' + trim(r.move_spread) + '+ puntos en contra desde la apertura (' + trim(r.move_total) + '+ el total). ' +
            'En el backtest esos picks son los que peor salieron.'),
          h('li', null, h('b', null, 'PASAR: '), 'no hay edge suficiente.')),
        p('El tamaño sugerido es 1/4 de Kelly con tope de ' + (r.stake_cap * 100).toFixed(0) + '% del bankroll por jugada.')),
      sec('Lo que el modelo no sabe',
        h('ul', null,
          h('li', null, 'Lesiones, suspensiones y cambios de quarterback.'),
          h('li', null, 'Clima. Un total que baja 10 puntos en la semana casi siempre es viento o lluvia.'),
          h('li', null, 'Motivación, descanso, viajes y juegos trampa.'),
          h('li', null, 'Todo lo que el mercado ya incorporó. Por eso la línea de cierre se equivoca menos que el modelo.')),
        p('Para meter esa información a mano, edita ', h('code', null, 'data/notas.csv'), ': un ajuste en puntos al margen del local o al total, y una nota por juego.')),
      sec('Cómo leer el track record',
        p('Todo es walk-forward: para la semana N solo se usa lo jugado antes. Los parámetros se calibraron con ' + (B.calibrado_con || 'temporadas pasadas') +
          ' y la temporada ' + B.temporada + ' se evaluó sin tocarla. El ROI simulado apuesta una unidad por JUGADA contra la línea de cierre, que es más dura que la línea del jueves.'),
        p('Ninguna proyección es segura. El máximo que se muestra es la probabilidad del modelo, y a su lado el error que tuvo en el backtest.')),
    ];
  }

  // ------------------------------------------------------------- arranque
  const VIEWS = { picks: renderPicks, parlay: renderParlay, record: renderRecord, como: renderHow };
  function currentView() {
    const name = (location.hash || '').replace(/^#/, '');
    return VIEWS[name] ? name : 'picks';
  }
  function render() {
    hideTip();
    const name = currentView();
    document.querySelectorAll('.tabs a').forEach((a) => {
      if (a.dataset.view === name) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
    if (state.error) {
      view.replaceChildren(h('section', { class: 'panel notice' }, h('h2', { class: 'h3' }, 'No pude cargar los picks'),
        h('p', null, state.error), h('p', { class: 'small' }, 'Genera los datos con python -m src.picks y abre el sitio desde un servidor (python -m http.server), no como archivo local.')));
      return;
    }
    if (!state.picks) return;
    pendingCharts = [];
    view.replaceChildren(...VIEWS[name]());
    flushCharts();
    if (refocus) {
      const el = document.getElementById(refocus);
      if (el) el.focus({ preventScroll: true });
      refocus = null;
    }
  }

  async function load() {
    try {
      const [picks, backtest] = await Promise.all(['data/picks.json', 'data/backtest.json'].map(async (url) => {
        const resp = await fetch(url, { cache: 'no-store' });
        if (!resp.ok) throw new Error(url + ' respondió HTTP ' + resp.status);
        return resp.json();
      }));
      state.picks = picks;
      state.backtest = backtest;
      const weekKey = picks.temporada + '-' + picks.semana;
      if (store.get('legsWeek', null) !== weekKey) {
        state.legs = {};
        store.set('legs', {});
        store.set('legsWeek', weekKey);
      }
      document.getElementById('hdr-week').textContent = picks.semana_label + ' · ' + picks.temporada;
      const n = allGames().length, mine = mineCount();
      document.getElementById('hdr-meta').textContent = 'Actualizado ' + stamp(picks.generado) + ' · ' + n + ' juegos' + (mine ? ', ' + mine + ' con tus líneas' : '');
      document.getElementById('foot-gen').textContent = 'Datos y líneas: API público de ESPN. Modelo: ' + picks.modelo.label + '.';
    } catch (err) {
      state.error = String((err && err.message) || err);
      document.getElementById('hdr-meta').textContent = 'Sin datos';
    }
    render();
  }

  window.addEventListener('hashchange', () => { render(); window.scrollTo(0, 0); view.focus({ preventScroll: true }); });
  let resizeTimer = null, lastWidth = window.innerWidth;
  window.addEventListener('resize', () => {
    if (window.innerWidth === lastWidth) return;
    lastWidth = window.innerWidth;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (currentView() === 'record') render(); }, 150);
  });
  window.addEventListener('scroll', hideTip, { passive: true });
  load();
})();
