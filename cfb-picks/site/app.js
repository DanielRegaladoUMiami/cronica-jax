/* CFB Picks: a static site that reads data/picks.json and data/backtest.json.
   No dependencies. Any text that comes from the data goes in through textContent. */
(function () {
  'use strict';

  // ----------------------------------------------------------------- helpers
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
  const b = (...kids) => h('b', null, kids);

  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const trim = (x, d = 1) => Number(x).toFixed(d).replace(/\.0+$/, '');
  // A probability is never shown as 100% or 0%: the display tops out at 99%.
  const pct = (p, d = 0) => (p == null ? '–' : clamp(p * 100, 1, 99).toFixed(d) + '%');
  const rate = (p, d = 1) => (p == null ? '–' : (p * 100).toFixed(d) + '%');
  function signed(x, d = 1, unit = '') {
    if (x == null) return '–';
    const txt = Math.abs(x).toFixed(d);
    if (Number(txt) === 0) return txt + unit;
    return (x > 0 ? '+' : MINUS) + txt + unit;
  }
  const edgeFmt = (e) => (e == null ? '–' : signed(e * 100, 1, '%'));
  const oddsFmt = (o) => (o == null ? 'no price' : (o > 0 ? '+' : MINUS) + Math.abs(o));
  const lineFmt = (x) => (x === 0 ? 'PK' : (x > 0 ? '+' : MINUS) + trim(Math.abs(x)));
  const toDecimal = (o) => (o > 0 ? 1 + o / 100 : 1 + 100 / Math.abs(o));
  function decimalToAmerican(dec) {
    if (!(dec > 1)) return null;
    return dec >= 2 ? Math.round((dec - 1) * 100) : -Math.round(100 / (dec - 1));
  }
  function fairOdds(p) {
    if (!(p > 0.005 && p < 0.995)) return null;
    return p >= 0.5 ? -Math.round((100 * p) / (1 - p)) : Math.round((100 * (1 - p)) / p);
  }
  const money = (x) => '$' + (Math.abs(x) >= 100 ? Math.round(x).toLocaleString('en-US') : x.toFixed(2));
  const oneIn = (p) => (!(p > 0) ? '–' : p >= 0.4 ? pct(p) : '1 in ' + Math.round(1 / p));
  const plural = (n, one, many) => n + ' ' + (n === 1 ? one : many || one + 's');
  const cap = (text) => text.charAt(0).toUpperCase() + text.slice(1);

  let dfDay = null, dfTime = null, dfStamp = null;
  try {
    dfDay = new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    dfTime = new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
    dfStamp = new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
  } catch (err) { /* no Intl: fall back to ISO strings */ }
  const dayKey = (d) => d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
  const timeFmt = (iso) => (dfTime ? dfTime.format(new Date(iso)) : iso);
  const dateFmt = (iso) => (dfDay ? dfDay.format(new Date(iso)) : iso.slice(0, 10));
  const kickFmt = (iso) => dateFmt(iso) + ' · ' + timeFmt(iso);
  const stamp = (iso) => (dfStamp ? dfStamp.format(new Date(iso)) : iso);
  function dayLabel(iso) {
    const key = dayKey(new Date(iso));
    if (key === dayKey(new Date())) return 'Today · ' + dateFmt(iso);
    if (key === dayKey(new Date(Date.now() + 864e5))) return 'Tomorrow · ' + dateFmt(iso);
    return dateFmt(iso);
  }

  const store = {
    get(key, fallback) {
      try {
        const raw = localStorage.getItem('cfb:' + key);
        return raw == null ? fallback : JSON.parse(raw);
      } catch (err) { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem('cfb:' + key, JSON.stringify(value)); } catch (err) { /* storage unavailable */ }
    },
  };

  // ------------------------------------------------------------------- state
  const state = {
    picks: null,
    backtest: null,
    error: null,
    scope: store.get('scope', 'mine'),
    market: store.get('market', 'all'),
    playsOnly: store.get('playsOnly', false),
    bankroll: store.get('bankroll', 1000),
    stake: store.get('stake', 10),
    legs: store.get('legs', {}),
    query: '',
    showAllPlays: false,
    parlayMarket: 'ml',
    slipOpen: false,
  };
  const view = document.getElementById('view');
  const tip = document.getElementById('tooltip');
  const reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  let refocus = null;

  function setState(key, value, focusId) {
    state[key] = value;
    if (key !== 'showAllPlays') store.set(key, value);
    refocus = focusId || null;
    render();
  }

  // ------------------------------------------------------------- data access
  const MARKET_NAME = { ml: 'Moneyline', spread: 'Spread', total: 'Total' };
  const MARKET_SHORT = { ml: 'ML', spread: 'Spread', total: 'Total' };
  const allGames = () => state.picks.games.filter((g) => !g.no_data);
  const mineCount = () => allGames().filter((g) => g.on_my_list).length;
  const useMine = () => state.scope === 'mine' && mineCount() > 0;
  const scopedGames = () => (useMine() ? allGames().filter((g) => g.on_my_list) : allGames());
  const visibleMarkets = () => (state.market === 'all' ? ['ml', 'spread', 'total'] : [state.market]);
  const sideKeys = (market) => (market === 'total' ? ['over', 'under'] : ['away', 'home']);
  const matchup = (g) => g.away.abbr + ' @ ' + g.home.abbr;

  function sideLabel(g, market, key, withMl) {
    if (market === 'total') return (key === 'over' ? 'Over ' : 'Under ') + trim(g.total.line);
    const team = g[key].name;
    if (market === 'ml') return team + (withMl ? ' ML' : '');
    const line = key === 'home' ? g.spread.home_line : -g.spread.home_line;
    return team + ' ' + lineFmt(line);
  }
  function pickOf(g, market) {
    const block = g[market];
    const key = block && block.pick;
    const side = key ? block[key] : null;
    return side ? { g, market, key, side } : null;
  }
  function collectPicks(games) {
    const out = [];
    for (const g of games) {
      for (const market of visibleMarkets()) {
        const p = pickOf(g, market);
        if (p && p.side.edge != null) out.push(p);
      }
    }
    return out.sort((x, y) => y.side.edge - x.side.edge);
  }
  const providerName = (g) => (g.market.espn && g.market.espn.provider) || 'ESPN';
  function sourceName(g, source) {
    if (source === 'espn') return providerName(g);
    if (source === 'assumed') return 'an assumed price';
    return source || '';
  }
  function pill(label) {
    const cls = { PLAY: 'play', AVOID: 'avoid', PASS: 'pass' }[label] || 'none';
    return h('span', { class: 'pill ' + cls }, label);
  }
  function luminance(hex) {
    const ch = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)));
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2];
  }
  function badge(team, small) {
    const hex = /^[0-9a-f]{6}$/i.test(team.color || '') ? team.color : null;
    const style = hex ? 'background:#' + hex + ';color:' + (luminance(hex) > 0.45 ? '#111111' : '#ffffff') : null;
    return h('span', { class: 'badge' + (hex ? '' : ' plain') + (small ? ' sm' : ''), style, 'aria-hidden': 'true' }, team.abbr);
  }
  function stakeText(side) {
    if (!side.stake) return null;
    const dollars = state.bankroll > 0 ? ' (' + money(side.stake * state.bankroll) + ')' : '';
    return (side.stake * 100).toFixed(1) + '% of bankroll' + dollars;
  }

  // ---------------------------------------------------------------- controls
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

  // ------------------------------------------------------------- parlay legs
  const isLeg = (id, market, key) => { const cur = state.legs[id]; return !!cur && cur.market === market && cur.key === key; };
  function legOf(g, market, key) {
    const side = g[market] && g[market][key];
    return side ? { g, market, key, side, label: sideLabel(g, market, key, true) } : null;
  }
  function selectedLegs() {
    const byId = new Map(allGames().map((g) => [g.id, g]));
    const legs = [];
    for (const [id, sel] of Object.entries(state.legs || {})) {
      const g = byId.get(id);
      const leg = g && sel ? legOf(g, sel.market, sel.key) : null;
      if (leg) legs.push(leg);
    }
    return legs.sort((x, y) => x.g.date.localeCompare(y.g.date));
  }
  function updateParlayCount() {
    const el = document.getElementById('parlay-count');
    const n = state.picks ? selectedLegs().length : 0;
    el.hidden = n === 0;
    el.textContent = String(n);
  }
  function setLeg(g, market, key) {
    const legs = Object.assign({}, state.legs);
    if (isLeg(g.id, market, key)) delete legs[g.id];
    else legs[g.id] = { market, key };   // one leg per game: two legs from the same game are not independent
    state.legs = legs;
    store.set('legs', legs);
    updateParlayCount();
  }

  // =================================================================== PICKS
  function renderPicks() {
    const P = state.picks;
    const out = [realityCheck()];
    if (P.warnings && P.warnings.length) {
      out.push(h('section', { class: 'callout' }, h('b', { class: 't' }, 'Data warnings'),
        h('ul', { class: 'alerts' }, P.warnings.map((w) => h('li', null, w)))));
    }
    out.push(topPlays());
    out.push(controls());
    const list = h('div', { class: 'stack', id: 'games-list' });
    fillGames(list);
    out.push(list);
    return out;
  }

  function realityCheck() {
    const P = state.picks, m = P.model;
    const play = P.evidence && P.evidence.all && P.evidence.all.PLAY;
    const box = h('section', { class: 'callout', 'aria-label': 'Reality check' }, h('b', { class: 't' }, 'Reality check'));
    if (!m.n_backtest) {
      box.append(h('p', null, 'No ' + P.season + ' games have been played yet, so these projections lean on last season’s ratings and are less reliable.'));
      return box;
    }
    const line = h('p', null, 'The model has picked ', b(rate(m.accuracy, 0) + ' of winners'), ' this season');
    if (play && play.n) {
      line.append(', but it has ', b('not beaten the betting market'), '. Past plays like the ones below went ' + play.won + '–' + play.lost +
        ' against closing lines (ROI ' + signed(play.roi * 100, 1, '%') + '). Read a PLAY as “my number disagrees with the market,” not as a sure thing. ');
    } else {
      line.append('. ');
    }
    line.append(h('a', { href: '#record' }, 'See the track record'));
    box.append(line);
    return box;
  }

  function topPlays() {
    const games = scopedGames();
    const picks = collectPicks(games);
    const plays = picks.filter((p) => p.side.label === 'PLAY');
    const avoid = picks.filter((p) => p.side.label === 'AVOID');
    const box = h('section', { class: 'panel' });
    box.append(h('div', { class: 'panel-head' }, h('h2', { class: 'h2' }, 'Best plays this week'),
      h('label', { class: 'field-input', for: 'bankroll' }, 'Bankroll $',
        h('input', { id: 'bankroll', type: 'number', min: '0', step: '50', inputmode: 'decimal', value: String(state.bankroll),
          onchange: (e) => setState('bankroll', Math.max(0, Number(e.target.value) || 0), 'bankroll') }))));
    const where = useMine() ? 'your ' + plural(games.length, 'tracked game') : plural(games.length, 'game');
    if (!plays.length) {
      box.append(h('p', { class: 'empty' }, 'No play clears the 5% edge bar in ' + where + '.'));
    } else {
      box.append(h('p', { class: 'sub' }, plural(plays.length, 'play') + ' in ' + where + ', ranked by edge: the model’s chance minus the market’s. Tap one to see the game.'));
      const shown = state.showAllPlays ? plays : plays.slice(0, 5);
      box.append(h('div', { class: 'plays' }, shown.map((p, i) => playRow(p, i + 1))));
      if (plays.length > 5) {
        box.append(h('button', { type: 'button', class: 'btn more', id: 'more-plays', onclick: () => setState('showAllPlays', !state.showAllPlays, 'more-plays') },
          state.showAllPlays ? 'Show top 5 only' : 'Show all ' + plays.length + ' plays'));
      }
      const exposure = plays.reduce((acc, p) => acc + p.side.stake, 0);
      if (plays.length > 3) {
        box.append(h('p', { class: 'small' }, 'Betting every play at the suggested stake would put ' + (exposure * 100).toFixed(0) + '% of your bankroll' +
          (state.bankroll > 0 ? ' (' + money(exposure * state.bankroll) + ')' : '') + ' at risk in one week. A moneyline and a spread on the same team are nearly the same bet, so pick one.'));
      }
    }
    if (avoid.length) {
      box.append(h('details', null,
        h('summary', { class: 'h3' }, plural(avoid.length, 'pick') + ' to avoid'),
        h('p', { class: 'small', style: 'margin-top:8px' }, 'These have an edge on paper, but the pick is unstable or the line moved against it. Picks like these did worst in the backtest.'),
        h('div', { class: 'plays' }, avoid.map((p) => playRow(p, null)))));
    }
    return box;
  }

  function playRow(p, rank) {
    const { g, market, key, side } = p;
    const stake = stakeText(side);
    let ctx = MARKET_NAME[market] + ' · ' + matchup(g) + ' · ' + kickFmt(g.date);
    if (rank && stake) ctx += ' · stake ' + stake;
    if (!rank && side.reasons.length) ctx += ' · ' + side.reasons[0];
    return h('button', { type: 'button', class: 'playrow', onclick: () => goToGame(g.id, market) },
      rank ? h('span', { class: 'rankno' }, String(rank)) : pill('AVOID'),
      h('span', { class: 'what' }, sideLabel(g, market, key, true), ' ', h('span', { class: 'price' }, '(' + oddsFmt(side.odds) + ')')),
      h('span', { class: 'edge' }, edgeFmt(side.edge), h('small', null, pct(side.p_model) + ' vs ' + pct(side.p_market))),
      h('span', { class: 'ctx' }, ctx));
  }

  function goToGame(id, market) {
    if (state.query) {
      state.query = '';
      const q = document.getElementById('q');
      if (q) q.value = '';
      fillGames(document.getElementById('games-list'));
    }
    const card = document.getElementById('g-' + id);
    if (!card) return;
    const det = card.querySelector('details[data-market="' + market + '"]');
    if (det) det.open = true;
    card.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
    card.classList.add('flash');
    setTimeout(() => card.classList.remove('flash'), 1800);
  }

  function controls() {
    const box = h('section', { class: 'controls', 'aria-label': 'Filter games' });
    const mine = mineCount(), total = allGames().length;
    box.append(h('div', { class: 'controls-row' },
      h('div', { class: 'search' },
        h('input', { id: 'q', type: 'search', placeholder: 'Find a team', autocomplete: 'off', value: state.query, 'aria-label': 'Find a team',
          oninput: (e) => { state.query = e.target.value; fillGames(document.getElementById('games-list')); } }))));
    const row = h('div', { class: 'controls-row' });
    if (mine) {
      row.append(seg('Games', [['mine', 'My games (' + mine + ')'], ['all', 'All games (' + total + ')']], useMine() ? 'mine' : 'all', (v, id) => setState('scope', v, id)));
    }
    row.append(seg('Bet type', [['all', 'All bets'], ['ml', 'Moneyline'], ['spread', 'Spread'], ['total', 'Total']], state.market, (v, id) => setState('market', v, id)));
    row.append(h('label', { class: 'toggle', for: 'plays-only' },
      h('input', { id: 'plays-only', type: 'checkbox', checked: state.playsOnly, onchange: (e) => setState('playsOnly', e.target.checked, 'plays-only') }), 'Plays only'));
    box.append(row);
    return box;
  }

  function filteredGames() {
    const q = state.query.trim().toLowerCase();
    let list = q ? allGames() : scopedGames();   // a search looks through every game
    if (q) list = list.filter((g) => [g.home.name, g.away.name, g.home.abbr, g.away.abbr].some((x) => String(x).toLowerCase().includes(q)));
    if (state.playsOnly) list = list.filter((g) => visibleMarkets().some((m) => { const p = pickOf(g, m); return p && p.side.label === 'PLAY'; }));
    return list.slice().sort((x, y) => x.date.localeCompare(y.date));
  }

  function fillGames(container) {
    if (!container) return;
    const games = filteredGames();
    const out = [];
    if (!games.length) {
      out.push(h('p', { class: 'empty' }, state.query ? 'No team matches “' + state.query + '”.' : state.playsOnly ? 'No game has a PLAY with these filters.' : 'No pending games with a projection this week.'));
    }
    let current = null, grid = null;
    for (const g of games) {
      const key = dayKey(new Date(g.date));
      if (key !== current) {
        current = key;
        const count = games.filter((x) => dayKey(new Date(x.date)) === key).length;
        out.push(h('div', { class: 'day' }, h('h2', { class: 'h2' }, dayLabel(g.date)), h('span', { class: 'small' }, plural(count, 'game'))));
        grid = h('div', { class: 'cards' });
        out.push(grid);
      }
      grid.append(gameCard(g));
    }
    const missing = state.picks.games.filter((g) => g.no_data && (!useMine() || g.on_my_list));
    if (missing.length && !state.query) {
      out.push(h('section', { class: 'panel' }, h('h2', { class: 'h3' }, 'No projection'),
        h('ul', { class: 'alerts' }, missing.map((g) => h('li', { class: 'cmp' }, g.away.name + ' @ ' + g.home.name + ': ' + (g.no_data_reason || 'no data'))))));
    }
    container.replaceChildren(...out);
  }

  function gameCard(g) {
    const pr = g.projection, model = state.picks.model, espn = g.market.espn;
    const favHome = pr.margin >= 0;
    const card = h('article', { class: 'panel game', id: 'g-' + g.id });
    const tags = h('div', { class: 'tags' });
    if (g.on_my_list) tags.append(h('span', { class: 'tag' }, 'My lines'));
    if (g.neutral) tags.append(h('span', { class: 'tag' }, 'Neutral site'));
    if (g.fcs) tags.append(h('span', { class: 'tag' }, 'vs FCS'));
    card.append(h('div', { class: 'game-top' }, h('span', { class: 'kick' }, kickFmt(g.date)), tags));

    const teams = h('div', { class: 'teams' });
    for (const key of ['away', 'home']) {
      const t = g[key];
      const isFav = (key === 'home') === favHome;
      const meta = [t.record, t.rank ? 'AP #' + t.rank : null, t.rating_rank ? 'model #' + t.rating_rank : null,
        key === 'home' && !g.neutral ? 'home' : null].filter(Boolean).join(' · ');
      teams.append(
        h('div', { class: 'team' + (isFav ? ' fav' : '') }, badge(t), h('div', { class: 'team-name' }, h('b', null, t.name), h('span', null, meta))),
        h('div', { class: 'proj' }, pr[key] == null ? '–' : String(Math.round(pr[key]))),
        h('div', { class: 'wp' }, pct(key === 'home' ? pr.p_home : pr.p_away)));
    }
    card.append(teams);
    card.append(h('div', { class: 'wpbar', role: 'img', 'aria-label': 'Win chance: ' + g.away.name + ' ' + pct(pr.p_away) + ', ' + g.home.name + ' ' + pct(pr.p_home) },
      h('i', { class: favHome ? null : 'fav', style: 'flex:' + Math.max(pr.p_away, 0.02) }),
      h('i', { class: favHome ? 'fav' : null, style: 'flex:' + Math.max(pr.p_home, 0.02) })));
    card.append(h('div', { class: 'cap' }, h('span', null, 'Projected score and win chance'),
      model.mae_margin != null ? h('span', null, 'typical miss ±' + trim(model.mae_margin) + ' pts') : null));

    const fav = favHome ? g.home : g.away;
    const lines = h('dl', { class: 'lines' });
    lines.append(h('dt', null, 'Model'), h('dd', null, fav.name + ' by ' + trim(Math.abs(pr.margin)) + (pr.total != null ? ' · total ' + trim(pr.total) : '')));
    if (g.market.margin != null) {
      const mFav = g.market.margin >= 0 ? g.home : g.away;
      let text = Math.abs(g.market.margin) < 0.01 ? 'Pick’em' : mFav.name + ' by ' + trim(Math.abs(g.market.margin));
      if (espn && espn.total != null) text += ' · total ' + trim(espn.total);
      const moved = Math.abs(g.market.spread_move || 0) >= 1 || Math.abs(g.market.total_move || 0) >= 1;
      if (espn && espn.spread_open != null && moved) {
        text += ' · opened ' + (espn.spread_open === 0 ? 'PK' : g.home.abbr + ' ' + lineFmt(espn.spread_open)) + (espn.total_open != null ? ', ' + trim(espn.total_open) : '');
      }
      lines.append(h('dt', null, 'Market'), h('dd', null, text));
    }
    if (pr.home_margin_adj || pr.total_adj) {
      lines.append(h('dt', null, 'Manual'), h('dd', null, signed(pr.home_margin_adj, 1) + ' pts to the home margin, ' + signed(pr.total_adj, 1) + ' to the total'));
    }
    card.append(lines);

    const markets = h('div', { class: 'markets' });
    for (const market of visibleMarkets()) {
      const row = marketRow(g, market);
      if (row) markets.append(row);
    }
    if (markets.children.length) card.append(markets);

    const notes = [...(g.alerts || []).map((a) => h('li', null, cap(a))), ...(g.price_notes || []).map((c) => h('li', { class: 'cmp' }, c))];
    if (notes.length) card.append(h('ul', { class: 'alerts' }, notes));
    if (g.notes) card.append(h('p', { class: 'note' }, b('Note: '), g.notes));

    const add = h('div', { class: 'addrow' }, h('span', { class: 'small' }, 'Add to parlay'));
    for (const key of ['away', 'home']) {
      const side = g.ml[key];
      if (!side) continue;
      add.append(h('button', { type: 'button', class: 'addchip', id: 'add-' + g.id + '-' + key, 'aria-pressed': String(isLeg(g.id, 'ml', key)),
        'aria-label': g[key].name + ' moneyline in the parlay',
        onclick: () => {
          setLeg(g, 'ml', key);
          for (const k of ['away', 'home']) {
            const chip = document.getElementById('add-' + g.id + '-' + k);
            if (chip) chip.setAttribute('aria-pressed', String(isLeg(g.id, 'ml', k)));
          }
        } }, g[key].abbr + ' ' + oddsFmt(side.odds)));
    }
    if (add.children.length > 1) card.append(add);
    return card;
  }

  function marketRow(g, market) {
    const p = pickOf(g, market);
    if (!p) return null;
    const side = p.side, block = g[market];
    const summary = h('summary', null,
      h('span', { class: 'mkt-type' }, MARKET_NAME[market]),
      h('span', { class: 'mkt-pick' }, sideLabel(g, market, p.key), ' ', h('span', { class: 'price' }, '(' + oddsFmt(side.odds) + ')')),
      pill(side.label));
    const sub = h('span', { class: 'mkt-sub' }, h('span', null, 'model ', b(pct(side.p_model))));
    if (side.p_market != null) sub.append(h('span', null, 'market ', b(pct(side.p_market))), h('span', null, 'edge ', b(edgeFmt(side.edge))));
    const stake = stakeText(side);
    if (stake) sub.append(h('span', null, 'stake ', b(stake)));
    if (side.label === 'AVOID' && side.reasons.length) sub.append(h('span', null, cap(side.reasons[0])));
    summary.append(sub);

    const body = h('div', { class: 'mkt-body' }, explain(g, market, p.key, side));
    const facts = [];
    if (side.odds != null) {
      facts.push(side.source === 'assumed'
        ? 'Price assumed at ' + oddsFmt(side.odds) + ' because none was captured. Confirm the real price at your book.'
        : 'Price ' + oddsFmt(side.odds) + ' from ' + sourceName(g, side.source) + '.');
    }
    const fair = fairOdds(side.p_model);
    if (fair != null) facts.push('The model’s fair price would be ' + oddsFmt(fair) + '.');
    if (side.ev != null) facts.push('Expected value at this price: ' + signed(side.ev * 100, 1, '%') + ' per unit staked.');
    const dv = block.devig;
    if (dv && dv.method === 'two_sided') facts.push('Market chance comes from both sides of the ' + sourceName(g, dv.source) + ' price, with the bookmaker’s cut removed.');
    if (dv && dv.method === 'one_sided') facts.push('Only one side was priced at ' + sourceName(g, dv.source) + ', so a typical 4.5% bookmaker cut was removed.');
    if (dv && dv.method === 'assumed') facts.push('With no price captured, the market chance assumes standard ' + MINUS + '110 on both sides.');
    const pr = g.projection;
    if (market !== 'total' && pr.margin_without_last2 != null) {
      const now = (pr.margin >= 0 ? g.home : g.away).name + ' by ' + trim(Math.abs(pr.margin));
      const alt = (pr.margin_without_last2 >= 0 ? g.home : g.away).name + ' by ' + trim(Math.abs(pr.margin_without_last2));
      facts.push('Without each team’s last 2 games the projection would be ' + alt + ' (now ' + now + ').');
    }
    if (market === 'total' && pr.total_without_last2 != null) {
      facts.push('Without each team’s last 2 games the projected total would be ' + trim(pr.total_without_last2) + ' (now ' + trim(pr.total) + ').');
    }
    body.append(h('ul', null, facts.map((f) => h('li', null, f))));

    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null, ['Side', 'Price', 'Model', 'Market', 'Edge'].map((t) => h('th', null, t)))));
    const tbody = h('tbody');
    for (const key of sideKeys(market)) {
      const sd = block[key];
      if (!sd) continue;
      tbody.append(h('tr', { class: key === p.key ? 'hl' : null },
        h('td', { class: 'wrapcell' }, sideLabel(g, market, key)), h('td', null, oddsFmt(sd.odds)), h('td', null, pct(sd.p_model, 1)),
        h('td', null, sd.p_market == null ? '–' : pct(sd.p_market, 1)), h('td', null, edgeFmt(sd.edge))));
    }
    table.append(tbody);
    body.append(h('div', { class: 'scroll' }, table));
    return h('details', { class: 'mkt' + (side.label === 'PLAY' ? ' is-play' : ''), 'data-market': market }, summary, body);
  }

  function explain(g, market, key, side) {
    const what = sideLabel(g, market, key);
    const verb = market === 'ml' ? 'to win' : market === 'spread' ? 'to cover' : 'to hit';
    if (side.p_market == null || side.edge == null) {
      return h('p', { class: 'why' }, 'No line is available for this market, so only the model’s number is shown: ', b(pct(side.p_model)), ' for ' + what + ' ' + verb + '.');
    }
    if (side.label === 'PLAY') {
      const stake = stakeText(side);
      return h('div', { class: 'why play' },
        h('p', null, 'The model gives ', b(what), ' a ', b(pct(side.p_model)), ' chance ' + verb + '. The market price implies ', b(pct(side.p_market)),
          ' once the bookmaker’s cut is removed. That gap of ', b(edgeFmt(side.edge)), ' is the edge.'),
        h('p', { style: 'margin-top:6px' }, 'The pick holds up without each team’s last two games, and the line has not moved against it.' +
          (stake ? ' Suggested stake: ' + stake + ', a quarter of the Kelly amount and never more than 2%.' : '')));
    }
    if (side.label === 'AVOID') {
      return h('div', { class: 'why avoid' },
        h('p', null, 'On paper there is an edge of ', b(edgeFmt(side.edge)), ' (model ' + pct(side.p_model) + ', market ' + pct(side.p_market) + '), but there is a red flag:'),
        h('ul', { style: 'margin-top:6px' }, side.reasons.map((r) => h('li', null, cap(r)))),
        h('p', { style: 'margin-top:6px' }, 'The model does not see injuries, weather or news. When the market disagrees this much, it usually knows something.'));
    }
    return h('p', { class: 'why' }, 'Not enough edge. The model says ', b(pct(side.p_model)), ' and the market ', b(pct(side.p_market)),
      ', a gap of ' + edgeFmt(side.edge) + '. A PLAY needs at least ' + (state.picks.rules.edge_min * 100).toFixed(0) + '%.');
  }

  // ================================================================== PARLAY
  function parlayMath(legs) {
    const out = { n: legs.length, pModel: null, pMarket: null, decimal: null };
    if (!legs.length) return out;
    out.pModel = legs.reduce((acc, l) => acc * l.side.p_model, 1);
    if (legs.every((l) => l.side.p_market != null)) out.pMarket = legs.reduce((acc, l) => acc * l.side.p_market, 1);
    if (legs.every((l) => l.side.odds != null)) out.decimal = legs.reduce((acc, l) => acc * toDecimal(l.side.odds), 1);
    return out;
  }

  function renderParlay() {
    const out = [];
    for (const saved of state.picks.parlays || []) out.push(savedParlay(saved));

    const builder = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Build a parlay'),
      h('p', { class: 'sub' }, 'Tap a side to add it to your ticket. One leg per game: the math multiplies each leg’s chance, which only works when the legs are independent.'));
    const row = h('div', { class: 'controls-row' });
    if (mineCount()) row.append(seg('Games', [['mine', 'My games'], ['all', 'All games']], useMine() ? 'mine' : 'all', (v, id) => setState('scope', v, id)));
    row.append(seg('Market', [['ml', 'Moneyline'], ['spread', 'Spread'], ['total', 'Total']], state.parlayMarket, (v, id) => { state.parlayMarket = v; refocus = id; render(); }));
    builder.append(row);
    const list = h('div');
    const market = state.parlayMarket;
    for (const g of scopedGames().slice().sort((x, y) => x.date.localeCompare(y.date))) {
      const opts = h('div', { class: 'legopts' });
      for (const key of sideKeys(market)) {
        const leg = legOf(g, market, key);
        if (!leg) continue;
        const id = 'leg-' + g.id + '-' + market + '-' + key;
        opts.append(h('button', { type: 'button', class: 'leg', id, 'aria-pressed': String(isLeg(g.id, market, key)),
          onclick: () => { setLeg(g, market, key); refocus = id; render(); } },
          h('b', null, sideLabel(g, market, key)), h('span', { class: 'o' }, oddsFmt(leg.side.odds)),
          h('span', { class: 'p' }, 'model ' + pct(leg.side.p_model) + (leg.side.p_market != null ? ' · market ' + pct(leg.side.p_market) : ''))));
      }
      if (opts.children.length) list.append(h('div', { class: 'legrow' }, h('span', { class: 'when' }, matchup(g) + ' · ' + kickFmt(g.date)), opts));
    }
    builder.append(list.children.length ? list : h('p', { class: 'empty' }, 'No lines for this market in the selected games.'));
    out.push(builder);
    out.push(slip());
    return out;
  }

  function savedParlay(saved) {
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Saved parlay: ' + saved.name));
    const legs = (saved.legs || []).filter((l) => !l.no_data);
    if (saved.p_model == null) {
      box.append(h('p', { class: 'empty' }, 'None of these legs has a pending game this week.'));
      return box;
    }
    const verdict = h('p', { class: 'verdict' }, 'The model gives this ' + legs.length + '-leg ticket an ', b(pct(saved.p_model, 1)), ' chance, about ', b(oneIn(saved.p_model)), '.');
    if (saved.p_market != null) verdict.append(' The market’s own numbers say ', b(pct(saved.p_market, 1)), ' (' + oneIn(saved.p_market) + ').');
    if (saved.implied_prob != null) {
      verdict.append(' The book pays ', b(oddsFmt(decimalToAmerican(saved.decimal))), ', which only breaks even if it hits ', b(pct(saved.implied_prob, 1)), ' of the time.');
    }
    box.append(verdict);
    const kv = h('div', { class: 'kv' });
    kv.append(h('div', null, h('div', { class: 'k' }, 'Model'), h('div', { class: 'v' }, pct(saved.p_model, 1)), h('div', { class: 's' }, 'chance all ' + legs.length + ' legs win')));
    if (saved.p_market != null) kv.append(h('div', null, h('div', { class: 'k' }, 'Market, no vig'), h('div', { class: 'v' }, pct(saved.p_market, 1)), h('div', { class: 's' }, 'what the prices themselves imply')));
    if (saved.implied_prob != null) {
      kv.append(h('div', null, h('div', { class: 'k' }, 'Needed to break even'), h('div', { class: 'v' }, pct(saved.implied_prob, 1)), h('div', { class: 's' }, 'pays ' + oddsFmt(decimalToAmerican(saved.decimal)) + ' (x' + saved.decimal.toFixed(2) + ')')));
      const ev = saved.ev_market != null ? saved.ev_market : saved.ev_model;
      kv.append(h('div', null, h('div', { class: 'k' }, 'Expected value'), h('div', { class: 'v ' + (ev >= 0 ? 'pos' : 'neg') }, signed(ev * 100, 0, '%')),
        h('div', { class: 's' }, saved.ev_market != null ? 'per $1 by the market’s numbers; ' + signed(saved.ev_model * 100, 0, '%') + ' by the model' : 'per $1, by the model')));
    }
    box.append(kv);
    box.append(h('p', { class: 'small' }, 'Why the gap: every leg is priced with the bookmaker’s cut built in, and in a parlay those cuts multiply.'));

    box.append(h('h3', { class: 'h3' }, 'Legs, weakest first'));
    const byId = new Map(allGames().map((g) => [g.id, g]));
    const grid = h('div', { class: 'legs' });
    for (const leg of legs.slice().sort((x, y) => x.p_model - y.p_model)) {
      const g = byId.get(leg.game_id);
      const weak = leg.team === saved.weakest;
      grid.append(h('div', { class: 'legitem' },
        g ? badge(g[leg.side], true) : h('span'),
        h('div', { class: 'legname' }, h('b', null, leg.team), weak ? h('span', { class: 'flag' }, 'Weakest leg') : null,
          h('small', null, (leg.venue === 'home' ? 'vs ' : 'at ') + leg.opponent)),
        h('div', { class: 'fig' }, pct(leg.p_model), h('small', null, (leg.p_market != null ? 'market ' + pct(leg.p_market) + ' \u00b7 ' : '') + oddsFmt(leg.odds))),
        h('div', { class: 'bar' + (weak ? ' weak' : ''), role: 'img', 'aria-label': 'Model chance ' + pct(leg.p_model) }, h('i', { style: 'width:' + clamp(leg.p_model * 100, 2, 100).toFixed(0) + '%' }))));
    }
    box.append(grid);
    for (const leg of (saved.legs || []).filter((l) => l.no_data)) box.append(h('p', { class: 'small' }, 'No data for ' + leg.team + '.'));
    box.append(h('div', null, h('button', { type: 'button', class: 'btn', onclick: () => {
      const next = {};
      for (const leg of legs) next[leg.game_id] = { market: 'ml', key: leg.side };
      state.legs = next;
      store.set('legs', next);
      state.parlayMarket = 'ml';
      state.slipOpen = true;
      refocus = 'slip';
      render();
    } }, 'Load into the builder')));
    return box;
  }

  function slip() {
    const legs = selectedLegs();
    const math = parlayMath(legs);
    const american = math.decimal != null ? decimalToAmerican(math.decimal) : null;
    const headline = !legs.length ? 'empty' : pct(math.pModel, 1) + ' · ' + (american != null ? 'pays ' + oddsFmt(american) : 'no price');
    const box = h('details', { class: 'slip', id: 'slip', open: state.slipOpen === true, ontoggle: (e) => { state.slipOpen = e.target.open; } });
    box.append(h('summary', null, h('span', { class: 'h3' }, 'Your ticket · ' + plural(legs.length, 'leg')), h('span', { class: 'slip-sum' }, headline)));
    const body = h('div', { class: 'slip-body' });
    box.append(body);
    if (!legs.length) {
      body.append(h('p', { class: 'small' }, 'Tap a side above to add it.'));
      return box;
    }
    const grid = h('div', { class: 'kv' });
    grid.append(h('div', null, h('div', { class: 'k' }, 'Model chance'), h('div', { class: 'v' }, pct(math.pModel, 1)), h('div', { class: 's' }, 'about ' + oneIn(math.pModel))));
    if (math.pMarket != null) grid.append(h('div', null, h('div', { class: 'k' }, 'Market, no vig'), h('div', { class: 'v' }, pct(math.pMarket, 1)), h('div', { class: 's' }, 'about ' + oneIn(math.pMarket))));
    if (math.decimal != null) {
      const ev = math.pModel * math.decimal - 1;
      grid.append(h('div', null, h('div', { class: 'k' }, 'Pays'), h('div', { class: 'v' }, oddsFmt(american)), h('div', { class: 's' }, 'breaks even at ' + pct(1 / math.decimal, 1))));
      grid.append(h('div', null, h('div', { class: 'k' }, 'Expected value'), h('div', { class: 'v ' + (ev >= 0 ? 'pos' : 'neg') }, signed(ev * 100, 0, '%')), h('div', { class: 's' }, 'per $1, by the model')));
    } else {
      grid.append(h('div', null, h('div', { class: 'k' }, 'Pays'), h('div', { class: 'v' }, '–'), h('div', { class: 's' }, 'a leg has no price')));
    }
    body.append(grid);
    if (math.decimal != null) {
      body.append(h('label', { class: 'field-input', for: 'slip-stake' }, 'Bet $',
        h('input', { id: 'slip-stake', type: 'number', min: '0', step: '1', inputmode: 'decimal', value: String(state.stake),
          onchange: (e) => setState('stake', Math.max(0, Number(e.target.value) || 0), 'slip-stake') }),
        h('span', null, 'returns ' + money(state.stake * math.decimal) + ' if every leg wins')));
    }
    const weakest = legs.reduce((x, y) => (x.side.p_model <= y.side.p_model ? x : y));
    body.append(h('p', { class: 'small' }, 'Weakest leg: ' + weakest.label + ' (' + pct(weakest.side.p_model) + '). Tap a leg to remove it.'));
    body.append(h('div', { class: 'slip-legs' }, legs.map((l) => h('button', { type: 'button', 'aria-label': 'Remove ' + l.label,
      onclick: () => { setLeg(l.g, l.market, l.key); refocus = 'slip'; render(); } }, l.label + ' ' + oddsFmt(l.side.odds) + ' ×'))));
    body.append(h('div', null, h('button', { type: 'button', class: 'btn ghost', onclick: () => { state.legs = {}; store.set('legs', {}); refocus = 'slip'; render(); } }, 'Clear ticket')));
    return box;
  }

  // ============================================================ TRACK RECORD
  function renderRecord() {
    const B = state.backtest, out = [];
    const sel = B.comparison.find((c) => c.selected) || B.comparison[0];
    const cur = sel.current, mkt = B.market;
    if (!cur || !cur.n) {
      return [h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Track record'),
        h('p', { class: 'empty' }, 'No ' + B.season + ' games have been played yet. The track record appears after the first week.'))];
    }
    const kpi = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'How the model has done in ' + B.season),
      h('p', { class: 'sub' }, 'Every week was predicted using only the games played before it, across ' + cur.n + ' games between top-division teams. Nothing here was fitted after the fact.'));
    const kv = h('div', { class: 'kv' });
    kv.append(h('div', null, h('div', { class: 'k' }, 'Picked the winner'), h('div', { class: 'v' }, rate(cur.accuracy)), h('div', { class: 's' }, mkt ? 'the market: ' + rate(mkt.accuracy) : '')));
    kv.append(h('div', null, h('div', { class: 'k' }, 'When favored by 14+'), h('div', { class: 'v' }, rate(cur.accuracy_14)), h('div', { class: 's' }, plural(cur.n_14, 'game'))));
    kv.append(h('div', null, h('div', { class: 'k' }, 'Typical miss, margin'), h('div', { class: 'v' }, cur.mae_margin.toFixed(1) + ' pts'), h('div', { class: 's' }, mkt ? 'the market: ' + mkt.mae_margin.toFixed(1) : '')));
    kv.append(h('div', null, h('div', { class: 'k' }, 'Typical miss, total'), h('div', { class: 'v' }, cur.mae_total.toFixed(1) + ' pts'), h('div', { class: 's' }, mkt ? 'the market: ' + mkt.mae_total.toFixed(1) : '')));
    kpi.append(kv);
    out.push(kpi);
    out.push(bettingPanel(B));
    out.push(comparisonPanel(B, sel));
    out.push(weeksPanel(B));
    out.push(calibrationPanel(B));
    return out;
  }

  function recordTable(block) {
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['Bet', 'W–L', 'Win %', 'ROI', 'Units'].map((t) => h('th', { class: t === 'Win %' ? 'opt' : null }, t)))));
    const body = h('tbody');
    for (const market of ['ml', 'spread', 'total', 'all']) {
      for (const label of ['PLAY', 'AVOID']) {
        const r = block[market] && block[market][label];
        if (!r || !r.n) continue;
        body.append(h('tr', { class: market === 'all' ? 'hl' : null },
          h('td', null, (market === 'all' ? 'All' : MARKET_SHORT[market]) + ' · ' + label.toLowerCase()),
          h('td', null, r.won + '–' + r.lost), h('td', { class: 'opt' }, rate(r.win_pct, 0)),
          h('td', { class: r.roi >= 0 ? 'pos' : 'neg' }, signed(r.roi * 100, 1, '%'), h('span', { class: 'pm' }, ' ±' + trim(r.roi_error * 100, 0))),
          h('td', { class: r.units >= 0 ? 'pos' : 'neg' }, signed(r.units, 0, 'u'))));
      }
    }
    table.append(body);
    return table;
  }

  function bettingPanel(B) {
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Would the plays have made money?'));
    const cur = (B.bets.current || {}).all, hist = (B.bets.history || {}).all;
    if (!cur || !cur.PLAY) {
      box.append(h('p', { class: 'empty' }, 'There are no closing lines yet to simulate against.'));
      return box;
    }
    const j = cur.PLAY, hp = hist && hist.PLAY;
    const verdict = h('p', { class: 'verdict' }, b(j.roi >= 0 ? 'So far, yes.' : 'Not so far.'), ' Betting one unit on every PLAY in ' + B.season + ' at the closing line went ' + j.won + '–' + j.lost +
      ' for ', b(signed(j.units, 1, ' units')), ' (ROI ' + signed(j.roi * 100, 1, '%') + ').');
    if (hp && hp.n) {
      verdict.append(' Across every season with lines (' + B.bets.seasons_with_lines.join(', ') + ') the ROI is ', b(signed(hp.roi * 100, 1, '%')), ' on ' + hp.n + ' bets' +
        (hp.roi < 0 && hp.roi > -0.08 ? ', roughly the bookmaker’s cut.' : '.'));
    }
    box.append(verdict);
    const sp = B.bets.history && B.bets.history.spread;
    if (sp && sp.AVOID.n && sp.PLAY.n && sp.AVOID.roi < sp.PLAY.roi) {
      box.append(h('p', { class: 'sub' }, 'The AVOID flag earns its keep: spread picks it flagged returned ' + signed(sp.AVOID.roi * 100, 1, '%') + ', against ' + signed(sp.PLAY.roi * 100, 1, '%') + ' for the plays.'));
    }
    const chart = h('div', { class: 'chart', id: 'chart-units' });
    box.append(h('h3', { class: 'h3' }, 'Running total in units, ' + B.season), chart);
    const weeks = B.weeks.filter((w) => w.plays > 0);
    queueChart(() => drawUnits(chart, weeks));
    const two = h('div', { class: 'two' });
    two.append(h('div', null, h('h3', { class: 'h3' }, B.season + ' by bet type'), h('div', { class: 'scroll' }, recordTable(B.bets.current))));
    two.append(h('div', null, h('h3', { class: 'h3' }, 'Seasons ' + B.bets.seasons_with_lines.join(', ')), h('div', { class: 'scroll' }, recordTable(B.bets.history))));
    box.append(two);
    box.append(h('p', { class: 'small' }, 'ROI is profit per unit staked; ± is one standard error. The closing line is the last price before kickoff, the hardest one to beat.'));
    return box;
  }

  function comparisonPanel(B, sel) {
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Models compared'),
      h('p', { class: 'sub' }, 'Average miss on the final margin in ' + B.season + ', in points. Lower is better. The lowest-error model is the one in use; the market’s closing line is there for reference.'));
    const rows = B.comparison.map((c) => ({ name: c.label.replace(/\s*\(.*\)$/, ''), value: c.current.mae_margin, hl: c.selected }));
    if (B.market) rows.push({ name: 'Market closing line', value: B.market.mae_margin, hl: false });
    rows.sort((x, y) => y.value - x.value);
    const max = Math.max(...rows.map((r) => r.value));
    const bars = h('div', { class: 'bars' });
    for (const r of rows) {
      bars.append(h('span', { class: 'name' + (r.hl ? ' hl' : '') }, r.name + (r.hl ? ' (in use)' : '')),
        h('span', null, h('span', { class: 'bar' + (r.hl ? ' hl' : ''), style: 'width:' + ((r.value / max) * 100).toFixed(1) + '%' })),
        h('span', { class: 'num' }, r.value.toFixed(2)));
    }
    box.append(bars);
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['Model', 'Winner', '14+ pts', 'Margin miss', 'Total miss', B.tuned_on + ' miss'].map((t) => h('th', null, t)))));
    const body = h('tbody');
    for (const c of B.comparison) {
      const a = c.current, hist = c.history || {};
      body.append(h('tr', { class: c.selected ? 'hl' : null },
        h('td', { class: 'wrapcell' }, c.label), h('td', null, rate(a.accuracy)), h('td', null, rate(a.accuracy_14)), h('td', null, a.mae_margin.toFixed(2)),
        h('td', null, a.mae_total == null ? '–' : a.mae_total.toFixed(2)), h('td', null, hist.mae_margin == null ? '–' : hist.mae_margin.toFixed(2))));
    }
    const ref = B.prototype_reference;
    body.append(h('tr', null, h('td', { class: 'wrapcell' }, 'Original prototype (reference)'), h('td', null, rate(ref.accuracy, 0)), h('td', null, rate(ref.accuracy_14, 0)),
      h('td', null, '~' + ref.mae_margin), h('td', null, '~' + ref.mae_total), h('td', null, '–')));
    table.append(body);
    box.append(h('details', null, h('summary', { class: 'small' }, 'See the full table'), h('div', { class: 'scroll', style: 'margin-top:8px' }, table),
      h('p', { class: 'small', style: 'margin-top:8px' }, 'Counting games against lower-division (FCS) teams, as the prototype did, the model in use picks ' + rate(sel.current_with_fcs.accuracy) +
        ' of winners and ' + rate(sel.current_with_fcs.accuracy_14) + ' when favored by 14 or more.')));
    return box;
  }

  function weeksPanel(B) {
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Week by week'),
      h('p', { class: 'sub' }, 'Margin and Total are the average miss in points. Plays is the record of that week’s PLAY picks against the closing line.'));
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['Week', 'Winners', 'Margin', 'Total', 'Plays', 'Units'].map((t) => h('th', { class: t === 'Total' ? 'opt' : null }, t)))));
    const body = h('tbody');
    for (const w of B.weeks) {
      body.append(h('tr', null, h('td', null, String(w.week)),
        h('td', null, h('span', { class: 'meter', 'aria-hidden': 'true' }, h('i', { style: 'width:' + (w.accuracy * 100).toFixed(0) + '%' })), rate(w.accuracy, 0), h('span', { class: 'pm opt' }, ' of ' + w.n)),
        h('td', null, w.mae_margin.toFixed(1)), h('td', { class: 'opt' }, w.mae_total == null ? '–' : w.mae_total.toFixed(1)),
        h('td', null, w.plays ? w.won + '–' + w.lost : '–'),
        h('td', { class: w.plays ? (w.units >= 0 ? 'pos' : 'neg') : null }, w.plays ? signed(w.units, 1, 'u') : '–')));
    }
    table.append(body);
    box.append(h('div', { class: 'scroll' }, table));
    box.append(h('p', { class: 'small' }, 'The first weeks have no plays: until every team has played ' + B.rules.min_games + ' games, its rating is too shaky to bet on.'));
    return box;
  }

  function calibrationPanel(B) {
    const cal = B.calibration;
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Can you trust the percentages?'),
      h('p', { class: 'sub' }, 'When the model gives its favorite a 70% chance, that team should win about 70% of the time. Dots on the diagonal mean the percentages are honest. In past seasons they were off by ' +
        (cal.mean_error == null ? '–' : trim(cal.mean_error * 100, 1) + ' points') + ' on average.'));
    const hist = B.history_seasons || [];
    const histName = hist.length ? hist[0] + '–' + hist[hist.length - 1] : 'Past seasons';
    box.append(h('div', { class: 'legend' },
      h('span', null, h('i', { class: 'dot', style: 'background:var(--series-1)' }), String(B.season)),
      h('span', null, h('i', { class: 'dot', style: 'background:var(--series-2)' }), histName),
      h('span', null, h('i', { style: 'background:var(--track);height:1px' }), 'Perfect calibration')));
    const chart = h('div', { class: 'chart', id: 'chart-cal' });
    box.append(chart);
    queueChart(() => drawCalibration(chart, cal.current, cal.history, String(B.season), histName));
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['Model said', B.season + ' won', 'Games', histName + ' won', 'Games'].map((t) => h('th', null, t)))));
    const body = h('tbody');
    const histBy = new Map(cal.history.map((x) => [x.from, x]));
    const curBy = new Map(cal.current.map((x) => [x.from, x]));
    const bins = Array.from(new Set([...curBy.keys(), ...histBy.keys()])).sort((x, y) => x - y);
    for (const lo of bins) {
      const c = curBy.get(lo), hb = histBy.get(lo);
      body.append(h('tr', null, h('td', null, Math.round(lo * 100) + '–' + Math.round((lo + 0.05) * 100) + '%'),
        h('td', null, c ? rate(c.observed, 0) : '–'), h('td', null, c ? String(c.n) : '–'),
        h('td', null, hb ? rate(hb.observed, 0) : '–'), h('td', null, hb ? String(hb.n) : '–')));
    }
    table.append(body);
    box.append(h('details', null, h('summary', { class: 'small' }, 'See it as a table'), h('div', { class: 'scroll', style: 'margin-top:8px' }, table)));
    return box;
  }

  // ------------------------------------------------------------------ charts
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
    if (!weeks.length) { el.append(h('p', { class: 'empty' }, 'No plays yet.')); return; }
    const W = Math.max(el.clientWidth, 260), H = 210, m = { l: 36, r: 54, t: 12, b: 30 };
    const values = weeks.map((w) => w.cumulative);
    const lo = Math.min(0, ...values), hi = Math.max(0, ...values);
    const step = niceStep((hi - lo) || 1, 4);
    const y0 = Math.floor(lo / step) * step, y1 = Math.ceil(hi / step) * step || step;
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const x = (i) => m.l + (weeks.length === 1 ? iw / 2 : (i / (weeks.length - 1)) * iw);
    const y = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * ih;
    const svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'aria-label': 'Running total of units won or lost on plays, by week' });
    for (let v = y0; v <= y1 + 1e-9; v += step) {
      svg.append(s('line', { class: Math.abs(v) < 1e-9 ? 'axis' : 'grid', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }),
        s('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, signed(v, 0)));
    }
    const every = Math.ceil(weeks.length / Math.max(Math.floor(iw / 40), 1));
    weeks.forEach((w, i) => {
      if (i % every === 0 || i === weeks.length - 1) svg.append(s('text', { x: x(i), y: H - 10, 'text-anchor': 'middle' }, 'Wk ' + w.week));
    });
    const pts = weeks.map((w, i) => [x(i), y(w.cumulative)]);
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
        'aria-label': 'Week ' + w.week + ': running total ' + signed(w.cumulative, 1) + ' units' });
      const on = (evt) => {
        cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
        dot.setAttribute('cx', x(i)); dot.setAttribute('cy', y(w.cumulative)); dot.setAttribute('visibility', 'visible');
        const box = svg.getBoundingClientRect();
        showTip(hit, [h('b', null, signed(w.cumulative, 1, 'u') + ' so far'), h('span', null, 'Week ' + w.week + ': ' + w.won + '–' + w.lost + ', ' + signed(w.units, 1, 'u'))],
          { x: box.left + (x(i) / W) * box.width, y: evt && evt.clientY ? evt.clientY : box.top + (y(w.cumulative) / H) * box.height });
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
    const observed = [...cur, ...hist].map((x) => x.observed);
    const yMin = Math.min(0.5, Math.floor(Math.min(...observed, 0.5) * 10) / 10);
    const x = (p) => m.l + ((p - 0.5) / 0.5) * iw;
    const y = (p) => m.t + (1 - (p - yMin) / (1 - yMin)) * ih;
    const svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'aria-label': 'Calibration: the chance the model gave its favorite against how often that team won' });
    for (let v = yMin; v <= 1 + 1e-9; v += 0.1) {
      svg.append(s('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }), s('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, Math.round(v * 100) + '%'));
    }
    for (let v = 0.5; v <= 1 + 1e-9; v += 0.1) svg.append(s('text', { x: x(v), y: H - m.b + 16, 'text-anchor': 'middle' }, Math.round(v * 100) + '%'));
    svg.append(s('text', { x: m.l + iw / 2, y: H - 6, 'text-anchor': 'middle' }, 'Chance the model gave its favorite'));
    svg.append(s('text', { x: 12, y: m.t + ih / 2, 'text-anchor': 'middle', transform: 'rotate(-90 12 ' + (m.t + ih / 2) + ')' }, 'How often it won'));
    svg.append(s('line', { class: 'axis', x1: x(0.5), y1: y(0.5), x2: x(1), y2: y(1) }));
    const series = [[hist, 'var(--series-2)', histName], [cur, 'var(--series-1)', curName]];
    for (const [bins, color] of series) {
      if (!bins.length) continue;
      const d = bins.map((bin, i) => (i ? 'L' : 'M') + x(bin.predicted).toFixed(1) + ' ' + y(bin.observed).toFixed(1)).join(' ');
      svg.append(s('path', { d, fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
    }
    for (const [bins, color, name] of series) {
      for (const bin of bins) {
        const cx = x(bin.predicted), cy = y(bin.observed);
        svg.append(s('circle', { cx, cy, r: 4.5, fill: color, stroke: 'var(--surface)', 'stroke-width': 2 }));
        const hit = s('circle', { cx, cy, r: 13, fill: 'transparent', tabindex: 0,
          'aria-label': name + ': the model said ' + rate(bin.predicted, 0) + ' and the favorite won ' + rate(bin.observed, 0) + ' of ' + bin.n + ' games' });
        const on = () => showTip(hit, [h('b', null, 'Won ' + rate(bin.observed, 0)), h('span', null, name + ' · model said ' + rate(bin.predicted, 0) + ' · ' + bin.n + ' games (±' + trim(bin.std_error * 100, 0) + ')')]);
        hit.addEventListener('pointerenter', on);
        hit.addEventListener('focus', on);
        hit.addEventListener('pointerleave', hideTip);
        hit.addEventListener('blur', hideTip);
        svg.append(hit);
      }
    }
    el.append(svg);
  }

  // ============================================================ HOW IT WORKS
  function renderHow() {
    const P = state.picks, B = state.backtest, r = P.rules, m = P.model;
    const sec = (title, ...kids) => h('section', { class: 'panel' }, h('h2', { class: 'h2' }, title), kids);
    const p = (...kids) => h('p', null, kids);
    const key = h('div', { class: 'key' },
      pill('PLAY'), h('div', null, 'The model’s chance beats the market’s by ' + (r.edge_min * 100).toFixed(0) + '% or more, the bet has positive expected value at the real price, and there are no red flags.'),
      pill('AVOID'), h('div', null, 'There is an edge on paper, but the pick flips when each team’s last 2 games are removed, the model is ' + trim(r.gap_margin) + '+ points away from the market spread (' +
        trim(r.gap_total) + '+ on a total), or the line moved ' + trim(r.move_spread) + '+ points against it since it opened (' + trim(r.move_total) + '+ on a total).'),
      pill('PASS'), h('div', null, 'The model and the market roughly agree. Nothing to do.'));
    const gloss = h('dl', { class: 'gloss' });
    const terms = [
      ['Moneyline', 'A bet on who wins. −300 means you risk $300 to win $100; +240 means you risk $100 to win $240.'],
      ['Spread', 'A bet on the margin. Indiana −7.5 wins if Indiana wins by 8 or more; the other side wins otherwise.'],
      ['Total', 'A bet on the combined score going over or under a number.'],
      ['Vig', 'The bookmaker’s cut, built into the prices. At −110 on both sides you need to win 52.4% of the time just to break even.'],
      ['Market chance', 'The win chance the prices imply once the vig is removed. It is the fairest read of what the market believes.'],
      ['Edge', 'The model’s chance minus the market chance. A positive edge means the model thinks the price is too generous.'],
      ['Stake', 'A quarter of the Kelly amount, which sizes a bet by its edge, and never more than ' + (r.stake_cap * 100).toFixed(0) + '% of your bankroll.'],
      ['Closing line', 'The last price before kickoff. It reflects everything known about the game, which makes it very hard to beat.'],
    ];
    for (const [term, text] of terms) gloss.append(h('div', null, h('dt', null, term), h('dd', null, text)));
    return [
      sec('The model in plain words', h('div', { class: 'prose' },
        p('Each team gets a rating: its average scoring margin, adjusted for how good its opponents were. A team’s rating depends on its opponents’ ratings and theirs on it, so all of them are solved together from the ' +
          m.training_games + ' games played this season.'),
        p('A projection is the difference between two ratings plus home-field advantage. A second set of ratings, built the same way on combined scores, gives the projected total and from it the score.'),
        p('Early in the season a rating leans on last year’s, and blowouts are capped so one 63–0 does not count for too much.'))),
      sec('From a projection to a percentage', h('div', { class: 'prose' },
        p('In testing, real margins landed about ' + trim(m.sigma_margin) + ' points away from the projection on a typical day (' + trim(m.sigma_total) + ' for totals). That spread turns a projected margin into a chance of winning, covering a spread or going over a total.'),
        p('No percentage is ever shown as 100%, and the typical miss sits next to every projection for the same reason.'))),
      sec('What the labels mean', key),
      sec('What the model does not know', h('div', { class: 'prose' },
        h('ul', null,
          h('li', null, 'Injuries, suspensions and quarterback changes.'),
          h('li', null, 'Weather. A total that drops 10 points during the week is almost always wind or rain.'),
          h('li', null, 'Motivation, rest, travel and look-ahead spots.'),
          h('li', null, 'Anything the market has already priced in, which is why the closing line misses by less than the model does.')),
        p('Notes and manual point adjustments can be added per game in ', h('code', null, 'data/notes.csv'), '.'))),
      sec('How the track record is built', h('div', { class: 'prose' },
        p('For week N the model only sees games played before week N. Its settings were tuned on ' + (B.tuned_on || 'past seasons') + ', and ' + B.season +
          ' was scored without touching them. The betting record stakes one unit on every PLAY at the closing line.'))),
      sec('Glossary', gloss),
    ];
  }

  // -------------------------------------------------------------------- boot
  const VIEWS = { picks: renderPicks, parlay: renderParlay, record: renderRecord, how: renderHow };
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
      view.replaceChildren(h('section', { class: 'callout' }, h('b', { class: 't' }, 'The picks could not be loaded'),
        h('p', null, state.error), h('p', null, 'Generate the data with python -m src.picks and open the site through a web server (python -m http.server -d site), not as a local file.')));
      return;
    }
    if (!state.picks) return;
    pendingCharts = [];
    view.replaceChildren(...VIEWS[name]());
    flushCharts();
    updateParlayCount();
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
        if (!resp.ok) throw new Error(url + ' answered HTTP ' + resp.status);
        return resp.json();
      }));
      state.picks = picks;
      state.backtest = backtest;
      const weekKey = picks.season + '-' + picks.week;
      if (store.get('legsWeek', null) !== weekKey) {   // last week's ticket does not carry over
        state.legs = {};
        store.set('legs', {});
        store.set('legsWeek', weekKey);
      }
      document.getElementById('hdr-week').textContent = picks.week_label + ' · ' + picks.season;
      const n = allGames().length, mine = mineCount();
      document.getElementById('hdr-meta').textContent = 'Updated ' + stamp(picks.generated) + ' · ' + plural(n, 'game') + (mine ? ', ' + mine + ' tracked' : '');
      document.getElementById('foot-gen').textContent = 'Scores and lines: ESPN’s public API. Model: ' + picks.model.label + '.';
    } catch (err) {
      state.error = String((err && err.message) || err);
      document.getElementById('hdr-meta').textContent = 'No data';
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
