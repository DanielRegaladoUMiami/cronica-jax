/* Picks site for two sports. The URL picks the sport and the section
   (#cfb/picks, #soccer/record ...) and the page reads that sport's JSON from data/.
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

  let dfDay = null, dfTime = null, dfStamp = null, dfShort = null;
  try {
    dfDay = new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    dfTime = new Intl.DateTimeFormat('en-US', { hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
    dfStamp = new Intl.DateTimeFormat('en-US', { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' });
    dfShort = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' });
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

  // ------------------------------------------------------------------ sports
  const SPORTS = {
    cfb: {
      title: 'CFB Picks', name: 'College football', game: 'game', games: 'games',
      files: ['data/picks.json', 'data/backtest.json'], command: 'python -m src.picks',
      market: { ml: 'Moneyline', spread: 'Spread', total: 'Total' },
      short: { ml: 'ML', spread: 'Spread', total: 'Total' },
    },
    soccer: {
      title: 'Soccer Picks', name: 'Soccer', game: 'match', games: 'matches',
      files: ['data/soccer_picks.json', 'data/soccer_backtest.json'], command: 'python -m src.soccer.picks',
      market: { ml: 'Match result', spread: 'Handicap', total: 'Total goals' },
      short: { ml: 'Result', spread: 'Handicap', total: 'Total' },
    },
  };
  const DATA = {};   // sport -> { picks, backtest } or { error }

  // ------------------------------------------------------------------- state
  const state = {
    sport: 'cfb',
    picks: null,
    backtest: null,
    error: null,
    scope: store.get('scope', 'mine'),
    league: store.get('league', 'all'),
    market: store.get('market', 'all'),
    playsOnly: store.get('playsOnly', false),
    bankroll: store.get('bankroll', 1000),
    stake: store.get('stake', 10),
    legs: {},
    query: '',
    showAllPlays: false,
    parlayMarket: 'ml',
    slipOpen: false,
  };
  const view = document.getElementById('view');
  const tip = document.getElementById('tooltip');
  const reduceMotion = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  let refocus = null;

  const S = () => SPORTS[state.sport];
  const isSoccer = () => state.sport === 'soccer';
  const legsKey = () => (state.sport === 'cfb' ? 'legs' : 'legs:' + state.sport);
  const seasonName = (obj) => String(obj.season_label || obj.season);
  const soccerSeason = (year) => year + '-' + String((year + 1) % 100).padStart(2, '0');
  function seasonList(years) {   // "2022, 2023" for football, "2022-23 to 2026-27" for soccer
    if (!isSoccer()) return years.join(', ');
    return years.length > 1 ? soccerSeason(years[0]) + ' to ' + soccerSeason(years[years.length - 1]) : years.map(soccerSeason).join('');
  }
  const gamesWord = (n) => plural(n, S().game, S().games);

  function setState(key, value, focusId) {
    state[key] = value;
    if (key !== 'showAllPlays') store.set(key, value);
    refocus = focusId || null;
    render();
  }

  // ------------------------------------------------------------- data access
  const allGames = () => state.picks.games.filter((g) => !g.no_data);
  const mineCount = () => allGames().filter((g) => g.on_my_list).length;
  const useMine = () => !isSoccer() && state.scope === 'mine' && mineCount() > 0;
  // A remembered league only counts while it has matches on the board; otherwise show them all.
  const leaguesOnBoard = () => (state.picks.leagues || []).filter((lg) => lg.games > 0);
  const currentLeague = () => (leaguesOnBoard().some((lg) => lg.code === state.league) ? state.league : 'all');
  const inScope = (g) => (isSoccer() ? currentLeague() === 'all' || g.league === currentLeague() : !useMine() || g.on_my_list);
  const scopedGames = () => allGames().filter(inScope);
  const visibleMarkets = () => (state.market === 'all' ? ['ml', 'spread', 'total'] : [state.market]);
  const mlKeys = () => (isSoccer() ? ['home', 'draw', 'away'] : ['away', 'home']);
  const sideKeys = (market) => (market === 'total' ? ['over', 'under'] : market === 'ml' ? mlKeys() : isSoccer() ? ['home', 'away'] : ['away', 'home']);
  // Soccer lists the home team first; American football writes "away @ home".
  const matchup = (g) => (isSoccer() ? g.home.abbr + ' v ' + g.away.abbr : g.away.abbr + ' @ ' + g.home.abbr);
  const fixture = (g) => (isSoccer() ? g.home.name + ' v ' + g.away.name : g.away.name + ' @ ' + g.home.name);
  const lineFmt = (x) => (x === 0 ? (isSoccer() ? '0' : 'PK') : (x > 0 ? '+' : MINUS) + trim(Math.abs(x)));

  function sideLabel(g, market, key, withMl) {
    if (market === 'total') return (key === 'over' ? 'Over ' : 'Under ') + trim(g.total.line);
    if (key === 'draw') return 'Draw';
    const team = g[key].name;
    if (market === 'ml') return team + (withMl ? (isSoccer() ? ' to win' : ' ML') : '');
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
      const id = 'seg-' + name.replace(/\W+/g, '') + '-' + String(val).replace(/\W+/g, '');
      box.append(h('button', { type: 'button', id, 'aria-pressed': String(val === value), onclick: () => onPick(val, id) }, text));
    }
    return box;
  }
  function leagueSeg() {
    const options = [['all', 'All leagues']].concat(leaguesOnBoard().map((lg) => [lg.code, lg.name]));
    return seg('League', options, currentLeague(), (v, id) => setState('league', v, id));
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
    if (!side) return null;
    // A total or a draw says nothing about which game it is, so the ticket adds the matchup.
    const bare = market === 'total' || key === 'draw';
    return { g, market, key, side, label: sideLabel(g, market, key, true) + (bare ? ' (' + matchup(g) + ')' : '') };
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
  function saveLegs(legs) {
    state.legs = legs;
    store.set(legsKey(), legs);
  }
  function setLeg(g, market, key) {
    const legs = Object.assign({}, state.legs);
    if (isLeg(g.id, market, key)) delete legs[g.id];
    else legs[g.id] = { market, key };   // one leg per game: two legs from the same game are not independent
    saveLegs(legs);
    updateParlayCount();
  }

  // =================================================================== PICKS
  function renderPicks() {
    const P = state.picks;
    const out = [realityCheck()];
    if (P.warnings && P.warnings.length) {
      out.push(h('section', { class: 'callout' }, h('b', { class: 't' }, isSoccer() ? 'Heads up' : 'Data warnings'),
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
    const P = state.picks, m = P.model, soccer = isSoccer();
    const play = P.evidence && P.evidence.all && P.evidence.all.PLAY;
    const box = h('section', { class: 'callout', 'aria-label': 'Reality check' }, h('b', { class: 't' }, 'Reality check'));
    if (!m.n_backtest) {
      box.append(h('p', null, 'No ' + seasonName(P) + ' ' + S().games + ' have been played yet, so these projections lean on last season’s ratings and are less reliable.'));
      return box;
    }
    const line = soccer
      ? h('p', null, 'The model has called ', b(rate(m.accuracy, 0) + ' of results'), ' right this season (home win, draw or away win)')
      : h('p', null, 'The model has picked ', b(rate(m.accuracy, 0) + ' of winners'), ' this season');
    if (play && play.n) {
      line.append(', but it has ', b('not beaten the betting market'), '. Past plays like the ones below went ' + play.won + '–' + play.lost +
        ' against closing ' + (soccer ? 'odds' : 'lines') + ' (ROI ' + signed(play.roi * 100, 1, '%') + '). Read a PLAY as “my number disagrees with the market,” not as a sure thing. ');
    } else {
      line.append('. ');
    }
    line.append(h('a', { href: '#' + state.sport + '/record' }, 'See the track record'));
    box.append(line);
    return box;
  }

  function topPlays() {
    const soccer = isSoccer();
    const games = scopedGames();
    const picks = collectPicks(games);
    const plays = picks.filter((p) => p.side.label === 'PLAY');
    const avoid = picks.filter((p) => p.side.label === 'AVOID');
    const box = h('section', { class: 'panel' });
    box.append(h('div', { class: 'panel-head' }, h('h2', { class: 'h2' }, soccer ? 'Best plays, next 7 days' : 'Best plays this week'),
      h('label', { class: 'field-input', for: 'bankroll' }, 'Bankroll $',
        h('input', { id: 'bankroll', type: 'number', min: '0', step: '50', inputmode: 'decimal', value: String(state.bankroll),
          onchange: (e) => setState('bankroll', Math.max(0, Number(e.target.value) || 0), 'bankroll') }))));
    const where = useMine() ? 'your ' + plural(games.length, 'tracked game') : gamesWord(games.length);
    if (!plays.length) {
      box.append(h('p', { class: 'empty' }, 'No play clears the ' + (state.picks.rules.edge_min * 100).toFixed(0) + '% edge bar in ' + where + '.'));
    } else {
      box.append(h('p', { class: 'sub' }, plural(plays.length, 'play') + ' in ' + where + ', ranked by edge: the model’s chance minus the market’s. Tap one to see the ' + S().game + '.'));
      const shown = state.showAllPlays ? plays : plays.slice(0, 5);
      box.append(h('div', { class: 'plays' }, shown.map((p, i) => playRow(p, i + 1))));
      if (plays.length > 5) {
        box.append(h('button', { type: 'button', class: 'btn more', id: 'more-plays', onclick: () => setState('showAllPlays', !state.showAllPlays, 'more-plays') },
          state.showAllPlays ? 'Show top 5 only' : 'Show all ' + plays.length + ' plays'));
      }
      const exposure = plays.reduce((acc, p) => acc + p.side.stake, 0);
      if (plays.length > 3) {
        box.append(h('p', { class: 'small' }, 'Betting every play at the suggested stake would put ' + (exposure * 100).toFixed(0) + '% of your bankroll' +
          (state.bankroll > 0 ? ' (' + money(exposure * state.bankroll) + ')' : '') + ' at risk in one week. ' +
          (soccer ? 'A result and a handicap on the same team are nearly the same bet, so pick one.' : 'A moneyline and a spread on the same team are nearly the same bet, so pick one.')));
      }
    }
    if (avoid.length) {
      box.append(h('details', null,
        h('summary', { class: 'h3' }, plural(avoid.length, 'pick') + ' to avoid'),
        h('p', { class: 'small', style: 'margin-top:8px' }, soccer
          ? 'These have an edge on paper, but a red flag is up: the edge is too big to be real, it is a result bet on an outsider, or the pick is unstable. Picks like these did worst in the backtest.'
          : 'These have an edge on paper, but the pick is unstable or the line moved against it. Picks like these did worst in the backtest.'),
        h('div', { class: 'plays' }, avoid.map((p) => playRow(p, null)))));
    }
    return box;
  }

  function playRow(p, rank) {
    const { g, market, key, side } = p;
    const stake = stakeText(side);
    let ctx = S().market[market] + ' · ' + matchup(g) + (g.league_name ? ' · ' + g.league_name : '') + ' · ' + kickFmt(g.date);
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

  function betTypeSeg(name, withAll, value, onPick) {
    const names = S().short;
    const options = [['ml', isSoccer() ? names.ml : 'Moneyline'], ['spread', names.spread], ['total', names.total]];
    return seg(name, withAll ? [['all', 'All bets']].concat(options) : options, value, onPick);
  }

  function controls() {
    const box = h('section', { class: 'controls', 'aria-label': 'Filter ' + S().games });
    const mine = mineCount(), total = allGames().length;
    box.append(h('div', { class: 'controls-row' },
      h('div', { class: 'search' },
        h('input', { id: 'q', type: 'search', placeholder: 'Find a team', autocomplete: 'off', value: state.query, 'aria-label': 'Find a team',
          oninput: (e) => { state.query = e.target.value; fillGames(document.getElementById('games-list')); } }))));
    if (isSoccer()) box.append(h('div', { class: 'controls-row' }, leagueSeg()));
    const row = h('div', { class: 'controls-row' });
    if (!isSoccer() && mine) {
      row.append(seg('Games', [['mine', 'My games (' + mine + ')'], ['all', 'All games (' + total + ')']], useMine() ? 'mine' : 'all', (v, id) => setState('scope', v, id)));
    }
    row.append(betTypeSeg('Bet type', true, state.market, (v, id) => setState('market', v, id)));
    row.append(h('label', { class: 'toggle', for: 'plays-only' },
      h('input', { id: 'plays-only', type: 'checkbox', checked: state.playsOnly, onchange: (e) => setState('playsOnly', e.target.checked, 'plays-only') }), 'Plays only'));
    box.append(row);
    return box;
  }

  function filteredGames() {
    const q = state.query.trim().toLowerCase();
    let list = q ? allGames() : scopedGames();   // a search looks through every game
    if (q) list = list.filter((g) => [g.home.name, g.away.name, g.home.abbr, g.away.abbr, g.home.full_name, g.away.full_name].some((x) => x && String(x).toLowerCase().includes(q)));
    if (state.playsOnly) list = list.filter((g) => visibleMarkets().some((m) => { const p = pickOf(g, m); return p && p.side.label === 'PLAY'; }));
    return list.slice().sort((x, y) => x.date.localeCompare(y.date));
  }

  function fillGames(container) {
    if (!container) return;
    const games = filteredGames();
    const out = [];
    if (!games.length) {
      out.push(h('p', { class: 'empty' }, state.query ? 'No team matches “' + state.query + '”.' : state.playsOnly ? 'No ' + S().game + ' has a PLAY with these filters.'
        : 'No pending ' + S().games + ' with a projection ' + (isSoccer() ? 'in the next 7 days.' : 'this week.')));
    }
    let current = null, grid = null;
    for (const g of games) {
      const key = dayKey(new Date(g.date));
      if (key !== current) {
        current = key;
        const count = games.filter((x) => dayKey(new Date(x.date)) === key).length;
        out.push(h('div', { class: 'day' }, h('h2', { class: 'h2' }, dayLabel(g.date)), h('span', { class: 'small' }, gamesWord(count))));
        grid = h('div', { class: 'cards' });
        out.push(grid);
      }
      grid.append(isSoccer() ? soccerCard(g) : gameCard(g));
    }
    const missing = state.picks.games.filter((g) => g.no_data && inScope(g));
    if (missing.length && !state.query) {
      out.push(h('section', { class: 'panel' }, h('h2', { class: 'h3' }, 'No projection'),
        h('ul', { class: 'alerts' }, missing.map((g) => h('li', { class: 'cmp' }, fixture(g) + ': ' + (g.no_data_reason || 'no data'))))));
    }
    container.replaceChildren(...out);
  }

  // Shared bottom half of a card: the markets, the notes and the parlay chips.
  function cardBody(card, g) {
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
    for (const key of mlKeys()) {
      const side = g.ml[key];
      if (!side) continue;
      const name = key === 'draw' ? 'Draw' : g[key].abbr;
      add.append(h('button', { type: 'button', class: 'addchip', id: 'add-' + g.id + '-' + key, 'aria-pressed': String(isLeg(g.id, 'ml', key)),
        'aria-label': (key === 'draw' ? 'Draw' : g[key].name + (isSoccer() ? ' to win' : ' moneyline')) + ' in the parlay',
        onclick: () => {
          setLeg(g, 'ml', key);
          for (const k of mlKeys()) {
            const chip = document.getElementById('add-' + g.id + '-' + k);
            if (chip) chip.setAttribute('aria-pressed', String(isLeg(g.id, 'ml', k)));
          }
        } }, name + ' ' + oddsFmt(side.odds)));
    }
    if (add.children.length > 1) card.append(add);
    return card;
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
    return cardBody(card, g);
  }

  // Soccer has three results, so the card shows home, draw and away with expected goals.
  function soccerCard(g) {
    const pr = g.projection, model = state.picks.model, mk = g.market, espn = mk.espn;
    const card = h('article', { class: 'panel game', id: 'g-' + g.id });
    card.append(h('div', { class: 'game-top' }, h('span', { class: 'kick' }, kickFmt(g.date)),
      h('div', { class: 'tags' }, h('span', { class: 'tag' }, g.league_name))));

    const chance = { home: pr.p_home, draw: pr.p_draw, away: pr.p_away };
    const teams = h('div', { class: 'teams' });
    for (const key of ['home', 'draw', 'away']) {
      if (key === 'draw') {
        teams.append(h('div', { class: 'team draw' + (pr.winner === 'draw' ? ' fav' : '') }, h('span', { class: 'badge plain', 'aria-hidden': 'true' }, 'X'),
          h('div', { class: 'team-name' }, h('b', null, 'Draw'))), h('div', { class: 'proj' }), h('div', { class: 'wp' }, pct(pr.p_draw)));
        continue;
      }
      const t = g[key];
      const wdl = /^\d+-\d+-\d+$/.test(t.record || '') ? t.record.split('-').map((n, i) => 'WDL'[i] + n).join(' ') : null;
      const meta = [wdl, t.rating_rank ? 'model #' + t.rating_rank : null, key].filter(Boolean).join(' · ');
      teams.append(
        h('div', { class: 'team' + (pr.winner === key ? ' fav' : '') }, badge(t), h('div', { class: 'team-name' }, h('b', null, t.name), h('span', null, meta))),
        h('div', { class: 'proj' }, pr[key] == null ? '–' : pr[key].toFixed(1)),
        h('div', { class: 'wp' }, pct(chance[key])));
    }
    card.append(teams);
    card.append(h('div', { class: 'wpbar', role: 'img',
      'aria-label': 'Chance of each result: ' + g.home.name + ' ' + pct(pr.p_home) + ', draw ' + pct(pr.p_draw) + ', ' + g.away.name + ' ' + pct(pr.p_away) },
      ['home', 'draw', 'away'].map((key) => h('i', { class: [key === 'draw' ? 'draw' : null, pr.winner === key ? 'fav' : null].filter(Boolean).join(' ') || null, style: 'flex:' + Math.max(chance[key], 0.02) }))));
    card.append(h('div', { class: 'cap' }, h('span', null, 'Expected goals and chances'),
      model.mae_margin != null ? h('span', null, 'typical miss ±' + model.mae_margin.toFixed(1) + ' goals') : null));

    const lines = h('dl', { class: 'lines' });
    const score = pr.likely_score || [];
    lines.append(h('dt', null, 'Model'), h('dd', null, (score.length === 2 ? 'Most likely score ' + g.home.abbr + ' ' + score[0] + '–' + score[1] + ' ' + g.away.abbr + ' · ' : '') +
      'total ' + pr.total.toFixed(1) + ' goals'));
    if (mk.p_home != null) {
      let text = g.home.abbr + ' ' + pct(mk.p_home) + ' · Draw ' + pct(mk.p_draw) + ' · ' + g.away.abbr + ' ' + pct(mk.p_away);
      if (espn && espn.total != null) text += ' · goals line ' + trim(espn.total);
      lines.append(h('dt', null, 'Market'), h('dd', null, text));
    }
    if (pr.home_goals_adj || pr.away_goals_adj) {
      lines.append(h('dt', null, 'Manual'), h('dd', null, signed(pr.home_goals_adj, 1) + ' goals for ' + g.home.name + ', ' + signed(pr.away_goals_adj, 1) + ' for ' + g.away.name));
    }
    card.append(lines);
    return cardBody(card, g);
  }

  function marketRow(g, market) {
    const p = pickOf(g, market);
    if (!p) return null;
    const side = p.side, block = g[market], soccer = isSoccer();
    const summary = h('summary', null,
      h('span', { class: 'mkt-type' }, S().market[market]),
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
    if (dv && dv.method === 'three_way') facts.push('Market chance comes from the three ' + sourceName(g, dv.source) + ' prices (home, draw, away), with the bookmaker’s cut removed.');
    if (dv && dv.method === 'two_sided') facts.push('Market chance comes from both sides of the ' + sourceName(g, dv.source) + ' price, with the bookmaker’s cut removed.');
    if (dv && dv.method === 'one_sided') facts.push('Only one side was priced at ' + sourceName(g, dv.source) + ', so a typical 4.5% bookmaker cut was removed.');
    if (dv && dv.method === 'assumed') facts.push('With no price captured, the market chance assumes standard ' + MINUS + '110 on both sides.');
    const pr = g.projection;
    if (soccer) {
      const line = market === 'total' ? block.line : market === 'spread' ? block.home_line : null;
      if (line != null && Number.isInteger(line)) facts.push('On a whole-number line an exact tie is a push and the stake comes back; the chances shown leave pushes out.');
      if (market !== 'total' && pr.p_home_without_last2 != null) {
        facts.push('Without each team’s last 2 matches the model would give ' + g.home.name + ' ' + pct(pr.p_home_without_last2) + ' to win (now ' + pct(pr.p_home) + ').');
      }
      if (market === 'total' && pr.total_without_last2 != null) {
        facts.push('Without each team’s last 2 matches the expected total would be ' + pr.total_without_last2.toFixed(1) + ' goals (now ' + pr.total.toFixed(1) + ').');
      }
    } else {
      if (market !== 'total' && pr.margin_without_last2 != null) {
        const now = (pr.margin >= 0 ? g.home : g.away).name + ' by ' + trim(Math.abs(pr.margin));
        const alt = (pr.margin_without_last2 >= 0 ? g.home : g.away).name + ' by ' + trim(Math.abs(pr.margin_without_last2));
        facts.push('Without each team’s last 2 games the projection would be ' + alt + ' (now ' + now + ').');
      }
      if (market === 'total' && pr.total_without_last2 != null) {
        facts.push('Without each team’s last 2 games the projected total would be ' + trim(pr.total_without_last2) + ' (now ' + trim(pr.total) + ').');
      }
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
    const soccer = isSoccer();
    const what = key === 'draw' ? 'a draw' : sideLabel(g, market, key);
    const verb = key === 'draw' ? '' : market === 'ml' ? ' to win' : market === 'spread' ? ' to cover' : ' to hit';
    if (side.p_market == null || side.edge == null) {
      return h('p', { class: 'why' }, 'No line is available for this market, so only the model’s number is shown: ', b(pct(side.p_model)), ' for ' + what + verb + '.');
    }
    if (side.label === 'PLAY') {
      const stake = stakeText(side);
      return h('div', { class: 'why play' },
        h('p', null, 'The model gives ', b(what), ' a ', b(pct(side.p_model)), ' chance' + verb + '. The market price implies ', b(pct(side.p_market)),
          ' once the bookmaker’s cut is removed. That gap of ', b(edgeFmt(side.edge)), ' is the edge.'),
        h('p', { style: 'margin-top:6px' }, (soccer
          ? 'The pick holds up without each team’s last two matches, and none of the red flags is up.'
          : 'The pick holds up without each team’s last two games, and the line has not moved against it.') +
          (stake ? ' Suggested stake: ' + stake + ', a quarter of the Kelly amount and never more than 2%.' : '')));
    }
    if (side.label === 'AVOID') {
      const ev = state.picks.evidence && state.picks.evidence.all;
      let closing = 'The model does not see injuries, weather or news. When the market disagrees this much, it usually knows something.';
      if (soccer) {
        closing = 'The model does not see injuries, suspensions or rotation.' + (ev && ev.AVOID && ev.AVOID.n && ev.PLAY && ev.PLAY.n
          ? ' In the backtest, flagged picks returned ' + signed(ev.AVOID.roi * 100, 1, '%') + ' against ' + signed(ev.PLAY.roi * 100, 1, '%') + ' for the plays.' : '');
      }
      return h('div', { class: 'why avoid' },
        h('p', null, 'On paper there is an edge of ', b(edgeFmt(side.edge)), ' (model ' + pct(side.p_model) + ', market ' + pct(side.p_market) + '), but there is a red flag:'),
        h('ul', { style: 'margin-top:6px' }, side.reasons.map((r) => h('li', null, cap(r)))),
        h('p', { style: 'margin-top:6px' }, closing));
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
      h('p', { class: 'sub' }, 'Tap a side to add it to your ticket. One leg per ' + S().game + ': the math multiplies each leg’s chance, which only works when the legs are independent.'));
    if (isSoccer()) builder.append(h('div', { class: 'controls-row' }, leagueSeg()));
    const row = h('div', { class: 'controls-row' });
    if (!isSoccer() && mineCount()) row.append(seg('Games', [['mine', 'My games'], ['all', 'All games']], useMine() ? 'mine' : 'all', (v, id) => setState('scope', v, id)));
    row.append(betTypeSeg('Market', false, state.parlayMarket, (v, id) => { state.parlayMarket = v; refocus = id; render(); }));
    builder.append(row);
    const list = h('div');
    const market = state.parlayMarket;
    for (const g of scopedGames().slice().sort((x, y) => x.date.localeCompare(y.date))) {
      const keys = sideKeys(market);
      const opts = h('div', { class: 'legopts' + (keys.length === 3 ? ' three' : '') });
      for (const key of keys) {
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
    builder.append(list.children.length ? list : h('p', { class: 'empty' }, 'No lines for this market in the selected ' + S().games + '.'));
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
        h('div', { class: 'fig' }, pct(leg.p_model), h('small', null, (leg.p_market != null ? 'market ' + pct(leg.p_market) + ' · ' : '') + oddsFmt(leg.odds))),
        h('div', { class: 'bar' + (weak ? ' weak' : ''), role: 'img', 'aria-label': 'Model chance ' + pct(leg.p_model) }, h('i', { style: 'width:' + clamp(leg.p_model * 100, 2, 100).toFixed(0) + '%' }))));
    }
    box.append(grid);
    for (const leg of (saved.legs || []).filter((l) => l.no_data)) box.append(h('p', { class: 'small' }, 'No data for ' + leg.team + '.'));
    box.append(h('div', null, h('button', { type: 'button', class: 'btn', onclick: () => {
      const next = {};
      for (const leg of legs) next[leg.game_id] = { market: 'ml', key: leg.side };
      saveLegs(next);
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
    body.append(h('div', null, h('button', { type: 'button', class: 'btn ghost', onclick: () => { saveLegs({}); refocus = 'slip'; render(); } }, 'Clear ticket')));
    return box;
  }

  // ============================================================ TRACK RECORD
  const kpiBox = (k, v, sub) => h('div', null, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v), h('div', { class: 's' }, sub || ''));
  const weekName = (w) => (isSoccer() ? (dfShort ? dfShort.format(new Date(w.week + 'T00:00:00Z')) : String(w.week)) : 'Wk ' + w.week);
  const weekLong = (w) => (isSoccer() ? 'Week of ' + weekName(w) : 'Week ' + w.week);

  function renderRecord() {
    const B = state.backtest;
    const sel = B.comparison.find((c) => c.selected) || B.comparison[0];
    const cur = sel.current;
    if (!cur || !cur.n) {
      return [h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Track record'),
        h('p', { class: 'empty' }, 'No ' + seasonName(B) + ' ' + S().games + ' have been played yet. The track record appears after the first week.'))];
    }
    return isSoccer() ? soccerRecord(B, sel) : footballRecord(B, sel);
  }

  function footballRecord(B, sel) {
    const cur = sel.current, mkt = B.market;
    const kpi = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'How the model has done in ' + B.season),
      h('p', { class: 'sub' }, 'Every week was predicted using only the games played before it, across ' + cur.n + ' games between top-division teams. Nothing here was fitted after the fact.'));
    kpi.append(h('div', { class: 'kv' },
      kpiBox('Picked the winner', rate(cur.accuracy), mkt ? 'the market: ' + rate(mkt.accuracy) : ''),
      kpiBox('When favored by 14+', rate(cur.accuracy_14), plural(cur.n_14, 'game')),
      kpiBox('Typical miss, margin', cur.mae_margin.toFixed(1) + ' pts', mkt ? 'the market: ' + mkt.mae_margin.toFixed(1) : ''),
      kpiBox('Typical miss, total', cur.mae_total.toFixed(1) + ' pts', mkt ? 'the market: ' + mkt.mae_total.toFixed(1) : '')));
    return [kpi, bettingPanel(B), comparisonPanel(B, sel), weeksPanel(B), calibrationPanel(B)];
  }

  function soccerRecord(B, sel) {
    const cur = sel.current, mkt = B.market;
    const kpi = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'How the model has done in ' + seasonName(B)),
      h('p', { class: 'sub' }, 'Every week was predicted using only the matches played before it, across ' + cur.n + ' matches in the five leagues. Nothing here was fitted after the fact.'));
    kpi.append(h('div', { class: 'kv' },
      kpiBox('Called the result', rate(cur.accuracy), mkt ? 'the odds’ favorite: ' + rate(mkt.accuracy) : 'home win, draw or away win'),
      kpiBox('Forecast error (RPS)', cur.rps.toFixed(3), (mkt ? 'the market: ' + mkt.rps.toFixed(3) + ' · ' : '') + 'lower is better'),
      kpiBox('Typical miss, total goals', cur.mae_total.toFixed(2), 'goal difference: ' + cur.mae_margin.toFixed(2)),
      kpiBox('Matches drawn', rate(cur.draw_observed, 0), 'the model expected ' + rate(cur.draw_predicted, 0))));
    return [kpi, bettingPanel(B), soccerComparison(B), leaguePanel(B), soccerWeeks(B), calibrationPanel(B)];
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
          h('td', null, (market === 'all' ? 'All' : S().short[market]) + ' · ' + label.toLowerCase()),
          h('td', null, r.won + '–' + r.lost), h('td', { class: 'opt' }, rate(r.win_pct, 0)),
          h('td', { class: r.roi >= 0 ? 'pos' : 'neg' }, signed(r.roi * 100, 1, '%'), r.roi_error == null ? null : h('span', { class: 'pm' }, ' ±' + trim(r.roi_error * 100, 0))),
          h('td', { class: r.units >= 0 ? 'pos' : 'neg' }, signed(r.units, 0, 'u'))));
      }
    }
    table.append(body);
    return table;
  }

  function bettingPanel(B) {
    const soccer = isSoccer(), season = seasonName(B);
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Would the plays have made money?'));
    const cur = (B.bets.current || {}).all, hist = (B.bets.history || {}).all;
    if (!cur || !cur.PLAY) {
      box.append(h('p', { class: 'empty' }, 'There are no closing lines yet to simulate against.'));
      return box;
    }
    const j = cur.PLAY, hp = hist && hist.PLAY;
    const verdict = !j.n
      ? h('p', { class: 'verdict' }, 'No PLAY has been settled yet in ' + season + '.')
      : h('p', { class: 'verdict' }, b(j.roi >= 0 ? 'So far, yes.' : 'Not so far.'), ' Betting one unit on every PLAY in ' + season + ' at the closing ' + (soccer ? 'odds' : 'line') + ' went ' + j.won + '–' + j.lost +
        ' for ', b(signed(j.units, 1, ' units')), ' (ROI ' + signed(j.roi * 100, 1, '%') + ').');
    if (hp && hp.n) {
      verdict.append(' Across every season with lines (' + seasonList(B.bets.seasons_with_lines) + ') the ROI is ', b(signed(hp.roi * 100, 1, '%')), ' on ' + hp.n + ' bets' +
        (hp.roi < 0 && hp.roi > -0.08 ? ', roughly the bookmaker’s cut.' : '.'));
    }
    box.append(verdict);
    const flagged = B.bets.history && B.bets.history[soccer ? 'all' : 'spread'];
    if (flagged && flagged.AVOID.n && flagged.PLAY.n && flagged.AVOID.roi < flagged.PLAY.roi) {
      box.append(h('p', { class: 'sub' }, 'The AVOID flag earns its keep: ' + (soccer ? 'the' : 'spread') + ' picks it flagged returned ' + signed(flagged.AVOID.roi * 100, 1, '%') +
        ', against ' + signed(flagged.PLAY.roi * 100, 1, '%') + ' for the plays.'));
    }
    const chart = h('div', { class: 'chart', id: 'chart-units' });
    box.append(h('h3', { class: 'h3' }, 'Running total in units, ' + season), chart);
    const weeks = B.weeks.filter((w) => w.plays > 0);
    queueChart(() => drawUnits(chart, weeks));
    const two = h('div', { class: 'two' });
    two.append(h('div', null, h('h3', { class: 'h3' }, season + ' by bet type'), h('div', { class: 'scroll' }, recordTable(B.bets.current))));
    two.append(h('div', null, h('h3', { class: 'h3' }, 'Seasons ' + seasonList(B.bets.seasons_with_lines)), h('div', { class: 'scroll' }, recordTable(B.bets.history))));
    box.append(two);
    box.append(h('p', { class: 'small' }, 'ROI is profit per unit staked; ± is one standard error. The closing ' + (soccer ? 'odds are' : 'line is') + ' the last price before kickoff, the hardest one to beat.' +
      (soccer && B.tuned_on ? ' The red flags behind AVOID were chosen on ' + B.tuned_on + '; ' + season + ' is the first season they were not fitted to.' : '')));
    return box;
  }

  function barRows(rows, digits) {
    const max = Math.max(...rows.map((r) => r.value));
    const bars = h('div', { class: 'bars' });
    for (const r of rows) {
      bars.append(h('span', { class: 'name' + (r.hl ? ' hl' : '') }, r.name + (r.hl ? ' (in use)' : '')),
        h('span', null, h('span', { class: 'bar' + (r.hl ? ' hl' : ''), style: 'width:' + ((r.value / max) * 100).toFixed(1) + '%' })),
        h('span', { class: 'num' }, r.value.toFixed(digits)));
    }
    return bars;
  }
  const shortLabel = (label) => label.replace(/\s*\(.*\)$/, '');

  function comparisonPanel(B, sel) {
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Models compared'),
      h('p', { class: 'sub' }, 'Average miss on the final margin in ' + B.season + ', in points. Lower is better. The lowest-error model is the one in use; the market’s closing line is there for reference.'));
    const rows = B.comparison.map((c) => ({ name: shortLabel(c.label), value: c.current.mae_margin, hl: c.selected }));
    if (B.market) rows.push({ name: 'Market closing line', value: B.market.mae_margin, hl: false });
    rows.sort((x, y) => y.value - x.value);
    box.append(barRows(rows, 2));
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

  function soccerComparison(B) {
    const season = seasonName(B), ref = B.reference;
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Models compared'),
      h('p', { class: 'sub' }, 'Forecast error in ' + season + ' (RPS): how far the three percentages landed from what happened. 0 would be perfect and lower is better. ' +
        'The betting market is the bar to clear; quoting league averages for every match is the floor.'));
    const rows = B.comparison.map((c) => ({ name: shortLabel(c.label), value: c.current.rps, hl: c.selected }));
    if (B.market) rows.push({ name: 'Market closing odds', value: B.market.rps, hl: false });
    if (ref && ref.current) rows.push({ name: 'League averages', value: ref.current.rps, hl: false });
    rows.sort((x, y) => y.value - x.value);
    box.append(barRows(rows, 3));
    box.append(h('p', { class: 'small' }, 'The Poisson model is the one in use because it is the only one that gives a chance for every scoreline, which totals and handicaps need.'));
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['Model', 'Result', 'RPS', 'Log loss', 'Total miss', (B.tuned_on || 'Past') + ' RPS'].map((t) => h('th', null, t)))));
    const body = h('tbody');
    const line = (label, a, hist, hl) => body.append(h('tr', { class: hl ? 'hl' : null },
      h('td', { class: 'wrapcell' }, label), h('td', null, rate(a.accuracy)), h('td', null, a.rps.toFixed(4)), h('td', null, a.log_loss.toFixed(4)),
      h('td', null, a.mae_total == null ? '–' : a.mae_total.toFixed(2)), h('td', null, hist && hist.rps != null ? hist.rps.toFixed(4) : '–')));
    for (const c of B.comparison) line(c.label, c.current, c.history, c.selected);
    if (B.market) line('Market closing odds', B.market, B.market_history, false);
    if (ref && ref.current) line(ref.label, ref.current, ref.history, false);
    table.append(body);
    box.append(h('details', null, h('summary', { class: 'small' }, 'See the full table'), h('div', { class: 'scroll', style: 'margin-top:8px' }, table),
      h('p', { class: 'small', style: 'margin-top:8px' }, 'Result is the share of matches where the most likely outcome happened. Log loss is a second measure of the same thing as RPS, harsher on confident misses.')));
    return box;
  }

  function leaguePanel(B) {
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'League by league'),
      h('p', { class: 'sub' }, 'The same model, scored separately for each league in ' + seasonName(B) + '.'));
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null, ['League', 'Result', 'RPS', 'Total miss'].map((t) => h('th', null, t)))));
    const body = h('tbody');
    for (const lg of B.by_league || []) {
      body.append(h('tr', null, h('td', { class: 'cell2' }, lg.name, h('small', null, plural(lg.n, 'match', 'matches'))),
        h('td', null, h('span', { class: 'meter', 'aria-hidden': 'true' }, h('i', { style: 'width:' + (lg.accuracy * 100).toFixed(0) + '%' })), rate(lg.accuracy, 0)),
        h('td', null, lg.rps.toFixed(3)), h('td', null, lg.mae_total.toFixed(2))));
    }
    table.append(body);
    box.append(h('div', { class: 'scroll' }, table));
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

  function soccerWeeks(B) {
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Week by week'),
      h('p', { class: 'sub' }, 'Results is the share of matches where the model’s most likely outcome happened. Plays is the record of that week’s PLAY picks against the closing odds.'));
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['Week of', 'Results', 'RPS', 'Total miss', 'Plays', 'Units'].map((t) => h('th', { class: t === 'RPS' || t === 'Total miss' ? 'opt' : null }, t)))));
    const body = h('tbody');
    for (const w of B.weeks) {
      body.append(h('tr', null, h('td', null, weekName(w)),
        h('td', null, h('span', { class: 'meter', 'aria-hidden': 'true' }, h('i', { style: 'width:' + (w.accuracy * 100).toFixed(0) + '%' })), rate(w.accuracy, 0), h('span', { class: 'pm opt' }, ' of ' + w.n)),
        h('td', { class: 'opt' }, w.rps == null ? '–' : w.rps.toFixed(3)), h('td', { class: 'opt' }, w.mae_total == null ? '–' : w.mae_total.toFixed(2)),
        h('td', null, w.plays ? w.won + '–' + w.lost : '–'),
        h('td', { class: w.plays ? (w.units >= 0 ? 'pos' : 'neg') : null }, w.plays ? signed(w.units, 1, 'u') : '–')));
    }
    table.append(body);
    box.append(h('div', { class: 'scroll' }, table));
    return box;
  }

  function calibrationPanel(B) {
    const cal = B.calibration, soccer = isSoccer(), season = seasonName(B);
    const off = cal.mean_error == null ? '–' : trim(cal.mean_error * 100, 1) + ' points';
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Can you trust the percentages?'),
      h('p', { class: 'sub' }, soccer
        ? 'Every home-win, draw and away-win percentage the model gave is pooled here. Outcomes it called 30% likely should happen about 30% of the time. Dots on the diagonal mean the percentages are honest. In past seasons they were off by ' + off + ' on average.'
        : 'When the model gives its favorite a 70% chance, that team should win about 70% of the time. Dots on the diagonal mean the percentages are honest. In past seasons they were off by ' + off + ' on average.'));
    const hist = B.history_seasons || [];
    const histName = !hist.length ? 'Past seasons' : soccer ? seasonList(hist) : hist[0] + '–' + hist[hist.length - 1];
    box.append(h('div', { class: 'legend' },
      h('span', null, h('i', { class: 'dot', style: 'background:var(--series-1)' }), season),
      h('span', null, h('i', { class: 'dot', style: 'background:var(--series-2)' }), histName),
      h('span', null, h('i', { style: 'background:var(--track);height:1px' }), 'Perfect calibration')));
    const chart = h('div', { class: 'chart', id: 'chart-cal' });
    box.append(chart);
    const words = soccer
      ? { xMin: 0, x: 'Chance the model gave', y: 'How often it happened', did: 'Happened', unit: 'cases', aria: 'Calibration: the chance the model gave an outcome against how often it happened' }
      : { xMin: 0.5, x: 'Chance the model gave its favorite', y: 'How often it won', did: 'Won', unit: 'games', aria: 'Calibration: the chance the model gave its favorite against how often that team won' };
    queueChart(() => drawCalibration(chart, cal.current, cal.history, season, histName, words));
    const head = soccer ? ['Model said', season + ' happened', 'Cases', histName + ' happened', 'Cases'] : ['Model said', season + ' won', 'Games', histName + ' won', 'Games'];
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null, head.map((t) => h('th', null, t)))));
    const body = h('tbody');
    const histBy = new Map(cal.history.map((x) => [x.from, x]));
    const curBy = new Map(cal.current.map((x) => [x.from, x]));
    const bins = Array.from(new Set([...curBy.keys(), ...histBy.keys()])).sort((x, y) => x - y);
    for (const lo of bins) {
      const c = curBy.get(lo), hb = histBy.get(lo);
      const hi = (c || hb).to;
      body.append(h('tr', null, h('td', null, Math.round(lo * 100) + '–' + Math.round((hi == null ? lo + 0.05 : hi) * 100) + '%'),
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
    const every = Math.ceil(weeks.length / Math.max(Math.floor(iw / (isSoccer() ? 52 : 40)), 1));
    weeks.forEach((w, i) => {
      if (i % every === 0 || i === weeks.length - 1) svg.append(s('text', { x: x(i), y: H - 10, 'text-anchor': 'middle' }, weekName(w)));
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
        'aria-label': weekLong(w) + ': running total ' + signed(w.cumulative, 1) + ' units' });
      const on = (evt) => {
        cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
        dot.setAttribute('cx', x(i)); dot.setAttribute('cy', y(w.cumulative)); dot.setAttribute('visibility', 'visible');
        const box = svg.getBoundingClientRect();
        showTip(hit, [h('b', null, signed(w.cumulative, 1, 'u') + ' so far'), h('span', null, weekLong(w) + ': ' + w.won + '–' + w.lost + ', ' + signed(w.units, 1, 'u'))],
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

  function drawCalibration(el, cur, hist, curName, histName, words) {
    el.replaceChildren();
    const W = Math.max(el.clientWidth, 260), H = Math.round(clamp(W * 0.72, 230, 360)), m = { l: 58, r: 14, t: 10, b: 42 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const xMin = words.xMin;
    const observed = [...cur, ...hist].map((bin) => bin.observed);
    const yMin = Math.min(xMin, Math.floor(Math.min(...observed, xMin) * 10) / 10);
    const x = (p) => m.l + ((p - xMin) / (1 - xMin)) * iw;
    const y = (p) => m.t + (1 - (p - yMin) / (1 - yMin)) * ih;
    const tick = xMin === 0 ? 0.2 : 0.1;
    const svg = s('svg', { viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, role: 'img', 'aria-label': words.aria });
    for (let v = yMin; v <= 1 + 1e-9; v += tick) {
      svg.append(s('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }), s('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, Math.round(v * 100) + '%'));
    }
    for (let v = xMin; v <= 1 + 1e-9; v += tick) svg.append(s('text', { x: x(v), y: H - m.b + 16, 'text-anchor': 'middle' }, Math.round(v * 100) + '%'));
    svg.append(s('text', { x: m.l + iw / 2, y: H - 6, 'text-anchor': 'middle' }, words.x));
    svg.append(s('text', { x: 11, y: m.t + ih / 2, 'text-anchor': 'middle', transform: 'rotate(-90 11 ' + (m.t + ih / 2) + ')' }, words.y));
    svg.append(s('line', { class: 'axis', x1: x(Math.max(xMin, yMin)), y1: y(Math.max(xMin, yMin)), x2: x(1), y2: y(1) }));
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
          'aria-label': name + ': the model said ' + rate(bin.predicted, 0) + ' and it ' + words.did.toLowerCase() + ' ' + rate(bin.observed, 0) + ' of ' + bin.n + ' ' + words.unit });
        const on = () => showTip(hit, [h('b', null, words.did + ' ' + rate(bin.observed, 0)), h('span', null, name + ' · model said ' + rate(bin.predicted, 0) + ' · ' + bin.n + ' ' + words.unit + ' (±' + trim(bin.std_error * 100, 0) + ')')]);
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
  const sec = (title, ...kids) => h('section', { class: 'panel' }, h('h2', { class: 'h2' }, title), kids);
  const para = (...kids) => h('p', null, kids);
  function glossary(terms) {
    const gloss = h('dl', { class: 'gloss' });
    for (const [term, text] of terms) gloss.append(h('div', null, h('dt', null, term), h('dd', null, text)));
    return gloss;
  }
  const renderHow = () => (isSoccer() ? soccerHow() : footballHow());

  function footballHow() {
    const P = state.picks, B = state.backtest, r = P.rules, m = P.model;
    const key = h('div', { class: 'key' },
      pill('PLAY'), h('div', null, 'The model’s chance beats the market’s by ' + (r.edge_min * 100).toFixed(0) + '% or more, the bet has positive expected value at the real price, and there are no red flags.'),
      pill('AVOID'), h('div', null, 'There is an edge on paper, but the pick flips when each team’s last 2 games are removed, the model is ' + trim(r.gap_margin) + '+ points away from the market spread (' +
        trim(r.gap_total) + '+ on a total), or the line moved ' + trim(r.move_spread) + '+ points against it since it opened (' + trim(r.move_total) + '+ on a total).'),
      pill('PASS'), h('div', null, 'The model and the market roughly agree. Nothing to do.'));
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
    return [
      sec('The model in plain words', h('div', { class: 'prose' },
        para('Each team gets a rating: its average scoring margin, adjusted for how good its opponents were. A team’s rating depends on its opponents’ ratings and theirs on it, so all of them are solved together from the ' +
          m.training_games + ' games played this season.'),
        para('A projection is the difference between two ratings plus home-field advantage. A second set of ratings, built the same way on combined scores, gives the projected total and from it the score.'),
        para('Early in the season a rating leans on last year’s, and blowouts are capped so one 63–0 does not count for too much.'))),
      sec('From a projection to a percentage', h('div', { class: 'prose' },
        para('In testing, real margins landed about ' + trim(m.sigma_margin) + ' points away from the projection on a typical day (' + trim(m.sigma_total) + ' for totals). That spread turns a projected margin into a chance of winning, covering a spread or going over a total.'),
        para('No percentage is ever shown as 100%, and the typical miss sits next to every projection for the same reason.'))),
      sec('What the labels mean', key),
      sec('What the model does not know', h('div', { class: 'prose' },
        h('ul', null,
          h('li', null, 'Injuries, suspensions and quarterback changes.'),
          h('li', null, 'Weather. A total that drops 10 points during the week is almost always wind or rain.'),
          h('li', null, 'Motivation, rest, travel and look-ahead spots.'),
          h('li', null, 'Anything the market has already priced in, which is why the closing line misses by less than the model does.')),
        para('Notes and manual point adjustments can be added per game in ', h('code', null, 'data/notes.csv'), '.'))),
      sec('How the track record is built', h('div', { class: 'prose' },
        para('For week N the model only sees games played before week N. Its settings were tuned on ' + (B.tuned_on || 'past seasons') + ', and ' + B.season +
          ' was scored without touching them. The betting record stakes one unit on every PLAY at the closing line.'))),
      sec('Glossary', glossary(terms)),
    ];
  }

  function soccerHow() {
    const P = state.picks, B = state.backtest, r = P.rules, m = P.model, cfg = m.settings || {};
    const months = cfg.half_life ? Math.round(cfg.half_life / 30.4) : null;
    const key = h('div', { class: 'key' },
      pill('PLAY'), h('div', null, 'The model’s chance beats the market’s by ' + (r.edge_min * 100).toFixed(0) + '% or more, the bet has positive expected value at the real price, and no red flag is up.'),
      pill('AVOID'), h('div', null, 'There is an edge on paper, but a red flag is up: the edge is ' + (r.gap_edge * 100).toFixed(0) + '+ points, which means the market strongly disagrees; it is a result bet on a side the market gives under ' +
        (r.ml_min_market * 100).toFixed(0) + '%; the pick flips when each team’s last 2 matches are removed; or a team has fewer than ' + r.min_games + ' league matches on record.'),
      pill('PASS'), h('div', null, 'The model and the market roughly agree. Nothing to do.'));
    const terms = [
      ['Match result', 'A bet on home win, draw or away win after 90 minutes plus stoppage time. −255 means you risk $255 to win $100; +425 means you risk $100 to win $425.'],
      ['Handicap', 'Goals added to a team before the bet is settled. Fulham +1.5 wins if Fulham wins, draws or loses by one. On a whole-number handicap an exact tie is a push and the stake comes back.'],
      ['Total goals', 'A bet on both teams’ goals combined going over or under a number.'],
      ['Expected goals', 'How many goals the model expects a team to score in this match, on average. 1.6 does not mean a 1.6 scoreline; it is the center of a range.'],
      ['Vig', 'The bookmaker’s cut, built into the prices. The three result prices add up to about 106%, not 100%.'],
      ['Market chance', 'The chance the prices imply once the vig is removed. It is the fairest read of what the market believes.'],
      ['Edge', 'The model’s chance minus the market chance. A positive edge means the model thinks the price is too generous.'],
      ['Stake', 'A quarter of the Kelly amount, which sizes a bet by its edge, and never more than ' + (r.stake_cap * 100).toFixed(0) + '% of your bankroll.'],
      ['RPS', 'Ranked probability score: the gap between the three percentages and what happened. 0 is perfect, and quoting league averages for every match scores about 0.23.'],
    ];
    const season = seasonName(B);
    return [
      sec('The model in plain words', h('div', { class: 'prose' },
        para('Every team gets two ratings: an attack rating for the goals it scores and a defense rating for the goals it keeps out. A team’s expected goals in a match come from its attack against the opponent’s defense, plus a home edge.'),
        para('All the ratings in a league are solved together from about three seasons of matches (' + m.training_games + ' across the five leagues). Recent matches count more' +
          (months ? ': a match’s weight halves every ' + months + ' months' : '') + '.'),
        para('Ratings are pulled toward the league average with the weight of ' + trim(cfg.k == null ? 5 : cfg.k) + ' matches, so a short hot streak does not produce an extreme rating. A newly promoted team starts below average until its results say otherwise.'))),
      sec('From expected goals to percentages', h('div', { class: 'prose' },
        para('Two expected-goal numbers give a probability for every scoreline from 0–0 up, with a small correction because real matches end 0–0 and 1–1 more often than the plain formula says. Adding up scorelines gives the chance of a home win, a draw and an away win, of going over a total and of covering a handicap.'),
        para('No percentage is ever shown as 100%. In testing the percentages were off by ' + (m.calibration_error == null ? 'about a point' : trim(m.calibration_error * 100, 1) + ' points') +
          ' on average, and the goal difference of a match lands ' + (m.mae_margin == null ? 'more than a goal' : m.mae_margin.toFixed(1) + ' goals') + ' away from the projection on a typical day.'))),
      sec('What the labels mean', key,
        h('p', { class: 'small' }, 'The red flags come from the ' + (B.tuned_on || 'past seasons') + ' backtest: picks carrying any of them lost far more than the bookmaker’s cut. A line that moved since it opened is shown on the card but is not a flag, because it did not separate good picks from bad ones.')),
      sec('What the model does not know', h('div', { class: 'prose' },
        h('ul', null,
          h('li', null, 'Injuries and suspensions. A missing striker or goalkeeper is invisible to it.'),
          h('li', null, 'Rotation around European nights and cup ties.'),
          h('li', null, 'A new coach, or what is at stake late in the season.'),
          h('li', null, 'Anything the market has already priced in, which is why the closing odds forecast better than the model does.')),
        para('Notes and manual goal adjustments can be added per match in ', h('code', null, 'data/soccer_notes.csv'), '.'))),
      sec('How the track record is built', h('div', { class: 'prose' },
        para('For each week the model only sees matches played before that week began. Its settings were tuned on ' + (B.tuned_on || 'past seasons') + ', and ' + season +
          ' was scored without touching them. The betting record stakes one unit on every PLAY at the closing odds ESPN published.'),
        para('One season of archived goal lines (2023-24) is left out of the betting record: that feed kept updating during matches, so it was not a price anyone could have bet before kickoff.'))),
      sec('Glossary', glossary(terms)),
    ];
  }

  // -------------------------------------------------------------------- boot
  const VIEWS = { picks: renderPicks, parlay: renderParlay, record: renderRecord, how: renderHow };
  function route() {
    const parts = (location.hash || '').replace(/^#/, '').split('/');
    const named = SPORTS[parts[0]] ? parts[0] : null;
    const name = named ? parts[1] : parts[0];
    const saved = store.get('sport', 'cfb');
    return { sport: named || (SPORTS[saved] ? saved : 'cfb'), view: VIEWS[name] ? name : 'picks' };
  }

  // Header, sport switch and tabs follow the sport and section in the URL.
  function chrome(name) {
    const sp = S(), P = state.picks;
    document.title = sp.title;
    document.getElementById('field').dataset.sport = state.sport;
    document.getElementById('hdr-title').textContent = sp.title;
    document.querySelectorAll('.sports a').forEach((a) => {
      a.setAttribute('href', '#' + a.dataset.sport + '/' + name);
      if (a.dataset.sport === state.sport) a.setAttribute('aria-current', 'true');
      else a.removeAttribute('aria-current');
    });
    document.querySelectorAll('.tabs a').forEach((a) => {
      a.setAttribute('href', '#' + state.sport + '/' + a.dataset.view);
      if (a.dataset.view === name) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
    const week = document.getElementById('hdr-week'), meta = document.getElementById('hdr-meta'), foot = document.getElementById('foot-gen');
    if (!P) {
      week.textContent = sp.name;
      meta.textContent = state.error ? 'No data' : 'Loading…';
      foot.textContent = '';
      return;
    }
    week.textContent = isSoccer() ? 'Top 5 leagues · ' + seasonName(P) : P.week_label + ' · ' + P.season;
    const mine = mineCount();
    meta.textContent = 'Updated ' + stamp(P.generated) + ' · ' + gamesWord(allGames().length) + (mine ? ', ' + mine + ' tracked' : '');
    foot.textContent = 'Scores and lines: ESPN’s public API. Model: ' + P.model.label + '.';
  }

  function render() {
    hideTip();
    const name = route().view;
    chrome(name);
    if (state.error) {
      view.replaceChildren(h('section', { class: 'callout' }, h('b', { class: 't' }, 'The picks could not be loaded'),
        h('p', null, state.error), h('p', null, 'Generate the data with ' + S().command + ' and open the site through a web server (python -m http.server -d site), not as a local file.')));
      updateParlayCount();
      return;
    }
    if (!state.picks) {
      view.replaceChildren(h('p', { class: 'loading' }, 'Loading the ' + S().name.toLowerCase() + ' picks…'));
      updateParlayCount();
      return;
    }
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

  // Point the page at one sport: its data (if already loaded) and its own parlay ticket.
  function activate(sport) {
    const changed = sport !== state.sport;
    state.sport = sport;
    const data = DATA[sport] || {};
    state.picks = data.picks || null;
    state.backtest = data.backtest || null;
    state.error = data.error || null;
    state.legs = store.get(legsKey(), {});
    if (state.picks) {   // a ticket only keeps legs whose game is still on the board
      const ids = new Set(allGames().map((g) => g.id));
      const kept = Object.fromEntries(Object.entries(state.legs || {}).filter(([id]) => ids.has(id)));
      if (Object.keys(kept).length !== Object.keys(state.legs || {}).length) saveLegs(kept);
    }
    if (changed) {
      state.query = '';
      state.showAllPlays = false;
      state.parlayMarket = 'ml';
      state.slipOpen = false;
    }
  }

  async function loadSport(sport) {
    if (DATA[sport]) return;
    try {
      const [picks, backtest] = await Promise.all(SPORTS[sport].files.map(async (url) => {
        const resp = await fetch(url, { cache: 'no-store' });
        if (!resp.ok) throw new Error(url + ' answered HTTP ' + resp.status);
        return resp.json();
      }));
      DATA[sport] = { picks, backtest };
    } catch (err) {
      DATA[sport] = { error: String((err && err.message) || err) };
    }
  }

  async function go() {
    const sport = route().sport;
    store.set('sport', sport);
    activate(sport);
    render();
    if (!DATA[sport]) {
      await loadSport(sport);
      if (route().sport !== sport) return;   // the visitor moved on while this was loading
      activate(sport);
      render();
    }
  }

  window.addEventListener('hashchange', () => { go(); window.scrollTo(0, 0); view.focus({ preventScroll: true }); });
  let resizeTimer = null, lastWidth = window.innerWidth;
  window.addEventListener('resize', () => {
    if (window.innerWidth === lastWidth) return;
    lastWidth = window.innerWidth;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (route().view === 'record') render(); }, 150);
  });
  window.addEventListener('scroll', hideTip, { passive: true });
  go();
})();
