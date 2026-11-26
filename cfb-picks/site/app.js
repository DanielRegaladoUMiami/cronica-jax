/* Picks site for three sections: NFL, college football and European soccer.
   The URL names the section and the tab (#nfl-players, #cfb-record, #soccer-picks ...)
   and the page reads that section's JSON from data/.
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

  // ----------------------------------------------------------------- leagues
  // The two football leagues share every view; what differs is the data and a few words.
  // Soccer is a different kind of page: a board of market odds and Kalshi prices, no model.
  const FOOTBALL_TABS = [['picks', 'Picks'], ['parlay', 'Parlay'], ['record', 'Track record'], ['how', 'How it works']];
  const DEFAULT_SPORT = 'nfl';
  const SPORTS = {
    cfb: {
      title: 'CFB Picks', name: 'College football', college: true, tabs: FOOTBALL_TABS,
      files: { picks: 'data/picks.json', backtest: 'data/backtest.json' }, command: 'python -m src.picks', notes: 'data/notes.csv',
      playoffs: {},
      blowout: 'one 63–0',
      blind: [
        'Injuries, suspensions and quarterback changes.',
        'Weather. A total that drops 10 points during the week is almost always wind or rain.',
        'Motivation, rest, travel and look-ahead spots.',
      ],
      spread: 'A bet on the margin. Indiana −7.5 wins if Indiana wins by 8 or more; the other side wins otherwise.',
    },
    nfl: {
      title: 'NFL Picks', name: 'NFL', college: false,
      tabs: [['picks', 'Picks'], ['players', 'Players'], ['parlay', 'Parlay'], ['record', 'Track record'], ['how', 'How it works']],
      files: { picks: 'data/nfl_picks.json', backtest: 'data/nfl_backtest.json', players: 'data/nfl_players.json' }, optional: ['players'],
      command: 'python -m src.picks --league nfl', notes: 'data/nfl_notes.csv',
      playoffs: { 19: ['WC', 'Wild Card'], 20: ['Div', 'Divisional round'], 21: ['Conf', 'Conference championships'], 22: ['SB', 'Super Bowl'] },
      blowout: 'one 45–3',
      blind: [
        'Injuries. A starting quarterback who is out can move a line by a touchdown, and the model does not notice.',
        'Weather. Wind and cold pull totals down during the week.',
        'Rest and travel: short weeks, byes and games played abroad.',
      ],
      spread: 'A bet on the margin. Jaguars −6 wins if Jacksonville wins by 7 or more; if it wins by exactly 6 the stake comes back.',
    },
    soccer: {
      title: 'Soccer Board', name: 'European cups', kind: 'board', tabs: [['picks', 'Matches'], ['how', 'How it works']],
      files: { board: 'data/soccer.json' }, command: 'python -m src.soccer.board',
    },
  };
  const DATA = {};   // section -> { picks, backtest, players } or { board }, or { error }

  // ------------------------------------------------------------------- state
  const state = {
    sport: DEFAULT_SPORT,
    picks: null,
    backtest: null,
    players: null,
    board: null,
    error: null,
    playersGame: null,
    playerOpen: null,
    soccerComp: store.get('soccerComp', 'all'),
    soccerPlayed: store.get('soccerPlayed', false),
    scope: store.get('scope', 'mine'),
    market: store.get('market', 'all'),
    allMarkets: store.get('allMarkets', true),
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
  const legsKey = () => (state.sport === 'cfb' ? 'legs' : 'legs:' + state.sport);

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
  // Moneyline, spread and total are all on by default; unticking the box leaves only who wins.
  const mlOnly = () => !state.allMarkets;
  const visibleMarkets = () => (mlOnly() ? ['ml'] : state.market === 'all' ? ['ml', 'spread', 'total'] : [state.market]);
  const countText = (right, wrong, ties) => right + ' right · ' + wrong + ' wrong' + (ties ? ' · ' + plural(ties, 'tie') : '');
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
  // The chance the evidence supports: the market's, moved toward the model's only as far as the track record backs it.
  const fairOf = (side) => (side.p_fair != null ? side.p_fair : side.p_model);
  const KALSHI_FEE = 0.07;
  // The side the model likes more than the market does. For a moneyline that is often not the pick.
  function leanOf(g, market) {
    const block = g[market];
    const key = block && (block.lean || block.pick);
    const side = key ? block[key] : null;
    return side ? { g, market, key, side } : null;
  }
  function collectPicks(games) {
    const out = [];
    for (const g of games) {
      for (const market of visibleMarkets()) {
        const p = leanOf(g, market);
        if (p && p.side.edge != null) out.push(p);
      }
    }
    return out.sort((x, y) => y.side.edge - x.side.edge);
  }
  const hasPlay = (g, market) => sideKeys(market).some((k) => g[market] && g[market][k] && g[market][k].label === 'PLAY');
  // What happened in past games where the model was 10+ points above the market.
  function disagreement(anchor) {
    const rows = ((anchor && anchor.buckets) || []).filter((r) => r.lo >= 0.0999);
    const n = rows.reduce((acc, r) => acc + r.n, 0);
    if (!n) return null;
    const avg = (k) => rows.reduce((acc, r) => acc + r[k] * r.n, 0) / n;
    return { n, model: avg('model'), market: avg('market'), actual: avg('actual') };
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
    if (!side) return null;
    // A total says nothing about which game it is, so the ticket adds the matchup.
    return { g, market, key, side, label: sideLabel(g, market, key, true) + (market === 'total' ? ' (' + matchup(g) + ')' : '') };
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
    if (!el) return;   // this section has no parlay tab
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
    const P = state.picks, m = P.model, B = state.backtest;
    const box = h('section', { class: 'callout', 'aria-label': 'Reality check' }, h('b', { class: 't' }, 'Reality check'));
    if (!m.n_backtest) {
      box.append(h('p', null, 'No ' + P.season + ' games have been played yet, so these projections lean on last season’s ratings and are less reliable.'));
      return box;
    }
    const line = h('p', null, 'On its own the model has picked ', b(rate(m.accuracy, 0) + ' of winners'), ' this season');
    if (m.correct != null) line.append(' (' + m.correct + ' right, ' + m.wrong + ' wrong)');
    const mkt = B && B.market;
    if (mkt && mkt.accuracy != null) line.append('; the betting favorite won ', b(rate(mkt.accuracy, 0)));
    line.append('. ');
    const anchor = P.evidence && P.evidence.anchor && P.evidence.anchor.ml;
    const d = disagreement(anchor);
    if (d) {
      line.append('When the two disagree, the market has been right: in ' + d.n + ' past games where the model gave a team 10+ points more than the market did, the model said ',
        b(pct(d.model)), ', the market said ', b(pct(d.market)), ' and the team won ', b(pct(d.actual)), '. ');
      line.append(anchor.trust > 0 ? 'So the chance shown here leans mostly on the market. ' : 'So the chance shown for every team is the market’s, with the model’s own number beside it. ');
    }
    line.append(h('a', { href: '#' + state.sport + '-record' }, 'See the track record'));
    box.append(line);
    return box;
  }

  function topPlays() {
    const P = state.picks, games = scopedGames();
    const box = h('section', { class: 'panel' });
    const likely = games.map((g) => pickOf(g, 'ml')).filter((p) => p && p.side.p_market != null).sort((x, y) => fairOf(y.side) - fairOf(x.side));
    const where = useMine() ? 'your ' + plural(games.length, 'tracked game') : plural(games.length, 'game');
    box.append(h('div', { class: 'panel-head' }, h('h2', { class: 'h2' }, 'Most likely winners this week')));
    if (!likely.length) {
      box.append(h('p', { class: 'empty' }, 'No game in ' + where + ' has a moneyline yet.'));
    } else {
      box.append(h('p', { class: 'sub' }, 'Ranked by chance of winning, across ' + where + '. A likely winner is not a bargain: the price already says how likely it is. Tap one to see the game.'));
      const shown = state.showAllPlays ? likely : likely.slice(0, 5);
      box.append(h('div', { class: 'plays' }, shown.map((p, i) => likelyRow(p, i + 1))));
      if (likely.length > 5) {
        box.append(h('button', { type: 'button', class: 'btn more', id: 'more-plays', onclick: () => setState('showAllPlays', !state.showAllPlays, 'more-plays') },
          state.showAllPlays ? 'Show top 5 only' : 'Show all ' + likely.length));
      }
    }
    const plays = [];
    for (const g of games) for (const market of visibleMarkets()) for (const key of sideKeys(market)) {
      const side = g[market] && g[market][key];
      if (side && side.label === 'PLAY') plays.push({ g, market, key, side });
    }
    plays.sort((x, y) => (y.side.value || 0) - (x.side.value || 0));
    if (plays.length) {
      box.append(h('h3', { class: 'h3' }, plural(plays.length, 'play')),
        h('p', { class: 'small' }, 'The price is ' + (P.rules.edge_min * 100).toFixed(0) + '%+ better than the chance, with no red flag.'),
        h('div', { class: 'plays' }, plays.map((p, i) => playRow(p, i + 1))));
    } else {
      box.append(h('p', { class: 'small' }, 'No play this week. A play needs a price ' + (P.rules.edge_min * 100).toFixed(0) +
        '%+ better than the chance to win, and the model has not shown it can find one: where it disagreed with the market, the market was right.'));
    }
    const gaps = collectPicks(games).filter((p) => p.side.edge >= P.rules.edge_min);
    if (gaps.length) {
      box.append(h('details', null,
        h('summary', { class: 'h3' }, 'Where the model disagrees with the market (' + gaps.length + ')'),
        h('p', { class: 'small', style: 'margin-top:8px' }, 'The model alone would take these. They are not picks: the track record shows the market’s number holding up, and the ones marked AVOID have a red flag on top (a line that moved against them, an unstable projection or a new quarterback).'),
        h('div', { class: 'plays' }, gaps.map((p) => playRow(p, null)))));
    }
    return box;
  }

  function likelyRow(p, rank) {
    const { g, key, side } = p;
    const other = g[key === 'home' ? 'away' : 'home'];
    const chance = fairOf(side);
    let ctx = (key === 'home' ? 'vs ' : 'at ') + other.name + ' · ' + kickFmt(g.date) + ' · the model alone says ' + pct(side.p_model);
    const k = g.kalshi && g.kalshi[key];
    if (k && k.yes_ask != null) {
      ctx += ' · Kalshi ' + cents(k.yes_ask) + ' (' + signed((chance - k.yes_ask - KALSHI_FEE * k.yes_ask * (1 - k.yes_ask)) * 100, 1, '¢') + ' after the fee)';
    }
    if ((g.qb_changes || []).length) ctx += ' · a quarterback change in this game';
    return h('button', { type: 'button', class: 'playrow', onclick: () => goToGame(g.id, 'ml') },
      h('span', { class: 'rankno' }, String(rank)),
      h('span', { class: 'what' }, g[key].name + ' to win ', h('span', { class: 'price' }, '(' + oddsFmt(side.odds) + ')')),
      h('span', { class: 'edge' }, pct(chance), h('small', null, 'chance to win')),
      h('span', { class: 'ctx' }, ctx));
  }

  function playRow(p, rank) {
    const { g, market, key, side } = p;
    const stake = stakeText(side);
    let ctx = MARKET_NAME[market] + ' · ' + matchup(g) + ' · ' + kickFmt(g.date);
    if (rank && stake) ctx += ' · stake ' + stake;
    if (!rank && side.reasons.length) ctx += ' · ' + side.reasons[0];
    return h('button', { type: 'button', class: 'playrow', onclick: () => goToGame(g.id, market) },
      rank ? h('span', { class: 'rankno' }, String(rank)) : pill(side.label),
      h('span', { class: 'what' }, sideLabel(g, market, key, true), ' ', h('span', { class: 'price' }, '(' + oddsFmt(side.odds) + ')')),
      h('span', { class: 'edge' }, edgeFmt(rank ? side.value : side.edge), h('small', null, rank ? pct(fairOf(side)) + ' vs ' + pct(side.p_market) : 'model ' + pct(side.p_model) + ' · market ' + pct(side.p_market))),
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
    if (!mlOnly()) row.append(seg('Bet type', [['all', 'All bets'], ['ml', 'Moneyline'], ['spread', 'Spread'], ['total', 'Total']], state.market, (v, id) => setState('market', v, id)));
    row.append(h('label', { class: 'toggle', for: 'plays-only' },
      h('input', { id: 'plays-only', type: 'checkbox', checked: state.playsOnly, onchange: (e) => setState('playsOnly', e.target.checked, 'plays-only') }), 'Plays only'));
    row.append(h('label', { class: 'toggle', for: 'all-markets' },
      h('input', { id: 'all-markets', type: 'checkbox', checked: state.allMarkets, onchange: (e) => setState('allMarkets', e.target.checked, 'all-markets') }), 'Spreads and totals too'));
    box.append(row);
    return box;
  }

  function filteredGames() {
    const q = state.query.trim().toLowerCase();
    let list = q ? allGames() : scopedGames();   // a search looks through every game
    if (q) list = list.filter((g) => [g.home.name, g.away.name, g.home.abbr, g.away.abbr].some((x) => String(x).toLowerCase().includes(q)));
    if (state.playsOnly) list = list.filter((g) => visibleMarkets().some((m) => hasPlay(g, m)));
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
    const favHome = pr.margin >= 0;   // the model's favorite, for its own line below
    const priced = g.ml.home && g.ml.away && g.ml.home.p_market != null;
    const chance = (key) => (priced ? fairOf(g.ml[key]) : key === 'home' ? pr.p_home : pr.p_away);
    const likelyHome = chance('home') >= chance('away');
    const card = h('article', { class: 'panel game', id: 'g-' + g.id });
    const tags = h('div', { class: 'tags' });
    if (g.on_my_list) tags.append(h('span', { class: 'tag' }, 'My lines'));
    if (g.neutral) tags.append(h('span', { class: 'tag' }, 'Neutral site'));
    if (g.fcs) tags.append(h('span', { class: 'tag' }, 'vs FCS'));
    card.append(h('div', { class: 'game-top' }, h('span', { class: 'kick' }, kickFmt(g.date)), tags));

    const teams = h('div', { class: 'teams' });
    for (const key of ['away', 'home']) {
      const t = g[key];
      const isFav = (key === 'home') === likelyHome;
      const meta = [t.record, t.rank ? 'AP #' + t.rank : null, t.rating_rank ? 'model #' + t.rating_rank : null,
        key === 'home' && !g.neutral ? 'home' : null].filter(Boolean).join(' · ');
      teams.append(
        h('div', { class: 'team' + (isFav ? ' fav' : '') }, badge(t), h('div', { class: 'team-name' }, h('b', null, t.name), h('span', null, meta))),
        h('div', { class: 'proj' }, pr[key] == null ? '–' : String(Math.round(pr[key]))),
        h('div', { class: 'wp' }, pct(chance(key))));
    }
    card.append(teams);
    card.append(h('div', { class: 'wpbar', role: 'img', 'aria-label': 'Win chance: ' + g.away.name + ' ' + pct(chance('away')) + ', ' + g.home.name + ' ' + pct(chance('home')) },
      h('i', { class: likelyHome ? null : 'fav', style: 'flex:' + Math.max(chance('away'), 0.02) }),
      h('i', { class: likelyHome ? 'fav' : null, style: 'flex:' + Math.max(chance('home'), 0.02) })));
    card.append(h('div', { class: 'cap' }, h('span', null, priced ? 'Model’s projected score · market’s chance to win' : 'Projected score and win chance'),
      model.mae_margin != null ? h('span', null, 'typical miss ±' + trim(model.mae_margin) + ' pts') : null));

    const fav = favHome ? g.home : g.away;
    const lines = h('dl', { class: 'lines' });
    const totals = !mlOnly();
    // Moneyline view: say who wins and how likely; the other view keeps the margin and total.
    lines.append(h('dt', null, 'Model'), h('dd', null, totals
      ? fav.name + ' by ' + trim(Math.abs(pr.margin)) + (pr.total != null ? ' · total ' + trim(pr.total) : '')
      : fav.name + ' to win, ' + pct(favHome ? pr.p_home : pr.p_away) + ' · by ' + trim(Math.abs(pr.margin))));
    if (g.market.margin != null) {
      const mFavKey = g.market.margin >= 0 ? 'home' : 'away', mFav = g[mFavKey];
      const mChance = g.ml[mFavKey] ? g.ml[mFavKey].p_market : null;
      let text = Math.abs(g.market.margin) < 0.01 ? 'Pick’em' : mFav.name + (!totals && mChance != null ? ' to win, ' + pct(mChance) + ' ·' : '') + ' by ' + trim(Math.abs(g.market.margin));
      if (totals && espn && espn.total != null) text += ' · total ' + trim(espn.total);
      const moved = Math.abs(g.market.spread_move || 0) >= 1 || (totals && Math.abs(g.market.total_move || 0) >= 1);
      if (espn && espn.spread_open != null && moved) {
        text += ' · opened ' + (espn.spread_open === 0 ? 'PK' : g.home.abbr + ' ' + lineFmt(espn.spread_open)) + (totals && espn.total_open != null ? ', ' + trim(espn.total_open) : '');
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

    const alerts = (g.alerts || []).filter((a) => totals || !/^the total /.test(a));   // a moved total says nothing about who wins
    const notes = [...alerts.map((a) => h('li', null, cap(a))), ...(g.price_notes || []).map((c) => h('li', { class: 'cmp' }, c))];
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
    const sub = h('span', { class: 'mkt-sub' });
    if (side.p_market == null) sub.append(h('span', null, 'model ', b(pct(side.p_model))));
    else if (market === 'ml') {
      sub.append(h('span', null, 'chance ', b(pct(fairOf(side)))), h('span', null, 'model alone ', b(pct(side.p_model))));
      const lean = leanOf(g, market);
      if (lean && lean.key !== p.key && lean.side.edge >= state.picks.rules.edge_min) sub.append(h('span', null, 'the model alone leans ' + g[lean.key].name));
    } else {
      sub.append(h('span', null, 'model ', b(pct(side.p_model))), h('span', null, 'market ', b(pct(side.p_market))), h('span', null, 'gap ', b(edgeFmt(side.edge))));
    }
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
    if (fair != null) facts.push('On its own the model would price it at ' + oddsFmt(fair) + '.');
    if (side.ev != null) facts.push('At a ' + pct(fairOf(side)) + ' chance, this price returns ' + signed(side.ev * 100, 1, '%') + ' per unit staked.');
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

    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null, ['Side', 'Price', 'Model', 'Market', 'Gap'].map((t) => h('th', null, t)))));
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
        h('p', null, 'Checked against the market, ', b(what), ' has a ', b(pct(fairOf(side))), ' chance ' + verb + '. The price implies ', b(pct(side.p_market)),
          ' once the bookmaker’s cut is removed. That gap of ', b(edgeFmt(side.value == null ? side.edge : side.value)), ' is the value.'),
        h('p', { style: 'margin-top:6px' }, 'The pick holds up without each team’s last two games, and the line has not moved against it.' +
          (stake ? ' Suggested stake: ' + stake + ', a quarter of the Kelly amount and never more than 2%.' : '')));
    }
    if (side.label === 'AVOID') {
      return h('div', { class: 'why avoid' },
        h('p', null, 'The model alone sees a gap of ', b(edgeFmt(side.edge)), ' (model ' + pct(side.p_model) + ', market ' + pct(side.p_market) + '), and there is a red flag on top:'),
        h('ul', { style: 'margin-top:6px' }, side.reasons.map((r) => h('li', null, cap(r)))),
        h('p', { style: 'margin-top:6px' }, 'The model does not see injuries, weather or news. When the market disagrees this much, it usually knows something.'));
    }
    if (side.edge >= state.picks.rules.edge_min) {
      return h('div', { class: 'why' },
        h('p', null, 'The model alone gives ', b(what), ' a ', b(pct(side.p_model)), ' chance ' + verb + '; the market says ', b(pct(side.p_market)), '.'),
        h('p', { style: 'margin-top:6px' }, 'That is a disagreement, not a play. In the track record, when the model was this far from the market, the market’s number was the one that held up.'));
    }
    if (market === 'ml') {
      return h('p', { class: 'why' }, b(what), ' is the likelier winner: ', b(pct(fairOf(side))), ' by the market, ' + pct(side.p_model) + ' by the model alone. The price already reflects it, so there is no play.');
    }
    return h('p', { class: 'why' }, 'The model says ', b(pct(side.p_model)), ' and the market ', b(pct(side.p_market)),
      ', a gap of ' + edgeFmt(side.edge) + '. Too close to act on.');
  }

  // ================================================================== PARLAY
  function parlayMath(legs) {
    const out = { n: legs.length, pModel: null, pMarket: null, decimal: null };
    if (!legs.length) return out;
    out.pModel = legs.reduce((acc, l) => acc * fairOf(l.side), 1);   // the chance to quote, not the model's own
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
    if (!mlOnly()) row.append(seg('Market', [['ml', 'Moneyline'], ['spread', 'Spread'], ['total', 'Total']], state.parlayMarket, (v, id) => { state.parlayMarket = v; refocus = id; render(); }));
    if (row.children.length) builder.append(row);
    const list = h('div');
    const market = mlOnly() ? 'ml' : state.parlayMarket;
    for (const g of scopedGames().slice().sort((x, y) => x.date.localeCompare(y.date))) {
      const opts = h('div', { class: 'legopts' });
      for (const key of sideKeys(market)) {
        const leg = legOf(g, market, key);
        if (!leg) continue;
        const id = 'leg-' + g.id + '-' + market + '-' + key;
        opts.append(h('button', { type: 'button', class: 'leg', id, 'aria-pressed': String(isLeg(g.id, market, key)),
          onclick: () => { setLeg(g, market, key); refocus = id; render(); } },
          h('b', null, sideLabel(g, market, key)), h('span', { class: 'o' }, oddsFmt(leg.side.odds)),
          h('span', { class: 'p' }, 'chance ' + pct(fairOf(leg.side)) + ' · model alone ' + pct(leg.side.p_model))));
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
    const chance = saved.p_fair != null ? saved.p_fair : saved.p_model;
    const legChance = (leg) => (leg.p_fair != null ? leg.p_fair : leg.p_model);
    const verdict = h('p', { class: 'verdict' }, 'This ' + legs.length + '-leg ticket has about a ', b(pct(chance, 1)), ' chance, ', b(oneIn(chance)), '.');
    if (Math.abs(saved.p_model - chance) >= 0.005) verdict.append(' The model alone says ' + pct(saved.p_model, 1) + ', but where it has disagreed with the market the market was right.');
    if (saved.implied_prob != null) {
      verdict.append(' The book pays ', b(oddsFmt(decimalToAmerican(saved.decimal))), ', which only breaks even if it hits ', b(pct(saved.implied_prob, 1)), ' of the time.');
    }
    box.append(verdict);
    const kv = h('div', { class: 'kv' });
    kv.append(h('div', null, h('div', { class: 'k' }, 'Chance'), h('div', { class: 'v' }, pct(chance, 1)), h('div', { class: 's' }, 'all ' + legs.length + ' legs win')));
    kv.append(h('div', null, h('div', { class: 'k' }, 'Model alone'), h('div', { class: 'v' }, pct(saved.p_model, 1)), h('div', { class: 's' }, 'before checking it against the market')));
    if (saved.implied_prob != null) {
      kv.append(h('div', null, h('div', { class: 'k' }, 'Needed to break even'), h('div', { class: 'v' }, pct(saved.implied_prob, 1)), h('div', { class: 's' }, 'pays ' + oddsFmt(decimalToAmerican(saved.decimal)) + ' (x' + saved.decimal.toFixed(2) + ')')));
      const ev = chance * saved.decimal - 1;
      kv.append(h('div', null, h('div', { class: 'k' }, 'Expected value'), h('div', { class: 'v ' + (ev >= 0 ? 'pos' : 'neg') }, signed(ev * 100, 0, '%')),
        h('div', { class: 's' }, 'per $1, at that chance')));
    }
    box.append(kv);
    box.append(h('p', { class: 'small' }, 'Why the gap: every leg is priced with the bookmaker’s cut built in, and in a parlay those cuts multiply.'));

    box.append(h('h3', { class: 'h3' }, 'Legs, weakest first'));
    const byId = new Map(allGames().map((g) => [g.id, g]));
    const grid = h('div', { class: 'legs' });
    for (const leg of legs.slice().sort((x, y) => legChance(x) - legChance(y))) {
      const g = byId.get(leg.game_id);
      const weak = leg.team === saved.weakest;
      grid.append(h('div', { class: 'legitem' },
        g ? badge(g[leg.side], true) : h('span'),
        h('div', { class: 'legname' }, h('b', null, leg.team), weak ? h('span', { class: 'flag' }, 'Weakest leg') : null,
          h('small', null, (leg.venue === 'home' ? 'vs ' : 'at ') + leg.opponent)),
        h('div', { class: 'fig' }, pct(legChance(leg)), h('small', null, 'model alone ' + pct(leg.p_model) + ' · ' + oddsFmt(leg.odds))),
        h('div', { class: 'bar' + (weak ? ' weak' : ''), role: 'img', 'aria-label': 'Chance ' + pct(legChance(leg)) }, h('i', { style: 'width:' + clamp(legChance(leg) * 100, 2, 100).toFixed(0) + '%' }))));
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
    grid.append(h('div', null, h('div', { class: 'k' }, 'Chance'), h('div', { class: 'v' }, pct(math.pModel, 1)), h('div', { class: 's' }, 'about ' + oneIn(math.pModel))));
    if (math.decimal != null) {
      const ev = math.pModel * math.decimal - 1;
      grid.append(h('div', null, h('div', { class: 'k' }, 'Pays'), h('div', { class: 'v' }, oddsFmt(american)), h('div', { class: 's' }, 'breaks even at ' + pct(1 / math.decimal, 1))));
      grid.append(h('div', null, h('div', { class: 'k' }, 'Expected value'), h('div', { class: 'v ' + (ev >= 0 ? 'pos' : 'neg') }, signed(ev * 100, 0, '%')), h('div', { class: 's' }, 'per $1, at that chance')));
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
    const weakest = legs.reduce((x, y) => (fairOf(x.side) <= fairOf(y.side) ? x : y));
    body.append(h('p', { class: 'small' }, 'Weakest leg: ' + weakest.label + ' (' + pct(fairOf(weakest.side)) + '). Tap a leg to remove it.'));
    body.append(h('div', { class: 'slip-legs' }, legs.map((l) => h('button', { type: 'button', 'aria-label': 'Remove ' + l.label,
      onclick: () => { setLeg(l.g, l.market, l.key); refocus = 'slip'; render(); } }, l.label + ' ' + oddsFmt(l.side.odds) + ' ×'))));
    body.append(h('div', null, h('button', { type: 'button', class: 'btn ghost', onclick: () => { saveLegs({}); refocus = 'slip'; render(); } }, 'Clear ticket')));
    return box;
  }

  // ============================================================ TRACK RECORD
  const kpiBox = (k, v, sub) => h('div', null, h('div', { class: 'k' }, k), h('div', { class: 'v' }, v), h('div', { class: 's' }, sub || ''));
  // NFL playoff rounds are weeks of their own; they go by name instead of "week 19".
  const weekName = (week) => (S().playoffs[week] ? S().playoffs[week][0] : 'Wk ' + week);
  const weekLong = (week) => (S().playoffs[week] ? S().playoffs[week][1] : 'Week ' + week);
  const bigLabel = (B) => trim(B.big_margin == null ? 14 : B.big_margin) + '+';

  function renderRecord() {
    const B = state.backtest, out = [];
    const sel = B.comparison.find((c) => c.selected) || B.comparison[0];
    const cur = sel.current, mkt = B.market;
    if (!cur || !cur.n) {
      return [h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Track record'),
        h('p', { class: 'empty' }, 'No ' + B.season + ' games have been played yet. The track record appears after the first week.'))];
    }
    const kpi = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'How the model has done in ' + B.season),
      h('p', { class: 'sub' }, 'Every week was predicted using only the games played before it, across ' + cur.n + ' games' +
        (S().college ? ' between top-division teams' : '') + '.' + (cur.wrong != null ? ' Its pick lost ' + cur.wrong + ' of them.' : '') + ' Nothing here was fitted after the fact.'));
    const bigRight = cur.accuracy_big == null ? null : Math.round(cur.accuracy_big * cur.n_big);
    const winner = kpiBox('Picked the winner', rate(cur.accuracy), cur.correct != null ? countText(cur.correct, cur.wrong, cur.ties) : '');
    const clear = kpiBox('When favored by ' + bigLabel(B), rate(cur.accuracy_big), bigRight == null ? plural(cur.n_big, 'game') : countText(bigRight, cur.n_big - bigRight));
    const margin = kpiBox('Typical miss, margin', cur.mae_margin.toFixed(1) + ' pts', mkt ? 'the market: ' + mkt.mae_margin.toFixed(1) : '');
    if (mlOnly()) {
      const favorite = kpiBox('The betting favorite won', mkt && mkt.accuracy != null ? rate(mkt.accuracy) : '–', mkt && mkt.correct != null ? countText(mkt.correct, mkt.wrong) : 'no closing lines yet');
      kpi.append(h('div', { class: 'kv' }, winner, favorite, clear, margin));
    } else {
      kpi.append(h('div', { class: 'kv' }, winner, clear, margin,
        kpiBox('Typical miss, total', cur.mae_total.toFixed(1) + ' pts', mkt ? 'the market: ' + mkt.mae_total.toFixed(1) : '')));
    }
    out.push(kpi);
    const disagree = disagreePanel(B);
    if (disagree) out.push(disagree);
    out.push(bettingPanel(B));
    out.push(comparisonPanel(B, sel));
    out.push(weeksPanel(B));
    out.push(calibrationPanel(B));
    return out;
  }

  // The evidence behind the rule: grouped by how far the model was above the market, what happened.
  function disagreePanel(B) {
    const A = B.anchor || {};
    const markets = (mlOnly() ? ['ml'] : ['ml', 'spread', 'total']).filter((m) => A[m] && A[m].buckets && A[m].buckets.length);
    if (!markets.length) return null;
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'When the model and the market disagree'));
    const d = disagreement(A.ml);
    if (d) {
      box.append(h('p', { class: 'verdict' }, b('The market has been right.'), ' In ' + d.n + ' games where the model gave a team 10+ points more than the market did, the model said ',
        b(pct(d.model)), ', the market said ', b(pct(d.market)), ' and the team won ', b(pct(d.actual)), '.'));
    }
    box.append(h('p', { class: 'sub' }, 'Each row groups past games by how far the model’s chance was above the market’s. If the model knew something the market did not, the last column would follow the model. ' +
      'It follows the market, so the chance shown on this site is the market’s and a gap alone is never a play.'));
    const two = h('div', { class: 'two' });
    for (const m of markets) {
      const body = h('tbody');
      for (const r of A[m].buckets) {
        const range = r.hi >= 1 ? Math.round(r.lo * 100) + '+ pts' : Math.round(r.lo * 100) + '–' + Math.round(r.hi * 100) + ' pts';
        body.append(h('tr', null, h('td', null, range), h('td', { class: 'opt' }, String(r.n)), h('td', null, pct(r.model, 1)), h('td', null, pct(r.market, 1)), h('td', null, b(pct(r.actual, 1)))));
      }
      two.append(h('div', null, h('h3', { class: 'h3' }, MARKET_NAME[m]), h('div', { class: 'scroll' }, h('table', { class: 'mini' },
        h('thead', null, h('tr', null, h('th', null, 'Model ahead by'), h('th', { class: 'opt' }, 'Games'), h('th', null, 'Model'), h('th', null, 'Market'), h('th', null, 'Happened'))), body))));
    }
    box.append(two);
    const w = A.ml && A.ml.weight;
    box.append(h('p', { class: 'small' }, 'Closing lines, seasons ' + B.bets.seasons_with_lines.join(', ') + '. “Model” and “Market” are the chance each gave the side; “Happened” is how often it won, covered or went over.' +
      (w == null ? '' : ' Put as one number, the share of the moneyline gap that came true is ' + signed(w, 2) + ' ± ' + trim(A.ml.error, 2) + ': 1 would mean the model was right, 0 that the market was.')));
    return box;
  }

  function recordTable(block) {
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['Bet', 'W–L', 'Win %', 'ROI', 'Units'].map((t) => h('th', { class: t === 'Win %' ? 'opt' : null }, t)))));
    const body = h('tbody');
    for (const market of (mlOnly() ? ['ml'] : ['ml', 'spread', 'total', 'all'])) {
      for (const label of ['PLAY', 'AVOID']) {
        const r = block[market] && block[market][label];
        if (!r || !r.n) continue;
        body.append(h('tr', { class: market === 'all' ? 'hl' : null },
          h('td', null, (market === 'all' ? 'All' : MARKET_SHORT[market]) + ' · ' + label.toLowerCase()),
          h('td', null, r.won + '–' + r.lost), h('td', { class: 'opt' }, rate(r.win_pct, 0)),
          h('td', { class: r.roi >= 0 ? 'pos' : 'neg' }, signed(r.roi * 100, 1, '%'), r.roi_error == null ? null : h('span', { class: 'pm' }, ' ±' + trim(r.roi_error * 100, 0))),
          h('td', { class: r.units >= 0 ? 'pos' : 'neg' }, signed(r.units, 0, 'u'))));
      }
    }
    table.append(body);
    return table;
  }

  function bettingPanel(B) {
    const ml = mlOnly(), kind = ml ? 'moneyline ' : '';
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'What if you had bet the model’s ' + kind + 'disagreements?'));
    const cur = (B.bets.current || {})[ml ? 'ml' : 'all'], hist = (B.bets.history || {})[ml ? 'ml' : 'all'];
    if (!cur || !cur.PLAY) {
      box.append(h('p', { class: 'empty' }, 'There are no closing lines yet to simulate against.'));
      return box;
    }
    const j = cur.PLAY, hp = hist && hist.PLAY;
    // Inside one standard error of zero, the honest answer is that it is too early to say.
    const even = j.roi_error != null && Math.abs(j.roi) <= j.roi_error;
    const verdict = !j.n
      ? h('p', { class: 'verdict' }, 'No ' + kind + 'bet of that kind has been settled yet in ' + B.season + '.')
      : h('p', { class: 'verdict' }, b(even ? 'Too few bets to say.' : j.roi >= 0 ? 'So far it would have paid.' : 'It would have lost.'), ' Betting one unit every time the model alone saw a ' + kind + 'play in ' + B.season + ' at the closing line went ' + j.won + '–' + j.lost +
        ' for ', b(signed(j.units, 1, ' units')), ' (ROI ' + signed(j.roi * 100, 1, '%') + (j.roi_error == null ? '' : ' ± ' + trim(j.roi_error * 100, 0) + '%') + ').');
    if (hp && hp.n) {
      const read = hp.roi >= 0 ? '.' : hp.roi > -0.065 ? ', roughly the bookmaker’s cut.' : ', which is worse than the bookmaker’s cut alone (about ' + MINUS + '4.5%).';
      verdict.append(' Across every season with lines (' + B.bets.seasons_with_lines.join(', ') + ') the ROI is ', b(signed(hp.roi * 100, 1, '%')), ' on ' + hp.n + ' bets' + read);
    }
    box.append(verdict);
    // Where the flag shows best: moneylines in the moneyline view, college spreads, everything pooled for the NFL.
    const where = ml ? 'ml' : S().college ? 'spread' : 'all';
    const flagged = B.bets.history && B.bets.history[where];
    if (flagged && flagged.AVOID.n && flagged.PLAY.n && flagged.AVOID.roi < flagged.PLAY.roi) {
      box.append(h('p', { class: 'sub' }, 'The AVOID flag earns its keep: ' + (where === 'all' ? '' : MARKET_NAME[where].toLowerCase() + ' ') + 'picks it flagged returned ' + signed(flagged.AVOID.roi * 100, 1, '%') +
        ', against ' + signed(flagged.PLAY.roi * 100, 1, '%') + ' for the plays.'));
    }
    const chart = h('div', { class: 'chart', id: 'chart-units' });
    box.append(h('h3', { class: 'h3' }, 'Running total in units, ' + B.season), chart);
    const weeks = B.weeks.map((w) => (ml ? Object.assign({ week: w.week }, w.ml || { plays: 0 }) : w)).filter((w) => w.plays > 0);
    queueChart(() => drawUnits(chart, weeks));
    const two = h('div', { class: 'two' });
    two.append(h('div', null, h('h3', { class: 'h3' }, B.season + (ml ? '' : ' by bet type')), h('div', { class: 'scroll' }, recordTable(B.bets.current))));
    two.append(h('div', null, h('h3', { class: 'h3' }, 'Seasons ' + B.bets.seasons_with_lines.join(', ')), h('div', { class: 'scroll' }, recordTable(B.bets.history))));
    box.append(two);
    box.append(h('p', { class: 'small' }, 'Here “play” is what the model alone would flag: 5%+ above the market with no red flag. The site no longer treats that as a play. ' +
      'ROI is profit per unit staked; ± is one standard error. The closing line is the last price before kickoff, the hardest one to beat.' +
      (S().college ? '' : ' The NFL red flags were set on this same record, so the gap between plays and flagged picks is a best case.')));
    return box;
  }

  function barRows(rows) {
    const max = Math.max(...rows.map((r) => r.value));
    const bars = h('div', { class: 'bars' });
    for (const r of rows) {
      bars.append(h('span', { class: 'name' + (r.hl ? ' hl' : '') }, r.name + (r.hl ? ' (in use)' : '')),
        h('span', null, h('span', { class: 'bar' + (r.hl ? ' hl' : ''), style: 'width:' + ((r.value / max) * 100).toFixed(1) + '%' })),
        h('span', { class: 'num' }, r.text));
    }
    return bars;
  }
  const shortLabel = (label) => label.replace(/\s*\(.*\)$/, '');

  function comparisonPanel(B, sel) {
    if (mlOnly()) return winnersCompared(B);
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Models compared'),
      h('p', { class: 'sub' }, 'Average miss on the final margin in ' + B.season + ', in points. Lower is better. The lowest-error model is the one in use; the market’s closing line is there for reference.'));
    const rows = B.comparison.map((c) => ({ name: shortLabel(c.label), value: c.current.mae_margin, text: c.current.mae_margin.toFixed(2), hl: c.selected }));
    if (B.market) rows.push({ name: 'Market closing line', value: B.market.mae_margin, text: B.market.mae_margin.toFixed(2), hl: false });
    rows.sort((x, y) => y.value - x.value);
    box.append(barRows(rows));
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['Model', 'Winner', bigLabel(B) + ' pts', 'Margin miss', 'Total miss', B.tuned_on + ' miss'].map((t) => h('th', null, t)))));
    const body = h('tbody');
    for (const c of B.comparison) {
      const a = c.current, hist = c.history || {};
      body.append(h('tr', { class: c.selected ? 'hl' : null },
        h('td', { class: 'wrapcell' }, c.label), h('td', null, rate(a.accuracy)), h('td', null, rate(a.accuracy_big)), h('td', null, a.mae_margin.toFixed(2)),
        h('td', null, a.mae_total == null ? '–' : a.mae_total.toFixed(2)), h('td', null, hist.mae_margin == null ? '–' : hist.mae_margin.toFixed(2))));
    }
    const ref = B.prototype_reference;
    if (ref) {
      body.append(h('tr', null, h('td', { class: 'wrapcell' }, 'Original prototype (reference)'), h('td', null, rate(ref.accuracy, 0)), h('td', null, rate(ref.accuracy_big, 0)),
        h('td', null, '~' + ref.mae_margin), h('td', null, '~' + ref.mae_total), h('td', null, '–')));
    }
    table.append(body);
    const more = h('details', null, h('summary', { class: 'small' }, 'See the full table'), h('div', { class: 'scroll', style: 'margin-top:8px' }, table));
    if (S().college && sel.current_with_fcs) {
      more.append(h('p', { class: 'small', style: 'margin-top:8px' }, 'Counting games against lower-division (FCS) teams, as the prototype did, the model in use picks ' + rate(sel.current_with_fcs.accuracy) +
        ' of winners and ' + rate(sel.current_with_fcs.accuracy_big) + ' when favored by ' + trim(B.big_margin) + ' or more.'));
    }
    box.append(more);
    return box;
  }

  // The moneyline view compares the models on the only thing a moneyline cares about: who won.
  function winnersCompared(B) {
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Who picks more winners'),
      h('p', { class: 'sub' }, 'Share of winners each model picked in ' + B.season + '. Higher is better. The betting favorite is the bar to clear.'));
    const rows = B.comparison.filter((c) => c.current.accuracy != null).map((c) => ({ name: shortLabel(c.label), value: c.current.accuracy, text: rate(c.current.accuracy), hl: c.selected }));
    if (B.market && B.market.accuracy != null) rows.push({ name: 'Betting favorite', value: B.market.accuracy, text: rate(B.market.accuracy), hl: false });
    rows.sort((x, y) => y.value - x.value);
    box.append(barRows(rows));
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null,
      ['Model', 'Right–wrong', 'Winner', bigLabel(B) + ' pts', B.tuned_on + ' winner'].map((t) => h('th', null, t)))));
    const body = h('tbody');
    const line = (label, a, past, hl) => body.append(h('tr', { class: hl ? 'hl' : null },
      h('td', { class: 'wrapcell' }, label), h('td', null, a.correct == null ? '–' : a.correct + '–' + a.wrong), h('td', null, rate(a.accuracy)),
      h('td', null, a.accuracy_big == null ? '–' : rate(a.accuracy_big)), h('td', null, past == null ? '–' : rate(past))));
    for (const c of B.comparison) line(c.label, c.current, (c.history || {}).accuracy, c.selected);
    if (B.market && B.market.accuracy != null) line('Betting favorite', B.market, null, false);
    table.append(body);
    box.append(h('details', null, h('summary', { class: 'small' }, 'See the full table'), h('div', { class: 'scroll', style: 'margin-top:8px' }, table),
      h('p', { class: 'small', style: 'margin-top:8px' }, 'The model in use was chosen for the smallest miss on the final margin, which is not always the one with the most winners.')));
    return box;
  }

  function weeksPanel(B) {
    const ml = mlOnly();
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'Week by week'),
      h('p', { class: 'sub' }, ml
        ? 'Right–wrong counts the games where the model’s pick won or lost. Plays is the record of that week’s moneyline PLAY picks against the closing line.'
        : 'Margin and Total are the average miss in points. Plays is the record of that week’s PLAY picks against the closing line.'));
    const head = ml ? ['Week', 'Winners', 'Right–wrong', 'Plays', 'Units'] : ['Week', 'Winners', 'Margin', 'Total', 'Plays', 'Units'];
    const table = h('table', { class: 'mini' }, h('thead', null, h('tr', null, head.map((t) => h('th', { class: t === 'Total' ? 'opt' : null }, t)))));
    const body = h('tbody');
    for (const w of B.weeks) {
      const bets = ml ? (w.ml || { plays: 0 }) : w;
      const row = h('tr', null, h('td', null, S().playoffs[w.week] ? S().playoffs[w.week][0] : String(w.week)),
        h('td', null, h('span', { class: 'meter', 'aria-hidden': 'true' }, h('i', { style: 'width:' + ((w.accuracy || 0) * 100).toFixed(0) + '%' })), rate(w.accuracy, 0), ml ? null : h('span', { class: 'pm opt' }, ' of ' + w.n)));
      if (ml) row.append(h('td', null, w.correct == null ? '–' : w.correct + '–' + w.wrong));
      else row.append(h('td', null, w.mae_margin.toFixed(1)), h('td', { class: 'opt' }, w.mae_total == null ? '–' : w.mae_total.toFixed(1)));
      row.append(h('td', null, bets.plays ? bets.won + '–' + bets.lost : '–'),
        h('td', { class: bets.plays ? (bets.units >= 0 ? 'pos' : 'neg') : null }, bets.plays ? signed(bets.units, 1, 'u') : '–'));
      body.append(row);
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
    const every = Math.ceil(weeks.length / Math.max(Math.floor(iw / 40), 1));
    weeks.forEach((w, i) => {
      if (i % every === 0 || i === weeks.length - 1) svg.append(s('text', { x: x(i), y: H - 10, 'text-anchor': 'middle' }, weekName(w.week)));
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
        'aria-label': weekLong(w.week) + ': running total ' + signed(w.cumulative, 1) + ' units' });
      const on = (evt) => {
        cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('visibility', 'visible');
        dot.setAttribute('cx', x(i)); dot.setAttribute('cy', y(w.cumulative)); dot.setAttribute('visibility', 'visible');
        const box = svg.getBoundingClientRect();
        showTip(hit, [h('b', null, signed(w.cumulative, 1, 'u') + ' so far'), h('span', null, weekLong(w.week) + ': ' + w.won + '–' + w.lost + ', ' + signed(w.units, 1, 'u'))],
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
    // The left margin holds the tick labels and, beside them, the axis title.
    const W = Math.max(el.clientWidth, 260), H = Math.round(clamp(W * 0.72, 230, 360)), m = { l: 58, r: 14, t: 10, b: 42 };
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
    svg.append(s('text', { x: 11, y: m.t + ih / 2, 'text-anchor': 'middle', transform: 'rotate(-90 11 ' + (m.t + ih / 2) + ')' }, 'How often it won'));
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
    const P = state.picks, B = state.backtest, r = P.rules, m = P.model, sp = S();
    const sec = (title, ...kids) => h('section', { class: 'panel' }, h('h2', { class: 'h2' }, title), kids);
    const p = (...kids) => h('p', null, kids);
    const key = h('div', { class: 'key' },
      pill('PLAY'), h('div', null, 'After the model is checked against the market, the chance still beats the price by ' + (r.edge_min * 100).toFixed(0) + '% or more, with positive expected value and no red flags. The track record has not given the model that credit, so plays are rare.'),
      pill('AVOID'), h('div', null, 'The model alone sees ' + (r.edge_min * 100).toFixed(0) + '%+, but the pick flips when each team’s last 2 games are removed, the model is ' + trim(r.gap_margin) + '+ points away from the market spread (' +
        trim(r.gap_total) + '+ on a total), or the line moved ' + trim(r.move_spread) + '+ points against it since it opened (' + trim(r.move_total) + '+ on a total).'),
      pill('PASS'), h('div', null, 'No value: the model agrees with the market, or disagrees without a record to back it.'));
    const gloss = h('dl', { class: 'gloss' });
    const terms = [
      ['Moneyline', 'A bet on who wins. −300 means you risk $300 to win $100; +240 means you risk $100 to win $240.'],
      ['Spread', sp.spread],
      ['Total', 'A bet on the combined score going over or under a number.'],
      ['Vig', 'The bookmaker’s cut, built into the prices. At −110 on both sides you need to win 52.4% of the time just to break even.'],
      ['Market chance', 'The win chance the prices imply once the vig is removed. It is the fairest read of what the market believes.'],
      ['Gap', 'The model’s chance minus the market’s. A gap is a disagreement. It only counts as value for the share of it that has come true in the past, and so far that share is zero.'],
      ['Stake', 'A quarter of the Kelly amount, which sizes a bet by its edge, and never more than ' + (r.stake_cap * 100).toFixed(0) + '% of your bankroll.'],
      ['Closing line', 'The last price before kickoff. It reflects everything known about the game, which makes it very hard to beat.'],
    ];
    for (const [term, text] of terms) gloss.append(h('div', null, h('dt', null, term), h('dd', null, text)));
    const lines = B.bets && B.bets.seasons_with_lines && B.bets.seasons_with_lines.length ? B.bets.seasons_with_lines.join(', ') : null;
    const out = [
      sec('The model in plain words', h('div', { class: 'prose' },
        p('Each team gets a rating: its average scoring margin, adjusted for how good its opponents were. A team’s rating depends on its opponents’ ratings and theirs on it, so all of them are solved together from the ' +
          m.training_games + ' games played this season.'),
        p('A projection is the difference between two ratings plus home-field advantage. A second set of ratings, built the same way on combined scores, gives the projected total and from it the score.'),
        p('Early in the season a rating leans on last year’s, and blowouts are capped so ' + sp.blowout + ' does not count for too much.'))),
      sec('From a projection to a percentage', h('div', { class: 'prose' },
        p('In testing, real margins landed about ' + trim(m.sigma_margin) + ' points away from the projection on a typical day (' + trim(m.sigma_total) + ' for totals). That spread turns a projected margin into a chance of winning, covering a spread or going over a total.'),
        p('No percentage is ever shown as 100%, and the typical miss sits next to every projection for the same reason.'))),
      sec('Why the chance shown is the market’s', h('div', { class: 'prose' },
        p('The model has its chance of winning and the betting market has another. In past seasons, whenever the two were far apart, the result followed the market’s number: the model does not see injuries, a change of quarterback or the weather, and the market does.'),
        p('So the chance on every card is the market’s, the model’s own number sits beside it, and the gap between them is something to look into, never a bet by itself. The moneyline pick is simply the team more likely to win. ',
          'If the model starts getting its disagreements right, the track record will show it and the rule will give it that much credit.'))),
      sec('What the labels mean', key),
      sec('What the model does not know', h('div', { class: 'prose' },
        h('ul', null,
          sp.blind.map((text) => h('li', null, text)),
          h('li', null, 'Anything the market has already priced in, which is why the closing line misses by less than the model does.')),
        p('Notes and manual point adjustments can be added per game in ', h('code', null, sp.notes), '.'))),
      sec('How the track record is built', h('div', { class: 'prose' },
        p('For week N the model only sees games played before week N. Its settings were tuned on ' + (B.tuned_on || 'past seasons') + ', and ' + B.season +
          ' was scored without touching them. The betting record stakes one unit on every PLAY at the closing line' +
          (lines ? ', for the seasons ESPN kept closing lines for (' + lines + ').' : '.')))),
      sec('Glossary', gloss),
    ];
    if (state.players) out.splice(out.length - 1, 0, playersHow(sec, p));
    return out;
  }

  // ================================================================= PLAYERS
  // NFL player projections next to the market: DraftKings lines and Kalshi's yes/no prices.
  const cents = (x) => (x == null ? '–' : Math.round(x * 100) + '¢');
  const points = (x) => (x == null ? '–' : signed(x * 100, 1));
  const FLAG_PILL = { PLAY: 'play', LEAN: 'pass', CHECK: 'avoid' };
  const flagPill = (label) => h('span', { class: 'pill ' + (FLAG_PILL[label] || 'none') }, label);
  const statName = (stat) => (state.players.stat_names[stat] || stat);
  const shortStat = { pass_yds: 'pass yds', pass_td: 'pass TD', pass_int: 'INT', pass_att: 'att', pass_cmp: 'cmp', rush_yds: 'rush yds', rush_att: 'carries', rec: 'rec', rec_yds: 'rec yds', td: 'TD' };
  const HEADLINE = { QB: ['pass_yds', 'pass_td', 'rush_yds'], RB: ['rush_yds', 'rec_yds', 'td'], WR: ['rec', 'rec_yds', 'td'], TE: ['rec', 'rec_yds', 'td'] };
  // A contract the way Kalshi words it: "Yes · 3+ receptions", "No · 50+ receiving yards".
  const contractText = (f) => (f.side === 'yes' ? 'Yes' : 'No') + ' · ' + f.t + '+ ' + statName(f.stat).toLowerCase();
  const anyTd = (lambda) => 1 - Math.exp(-lambda);   // chance of at least one touchdown from the expected number
  const share = (x) => Math.round(x * 100) + '%';

  function playersGames() { return state.players.games; }
  function currentGame() {
    const games = playersGames();
    return games.find((g) => g.game_id === state.playersGame) || games.find((g) => g.status === 'scheduled') || games[0];
  }
  function openPlayer(gameId, playerId) {
    state.playersGame = gameId;
    state.playerOpen = playerId;
    render();
    const el = document.getElementById('pl-' + playerId);
    if (el) {
      el.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' });
      el.classList.add('flash');
      setTimeout(() => el.classList.remove('flash'), 1800);
    }
  }

  function renderPlayers() {
    const P = state.players;
    if (!P) {
      return [h('section', { class: 'callout' }, h('b', { class: 't' }, 'No player projections yet'),
        h('p', null, 'Generate them with python -m src.players.week after the week’s picks.'))];
    }
    const out = [playersCheck(P)];
    if (P.warnings && P.warnings.length) {
      out.push(h('section', { class: 'callout' }, h('b', { class: 't' }, 'Data warnings'), h('ul', { class: 'alerts' }, P.warnings.map((w) => h('li', null, w)))));
    }
    out.push(flaggedPanel(P));
    out.push(gamePicker(P));
    const game = currentGame();
    if (game) out.push(h('div', { class: 'cards wide' }, game.teams.map((t) => teamSheet(game, t))));
    out.push(playersEvidence(P));
    return out;
  }

  function playersCheck(P) {
    const err = P.backtest.error_season || {}, rec = P.record || {};
    const box = h('section', { class: 'callout', 'aria-label': 'Reality check' }, h('b', { class: 't' }, 'Reality check'));
    const line = h('p', null, 'These are projections for each player, tested on ' + P.backtest.games + ' past games. ');
    if (err.rec_yds && err.rush_yds) {
      line.append('This season they miss by ', b(trim(err.rec_yds.mae) + ' receiving yards'), ' and ', b(trim(err.rush_yds.mae) + ' rushing yards'),
        ' on a typical player, a little better than his season average (' + trim(err.rec_yds.mae_avg) + ' and ' + trim(err.rush_yds.mae_avg) + '). ');
    }
    const n = (rec.plays && rec.plays.n) || 0;
    line.append(n
      ? 'Flagged contracts settled so far: ' + rec.plays.won + ' won and ' + rec.plays.lost + ' lost.'
      : b('No flagged contract has been settled yet'), n ? '' : ': the betting record starts this week, so nothing here is proven against Kalshi’s prices.');
    box.append(line);
    return box;
  }

  function flaggedPanel(P) {
    const by = (label) => P.flagged.filter((f) => f.label === label);
    const plays = by('PLAY'), leans = by('LEAN'), checks = by('CHECK');
    const box = h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', { class: 'h2' }, 'Kalshi contracts the model flags')));
    const rule = P.rule;
    if (!plays.length) {
      box.append(h('p', { class: 'empty' }, 'No contract clears the bar this week: likely to come in (' + (rule.min_prob * 100).toFixed(0) + '%+), and priced ' + (rule.edge_play * 100).toFixed(0) +
        '+ points below what both the model and the DraftKings line say.'));
    } else {
      box.append(h('p', { class: 'sub' }, plural(plays.length, 'contract') + ' likely to come in (' + (rule.min_prob * 100).toFixed(0) + '%+), where the model and the DraftKings line both say Kalshi’s price is ' +
        (rule.edge_play * 100).toFixed(0) + '+ points too low after the fee. Long shots are never flagged. Tap one to see the player.'));
      box.append(h('div', { class: 'plays' }, plays.map((f, i) => flagRow(f, i + 1))));
    }
    const fold = (rows, one, many, text) => (rows.length ? h('details', null, h('summary', { class: 'h3' }, plural(rows.length, one, many)),
      h('p', { class: 'small', style: 'margin-top:8px' }, text), h('div', { class: 'plays' }, rows.slice(0, 25).map((f) => flagRow(f, null)))) : null);
    box.append(fold(leans, 'lean', 'leans', 'Likely contracts the model thinks are cheap, without a DraftKings line that fully backs it or with too few contracts traded for the price to mean much.'));
    box.append(fold(checks, 'contract to check first', 'contracts to check first', 'The model is far from the market here, or the player’s situation changed. A big gap usually means the market knows something: read the news before touching these.'));
    return box;
  }

  function flagRow(f, rank) {
    let ctx = f.name + ' · ' + f.team + ' · ' + f.game + ' · ' + kickFmt(f.kickoff) + ' · projection ' + trim(f.proj, 1) + (f.dk != null ? ' (DraftKings line ' + trim(f.dk, 1) + ')' : '') +
      (f.p_dk != null ? ' · DraftKings implies ' + pct(f.p_dk) : '');
    if (!rank && f.reasons.length) ctx += ' · ' + f.reasons[0];
    return h('button', { type: 'button', class: 'playrow', onclick: () => openPlayer(f.game_id, f.athlete_id) },
      rank ? h('span', { class: 'rankno' }, String(rank)) : flagPill(f.label),
      h('span', { class: 'what' }, contractText(f), ' ', h('span', { class: 'price' }, '(' + cents(f.price) + ')')),
      h('span', { class: 'edge' }, points(f.edge), h('small', null, pct(f.p) + ' vs ' + cents(f.price))),
      h('span', { class: 'ctx' }, ctx));
  }

  function gamePicker(P) {
    const game = currentGame();
    const select = h('select', { id: 'players-game', 'aria-label': 'Game', onchange: (e) => { state.playersGame = e.target.value; state.playerOpen = null; refocus = 'players-game'; render(); } },
      playersGames().map((g) => h('option', { value: g.game_id, selected: game && g.game_id === game.game_id },
        g.away + ' @ ' + g.home + ' · ' + kickFmt(g.kickoff) + (g.status === 'scheduled' ? '' : ' · ' + (g.status === 'final' ? 'played' : 'started')))));
    const note = !game ? null : game.status !== 'scheduled' ? 'This game has started: the numbers are what the model said before kickoff.'
      : !game.has_kalshi && !game.has_dk ? 'No player lines or Kalshi prices for this game yet.'
      : !game.has_kalshi ? 'No Kalshi prices for this game yet.' : null;
    return h('section', { class: 'controls', 'aria-label': 'Choose a game' },
      h('div', { class: 'controls-row' }, h('label', { class: 'h3', for: 'players-game' }, 'Game'), h('div', { class: 'pick' }, select)),
      h('p', { class: 'small' }, 'Projections and prices as of ' + stamp(P.generated_at) + '.' + (note ? ' ' + note : '')));
  }

  function teamSheet(game, t) {
    const opp = game.teams.find((x) => x.id !== t.id);
    const card = h('article', { class: 'panel game' });
    const tags = h('div', { class: 'tags' });
    if (t.implied_points != null) tags.append(h('span', { class: 'tag' }, 'Line implies ' + trim(t.implied_points, 1) + ' pts'));
    card.append(h('div', { class: 'game-top' },
      h('div', { class: 'team fav' }, badge(t), h('div', { class: 'team-name' }, h('b', null, t.name), h('span', null, (t.side === 'home' ? 'home vs ' : 'at ') + opp.name))), tags));
    const lines = h('dl', { class: 'lines' });
    lines.append(h('dt', null, 'Quarterback'), h('dd', null, t.qb || 'not known'));
    lines.append(h('dt', null, 'Plan'), h('dd', null, trim(t.pass_att, 0) + ' passes for ' + t.pass_yds + ' yds · ' + trim(t.rush_att, 0) + ' runs for ' + t.rush_yds + ' yds'));
    lines.append(h('dt', null, opp.abbr + ' defense'), h('dd', null,
      'vs the pass: ' + defText(t.opp_pass, 'yards per pass') + ' · vs the run: ' + defText(t.opp_rush, 'yards per carry')));
    card.append(lines);
    const alerts = [];
    if (t.qb_change) alerts.push(h('li', null, t.qb + ' starts instead of ' + t.usual_qb + ': fewer passes and a wider margin of error for everyone.'));
    for (const m of t.missing) {
      const part = m.pos === 'RB' ? share(m.carry_share) + ' of the carries' : share(m.target_share) + ' of the targets';
      alerts.push(h('li', null, m.name + ' (' + m.pos + ') is out: he had ' + part + ', now split among the others.'));
    }
    const others = t.out.filter((o) => o.name);
    if (others.length) alerts.push(h('li', { class: 'cmp' }, 'Also out: ' + others.map((o) => o.name + ' (' + o.pos + ')').join(', ')));
    if (alerts.length) card.append(h('ul', { class: 'alerts' }, alerts));
    const list = h('div', { class: 'markets' });
    for (const p of t.players) list.append(playerRow(game, t, p));
    card.append(list);
    return card;
  }
  function defText(x, unit) {
    if (Math.abs(x) < 0.01) return 'average';
    return Math.abs(x * 100).toFixed(0) + '% ' + (x > 0 ? 'more' : 'fewer') + ' ' + unit + ' than average';
  }

  function playerRow(game, t, p) {
    const head = HEADLINE[p.pos] || HEADLINE.WR;
    const bits = [];
    for (const stat of head) {
      const e = p.stats[stat];
      if (!e) continue;
      bits.push(stat === 'td' ? h('span', null, 'TD ', b(pct(anyTd(e.proj)))) : h('span', null, b(trim(e.proj, e.proj < 10 ? 1 : 0)), ' ' + shortStat[stat]));
    }
    const flags = Object.values(p.stats).map((e) => e.best).filter(Boolean);
    const top = flags.sort((x, y) => (x.label === 'PLAY' ? -1 : 1) - (y.label === 'PLAY' ? -1 : 1) || y.edge - x.edge)[0];
    const det = h('details', { class: 'mkt pl' + (top && top.label === 'PLAY' ? ' is-play' : ''), id: 'pl-' + p.id, open: state.playerOpen === p.id });
    det.addEventListener('toggle', () => { if (det.open) state.playerOpen = p.id; else if (state.playerOpen === p.id) state.playerOpen = null; });
    det.append(h('summary', null,
      h('span', { class: 'mkt-type' }, h('span', null, p.pos + (p.status ? ' · ' + p.status : '') + (p.flags.includes('returning') ? ' · back after missing games' : ''))),
      h('span', { class: 'mkt-pick' }, p.name), top ? flagPill(top.label) : h('span', null),
      h('span', { class: 'mkt-sub' }, bits)));
    const body = h('div', { class: 'mkt-body' });
    const rows = [];
    for (const [stat, e] of Object.entries(p.stats)) {
      // Under the name: where half of his games land, or his chance to score.
      const hint = stat === 'td' ? pct(anyTd(e.proj)) + ' to score' : e.lo != null && e.hi != null ? 'half the time ' + trim(e.lo, 0) + '–' + trim(e.hi, 0) : null;
      rows.push(h('tr', null,
        h('td', { class: 'cell2 wrapcell' }, LONG_STAT[stat] || statName(stat), hint ? h('small', null, hint) : null),
        h('td', null, b(trim(e.proj, e.proj < 10 ? 2 : 1))),
        h('td', null, e.avg == null ? '–' : trim(e.avg, 1)),
        h('td', { class: 'cell2' }, e.dk == null ? '–' : trim(e.dk, 1), e.dk != null && e.p_over_dk != null ? h('small', null, pct(e.p_over_dk) + ' over') : null)));
    }
    body.append(h('div', { class: 'scroll' }, h('table', { class: 'mini' },
      h('thead', null, h('tr', null, h('th', null, 'Statistic'), h('th', null, 'Model'), h('th', null, 'Avg'), h('th', null, 'DraftKings'))),
      h('tbody', null, rows))));
    for (const [stat, e] of Object.entries(p.stats)) {
      if (!e.ladder.length) continue;
      if (e.best) {
        const cls = e.best.label === 'PLAY' ? ' play' : e.best.label === 'CHECK' ? ' avoid' : '';
        body.append(h('div', { class: 'why' + cls }, flagPill(e.best.label), ' ', b(contractText({ side: e.best.side, t: e.best.t, stat }) + ' at ' + cents(e.best.price)),
          ': the model gives it ' + pct(e.best.p) + ', which leaves ' + points(e.best.edge) + ' points after the fee.' +
          (e.best.reasons.length ? ' ' + cap(e.best.reasons.join('; ')) + '.' : '')));
      }
      body.append(h('details', { class: 'ladder' }, h('summary', { class: 'small' }, 'Kalshi prices for ' + statName(stat).toLowerCase() + ' (' + e.ladder.length + ')'),
        h('div', { class: 'scroll' }, h('table', { class: 'mini' },
          h('thead', null, h('tr', null, h('th', null, 'Contract'), h('th', null, 'Model'), h('th', null, 'Yes'), h('th', null, 'No'), h('th', null, 'Edge yes'), h('th', null, 'Edge no'))),
          h('tbody', null, e.ladder.map((r) => h('tr', null,
            h('td', null, r.t + '+'), h('td', null, pct(r.p)), h('td', null, cents(r.yes_ask)), h('td', null, cents(r.no_ask)),
            h('td', { class: r.edge_yes >= P_EDGE ? 'pos' : null }, points(r.edge_yes)), h('td', { class: r.edge_no >= P_EDGE ? 'pos' : null }, points(r.edge_no)))))))));
    }
    if (p.games < 4) body.append(h('p', { class: 'small' }, 'Only ' + plural(p.games, 'game') + ' this season: his role is not established, so treat these numbers as rough.'));
    det.append(body);
    return det;
  }
  const P_EDGE = 0.03;
  const LONG_STAT = { td: 'Touchdowns', pass_int: 'Interceptions' };

  function playersEvidence(P) {
    const E = P.backtest, ing = E.ingredients || {};
    const box = h('section', { class: 'panel' }, h('h2', { class: 'h2' }, 'How far to trust these'));
    const rows = [];
    for (const stat of ['pass_yds', 'pass_td', 'rush_yds', 'rush_att', 'rec', 'rec_yds']) {
      const e = (E.error_season || {})[stat] || (E.error || {})[stat];
      if (!e) continue;
      rows.push(h('tr', null, h('td', null, statName(stat)), h('td', null, b(trim(e.mae, 2))), h('td', null, trim(e.mae_avg, 2)),
        h('td', { class: e.mae < e.mae_avg ? 'pos' : 'neg' }, signed((1 - e.mae / e.mae_avg) * 100, 0, '%')), h('td', { class: 'opt' }, String(e.n))));
    }
    const left = h('div', null, h('h3', { class: 'h3' }, 'Typical miss, ' + P.season),
      h('div', { class: 'scroll' }, h('table', { class: 'mini' },
        h('thead', null, h('tr', null, h('th', null, 'Statistic'), h('th', null, 'Model'), h('th', null, 'Season avg'), h('th', null, 'Gain'), h('th', { class: 'opt' }, 'Player-games'))),
        h('tbody', null, rows))),
      h('p', { class: 'small' }, 'How far the projection landed from what the player did, against simply using his average per game. Every week was projected with only what was known before it.'));
    const facts = h('ul', null);
    const run = ing.absent_runners && ing.absent_runners.rush_att, runY = ing.absent_runners && ing.absent_runners.rush_yds;
    if (run && runY) {
      facts.append(h('li', null, b('Who is out matters most. '), 'When a team is missing a main running back, the ones left carry ' + trim(-run.bias_without, 0) + ' more times and gain ' +
        trim(-runY.bias_without, 0) + ' more yards than their average says. Knowing who is out cuts the miss on carries from ' + trim(run.mae_without, 1) + ' to ' + trim(run.mae, 1) + '.'));
    }
    const recv = ing.absent_receivers && ing.absent_receivers.rec_yds;
    if (recv) facts.append(h('li', null, 'When a main receiver is out, the others gain about ' + trim(-recv.bias_without, 0) + ' receiving yards each. It happens in ' + share(ing.share_absent_receivers) + ' of team games.'));
    const opp = ing.opponent && ing.opponent.pass_yds;
    if (opp) {
      facts.append(h('li', null, b('The opponent matters less than it seems. '), 'Dropping the defense adjustment moves the miss on passing yards from ' + trim(opp.mae, 1) + ' to ' + trim(opp.mae_without, 1) +
        ': one game is too noisy for the defense to show.'));
    }
    const cal = E.calibration || {};
    const gaps = Object.values(cal).map((c) => c.gap);
    if (gaps.length) facts.append(h('li', null, 'The percentages hold up: across ' + Object.values(cal).reduce((a, c) => a + c.n, 0).toLocaleString('en-US') + ' threshold checks, what the model called 60% happened about 60% of the time (off by ' +
      (Math.max(...gaps) * 100).toFixed(0) + ' point at most on these games, 1 to 4 on a season it had not seen).'));
    facts.append(h('li', null, 'What it cannot know: a game plan, the weather, a player on a snap count. When the model is far from the market, assume the market knows.'));
    box.append(h('div', { class: 'two' }, left, h('div', { class: 'prose' }, h('h3', { class: 'h3' }, 'What moves the numbers'), facts)));
    return box;
  }


  function playersHow(sec, p) {
    const r = state.players.rule;
    return sec('How the player projections work', h('div', { class: 'prose' },
      p('Each player’s numbers are built from the top down, in the order things happen on the field.'),
      h('ul', null,
        h('li', null, b('The team. '), 'How many passes and runs it calls and how efficient it is, adjusted for the defense it faces.'),
        h('li', null, b('The quarterback. '), 'The passing numbers belong to whoever starts. A backup changes every receiver’s projection too.'),
        h('li', null, b('The role. '), 'Each player’s share of the targets and carries in his recent games.'),
        h('li', null, b('Who is available. '), 'Shares are split among the players who will play. When a teammate is out, the man who replaces him takes most of the vacancy.'),
        h('li', null, b('His own efficiency. '), 'Catch rate, yards per target and per carry, pulled toward the average of his position when he has had few chances.')),
      p('A projection is an average. To price a contract such as “250+ passing yards”, the model uses how real results have spread around projections of that size in the backtest.'),
      p('Only likely things are flagged. A contract under ' + cents(r.min_price) + ', or one the model gives less than ' + (r.min_prob * 100).toFixed(0) +
        '%, is a long shot: on Kalshi those win less often than their price says, and a few points of error in the model is a third of what they are worth.'),
      p(b('PLAY'), ': a likely contract where the model and the DraftKings line both leave ' + (r.edge_play * 100).toFixed(0) + '+ points after Kalshi’s fee, the price is between ' + cents(r.min_price) + ' and ' + cents(r.max_price) +
        ' with bid and ask close together, enough contracts have traded, and the player is not questionable, just back from injury or playing with a new quarterback. ',
        b('LEAN'), ': the model is alone. ', b('CHECK'), ': the model is more than ' + (r.max_gap * 100).toFixed(0) + ' points from the market, which usually means the market knows something.'),
      p('The model alone is not enough for a PLAY because of what the team picks showed: when the model disagreed with the market, the market was right.')));
  }

  // ================================================================== SOCCER
  // European cups: the sportsbook's odds with the margin removed, next to Kalshi's prices.
  const COMP_NAME = { ucl: 'Champions League', uel: 'Europa League', uecl: 'Conference League' };
  const OUTCOMES = ['home', 'draw', 'away'];
  const LIKELY_PRICE = 0.30;   // below it a contract is a long shot and never listed as a good price
  const upcoming = (m) => m.status === 'scheduled';
  const boardMatches = () => state.board.matches.filter((m) => state.soccerComp === 'all' || m.competition === state.soccerComp);

  function renderSoccer() {
    const B = state.board;
    const out = [h('section', { class: 'callout', 'aria-label': 'Read this first' }, h('b', { class: 't' }, 'Read this first'),
      h('p', null, b('There is no model here. '), 'The percentages are the sportsbook’s odds with its margin taken out: what the betting market thinks. ',
        'Next to them is what Kalshi charges for each result, including “No” on a team, which wins when that team draws or loses. ',
        'A price below the market’s chance means the two disagree. It is not a tested edge.'))];
    if (B.warnings && B.warnings.length) {
      out.push(h('section', { class: 'callout' }, h('b', { class: 't' }, 'Data warnings'), h('ul', { class: 'alerts' }, B.warnings.map((w) => h('li', null, w)))));
    }
    out.push(bargains(B));
    out.push(soccerControls(B));
    const list = h('div', { class: 'stack' });
    fillMatches(list);
    out.push(list);
    return out;
  }

  function bargains(B) {
    const rows = [];
    for (const m of B.matches) {
      if (!upcoming(m)) continue;
      for (const c of m.contracts) if (c.price >= LIKELY_PRICE && c.price <= 0.95) rows.push({ c, m });
    }
    rows.sort((x, y) => y.c.edge - x.c.edge);
    const box = h('section', { class: 'panel' }, h('div', { class: 'panel-head' }, h('h2', { class: 'h2' }, 'Kalshi against the sportsbook')));
    if (!rows.length) {
      box.append(h('p', { class: 'empty' }, 'No match to play has both sportsbook odds and Kalshi prices right now.'));
      return box;
    }
    const best = rows[0].c.edge, n = rows.length;
    box.append(h('p', { class: 'verdict' }, best < 0.01
      ? ['Nothing is cheap. Of ' + n + ' contracts on results that are not long shots, the best one leaves ', b(signed(best * 100, 1, '¢')), ' after Kalshi’s fee. Kalshi is priced where the sportsbooks are.']
      : [plural(rows.filter((r) => r.c.edge >= 0.01).length, 'contract') + ' cost less than the sportsbook’s fair price after the fee. The widest gap is ', b(signed(best * 100, 1, '¢')), '.']));
    box.append(h('div', { class: 'plays' }, rows.slice(0, 6).map(({ c, m }) => h('button', { type: 'button', class: 'playrow', onclick: () => goToMatch(m.id) },
      h('span', { class: 'rankno' }, cents(c.price)),
      h('span', { class: 'what' }, c.text),
      h('span', { class: 'edge' + (c.edge > 0 ? ' pos' : '') }, signed(c.edge * 100, 1, '¢'), h('small', null, 'worth ' + (c.fair * 100).toFixed(1) + '¢')),
      h('span', { class: 'ctx' }, m.home.short + ' v ' + m.away.short + ' · ' + COMP_NAME[m.competition] + ' · ' + kickFmt(m.kickoff))))));
    box.append(h('p', { class: 'small' }, 'Price is what you pay now. “Worth” is the market’s chance of that result. The gap is what is left after a fee of about ' +
      (B.fee.taker_rate * 100).toFixed(0) + '% × price × (1 − price) per contract. Contracts under ' + cents(LIKELY_PRICE) +
      ' are left out of this list: they are long shots, and on Kalshi those win less often than their price says.'));
    return box;
  }

  function goToMatch(id) {
    const card = document.getElementById('m-' + id);
    if (!card) return;
    card.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
    card.classList.add('flash');
    setTimeout(() => card.classList.remove('flash'), 1800);
  }

  function soccerControls(B) {
    const options = [['all', 'All']].concat(B.competitions.filter((c) => c.matches).map((c) => [c.key, c.name.replace(' League', '')]));
    const row = h('div', { class: 'controls-row' }, seg('Competition', options, state.soccerComp, (v, id) => setState('soccerComp', v, id)),
      h('label', { class: 'toggle', for: 'played' },
        h('input', { id: 'played', type: 'checkbox', checked: state.soccerPlayed, onchange: (e) => setState('soccerPlayed', e.target.checked, 'played') }), 'Show played matches'));
    return h('section', { class: 'controls', 'aria-label': 'Filter matches' }, row);
  }

  function fillMatches(container) {
    const matches = boardMatches().filter((m) => state.soccerPlayed || m.status !== 'final').slice().sort((x, y) => x.kickoff.localeCompare(y.kickoff));
    const out = [];
    if (!matches.length) out.push(h('p', { class: 'empty' }, 'No matches to play in the next few days. Tick “Show played matches” to see the last results.'));
    let current = null, grid = null;
    for (const m of matches) {
      const key = dayKey(new Date(m.kickoff));
      if (key !== current) {
        current = key;
        const count = matches.filter((x) => dayKey(new Date(x.kickoff)) === key).length;
        out.push(h('div', { class: 'day' }, h('h2', { class: 'h2' }, dayLabel(m.kickoff)), h('span', { class: 'small' }, plural(count, 'match', 'matches'))));
        grid = h('div', { class: 'cards' });
        out.push(grid);
      }
      grid.append(matchCard(m));
    }
    container.replaceChildren(...out);
  }

  function matchCard(m) {
    const card = h('article', { class: 'panel game', id: 'm-' + m.id });
    const played = m.status === 'final', live = m.status === 'in_progress';
    card.append(h('div', { class: 'game-top' }, h('span', { class: 'kick' }, played ? dateFmt(m.kickoff) + ' · full time' : live ? 'Being played' : timeFmt(m.kickoff)),
      h('div', { class: 'tags' }, h('span', { class: 'tag' }, COMP_NAME[m.competition]))));
    const fair = m.fair;
    const favKey = fair ? OUTCOMES.reduce((a, k) => (fair[k] > fair[a] ? k : a), 'home') : null;
    const teams = h('div', { class: 'teams' });
    for (const key of ['home', 'away']) {
      const t = m[key];
      const meta = [t.rank ? '#' + t.rank + ' of 36' : null, t.record ? t.record + ' (W-D-L)' : null, key === 'home' ? 'home' : null].filter(Boolean).join(' · ');
      teams.append(
        h('div', { class: 'team' + (favKey === key ? ' fav' : '') }, h('span', { class: 'badge plain', 'aria-hidden': 'true' }, t.abbr || ''),
          h('div', { class: 'team-name' }, h('b', null, t.name), h('span', null, meta))),
        h('div', { class: 'proj' }, t.score == null ? '' : String(t.score)),
        h('div', { class: 'wp' }, fair ? pct(fair[key]) : '–'));
    }
    card.append(teams);
    if (fair) {
      card.append(h('div', { class: 'wpbar three', role: 'img', 'aria-label': 'Market chance: ' + m.home.short + ' ' + pct(fair.home) + ', draw ' + pct(fair.draw) + ', ' + m.away.short + ' ' + pct(fair.away) },
        OUTCOMES.map((k) => h('i', { class: k === favKey ? 'fav' : k === 'draw' ? 'draw' : null, style: 'flex:' + Math.max(fair[k], 0.02) }))));
      const open = m.fair_open;
      const moved = open && OUTCOMES.some((k) => Math.abs(open[k] - fair[k]) >= 0.03);
      card.append(h('div', { class: 'cap' }, h('span', null, 'Market chance · draw ' + pct(fair.draw)),
        moved ? h('span', null, 'opened ' + OUTCOMES.map((k) => Math.round(open[k] * 100)).join(' / ')) : h('span', null, (m.book && m.book.provider) || '')));
    } else if (!played) {
      card.append(h('p', { class: 'small' }, 'No sportsbook odds for this match yet, so there is nothing to compare Kalshi with.'));
    }
    if (m.note) card.append(h('p', { class: 'note' }, m.note));
    if (played || !fair) return card;
    if (!m.contracts.length) {
      card.append(h('p', { class: 'small' }, 'No Kalshi prices for this match.'));
      return card;
    }
    const find = (outcome, side) => m.contracts.find((c) => c.outcome === outcome && c.side === side);
    const cell = (c) => (c ? h('td', { class: 'cell2' }, b(cents(c.price)), h('small', { class: c.edge > 0 ? 'pos' : null }, signed(c.edge * 100, 1, '¢'))) : h('td', null, '–'));
    const label = { home: m.home.short + ' win', draw: 'Draw', away: m.away.short + ' win' };
    card.append(h('div', { class: 'scroll' }, h('table', { class: 'mini' },
      h('thead', null, h('tr', null, h('th', null, 'Result'), h('th', null, 'Market'), h('th', null, 'Kalshi yes'), h('th', null, 'Kalshi no'))),
      h('tbody', null, OUTCOMES.map((k) => h('tr', null, h('td', null, label[k]), h('td', null, pct(fair[k], 1)), cell(find(k, 'yes')), cell(find(k, 'no'))))))));
    card.append(h('p', { class: 'small' }, '“No” on a team wins if it draws or loses. Under each price: what is left after the fee, in cents per contract.'));
    return card;
  }

  function renderSoccerHow() {
    const B = state.board;
    const sec = (title, ...kids) => h('section', { class: 'panel' }, h('h2', { class: 'h2' }, title), kids);
    const p = (...kids) => h('p', null, kids);
    return [
      sec('What this page is', h('div', { class: 'prose' },
        p('The football pages have a model that was tested on past seasons. This one does not: nothing in this project has been tested on European cup matches, so it shows the market instead of a prediction.'),
        p('For every Champions League, Europa League and Conference League match it puts two things side by side: what the sportsbook thinks, and what Kalshi charges.'))),
      sec('From odds to a chance', h('div', { class: 'prose' },
        p('A sportsbook’s three prices (home, draw, away) add up to more than 100%: the extra is its margin. Taking it out leaves the market’s chance of each result. ',
          'The margin sits mostly on the long shots, so it is removed with Shin’s method instead of an even split, which would overstate the underdog.'))),
      sec('Yes, No and the draw', h('div', { class: 'prose' },
        p('On Kalshi each result is its own yes/no contract that pays $1. “Yes” on a team wins only if it wins inside 90 minutes plus stoppage time. ',
          b('“No” on a team wins if it draws or loses'), ', so it is the same as backing the other team and the draw together.'),
        p('That does not make “No” safer. “No” on a 14% underdog costs about 86¢ to win 14¢: it has to come in 86 times out of 100 just to break even, and the market says it does exactly that often.'))),
      sec('What “left after the fee” means', h('div', { class: 'prose' },
        p('A contract is worth its chance: a result the market gives 42% is worth 42¢. The number under each price is that value, minus the price, minus Kalshi’s fee (about ' +
          (B.fee.taker_rate * 100).toFixed(0) + '% × price × (1 − price) per contract, so 1.7¢ at 50¢ and almost nothing near 0 or 100).'),
        p('When it is positive, Kalshi is charging less than the sportsbook implies. That is a disagreement between two markets, and either one can be the one that is wrong. Lineups come out an hour before kickoff and move both.'))),
    ];
  }

  // -------------------------------------------------------------------- boot
  const VIEWS = { picks: renderPicks, players: renderPlayers, parlay: renderParlay, record: renderRecord, how: renderHow };
  const BOARD_VIEWS = { picks: renderSoccer, how: renderSoccerHow };
  const viewsOf = (sport) => (SPORTS[sport].kind === 'board' ? BOARD_VIEWS : VIEWS);
  const hasTab = (sport, name) => SPORTS[sport].tabs.some(([view]) => view === name);
  // "#nfl-players" names the section and the tab; a bare "#record" keeps the last section used.
  function route() {
    const raw = (location.hash || '').replace(/^#/, '');
    const [head, ...rest] = raw.split('-');
    const named = SPORTS[head] ? head : null;
    const name = named ? rest.join('-') : raw;
    const saved = store.get('sport', DEFAULT_SPORT);
    const sport = named || (SPORTS[saved] ? saved : DEFAULT_SPORT);
    return { sport, view: hasTab(sport, name) ? name : 'picks' };
  }

  // Header, section switch and tabs follow the section and tab in the URL.
  function chrome(name) {
    const sp = S(), P = state.picks, B = state.board;
    document.documentElement.setAttribute('data-sport', state.sport);
    document.title = sp.title;
    document.getElementById('hdr-title').textContent = sp.title;
    document.querySelectorAll('.sports a').forEach((a) => {
      const other = a.dataset.sport;
      a.setAttribute('href', '#' + other + '-' + (hasTab(other, name) ? name : 'picks'));
      if (other === state.sport) a.setAttribute('aria-current', 'true');
      else a.removeAttribute('aria-current');
    });
    document.querySelector('.tabs-inner').replaceChildren(...sp.tabs.map(([tab, text]) =>
      h('a', { href: '#' + state.sport + '-' + tab, 'data-view': tab, 'aria-current': tab === name ? 'page' : null }, text,
        tab === 'parlay' ? h('span', { class: 'count', id: 'parlay-count', hidden: true }) : null)));
    const week = document.getElementById('hdr-week'), meta = document.getElementById('hdr-meta'), foot = document.getElementById('foot-gen');
    const note = document.getElementById('foot-note');
    note.textContent = sp.kind === 'board'
      ? 'These are the betting market’s numbers, not a prediction. No result is a sure thing.'
      : 'These are statistical projections and they miss often. No pick is a sure thing: the team model knows nothing about weather or motivation.';
    if (sp.kind === 'board') {
      week.textContent = 'Champions · Europa · Conference';
      foot.textContent = B ? 'Odds: DraftKings through ESPN’s public API. Prices: Kalshi’s public market data.' : '';
      meta.textContent = B ? 'Updated ' + stamp(B.generated_at) + ' · ' + plural(B.matches.filter(upcoming).length, 'match', 'matches') + ' to play'
        : state.error ? 'No data' : 'Loading…';
      return;
    }
    if (!P) {
      week.textContent = sp.name;
      meta.textContent = state.error ? 'No data' : 'Loading…';
      foot.textContent = '';
      return;
    }
    week.textContent = P.week_label + ' · ' + P.season;
    const n = allGames().length, mine = mineCount();
    meta.textContent = 'Updated ' + stamp(P.generated) + ' · ' + plural(n, 'game') + (mine ? ', ' + mine + ' tracked' : '');
    foot.textContent = 'Scores, lines and player numbers: ESPN’s public API. ' + (state.players ? 'Player prices: Kalshi. ' : '') + 'Model: ' + P.model.label + '.';
  }

  function render() {
    hideTip();
    const name = route().view;
    chrome(name);
    const sp = S(), ready = sp.kind === 'board' ? state.board : state.picks;
    if (state.error) {
      view.replaceChildren(h('section', { class: 'callout' }, h('b', { class: 't' }, 'This section could not be loaded'),
        h('p', null, state.error), h('p', null, 'Generate the data with ' + sp.command + ' and open the site through a web server (python -m http.server -d site), not as a local file.')));
      updateParlayCount();
      return;
    }
    if (!ready) {
      view.replaceChildren(h('p', { class: 'loading' }, 'Loading ' + sp.name + '…'));
      updateParlayCount();
      return;
    }
    pendingCharts = [];
    view.replaceChildren(...viewsOf(state.sport)[name]());
    flushCharts();
    updateParlayCount();
    if (refocus) {
      const el = document.getElementById(refocus);
      if (el) el.focus({ preventScroll: true });
      refocus = null;
    }
  }

  // Point the page at one section: its data (if already loaded) and its own parlay ticket.
  function activate(sport) {
    const changed = sport !== state.sport;
    state.sport = sport;
    const data = DATA[sport] || {};
    state.picks = data.picks || null;
    state.backtest = data.backtest || null;
    state.players = data.players || null;
    state.board = data.board || null;
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
      state.playersGame = null;
      state.playerOpen = null;
    }
  }

  async function loadSport(sport) {
    if (DATA[sport]) return;
    const spec = SPORTS[sport], got = {};
    try {
      await Promise.all(Object.entries(spec.files).map(async ([key, url]) => {
        try {
          const resp = await fetch(url, { cache: 'no-store' });
          if (!resp.ok) throw new Error(url + ' answered HTTP ' + resp.status);
          got[key] = await resp.json();
        } catch (err) {
          if (!(spec.optional || []).includes(key)) throw err;   // an optional file only hides its tab's content
        }
      }));
      DATA[sport] = got;
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
