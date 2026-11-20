# CFB Picks

Sistema semanal de predicciones de college football: baja todos los juegos FBS
del API público de ESPN, ajusta un modelo de ratings, lo valida con un backtest
walk-forward y publica los picks en un sitio estático.

## Qué tan bueno es (y qué tan bueno no es)

Backtest walk-forward de la temporada 2026, 635 juegos FBS vs FBS que el modelo
nunca vio al calibrarse:

| Modelo | Ganador | Con 14+ pts | Error margen | Error total |
|---|---|---|---|---|
| Baseline (puntos vs % de victorias del rival) | 65.7% | 79.5% | 14.72 | 13.08 |
| **SRS (elegido)** | **70.4%** | **90.6%** | **12.83** | **13.01** |
| Elo con margen de victoria | 68.7% | 83.3% | 13.29 | n/a |
| Línea de cierre del mercado | 74.6% | n/a | 11.72 | 12.42 |
| Prototipo original (referencia) | 73% | 92% | ~15.5 | ~20 |

Contando los juegos contra FCS, como hacía el prototipo, el SRS acierta 74.4%
y 92.5% con 14+ puntos de ventaja.

Dos cosas que conviene tener presentes:

- **El modelo está bien calibrado**: cuando dice 70–75%, su favorito gana ~73%.
- **El modelo no le gana al mercado.** Aplicando la regla JUGADA a 1,764
  apuestas pasadas contra la línea de cierre (2021, 2022 y 2026), el resultado
  fue 809–944 con ROI de −5.4% ± 2.6%: pierde lo que cobra la casa. Los picks
  marcados EVITAR salieron peor (−13% en spreads). Lee una JUGADA como "aquí mi
  número difiere del mercado", no como una apuesta con ventaja demostrada.

Estos números se recalculan en cada corrida y se ven en la pestaña *Track
record* del sitio.

## Correrlo local

```bash
cd cfb-picks
pip install -r requirements.txt

python -m src.picks --week 12     # baja datos, corre el backtest y genera los picks
python -m http.server -d site     # abre http://localhost:8000
```

`python -m src.picks` sin `--week` usa la semana en curso. Con `--skip-build`
no vuelve a descargar nada (útil después de editar odds o notas).

Otros comandos:

| Comando | Para qué |
|---|---|
| `python -m src.data.build` | Solo descargar juegos a `data/games.parquet` |
| `python -m src.data.odds_history` | Solo líneas de cierre de juegos ya jugados |
| `python -m src.compare` | Tabla comparativa de modelos (`--select` guarda el de menor error) |
| `python -m src.model.tune` | Recalibrar hiperparámetros con temporadas pasadas |
| `python -m pytest` | Tests |

## Actualizar odds a mano

Edita `data/odds_manual.csv`, una fila por juego. Deja vacío lo que no tengas.

```csv
season,week,away,home,ml_away,ml_home,spread_away,spread_away_odds,spread_home_odds,total,over_odds,under_odds,book,captured_at
2026,12,Indiana,Washington,-300,+240,-7.5,-110,,43.5,-115,-105,mi casa,2026-11-20
2026,12,Clemson,Duke,,-475,,,,,,,mi casa,2026-11-20
```

- `spread_away` es la línea del **visitante** (`-7.5` = visitante favorito).
- Los equipos se escriben como en ESPN (`Miami (FL)`, `Pitt`, `Hawaii` también
  funcionan). Si un nombre no se reconoce, la fila se ignora y sale un aviso en
  el sitio: nunca se adivina.
- Tus líneas y momios tienen prioridad. Lo que falte se completa con ESPN
  (DraftKings) y queda etiquetado con su fuente. Un precio de spread o total que
  no esté en ningún lado se asume en −110 y se marca como "asumido".
- La probabilidad del mercado sin vig se calcula con los dos lados de una misma
  casa; si solo capturaste un lado, se usan los dos de ESPN.

Después corre `python -m src.picks --week 12 --skip-build`.

## Notas y ajuste manual

El modelo no sabe de lesiones, clima ni motivación. En `data/notas.csv` puedes
mover la proyección y dejar una nota por juego (el ejemplo es inventado):

```csv
season,week,away,home,ajuste_margen_local,ajuste_total,notas
2026,12,LSU,Tennessee,-3,-6,QB titular de Tennessee duda; viento de 25 mph
```

`ajuste_margen_local` suma puntos al margen del local (negativo favorece al
visitante); `ajuste_total` suma puntos al total.

En `data/parlays.csv` se guardan parlays de moneyline (una fila por pierna)
para que el sitio los evalúe cada semana.

## Cómo funciona

1. **Datos** (`src/data/`): scoreboard de ESPN semana por semana, con caché en
   disco. Las semanas ya jugadas no se vuelven a pedir. Los equipos FBS salen
   de los standings de cada temporada; todo lo demás cuenta como FCS.
2. **Modelo** (`src/model/`): tres candidatos.
   - `baseline.py`: puntos anotados y permitidos en función del % de victorias
     del rival. Es una reconstrucción del prototipo a partir de su descripción.
   - `srs.py`: ratings ajustados por calendario (el rating de un equipo es su
     margen promedio más el rating promedio de sus rivales), con tope a las
     palizas, encogimiento hacia el rating del año anterior y ventaja de local
     estimada. Un sistema igual sobre los totales da el marcador.
   - `elo.py`: Elo con multiplicador de margen, arrastrado entre temporadas.
3. **Backtest** (`src/model/backtest.py`): para predecir la semana W solo se
   usan semanas anteriores. Los hiperparámetros se calibran con 2022–2025
   (`tune.py`) y 2026 se evalúa sin tocarla.
4. **Probabilidades** (`src/model/prob.py`): normal con la desviación estándar
   real del backtest (15.8 pts el margen, 16.1 el total).
5. **Reglas** (`src/model/rules.py`):
   - **JUGADA**: edge ≥ 5%, valor esperado positivo al momio real y sin alertas.
   - **EVITAR**: hay edge en papel, pero el pick cambia de lado al quitar los
     últimos 2 partidos de cada equipo, el modelo difiere 5+ puntos del spread
     (7+ del total), o la línea se movió 3+ puntos en contra desde la apertura
     (4+ el total).
   - **PASAR**: no hay edge suficiente.
   - Tamaño: 1/4 de Kelly con tope de 2% del bankroll.
6. **Sitio** (`site/`): HTML y JS sin backend que leen `site/data/picks.json` y
   `site/data/backtest.json`.

## Deploy en GitHub Pages

El workflow `.github/workflows/cfb-picks.yml` corre el pipeline cada jueves y
sábado en la mañana (de agosto a enero) y publica `site/`. Para activarlo:

1. En GitHub: **Settings → Pages → Source: GitHub Actions**.
2. Lleva estos archivos a la rama por defecto (`main`). GitHub solo ejecuta los
   horarios programados desde ahí.
3. La primera vez puedes lanzarlo a mano en **Actions → CFB Picks → Run
   workflow**.

GitHub desactiva los workflows programados tras 60 días sin actividad en el
repo; se reactivan desde la pestaña Actions. Si el API de ESPN falla, el
workflow falla y el sitio conserva la versión anterior: no se publican datos
inventados.

## Límites conocidos

- Solo usa marcadores finales. No ve jugadas, yardas, lesiones ni clima.
- Los rivales FCS se tratan como un solo equipo promedio.
- El ROI simulado usa la línea de cierre, que no es la que ves el jueves.
- Las líneas de cierre históricas de ESPN solo están completas para 2021, 2022
  y 2026, así que la muestra de apuestas es de tres temporadas.
