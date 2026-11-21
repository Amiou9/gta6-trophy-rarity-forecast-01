"""gta6-trophy-rarity-forecast-01 forecaster (Python 3.13, standard library only).

Forecasts every GTA VI trophy's Exophase earn rate at the case target as five
quantiles, scored by pinball loss on the logit scale. Beats the persistence
baseline by (a) extrapolating a *saturating* growth curve instead of assuming
the last rate stands still, and (b) calibrating the spread from the history
instead of using the house's fixed 0.5*sqrt(days).

Model, per trophy
-----------------
Time is measured in hours since the first read of the case's history, on a
log scale: x = log1p(hours).  Cumulative adoption curves (players earning a
trophy over time) are close to linear in log-time and flatten as 1/t, so a
linear fit in x extrapolates to a 4.5-day horizon without the runaway growth
a linear-in-time trend would give.  Two complementary fits are averaged in
logit space:

  A. Direct: logit(rate) ~ a + b * x, fitted on the *distinct* reads only
     (consecutive identical reads carry no information and would otherwise
     overweight the flat stretches while Exophase sits unchanged for hours).
  B. Counts: log(earners) ~ aE + bE * x and log(players_tracked) ~ aN + bN * x,
     then rate = earners / players.  Earners are (near-)monotone, so this fit
     is immune to the mechanical dilution drop in the rate when a refresh
     adds a wave of new tracked players; the players fit then re-applies the
     dilution at the target.  For small rates A and B agree; for large rates
     A (logit) saturates correctly, B keeps the dilution honest.

Shrinkage: a trophy's slope b is shrunk toward the median slope of its tier
(same-tier trophies behave alike), with weight n/(n+6) on its own fit, n the
number of distinct reads in the recent window.  Noisy/rare trophies and
trophies with no movement at all therefore borrow the group drift instead
of extrapolating noise, and a trophy stuck at 0% still drifts up with its
tier.  The extrapolated logit move is capped at +/-1.5 so a steep early
slope cannot push a rate to 99%; the fits themselves use only the recent
window (last 1.0 of log-time), because a saturating curve's recent slope
is the one that extrapolates -- a whole-history slope overstates it.

Spread (logit sd), per trophy:
    spread = sqrt(0.05^2 + (0.30*sqrt(days))^2 + (se_b * dx)^2)
where days is the horizon in days, dx the log-time distance to the target,
and se_b the standard error of the fitted slope.  When the history is flat,
residuals and se_b are ~0 and the spread collapses to the 0.05 floor plus
the horizon term -- much narrower than persistence's 0.5*sqrt(days), which
is why the backtest on the flat practice histories wins on spread alone.
When the history actually moves, residuals/se_b widen the spread honestly;
the pinball loss rewards an honest spread, not a narrow one.

Edge cases: rate 100% stays ~99.99 (clamped, as the scorer clamps);
rate 0% is treated as 0.01% before the logit and drifts with its tier;
null rates are skipped; a case with no usable movement falls back to the
last rate with the horizon spread.  A per-case failure never aborts the
run: that case gets a valid persistence-style forecast instead.

Deterministic: sorted input files, no randomness, no dict-order dependence.
Runs in well under 8 s per case (O(reads) per trophy, no search).

Grader: `python3 -P -s /bot/main.py`, no network.  Local test:
    mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Spread terms (logit units): floor, per-sqrt-day, slope-uncertainty.
SPREAD_FLOOR = 0.05
SPREAD_PER_SQRT_DAY = 0.30
MAX_LOGIT_MOVE = 1.5
LOGIT_MIN = math.log(0.01 / 99.99)
LOGIT_MAX = math.log(99.99 / 0.01)


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s):
    """ISO-8601 timestamp -> aware datetime, or None."""
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, se_b, resid_sd, n, sxx)."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0, 0.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx <= 0:
        return (my, 0.0, 0.0, 0.0, n, sxx)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = n - 2
    var = sum(r * r for r in resid) / dof if dof > 0 else 0.0
    se_b = math.sqrt(var / sxx) if sxx > 0 else 0.0
    return (a, b, se_b, math.sqrt(var), n, sxx)


def median(vals):
    vals = sorted(vals)
    if not vals:
        return 0.0
    m = len(vals) // 2
    if len(vals) % 2:
        return vals[m]
    return 0.5 * (vals[m - 1] + vals[m])


def forecast_case(case: dict, name: str) -> dict:
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = max(0.0, float(case.get("horizon_minutes") or 0) / 1440.0)

    # --- time axis: hours since first read, log scale -------------------
    times = [parse_time(h.get("at")) for h in history]
    t0 = next((t for t in times if t is not None), None)
    if t0 is None:
        # No timestamps: fall back to read index as time.
        hours = [float(i) * 0.5 for i in range(len(history))]
    else:
        hours = [
            max(0.0, (t - t0).total_seconds() / 3600.0) if t is not None else 0.0
            for t in times
        ]
    xs_all = [math.log1p(h) for h in hours]

    target_t = parse_time(case.get("target"))
    if target_t is not None and t0 is not None:
        h_target = max(0.0, (target_t - t0).total_seconds() / 3600.0)
    elif hours:
        h_target = hours[-1] + float(case.get("horizon_minutes") or 0) / 60.0
    else:
        h_target = float(case.get("horizon_minutes") or 0) / 60.0
    x_target = math.log1p(h_target)
    x_last = xs_all[-1] if xs_all else 0.0
    dx = max(0.0, x_target - x_last)

    # --- players-tracked fit (all reads; N moves even when rates don't) --
    n_fit = (0.0, 0.0, 0.0, 0.0, 0, 0.0)
    n_last = None
    if history:
        pts = [
            (x, math.log(max(1.0, float(h.get("players_tracked") or 1))))
            for x, h in zip(xs_all, history)
            if h.get("players_tracked")
        ]
        if len(pts) >= 2:
            n_fit = ols([p[0] for p in pts], [p[1] for p in pts])
        n_last = float(history[-1].get("players_tracked") or 0) or None
    log_n_target = n_fit[0] + n_fit[1] * x_target if n_fit[4] >= 2 else (
        math.log(n_last) if n_last else 0.0
    )

    # --- per-trophy fits --------------------------------------------------
    # First pass: raw slopes, for the tier-median shrinkage prior.
    fits = []
    for i, trophy in enumerate(trophies):
        # Distinct reads: keep a read when this trophy's rate (or the
        # players count) changed from the previous kept read, plus first/last.
        pts = []  # (x, logit rate)
        epts = []  # (x, log earners)
        prev_rate = None
        prev_n = None
        last_rate = None
        for k, h in enumerate(history):
            rates = h.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is not None:
                last_rate = rate
            n = h.get("players_tracked")
            if rate is None or n is None:
                continue
            changed = (
                prev_rate is None or rate != prev_rate or n != prev_n
                or k == len(history) - 1
            )
            if changed:
                pts.append((xs_all[k], logit(rate)))
                earners = max(0.0, float(rate) * float(n) / 100.0)
                if earners > 0:
                    epts.append((xs_all[k], math.log(earners)))
                prev_rate, prev_n = rate, n
        # Recent-window fit: only distinct points in the last 1.0 of log-time
        # (roughly the last two-thirds of the history's span).  A growth
        # curve's slope falls as it saturates, so the *recent* slope -- not
        # the average slope since launch -- is what extrapolates.  Tuned on
        # two synthetic launch regimes (rate-saturating and earners-
        # saturating): the recent window beats a whole-history fit on both,
        # while the whole-history fit loses to persistence in the second.
        x_recent = (xs_all[-1] - 1.0) if xs_all else 0.0
        pts_r = [p for p in pts if p[0] >= x_recent] or pts[-2:]
        epts_r = [p for p in epts if p[0] >= x_recent] or epts[-2:]
        if len(pts_r) >= 2:
            a, b, se_b, sd, npts, sxx = ols([p[0] for p in pts_r], [p[1] for p in pts_r])
        elif pts_r:
            a, b, se_b, sd, npts, sxx = (pts_r[0][1], 0.0, 0.0, 0.0, 1, 0.0)
        else:
            a, b, se_b, sd, npts, sxx = (logit(last_rate if last_rate is not None else 0.0), 0.0, 0.0, 0.0, 0, 0.0)
        if len(epts_r) >= 2:
            ae, be, _, _, _, _ = ols([p[0] for p in epts_r], [p[1] for p in epts_r])
        elif epts_r:
            ae, be = epts_r[0][1], 0.0
        else:
            ae, be = (math.log(max(1e-9, (last_rate or 0.0) * (n_last or 1) / 100.0)), 0.0)
        fits.append({
            "tier": trophy.get("tier"),
            "last_rate": last_rate,
            "a": a, "b": b, "se_b": se_b, "sd": sd, "n": npts,
            "ae": ae, "be": be,
            "last_logit": logit(last_rate if last_rate is not None else 0.0),
        })

    # Tier-median slope prior (overall median as fallback).
    by_tier = {}
    for f in fits:
        by_tier.setdefault(f["tier"], []).append(f["b"])
    tier_med = {t: median(v) for t, v in by_tier.items()}
    overall_med = median([f["b"] for f in fits])

    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        prior = tier_med.get(f["tier"], overall_med)
        w = f["n"] / (f["n"] + 6.0) if f["n"] else 0.0
        b = w * f["b"] + (1.0 - w) * prior
        # A: direct logit extrapolation from the last logit.
        move_a = b * dx
        # B: earners minus players at the target, in logit space.
        log_e_target = f["ae"] + f["be"] * x_target
        # Re-anchor B at the last read so a level mismatch in the E fit
        # cannot shift the forecast: only its *slope* matters.
        if f["n"] >= 2 and n_last:
            log_e_last = math.log(max(1e-9, (f["last_rate"] or 0.0) * n_last / 100.0)) if (f["last_rate"] or 0.0) > 0 else None
        else:
            log_e_last = None
        if log_e_last is not None:
            # Slope of log-rate implied by B: (logE slope - logN slope).
            b_en = f["be"] - n_fit[1]
            move_b = b_en * dx
            # Weight B (counts) 0.7 / A (direct) 0.3: B sees a saturating
            # earners curve turning over against still-growing players,
            # which A (rate-only) misses until the rate itself turns.
            move = 0.3 * move_a + 0.7 * move_b
        else:
            move = move_a
        move = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, move))
        med_logit = f["last_logit"] + move
        # A trophy at exactly 100% in the last read: no earners info can
        # move it down (E == N by construction); keep it at the cap.
        if (f["last_rate"] or 0.0) >= 99.99 and f["n"] <= 1:
            med_logit = LOGIT_MAX
        med_logit = max(LOGIT_MIN, min(LOGIT_MAX, med_logit))

        spread = math.sqrt(
            SPREAD_FLOOR ** 2
            + (SPREAD_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
            + (f["se_b"] * dx) ** 2
        )
        # Volatile histories widen the spread a little more.
        spread = math.sqrt(spread ** 2 + (0.5 * f["sd"]) ** 2)
        qs = []
        for z in Z:
            q = round4(clamp_pct(sigmoid(med_logit + z * spread)))
            qs.append(q)
        # Enforce non-decreasing after rounding.
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_case(case: dict) -> dict:
    """Valid persistence-style output, used if the real forecast fails."""
    history = case.get("history") or []
    last_rates = (history[-1].get("rates") or []) if history else []
    horizon_days = max(0.0, float(case.get("horizon_minutes") or 0) / 1440.0)
    spread = 0.5 * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i] if i < len(last_rates) and last_rates[i] is not None else 1.0
        lg = logit(rate)
        qs = [round4(clamp_pct(sigmoid(lg + z * spread))) for z in Z]
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case, path.name)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = fallback_case(case)
            except Exception:
                continue
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
