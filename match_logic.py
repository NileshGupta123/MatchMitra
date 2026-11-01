"""MatchMitra - Python calculations for the simulated match.

All match facts are calculated here in plain Python (last 5 overs, run rates,
partnerships, momentum, key moments). The LLM only EXPLAINS these facts. It never
calculates or guesses them.

Two rules protect the fan from spoilers and invented facts:
  1. get_balls_upto() returns only the balls up to the current replay position.
     Every other function here works on that slice, so future balls never leak.
  2. Every number the LLM sees comes from this file.

Run `python match_logic.py` to execute the self-test at the bottom of this file.
"""

import copy
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "match.json"

WINDOW_OVERS = 5            # "last 5 overs" for the catch-up summary
MOMENTUM_OVERS = 2          # recent overs used to judge momentum
EXPENSIVE_OVER_RUNS = 15    # an over with at least this many runs is "expensive"
RECENT_MOMENTS_SHOWN = 5    # how many recent key moments go into the LLM context


# ---------------------------------------------------------------------------
# Loading and slicing
# ---------------------------------------------------------------------------

def load_match(path=DATA_PATH):
    """Load the simulated match from JSON and run basic sanity checks."""
    with open(path, encoding="utf-8") as f:
        match = json.load(f)
    for key in ("match_info", "teams", "target", "second_innings_balls", "batting_order"):
        if key not in match:
            raise ValueError(f"match.json is missing the '{key}' field")
    if not match["second_innings_balls"]:
        raise ValueError("match.json has no ball-by-ball data. Run generate_match.py first.")
    return match


def get_balls_upto(match, ball_index):
    """Return the balls from the start of the chase up to AND including ball_index.

    ball_index is a 0-based position in the replay. This is the only place where the
    replay position is applied, so no future ball can ever reach the LLM.
    """
    balls = match["second_innings_balls"]
    if not isinstance(ball_index, int) or isinstance(ball_index, bool):
        raise ValueError("ball_index must be an integer")
    if ball_index < 0 or ball_index >= len(balls):
        raise ValueError(f"ball_index must be between 0 and {len(balls) - 1}")
    return balls[: ball_index + 1]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ball_total(ball):
    """Total runs added to the score by one delivery (bat runs plus extras)."""
    return ball["runs_off_bat"] + ball["extras"]


def ball_label(ball):
    """Return the over.ball label of a delivery, for example '3.4'.

    For a wide or a no-ball the label shows the ball that is being bowled again.
    """
    if ball["is_legal_delivery"]:
        return f"{ball['over'] - 1}.{ball['ball']}"
    return f"{ball['over'] - 1}.{ball['ball'] + 1}"


def ordinal(number):
    """Return a number with its English ordinal suffix: 1st, 2nd, 3rd, 4th, 11th, 21st."""
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def when_text(ball):
    """Plain-language description of when a delivery happened.

    The LLM is given this text instead of making it up from a label like '2.3'. Reading
    cricket labels itself caused mistakes such as '3rd ball of the 4th over'.
    """
    over_text = f"the {ordinal(ball['over'])} over"
    if ball["extras_type"] == "wide":
        return f"a wide in {over_text}"
    if ball["extras_type"] == "no_ball":
        return f"a no-ball in {over_text}"
    return f"ball {ball['ball']} of {over_text}"


def _initial_state(match):
    """Match state before the first ball of the chase."""
    target = match["target"]
    total_balls = match["total_overs"] * 6
    return {
        "score": 0,
        "wickets": 0,
        "overs_completed": "0.0",
        "balls_remaining": total_balls,
        "runs_needed": target,
        "current_run_rate": 0.0,
        "required_run_rate": round(target * 6 / total_balls, 2),
    }


def _state_from_ball(ball):
    """Scoreboard state right after the given delivery."""
    return {
        "score": ball["score"],
        "wickets": ball["wickets"],
        "overs_completed": ball["overs_completed"],
        "balls_remaining": ball["balls_remaining"],
        "runs_needed": ball["runs_needed"],
        "current_run_rate": ball["current_run_rate"],
        "required_run_rate": ball["required_run_rate"],
    }


def batter_stats(balls):
    """Runs, balls faced, fours and sixes for every batter in the given balls.

    A wide is not counted as a ball faced. A no-ball is counted as faced.
    """
    stats = {}
    for b in balls:
        s = stats.setdefault(
            b["batter"], {"runs": 0, "balls": 0, "fours": 0, "sixes": 0, "out": False}
        )
        s["runs"] += b["runs_off_bat"]
        if b["extras_type"] != "wide":
            s["balls"] += 1
        if b["runs_off_bat"] == 4:
            s["fours"] += 1
        if b["runs_off_bat"] == 6:
            s["sixes"] += 1
        if b["wicket"]:
            stats[b["wicket"]["player_out"]]["out"] = True
    return stats


def bowler_figures(balls):
    """Overs, runs conceded and wickets for every bowler in the given balls."""
    figures = {}
    for b in balls:
        f = figures.setdefault(b["bowler"], {"legal_balls": 0, "runs": 0, "wickets": 0})
        f["runs"] += _ball_total(b)
        if b["is_legal_delivery"]:
            f["legal_balls"] += 1
        if b["wicket"]:
            f["wickets"] += 1
    result = {}
    for name, f in figures.items():
        result[name] = {
            "overs": f"{f['legal_balls'] // 6}.{f['legal_balls'] % 6}",
            "runs": f["runs"],
            "wickets": f["wickets"],
        }
    return result


def batters_at_crease(match, balls):
    """Names of the two batters at the crease after the latest delivery."""
    last = balls[-1]
    pair = [last["batter"], last["non_striker"]]
    if last["wicket"]:
        pair.remove(last["wicket"]["player_out"])
        order = match["batting_order"]
        new_index = last["wickets"] + 1  # two openers, then one new batter per wicket
        if new_index < len(order):
            pair.append(order[new_index])
    return pair


def _partnership_segment(balls, ended=False):
    """Balls that belong to a partnership.

    ended=False: the partnership since the latest wicket (empty right after a wicket).
    ended=True: if the latest ball is a wicket, the partnership that this wicket ended.
    """
    search = balls[:-1] if (ended and balls[-1]["wicket"]) else balls
    start = 0
    for i in range(len(search) - 1, -1, -1):
        if search[i]["wicket"]:
            start = i + 1
            break
    return balls[start:]


def _partnership_summary(segment, batters=None):
    """Runs, legal balls and batter names of a partnership segment."""
    if not segment:
        return {"runs": 0, "balls": 0, "batters": batters or []}
    names = sorted({b["batter"] for b in segment} | {b["non_striker"] for b in segment})
    return {
        "runs": sum(_ball_total(b) for b in segment),
        "balls": sum(1 for b in segment if b["is_legal_delivery"]),
        "batters": batters or names,
    }


def current_partnership(match, balls):
    """Partnership of the two batters now at the crease."""
    segment = _partnership_segment(balls, ended=False)
    return _partnership_summary(segment, batters_at_crease(match, balls))


def partnership_ended_by_wicket(balls):
    """Partnership that the latest wicket ended (the latest ball must be a wicket)."""
    segment = _partnership_segment(balls, ended=True)
    return _partnership_summary(segment)


# ---------------------------------------------------------------------------
# Feature 1: last 5 overs
# ---------------------------------------------------------------------------

def last_overs_summary(balls, n=WINDOW_OVERS):
    """Facts about the last n overs (including the current over if it is in progress)."""
    last = balls[-1]
    current_over = last["over"]
    first_over = max(1, current_over - n + 1)
    window = [b for b in balls if b["over"] >= first_over]

    legal = sum(1 for b in window if b["is_legal_delivery"])
    runs = sum(_ball_total(b) for b in window)
    fours = sum(1 for b in window if b["runs_off_bat"] == 4)
    sixes = sum(1 for b in window if b["runs_off_bat"] == 6)
    dots = sum(1 for b in window if b["is_legal_delivery"] and _ball_total(b) == 0)
    extras = sum(b["extras"] for b in window)

    per_over = []
    for over in range(first_over, current_over + 1):
        over_balls = [b for b in window if b["over"] == over]
        end_of_over = over_balls[-1]
        per_over.append({
            "over": over,
            "runs": sum(_ball_total(b) for b in over_balls),
            "wickets": sum(1 for b in over_balls if b["wicket"]),
            "score_after": f"{end_of_over['score']}/{end_of_over['wickets']}",
            "required_rate_after": end_of_over["required_run_rate"],
            "runs_needed_after": end_of_over["runs_needed"],
            "balls_remaining_after": end_of_over["balls_remaining"],
            "complete": over < current_over or last["ball"] == 6,
        })

    wickets = [
        {
            "when": when_text(b),
            "player_out": b["wicket"]["player_out"],
            "type": b["wicket"]["type"],
            "bowler": b["bowler"],
            "score_at_fall": f"{b['score']}/{b['wickets']}",
        }
        for b in window if b["wicket"]
    ]

    first_index = len(balls) - len(window)
    before_window = balls[first_index - 1] if first_index > 0 else None
    state_before = _state_from_ball(before_window) if before_window else None

    return {
        "from_over": first_over,
        "to_over": current_over,
        "includes_partial_over": last["ball"] < 6,
        "legal_balls": legal,
        "runs": runs,
        "run_rate": round(runs * 6 / legal, 2) if legal else 0.0,
        "wickets": wickets,
        "fours": fours,
        "sixes": sixes,
        "dot_balls": dots,
        "extras": extras,
        "per_over": per_over,
        "score_before_window": (
            f"{state_before['score']}/{state_before['wickets']}" if state_before else "0/0"
        ),
        "required_rate_before_window": (
            state_before["required_run_rate"] if state_before else _initial_required_rate(balls)
        ),
        "required_rate_now": last["required_run_rate"],
    }


def _initial_required_rate(balls):
    """Required rate before ball one, rebuilt from the first delivery's totals."""
    first = balls[0]
    total_target = first["runs_needed"] + first["score"]
    total_balls = first["balls_remaining"] + (1 if first["is_legal_delivery"] else 0)
    return round(total_target * 6 / total_balls, 2)


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def momentum(match, balls):
    """A simple, rule-based momentum signal. The LLM only explains it."""
    last = balls[-1]
    chasing = match["teams"]["chasing"]
    bowling = match["teams"]["bowling"]

    if last["balls_remaining"] == 0 or last["runs_needed"] <= 0:
        return {"leader": "match over", "reason": "The innings is complete."}

    current_over = last["over"]
    first_over = max(1, current_over - MOMENTUM_OVERS + 1)
    recent = [b for b in balls if b["over"] >= first_over]
    legal = sum(1 for b in recent if b["is_legal_delivery"])
    recent_rate = round(sum(_ball_total(b) for b in recent) * 6 / legal, 2) if legal else 0.0
    recent_wickets = sum(1 for b in recent if b["wicket"])
    required = last["required_run_rate"]

    if recent_rate >= required + 1:
        leader = chasing
        reason = (
            f"{chasing} scored at {recent_rate} an over in the last {MOMENTUM_OVERS} overs, "
            f"faster than the {required} an over they need."
        )
    elif recent_rate <= required - 1 or (recent_wickets >= 1 and recent_rate < required):
        leader = bowling
        reason = (
            f"{chasing} scored at only {recent_rate} an over in the last {MOMENTUM_OVERS} overs "
            f"with {recent_wickets} wicket(s) lost, below the {required} an over they need."
        )
    else:
        leader = "balanced"
        reason = (
            f"{chasing} scored at {recent_rate} an over in the last {MOMENTUM_OVERS} overs, "
            f"close to the {required} an over they need."
        )

    return {
        "leader": leader,
        "recent_run_rate": recent_rate,
        "required_run_rate": required,
        "wickets_in_recent_overs": recent_wickets,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Feature 2: key moments and moment context
# ---------------------------------------------------------------------------

def _over_runs_ending_at(balls, index):
    """Runs scored in the over that the ball at `index` belongs to, up to that ball."""
    over = balls[index]["over"]
    return sum(_ball_total(b) for b in balls[: index + 1] if b["over"] == over)


def key_moments(balls):
    """List the key moments (wickets, sixes, fours, expensive overs) seen so far.

    Each moment holds the index of its ball in the replay. When a ball is both a
    boundary and the end of an expensive over, the boundary type is kept and the
    flag 'also_expensive_over' is set.
    """
    moments = []
    for i, b in enumerate(balls):
        moment_type = None
        headline = None
        if b["wicket"]:
            moment_type = "wicket"
            headline = (
                f"WICKET: {b['wicket']['player_out']} is out ({b['wicket']['type']}), "
                f"bowler {b['bowler']}"
            )
        elif b["runs_off_bat"] == 6:
            moment_type = "six"
            headline = f"SIX: {b['batter']} hits {b['bowler']} for a six"
        elif b["runs_off_bat"] == 4:
            moment_type = "four"
            headline = f"FOUR: {b['batter']} hits {b['bowler']} for four"

        ends_expensive_over = (
            b["is_legal_delivery"]
            and b["ball"] == 6
            and _over_runs_ending_at(balls, i) >= EXPENSIVE_OVER_RUNS
        )
        if ends_expensive_over:
            over_runs = _over_runs_ending_at(balls, i)
            if moment_type is None:
                moment_type = "expensive_over"
                headline = f"EXPENSIVE OVER: {over_runs} runs off {b['bowler']}'s over"

        if moment_type:
            moment = {
                "ball_index": i,
                "type": moment_type,
                "over": b["over"],
                "ball": ball_label(b),
                "when": when_text(b),
                "headline": headline,
                "score_after": f"{b['score']}/{b['wickets']}",
            }
            if ends_expensive_over and moment_type != "expensive_over":
                moment["also_expensive_over"] = True
            moments.append(moment)
    return moments


def moment_context(match, balls):
    """Facts about the latest ball in `balls`, for the 'Explain This Moment' feature."""
    last = balls[-1]
    after = _state_from_ball(last)

    if last["wicket"]:
        moment_type = "wicket"
    elif last["runs_off_bat"] == 6:
        moment_type = "six"
    elif last["runs_off_bat"] == 4:
        moment_type = "four"
    elif (
        last["is_legal_delivery"]
        and last["ball"] == 6
        and _over_runs_ending_at(balls, len(balls) - 1) >= EXPENSIVE_OVER_RUNS
    ):
        moment_type = "expensive_over"
    else:
        moment_type = "ordinary_delivery"

    # An expensive over is judged over the WHOLE over, so compare with the start of that
    # over. Every other moment is compared with the ball just before it.
    if moment_type == "expensive_over":
        first_of_over = next(i for i, b in enumerate(balls) if b["over"] == last["over"])
        compared_from = "start of the over"
    else:
        first_of_over = len(balls) - 1
        compared_from = "the previous ball"
    before = _state_from_ball(balls[first_of_over - 1]) if first_of_over > 0 else _initial_state(match)

    batter = batter_stats(balls).get(last["batter"], {})
    over_balls = [b for b in balls if b["over"] == last["over"]]
    context = {
        "moment_type": moment_type,
        "when": when_text(last),
        "over": last["over"],
        "batter": last["batter"],
        "bowler": last["bowler"],
        "runs_off_bat": last["runs_off_bat"],
        "extras": last["extras"],
        "wicket": last["wicket"],
        "state_before": before,
        "state_before_is": compared_from,
        "state_after": after,
        "required_rate_change": (
            round(after["required_run_rate"] - before["required_run_rate"], 2)
            if after["required_run_rate"] is not None else None
        ),
        "batter_innings": {
            "runs": batter.get("runs"),
            "balls": batter.get("balls"),
            "fours": batter.get("fours"),
            "sixes": batter.get("sixes"),
        },
        "over_so_far": {
            "over": last["over"],
            "bowler": last["bowler"],
            "runs": sum(_ball_total(b) for b in over_balls),
            "wickets": sum(1 for b in over_balls if b["wicket"]),
            "ball_by_ball_runs": [_ball_total(b) for b in over_balls],
        },
    }

    if moment_type == "expensive_over":
        # Who scored the runs in this over (the batter of the last ball is not always the hero).
        all_stats = batter_stats(balls)
        runs_in_over = {}
        for b in over_balls:
            runs_in_over[b["batter"]] = runs_in_over.get(b["batter"], 0) + b["runs_off_bat"]
        context["batters_in_over"] = [
            {
                "name": name,
                "runs_in_over": runs,
                "innings_runs": all_stats[name]["runs"],
                "innings_balls": all_stats[name]["balls"],
            }
            for name, runs in sorted(runs_in_over.items(), key=lambda item: -item[1])
        ]

    if moment_type == "wicket":
        context["partnership_ended"] = partnership_ended_by_wicket(balls)
        context["wickets_fallen"] = last["wickets"]
    else:
        context["current_partnership"] = current_partnership(match, balls)
    return context


# ---------------------------------------------------------------------------
# Full context for the LLM (catch-up and chat)
# ---------------------------------------------------------------------------

def build_match_context(match, balls):
    """Everything the LLM may use to answer, built only from balls up to now."""
    last = balls[-1]
    stats = batter_stats(balls)
    crease = batters_at_crease(match, balls)

    wickets_so_far = []
    for b in balls:
        if b["wicket"]:
            out = b["wicket"]["player_out"]
            wickets_so_far.append({
                "when": when_text(b),
                "player_out": out,
                "type": b["wicket"]["type"],
                "bowler": b["bowler"],
                "score_at_fall": f"{b['score']}/{b['wickets']}",
                "batter_runs": stats[out]["runs"],
                "batter_balls": stats[out]["balls"],
            })

    return {
        "note": "Simulated match data with fictional teams and players, not a real match.",
        "match": match["match_info"]["title"],
        "chasing_team": match["teams"]["chasing"],
        "bowling_team": match["teams"]["bowling"],
        "first_innings": match["first_innings"],
        "target": match["target"],
        "current_state": _state_from_ball(last),
        "latest_ball": {
            "when": when_text(last),
            "batter": last["batter"],
            "bowler": last["bowler"],
            "runs_off_bat": last["runs_off_bat"],
            "extras": last["extras"],
            "wicket": last["wicket"],
        },
        "batters_at_crease": [
            {"name": name, **{k: stats.get(name, {}).get(k, 0) for k in ("runs", "balls")}}
            for name in crease
        ],
        "current_partnership": current_partnership(match, balls),
        "last_5_overs": last_overs_summary(balls),
        "momentum": momentum(match, balls),
        "wickets_so_far": wickets_so_far,
        "bowling_figures": bowler_figures(balls),
        "recent_key_moments": [
            {k: m[k] for k in ("when", "type", "headline", "score_after")}
            for m in key_moments(balls)[-RECENT_MOMENTS_SHOWN:]
        ],
    }


# ---------------------------------------------------------------------------
# Data consistency check
# ---------------------------------------------------------------------------

def check_match_consistency(match):
    """Check the ball-by-ball data for impossible or contradictory states.

    Returns a list of problems. An empty list means the data is consistent. It checks the
    running score and wickets, legal balls remaining, runs needed, both run rates, extras,
    six legal balls per over, and strike rotation between consecutive deliveries.
    """
    problems = []
    balls = match["second_innings_balls"]
    target = match["target"]
    total_balls = match["total_overs"] * 6

    score = wickets = legal = 0
    legal_per_over = {}
    for i, b in enumerate(balls):
        score += _ball_total(b)
        wickets += 1 if b["wicket"] else 0
        if b["is_legal_delivery"]:
            legal += 1
            legal_per_over[b["over"]] = legal_per_over.get(b["over"], 0) + 1

        expected = [
            ("score", b["score"], score),
            ("wickets", b["wickets"], wickets),
            ("balls_remaining", b["balls_remaining"], total_balls - legal),
            ("runs_needed", b["runs_needed"], target - score),
        ]
        for name, got, want in expected:
            if got != want:
                problems.append(f"delivery {i}: {name} is {got}, expected {want}")

        if legal and abs(b["current_run_rate"] - round(score * 6 / legal, 2)) > 0.011:
            problems.append(f"delivery {i}: current run rate {b['current_run_rate']} is wrong")
        if b["balls_remaining"] and abs(
            b["required_run_rate"] - round(b["runs_needed"] * 6 / b["balls_remaining"], 2)
        ) > 0.011:
            problems.append(f"delivery {i}: required run rate {b['required_run_rate']} is wrong")
        if b["extras_type"] in ("wide", "no_ball") and b["is_legal_delivery"]:
            problems.append(f"delivery {i}: a wide or no-ball is marked as a legal delivery")
        if b["extras_type"] is None and not b["is_legal_delivery"]:
            problems.append(f"delivery {i}: an illegal delivery has no extras type")
        if b["batter"] == b["non_striker"]:
            problems.append(f"delivery {i}: the same batter is at both ends")

    for over, count in sorted(legal_per_over.items()):
        if count != 6:
            problems.append(f"over {over} has {count} legal balls instead of 6")

    # Strike rotation: odd bat runs swap the batters (a wide does not), and the end of an
    # over swaps them again. Deliveries right after a wicket are skipped, because the new
    # batter's position is not stored.
    for i, (prev, nxt) in enumerate(zip(balls, balls[1:])):
        if prev["wicket"]:
            continue
        swap = prev["runs_off_bat"] % 2 == 1 and prev["extras_type"] != "wide"
        if prev["is_legal_delivery"] and prev["ball"] == 6:
            swap = not swap
        want = (
            (prev["non_striker"], prev["batter"]) if swap else (prev["batter"], prev["non_striker"])
        )
        if (nxt["batter"], nxt["non_striker"]) != want:
            problems.append(f"delivery {i + 1}: strike rotation is wrong")
    return problems


# ---------------------------------------------------------------------------
# Self-test: python match_logic.py
# ---------------------------------------------------------------------------

def _index_of_over_end(match, over):
    """Index of the last delivery of the given over."""
    balls = match["second_innings_balls"]
    return max(i for i, b in enumerate(balls) if b["over"] == over)


def _self_test():
    match = load_match()
    balls = match["second_innings_balls"]
    print(f"Loaded {len(balls)} deliveries for {match['match_info']['title']}.")

    # 1. No-spoiler slicing.
    first = get_balls_upto(match, 0)
    assert len(first) == 1 and first[0]["score"] == 1
    for bad in (-1, len(balls), 3.5, True):
        try:
            get_balls_upto(match, bad)
            raise AssertionError(f"ball_index {bad!r} should have been rejected")
        except ValueError:
            pass
    print("OK: slicing is correct and rejects bad indexes.")

    # 2. Final state.
    final = _state_from_ball(get_balls_upto(match, len(balls) - 1)[-1])
    assert final["score"] == 172 and final["wickets"] == 6 and final["balls_remaining"] == 0
    print("OK: final state is 172/6 with 0 balls remaining.")

    # 2b. The whole dataset is consistent, and the checker really catches mistakes.
    problems = check_match_consistency(match)
    assert problems == [], problems[:3]
    wrong_score = copy.deepcopy(match)
    wrong_score["second_innings_balls"][30]["score"] += 1
    assert any("score" in p for p in check_match_consistency(wrong_score))
    wrong_legal = copy.deepcopy(match)
    wide_index = next(i for i, b in enumerate(balls) if b["extras_type"] == "wide")
    wrong_legal["second_innings_balls"][wide_index]["is_legal_delivery"] = True
    assert any("legal" in p for p in check_match_consistency(wrong_legal))
    wrong_strike = copy.deepcopy(match)
    ball = wrong_strike["second_innings_balls"][25]
    ball["batter"], ball["non_striker"] = ball["non_striker"], ball["batter"]
    assert any("strike rotation" in p for p in check_match_consistency(wrong_strike))
    print(f"OK: all {len(balls)} deliveries are consistent (score, wickets, run rates, strike), and bad data is caught.")

    # 3. Last 5 overs at the end of over 10 (overs 6 to 10).
    i10 = _index_of_over_end(match, 10)
    summary = last_overs_summary(get_balls_upto(match, i10))
    assert (summary["from_over"], summary["to_over"]) == (6, 10)
    assert summary["runs"] == 44, summary["runs"]
    assert summary["fours"] == 4 and summary["sixes"] == 1
    assert summary["wickets"] == [] and not summary["includes_partial_over"]
    assert summary["score_before_window"] == "39/2"
    last_over = summary["per_over"][-1]
    assert last_over["score_after"] == "83/2" and last_over["required_rate_after"] == 8.9
    print("OK: last 5 overs at end of over 10 = 44 runs, 4 fours, 1 six, no wickets.")

    # 3b. Per-over required rate: it rose to 11.0 after over 13 and fell to 9.33 after over 14.
    summary14 = last_overs_summary(get_balls_upto(match, _index_of_over_end(match, 14)))
    rates = {o["over"]: o["required_rate_after"] for o in summary14["per_over"]}
    assert rates[13] == 11.0 and rates[14] == 9.33
    needs = {o["over"]: (o["runs_needed_after"], o["balls_remaining_after"]) for o in summary14["per_over"]}
    assert needs[13] == (77, 42) and needs[14] == (56, 36), needs
    print("OK: per-over required rate shows the drop in over 14 (11.0 to 9.33).")
    print("OK: per-over runs needed and balls remaining are given (77 off 42 before over 14, 56 off 36 after).")

    # 3c. Spoken time text, so the LLM never converts labels like 2.3 by itself.
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 14, 20, 21)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "14th", "20th", "21st"
    ]
    assert when_text(balls[14]) == "ball 3 of the 3rd over"
    assert when_text(balls[59]) == "a wide in the 10th over"
    assert when_text(balls[97]) == "a no-ball in the 17th over"
    print("OK: when_text gives 'ball 3 of the 3rd over' (not the label 2.3).")

    # 4. Early in the match the window is shorter than 5 overs.
    summary = last_overs_summary(get_balls_upto(match, 9))
    assert (summary["from_over"], summary["to_over"]) == (1, 2)
    assert summary["includes_partial_over"]
    print("OK: window is shorter than 5 overs at the start of the match.")

    # 5. Key moments never include future balls.
    moments_all = key_moments(balls)
    moments_early = key_moments(get_balls_upto(match, i10))
    assert all(m["ball_index"] <= i10 for m in moments_early)
    assert sum(1 for m in moments_all if m["type"] == "wicket") == 6
    assert any(m["type"] == "expensive_over" and m["over"] == 14 for m in moments_all)
    print(f"OK: {len(moments_all)} key moments in total, 6 wickets, over 14 flagged expensive.")

    # 6. First wicket moment context.
    first_wicket = next(m for m in moments_all if m["type"] == "wicket")
    ctx = moment_context(match, get_balls_upto(match, first_wicket["ball_index"]))
    assert ctx["wicket"]["player_out"] == "R. Nair"
    assert ctx["batter_innings"]["runs"] == 7 and ctx["batter_innings"]["balls"] == 8
    assert ctx["partnership_ended"]["runs"] == 21
    assert ctx["when"] == "ball 3 of the 3rd over"
    print("OK: first wicket context (R. Nair 7 off 8, partnership of 21 ended).")

    # 7. Expensive over context.
    over14 = next(m for m in moments_all if m["type"] == "expensive_over")
    ctx = moment_context(match, get_balls_upto(match, over14["ball_index"]))
    assert ctx["over_so_far"]["runs"] == 21
    assert ctx["over_so_far"]["ball_by_ball_runs"] == [4, 6, 6, 1, 4, 0]
    assert ctx["state_before_is"] == "start of the over"
    assert ctx["state_before"]["score"] == 95 and ctx["state_before"]["required_run_rate"] == 11.0
    assert ctx["state_after"]["required_run_rate"] == 9.33
    assert ctx["required_rate_change"] == -1.67
    heroes = {b["name"]: b["runs_in_over"] for b in ctx["batters_in_over"]}
    assert heroes == {"S. Joshi": 17, "M. Deshmukh": 4}
    assert ctx["batters_in_over"][0]["name"] == "S. Joshi"
    print("OK: expensive over context (21 runs, S. Joshi 17 of them, required rate 11.0 to 9.33).")

    # 8. Momentum at three points of the match.
    expected = {
        12: match["teams"]["bowling"],   # slow overs and a wicket
        14: match["teams"]["chasing"],   # 25 runs in two overs
        18: match["teams"]["bowling"],   # wicket and a slow over
    }
    for over, leader in expected.items():
        m = momentum(match, get_balls_upto(match, _index_of_over_end(match, over)))
        assert m["leader"] == leader, (over, m)
    print("OK: momentum is rule-based and sensible after overs 12, 14 and 18.")

    # 9. Full context is JSON-serializable and spoiler-free.
    ctx = build_match_context(match, get_balls_upto(match, i10))
    text = json.dumps(ctx)
    assert ctx["current_state"]["score"] == 83 and ctx["current_state"]["wickets"] == 2
    assert "172/6" not in text and "J. Naik" in text  # Naik bowled over 8 already
    assert "T. Gaikwad" not in text  # batter who has not come in yet
    assert '"ball":' not in text, "raw ball labels must not reach the LLM"
    print(f"OK: full context built ({len(text)} characters), no future information.")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()