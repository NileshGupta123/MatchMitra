"""Generate data/match.json: a SIMULATED T20 chase with fictional teams and players.

Run once from the project folder:
    python generate_match.py

The match is scripted over by over (OVER_PLAN below). This script plays the plan
ball by ball and calculates every running total (score, wickets, balls remaining,
required run rate), so the numbers in match.json are always consistent.
The data is simulated. It is NOT a real match.
"""

import json
from pathlib import Path

TARGET = 172
TOTAL_BALLS = 120

CHASING_TEAM = "Sahyadri Strikers"
BOWLING_TEAM = "Konkan Kings"

# Chasing team batting order (fictional players).
BATTING_ORDER = [
    "A. Kadam",
    "R. Nair",
    "S. Joshi",
    "V. Pawar",
    "M. Deshmukh",
    "K. Shinde",
    "P. More",
    "T. Gaikwad",
    "N. Sawant",
    "D. Kulkarni",
    "H. Bhatt",
]

# Bowler for each over of the chase (fictional players, max 4 overs each).
OVER_BOWLERS = {
    1: "S. Rane", 2: "V. Salvi", 3: "S. Rane", 4: "V. Salvi", 5: "J. Naik",
    6: "O. Chavan", 7: "G. Patil", 8: "J. Naik", 9: "O. Chavan", 10: "G. Patil",
    11: "L. Mhatre", 12: "J. Naik", 13: "O. Chavan", 14: "L. Mhatre", 15: "G. Patil",
    16: "V. Salvi", 17: "S. Rane", 18: "V. Salvi", 19: "S. Rane", 20: "J. Naik",
}

# Ball-by-ball script for each over.
#   integer  = runs off the bat on a legal ball (0, 1, 2, 3, 4, 6)
#   "wd"     = wide (1 extra run, ball is not counted)
#   "nb"     = no-ball (1 extra run, ball is not counted)
#   "W:type" = wicket on a legal ball (the striker is out)
OVER_PLAN = {
    1: [1, 0, 4, 0, 1, 1],
    2: [0, 6, 1, 0, 2, 0],
    3: [4, 1, "W:caught", 0, 1, 1],
    4: [1, 4, 0, 6, 0, 1],
    5: [0, 1, "W:bowled", 1, 0, 2],
    6: [4, 0, 1, 4, 1, 0],
    7: [1, 1, 0, 2, 1, 1],
    8: [1, 4, 0, 1, 6, 1],
    9: [1, 0, 2, 1, 1, 1],
    10: [1, 4, 1, 0, 1, "wd", 1],
    11: [0, 1, 1, 0, 2, 1],
    12: [1, 1, 0, "W:lbw", 1, 0],
    13: [1, 0, 1, 0, 1, 1],
    14: [4, 6, 6, 1, 4, 0],
    15: [1, 2, 1, 0, 4, 1],
    16: [1, 1, "W:caught", 1, 0, 2],
    17: ["nb", 6, 0, 1, 4, 1, 2],
    18: [1, 1, 0, "W:bowled", 2, 1],
    19: [1, 4, 0, 6, 1, 2],
    20: [0, "W:caught", 2, 4, 0, 2],
}

FIRST_INNINGS = {
    "team": BOWLING_TEAM,
    "score": 171,
    "wickets": 6,
    "overs": "20.0",
    "top_batters": [
        {"name": "A. Salunkhe", "runs": 62, "balls": 41},
        {"name": "R. Bhosale", "runs": 38, "balls": 24},
    ],
    "top_bowlers": [
        {"name": "D. Jadhav", "figures": "2/28"},
        {"name": "M. Ghag", "figures": "2/31"},
    ],
}


def play_chase():
    """Play OVER_PLAN ball by ball and return the list of ball records."""
    balls = []
    score = 0
    wickets = 0
    legal_balls = 0
    striker = BATTING_ORDER[0]
    non_striker = BATTING_ORDER[1]
    next_batter = 2

    for over in range(1, 21):
        legal_in_over = 0
        bowler = OVER_BOWLERS[over]

        for token in OVER_PLAN[over]:
            runs_off_bat = 0
            extras = 0
            extras_type = None
            wicket = None
            is_legal = True

            if token == "wd":
                extras, extras_type, is_legal = 1, "wide", False
            elif token == "nb":
                extras, extras_type, is_legal = 1, "no_ball", False
            elif isinstance(token, str) and token.startswith("W:"):
                wicket = {"type": token.split(":")[1], "player_out": striker}
            else:
                runs_off_bat = token

            batter = striker
            facing_non_striker = non_striker

            score += runs_off_bat + extras
            if is_legal:
                legal_balls += 1
                legal_in_over += 1
            if wicket:
                wickets += 1

            balls_remaining = TOTAL_BALLS - legal_balls
            runs_needed = TARGET - score
            required_rate = (
                round(runs_needed * 6 / balls_remaining, 2) if balls_remaining > 0 else None
            )
            current_rate = round(score * 6 / legal_balls, 2) if legal_balls > 0 else 0.0

            balls.append({
                "innings": 2,
                "over": over,
                "ball": legal_in_over,
                "overs_completed": f"{legal_balls // 6}.{legal_balls % 6}",
                "batter": batter,
                "non_striker": facing_non_striker,
                "bowler": bowler,
                "runs_off_bat": runs_off_bat,
                "extras": extras,
                "extras_type": extras_type,
                "is_legal_delivery": is_legal,
                "wicket": wicket,
                "score": score,
                "wickets": wickets,
                "balls_remaining": balls_remaining,
                "runs_needed": runs_needed,
                "current_run_rate": current_rate,
                "required_run_rate": required_rate,
            })

            # Strike rotation: odd runs swap the batters, a wicket brings in a new batter.
            if wicket:
                striker = BATTING_ORDER[next_batter]
                next_batter += 1
            elif is_legal and runs_off_bat % 2 == 1:
                striker, non_striker = non_striker, striker

            # End of over: batters swap ends.
            if is_legal and legal_in_over == 6:
                striker, non_striker = non_striker, striker

        assert legal_in_over == 6, f"Over {over} has {legal_in_over} legal balls, expected 6"

    return balls


def validate(balls):
    """Check that the scripted chase ends as designed: a win off the last ball."""
    last = balls[-1]
    assert last["balls_remaining"] == 0, "Chase must use all 120 balls"
    assert last["score"] == TARGET, f"Final score {last['score']} must equal target {TARGET}"
    for ball in balls[:-1]:
        assert ball["score"] < TARGET, "Target must not be reached before the last ball"
    assert last["wickets"] < 10, "All out is not allowed in this scenario"


def print_summary(balls):
    """Print a quick human-readable summary so the match can be eyeballed."""
    print("\nOver-by-over summary (score/wickets after each over):")
    for over in range(1, 21):
        last_ball = [b for b in balls if b["over"] == over][-1]
        runs = sum(b["runs_off_bat"] + b["extras"] for b in balls if b["over"] == over)
        rrr = last_ball["required_run_rate"]
        print(
            f"  Over {over:>2} ({last_ball['bowler']:<10}) runs={runs:>2}  "
            f"score={last_ball['score']}/{last_ball['wickets']}  "
            f"need={last_ball['runs_needed']:>3}  RRR={rrr}"
        )

    batter_stats = {}
    for b in balls:
        stats = batter_stats.setdefault(b["batter"], {"runs": 0, "balls": 0})
        stats["runs"] += b["runs_off_bat"]
        if b["is_legal_delivery"]:
            stats["balls"] += 1
    print("\nBatter summary:")
    for name, stats in batter_stats.items():
        print(f"  {name:<14} {stats['runs']:>3} ({stats['balls']})")


def main():
    balls = play_chase()
    validate(balls)

    match = {
        "match_info": {
            "title": f"{CHASING_TEAM} vs {BOWLING_TEAM}",
            "format": "T20 (20 overs per side)",
            "venue": "Sahyadri Oval (fictional)",
            "simulated": True,
            "disclaimer": "Simulated match data, not a real live match. All teams and players are fictional.",
        },
        "teams": {"chasing": CHASING_TEAM, "bowling": BOWLING_TEAM},
        "target": TARGET,
        "total_overs": 20,
        "first_innings": FIRST_INNINGS,
        "batting_order": BATTING_ORDER,
        "second_innings_balls": balls,
    }

    out_path = Path(__file__).parent / "data" / "match.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the header with indentation and keep one ball per line for easy reading.
    header = {k: v for k, v in match.items() if k != "second_innings_balls"}
    header_text = json.dumps(header, indent=2)[:-2]  # drop the closing "\n}"
    ball_lines = ",\n".join("    " + json.dumps(b) for b in balls)
    out_path.write_text(
        header_text + ',\n  "second_innings_balls": [\n' + ball_lines + "\n  ]\n}\n",
        encoding="utf-8",
    )

    # Re-read the file to make sure it is valid JSON.
    with open(out_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded["second_innings_balls"]) == len(balls)

    print(f"Wrote {out_path} with {len(balls)} delivery records.")
    print_summary(balls)
    last = balls[-1]
    print(f"\nFinal: {CHASING_TEAM} {last['score']}/{last['wickets']} chasing {TARGET}, "
          f"won off the last ball.")


if __name__ == "__main__":
    main()