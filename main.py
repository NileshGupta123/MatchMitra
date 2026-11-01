"""MatchMitra - FastAPI app and routes.

Start the server:   uvicorn main:app --reload
Run the API tests:  python main.py     (offline, uses a fake AI, no API key needed)

Routes:
    GET  /                  the web page
    GET  /api/health        quick health check
    GET  /api/match         match info and the ball-by-ball replay data
    GET  /api/moments       key moments up to a replay position
    POST /api/catchup       Feature 1: summary of the last 5 overs
    POST /api/explain       Feature 2: explain the moment at a replay position
    POST /api/ask           Feature 3: answer a fan's question

Every AI route builds its facts with match_logic.py from the balls up to the replay
position only, then asks the LLM (through llm_client.py) to explain them.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

import match_logic as ml
import prompts
from llm_client import LLMError, generate, get_model_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("matchmitra.api")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
MAX_QUESTION_LENGTH = 300

try:
    MATCH = ml.load_match()
except (OSError, ValueError) as error:
    raise RuntimeError(
        f"Could not load data/match.json ({error}). Run `python generate_match.py` first."
    ) from error

app = FastAPI(title="MatchMitra", description="AI cricket companion (simulated match data)")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Catch-up and explain answers depend only on the replay position, so a repeated click
# can reuse the answer. This saves Groq calls and avoids rate limits during a demo.
_answer_cache = {}


# ---------------------------------------------------------------------------
# Request models and errors
# ---------------------------------------------------------------------------

class PositionRequest(BaseModel):
    """A replay position: the 0-based index of the latest ball the fan has seen."""

    ball_index: int = Field(..., ge=0)


class AskRequest(PositionRequest):
    """A fan's question at a replay position."""

    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_LENGTH)


class ApiError(Exception):
    """A client error (bad input) with a message that is safe to show to the fan."""

    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@app.exception_handler(ApiError)
def handle_api_error(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.exception_handler(LLMError)
def handle_llm_error(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.exception_handler(RequestValidationError)
def handle_validation_error(request, exc):
    message = "The request was not valid."
    for err in exc.errors():
        field = err["loc"][-1] if err["loc"] else ""
        if field == "question":
            message = f"Please type a question (up to {MAX_QUESTION_LENGTH} characters)."
            break
        if field == "ball_index":
            message = "ball_index must be a whole number, 0 or higher."
            break
    return JSONResponse(status_code=422, content={"error": message})


@app.exception_handler(Exception)
def handle_unexpected_error(request, exc):
    logger.error("Unexpected error on %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong on our side. Please try again."},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _balls_upto(ball_index):
    """Balls up to the replay position, or a clear 400 error if the index is invalid."""
    try:
        return ml.get_balls_upto(MATCH, ball_index)
    except ValueError as exc:
        raise ApiError(str(exc), 400) from exc


def _as_of(balls):
    """Short description of the replay position, shown next to every AI answer."""
    last = balls[-1]
    return {
        "ball": ml.ball_label(last),
        "score": f"{last['score']}/{last['wickets']}",
        "overs": last["overs_completed"],
    }


def _public_ball(index, ball):
    """The fields of one delivery that the web page needs for the replay."""
    return {
        "index": index,
        "label": ml.ball_label(ball),
        "over": ball["over"],
        "batter": ball["batter"],
        "bowler": ball["bowler"],
        "runs_off_bat": ball["runs_off_bat"],
        "extras": ball["extras"],
        "extras_type": ball["extras_type"],
        "is_legal_delivery": ball["is_legal_delivery"],
        "wicket": ball["wicket"],
        "score": ball["score"],
        "wickets": ball["wickets"],
        "overs_completed": ball["overs_completed"],
        "balls_remaining": ball["balls_remaining"],
        "runs_needed": ball["runs_needed"],
        "current_run_rate": ball["current_run_rate"],
        "required_run_rate": ball["required_run_rate"],
    }


def _cached_answer(kind, ball_index, build_prompt):
    """Return a cached AI answer for (kind, position), or ask the LLM and cache it."""
    key = (kind, ball_index)
    if key in _answer_cache:
        logger.info("%s answer for ball_index=%s served from cache", kind, ball_index)
        return _answer_cache[key]
    system_prompt, user_prompt = build_prompt()
    answer = generate(system_prompt, user_prompt)
    _answer_cache[key] = answer
    return answer


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(status_code=404, content={"error": "static/index.html is missing."})
    return FileResponse(index_file)


@app.get("/api/health")
def health():
    return {"status": "ok", "model": get_model_name(), "simulated_data": True}


@app.get("/api/match")
def get_match():
    balls = [_public_ball(i, b) for i, b in enumerate(MATCH["second_innings_balls"])]
    return {
        "match_info": MATCH["match_info"],
        "teams": MATCH["teams"],
        "target": MATCH["target"],
        "total_overs": MATCH["total_overs"],
        "first_innings": MATCH["first_innings"],
        "total_balls": len(balls),
        "balls": balls,
    }


@app.get("/api/moments")
def get_moments(ball_index: int = Query(..., ge=0)):
    balls = _balls_upto(ball_index)
    return {"moments": ml.key_moments(balls)}


@app.post("/api/catchup")
def catchup(req: PositionRequest):
    balls = _balls_upto(req.ball_index)

    def build():
        return prompts.catchup_prompt(ml.build_match_context(MATCH, balls))

    answer = _cached_answer("catchup", req.ball_index, build)
    return {"answer": answer, "as_of": _as_of(balls)}


@app.post("/api/explain")
def explain(req: PositionRequest):
    balls = _balls_upto(req.ball_index)
    moment = ml.moment_context(MATCH, balls)

    def build():
        return prompts.explain_prompt(MATCH, moment)

    answer = _cached_answer("explain", req.ball_index, build)
    return {
        "answer": answer,
        "moment_type": moment["moment_type"],
        "as_of": _as_of(balls),
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    balls = _balls_upto(req.ball_index)
    context = ml.build_match_context(MATCH, balls)
    system_prompt, user_prompt = prompts.ask_prompt(req.question, context)
    answer = generate(system_prompt, user_prompt)
    return {"answer": answer, "as_of": _as_of(balls)}


# ---------------------------------------------------------------------------
# Self-test: python main.py  (offline, fake AI, no API key needed)
# ---------------------------------------------------------------------------

def _self_test():
    from fastapi.testclient import TestClient

    global generate
    real_generate = generate
    logging.disable(logging.CRITICAL)  # expected failures below would only add noise
    client = TestClient(app, raise_server_exceptions=False)

    calls = []

    def fake_generate(system_prompt, user_prompt):
        calls.append((system_prompt, user_prompt))
        return "Fake answer from the test AI."

    def index_of_over_end(over):
        balls = MATCH["second_innings_balls"]
        return max(i for i, b in enumerate(balls) if b["over"] == over)

    try:
        generate = fake_generate

        # 1. Health, page and match data.
        assert client.get("/api/health").json()["status"] == "ok"
        page = client.get("/")
        assert page.status_code == 200 and "MatchMitra" in page.text
        data = client.get("/api/match").json()
        assert data["total_balls"] == len(MATCH["second_innings_balls"]) == len(data["balls"])
        assert data["match_info"]["simulated"] is True
        assert "Simulated" in data["match_info"]["disclaimer"]
        assert data["balls"][0]["index"] == 0 and data["balls"][-1]["score"] == 172
        print(f"OK: page, health and match data work ({data['total_balls']} balls, simulated label present).")

        # 2. Moments only include balls up to the replay position.
        i10 = index_of_over_end(10)
        moments = client.get(f"/api/moments?ball_index={i10}").json()["moments"]
        assert moments and all(m["ball_index"] <= i10 for m in moments)
        print(f"OK: /api/moments returns {len(moments)} moments up to ball {i10}, none from the future.")

        # 3. Bad positions give clear errors.
        r = client.get("/api/moments?ball_index=-1")
        assert r.status_code == 422 and "ball_index" in r.json()["error"]
        r = client.post("/api/catchup", json={"ball_index": 99999})
        assert r.status_code == 400 and "between 0 and" in r.json()["error"]
        r = client.post("/api/catchup", json={"ball_index": "abc"})
        assert r.status_code == 422
        r = client.post("/api/catchup", content="not json", headers={"content-type": "application/json"})
        assert r.status_code == 422 and "error" in r.json()
        print("OK: invalid ball_index values give clear 400 and 422 errors.")

        # 4. Catch-up: answer, no spoilers in the prompt, and caching.
        calls.clear()
        _answer_cache.clear()
        r = client.post("/api/catchup", json={"ball_index": i10})
        body = r.json()
        assert r.status_code == 200 and body["answer"] == "Fake answer from the test AI."
        assert body["as_of"] == {"ball": "9.6", "score": "83/2", "overs": "10.0"}
        system_prompt, user_prompt = calls[0]
        assert "last_5_overs" in user_prompt and "ONLY the MATCH DATA" in system_prompt
        assert "172/6" not in user_prompt and "T. Gaikwad" not in user_prompt
        client.post("/api/catchup", json={"ball_index": i10})
        assert len(calls) == 1, "second identical catch-up should come from the cache"
        print("OK: catch-up works, prompt has no future information, repeat click uses the cache.")

        # 5. Explain a wicket.
        first_wicket = next(m for m in ml.key_moments(MATCH["second_innings_balls"]) if m["type"] == "wicket")
        calls.clear()
        r = client.post("/api/explain", json={"ball_index": first_wicket["ball_index"]})
        body = r.json()
        assert r.status_code == 200 and body["moment_type"] == "wicket"
        assert "R. Nair" in calls[0][1] and "partnership_ended" in calls[0][1]
        print("OK: explain works for a wicket (moment_type, player and partnership are in the prompt).")

        # 6. Ask: the question is passed as a question only, and bad questions are rejected.
        calls.clear()
        r = client.post("/api/ask", json={"ball_index": i10, "question": "  Why is this wicket important?  "})
        assert r.status_code == 200 and r.json()["answer"]
        assert "<<<\nWhy is this wicket important?\n>>>" in calls[0][1]
        for bad in ("", "   ", "x" * (MAX_QUESTION_LENGTH + 1)):
            r = client.post("/api/ask", json={"ball_index": i10, "question": bad})
            assert r.status_code == 422 and "question" in r.json()["error"], bad
        print("OK: ask passes a trimmed question and rejects empty or too long questions.")

        # 7. LLM failures become friendly JSON errors, never a crash.
        def raising(status, message):
            def _raise(system_prompt, user_prompt):
                raise LLMError(message, status)
            return _raise

        _answer_cache.clear()
        generate = raising(429, "Please wait a few seconds and try again.")
        r = client.post("/api/ask", json={"ball_index": i10, "question": "Who is winning?"})
        assert r.status_code == 429 and r.json() == {"error": "Please wait a few seconds and try again."}
        generate = raising(503, "GROQ_API_KEY is missing.")
        r = client.post("/api/catchup", json={"ball_index": i10})
        assert r.status_code == 503 and "GROQ_API_KEY" in r.json()["error"]
        assert (("catchup", i10)) not in _answer_cache, "failed answers must not be cached"

        def crash(system_prompt, user_prompt):
            raise RuntimeError("boom")

        generate = crash
        r = client.post("/api/explain", json={"ball_index": i10})
        assert r.status_code == 500 and "Something went wrong" in r.json()["error"]
        print("OK: LLM failures (429, 503) and unexpected crashes return friendly JSON errors.")
    finally:
        generate = real_generate
        _answer_cache.clear()
        logging.disable(logging.NOTSET)

    print("\nAll API tests passed.")


if __name__ == "__main__":
    _self_test()
