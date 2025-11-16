import os
import time
from collections import defaultdict, deque
from datetime import datetime

from flask import Flask, render_template, request, jsonify, session
import secrets
from google import genai
from dotenv import load_dotenv
import pandas as pd 
import json

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# -----------------------------
# In-memory user state
# -----------------------------
user_state = defaultdict(lambda: {
    "habits": deque(maxlen=200),
    "energy": 5,         # 0–10
    "mood": 0.6,         # 0–1
    "chat_history": [],
    "health_snapshot": None,
    "health_scores": None,
    "good_streak": 0,        # consecutive healthy habits
    "just_hit_streak": False,
    "last_update": time.time()  # last time we touched state (for decay)
})


def get_user_id():
    if "uid" not in session:
        session["uid"] = secrets.token_hex(8)
    return session["uid"]


# -----------------------------
# Habit + simple analytics
# -----------------------------
def add_habit(user_id, text, source="user"):
    """
    Log a habit with:
    - crude health category (sleep/exercise/nutrition/mental/other)
    - overall quality: healthy / unhealthy / neutral
    And update pet energy/mood based on quality.
    """
    state = user_state[user_id]
    ts = time.time()
    text_lower = text.lower()

    # Category for analytics
    if any(k in text_lower for k in ["sleep", "nap", "bed", "rest"]):
        category = "sleep"
    elif any(k in text_lower for k in ["run", "gym", "walk", "exercise", "workout", "steps"]):
        category = "exercise"
    elif any(k in text_lower for k in ["water", "hydrate", "drink", "veggie", "salad", "fruit"]):
        category = "nutrition"
    elif any(k in text_lower for k in ["meditate", "journal", "therapy", "breathe", "mindful"]):
        category = "mental"
    else:
        category = "other"

    # overall quality (healthy / unhealthy / neutral)
    result = classify_habit_quality(text)
    quality = result["quality"]
    severity = result["severity"]

    state["habits"].append({
        "text": text,
        "category": category,
        "quality": quality,
        "timestamp": ts,
        "source": source,
    })

    if quality == "healthy":
        state["energy"] = min(10, state["energy"] + (1 + 2 * severity))
        state["mood"] = min(1.0, state["mood"] + (0.05 + 0.10 * severity))
    elif quality == "unhealthy":
        energy_drop = 1 + severity * 3
        mood_drop   = 0.02 + severity * 0.12
        state["energy"] = max(0.0, state["energy"] - energy_drop)
        state["mood"]   = max(0.0, state["mood"] - mood_drop)
    else:
        state["energy"] = min(10, state["energy"] + 0.5)
        state["mood"]   = min(1.0, state["mood"] + 0.01)


    # --- Streak tracking: 3 healthy habits in a row ---
    if quality == "healthy":
        prev = state.get("good_streak", 0)
        state["good_streak"] = prev + 1
        # trigger a "just hit streak" flag when going from 2 -> 3+
        state["just_hit_streak"] = (prev < 3 and state["good_streak"] >= 3)
    else:
        state["good_streak"] = 0
        state["just_hit_streak"] = False

    # Update last_update timestamp since we just did something good/bad
    state["last_update"] = time.time()



def count_habits_today(user_id):
    state = user_state[user_id]
    today = datetime.utcnow().date()
    c = 0
    for h in state["habits"]:
        if datetime.utcfromtimestamp(h["timestamp"]).date() == today:
            c += 1
    return c


def health_trend_summary(user_id):
    """
    Tiny 'data science' block:
    Count habits per category and compare last 24h vs older.
    """
    state = user_state[user_id]
    habits = list(state["habits"])
    now = time.time()
    last_24h = [h for h in habits if now - h["timestamp"] <= 24 * 3600]
    older = [h for h in habits if now - h["timestamp"] > 24 * 3600]

    def count_by_cat(entries):
        from collections import defaultdict as dd
        counts = dd(int)
        for h in entries:
            counts[h["category"]] += 1
        return counts

    return {
        "recent_counts": count_by_cat(last_24h),
        "older_counts": count_by_cat(older),
        "recent_total": len(last_24h),
        "older_total": len(older),
    }

def apply_passive_decay(user_id):
    """
    Slowly decrease energy and mood if the user has been inactive
    (no chat/habit) for a while. Called whenever the user next interacts.
    """
    state = user_state[user_id]
    now = time.time()
    last = state.get("last_update")
    if last is None:
        state["last_update"] = now
        return

    dt_hours = (now - last) / 3600.0
    # Ignore tiny gaps
    if dt_hours < 0.25:  # < 15 minutes
        return

    # For each hour of inactivity:
    # - energy loses 0.5
    # - mood loses 0.03
    decay_hours = min(dt_hours, 12.0)  # cap, so it doesn't crash to zero
    energy_loss = 0.5 * decay_hours
    mood_loss = 0.03 * decay_hours

    state["energy"] = max(0.0, state["energy"] - energy_loss)
    state["mood"] = max(0.0, state["mood"] - mood_loss)

    state["last_update"] = now



# -----------------------------
# AI / LLM hooks (single personality)
# -----------------------------
def generate_pet_reply(user_id, user_message, as_habit=False):
    """
    ONE unified TamaCare personality powered by Gemini.
    Uses:
      - user_message (what the user just typed)
      - as_habit (True if they hit 'Track Habit')
      - health_trend_summary(user_id) for context (recent habits/categories)
      - current energy/mood from user_state
    Returns a short, friendly pet reply string.
    """

    state = user_state[user_id]
    trends = health_trend_summary(user_id)

    energy = state["energy"]
    mood = state["mood"]
    recent_total = trends["recent_total"]
    recent_counts = trends["recent_counts"]

    cat_parts = []
    for cat in ["exercise", "sleep", "nutrition", "mental", "other"]:
        if cat in recent_counts:
            cat_parts.append(f"{cat}: {recent_counts[cat]}")
    cats_str = ", ".join(cat_parts) if cat_parts else "no logged habits yet"

    pet_state_summary = (
        f"Energy (0-10): {energy}. Mood (0-1): {mood:.2f}. "
        f"Habits in last 24h: {recent_total}. "
        f"Breakdown by category: {cats_str}."
    )

    habit_flag_text = "This message WAS logged as a health habit." if as_habit else \
        "This message was general chat, NOT directly logged as a habit."

    system_instruction = """
        You are TamaCare, a tiny retro-styled virtual pet that lives off the user's healthy choices.
        Your vibe:
        - Chill and honest
        - Gives emotionally modern, kind, non-judgmental wellness feedback
        - Short, punchy responses (1–3 sentences max)
        - Occasionally uses light retro/gaming metaphors,

        Goals:
        1. Acknowledge what the user said.
        2. Reflect their recent health trend briefly.
        3. Gently encourage a realistic next step (if appropriate).
        4. NEVER give medical diagnoses or tell the user to ignore doctors.
        Always be general and supportive, not clinical.
    """

    user_context = f"""
        USER MESSAGE:
        {user_message}

        CONTEXT ABOUT USER HEALTH FROM TAMA STATE:
        {pet_state_summary}

        EVENT INFO:
        {habit_flag_text}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_instruction, user_context]
        )

        text = (response.text or "").strip()
        if not text:
            raise ValueError("Empty Gemini response")
        return text

    except Exception as e:
        print("Gemini error in generate_pet_reply:", e)
        if as_habit:
            return (
                "⌛ TamaCare here. That definitely counts as taking care of yourself. "
                "My AI brain is a bit offline, but I’m still proud of you."
            )
        else:
            return (
                "⌛ TamaCare here. Thanks for checking in with me. "
                "Even when my AI brain hiccups, your small steps still matter."
            )


import json  # make sure this is at the top of app.py

def classify_habit_quality(text: str) -> dict:
    """
    Classify a habit description using Gemini + fallback heuristics.

    ALWAYS returns:
        {"quality": "healthy" | "unhealthy" | "neutral", "severity": float}

    - quality: how the habit affects health overall
    - severity: 0.0–1.0, only meaningful when quality == "unhealthy"
               (0.0 is barely harmful, 1.0 is extremely harmful)
    """

    t = (text or "").strip()
    if not t:
        return {"quality": "neutral", "severity": 0.0}

    # ---------- 1) Try Gemini in a SUPER simple format ----------
    try:
        prompt = f"""
        You are a classifier for health-related behaviors.

        Given a short sentence describing something a person did,
        respond ONLY on the FIRST LINE in this exact format:

        quality|severity

        Where:
        - quality is one of: healthy, unhealthy, neutral
        - severity is a number from 0.0 (barely harmful) to 1.0 (extremely harmful)
        - For clearly healthy or neutral habits, use severity 0.0

        NO extra words, no JSON, no code fences, no explanation.
        Just:
        healthy|0.0
        or:
        unhealthy|0.7

        HABIT: "{t}"
        """

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        raw = (resp.text or "").strip()
        # Take the first non-empty line
        first_line = ""
        for line in raw.splitlines():
            if line.strip():
                first_line = line.strip()
                break

        # Expected formats:
        #   unhealthy|0.6
        #   healthy|0
        #   neutral|0.0
        if "|" in first_line:
            q_str, s_str = first_line.split("|", 1)
            q_str = q_str.strip().strip('"').strip("'")
            s_str = s_str.strip().strip('"').strip("'")
        else:
            # If Gemini ignores the format, treat whole thing as quality and severity 0
            q_str, s_str = first_line.strip(), "0.0"

        q = q_str.lower()
        if q not in {"healthy", "unhealthy", "neutral"}:
            q = "neutral"

        try:
            s = float(s_str)
        except Exception:
            s = 0.0

        s = max(0.0, min(1.0, s))
        return {"quality": q, "severity": s}

    except Exception as e:
        # If ANYTHING goes wrong with Gemini, fall back to keyword rules
        print("Gemini classify error in classify_habit_quality:", e)

    # ---------- 2) Fallback heuristic (no model needed) ----------
    tl = t.lower()

    # Quit / stopped unhealthy habit → healthy
    if ("quit" in tl or "stopped" in tl or "gave up" in tl) and any(
        bad in tl for bad in ["smoking", "cigarettes", "vaping", "vape", "drinking", "alcohol"]
    ):
        return {"quality": "healthy", "severity": 0.0}

    very_unhealthy = [
        "smoked", "smoke", "cigarette", "cigarettes",
        "vape", "vaping",
        "weed", "joint",
        "drunk", "wasted", "hungover", "blackout",
        "binge drinking", "blackout drunk",
    ]

    mildly_unhealthy = [
        "binge", "binged",
        "fast food", "junk food", "mcdonald", "burger king", "kfc",
        "chips", "soda", "coke", "pepsi",
    ]

    healthy = [
        "walk", "walked", "run", "ran", "jog", "gym", "workout", "exercise",
        "pushup", "push-ups", "pullup", "yoga", "stretch",
        "water", "hydrated", "hydrate",
        "salad", "veggies", "vegetable", "fruit",
        "slept", "sleep", "nap",
        "meditate", "meditated", "journaling", "journal", "therapy", "breathed"
    ]

    if any(k in tl for k in very_unhealthy):
        return {"quality": "unhealthy", "severity": 0.8}

    if any(k in tl for k in mildly_unhealthy):
        return {"quality": "unhealthy", "severity": 0.4}

    if any(k in tl for k in healthy):
        return {"quality": "healthy", "severity": 0.0}

    # Default
    return {"quality": "neutral", "severity": 0.0}


def generate_state_bubble(user_id) -> str:
    """
    Short 1-sentence summary of how the pet feels based on current stats.
    Uses Gemini; falls back to a rule-based line if something fails.
    """
    state = user_state[user_id]
    trends = health_trend_summary(user_id)

    energy = state["energy"]
    mood = state["mood"]
    recent_total = trends["recent_total"]

    system_instruction = """
    You are TamaCare, a tiny retro-style health pet.

    Write ONE short phrase (max ~3 words) that describes how you feel,
    based on the current health state. Tone:
    - slightly playful
    - no medical advice
    - no emojis

    Examples:
    - "Go exercise"
    - "I’m low on fuel."
    - "I feel strong."
    - "Rest would really help."
    Only output the sentence, nothing else.
    """

    user_context = f"""
    PET STATE:
    Energy (0-10): {energy}
    Mood (0-1): {mood:.2f}
    Recent logged health actions (last 24h): {recent_total}
    """

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[system_instruction, user_context],
        )
        text = (resp.text or "").strip()
        if text:
            return text
    except Exception as e:
        print("Gemini error in generate_state_bubble:", e)

    # fallback
    if energy <= 2:
        return "I’m exhausted."
    if mood >= 0.8:
        return "I’m feeling really good."
    return "I feel okay"


def generate_voice_url(text):
    """
    ElevenLabs hook: generate TTS and return a URL.
    For now we just return None.
    """
    return None


# -----------------------------
# Health CSV → scores → TamaCare state
# -----------------------------
CSV_PATH = "cleaned_output.csv"


def _get_val(row, col, default=None):
    if col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return val


def compute_health_from_csv(user_id, csv_path: str = CSV_PATH):
    """
    Load the latest row from cleaned_output.csv and compute:
    - sleep_score
    - activity_score
    - recovery_score
    - calm_score
    Then map to TamaCare energy (0-10) and mood (0-1).
    Returns (stats_dict, scores_dict, snapshot_dict, message_str)
    or (None, None, None, error_message) on error.
    """
    if not os.path.exists(csv_path):
        return None, None, None, f"CSV file not found at {csv_path}"

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, None, None, f"Failed to read CSV: {e}"

    if df.empty:
        return None, None, None, "CSV is empty."

    # Ensure Date/Time if present
    if "Date/Time" in df.columns:
        df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")
        df = df.sort_values("Date/Time")

    # Choose last row that has at least one key metric
    key_cols = [
        "Sleep Analysis [Total] (hr)",
        "Step Count (count)",
        "Apple Exercise Time (min)",
        "Heart Rate Variability (ms)",
    ]
    available_keys = [c for c in key_cols if c in df.columns]
    if available_keys:
        valid = df.dropna(subset=available_keys, how="all")
        if valid.empty:
            row = df.iloc[-1]
        else:
            row = valid.iloc[-1]
    else:
        row = df.iloc[-1]

    # Snapshot info for UI (raw values used)
    sleep_hours = _get_val(row, "Sleep Analysis [Total] (hr)", 0.0)
    steps = _get_val(row, "Step Count (count)", 0) or 0
    exercise_min = _get_val(row, "Apple Exercise Time (min)", 0.0) or 0.0
    hrv_ms = _get_val(row, "Heart Rate Variability (ms)", 0.0) or 0.0
    avg_hr = _get_val(row, "Heart Rate [Avg] (count/min)", 0.0) or 0.0
    env_audio = _get_val(row, "Environmental Audio Exposure (dBASPL)", 0.0) or 0.0
    headphones_audio = _get_val(row, "Headphone Audio Exposure (dBASPL)", 0.0) or 0.0
    daylight_min = _get_val(row, "Time in Daylight (min)", 0.0) or 0.0
    date_str = None
    if "Date/Time" in row.index and pd.notna(row["Date/Time"]):
        try:
            date_str = pd.to_datetime(row["Date/Time"]).strftime("%Y-%m-%d")
        except Exception:
            date_str = str(row["Date/Time"])

    # --- SleepScore (0–1) ---
    h = float(sleep_hours or 0.0)
    if h <= 0:
        sleep_score = 0.4
    elif h < 5:
        sleep_score = 0.2
    elif h < 6:
        sleep_score = 0.4
    elif h <= 8.5:
        sleep_score = 0.9
    elif h <= 10:
        sleep_score = 0.7
    else:
        sleep_score = 0.4

    # --- ActivityScore (0–1) ---
    steps_cap = min(float(steps), 15000.0)
    steps_score = steps_cap / 15000.0
    ex_cap = min(float(exercise_min), 60.0)
    exercise_score = ex_cap / 60.0
    activity_score = 0.7 * steps_score + 0.3 * exercise_score
    activity_score = max(0.0, min(1.0, activity_score))

    # --- RecoveryScore (0–1) ---
    # HRV assumed ~10–80ms typical range
    hrv = float(hrv_ms)
    hrv_score = (hrv - 10.0) / (80.0 - 10.0)
    hrv_score = max(0.0, min(1.0, hrv_score))

    # Avg HR assumed 50–100 bpm range
    hr = float(avg_hr)
    if hr <= 0:
        hr_score = 0.5
    elif hr < 60:
        hr_score = 1.0
    elif hr < 75:
        hr_score = 0.7
    elif hr <= 100:
        hr_score = 0.3
    else:
        hr_score = 0.2
    recovery_score = 0.6 * hrv_score + 0.4 * hr_score
    recovery_score = max(0.0, min(1.0, recovery_score))

    # --- CalmScore (inverse stress) ---
    # Noise: 50 dB = calm (1), 80 dB = loud (0)
    noise = max(float(env_audio), float(headphones_audio))
    if noise <= 40:
        noise_score = 1.0
    elif noise >= 85:
        noise_score = 0.0
    else:
        noise_score = 1.0 - ((noise - 40.0) / (85.0 - 40.0))
    noise_score = max(0.0, min(1.0, noise_score))

    # Daylight: 0–60min -> 0–1
    daylight_score = min(float(daylight_min), 60.0) / 60.0
    raw_stress = (1.0 - noise_score) + (1.0 - daylight_score)
    calm_score = 1.0 - min(1.0, raw_stress / 2.0)

    # --- Map scores → energy & mood ---
    energy = 10.0 * (0.5 * sleep_score + 0.5 * activity_score)
    energy = max(0.0, min(10.0, energy))

    mood = 0.4 * sleep_score + 0.3 * recovery_score + 0.3 * calm_score
    mood = max(0.0, min(1.0, mood))

    # Update user state
    state = user_state[user_id]
    state["energy"] = energy
    state["mood"] = mood

    snapshot = {
        "date": date_str,
        "sleep_hours": round(h, 2),
        "steps": int(steps),
        "exercise_min": round(float(exercise_min), 1),
        "hrv_ms": round(float(hrv_ms), 1),
        "avg_hr": round(float(avg_hr), 1),
        "env_audio": round(float(env_audio), 1),
        "headphones_audio": round(float(headphones_audio), 1),
        "daylight_min": round(float(daylight_min), 1),
    }

    scores = {
        "sleep_score": round(sleep_score, 3),
        "activity_score": round(activity_score, 3),
        "recovery_score": round(recovery_score, 3),
        "calm_score": round(calm_score, 3),
    }

    state["health_snapshot"] = snapshot
    state["health_scores"] = scores

    # Human-readable message for chat log / UI
    msg_parts = []
    if date_str:
        msg_parts.append(f"Health data from {date_str}.")
    if h > 0:
        msg_parts.append(f"Slept {h:.1f}h")
    if steps > 0:
        msg_parts.append(f"{int(steps)} steps")
    if exercise_min > 0:
        msg_parts.append(f"{exercise_min:.0f} min exercise")
    if hrv_ms > 0:
        msg_parts.append(f"HRV {hrv_ms:.0f} ms")

    message = " | ".join(msg_parts) if msg_parts else "Loaded health data from CSV."

    stats = {
        "energy": state["energy"],
        "mood": state["mood"],
        "habits_today": count_habits_today(user_id),
    }
    
    state["last_update"] = time.time()

    return stats, scores, snapshot, message


# -----------------------------
# Apple Watch / Health sync hook (manual JSON)
# -----------------------------
@app.route("/hk_sync", methods=["POST"])
def hk_sync():
    """
    Endpoint for iOS / Apple Watch / Shortcuts to send health stats.
    e.g. { "steps_today": 6200, "sleep_hours": 7.2 }
    """
    user_id = get_user_id()
    data = request.get_json(force=True)

    steps = data.get("steps_today")
    sleep_hours = data.get("sleep_hours")

    if steps is not None:
        add_habit(user_id, f"{steps} steps today (synced from watch)", source="watch")

    if sleep_hours is not None:
        add_habit(user_id, f"Slept about {sleep_hours} hours last night (synced from watch)", source="watch")

    return jsonify({"ok": True})


# -----------------------------
# NEW: Load health from CSV
# -----------------------------
@app.route("/load_health", methods=["POST"])
def load_health():
    """
    Load latest health data from cleaned_output.csv,
    map to TamaCare state, and return updated stats + snapshot.
    """
    user_id = get_user_id()
    stats, scores, snapshot, message = compute_health_from_csv(user_id)

    if stats is None:
        return jsonify({"error": message}), 400

    # Also generate a fresh bubble line based on new energy/mood
    bubble = generate_state_bubble(user_id)
    
    stats["state_bubble"] = bubble
    # Streak info (CSV load shouldn’t increment streak, but we can show if active)
    state = user_state[user_id]
    stats["streak_active"] = state.get("good_streak", 0) >= 3
    stats["streak_just_hit"] = state.pop("just_hit_streak", False)

    return jsonify({
        "message": message,
        "stats": stats,
        "health_scores": scores,
        "health_snapshot": snapshot,
    })



# -----------------------------
# Routes: index + chat + habit
# -----------------------------
@app.route("/")
def index():
    _ = get_user_id()
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    General chat, no automatic habit logging.
    """
    user_id = get_user_id()
    apply_passive_decay(user_id)  # NEW
    data = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    state = user_state[user_id]
    reply = generate_pet_reply(user_id, message, as_habit=False)
    audio_url = generate_voice_url(reply)

    ts = datetime.utcnow().isoformat()
    state["chat_history"].append({"role": "user", "text": message, "timestamp": ts})
    state["chat_history"].append({"role": "pet", "text": reply, "timestamp": ts})

    return jsonify({
        "pet_reply": reply,
        "audio_url": audio_url,
        "stats": {
            "energy": state["energy"],
            "mood": state["mood"],
            "habits_today": count_habits_today(user_id),
            "state_bubble": generate_state_bubble(user_id),
            "streak_active": state.get("good_streak", 0) >= 3,
            "streak_just_hit": state.pop("just_hit_streak", False),
        }
    })



@app.route("/habit", methods=["POST"])
def habit():
    """
    “Track Habit” — same input box, but counts as health action.
    """
    user_id = get_user_id()
    apply_passive_decay(user_id)  # NEW
    data = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    state = user_state[user_id]

    add_habit(user_id, message, source="user")
    reply = generate_pet_reply(user_id, message, as_habit=True)
    audio_url = generate_voice_url(reply)

    ts = datetime.utcnow().isoformat()
    state["chat_history"].append({"role": "user", "text": message + " [habit]", "timestamp": ts})
    state["chat_history"].append({"role": "pet", "text": reply, "timestamp": ts})

    return jsonify({
        "pet_reply": reply,
        "audio_url": audio_url,
        "stats": {
            "energy": state["energy"],
            "mood": state["mood"],
            "habits_today": count_habits_today(user_id),
            "state_bubble": generate_state_bubble(user_id),
            "streak_active": state.get("good_streak", 0) >= 3,
            "streak_just_hit": state.pop("just_hit_streak", False),
        }
    })


if __name__ == "__main__":
    app.run(debug=True)
