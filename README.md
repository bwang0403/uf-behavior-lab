# Resurgence Verbal Behavior Experiment

A behavioral psychology experiment system for studying **resurgence** — the re-emergence of previously reinforced verbal behavior when a more recently reinforced behavior is extinguished.

---

## Table of Contents

1. [Research Design](#research-design)
2. [System Architecture](#system-architecture)
3. [File Structure](#file-structure)
4. [Setup & Installation](#setup--installation)
5. [Running an Experiment](#running-an-experiment)
6. [Configuration Reference](#configuration-reference)
7. [Word Lists](#word-lists)
8. [Data Output](#data-output)
9. [Frontend Development Guide](#frontend-development-guide)
10. [Backend Development Guide](#backend-development-guide)
11. [Known Issues & TODOs](#known-issues--todos)

---

## Research Design

### Participant Groups

| Group | Lists Used | Description |
|-------|-----------|-------------|
| 2-list | List 1 + List 2 | Standard resurgence design |
| 3-list | List 1 + List 2 + List 3 | Extended design with additional comparison list |

### Phase Structure

| Phase | Reinforced List | Extinguished | Purpose |
|-------|----------------|--------------|---------|
| 1 | List 1 (animals) | — | Establish target behavior |
| 2 | List 2 (professions) | List 1 | Establish alternative behavior |
| 3 | None | All lists | Test for resurgence of List 1 |

### Cycle Flow

Each cycle proceeds as follows:

```
Window opens (3s default)
    -> Participant speaks a word
    -> ASR transcribes it
    -> Matched against current reinforced list
    -> If correct and not repeated: chime plays (reinforcement)
    -> Log result to database
    -> Check phase-switch criteria
    -> Inter-trial interval (0.5s default)
    -> Next cycle begins
```

### Phase Switch Criteria

A phase ends when **either** condition is met:
- Participant says all words on the current reinforced list (list exhausted)
- X consecutive cycles with no detectable response (default: 5 cycles)

The experimenter can also manually advance the phase via the **NEXT PHASE** button.

---

## System Architecture

```
Browser (experimenter)          Browser (participant)
   experimenter.html                participant.html
        |  POST /cmd/<action>            |  GET /poll?since=seq
        |  GET /poll?since=seq           |
        +----------+---------------------+
                   |
              server.py  (Flask, port 5000)
                   |
              trial_loop.py  (experiment thread)
                   |
         audio.py  matcher.py  logger.py
```

### Communication Protocol

**Server -> Browser (events):** Browsers poll `GET /poll?since=<lastSeq>` every 200ms. The server returns a JSON array of new events since that sequence number. Each event has the shape:
```json
{"seq": 42, "event": "cycle_start", "data": {"window_seconds": 3.0}}
```

**Browser -> Server (commands):** HTTP POST to `/cmd/<action>`. Available actions: `start`, `stop`, `pause`, `next_phase`, `acknowledge`.

This polling architecture was chosen because:
- Server-Sent Events (SSE) was buffered by Werkzeug and events never reached the browser
- Socket.IO emission from background threads was unreliable
- Polling is simple, reliable, and requires zero configuration

### Event Types

| Event | Data Fields | Triggered When |
|-------|-------------|----------------|
| `model_status` | `ready: bool` | ASR ready status |
| `session_started` | — | START SESSION clicked and confirmed |
| `session_ended` | — | Experiment thread exits (natural or crash) |
| `session_stopped` | — | STOP button clicked |
| `session_complete` | `summary: {...}` | All 3 phases finished normally |
| `status_update` | `phase, cycle, last_response, remaining` | After every cycle |
| `cycle_start` | `window_seconds` | Recording window opens |
| `cycle_end` | — | Recording window closes |
| `iti_change` | `active: bool` | Inter-trial interval starts/ends |
| `pause_change` | `paused: bool` | PAUSE or RESUME clicked |
| `instruction` | `text: str` | Instruction screen shown to participant |
| `instruction_done` | — | CONTINUE clicked by experimenter |
| `server_error` | `message: str` | Unhandled exception in experiment thread |

---

## File Structure

```
E:\Lab\
├── main.py                   Entry point. Pre-flight checks, config validation, launches server.
├── server.py                 Flask backend. Event log, HTTP routes, model preload.
├── trial_loop.py             Core experiment loop. ExperimentRunner class.
├── audio.py                  Microphone recording (VAD) and reinforcement sound playback.
├── matcher.py                Word list management and response matching (singular/plural aware).
├── logger.py                 SQLite session logging and CSV export.
├── config.yaml               All tunable parameters. Edit before each session.
├── requirements.txt          Python dependencies.
├── run.bat                   Double-click launcher for lab use.
│
├── frontend/
│   ├── experimenter.html     Experimenter control panel (served at http://localhost:5000/)
│   └── participant.html      Participant display screen (served at http://localhost:5000/participant)
│
├── list/
│   ├── list1_animals.txt     ~50 animal words (one per line, lowercase)
│   ├── list2_professions.txt ~50 profession words
│   ├── list3_household.txt   ~50 household items (3-list group only)
│   └── list_practice.txt    10 practice words (shown before Phase 1)
│
├── audio/
│   └── reward.wav            Reinforcement chime (auto-generated if missing)
│
└── data/
    ├── resurgence.db         SQLite database (all sessions, all participants)
    └── P001_20260228_*.csv   Per-session CSV export
```

---

## Setup & Installation

### Prerequisites

- Windows 10/11
- Anaconda or Miniconda
- A working microphone

### First-time Setup

```bash
# 1. Create and activate the conda environment
conda create -n resurgence python=3.10
conda activate resurgence

# 2. Install dependencies
cd E:\Lab
pip install -r requirements.txt

# 3. Verify microphone works
python audio.py
# You should hear a chime, then it will record for 3 seconds
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `faster-whisper` | Local speech recognition (ASR) |
| `openai` | Optional hosted ASR provider |
| `sounddevice` | Microphone recording and audio playback |
| `soundfile` | WAV file reading/writing |
| `numpy` | Audio signal processing |
| `pyyaml` | Config file parsing |
| `flask` | Web server |

### ASR Provider

By default, `config.yaml` uses local `faster-whisper`:

```yaml
asr:
  provider: "local"
  model_size: "base"
```

The first local run downloads the Whisper `base` model (~150MB) automatically. It is cached at:
```
C:\Users\<username>\.cache\huggingface\
```
After the first download, the model loads from cache (no internet needed).

For more consistent recognition across devices, use hosted ASR instead:

```yaml
asr:
  provider: "openai"
  openai_model: "gpt-4o-transcribe"
```

Then set `OPENAI_API_KEY` in the environment before starting the app:

```powershell
$env:OPENAI_API_KEY = "sk-..."
python main.py
```

Hosted ASR still uses the same microphone recording flow, but transcription no longer depends on the laptop's local Whisper model or model cache.

---

## Running an Experiment

### Before Each Session

1. Open `config.yaml` and update:
   ```yaml
   experiment:
     participant_id: "P002"   # unique ID for this participant
     group: 2                 # 2 or 3
   ```

2. Verify the word lists are finalized in `list/`.

### Research Decisions To Finalize Before Data Collection

Before running real participants, the research team should explicitly finalize the items below and keep them stable across sessions unless the protocol says otherwise:

- **Participant assignment**
  `experiment.participant_id` naming convention, plus whether the participant is assigned to the 2-list or 3-list group via `experiment.group`.
- **Final stimulus sets**
  The exact contents of `list1`, `list2`, `list3`, and the practice list. The matcher treats regular singular/plural forms as equivalent, but it does **not** handle irregular plurals well (`person/people`, `mouse/mice`), so lists should avoid those forms or the code should be extended first.
- **Phase-switching rule**
  The threshold for `switching.no_response_streak`, and whether manual `NEXT PHASE` presses are allowed in the actual protocol or only during pilot/testing sessions.
- **Timing parameters**
  `trial.window_seconds`, `trial.min_response_gap`, and `trial.iti_seconds`. These determine how long a participant can respond, how much post-speech silence ends a recording, and the gap between cycles.
- **ASR settings**
  `asr.provider`, `asr.model_size`, `asr.openai_model`, and `asr.language`. The default CPU-friendly local setting is `base`, but if recognition accuracy is not acceptable in pilot runs, decide whether to move to local `small` or hosted `openai`.
- **Practice-round policy**
  Whether `practice.enabled` stays on, how long practice should run (`practice.no_response_streak`), and what instruction text is shown before practice.
- **Instruction wording**
  The exact text shown in `instructions.intro`, `instructions.between_phases`, and `instructions.end`, because those prompts become part of the participant-facing procedure.
- **Analysis rule for resurgence**
  The current code counts resurgence as a **List-1 response in Phase 3 that is not a repeat**. Confirm this matches the intended analysis plan before collecting formal data.

### Starting the Server

```bash
# Anaconda Prompt
conda activate resurgence
cd E:\Lab
python main.py
```

The terminal will show session info and ask `Start server? (y/n)`. Type `y`.

Two browser windows will open automatically:
- **Experimenter panel**: `http://localhost:5000/` — control interface
- **Participant screen**: `http://localhost:5000/participant` — shown to the participant (full screen)

### Session Flow

```
Experimenter panel shows "LOADING WHISPER MODEL..."
    -> Model loads (10-30 seconds on first run, ~5s from cache)
    -> Panel shows "MODEL READY — click START SESSION"
    -> Experimenter clicks START SESSION
    -> Participant screen shows intro instruction text
    -> Participant clicks "Start Task"
    -> Phase words are flashed one-by-one on the screen (2 seconds each to control reading pacing)
    -> 3-2-1 Countdown begins
    -> Cycle 1 begins (blue arc countdown visible on participant screen)
    -> ... cycles run automatically ...
    -> Phase switch (auto or manual NEXT PHASE button)
    -> Next phase's shuffled word list is flashed one-by-one
    -> Phases 2 and 3 run similarly
    -> Session complete: summary shown on experimenter panel
    -> Data saved to data/ folder
```

### Experimenter Controls

| Button | Function |
|--------|----------|
| START SESSION | Begin experiment (requires ASR ready) |
| STOP | Immediately end session (data is saved) |
| PAUSE / RESUME | Freeze experiment mid-session |
| NEXT PHASE | Manually advance to next phase immediately |
| CONTINUE | Dismiss instruction screen (appears during instructions) |

### Participant Screen States

| Display | Meaning |
|---------|---------|
| "WAITING TO START" | Session not yet started |
| Instruction text (full screen) | Read instructions; participant must click "Start Task" |
| Flashing Words (Word List) | Phase-specific words are presented one at a time for 2 seconds each to control exposure |
| 3-2-1 Countdown | Getting ready to start the actual response cycles |
| Blue arc counting down | Recording window is open — participant should speak now |
| Arc stops | Cycle ended; wait for next cycle |
| "PAUSED" overlay | Session is paused |
| "SESSION COMPLETE" | All done |

---

## Configuration Reference

All parameters are in `config.yaml`. Edit this file before each session.

```yaml
# ─── Experiment Identity ───────────────────────────────────────────────────────
experiment:
  participant_id: "P001"      # Shown in experimenter panel; used as filename prefix
  group: 2                    # 2 = two-list group, 3 = three-list group

# ─── Phase Definitions ────────────────────────────────────────────────────────
phases:
  - name: "Phase 1"
    reinforced_list: 1        # List number that earns reinforcement (chime)
    extinction_lists: []

  - name: "Phase 2"
    reinforced_list: 2
    extinction_lists: [1]

  - name: "Phase 3"
    reinforced_list: null     # null = no reinforcement (full extinction)
    extinction_lists: [1, 2, 3]

# ─── Phase Switching ──────────────────────────────────────────────────────────
switching:
  no_response_streak: 5       # Switch phase after this many consecutive no-response cycles

# ─── Trial / Cycle Timing ─────────────────────────────────────────────────────
trial:
  window_seconds: 3.0         # Max duration of each response window
  min_response_gap: 0.3       # Seconds of silence before cutting off recording
  iti_seconds: 0.5            # Inter-trial interval between cycles (0 = no gap)

# ─── Word Lists ───────────────────────────────────────────────────────────────
lists:
  path: "list/"
  list1: "list1_animals.txt"
  list2: "list2_professions.txt"
  list3: "list3_household.txt"  # Only used if group: 3

# ─── Audio ────────────────────────────────────────────────────────────────────
audio:
  reinforcement_sound: "audio/reward.wav"   # Auto-generated if missing
  sample_rate: 16000
  channels: 1

# ─── Speech Recognition ───────────────────────────────────────────────────────
asr:
  provider: "local"          # local or openai
  model_size: "base"          # base (fast) or small (more accurate, slower)
  beam_size: 5
  openai_model: "gpt-4o-transcribe"
  prefer_vocabulary_matches: true
  language: "en"              # Language code

# ─── Practice Round ───────────────────────────────────────────────────────────
practice:
  enabled: true               # Set false to skip
  list: "list/list_practice.txt"
  no_response_streak: 3       # End practice after 3 consecutive no-responses
  instruction: |
    Practice round. Say any word from the practice list.
    You will hear a chime for correct answers.

# ─── Instruction Texts ────────────────────────────────────────────────────────
instructions:
  intro: |
    Welcome to the experiment. ...
  between_phases: |
    Please wait. The next part will begin shortly.
  end: |
    The experiment is complete. Thank you!

# ─── Data Storage ─────────────────────────────────────────────────────────────
data:
  db_path: "data/resurgence.db"
  export_csv: true
```

---

## Word Lists

Word lists are plain text files in `list/`, one word per line, lowercase.

```
cat
dog
elephant
...
```

### Matching Rules

- Case-insensitive: "CAT", "Cat", "cat" all match "cat"
- Singular/plural equivalent: "cats" matches "cat" and vice versa
- A word can only be reinforced **once per session** — saying it again counts as a repeat
- Words not on any list are recorded as "novel"

### Irregular plurals

The matcher uses rule-based pluralization and does **not** handle irregular forms (mouse/mice, person/people, etc.). When finalizing word lists, prefer regular plurals or words where singular=plural.

### Practice List

`list/list_practice.txt` contains words used only in the practice round. These are not logged in the experiment data. Current default words:
```
apple, chair, pencil, window, flower, bottle, clock, blanket, candle, pillow
```

---

## Data Output

### SQLite Database

Located at `data/resurgence.db`. Contains all sessions from all participants.

**`sessions` table** — one row per session:
| Column | Type | Description |
|--------|------|-------------|
| session_id | TEXT | `P001_20260228_143022` |
| participant | TEXT | Participant ID from config |
| group_num | INTEGER | 2 or 3 |
| started_at | TEXT | ISO timestamp |
| ended_at | TEXT | ISO timestamp |

**`trials` table** — one row per cycle:
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment |
| session_id | TEXT | Links to sessions table |
| phase | INTEGER | 1, 2, or 3 |
| cycle | INTEGER | Cycle number within session (1-indexed) |
| timestamp | REAL | Unix time when cycle started |
| response_raw | TEXT | Raw ASR transcript (NULL if no response) |
| response_word | TEXT | Canonical matched word (NULL if no match) |
| matched_list | INTEGER | Which list it matched (NULL if novel or no response) |
| is_novel | INTEGER | 1 = word not on any list |
| is_repeat | INTEGER | 1 = word already said this session |
| reinforced | INTEGER | 1 = chime played |
| response_time | REAL | Seconds from cycle start to first speech (NULL if no response) |

### CSV Export

A CSV is automatically exported to `data/` at the end of each session. File name format:
```
P001_20260228_143022.csv
```

### Analysis Examples

```python
import pandas as pd

df = pd.read_csv("data/P001_20260228_143022.csv")

# Response rate per phase
df.groupby('phase')['response_word'].count()

# Resurgence: List-1 novel responses in Phase 3
resurgence = df[(df.phase == 3) & (df.matched_list == 1) & (df.is_repeat == 0)]
print(f"Resurgence responses: {len(resurgence)}")

# Mean response time per phase
df.dropna(subset=['response_time']).groupby('phase')['response_time'].mean()

# Novel responses (not on any list)
df[df.is_novel == 1]
```

---

## Frontend Development Guide

The two HTML files in `frontend/` are self-contained — no build step, no npm, no framework. They use vanilla JavaScript with a simple polling loop.

### Adding a New Event Handler

In either HTML file, find the `handleEvent(event, data)` function and add a new `case`:

```javascript
function handleEvent(event, data) {
  switch(event) {
    case 'cycle_start':
      startCycle(data.window_seconds);
      break;

    // Add your new event here:
    case 'my_new_event':
      doSomething(data.my_field);
      break;
  }
}
```

### Adding a New Button/Command

Add a button in the HTML:
```html
<button id="btn-myaction" onclick="post('myaction')" disabled>MY ACTION</button>
```

Add the command handler in `server.py`:
```python
elif action == "myaction":
    if _runner:
        _runner.do_something()
    return jsonify({"ok": True})
```

### Participant Screen Layout

The participant screen uses a fixed 520x520 SVG stage centered on the page:
- **Arc**: `<circle id="arc">` — stroke-dashoffset controls fill; `renderArc(ratio)` updates it
- **Time display**: `<div id="time-display">` — shows countdown number
- **Status text**: `<div id="status-text">` — shows "WAITING TO START" etc.
- **Instruction overlay**: `<div id="instruction-overlay">` — full-screen dark overlay
- **Pause overlay**: `<div id="pause-overlay">` — semi-transparent amber overlay

To change colors or timing, edit `arcColor()` (color lerp) and `startCycle()` (setInterval logic).

### Experimenter Panel Layout

The experimenter panel is a centered vertical stack of cards (460px wide):
- **Stats card**: phase, cycle, last response, remaining count
- **Button row 1**: START / STOP
- **Button row 2**: PAUSE / NEXT PHASE
- **CONTINUE button**: hidden by default, shown during instruction screens
- **Summary card**: hidden by default, shown after session complete

---

## Backend Development Guide

### Adding a New Event

In `server.py`, call `_push()` anywhere (thread-safe):

```python
_push("my_event", {"key": "value"})
```

The event will be delivered to all polling browsers within 200ms.

### Adding a New Callback in ExperimentRunner

In `trial_loop.py`:

```python
class ExperimentRunner:
    def __init__(self, ...):
        ...
        self.on_my_event = None   # add the callback attribute

    def some_method(self):
        ...
        if self.on_my_event:
            self.on_my_event(some_data)
```

In `server.py`, wire it up after creating the runner:

```python
_runner.on_my_event = lambda data: _push("my_event", {"value": data})
```

### Modifying the Experiment Loop

The main loop in `trial_loop.py` is `ExperimentRunner.run()`. The cycle structure is:

```python
while self.running and self.current_phase_idx < len(phases):
    self._wait_if_paused()          # blocks if paused

    # 1. Record
    wav_path, response_time = record_until_silence(...)

    # 2. Transcribe
    word, transcript = transcribe(
        self.model,
        wav_path,
        language,
        vocabulary=self._experiment_vocabulary(),
        asr_cfg=self.cfg.get("asr", {}),
    )

    # 3. Match
    match_result = self.list_manager.match(word)

    # 4. Reinforce
    if should_reinforce(match_result, reinforced_list):
        play_reinforcement(self.reward_path)

    # 5. Log
    self.logger.log_trial(...)

    # 6. ITI
    time.sleep(iti_seconds)

    # 7. Check phase switch
    if check_phase_switch(...):
        self.current_phase_idx += 1
```

### Thread Safety Notes

- `_push()` in server.py is thread-safe (uses `_log_lock`)
- `ExperimentRunner.run()` runs in a daemon thread; all other methods may be called from the Flask thread
- `_pause_event` and `_instruction_event` are `threading.Event` objects — safe to set/clear from any thread
- `_force_phase_switch` is a simple bool flag — technically a race condition but harmless in practice (worst case: skips one extra cycle)
- Never call Flask's `jsonify()` or `request` from the experiment thread

### Adding a New Config Parameter

1. Add to `config.yaml` with a sensible default and inline comment
2. Access in code via `self.cfg["section"]["key"]`
3. Document in the Configuration Reference section of this README

---

## Known Issues & TODOs

### Bugs

- [ ] **No visual feedback after response**: The participant screen arc just stops when a cycle ends. There is no indication of whether the response was reinforced or not. A brief flash (green for reinforcement, no change for no reinforcement) would improve usability.

- [ ] **Phase name not shown**: The experimenter panel shows phase number (1/2/3) but not the phase name from config.

### Features Planned

- [ ] **Response feedback on participant screen**: Flash color or text after each response
- [ ] **Experimenter notes field**: Free-text field on experimenter panel to note observations
- [ ] **Live response rate graph**: Show a running chart of responses per phase on experimenter panel
- [ ] **3-list group end-to-end test**: 2-list group tested and working; 3-list group needs validation
- [ ] **Export to R-friendly format**: Add an optional export with computed variables (resurgence onset, response rate per 5-cycle bin, etc.)

### Research Parameters (To Be Confirmed)

- [ ] Exact word lists — finalize the real stimuli in `list/` before data collection
- [ ] `no_response_streak` threshold value — confirm the final switching rule from the protocol/literature
- [ ] Whether `window_seconds`, `min_response_gap`, or `iti_seconds` should differ between phases
- [ ] Whether manual `NEXT PHASE` overrides are allowed outside pilot/testing sessions
- [ ] Whether to track inter-response time *across* cycles (currently only within-cycle RT is logged)
- [ ] Definition of resurgence for analysis — currently: any List-1 response in Phase 3 that is not a repeat

---

## Troubleshooting

### "Session already running" error

This should not happen with the current code. If it does:
1. Stop the server (Ctrl+C in terminal)
2. Restart with `python main.py`

### No chime when participant speaks

1. Check `data/resurgence.db` — is the cycle being logged? (Use DB Browser for SQLite)
2. Check terminal output — what word is being recognized?
3. If the word is recognized but not reinforced, it may be a repeat or not on the current phase's list
4. If no word is recognized, check microphone levels — RMS should be ~0.027 when speaking; adjust `silence_threshold` in `audio.py`

### Arc countdown not animating

1. Open browser DevTools (F12) -> Console — any errors?
2. Open `http://localhost:5000/poll?since=0` in browser — do you see `cycle_start` events after clicking Start?
3. If events are present but arc is not animating, check `handleEvent` in `participant.html`

### ASR keeps transcribing wrong words

1. Use words from the current experiment list; practice words such as `mountain` will not chime during Phase 1.
2. Try `model_size: "small"` in `config.yaml` for local ASR (more accurate, slower).
3. For cross-device consistency, set `asr.provider: "openai"` and set `OPENAI_API_KEY`.
4. Check for background noise and microphone placement.

### Database locked error

Only one experiment session can write to the database at a time. If you see a locked error:
1. Ensure only one instance of `python main.py` is running
2. Check if a previous session crashed mid-write — restart the server