# Resurgence Verbal Behavior Experiment

A behavioral psychology experiment system for studying **resurgence** — the re-emergence of previously reinforced verbal behavior when a more recently reinforced behavior is extinguished.

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
| ----- | ----- | ----- |
| **2-list** | List 1 + List 2 | Standard resurgence design |
| **3-list** | List 1 + List 2 + List 3 | Extended design with additional comparison list |

### Phase Structure
| Phase | Reinforced List | Extinguished | Purpose |
| ----- | ----- | ----- | ----- |
| **1** | List 1 (animals) | — | Establish target behavior |
| **2** | List 2 (professions) | List 1 | Establish alternative behavior |
| **3** | None | All lists | Test for resurgence of List 1 |

### Cycle Flow
Each cycle proceeds as follows:
1. Window opens (3s default)
2. Participant speaks a word
3. ASR transcribes it
4. Matched against current reinforced list
5. If correct and not repeated: chime plays (reinforcement)
6. Log result to database
7. Check phase-switch criteria
8. Inter-trial interval (0.5s default)
9. Next cycle begins

### Phase Switch Criteria
A phase ends when **either** condition is met:
* Participant says all words on the current reinforced list (list exhausted)
* `X` consecutive cycles with no detectable response (default: 5 cycles)

The experimenter can also manually advance the phase via the **NEXT PHASE** button.

---

## System Architecture

```text
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
* **Server -> Browser (events):** Browsers poll `GET /poll?since=<lastSeq>` every 200ms. The server returns a JSON array of new events since that sequence number.
* **Browser -> Server (commands):** HTTP POST to `/cmd/<action>`. Available actions: `start`, `stop`, `pause`, `next_phase`, `acknowledge`.

### Event Types
| Event | Data Fields | Triggered When |
| ----- | ----- | ----- |
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
| `phase_change` | Phase metadata | When moving to a new phase |

---

## File Structure

```text
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
│   └── list_practice.txt     10 practice words (shown before Phase 1)
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
* Windows 10/11
* Anaconda or Miniconda
* A working microphone

### First-time Setup
```bash
# 1. Create and activate the conda environment
conda create -n resurgence python=3.10
conda activate resurgence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify microphone works
python audio.py  # You should hear a chime, then it will record for 3 seconds
```

### ASR Provider
By default, `config.yaml` uses local `faster-whisper`:
```yaml
asr:
  provider: "local"
  model_size: "base"
```
The first local run downloads the Whisper base model (~150MB) automatically. After the first download, the model loads from cache (no internet needed).

For more consistent recognition across devices, use hosted ASR instead:
```yaml
asr:
  provider: "openai"
  openai_model: "gpt-4o-transcribe"
```
Then set `OPENAI_API_KEY` in the environment before starting the app.

---

## Running an Experiment

### Before Each Session
1. Open `config.yaml` and update the `experiment` section (`participant_id`, `group`).
2. Verify the word lists are finalized in `list/`.

### Starting the Server
```bash
conda activate resurgence
python main.py
```
Two browser windows will open automatically:
* **Experimenter panel**: `http://localhost:5000/` — control interface
* **Participant screen**: `http://localhost:5000/participant` — shown to the participant (full screen)

### Session Flow
```text
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
| ----- | ----- |
| **START SESSION** | Begin experiment (requires ASR ready) |
| **STOP** | Immediately end session (data is saved) |
| **PAUSE / RESUME** | Freeze experiment mid-session |
| **NEXT PHASE** | Manually advance to next phase immediately |
| **CONTINUE** | Dismiss instruction screen (appears during instructions) |

### Participant Screen States
| Display | Meaning |
| ----- | ----- |
| **"WAITING TO START"** | Session not yet started |
| **Instruction text (full screen)** | Read instructions; participant must click "Start Task" |
| **Flashing Words (Word List)** | Phase-specific words are presented one at a time for 2 seconds each to control exposure |
| **3-2-1 Countdown** | Getting ready to start the actual response cycles |
| **Blue arc counting down** | Recording window is open — participant should speak now |
| **Arc stops** | Cycle ended; wait for next cycle |
| **"PAUSED" overlay** | Session is paused |
| **"SESSION COMPLETE"** | All done |

---

## Configuration Reference
All parameters are in `config.yaml`. Edit this file before each session. Detailed comments are included inside the file itself.

---

## Word Lists
Word lists are plain text files in `list/`, one word per line, lowercase.
### Matching Rules
* **Case-insensitive**: "CAT", "Cat", "cat" all match "cat"
* **Singular/plural equivalent**: "cats" matches "cat" and vice versa
* A word can only be reinforced *once per session* — saying it again counts as a repeat
* Words not on any list are recorded as "novel"

*Note: The matcher uses rule-based pluralization and does not handle irregular forms well (e.g., mouse/mice).*

---

## Data Output
### SQLite Database
Located at `data/resurgence.db`. Contains all sessions from all participants. Includes tables for `sessions` and `trials`.

### CSV Export
A CSV is automatically exported to `data/` at the end of each session. File name format: `P001_20260228_143022.csv`.

---

## Frontend Development Guide
The two HTML files in `frontend/` are self-contained. They use vanilla JavaScript with a simple polling loop.
To modify participant screen timing or word flash animations, locate the respective CSS (`@keyframes fadeInOut`) or JS (`flashWordsOneByOne`, `startCycle`) in `participant.html`.

## Backend Development Guide
In `server.py`, use `_push()` to trigger events to the frontend.
The core logic resides in `trial_loop.py` inside the `ExperimentRunner` class.

## Known Issues & TODOs
* [ ] **Response feedback on participant screen**: Flash color or text after each response.
* [ ] **Experimenter notes field**: Free-text field on experimenter panel to note observations.
* [ ] **Live response rate graph**: Show a running chart of responses per phase on experimenter panel.
