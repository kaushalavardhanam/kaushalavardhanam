# Coherence Check: Branch and Simulator Setup (Issue #13, sub-task 1)

This note covers the environment needed for the MITRA coherence check work.
It has four parts: the working branch, running the agent in the simulator,
logging spoken lines, and judge-model credentials.

Note: this file was written without running the simulator. Fill in the
"Verification checklist" at the bottom once you have run the agent locally.

## 1. Working branch

Branch from `asr-cpu-lightweight`:

```bash
git fetch origin
git checkout asr-cpu-lightweight
git pull --ff-only origin asr-cpu-lightweight
git checkout -b agent-mitra-coherence-check-1
```

All coherence-check work for issue #13 goes on `agent-mitra-coherence-check-1`.

## 2. Install and launch (Reachy Mini MuJoCo simulator)

`mitra/README.md` is the source of truth for installing dependencies and
launching. In short:

```bash
cd mitra
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # or the install step given in mitra/README.md
```

Start the simulator and the agent in separate terminals, following the
README:

1. Start the Reachy Mini MuJoCo simulator/daemon as described in
   `mitra/README.md`.
2. In a second terminal, with the same venv active:
   ```bash
   cd mitra
   python main.py
   ```

Expected behaviour: the agent starts without errors, listens through the ASR
(CPU-lightweight) path, and answers in Sanskrit. The simulated robot moves
or speaks accordingly.

## 3. Logging each spoken line

Use `mitra/coherence/spoken_log.py`. It appends one JSON object per line
(JSONL) to `mitra/logs/spoken_lines.jsonl`, or to the path in
`MITRA_SPOKEN_LOG`.

Call it wherever `main.py` sends a reply to TTS or speech output:

```python
from coherence.spoken_log import log_spoken_line

log_spoken_line(text=reply_text, role="agent", turn_id=turn_id)
# optionally log the user's transcribed input too:
log_spoken_line(text=user_text, role="user", turn_id=turn_id)
```

Each record contains: `ts` (UTC ISO-8601), `role`, `turn_id`, `text`, and
any extra keyword fields. Text is written as UTF-8 without ASCII escaping, so
Devanagari stays readable.

Read the log back with:

```bash
python -m coherence.spoken_log          # prints the log
```

Run this from the `mitra/` directory. The `logs/` directory is created
automatically. Do not commit log files.

## 4. Judge-model API credentials (GPT5.6)

Credentials are read from environment variables. They are never stored in
the repo.

```bash
cp mitra/.env.example mitra/.env
# edit mitra/.env and fill in the values
set -a; source mitra/.env; set +a
python -m coherence.check_env           # run from mitra/
```

Variables:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | API key for the judge model |
| `MITRA_JUDGE_MODEL` | Model identifier for GPT5.6, exactly as your API account lists it |
| `OPENAI_BASE_URL` | Optional; only if using a non-default endpoint |
| `MITRA_SPOKEN_LOG` | Optional; override the spoken-line log path |

The exact model identifier depends on the provider account, so it is
configurable through `MITRA_JUDGE_MODEL` and not hard-coded. Make sure
`mitra/.env` is git-ignored before committing anything (`git status` must
not list it).

## Verification checklist

- [ ] Branch `agent-mitra-coherence-check-1` created from `asr-cpu-lightweight`
- [ ] Dependencies installed per `mitra/README.md`
- [ ] Simulator launched; `python main.py` runs end-to-end
- [ ] At least one Sanskrit reply produced
- [ ] `spoken_lines.jsonl` receives a line per spoken reply
- [ ] `python -m coherence.check_env` reports all required variables set