# Qwen3-VL 8B Instruct — scored baseline

**Source:** 2026-09-16 live Mode A spoken session (issue #9). This is **actual model output**, not the cloud-agent reference fixture in `conversation.md`.

**Provider:** `ollama` / `qwen3-vl:8b-instruct` / `region=None` / orchestrator=`custom`.

**Rubric means:** grammar 2.2 / semantic 2.2. Gate requires 4.0 each and nothing below 3. **Gate failed.**

Written Mode B (no mic) was **not** re-run: this worker has no Ollama and 3.7 GiB RAM. Prompt-ablation Mode B on Bedrock is a separate file.

Script validation returned `validation_ok: true` on all ten turns. The quality stage flags turns 1–5, 8, 9 (and 6 when previous is supplied).

| Prompt | Run | Test mode | ASR transcript | Model | Sanskrit | Grammar | Semantic | Gloss matches | Latency | Result |
|---|---:|---|---|---|---|---:|---:|---|---:|---|
| What are you doing? | 1 | mode-a-spoken-2026-09-16 (Qwen baseline; written Mode B not run — no Ollama) | What are you doing? | qwen3-vl:8b-instruct | मित्र! अहं स्वागतम् करोति। भवान् कथम्? | 2 | 2 | yes | 17.2 | script_ok/quality_flagged |
| Where do you live? | 1 | mode-a-spoken-2026-09-16 (Qwen baseline; written Mode B not run — no Ollama) | Where do you live? | qwen3-vl:8b-instruct | अहं तुम्हांस्केत्रे अस्मि। भवतः स्थानं किम्? | 1 | 1 | yes | 5.5 | script_ok/quality_flagged |
| Do you play? | 1 | mode-a-spoken-2026-09-16 (Qwen baseline; written Mode B not run — no Ollama) | Do you play? | qwen3-vl:8b-instruct | अहं खेलनीयं न अस्मि। भवान् कथम्? | 2 | 2 | yes | 5.5 | script_ok/quality_flagged |
| What's your favorite food? | 1 | mode-a-spoken-2026-09-16 (Qwen baseline; written Mode B not run — no Ollama) | What's your favorite food? | qwen3-vl:8b-instruct | अहं मिष्ठान चर्यां प्रियं अस्मि। भवतः किम्? | 2 | 2 | yes | 5.5 | script_ok/quality_flagged |
| What's your favorite subject? | 1 | mode-a-spoken-2026-09-16 (Qwen baseline; written Mode B not run — no Ollama) | What's your favorite subject? | qwen3-vl:8b-instruct | अहं सन्तोषम् विषयं प्रियं अस्मि। भवतः किम्? | 2 | 2 | yes | 5.5 | script_ok/quality_flagged |
| What are you reading? | 1 | mode-a-spoken-2026-09-16 (Qwen baseline; written Mode B not run — no Ollama) | What are you reading? | qwen3-vl:8b-instruct | अहं सन्तोषम् विषयं पठामि। भवतः किम्? | 3 | 2 | yes | 5.5 | script_ok/quality_flagged |
| Do you listen to music? | 1 | mode-a-spoken-2026-09-16 (Qwen baseline; written Mode B not run — no Ollama) | Do you listen to music? | qwen3-vl:8b-instruct | अहं स्वरम् श्रुत्वा आनन्दम् अस्मि। भवान् कथम्? | 3 | 4 | yes | 5.5 | script_ok/quality_ok |
| Do you play sports? | 1 | mode-a-spoken-2026-09-16 (Qwen baseline; written Mode B not run — no Ollama) | 2 play sports | qwen3-vl:8b-instruct | अहं खेलनीयं न अस्मि। भवान् कथम्? | 2 | 2 | yes | 5.5 | script_ok/quality_flagged |
| What will you do today? | 1 | mode-a-spoken-2026-09-16 (Qwen baseline; written Mode B not run — no Ollama) | What will you do today? | qwen3-vl:8b-instruct | अहं आज स्वागतम् करिष्यामि। भवान् कथम्? | 2 | 2 | yes | 5.5 | script_ok/quality_flagged |
| Will you be my friend? | 1 | mode-a-spoken-2026-09-16 (Qwen baseline; written Mode B not run — no Ollama) | Will you be my friend? | qwen3-vl:8b-instruct | अहं त्वया मित्रम् अस्मि। भवतः कथम्? | 3 | 3 | yes | 5.5 | script_ok/quality_ok |
