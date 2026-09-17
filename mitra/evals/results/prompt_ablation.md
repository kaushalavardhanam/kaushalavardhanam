# Prompt ablation (issue #9)

Prompt changes and model changes are reported separately. `SANSKRIT_SYSTEM_PROMPT_V1` is the pre-#9 prompt. `SANSKRIT_SYSTEM_PROMPT` (v2) adds Hindi bans, no greeting closer, no previous-turn reuse, 1sg agreement, and no English self-corrections.

| Condition | Model | Prompt | Grammar | Semantic | Hard fail | Gate | Notes |
|---|---|---|---:|---:|---|---|---|
| Mode A spoken 2026-09-16 | `qwen3-vl:8b-instruct` | v1 (then-current) | 2.2 | 2.2 | yes (8/10 &lt; 3) | **fail** | Actual offline default. Hindi, greeting template, `validation_ok: true` on all ten. |
| Mode B written 2026-09-08 | `us.anthropic.claude-sonnet-4-6` | v1 | 4.5 | 4.8 | no | **pass** | Existing #8 result. Reciprocal `भवान्` questions still present. |
| Mode B written 2026-09-08 | `us.openai.gpt-5.6-sol` | v1 | 4.6 | 4.8 | no | **pass** | Existing #8 result. |
| Mode B written 2026-09-08 | `us.amazon.nova-pro-v1:0` | v1 | 3.4 | 4.3 | yes | **fail** | Grammar. Surface in the PR: Nova does not clear #7. |
| Mode B written 2026-09-08 | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | v1 | 3.6 | 4.4 | yes | **fail** | Hindi `खेल` + English gloss. Surface in the PR. |
| Mode B written 2026-09-17 | `us.anthropic.claude-sonnet-4-6` | v2 | 4.2 | 4.3 | yes (sports) | **fail** | Prompt-only change vs the #8 Sonnet row. Sports leaked English self-correction and Hindi `खेलः`. |
| Mode B written Qwen | `qwen3-vl:8b-instruct` | v2 | — | — | — | not run | No Ollama / 3.7 GiB RAM. Do not attribute a Qwen fix to v2. |

v2 removes the greeting closer that every Qwen turn appended, and names the Hindi lexemes from the failing log. On Sonnet it also *primed* a Hindi+English leak on “Do you play sports?”. The live orchestrator would now retry that turn (script fail + `hindi_lexeme`). Raw Converse (this table) does not retry.

**Do not** read “Sonnet + v2” as the quality-gate configuration. The measured passing configuration remains **Sonnet 4.6 or GPT-5.6 Sol + prompt v1**.
