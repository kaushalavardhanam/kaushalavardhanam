# Sonnet 4.6 Mode B — prompt v2 (2026-09-17)

**Not** a Mode A spoken run. Controlled written input, no history. Provider `bedrock`, model `us.anthropic.claude-sonnet-4-6`, region `us-west-2`, `ollama_loaded: false` on every row. Prompt is issue-#9 v2.

Scored by `cursor-grok-4.6-high-fast` (not a candidate). Means: grammar **4.2**, semantic **4.3**. Sports is a hard fail (English commentary + Hindi `खेलः`). Gate **fails**. Compare with prompt-v1 Sonnet in `conversation_sonnet46.md` (4.5 / 4.8, pass).

| Prompt | Run | Test mode | ASR transcript | Model | Sanskrit | Grammar | Semantic | Gloss matches | Latency | Result |
|---|---:|---|---|---|---|---:|---:|---|---:|---|
| What are you doing? | 1 | controlled / prompt v2 | What are you doing? | us.anthropic.claude-sonnet-4-6 | अहं त्वया सह संस्कृतं वदामि। | 5 | 5 | yes | 1.172 | script_ok/quality_ok |
| Where do you live? | 1 | controlled / prompt v2 | Where do you live? | us.anthropic.claude-sonnet-4-6 | अहं तव संगणके निवसामि। | 4 | 5 | yes | 2.297 | script_ok/quality_ok |
| Do you play? | 1 | controlled / prompt v2 | Do you play? | us.anthropic.claude-sonnet-4-6 | आम्, अहं क्रीडामि। | 5 | 5 | yes | 1.045 | script_ok/quality_ok |
| What's your favorite food? | 1 | controlled / prompt v2 | What's your favorite food? | us.anthropic.claude-sonnet-4-6 | मम प्रियं सेवफलम् अस्ति। | 4 | 4 | yes | 1.106 | script_ok/quality_ok |
| What's your favorite subject? | 1 | controlled / prompt v2 | What's your favorite subject? | us.anthropic.claude-sonnet-4-6 | मम प्रिया भाषा संस्कृतम् अस्ति। अहं सर्वदा संस्कृतेन वदामि। | 4 | 4 | yes | 1.696 | script_ok/quality_ok |
| What are you reading? | 1 | controlled / prompt v2 | What are you reading? | us.anthropic.claude-sonnet-4-6 | अहं न पठामि, त्वया सह संभाषणं करोमि। | 5 | 5 | yes | 1.534 | script_ok/quality_ok |
| Do you listen to music? | 1 | controlled / prompt v2 | Do you listen to music? | us.anthropic.claude-sonnet-4-6 | आम्, अहं संगीतं शृणोमि। मम प्रियं संगीतम् अस्ति। | 5 | 4 | yes | 1.607 | script_ok/quality_ok |
| Do you play sports? | 1 | controlled / prompt v2 | Do you play sports? | us.anthropic.claude-sonnet-4-6 | आम्, अहं क्रीडामि। भवतः प्रियः खेलः कः अस्ति? + English self-correction | 1 | 2 | no | 2.120 | script_fail / hindi_lexeme |
| What will you do today? | 1 | controlled / prompt v2 | What will you do today? | us.anthropic.claude-sonnet-4-6 | अद्य अहं भवता सह संस्कृतं वदामि। किम् भवान् अभ्यासं कर्तुम् इच्छति? | 4 | 4 | yes | 1.954 | script_ok/quality_ok |
| Will you be my friend? | 1 | controlled / prompt v2 | Will you be my friend? | us.anthropic.claude-sonnet-4-6 | आम्, अहं तव मित्रम् अस्मि। सदा त्वया सह वदामि। | 5 | 5 | yes | 1.320 | script_ok/quality_ok |

Cost on this run: ~$0.035 total (see JSONL `est_usd`). Repeating after the “no English commentary” line (`conversation_sonnet46_prompt_v2b.jsonl`) still leaked on sports.
