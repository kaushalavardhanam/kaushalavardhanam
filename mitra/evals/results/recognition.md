# Recognition corpus — measured WER/CER

Audio was **not** committed. Hypotheses are the live 2026-09-16 ASR transcripts for the ten English scenarios.

| Language | Items scored | Micro-WER | Mean item WER |
|---|---:|---:|---:|
| en | 10 | 0.048 | 0.05 |
| sa | 0 | — | no consented WAV |

Turn 8 (`Do you play sports?` → `2 play sports`) is the only error. The sentence-recognition objective from #7 is restated: English WER on this set is 0.048, below a 0.10 working bar. Residual: leading-function-word truncation / Do→2. `first_error_stage` is now `asr`.
