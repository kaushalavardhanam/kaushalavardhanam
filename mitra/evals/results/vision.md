# Vision + tool-result comparison (issue #9 §10)

**Images:** `evals/fixtures/vision/{apple,croissant,duck}.jpg` — synthetic stand-ins, identical across candidates, **not** MuJoCo `minimal` captures. A simulator JPEG swap must keep these filenames.

**Prompt:** `SANSKRIT_SYSTEM_PROMPT` (v2) + `VISION_JSON_INSTRUCTION` + “What is this?”  
**Qwen:** not run (no Ollama).  
**Ollama contacted:** false on both Bedrock rows.

| Object | Model | object_en | Generated name | Spoken after lexicon | Grounded? | Latency (s) |
|---|---|---|---|---|---|---:|
| apple | Sonnet 4.6 | apple | सेवफलम् | एतत् सेवफलम् अस्ति। | yes | 1.54 |
| apple | Nova Pro | apple | सेवफलम् | एतत् सेवफलम् अस्ति। | yes | 2.15 |
| croissant | Sonnet 4.6 | dumpling / half-moon pastry | पिष्टकः | एतत् पिष्टकः अस्ति। | partial (pastry, not croissant) | 2.12 |
| croissant | Nova Pro | **fan** | क्रितान्का | एतत् क्रितान्का अस्ति। | no | 1.92 |
| duck | Sonnet 4.6 | duck | हंसशावकः | एतत् कारण्डवः अस्ति। (seed lexicon) | yes | 1.92 |
| duck | Nova Pro | duck | हंस | एतत् कारण्डवःो अस्ति। (seed lexicon, broken sandhi) | yes | 2.09 |

Lexicon: apple is seed-verified (`सेवफलम्`) and present after substitution. Duck is also in the seed as `कारण्डवः`; the corpus marks it unverified, but the store still substitutes — that is why spoken ≠ generated for duck.

`capture_image` was not invoked: this helper sends the JPEG in the user message (Mode B vision). Tool-argument validity on the live `capture_image` path still needs the simulator.

**Conclusion:** On identical images, both VLMs name a red-circle apple correctly. Sonnet is more grounded on the croissant stand-in; Nova invents “fan”. Duck naming is lexicon-dominated. This is **not** a MuJoCo vision pass.
