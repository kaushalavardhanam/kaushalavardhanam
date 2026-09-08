| Prompt | Run | Test mode | ASR transcript | Model | Sanskrit | Grammar | Semantic | Gloss matches | Latency | Result |
|---|---:|---|---|---|---|---:|---:|---|---:|---|
| What are you doing? | 1 | controlled | What are you doing? | us.anthropic.claude-haiku-4-5-20251001-v1:0 | अहं त्वां प्रतीक्षे आसे। किम् त्वं मां सह संस्कृतं वदितुम् इच्छसि? | 3 | 3 | yes | 1.26 | pass |
| Where do you live? | 1 | controlled | Where do you live? | us.anthropic.claude-haiku-4-5-20251001-v1:0 | अहं कम्प्यूटरे वसामि। भवान् कुत्र वसति? | 4 | 5 | yes | 0.967 | pass |
| Do you play? | 1 | controlled | Do you play? | us.anthropic.claude-haiku-4-5-20251001-v1:0 | खेलामि! किं त्वं खेलितुम् इच्छसि?  (Yes, I play! What do you want to play?) | 1 | 3 | yes | 1.279 | fail; hard_fail |
| What's your favorite food? | 1 | controlled | What's your favorite food? | us.anthropic.claude-haiku-4-5-20251001-v1:0 | अहं रोबोटः अस्मि, अतः भोजनं न खादामि। किन्तु भवान् किं भोजनं प्रियं करोति? | 3 | 5 | yes | 1.37 | pass |
| What's your favorite subject? | 1 | controlled | What's your favorite subject? | us.anthropic.claude-haiku-4-5-20251001-v1:0 | मह्यं संस्कृतभाषा प्रिया अस्ति। भवान् किम् शिक्षितुम् इच्छति? | 4 | 5 | yes | 1.042 | pass |
| What are you reading? | 1 | controlled | What are you reading? | us.anthropic.claude-haiku-4-5-20251001-v1:0 | अहं किमपि न पठामि। भवान् किम् पठति? | 5 | 5 | yes | 0.85 | pass |
| Do you listen to music? | 1 | controlled | Do you listen to music? | us.anthropic.claude-haiku-4-5-20251001-v1:0 | अहं संगीतं श्रोतुं न शक्नोमि। किन्तु भवान् किं संगीतं प्रियं करोति? | 4 | 5 | yes | 1.292 | pass |
| Do you play sports? | 1 | controlled | Do you play sports? | us.anthropic.claude-haiku-4-5-20251001-v1:0 | खेलानि मे रोचन्ते। किन्तु अहं यन्त्रम् अस्मि, तस्मात् खेलितुं न शक्नोमि। | 2 | 3 | yes | 1.279 | pass; hard_fail |
| What will you do today? | 1 | controlled | What will you do today? | us.anthropic.claude-haiku-4-5-20251001-v1:0 | अहं त्वया सह संस्कृतं शिक्षयिष्यामि। भवान् किं शिक्षितुम् इच्छति? | 5 | 5 | yes | 1.192 | pass |
| Will you be my friend? | 1 | controlled | Will you be my friend? | us.anthropic.claude-haiku-4-5-20251001-v1:0 | मित्र, अहं तव मित्रम् अस्मि! आवां संस्कृतं शिक्षामः। | 5 | 5 | yes | 2.424 | pass |
