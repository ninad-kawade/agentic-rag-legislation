# Phase 5 Evaluation Report

Run date: 2026-11-15  
Results: `eval/results.jsonl` (50 saved API answers)  
Detailed RAGAS scores: `eval/scores_detailed.jsonl` (40 answerable questions)  
RAGAS version: 0.2.3

RAGAS faithfulness, context precision, and context recall use the saved API answers, expected answers, and cited snippets. Null metric cells are retained and excluded from means; each table reports available judgments in Diagnostics.

## Overall

| Measure | Correct / n | Score |
|---|---:|---:|
| Abstention accuracy (out-of-corpus) | 10 / 10 | 100.0% |
| Answerable non-abstention rate | 40 / 40 | 100.0% |
| Citation accuracy (exact expected chunk ID cited) | 40 / 40 | 100.0% |
| Faithfulness (answerable only) | — | 0.912 |
| Context precision (answerable only) | — | 0.975 |
| Context recall (answerable only) | — | 0.950 |

## Category breakdown

| Category | n | Correct abstention behavior | Citation accuracy | Faithfulness | Context precision | Context recall | Average retries |
|---|---:|---:|---:|---:|---:|---:|---:|
| Easy | 30 | 100.0% (30/30) | 100.0% | 1.000 | 1.000 | 1.000 | 0.133 |
| Hard | 10 | 100.0% (10/10) | 100.0% | 0.700 | 0.900 | 0.800 | 0.200 |
| Abstention | 10 | 100.0% (10/10) | N/A | N/A | N/A | N/A | 2.000 |

Average retries across all 50 questions: **0.520**.

## Diagnostics

### RAGAS job success/failure breakdown

A finite numeric score is a successful job; null is an omitted/failed judgment. The aggregate means above are calculated from this same detailed file.

| Category | Metric | Succeeded | Omitted | Total |
|---|---|---:|---:|---:|
| Easy | Faithfulness | 12 | 18 | 30 |
| Easy | Context Precision | 15 | 15 | 30 |
| Easy | Context Recall | 15 | 15 | 30 |
| Hard | Faithfulness | 5 | 5 | 10 |
| Hard | Context Precision | 5 | 5 | 10 |
| Hard | Context Recall | 5 | 5 | 10 |

### Hard questions: retries and faithfulness

| ID | Question | Retries | Faithfulness |
|---:|---|---:|---:|
| 31 | Which Implementing Regulation is amended by Commission Implementing Regulation (EU) 2020/2041? | 0 | N/A |
| 32 | Which Implementing Regulation is amended by Commission Implementing Regulation (EU) 2020/2042? | 0 | 1.000 |
| 33 | In Council Decision (EU) 2020/2062, which republic is the other party to the Stabilisation and Association Agreement? | 0 | N/A |
| 34 | In Council Decision (EU) 2020/2072, which republic is the other party to the Stabilisation and Association Agreement? | 0 | 1.000 |
| 35 | According to the existing toy-safety rule described in Commission Directive (EU) 2020/2088, how many allergenic fragrances must be listed when added above the stated concentration? | 0 | N/A |
| 36 | How many allergenic fragrances does Commission Directive (EU) 2020/2089 describe as generally prohibited in toys? | 1 | 0.500 |
| 37 | Which part of Regulation (EC) No 1008/2008 is Commission Delegated Regulation (EU) 2020/2114 concerned with? | 0 | N/A |
| 38 | Which part of Regulation (EC) No 1008/2008 is Commission Delegated Regulation (EU) 2020/2115 concerned with? | 0 | 0.000 |
| 39 | Which prior implementing regulation’s anti-dumping measures are investigated for possible circumvention by Commission Implementing Regulation (EU) 2020/2161? | 0 | N/A |
| 40 | Which two prior implementing regulations’ anti-dumping measures are investigated for possible circumvention by Commission Implementing Regulation (EU) 2020/2162? | 1 | 1.000 |

### Abstention retry check

Retries by question ID: 41=2, 42=2, 43=2, 44=2, 45=2, 46=2, 47=2, 48=2, 49=2, 50=2.
All 10 abstention questions used exactly 2 retries: **yes**.

### Metric notes

- A failed metric remains as `null` in `scores_detailed.jsonl`; it is not dropped or imputed.
- `scores_progress.json` records completed and omitted jobs so an interrupted run can resume at the next unfinished question/metric pair.
- RAGAS uses the returned citation snippets as retrieved contexts. Missing citation contexts remain empty.
