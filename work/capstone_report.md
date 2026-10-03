# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Ayushman Lohani
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/ayushmanlohani/flyrank-ml-internship
- **Date:** 2026

## 0. Abstract

Content teams must decide which pages to refresh first — can trailing signals rank that queue? We use the bundled slice (30,000 pages, 32 clients, base rate 54.2%). We compare a hand-rule baseline against a Random Forest on the same client-grouped holdout (seed 42). The model shows directional skill (grouped AUC 0.61 vs 0.50) but loses at the top (P50 0.56 vs 0.70); random split overstates at 0.77. We ship a human-reviewed queue, not an autopublisher.

## 1. Problem framing

Decision: review order each month. Unit: one pseudonymized page. Output: rank score + reason codes. Action: review top-50, check intent/seasonality, then refresh/monitor/skip. Wrong call costs editor time or continued decay. ML helps: multi-signal ranking beats single thresholds.

## 2. Data safety

Modeled file: `data/raw/content_refresh_anonymized.csv` only. Excluded from features: `trend_direction`, `trend_pct` (label-derived), `content_id`/`client_id` (grouping only). No names, domains, URLs, queries anywhere. Rates are 0–100 scale. Warehouse (~79M rows) is context only; numbers below are slice-measured.

## 3. Baseline

`(days_since_last_update >= 180) * impressions_90d`. Fair: same test rows/metrics. Grouped holdout: AUC 0.500, P50 0.70, P200 0.575.

## 4. Model / analysis

RandomForest(100, depth 10, seed 42). 27 features: 19 numeric (incl. 4 log transforms) + 8 categorical (label-encoded) — see paper §3 for exact list. Target in one sentence: page trend bucket is "down". Fits lane: non-linear staleness cliff + visibility×age interactions + stable `predict_proba` ranking.

## 5. Evaluation

GroupShuffleSplit on `client_id`, 20% clients (25/7), seed 42 — no client in both sides. Metrics on same split: AUC 0.61 vs 0.50; P50 0.56 vs 0.70; P200 0.575 tie. Errors: worst on 1–2 impression fresh pages (single click = 100% swing). Random-split contrast ~0.77 proves 0.16 memorization gap.

## 6. Interpretation

Top signals: `log_impressions_90d` (0.17), `days_with_impressions` (0.138), `avg_position` (0.112), `content_age_days` (0.105); length weak (~0.06). Surprise/negative: baseline beats model at P50 — hard 180d+visible filter nails obvious head; model wins breadth. Well-understood no-effect: word count alone does not separate decline.

## 7. Recommendation

P0 refresh_and_review_ctr (visible+declining+low CTR) → fix meta+intent. P0 refresh (visible+declining) → update stale parts. P1 engagement fix. P2 striking-distance expand. Hold monitor for low-vol/fresh. Monthly top-50 human gate; no autopublish brand/legal. Confidence directional: ~56–58% hits top-50/200, not guarantees.

## 8. Reproducibility

`pip install -r requirements.txt`, then `python scripts/run_all.py` or Run All on `work/notebooks/capstone.ipynb` (bundled CSV only, seed 42). Env: Python 3.11, pandas, sklearn, numpy. Receipt: `work/outputs/playbook_metrics.json` committed. `scripts/` untouched.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — https://flyrank.ai
