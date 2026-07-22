# Build Plan — XIPL Tuck Try-On Assessment

Governing rule: 50% of every question's marks is "does it run from the README."
Reproducibility beats cleverness. Document every failure honestly.

## Phase 0 — Setup
- [ ] Create GitHub repo, push this scaffold
- [ ] Open Colab, confirm T4 GPU, mount Google Drive
- [ ] Source 2 extra pairs (pair_04, pair_05) into assets/test_pairs/
- [ ] Update assets/test_pairs/pairs_manifest.csv rows pair_04 / pair_05

## Phase 1 — Q1: Garment & body understanding (15) — EASY, DO FIRST
- [ ] Run q1_understanding/q1_attributes.py in Colab
- [ ] JSON for 5 pairs + 3 edge persons saved to q1_understanding/outputs/
- [ ] Side pose classifies as "side", seated as "seated"
- [ ] Fill README Q1 section

## Phase 2 — Q3 SPIKE: prove CatVTON runs on ONE pair (de-risk early)
- [ ] One try-on output image exists at reduced res
- [ ] Note VRAM workaround used

## Phase 3 — Q2: Parsing & segmentation (20) — built to feed Q3
- [ ] Garment cutouts (rembg / RMBG-1.4)
- [ ] Human parsing + agnostic image (SCHP or SAM+GroundingDINO)
- [ ] Run on provided persons + crossed-arms edge case

## Phase 4 — Q3 FULL: try-on inference (25)
- [ ] 5 output images (3 given pairs + 2 sourced)
- [ ] Constraints log in README

## Phase 5 — Q4: automated evaluation (20) — YOUR STRENGTH
- [ ] Garment fidelity (OpenCLIP/DINOv2)
- [ ] Identity preservation (InsightFace)
- [ ] VLM-as-judge (reuse Q1 model + rubric)
- [ ] Fill evaluation_template_q4.csv, commit rubric prompt

## Phase 6 — Q5: web demo (20)
- [ ] Gradio app: upload -> result + Q4 scores + Q1 attributes
- [ ] Guardrails: reject no-person, warn side/seated, show ETA
- [ ] Guardrails fire on no_person / seated / side edge images

## Phase 7 — Package & submit
- [ ] Every README section filled + honest failure log
- [ ] Colab links with outputs saved
- [ ] Demo video (<= 5 min) showing guardrails
- [ ] Reply to email: repo + video + Colab links

## Cut line (if time runs out)
Ship: Q1 done well + Q3 on 3 given pairs (even rough) + Q4 on those + Q5 shell
with guardrails + honest README. That beats a broken attempt at all five.
