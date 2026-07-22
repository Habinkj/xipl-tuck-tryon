# Virtual Try-On Assessment — Submission

**Candidate name:** Habin
**Email:**
**Date:**
**GitHub repo link:**
**Demo video link (max 5 min):**
**Colab notebook links (if used):**

> Pipeline: two photos in (a person + a garment) → a "wearing it" image out, with
> automatic quality scores and a simple web demo around it. See PLAN.md for build order.

---

## Q1 — Garment & Body Understanding
- VLM chosen and why: MiniCPM-V-2.6 (int4) — follows structured JSON instructions
  reliably and fits a free Colab T4 in 4-bit (~7 GB VRAM).
- How to run: `python q1_understanding/q1_attributes.py` (see script header for Colab setup)
- Pairing assumption: person_0X is paired with garment_0X for the 5 pairs; edge-case
  persons are run person-only (garment fields null) to test pose classification.
- Known limitations:

## Q2 — Human Parsing & Segmentation
- Models used (parsing / background removal):
- How to run:
- Edge cases handled / failed:

## Q3 — End-to-End Try-On
- Try-on model chosen and why:
- Hardware used (GPU, VRAM):
- Constraints hit and workarounds:
- How to run:

## Q4 — Automated Quality Evaluation
- Metrics implemented:
- VLM-as-judge rubric prompt (paste it here):
- Results: see evaluation_template_q4.csv (committed)

## Q5 — Web Demo
- Framework (Gradio/Streamlit):
- How to launch:
- Guardrails implemented:

## Honest failure log
- (List anything that did not work and what you tried. Documented failures earn partial credit.)

## Model licenses
- MiniCPM-V-2.6: Apache 2.0
- (add each model + license as you use them)
