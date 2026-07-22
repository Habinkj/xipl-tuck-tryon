"""
Q1 — Garment & Body Understanding with a Vision-Language Model
--------------------------------------------------------------
Takes a person photo + a garment photo and writes structured JSON describing
the garment (type, sleeves, neckline, color, pattern) and the person (pose,
upper/lower body visibility). Output matches assets/sample_output_q1.json.

HOW TO RUN IN COLAB (first time — do these in a cell ABOVE this script):
  1) Runtime > Change runtime type > T4 GPU > Save
  2) Install deps:
       !pip install -q transformers==4.44.2 accelerate pillow torch
       !pip install -q "bitsandbytes>=0.43.0"    # for the int4 model on T4
  3) Put the repo on Colab (either clone your GitHub repo, or upload the
     'assets' folder). Then run:
       !python q1_understanding/q1_attributes.py
  Outputs are written to q1_understanding/outputs/.

MODEL: openbmb/MiniCPM-V-2_6-int4  (Apache 2.0). The int4 build fits a T4.
If it OOMs or fails to load, see the FALLBACK note at the bottom.
"""

import os
import json
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# ----- paths -------------------------------------------------------------
ASSETS = "assets"
PERSON_DIR = os.path.join(ASSETS, "test_pairs", "person")
GARMENT_DIR = os.path.join(ASSETS, "test_pairs", "garment")
EDGE_DIR = os.path.join(ASSETS, "edge_cases")
OUT_DIR = os.path.join("q1_understanding", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ID = "openbmb/MiniCPM-V-2_6-int4"

# ----- the exact schema we must produce (mirrors sample_output_q1.json) ---
GARMENT_KEYS = ["type", "sleeve_length", "neckline", "primary_color", "pattern"]
PERSON_KEYS = ["pose_category", "upper_body_visible", "lower_body_visible"]

# What we ask the model. We DEMAND raw JSON only, no prose, no code fences.
GARMENT_PROMPT = (
    "You are labeling a flat product photo of a single clothing garment. "
    "Return ONLY a JSON object, no extra text, with exactly these keys: "
    '"type" (e.g. t-shirt, shirt, tank-top, dress, hoodie), '
    '"sleeve_length" (one of: sleeveless, short, three-quarter, long), '
    '"neckline" (e.g. crew, v-neck, scoop, collared, strappy), '
    '"primary_color" (single word), '
    '"pattern" (e.g. solid, striped, graphic print, floral, checked).'
)
PERSON_PROMPT = (
    "You are labeling a photo of a person for a virtual try-on system. "
    "Return ONLY a JSON object, no extra text, with exactly these keys: "
    '"pose_category" (EXACTLY one of: "front-facing", "side", "seated"), '
    '"upper_body_visible" (true or false), '
    '"lower_body_visible" (true or false). '
    "Rule: if the person is sitting, pose_category is \"seated\"; "
    "if the body/face is turned to the side (profile), it is \"side\"; "
    "otherwise \"front-facing\"."
)


def load_model():
    print(f"Loading {MODEL_ID} ... (first run downloads weights, be patient)")
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        attn_implementation="sdpa", torch_dtype=torch.float16,
    )
    model = model.eval().cuda()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, tok


def ask(model, tok, images, question):
    """Send 1-2 images + a question, return the raw text answer."""
    content = list(images) + [question]
    msgs = [{"role": "user", "content": content}]
    answer = model.chat(image=None, msgs=msgs, tokenizer=tok,
                        sampling=False, temperature=0.0)
    return answer


def parse_json(text, required_keys):
    """Strip code fences / stray text, parse JSON, fill any missing key with None."""
    t = text.strip()
    if "```" in t:                       # remove ```json ... ``` fences
        t = t.split("```")[1] if t.count("```") >= 2 else t.replace("```", "")
        t = t.replace("json", "", 1).strip()
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end != -1:
        t = t[start:end + 1]
    try:
        obj = json.loads(t)
    except Exception:
        obj = {}
    return {k: obj.get(k) for k in required_keys}, text


def describe_garment(model, tok, path):
    img = Image.open(path).convert("RGB")
    raw = ask(model, tok, [img], GARMENT_PROMPT)
    attrs, _ = parse_json(raw, GARMENT_KEYS)
    return attrs, raw


def describe_person(model, tok, path):
    img = Image.open(path).convert("RGB")
    raw = ask(model, tok, [img], PERSON_PROMPT)
    attrs, _ = parse_json(raw, PERSON_KEYS)
    return attrs, raw


def build_record(person_file, garment_file, model, tok):
    rec = {"person_image": person_file, "garment_image": garment_file}
    notes = []
    if garment_file:
        g, graw = describe_garment(model, tok, os.path.join(GARMENT_DIR, garment_file))
        rec["garment_attributes"] = g
        notes.append("garment_raw=" + graw.replace("\n", " ")[:160])
    else:
        rec["garment_attributes"] = {k: None for k in GARMENT_KEYS}
    ppath = os.path.join(EDGE_DIR, person_file) if person_file.endswith(".jpg") \
        else os.path.join(PERSON_DIR, person_file)
    p, praw = describe_person(model, tok, ppath)
    rec["person_attributes"] = p
    notes.append("person_raw=" + praw.replace("\n", " ")[:160])
    rec["model_used"] = MODEL_ID
    rec["confidence_notes"] = " | ".join(notes)
    return rec


# ----- the images we must cover ------------------------------------------
# 5 official pairs (person_0X + garment_0X) — this consumes all 10 test images
PAIRS = [(f"person_0{i}.png", f"garment_0{i}.jpg") for i in range(1, 6)]
# 3 edge-case persons (person-only) — pose classification is graded here
EDGE_PERSONS = ["person_side_pose.jpg", "person_seated.jpg", "person_crossed_arms.jpg"]


def main():
    model, tok = load_model()

    for person_file, garment_file in PAIRS:
        print(f"\n--- {person_file} + {garment_file} ---")
        rec = build_record(person_file, garment_file, model, tok)
        out = os.path.join(OUT_DIR, person_file.replace(".png", "") + ".json")
        with open(out, "w") as f:
            json.dump(rec, f, indent=2)
        print("saved", out, "->", rec["person_attributes"], rec["garment_attributes"])

    for person_file in EDGE_PERSONS:
        print(f"\n--- edge: {person_file} ---")
        rec = build_record(person_file, None, model, tok)
        out = os.path.join(OUT_DIR, person_file.replace(".jpg", "") + ".json")
        with open(out, "w") as f:
            json.dump(rec, f, indent=2)
        print("saved", out, "-> pose:", rec["person_attributes"].get("pose_category"))

    print("\nDONE. Check q1_understanding/outputs/. Verify side->'side', seated->'seated'.")


if __name__ == "__main__":
    main()

# FALLBACK if MiniCPM int4 won't load on your T4:
#   - Try the full model "openbmb/MiniCPM-V-2_6" with load in 4-bit via bitsandbytes,
#     or switch to "microsoft/Florence-2-large" (much lighter, MIT) — but Florence-2
#     needs task-prompt style calls, so keep MiniCPM if you can. Ask me and I'll adapt.
