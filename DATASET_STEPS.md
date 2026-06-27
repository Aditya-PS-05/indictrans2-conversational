# Building your own large English↔Hindi parallel dataset

Three ways to get parallel sentences. You'll combine them:

1. **Collect naturally-parallel text** — real human translations (subtitles, gov
   docs, Wikipedia, books). Highest quality, most effort to find/align.
2. **Mine** parallel sentences from comparable corpora — match EN/HI sentences by
   embedding similarity. Medium quality, scalable.
3. **Generate synthetic pairs** — machine-translate monolingual text. The realistic
   way to get LARGE solo, fast. Quality = your MT model's quality.

`build_en_hi_dataset.py` implements **method 3** (the scalable core). Mix in 1 & 2
for quality once you have the basics working.

---

## Step 1 — Decide scope
- Direction: EN→HI (script does this; reverse for back-translation).
- Target size: start 5k smoke test → scale to 1–5M.
- Domain: general (Wikipedia) vs specific (news/medical/chat) — changes the source.

## Step 2 — Sources
- **Monolingual EN** (synthetic): `wikimedia/wikipedia`, OSCAR, CC100-en, news crawls.
- **Monolingual HI**: AI4Bharat IndicCorp, OSCAR-hi, Hindi Wikipedia.
- **Naturally-parallel** (quality): OPUS, OpenSubtitles en-hi, Wikipedia
  interlanguage links, government bilingual PDFs.

## Step 3 — Clean + segment
- Strip HTML/URLs/boilerplate. Split into sentences.
- English → `nltk` / `spacy`. **Hindi → `indic-nlp-library`** (English splitters
  break on Devanagari). Drop sentences <4 or >40 words.

## Step 4 — Build pairs
- **Synthetic** (this script): NLLB-200 or IndicTrans2 (AI4Bharat, best for Indic).
  Batch ~64 sentences on the A10G.
- **Mining** (optional, higher quality): embed both sides with **LaBSE**, keep pairs
  with cosine > 0.75.

## Step 5 — Filter pairs (this decides quality — most important)
- **Language ID** with fastText `lid.176.bin`: confirm EN side is English, HI is Hindi.
- **Length-ratio**: drop pairs outside ~0.5–2.0 (catches misalignment).
- **Dedup**: exact + near-dup (MinHash).
- Drop empties, `en == hi` copies, junk.
- Optional: keep only pairs above a LaBSE similarity threshold.

## Step 6 — Spot-check
Read 100 random pairs. If many are wrong, raise thresholds. Eyeballing catches what
metrics miss.

## Step 7 — Split + save
- `train_test_split`, `save_to_disk`, JSONL as `{"translation": {"en","hi"}}`.
- Optional `push_to_hub`.

---

## Run it
```bash
# on the A10G box
pip install "datasets>=2.18" transformers torch sentencepiece nltk fasttext-wheel tqdm
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin   # quality filter

python build_en_hi_dataset.py --target 5000   --out data_smoke    # sanity check first
python build_en_hi_dataset.py --target 2000000 --out data_full    # real run
```

## Reality check
A good corpus is mostly **cleaning/filtering**, not collecting — expect to throw away
30–70% of raw pairs. Synthetic data inherits the MT model's errors: fine for
bootstrapping/learning, but mix in real parallel sources (OPUS, subtitles) for quality.

## Upgrades when ready
- Swap NLLB → **IndicTrans2** (`ai4bharat/indictrans2-en-indic-1B`) for better Hindi.
- Add **back-translation**: translate Hindi monolingual → English, append those pairs.
- Add **LaBSE filtering** in `flush()` to drop low-similarity synthetic pairs.
- Use Hindi sentence segmentation (`indic-nlp-library`) when you mine real HI text.
