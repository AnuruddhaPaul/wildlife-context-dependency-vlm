"""Numbers reported in the paper / JK.pdf (used ONLY for comparison, never for tuning)."""
MODELS = ["ViT-B/32", "ViT-L/14", "BioCLIP"]

# Table 1: Acc_full, Acc_ultra_tight, CDS, CR, PS, BG   (percent)
TARGETS = {
    "ViT-B/32": dict(acc_full=33.2, acc_ultra_tight=34.5, CDS=-4.0, CR=86.0, PS=45.7, BG=12.9),
    "ViT-L/14": dict(acc_full=62.5, acc_ultra_tight=54.1, CDS=13.4, CR=70.0, PS=25.2, BG=16.5),
    "BioCLIP": dict(acc_full=40.4, acc_ultra_tight=25.4, CDS=37.0, CR=46.5, PS=37.4, BG=11.8),
}
# Table 2 / 4 / 5 (percent)
EXP2 = {"ViT-B/32": (18.8, 16.2), "ViT-L/14": (40.2, 28.1), "BioCLIP": (24.8, 11.5)}  # baseline, mismatch
HABITAT_ACC = {"ViT-B/32": dict(arctic=17.4, desert=24.3, aquatic=12.0, urban=7.3),
               "ViT-L/14": dict(arctic=25.3, desert=34.8, aquatic=27.4, urban=27.0),
               "BioCLIP": dict(arctic=13.0, desert=16.4, aquatic=6.7, urban=11.3)}
PROMPT_ACC = {"ViT-B/32": dict(neutral=18.8, habitat=20.7, wildlife=28.1, scientific=18.3),
              "ViT-L/14": dict(neutral=40.2, habitat=44.3, wildlife=46.9, scientific=36.3),
              "BioCLIP": dict(neutral=24.8, habitat=18.4, wildlife=17.4, scientific=18.3)}
# Other statements in the text
EXTRA = {"ViT-B/32 medium-crop acc": ("ViT-B/32", "medium", 38.7),
         "BioCLIP ultra_tight mean conf": ("BioCLIP", "ultra_tight_conf", 66.7)}
CONF_ACC_GAP_UT = {"ViT-B/32": 7.9, "ViT-L/14": 1.7, "BioCLIP": 41.3}

# Known points of the Exp-1 curves in the paper (accuracy %, mean confidence %); only these are printed in the text
PAPER_LEVEL_ACC = {"ViT-B/32": {"full": 33.2, "medium": 38.7, "ultra_tight": 34.5},
                   "ViT-L/14": {"full": 62.5, "ultra_tight": 54.1}, "BioCLIP": {"full": 40.4, "ultra_tight": 25.4}}
PAPER_LEVEL_CONF = {"ViT-B/32": {"ultra_tight": 34.5 + 7.9}, "ViT-L/14": {"ultra_tight": 54.1 + 1.7}, "BioCLIP": {"ultra_tight": 66.7}}
# JK Table 1 / paper section 3.1: species counts of the evaluation set
PAPER_COUNTS = {"opossum": 100, "raccoon": 100, "coyote": 100, "deer": 100, "rabbit": 100, "bobcat": 100, "cat": 100, "bird": 100,
                "dog": 100, "fox": 100, "lizard": 100, "mountain_lion": 44, "badger": 28, "rodent": 100, "skunk": 100,
                "squirrel": 100, "car": 100}
# JK Table 10: ViT-L/14 per-species accuracy (%) by prompt: neutral, habitat, wildlife, scientific
TABLE10_L14 = {
    "mountain_lion": (90.9, 93.2, 95.5, 95.5), "deer": (75.0, 75.0, 72.0, 69.0), "skunk": (64.0, 81.0, 68.0, 63.0),
    "coyote": (65.0, 53.0, 54.0, 49.0), "bobcat": (61.0, 40.0, 39.0, 39.0), "fox": (35.0, 55.0, 40.0, 53.0),
    "opossum": (51.0, 41.0, 73.0, 25.0), "dog": (25.0, 49.0, 55.0, 23.0), "raccoon": (47.0, 43.0, 25.0, 28.0),
    "bird": (44.0, 42.0, 42.0, 28.0), "cat": (2.0, 24.0, 44.0, 4.0), "rabbit": (33.0, 33.0, 37.0, 27.0),
    "car": (28.0, 59.0, 90.0, 43.0), "badger": (25.0, 3.6, 10.7, 14.3), "rodent": (11.0, 15.0, 10.0, 28.0),
    "squirrel": (23.0, 23.0, 21.0, 23.0), "lizard": (21.0, 21.0, 22.0, 23.0)}

# Table 6: per-species CDS (%): ViT-B/32, ViT-L/14, BioCLIP
SPECIES_CDS = {
    "opossum": (62.5, 44.4, 100.0), "rodent": (-25.0, 46.2, 100.0), "badger": (52.0, 69.2, 68.8),
    "raccoon": (-40.0, 32.3, 68.6), "cat": (0.0, -142.9, 50.0), "squirrel": (-4.1, 13.1, 50.0),
    "skunk": (19.0, 25.0, 41.1), "coyote": (15.8, 21.0, 36.8), "lizard": (-22.2, 6.5, 33.3),
    "fox": (-50.0, 26.2, 27.8), "rabbit": (-33.3, 13.2, 27.1), "deer": (-25.9, 2.3, 26.3),
    "bobcat": (10.0, 13.7, 24.1), "bird": (-16.0, -4.1, 17.9), "mountain_lion": (15.6, 16.2, 0.0),
    "dog": (-100.0, -35.6, -12.8), "car": (0.0, 0.0, 0.0),
}
