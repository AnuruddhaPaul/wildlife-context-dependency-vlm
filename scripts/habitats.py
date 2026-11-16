"""Experiment 2 species -> incompatible-habitat mapping (RECONSTRUCTED, no earlier mapping was recovered).

Constraints from the task: badger + mountain_lion get arctic+desert; the other 15 species get two distinct
habitats such that arctic=11, desert=6, aquatic=10, urban=3 species; no desert for lizard, no aquatic for deer.
With the per-box sample this yields arctic 1,172 / desert 672 / aquatic 1,000 / urban 300 = 3,144 predictions.
"""
HABITAT_COLORS = {"arctic": (240, 248, 255), "desert": (237, 201, 175),
                  "aquatic": (70, 130, 180), "urban": (169, 169, 169)}
HABITATS = list(HABITAT_COLORS)

HABITAT_MAP = {
    # species: (habitat 1, habitat 2, rationale)
    "badger": ("arctic", "desert", "required by task (burrowing woodland/grassland mustelid)"),
    "mountain_lion": ("arctic", "desert", "required by task"),
    "opossum": ("arctic", "desert", "warm-temperate woodland species, poorly adapted to cold or arid terrain"),
    "raccoon": ("arctic", "desert", "riparian/woodland species; snow and dry sand are both foreign"),
    "squirrel": ("arctic", "desert", "forest/tree-dependent species"),
    "bobcat": ("arctic", "urban", "solitary forest/scrub carnivore; avoids dense built-up areas"),
    "deer": ("arctic", "urban", "task: no aquatic for deer; woodland browser"),
    "lizard": ("aquatic", "urban", "task: no desert for lizard; ectotherm, not aquatic, not built-up"),
    "bird": ("arctic", "aquatic", "terrestrial passerine/quail-type birds, not snow or open-water species"),
    "coyote": ("arctic", "aquatic", "desert/scrub-adapted canid"),
    "fox": ("arctic", "aquatic", "scrub/woodland canid"),
    "rabbit": ("arctic", "aquatic", "scrub/grassland lagomorph"),
    "rodent": ("arctic", "aquatic", "terrestrial small rodents"),
    "skunk": ("arctic", "aquatic", "terrestrial scrub/woodland omnivore"),
    "car": ("desert", "aquatic", "non-animal control: no natural habitat, placed in sand / on water"),
    "cat": ("desert", "aquatic", "domestic cat, placed on sand / on water"),
    "dog": ("desert", "aquatic", "domestic dog, placed on sand / on water"),
}

EXPECTED_HABITAT_TOTALS = {"arctic": 1172, "desert": 672, "aquatic": 1000, "urban": 300}


def habitats_for(species: str) -> tuple[str, str]:
    return HABITAT_MAP[species][:2]


def check_totals(species_counts: dict[str, int]) -> dict[str, int]:
    totals = {h: 0 for h in HABITATS}
    for sp, n in species_counts.items():
        h1, h2 = habitats_for(sp)
        assert h1 != h2
        totals[h1] += n
        totals[h2] += n
    return totals
