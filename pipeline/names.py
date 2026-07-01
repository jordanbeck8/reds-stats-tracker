"""Player-name normalization for fuzzy merges across data sources."""

import unicodedata


def normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy merge across data source naming differences."""
    name = str(name)
    name = "".join(
        c for c in unicodedata.normalize("NFD", name) if unicodedata.category(c) != "Mn"
    )
    for suffix in (" Jr.", " Sr.", " II", " III", " IV"):
        name = name.replace(suffix, "")
    return name.strip().lower()
