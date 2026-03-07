"""
matcher.py
Handles word list loading and response matching.

Rules:
- Matching is case-insensitive
- Singular and plural forms are treated as equivalent
- Once a word is matched, both singular and plural are removed from available words
- Words not on any list are recorded as "novel"
"""

import re


# ─── Simple English Pluralization Rules ───────────────────────────────────────

def get_variants(word: str) -> set[str]:
    """
    Return a set of likely singular/plural variants for a word.
    Covers the most common English patterns; edge cases (e.g. mouse/mice) are not handled.
    """
    word = word.lower().strip()
    variants = {word}

    # If already plural, try to get singular
    if word.endswith("ies"):
        variants.add(word[:-3] + "y")       # babies → baby
    elif word.endswith("ves"):
        variants.add(word[:-3] + "f")       # wolves → wolf
        variants.add(word[:-3] + "fe")      # knives → knife
    elif word.endswith("ses") or word.endswith("xes") or word.endswith("zes"):
        variants.add(word[:-2])             # buses → bus
    elif word.endswith("s") and not word.endswith("ss"):
        variants.add(word[:-1])             # cats → cat

    # If singular, add plural
    base = min(variants, key=len)           # pick shortest as base
    if base.endswith("y") and not base[-2] in "aeiou":
        variants.add(base[:-1] + "ies")    # baby → babies
    elif base.endswith(("s", "x", "z", "ch", "sh")):
        variants.add(base + "es")           # bus → buses
    elif base.endswith("f"):
        variants.add(base[:-1] + "ves")    # wolf → wolves
    elif base.endswith("fe"):
        variants.add(base[:-2] + "ves")    # knife → knives
    else:
        variants.add(base + "s")            # cat → cats

    return variants


# ─── List Manager ─────────────────────────────────────────────────────────────

class ListManager:
    """
    Loads word lists from txt files and tracks which words are still available.
    """

    def __init__(self, list_paths: dict[int, str]):
        """
        Args:
            list_paths: {list_number: filepath}, e.g. {1: "lists/list1_animals.txt"}
        """
        # All words on each list (never changes after loading)
        self.full_lists: dict[int, set[str]] = {}

        # Words still available to be said (shrinks as participant responds)
        self.available: dict[int, set[str]] = {}

        for list_num, path in list_paths.items():
            words = self._load(path)
            self.full_lists[list_num] = words
            self.available[list_num] = set(words)  # copy

    def _load(self, path: str) -> set[str]:
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}

    def match(self, response: str) -> dict:
        """
        Try to match a response against all available words across all lists.

        Returns a result dict:
        {
            "raw":        original response string,
            "word":       matched canonical word (or None),
            "list":       list number matched (or None),
            "novel":      True if not on any list,
            "repeat":     True if word was already used,
        }
        """
        raw = response.strip().lower()
        # Strip punctuation Whisper might add
        word = re.sub(r"[^a-z\-]", "", raw)

        if not word:
            return {"raw": raw, "word": None, "list": None, "novel": False, "repeat": False}

        variants = get_variants(word)

        # Check available words first (not yet said)
        for list_num, available_words in self.available.items():
            match = variants & available_words
            if match:
                canonical = min(match, key=len)  # use shortest form as canonical
                self._consume(list_num, variants)
                return {"raw": raw, "word": canonical, "list": list_num, "novel": False, "repeat": False}

        # Check if it was on a list but already used
        for list_num, full_words in self.full_lists.items():
            if variants & full_words:
                canonical = min(variants & full_words, key=len)
                return {"raw": raw, "word": canonical, "list": list_num, "novel": False, "repeat": True}

        # Not on any list → novel
        return {"raw": raw, "word": word, "list": None, "novel": True, "repeat": False}

    def _consume(self, list_num: int, variants: set[str]):
        """Remove all variants of a matched word from available words."""
        self.available[list_num] -= variants

    def is_exhausted(self, list_num: int) -> bool:
        """True if all words on a list have been said."""
        return len(self.available.get(list_num, set())) == 0

    def remaining_count(self, list_num: int) -> int:
        return len(self.available.get(list_num, set()))


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    base = os.path.dirname(os.path.dirname(__file__))
    manager = ListManager({
        1: os.path.join(base, "lists/list1_animals.txt"),
        2: os.path.join(base, "lists/list2_professions.txt"),
    })

    tests = ["cat", "cats", "CAT", "dogs", "doctor", "Doctors", "kitten", "cat"]
    print(f"{'Response':<12} {'Word':<12} {'List':<6} {'Novel':<8} {'Repeat'}")
    print("-" * 50)
    for t in tests:
        r = manager.match(t)
        print(f"{t:<12} {str(r['word']):<12} {str(r['list']):<6} {str(r['novel']):<8} {r['repeat']}")

    print(f"\nAnimals remaining: {manager.remaining_count(1)}")
    print(f"List 1 exhausted: {manager.is_exhausted(1)}")
