# turnstile_gen.py — Generate parity-counting sentences

import random
from typing import Optional, Tuple


def generate_turnstile_sample(
    max_turnstiles: int = 10,
    seed: Optional[int] = None,
    bias_equal: float = 0.5,
) -> Tuple[str, int]:
    """
    Generiert einen Sample basierend auf der Parität der !-Anzahl.

    Args:
        max_turnstiles: Maximale Anzahl an !.
        seed:           Optionaler Seed für Reproduzierbarkeit.
        bias_equal:     Wahrscheinlichkeit für ein Label 1 (gerade Anzahl).

    Returns:
        (sentence, label)
        Beispiel: ("!!!! ", 1) oder ("!!! ", 0)
    """
    if seed is not None:
        random.seed(seed)

    # Entscheiden, ob das Ergebnis 1 (gerade) oder 0 (ungerade) sein soll
    if random.random() < bias_equal:
        # Erzeuge eine gerade Zahl (0, 2, 4, ...)
        n = random.randint(0, max_turnstiles // 2) * 2
        label = 1
    else:
        # Erzeuge eine ungerade Zahl (1, 3, 5, ...)
        # Falls max_turnstiles 0 ist, weichen wir auf 1 aus
        limit = max(1, max_turnstiles)
        n = random.randint(0, (limit - 1) // 2) * 2 + 1
        label = 0

    # Falls n durch Zufall über max_turnstiles gerutscht ist, korrigieren
    if n > max_turnstiles:
        n -= 2

    sentence = "!" * n + " "

    return sentence, label


def generate_turnstile_function(
    max_turnstiles: int = 10,
    seed: Optional[int] = None,
    bias_equal: float = 0.5,
    **kwargs,
) -> Tuple[str, int]:
    return generate_turnstile_sample(
        max_turnstiles=max_turnstiles,
        seed=seed,
        bias_equal=bias_equal,
    )


if __name__ == "__main__":
    print("Sample parity sentences (even=1, odd=0):")
    print("-" * 40)
    for i in range(20):
        # Wir nutzen i als seed, damit es unterschiedliche Längen gibt
        sentence, label = generate_turnstile_sample(
            max_turnstiles=10, seed=i
        )
        # Darstellung: "! " -> 0
        print(f"  {sentence.strip()} -> {label}")
