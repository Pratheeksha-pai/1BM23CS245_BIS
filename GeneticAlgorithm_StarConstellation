# Star Constellation using genetic algorithm

import math
import random
from typing import List, Tuple

Point = Tuple[float, float]
Chromosome = List[int]

# ---------- Geometry helpers ----------
def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _orientation(p: Point, q: Point, r: Point) -> int:
    """Returns orientation of ordered triplet (p, q, r):
       0 -> colinear, 1 -> clockwise, 2 -> counterclockwise
    """
    val = (q[1]-p[1])*(r[0]-q[0]) - (q[0]-p[0])*(r[1]-q[1])
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2

def _on_segment(p: Point, q: Point, r: Point) -> bool:
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

def segments_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    """Proper/general intersection with colinear handling (excluding shared endpoints)."""
    # Exclude if sharing any endpoint exactly
    if p1 == p2 or p1 == q2 or q1 == p2 or q1 == q2:
        return False
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    # Colinear special cases
    if o1 == 0 and _on_segment(p1, p2, q1): return True
    if o2 == 0 and _on_segment(p1, q2, q1): return True
    if o3 == 0 and _on_segment(p2, p1, q2): return True
    if o4 == 0 and _on_segment(p2, q1, q2): return True
    return False

# ---------- Fitness (length + crossings) ----------
def path_length(order: Chromosome, stars: List[Point], cycle: bool) -> float:
    total = 0.0
    for i in range(len(order)-1):
        total += dist(stars[order[i]], stars[order[i+1]])
    if cycle and len(order) > 1:
        total += dist(stars[order[-1]], stars[order[0]])
    return total

def count_crossings(order: Chromosome, stars: List[Point], cycle: bool) -> int:
    # Build the list of edges in this order
    edges = []
    for i in range(len(order)-1):
        edges.append((order[i], order[i+1]))
    if cycle and len(order) > 1:
        edges.append((order[-1], order[0]))

    crossings = 0
    for i in range(len(edges)):
        a1, a2 = edges[i]
        p1, q1 = stars[a1], stars[a2]
        for j in range(i+1, len(edges)):
            b1, b2 = edges[j]
            # Skip adjacent/overlapping edges (they share a vertex)
            if len({a1, a2, b1, b2}) < 4:
                continue
            p2, q2 = stars[b1], stars[b2]
            if segments_intersect(p1, q1, p2, q2):
                crossings += 1
    return crossings

def fitness(order: Chromosome, stars: List[Point], crossing_weight: float, cycle: bool) -> float:
    """Lower is better."""
    L = path_length(order, stars, cycle)
    X = count_crossings(order, stars, cycle)
    return L + crossing_weight * X

# ---------- GA operators ----------
def tournament_selection(pop: List[Chromosome], stars: List[Point], k: int, crossing_weight: float, cycle: bool) -> Chromosome:
    """Pick best of k random individuals (min fitness)."""
    contenders = random.sample(pop, k)
    contenders.sort(key=lambda c: fitness(c, stars, crossing_weight, cycle))
    return contenders[0][:]

def order_crossover(parent1: Chromosome, parent2: Chromosome) -> Chromosome:
    """OX (Order Crossover) for permutations."""
    n = len(parent1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    # Copy slice from parent1
    child[a:b+1] = parent1[a:b+1]
    # Fill remaining from parent2 preserving order
    p2_idx = 0
    for i in range(n):
        if child[i] == -1:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
            p2_idx += 1
    return child

def swap_mutation(ch: Chromosome, mutation_rate: float) -> None:
    n = len(ch)
    if random.random() < mutation_rate:
        i, j = random.sample(range(n), 2)
        ch[i], ch[j] = ch[j], ch[i]

# ---------- GA driver ----------
def run_ga(
    stars: List[Point],
    population_size: int = 200,
    generations: int = 500,
    tournament_k: int = 4,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.2,
    elitism: int = 4,
    crossing_weight: float = 50.0,
    cycle: bool = True,
    seed: int = 42
):
    """Returns best_order, best_fitness, history (list of best fitness by generation)."""
    random.seed(seed)
    n = len(stars)
    if n < 2:
        raise ValueError("Need at least 2 stars.")

    # Initialize population with random permutations
    base = list(range(n))
    population = [random.sample(base, n) for _ in range(population_size)]

    # Evaluate initial best
    def fit(ch): return fitness(ch, stars, crossing_weight, cycle)
    population.sort(key=fit)
    best = population[0][:]
    best_f = fit(best)
    history = [best_f]

    for gen in range(generations):
        new_pop: List[Chromosome] = []
        # Elitism
        elites = population[:elitism]
        new_pop.extend([e[:] for e in elites])

        # Fill rest with offspring
        while len(new_pop) < population_size:
            p1 = tournament_selection(population, stars, tournament_k, crossing_weight, cycle)
            p2 = tournament_selection(population, stars, tournament_k, crossing_weight, cycle)
            if random.random() < crossover_rate:
                child = order_crossover(p1, p2)
            else:
                child = p1[:]
            swap_mutation(child, mutation_rate)
            new_pop.append(child)

        population = new_pop
        population.sort(key=fit)

        # Track best
        if fit(population[0]) < best_f:
            best = population[0][:]
            best_f = fit(best)

        history.append(best_f)

    return best, best_f, history

# ---------- Example usage ----------
if __name__ == "__main__":
    # Example: a synthetic "sky" of 20 stars (you can replace with real data)
    stars = [
        (2.1, 9.7), (1.4, 8.3), (3.0, 7.8), (5.2, 9.1), (7.4, 9.5),
        (9.1, 8.6), (8.5, 6.9), (9.7, 5.2), (7.8, 4.1), (5.6, 3.5),
        (3.3, 3.9), (1.5, 4.8), (0.9, 6.2), (2.6, 5.8), (4.2, 6.8),
        (6.1, 7.4), (6.7, 5.6), (4.9, 5.1), (3.8, 4.9), (2.0, 3.2),
    ]

    # Tune crossing_weight:
    #   higher -> prioritizes fewer crossings (good for "constellation-like" lines)
    #   lower  -> prioritizes shorter total length
    best_order, best_score, history = run_ga(
        stars,
        population_size=80,   # instead of 200
        generations=150,      # instead of 600
        crossing_weight=50.0,
        cycle=False,
        seed=123
    )

    # Report
    total_len = path_length(best_order, stars, cycle=False)
    crossings = count_crossings(best_order, stars, cycle=False)
    print("Best order (0-based indices):", best_order)
    print(f"Total length: {total_len:.3f}")
    print(f"Crossings: {crossings}")
    print(f"Objective score: {best_score:.3f}")
    print("Best order as coordinates:")
    for idx in best_order:
        print(stars[idx])
