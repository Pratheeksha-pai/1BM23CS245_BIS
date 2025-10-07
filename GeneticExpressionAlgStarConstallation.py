import math
import random
from typing import List, Tuple

Point = Tuple[float, float]

# -------- Geometry helpers (from your original code) --------
def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _orientation(p: Point, q: Point, r: Point) -> int:
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if abs(val) < 1e-12:
        return 0
    return 1 if val > 0 else 2

def _on_segment(p: Point, q: Point, r: Point) -> bool:
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

def segments_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    if p1 == p2 or p1 == q2 or q1 == p2 or q1 == q2:
        return False
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1): return True
    if o2 == 0 and _on_segment(p1, q2, q1): return True
    if o3 == 0 and _on_segment(p2, p1, q2): return True
    if o4 == 0 and _on_segment(p2, q1, q2): return True
    return False

def path_length(order: List[int], stars: List[Point], cycle: bool) -> float:
    total = 0.0
    for i in range(len(order) - 1):
        total += dist(stars[order[i]], stars[order[i + 1]])
    if cycle and len(order) > 1:
        total += dist(stars[order[-1]], stars[order[0]])
    return total

def count_crossings(order: List[int], stars: List[Point], cycle: bool) -> int:
    edges = []
    for i in range(len(order) - 1):
        edges.append((order[i], order[i + 1]))
    if cycle and len(order) > 1:
        edges.append((order[-1], order[0]))
    crossings = 0
    for i in range(len(edges)):
        a1, a2 = edges[i]
        p1, q1 = stars[a1], stars[a2]
        for j in range(i + 1, len(edges)):
            b1, b2 = edges[j]
            if len({a1, a2, b1, b2}) < 4:
                continue
            p2, q2 = stars[b1], stars[b2]
            if segments_intersect(p1, q1, p2, q2):
                crossings += 1
    return crossings

# -------- Protected division --------
def pdiv(a, b):
    try:
        return a / b if abs(b) > 1e-6 else 1.0
    except:
        return 1.0

# -------- Expression tree nodes --------
class Node:
    def evaluate(self, x, y):
        raise NotImplementedError
    def copy(self):
        raise NotImplementedError
    def size(self):
        raise NotImplementedError
    def get_subtree(self, index):
        raise NotImplementedError
    def replace_subtree(self, index, subtree):
        raise NotImplementedError

class Const(Node):
    def __init__(self, val):
        self.val = val
    def evaluate(self, x, y):
        return self.val
    def copy(self):
        return Const(self.val)
    def size(self):
        return 1
    def get_subtree(self, index):
        if index == 0:
            return self
        raise IndexError
    def replace_subtree(self, index, subtree):
        if index == 0:
            return subtree
        raise IndexError
    def __repr__(self):
        return f"{self.val:.2f}"

class VarX(Node):
    def evaluate(self, x, y):
        return x
    def copy(self):
        return VarX()
    def size(self):
        return 1
    def get_subtree(self, index):
        if index == 0:
            return self
        raise IndexError
    def replace_subtree(self, index, subtree):
        if index == 0:
            return subtree
        raise IndexError
    def __repr__(self):
        return "x"

class VarY(Node):
    def evaluate(self, x, y):
        return y
    def copy(self):
        return VarY()
    def size(self):
        return 1
    def get_subtree(self, index):
        if index == 0:
            return self
        raise IndexError
    def replace_subtree(self, index, subtree):
        if index == 0:
            return subtree
        raise IndexError
    def __repr__(self):
        return "y"

class OpNode(Node):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def size(self):
        return 1 + self.left.size() + self.right.size()
    def get_subtree(self, index):
        if index == 0:
            return self
        index -= 1
        left_size = self.left.size()
        if index < left_size:
            return self.left.get_subtree(index)
        else:
            return self.right.get_subtree(index - left_size)
    def replace_subtree(self, index, subtree):
        if index == 0:
            return subtree
        index -= 1
        left_size = self.left.size()
        if index < left_size:
            self.left = self.left.replace_subtree(index, subtree)
        else:
            self.right = self.right.replace_subtree(index - left_size, subtree)
        return self

class Add(OpNode):
    def evaluate(self, x, y):
        return self.left.evaluate(x, y) + self.right.evaluate(x, y)
    def copy(self):
        return Add(self.left.copy(), self.right.copy())
    def __repr__(self):
        return f"({self.left}+{self.right})"

class Sub(OpNode):
    def evaluate(self, x, y):
        return self.left.evaluate(x, y) - self.right.evaluate(x, y)
    def copy(self):
        return Sub(self.left.copy(), self.right.copy())
    def __repr__(self):
        return f"({self.left}-{self.right})"

class Mul(OpNode):
    def evaluate(self, x, y):
        return self.left.evaluate(x, y) * self.right.evaluate(x, y)
    def copy(self):
        return Mul(self.left.copy(), self.right.copy())
    def __repr__(self):
        return f"({self.left}*{self.right})"

class Div(OpNode):
    def evaluate(self, x, y):
        return pdiv(self.left.evaluate(x, y), self.right.evaluate(x, y))
    def copy(self):
        return Div(self.left.copy(), self.right.copy())
    def __repr__(self):
        return f"({self.left}/{self.right})"

# -------- Generate random expression --------
def random_expr(depth=3):
    if depth <= 0:
        terminal = random.choice([Const(random.uniform(-10, 10)), VarX(), VarY()])
        return terminal
    else:
        op = random.choice([Add, Sub, Mul, Div])
        return op(random_expr(depth - 1), random_expr(depth - 1))

# -------- Convert expression to star order --------
def expr_to_order(expr: Node, stars: List[Point]) -> List[int]:
    scored = [(i, expr.evaluate(x, y)) for i, (x, y) in enumerate(stars)]
    scored.sort(key=lambda x: x[1])
    order = [i for i, _ in scored]
    return order

# -------- Fitness --------
def fitness_expr(expr: Node, stars: List[Point], crossing_weight=50.0, cycle=False) -> float:
    order = expr_to_order(expr, stars)
    L = path_length(order, stars, cycle)
    X = count_crossings(order, stars, cycle)
    return L + crossing_weight * X

# -------- Genetic operators --------

def mutate_expr(expr: Node, prob_mut=0.1, max_depth=3) -> Node:
    if random.random() < prob_mut or max_depth <= 0:
        return random_expr(max_depth)
    else:
        if isinstance(expr, OpNode):
            expr.left = mutate_expr(expr.left, prob_mut, max(0, max_depth - 1))
            expr.right = mutate_expr(expr.right, prob_mut, max(0, max_depth - 1))
        return expr


def crossover_expr(expr1: Node, expr2: Node, prob_cross=0.7) -> Node:
    if random.random() > prob_cross:
        return expr1.copy()

    size1 = expr1.size()
    size2 = expr2.size()
    if size1 == 1 or size2 == 1:
        return expr2.copy()

    # Random subtree swap:
    cross_point1 = random.randint(1, size1 - 1)
    cross_point2 = random.randint(1, size2 - 1)

    subtree1 = expr1.get_subtree(cross_point1)
    subtree2 = expr2.get_subtree(cross_point2)

    new_expr = expr1.copy()
    new_expr = new_expr.replace_subtree(cross_point1, subtree2.copy())

    return new_expr

# -------- GEA driver --------
def run_gea(
    stars: List[Point],
    population_size=100,
    generations=100,
    crossing_weight=50.0,
    cycle=False,
    mutation_rate=0.2,
    crossover_rate=0.7,
    seed=42,
):
    random.seed(seed)
    pop = [random_expr(depth=3) for _ in range(population_size)]

    def fit(e): return fitness_expr(e, stars, crossing_weight, cycle)

    best = min(pop, key=fit)
    best_f = fit(best)

    for gen in range(generations):
        new_pop = []
        elite_n = max(1, population_size // 20)  # 5% elitism
        sorted_pop = sorted(pop, key=fit)
        elites = sorted_pop[:elite_n]
        new_pop.extend([e.copy() for e in elites])

        while len(new_pop) < population_size:
            # Tournament selection size=3
            contenders = random.sample(pop, 3)
            parent1 = min(contenders, key=fit)
            contenders = random.sample(pop, 3)
            parent2 = min(contenders, key=fit)

            if random.random() < crossover_rate:
                child = crossover_expr(parent1, parent2)
            else:
                child = parent1.copy()

            child = mutate_expr(child, mutation_rate)

            new_pop.append(child)

        pop = new_pop

        current_best = min(pop, key=fit)
        current_best_f = fit(current_best)
        if current_best_f < best_f:
            best = current_best
            best_f = current_best_f

        if gen % 10 == 0 or gen == generations - 1:
            print(f"Gen {gen}: Best fitness = {best_f:.3f}")

    best_order = expr_to_order(best, stars)
    return best, best_order, best_f

# -------- Main --------
if __name__ == "__main__":
    # Example star coordinates
    stars = [
        (2.1, 9.7), (1.4, 8.3), (3.0, 7.8), (5.2, 9.1), (7.4, 9.5),
        (9.1, 8.6), (8.5, 6.9), (9.7, 5.2), (7.8, 4.1), (5.6, 3.5),
        (3.3, 3.9), (1.5, 4.8), (0.9, 6.2), (2.6, 5.8), (4.2, 6.8),
        (6.1, 7.4), (6.7, 5.6), (4.9, 5.1), (3.8, 4.9), (2.0, 3.2),
    ]

    best_expr, best_order, best_score = run_gea(
        stars,
        population_size=100,
        generations=100,
        crossing_weight=50.0,
        cycle=False,
        mutation_rate=0.2,
        crossover_rate=0.7,
        seed=123,
    )

    print("\nBest expression:", best_expr)
    print("Best order:", best_order)
    print("Total length:", path_length(best_order, stars, cycle=False))
    print("Crossings:", count_crossings(best_order, stars, cycle=False))
    print("Objective score:", best_score)
