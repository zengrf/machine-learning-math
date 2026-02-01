"""
Generate (u, v, w, c) data by random walks down the strong Bruhat order.
Run with: sage -python schubert_random_walk_dataset.py
"""
import json
import random
from sage.all import SchubertPolynomialRing, Permutations, ZZ

# ---- Config ----
N_UV = 5  # u, v live in S_5
N_W = 7   # w lives in S_7
SAMPLES_POS = 50000
SAMPLES_ZERO = 50000
SEED = 32
OUT_PATH = "schubert_random_walk_data_100000.jsonl"
MAX_ATTEMPTS = 50000

random.seed(SEED)

# Schubert polynomial ring (same approach as in schubert_data_generation.ipynb)
X = SchubertPolynomialRing(ZZ)


def _inv_count_fast(p_list):
    """Fenwick-based inversion count for a 1-line permutation list."""
    n = len(p_list)
    bit = [0] * (n + 1)

    def _add(i, v):
        while i <= n:
            bit[i] += v
            i += i & -i

    def _sum(i):
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    inv = 0
    for i, val in enumerate(p_list, start=1):
        inv += i - 1 - _sum(val)
        _add(val, 1)
    return inv


def bruhat_length(p):
    """Return Coxeter length of a permutation (number of inversions)."""
    if hasattr(p, "length"):
        return int(p.length())
    if hasattr(p, "coxeter_length"):
        return int(p.coxeter_length())
    return _inv_count_fast(list(p))


def _lower_covers_bruteforce(p_list):
    """Fallback strong Bruhat lower covers from a 1-line list permutation."""
    n = len(p_list)
    covers = []
    base_len = bruhat_length(p_list)
    for i in range(n):
        for j in range(i + 1, n):
            if p_list[i] > p_list[j]:
                q = list(p_list)
                q[i], q[j] = q[j], q[i]
                if bruhat_length(q) == base_len - 1:
                    covers.append(q)
    return covers


def bruhat_lower_covers(p):
    """Return strong Bruhat lower covers for permutation p."""
    if hasattr(p, "bruhat_lower_covers"):
        return list(p.bruhat_lower_covers())
    # fallback for list/tuple
    return _lower_covers_bruteforce(list(p))


def random_walk_down(start, steps):
    """Random walk down the strong Bruhat order from start for given steps."""
    p = start
    for _ in range(steps):
        covers = bruhat_lower_covers(p)
        if not covers:
            break
        p = random.choice(covers)
    return p


def embed_perm(p_list, target_n):
    """Embed a permutation into S_target_n by appending fixed points."""
    return list(p_list) + list(range(len(p_list) + 1, target_n + 1))


def standardize_first_k(p_list, k):
    """Standardize the first k entries to a permutation in S_k."""
    first = list(p_list)[:k]
    order = sorted((v, i) for i, v in enumerate(first))
    ranks = [0] * k
    for rank, (_, i) in enumerate(order, start=1):
        ranks[i] = rank
    return ranks


def schubert_constant(u, v, w):
    """Compute c_{u,v}^w using Schubert polynomial multiplication."""
    u_emb = embed_perm(u, N_W)
    v_emb = embed_perm(v, N_W)
    product = X(u_emb) * X(v_emb)
    for perm, coeff in list(product):
        if list(perm) == list(w):
            return int(coeff)
    return 0


def to_oneline(p):
    return list(p)

def reduced_word(p_list):
    """Return a reduced word for a 1-line permutation list."""
    perm = Permutations(len(p_list))(p_list)
    return list(perm.reduced_word())

def _sample_one(target_c_positive):
    """Sample a single row matching the target c condition."""
    P_w = list(Permutations(N_W))
    for _ in range(MAX_ATTEMPTS):
        w = random.choice(P_w)
        # Use the standardized first N_UV entries of w as the S_5 start point.
        w5 = standardize_first_k(w, N_UV)
        w5_len = bruhat_length(w5)
        w_len = bruhat_length(w)
        if w_len > 2 * w5_len:
            continue
        steps_u = random.randint(0, w5_len)
        steps_v = random.randint(0, w5_len)
        u = random_walk_down(w5, steps_u)
        v = random_walk_down(w5, steps_v)
        if bruhat_length(u) + bruhat_length(v) != w_len:
            continue
        c = schubert_constant(u, v, w)
        if target_c_positive:
            if c <= 0:
                continue
        else:
            if c != 0:
                continue
        u_list = to_oneline(u)
        v_list = to_oneline(v)
        w_list = to_oneline(w)
        return [
            int(c),
            u_list,
            reduced_word(u_list),
            v_list,
            reduced_word(v_list),
            w_list,
            reduced_word(w_list),
        ]
    raise RuntimeError(
        "Failed to generate a sample matching the target condition. "
        "Try increasing MAX_ATTEMPTS or changing SEED."
    )


def _generate_and_write(target_c_positive, f, label):
    target = SAMPLES_POS if target_c_positive else SAMPLES_ZERO
    for i in range(target):
        row = _sample_one(target_c_positive)
        f.write(json.dumps(row) + "\n")
        if (i + 1) % 1000 == 0:
            print(f"{label}: wrote {i + 1}/{target}")


def main():
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        _generate_and_write(True, f, "positive")
        _generate_and_write(False, f, "zero")
    print(
        f"Wrote {SAMPLES_POS} positive and {SAMPLES_ZERO} zero rows to {OUT_PATH}"
    )


if __name__ == "__main__":
    main()
