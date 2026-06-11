// General maxRPA screen for n species and m reactions.
//
// Supports:
//   exact  ./maxrpa_ns_m_screen exact  <n> <m>
//   sample ./maxrpa_ns_m_screen sample <n> <m> <samples> <seed>
//
// The special-pair algebraic condition is exact.  The q-feasibility test uses
// nullspace geometry in small dimension with floating-point row reduction; this
// is suitable for the exploratory n=2..5, m=2..8 sweep.  The previously reported
// n=3 exact values are taken from the dedicated integer solvers.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Vec = std::vector<int>;

// One directed mass-action reaction alpha -> beta.  The vector zeta is the
// stoichiometric change beta - alpha, i.e. one column of the stoichiometric
// matrix in the Gupta-Khammash condition.
struct Reaction {
    Vec alpha;
    Vec beta;
    Vec zeta;
};

static constexpr double EPS = 1e-10;

static Vec sub(const Vec& a, const Vec& b) {
    Vec out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

static double dot_double(const std::vector<double>& a, const Vec& b) {
    double out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        out += a[i] * static_cast<double>(b[i]);
    }
    return out;
}

static bool equal_vec(const Vec& a, const Vec& b) {
    return a == b;
}

// Rank of the span of a list of stoichiometric vectors in R^n. Since these
// vectors are the columns of S, this is exactly rank(S).
static int vector_span_rank(const std::vector<Vec>& vectors, int n) {
    std::vector<std::vector<double>> a(vectors.size(), std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        for (int j = 0; j < n; ++j) {
            a[i][j] = static_cast<double>(vectors[i][j]);
        }
    }

    int r = 0;
    for (int c = 0; c < n && r < static_cast<int>(a.size()); ++c) {
        int pivot = r;
        for (int i = r + 1; i < static_cast<int>(a.size()); ++i) {
            if (std::fabs(a[i][c]) > std::fabs(a[pivot][c])) {
                pivot = i;
            }
        }
        if (std::fabs(a[pivot][c]) <= EPS) {
            continue;
        }
        std::swap(a[r], a[pivot]);
        const double inv = 1.0 / a[r][c];
        for (int j = c; j < n; ++j) {
            a[r][j] *= inv;
        }
        for (int i = r + 1; i < static_cast<int>(a.size()); ++i) {
            const double factor = a[i][c];
            if (std::fabs(factor) <= EPS) {
                continue;
            }
            for (int j = c; j < n; ++j) {
                a[i][j] -= factor * a[r][j];
            }
        }
        ++r;
    }
    return r;
}

// Species that never appear in any reactant or product complex are disconnected
// padding dimensions.  They should not make an otherwise full-rank embedded CRN
// fail.  A species is active only if it participates in at least one selected
// reaction complex; merely having a zero stoichiometric row is not enough.
static int active_species_count(const std::vector<int>& idx,
                                const std::vector<Reaction>& reactions,
                                int n) {
    std::vector<bool> active(n, false);
    for (int reaction_idx : idx) {
        const Reaction& reaction = reactions[reaction_idx];
        for (int species = 0; species < n; ++species) {
            if (reaction.alpha[species] != 0 || reaction.beta[species] != 0) {
                active[species] = true;
            }
        }
    }
    return static_cast<int>(std::count(active.begin(), active.end(), true));
}

// Enumerate all complexes with total molecularity <= 2 in n labelled species.
// The recursion distributes at most two indistinguishable molecules over n
// coordinates.  For n=3 this gives the expected 10 complexes:
// 0, Xi, 2Xi, and Xi+Xj.
static void gen_complexes_rec(int n, int pos, int remaining, Vec& cur, std::vector<Vec>& out) {
    if (pos == n) {
        out.push_back(cur);
        return;
    }
    for (int v = 0; v <= remaining; ++v) {
        cur[pos] = v;
        gen_complexes_rec(n, pos + 1, remaining - v, cur, out);
    }
}

static std::vector<Vec> build_complexes(int n) {
    std::vector<Vec> complexes;
    Vec cur(n, 0);
    gen_complexes_rec(n, 0, 2, cur, complexes);
    std::sort(complexes.begin(), complexes.end());
    return complexes;
}

// Build every nontrivial directed reaction between allowed complexes.  Reversing
// a reaction gives a distinct reaction, while alpha -> alpha is excluded.
static std::vector<Reaction> build_reactions(const std::vector<Vec>& complexes) {
    std::vector<Reaction> reactions;
    for (const Vec& alpha : complexes) {
        for (const Vec& beta : complexes) {
            if (!equal_vec(alpha, beta)) {
                reactions.push_back({alpha, beta, sub(beta, alpha)});
            }
        }
    }
    return reactions;
}

// Return a basis for {q : A q = 0}, where the rows of A are the stoichiometric
// changes of the non-special reactions.  In the maxRPA test, q must annihilate
// all these rows before it can assign opposite signs to the two special
// reactions.
//
// This is Gaussian elimination to reduced row echelon form.  Pivot columns are
// dependent variables; every non-pivot column is set once to 1 to construct one
// basis vector of the nullspace.  The entries here are small integers, but this
// exploratory sweep uses double arithmetic for speed and simplicity.
static std::vector<std::vector<double>> nullspace_basis(const std::vector<Vec>& rows, int n) {
    std::vector<std::vector<double>> a(rows.size(), std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (int j = 0; j < n; ++j) {
            a[i][j] = static_cast<double>(rows[i][j]);
        }
    }

    std::vector<int> pivot_cols;
    int r = 0;
    for (int c = 0; c < n && r < static_cast<int>(a.size()); ++c) {
        int pivot = r;
        for (int i = r + 1; i < static_cast<int>(a.size()); ++i) {
            if (std::fabs(a[i][c]) > std::fabs(a[pivot][c])) {
                pivot = i;
            }
        }
        if (std::fabs(a[pivot][c]) <= EPS) {
            continue;
        }
        std::swap(a[r], a[pivot]);
        const double inv = 1.0 / a[r][c];
        for (int j = c; j < n; ++j) {
            a[r][j] *= inv;
        }
        for (int i = 0; i < static_cast<int>(a.size()); ++i) {
            if (i == r) {
                continue;
            }
            const double factor = a[i][c];
            if (std::fabs(factor) <= EPS) {
                continue;
            }
            for (int j = c; j < n; ++j) {
                a[i][j] -= factor * a[r][j];
            }
        }
        pivot_cols.push_back(c);
        ++r;
    }

    std::vector<bool> is_pivot(n, false);
    for (int c : pivot_cols) {
        is_pivot[c] = true;
    }

    std::vector<std::vector<double>> basis;
    for (int free_col = 0; free_col < n; ++free_col) {
        if (is_pivot[free_col]) {
            continue;
        }
        std::vector<double> v(n, 0.0);
        v[free_col] = 1.0;
        for (int row = 0; row < static_cast<int>(pivot_cols.size()); ++row) {
            v[pivot_cols[row]] = -a[row][free_col];
        }
        basis.push_back(v);
    }
    return basis;
}

// Given a nullspace basis, decide whether some admissible q can satisfy
//
//     q . za > 0  and  q . zb < 0.
//
// Write q = sum_i c_i b_i.  Then the two inequalities become
//
//     c . pa > 0  and  c . pb < 0,
//
// where pa_i = b_i . za and pb_i = b_i . zb.  If the two induced forms are not
// collinear, there is always a coefficient direction c that gives opposite
// signs.  If they are collinear, opposite signs are possible only when they
// point in opposite directions.
static bool forms_allow_opposite_signs(const std::vector<std::vector<double>>& basis,
                                       const Vec& za,
                                       const Vec& zb) {
    const int d = static_cast<int>(basis.size());
    if (d == 0) {
        return false;
    }

    std::vector<double> pa(d, 0.0);
    std::vector<double> pb(d, 0.0);
    for (int i = 0; i < d; ++i) {
        pa[i] = dot_double(basis[i], za);
        pb[i] = dot_double(basis[i], zb);
    }

    const auto norm2 = [](const std::vector<double>& v) {
        double out = 0.0;
        for (double x : v) {
            out += x * x;
        }
        return out;
    };

    if (norm2(pa) <= EPS * EPS || norm2(pb) <= EPS * EPS) {
        return false;
    }

    bool collinear = true;
    for (int i = 0; i < d; ++i) {
        for (int j = i + 1; j < d; ++j) {
            if (std::fabs(pa[i] * pb[j] - pa[j] * pb[i]) > 1e-9) {
                collinear = false;
                break;
            }
        }
        if (!collinear) {
            break;
        }
    }
    if (!collinear) {
        return true;
    }

    for (int i = 0; i < d; ++i) {
        if (std::fabs(pa[i]) > EPS) {
            return pa[i] * pb[i] < -EPS;
        }
    }
    return false;
}

// q-feasibility part of the deterministic maxRPA characterization:
// other reactions must have zero q-charge, while the two special reactions must
// have strictly opposite q-charges.
static bool q_test(const std::vector<Vec>& others, const Vec& za, const Vec& zb, int n) {
    return forms_allow_opposite_signs(nullspace_basis(others, n), za, zb);
}

// Structural rank condition for the selected stoichiometric matrix
// S = [zeta_1 ... zeta_m].  We require full row rank only after ignoring species
// that are completely disconnected from the selected CRN.  Thus an embedded
// two-species network inside a five-species ambient space can pass if its two
// active rows are full rank, while a species that participates but is
// stoichiometrically dependent still causes failure.
static bool full_active_row_rank_S(const std::vector<int>& idx,
                                   const std::vector<Reaction>& reactions,
                                   int n) {
    std::vector<Vec> zetas;
    zetas.reserve(idx.size());
    for (int reaction_idx : idx) {
        zetas.push_back(reactions[reaction_idx].zeta);
    }
    return vector_span_rank(zetas, n) == active_species_count(idx, reactions, n);
}

// Algebraic/reactant-pattern condition for the special pair.  The output species
// is fixed to X1, so the two reactants must differ in alpha[0] and agree in all
// other species coordinates.
static bool reactant_pattern_ok(const Reaction& a, const Reaction& b, int n) {
    if (a.alpha[0] == b.alpha[0]) {
        return false;
    }
    for (int i = 1; i < n; ++i) {
        if (a.alpha[i] != b.alpha[i]) {
            return false;
        }
    }
    return true;
}

// A CRN passes if at least one ordered special pair (a,b) satisfies both parts
// of the characterization.  The order matters because q . zeta_a must be
// positive and q . zeta_b negative, although the outer CRN itself is an
// unordered set of reaction indices.
static bool crn_passes(const std::vector<int>& idx,
                       const std::vector<Reaction>& reactions,
                       int n) {
    if (!full_active_row_rank_S(idx, reactions, n)) {
        return false;
    }

    const int m = static_cast<int>(idx.size());
    for (int ia = 0; ia < m; ++ia) {
        for (int ib = 0; ib < m; ++ib) {
            if (ia == ib) {
                continue;
            }
            const Reaction& a = reactions[idx[ia]];
            const Reaction& b = reactions[idx[ib]];
            if (!reactant_pattern_ok(a, b, n)) {
                continue;
            }

            std::vector<Vec> others;
            others.reserve(std::max(0, m - 2));
            for (int j = 0; j < m; ++j) {
                if (j != ia && j != ib) {
                    others.push_back(reactions[idx[j]].zeta);
                }
            }

            if (q_test(others, a.zeta, b.zeta, n)) {
                return true;
            }
        }
    }
    return false;
}

using UInt128 = unsigned __int128;

// Compute binomial coefficients large enough for the n=5, m=10 totals.  Some of
// these totals exceed 64 bits, but the sampled/exact hit counts printed below
// still fit in unsigned long long for the workloads used here.
static UInt128 choose(unsigned long long n, unsigned long long k) {
    if (k > n) {
        return 0;
    }
    if (k > n - k) {
        k = n - k;
    }
    UInt128 out = 1;
    for (unsigned long long i = 1; i <= k; ++i) {
        out = out * (n - k + i) / i;
    }
    return out;
}

static std::string to_string_u128(UInt128 value) {
    if (value == 0) {
        return "0";
    }
    std::string out;
    while (value > 0) {
        const int digit = static_cast<int>(value % 10);
        out.push_back(static_cast<char>('0' + digit));
        value /= 10;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

static unsigned long long gcd_ull(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        const unsigned long long r = a % b;
        a = b;
        b = r;
    }
    return a;
}

// Exact mode visits every unordered m-subset of the directed reaction list in
// lexicographic order.  The increasing start index is what prevents duplicate
// CRNs such as {r1,r2} and {r2,r1}.
static void enumerate_combinations(int n_reactions,
                                   int m,
                                   int start,
                                   std::vector<int>& current,
                                   const std::vector<Reaction>& reactions,
                                   int n_species,
                                   unsigned long long& count) {
    if (static_cast<int>(current.size()) == m) {
        if (crn_passes(current, reactions, n_species)) {
            ++count;
        }
        return;
    }

    const int remaining = m - static_cast<int>(current.size());
    for (int i = start; i <= n_reactions - remaining; ++i) {
        current.push_back(i);
        enumerate_combinations(n_reactions, m, i + 1, current, reactions, n_species, count);
        current.pop_back();
    }
}

static unsigned long long run_exact(const std::vector<Reaction>& reactions, int n_species, int m) {
    unsigned long long count = 0;
    std::vector<int> current;
    current.reserve(m);
    enumerate_combinations(static_cast<int>(reactions.size()), m, 0, current, reactions, n_species, count);
    return count;
}

// Sample mode estimates the fraction by drawing random unordered CRNs.  We
// shuffle the full reaction list, take the first m reactions, and sort their
// indices so the downstream test sees the same representation as exact mode.
static unsigned long long run_sample(const std::vector<Reaction>& reactions,
                                     int n_species,
                                     int m,
                                     unsigned long long samples,
                                     unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::vector<int> pool(reactions.size());
    std::iota(pool.begin(), pool.end(), 0);
    std::vector<int> current(m);
    unsigned long long count = 0;

    for (unsigned long long s = 0; s < samples; ++s) {
        std::shuffle(pool.begin(), pool.end(), rng);
        std::copy(pool.begin(), pool.begin() + m, current.begin());
        std::sort(current.begin(), current.end());
        if (crn_passes(current, reactions, n_species)) {
            ++count;
        }
    }
    return count;
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " exact  <n_species> <m_reactions>\n"
                  << "  " << argv[0] << " sample <n_species> <m_reactions> <samples> <seed>\n";
        return EXIT_FAILURE;
    }

    const std::string mode = argv[1];
    const int n_species = std::stoi(argv[2]);
    const int m = std::stoi(argv[3]);
    if (n_species < 2 || n_species > 5 || m < 2 || m > 10) {
        std::cerr << "Expected 2 <= n_species <= 5 and 2 <= m <= 10.\n";
        return EXIT_FAILURE;
    }

    const std::vector<Vec> complexes = build_complexes(n_species);
    const std::vector<Reaction> reactions = build_reactions(complexes);
    const UInt128 total = choose(reactions.size(), static_cast<unsigned long long>(m));

    // In exact mode, denominator is the full CRN count C(R,m).  In sample mode,
    // denominator is the number of Monte Carlo samples; total CRNs is still
    // printed as context for the sampled search space.
    unsigned long long count = 0;
    unsigned long long denominator = 0;
    unsigned long long samples = 0;
    unsigned long long seed = 0;

    if (mode == "exact") {
        count = run_exact(reactions, n_species, m);
        denominator = static_cast<unsigned long long>(total);
    } else if (mode == "sample") {
        if (argc != 6) {
            std::cerr << "Sample mode requires <samples> and <seed>.\n";
            return EXIT_FAILURE;
        }
        samples = std::stoull(argv[4]);
        seed = std::stoull(argv[5]);
        count = run_sample(reactions, n_species, m, samples, seed);
        denominator = samples;
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return EXIT_FAILURE;
    }

    const unsigned long long g = gcd_ull(count, denominator);
    std::cout << "mode: " << mode << "\n";
    std::cout << "number of species: " << n_species << "\n";
    std::cout << "number of complexes: " << complexes.size() << "\n";
    std::cout << "number of possible directed reactions: " << reactions.size() << "\n";
    std::cout << "m reactions: " << m << "\n";
    std::cout << "total CRNs: " << to_string_u128(total) << "\n";
    if (mode == "sample") {
        std::cout << "samples: " << samples << "\n";
        std::cout << "seed: " << seed << "\n";
    }
    std::cout << "maxRPA count: " << count << "\n";
    std::cout << "reduced fraction: " << (count / g) << " / " << (denominator / g) << "\n";
    std::cout << std::setprecision(17)
              << "decimal portion: " << static_cast<long double>(count) / denominator << "\n";
    return EXIT_SUCCESS;
}
