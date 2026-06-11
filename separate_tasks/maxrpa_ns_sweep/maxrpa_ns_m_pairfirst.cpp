// Pair-first exact maxRPA enumerator.
//
// Instead of enumerating all CRNs and then looking for a special reactant pair,
// this enumerates a candidate special ordered pair first and then chooses the
// remaining reactions.  To avoid double-counting CRNs with several special
// pairs, a completed CRN is accepted only when the chosen pair is the canonical
// lexicographically first special ordered pair in that CRN.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using Vec = std::vector<int>;

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

static int active_species_count(const std::vector<int>& crn,
                                const std::vector<Reaction>& reactions,
                                int n) {
    std::vector<bool> active(n, false);
    for (int idx : crn) {
        const Reaction& reaction = reactions[idx];
        for (int species = 0; species < n; ++species) {
            if (reaction.alpha[species] != 0 || reaction.beta[species] != 0) {
                active[species] = true;
            }
        }
    }
    return static_cast<int>(std::count(active.begin(), active.end(), true));
}

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

static std::vector<Reaction> build_reactions(const std::vector<Vec>& complexes) {
    std::vector<Reaction> reactions;
    for (const Vec& alpha : complexes) {
        for (const Vec& beta : complexes) {
            if (alpha != beta) {
                reactions.push_back({alpha, beta, sub(beta, alpha)});
            }
        }
    }
    return reactions;
}

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

    auto norm2 = [](const std::vector<double>& v) {
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

static bool q_test(const std::vector<int>& crn,
                   int ia,
                   int ib,
                   const std::vector<Reaction>& reactions,
                   int n) {
    std::vector<Vec> others;
    others.reserve(crn.size() - 2);
    for (int idx : crn) {
        if (idx != ia && idx != ib) {
            others.push_back(reactions[idx].zeta);
        }
    }
    return forms_allow_opposite_signs(
        nullspace_basis(others, n),
        reactions[ia].zeta,
        reactions[ib].zeta
    );
}

static bool full_active_row_rank_S(const std::vector<int>& crn,
                                   const std::vector<Reaction>& reactions,
                                   int n) {
    std::vector<Vec> zetas;
    zetas.reserve(crn.size());
    for (int idx : crn) {
        zetas.push_back(reactions[idx].zeta);
    }
    return vector_span_rank(zetas, n) == active_species_count(crn, reactions, n);
}

static std::pair<int, int> canonical_q_passing_pair(const std::vector<int>& crn,
                                                    const std::vector<Reaction>& reactions,
                                                    int n) {
    if (!full_active_row_rank_S(crn, reactions, n)) {
        return {-1, -1};
    }

    for (int a : crn) {
        for (int b : crn) {
            if (
                a != b &&
                reactant_pattern_ok(reactions[a], reactions[b], n) &&
                q_test(crn, a, b, reactions, n)
            ) {
                return {a, b};
            }
        }
    }
    return {-1, -1};
}

static unsigned long long enumerate_rest(const std::vector<int>& available,
                                         int pos,
                                         int needed,
                                         std::vector<int>& crn,
                                         int pair_a,
                                         int pair_b,
                                         const std::vector<Reaction>& reactions,
                                         int n) {
    if (needed == 0) {
        std::vector<int> completed = crn;
        std::sort(completed.begin(), completed.end());
        const auto canon = canonical_q_passing_pair(completed, reactions, n);
        return (canon.first == pair_a && canon.second == pair_b) ? 1ULL : 0ULL;
    }
    if (pos > static_cast<int>(available.size()) - needed) {
        return 0ULL;
    }

    unsigned long long count = 0;
    for (int i = pos; i <= static_cast<int>(available.size()) - needed; ++i) {
        crn.push_back(available[i]);
        count += enumerate_rest(available, i + 1, needed - 1, crn, pair_a, pair_b, reactions, n);
        crn.pop_back();
    }
    return count;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <n_species> <m_reactions>\n";
        return EXIT_FAILURE;
    }
    const int n = std::stoi(argv[1]);
    const int m = std::stoi(argv[2]);
    if (n < 2 || n > 5 || m < 2 || m > 6) {
        std::cerr << "Expected 2 <= n <= 5 and 2 <= m <= 6.\n";
        return EXIT_FAILURE;
    }

    const auto reactions = build_reactions(build_complexes(n));
    std::vector<int> all(reactions.size());
    std::iota(all.begin(), all.end(), 0);

    unsigned long long count = 0;
    std::vector<int> crn;
    crn.reserve(m);

    for (int a = 0; a < static_cast<int>(reactions.size()); ++a) {
        for (int b = 0; b < static_cast<int>(reactions.size()); ++b) {
            if (a == b || !reactant_pattern_ok(reactions[a], reactions[b], n)) {
                continue;
            }
            std::vector<int> available;
            available.reserve(reactions.size() - 2);
            for (int idx : all) {
                if (idx != a && idx != b) {
                    available.push_back(idx);
                }
            }
            crn = {a, b};
            count += enumerate_rest(available, 0, m - 2, crn, a, b, reactions, n);
        }
    }

    std::cout << "number of species: " << n << "\n";
    std::cout << "number of possible directed reactions: " << reactions.size() << "\n";
    std::cout << "m reactions: " << m << "\n";
    std::cout << "maxRPA count: " << count << "\n";
}
