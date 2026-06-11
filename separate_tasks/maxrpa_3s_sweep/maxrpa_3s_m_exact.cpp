// Exact enumeration of 3-species, m-reaction CRNs satisfying deterministic
// maxRPA under the Gupta-Khammash stoichiometric characterization.
//
// Usage:
//   ./maxrpa_3s_m_exact 4
//
// The pass/fail decision is exact integer arithmetic.  Floating point is used
// only to print the final decimal portion.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using Vec3 = std::array<int, 3>;

struct Reaction {
    Vec3 alpha;
    Vec3 beta;
    Vec3 zeta;
};

static Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

static Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

static int dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static bool is_zero(const Vec3& v) {
    return v[0] == 0 && v[1] == 0 && v[2] == 0;
}

static bool collinear(const Vec3& a, const Vec3& b) {
    return is_zero(cross(a, b));
}

static bool equal_vec(const Vec3& a, const Vec3& b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

static std::vector<Vec3> build_complexes() {
    std::vector<Vec3> complexes;
    for (int x1 = 0; x1 <= 2; ++x1) {
        for (int x2 = 0; x2 <= 2; ++x2) {
            for (int x3 = 0; x3 <= 2; ++x3) {
                if (x1 + x2 + x3 <= 2) {
                    complexes.push_back({x1, x2, x3});
                }
            }
        }
    }
    std::sort(complexes.begin(), complexes.end());
    return complexes;
}

static std::vector<Reaction> build_reactions(const std::vector<Vec3>& complexes) {
    std::vector<Reaction> reactions;
    for (const Vec3& alpha : complexes) {
        for (const Vec3& beta : complexes) {
            if (!equal_vec(alpha, beta)) {
                reactions.push_back({alpha, beta, sub(beta, alpha)});
            }
        }
    }
    return reactions;
}

static int rank_rows(const std::vector<Vec3>& rows, Vec3& r1, Vec3& r2) {
    r1 = {0, 0, 0};
    r2 = {0, 0, 0};

    for (const Vec3& row : rows) {
        if (!is_zero(row)) {
            r1 = row;
            break;
        }
    }
    if (is_zero(r1)) {
        return 0;
    }

    for (const Vec3& row : rows) {
        if (!is_zero(row) && !collinear(r1, row)) {
            r2 = row;
            break;
        }
    }
    if (is_zero(r2)) {
        return 1;
    }

    const Vec3 normal = cross(r1, r2);
    for (const Vec3& row : rows) {
        if (dot(normal, row) != 0) {
            return 3;
        }
    }
    return 2;
}

static std::vector<Vec3> null_basis_rank_one(const Vec3& v) {
    const std::array<Vec3, 3> axes = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    std::vector<Vec3> basis;
    for (const Vec3& axis : axes) {
        const Vec3 candidate = cross(v, axis);
        if (is_zero(candidate)) {
            continue;
        }
        if (basis.empty()) {
            basis.push_back(candidate);
        } else if (!collinear(basis.front(), candidate)) {
            basis.push_back(candidate);
            break;
        }
    }
    return basis;
}

static bool forms_allow_opposite_signs(const std::vector<Vec3>& basis,
                                       const Vec3& za,
                                       const Vec3& zb) {
    std::vector<int> pa;
    std::vector<int> pb;
    pa.reserve(basis.size());
    pb.reserve(basis.size());
    for (const Vec3& q_basis : basis) {
        pa.push_back(dot(q_basis, za));
        pb.push_back(dot(q_basis, zb));
    }

    const bool a_zero = std::all_of(pa.begin(), pa.end(), [](int x) { return x == 0; });
    const bool b_zero = std::all_of(pb.begin(), pb.end(), [](int x) { return x == 0; });
    if (a_zero || b_zero) {
        return false;
    }

    bool induced_collinear = true;
    for (std::size_t i = 0; i < pa.size(); ++i) {
        for (std::size_t j = i + 1; j < pa.size(); ++j) {
            if (static_cast<long long>(pa[i]) * pb[j] != static_cast<long long>(pa[j]) * pb[i]) {
                induced_collinear = false;
                break;
            }
        }
        if (!induced_collinear) {
            break;
        }
    }

    if (!induced_collinear) {
        return true;
    }

    for (std::size_t i = 0; i < pa.size(); ++i) {
        if (pa[i] != 0) {
            return static_cast<long long>(pa[i]) * pb[i] < 0;
        }
    }
    return false;
}

static bool q_test(const std::vector<Vec3>& others, const Vec3& za, const Vec3& zb) {
    Vec3 r1;
    Vec3 r2;
    const int rank = rank_rows(others, r1, r2);

    if (rank == 3) {
        return false;
    }
    if (rank == 2) {
        const Vec3 q = cross(r1, r2);
        return static_cast<long long>(dot(q, za)) * dot(q, zb) < 0;
    }
    if (rank == 1) {
        return forms_allow_opposite_signs(null_basis_rank_one(r1), za, zb);
    }
    const std::vector<Vec3> basis = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    return forms_allow_opposite_signs(basis, za, zb);
}

static bool reactant_pattern_ok(const Reaction& a, const Reaction& b) {
    return a.alpha[0] != b.alpha[0] &&
           a.alpha[1] == b.alpha[1] &&
           a.alpha[2] == b.alpha[2];
}

static bool crn_passes(const std::vector<int>& idx, const std::vector<Reaction>& reactions) {
    const int m = static_cast<int>(idx.size());
    for (int ia = 0; ia < m; ++ia) {
        for (int ib = 0; ib < m; ++ib) {
            if (ia == ib) {
                continue;
            }

            const Reaction& a = reactions[idx[ia]];
            const Reaction& b = reactions[idx[ib]];
            if (!reactant_pattern_ok(a, b)) {
                continue;
            }

            std::vector<Vec3> others;
            others.reserve(std::max(0, m - 2));
            for (int j = 0; j < m; ++j) {
                if (j != ia && j != ib) {
                    others.push_back(reactions[idx[j]].zeta);
                }
            }

            if (q_test(others, a.zeta, b.zeta)) {
                return true;
            }
        }
    }
    return false;
}

static unsigned long long choose(unsigned long long n, unsigned long long k) {
    if (k > n) {
        return 0;
    }
    if (k > n - k) {
        k = n - k;
    }
    unsigned long long out = 1;
    for (unsigned long long i = 1; i <= k; ++i) {
        out = out * (n - k + i) / i;
    }
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

static void enumerate_combinations(int n,
                                   int m,
                                   int start,
                                   std::vector<int>& current,
                                   const std::vector<Reaction>& reactions,
                                   unsigned long long& count) {
    if (static_cast<int>(current.size()) == m) {
        if (crn_passes(current, reactions)) {
            ++count;
        }
        return;
    }

    const int remaining = m - static_cast<int>(current.size());
    for (int i = start; i <= n - remaining; ++i) {
        current.push_back(i);
        enumerate_combinations(n, m, i + 1, current, reactions, count);
        current.pop_back();
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <number_of_reactions>\n";
        return EXIT_FAILURE;
    }

    const int m = std::stoi(argv[1]);
    if (m < 2 || m > 6) {
        std::cerr << "This sweep expects 2 <= m <= 6.\n";
        return EXIT_FAILURE;
    }

    const std::vector<Vec3> complexes = build_complexes();
    const std::vector<Reaction> reactions = build_reactions(complexes);
    const unsigned long long total = choose(reactions.size(), static_cast<unsigned long long>(m));

    if (complexes.size() != 10 || reactions.size() != 90) {
        std::cerr << "Sanity check failed: complexes=" << complexes.size()
                  << ", reactions=" << reactions.size() << "\n";
        return EXIT_FAILURE;
    }

    unsigned long long count = 0;
    std::vector<int> current;
    current.reserve(m);
    enumerate_combinations(static_cast<int>(reactions.size()), m, 0, current, reactions, count);

    const unsigned long long g = gcd_ull(count, total);
    std::cout << "number of complexes: " << complexes.size() << " (expected 10)\n";
    std::cout << "number of possible directed reactions: " << reactions.size()
              << " (expected 90)\n";
    std::cout << "m reactions: " << m << "\n";
    std::cout << "total CRNs: " << total << "\n";
    std::cout << "maxRPA count: " << count << "\n";
    std::cout << "reduced fraction: " << (count / g) << " / " << (total / g) << "\n";
    std::cout << std::setprecision(17)
              << "decimal portion: " << static_cast<long double>(count) / total << "\n";
    return EXIT_SUCCESS;
}
