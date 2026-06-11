// Exact enumeration of 3-species, 5-reaction CRNs satisfying deterministic
// maxRPA under the Gupta-Khammash stoichiometric characterization.
//
// The implementation is intentionally standalone and uses only integer
// arithmetic.  In dimension three, the q-test can be reduced to exact rank,
// cross-product, and collinearity checks.

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
            if (equal_vec(alpha, beta)) {
                continue;
            }
            reactions.push_back({alpha, beta, sub(beta, alpha)});
        }
    }
    return reactions;
}

static int rank_rows(const std::array<Vec3, 3>& rows, Vec3& r1, Vec3& r2) {
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

    // If the two induced forms are not collinear in the nullspace dual, choose a
    // vector that makes them have opposite signs.  If they are collinear, this is
    // possible exactly when their scalar multiplier is negative.
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

static std::vector<Vec3> null_basis_rank_one(const Vec3& v) {
    const std::array<Vec3, 3> axes = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    std::vector<Vec3> candidates;
    for (const Vec3& axis : axes) {
        Vec3 c = cross(v, axis);
        if (!is_zero(c)) {
            candidates.push_back(c);
        }
    }

    std::vector<Vec3> basis;
    for (const Vec3& candidate : candidates) {
        if (basis.empty()) {
            basis.push_back(candidate);
        } else if (!collinear(basis.front(), candidate)) {
            basis.push_back(candidate);
            break;
        }
    }
    return basis;
}

static bool q_test(const std::array<Vec3, 3>& others,
                   const Vec3& za,
                   const Vec3& zb) {
    Vec3 r1;
    Vec3 r2;
    const int rank = rank_rows(others, r1, r2);

    if (rank == 3) {
        return false;
    }

    if (rank == 2) {
        const Vec3 q = cross(r1, r2);
        const int da = dot(q, za);
        const int db = dot(q, zb);
        return static_cast<long long>(da) * db < 0;
    }

    if (rank == 1) {
        const std::vector<Vec3> basis = null_basis_rank_one(r1);
        return forms_allow_opposite_signs(basis, za, zb);
    }

    const std::vector<Vec3> basis = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    return forms_allow_opposite_signs(basis, za, zb);
}

static bool reactant_pattern_ok(const Reaction& a, const Reaction& b) {
    return a.alpha[0] != b.alpha[0] &&
           a.alpha[1] == b.alpha[1] &&
           a.alpha[2] == b.alpha[2];
}

static bool crn_passes(const std::array<int, 5>& idx,
                       const std::vector<Reaction>& reactions) {
    for (int ia = 0; ia < 5; ++ia) {
        for (int ib = 0; ib < 5; ++ib) {
            if (ia == ib) {
                continue;
            }

            const Reaction& a = reactions[idx[ia]];
            const Reaction& b = reactions[idx[ib]];
            if (!reactant_pattern_ok(a, b)) {
                continue;
            }

            std::array<Vec3, 3> others;
            int k = 0;
            for (int j = 0; j < 5; ++j) {
                if (j != ia && j != ib) {
                    others[k++] = reactions[idx[j]].zeta;
                }
            }

            if (q_test(others, a.zeta, b.zeta)) {
                return true;
            }
        }
    }
    return false;
}

static unsigned long long choose_5(unsigned long long n) {
    return n * (n - 1) * (n - 2) * (n - 3) * (n - 4) / 120ULL;
}

static unsigned long long gcd_ull(unsigned long long a, unsigned long long b) {
    while (b != 0) {
        const unsigned long long r = a % b;
        a = b;
        b = r;
    }
    return a;
}

int main() {
    const std::vector<Vec3> complexes = build_complexes();
    const std::vector<Reaction> reactions = build_reactions(complexes);
    const unsigned long long total = choose_5(reactions.size());

    if (complexes.size() != 10 || reactions.size() != 90 || total != 43949268ULL) {
        std::cerr << "Sanity check failed: complexes=" << complexes.size()
                  << ", reactions=" << reactions.size()
                  << ", total=" << total << "\n";
        return EXIT_FAILURE;
    }

    unsigned long long count = 0;
    std::array<int, 5> idx;

    for (idx[0] = 0; idx[0] < 86; ++idx[0]) {
        for (idx[1] = idx[0] + 1; idx[1] < 87; ++idx[1]) {
            for (idx[2] = idx[1] + 1; idx[2] < 88; ++idx[2]) {
                for (idx[3] = idx[2] + 1; idx[3] < 89; ++idx[3]) {
                    for (idx[4] = idx[3] + 1; idx[4] < 90; ++idx[4]) {
                        if (crn_passes(idx, reactions)) {
                            ++count;
                        }
                    }
                }
            }
        }
    }

    const unsigned long long g = gcd_ull(count, total);

    std::cout << "number of complexes: " << complexes.size() << " (expected 10)\n";
    std::cout << "number of possible directed reactions: " << reactions.size()
              << " (expected 90)\n";
    std::cout << "total CRNs: " << total << " (expected 43,949,268)\n";
    std::cout << "maxRPA count: " << count << "\n";
    std::cout << "reduced fraction: " << (count / g) << " / " << (total / g) << "\n";
    std::cout << std::setprecision(17)
              << "decimal portion: " << static_cast<long double>(count) / total << "\n";

    return EXIT_SUCCESS;
}
