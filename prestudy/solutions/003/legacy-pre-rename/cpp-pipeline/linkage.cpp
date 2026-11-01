// Full task-008 ASCII pipeline, using only the C++17 standard library.
// No fast-math; each f32 call is an intentional reference rounding boundary.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Key = std::array<uint32_t, 5>;
struct KeyHash {
    size_t operator()(const Key& a) const {
        size_t h = 1469598103934665603ULL;
        for (auto x : a) { h ^= x; h *= 1099511628211ULL; }
        return h;
    }
};
struct Value {
    std::string text;
    std::vector<std::pair<uint16_t, uint32_t>> grams;
};
struct Field {
    std::unordered_map<std::string, uint32_t> ids;
    std::vector<Value> values;
    uint32_t add(const char* ptr, size_t width) {
        size_t len = 0;
        while (len < width && ptr[len]) ++len;
        std::string text(ptr, len);
        auto found = ids.find(text);
        if (found != ids.end()) return found->second;
        uint32_t id = static_cast<uint32_t>(values.size());
        Value value;
        value.text = text;
        std::vector<uint16_t> codes;
        for (size_t k = 1; k < len; ++k)
            codes.push_back((static_cast<uint16_t>(static_cast<unsigned char>(ptr[k-1])) << 8)
                           | static_cast<unsigned char>(ptr[k]));
        std::sort(codes.begin(), codes.end());
        for (auto code : codes) {
            if (!value.grams.empty() && value.grams.back().first == code)
                ++value.grams.back().second;
            else value.grams.emplace_back(code, 1);
        }
        values.push_back(std::move(value));
        ids.emplace(std::move(text), id);
        return id;
    }
};
double f32(double x) { return static_cast<double>(static_cast<float>(x)); }
double bigram(const Value& m, const Value& u) {
    const auto lm = m.text.size(), lu = u.text.size();
    if (!lu) return 0.;
    if (m.text == u.text) return 1.;
    double g = 0.;
    if (lm < 3 || lu < 3) {
        bool contains = m.text.find(u.text) != std::string::npos
                     || u.text.find(m.text) != std::string::npos;
        if (contains) g = f32(static_cast<double>(std::min(lm, lu))
                             / (static_cast<double>(lm + lu) - 1.));
    } else {
        size_t p = 0, q = 0;
        uint64_t count = 0;
        while (p < m.grams.size() && q < u.grams.size()) {
            if (m.grams[p].first == u.grams[q].first) {
                count += m.grams[p].second; ++p; ++q;
            } else if (m.grams[p].first < u.grams[q].first) ++p;
            else ++q;
        }
        g = f32(static_cast<double>(count) * 2. / static_cast<double>(lm + lu - 2));
    }
    int wink = 0;
    for (size_t k = 1; k <= 4; ++k) {
        size_t ml = std::min(k, lm), ul = std::min(k, lu);
        if (ml == ul && m.text.compare(0, ml, u.text, 0, ul) == 0) ++wink;
    }
    return f32(g + wink * (1. - g) / 10.);
}
struct Pair { int64_t i, j; double score; };
double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
}
}

extern "C" {
struct Output {
    int64_t count;
    int64_t perfect;
    int64_t candidates;
    int64_t* i;
    int64_t* j;
    double* score;
    double feature_s;
    double index_s;
    double search_s;
    double output_s;
    char* error;
};
void linkage_free(Output* out) {
    if (!out) return;
    std::free(out->i); std::free(out->j); std::free(out->score);
    std::free(out->error); delete out;
}
// Column buffers are fixed-width zero-padded ASCII strings. The first five
// columns are master, the second five using. Field 4 (state) is required.
Output* linkage_run(const char** buffers, const int64_t* widths, int64_t nm, int64_t nu,
                    const double* wm, const double* wnm, double minscore,
                    double minbigram, double biscale, int nthreads) {
    Output* out = new Output{};
    try {
        auto start = Clock::now();
        std::array<Field, 5> fields;
        std::vector<Key> master(nm), using_rows(nu);
        for (int v = 0; v < 5; ++v) {
            fields[v].ids.reserve(static_cast<size_t>(nm + nu));
            for (int64_t i = 0; i < nm; ++i)
                master[i][v] = fields[v].add(buffers[v] + i*widths[v], widths[v]);
            for (int64_t j = 0; j < nu; ++j)
                using_rows[j][v] = fields[v].add(buffers[v+5] + j*widths[v+5], widths[v+5]);
        }
        auto featured = Clock::now();
        std::unordered_map<Key, std::vector<int64_t>, KeyHash> exact;
        exact.reserve(nu);
        for (int64_t j = 0; j < nu; ++j) exact[using_rows[j]].push_back(j);
        std::vector<uint8_t> removed_master(nm, 0), removed_using(nu, 0);
        std::vector<Pair> perfect;
        for (int64_t i = 0; i < nm; ++i) {
            auto p = exact.find(master[i]);
            if (p == exact.end()) continue;
            removed_master[i] = 1;
            for (auto j : p->second) {
                perfect.push_back({i, j, 1.});
                removed_using[j] = 1;
            }
        }
        out->perfect = perfect.size();
        exact.clear(); exact.rehash(0);
        std::array<std::unordered_map<uint64_t, std::vector<int64_t>>, 4> index;
        for (auto& map : index) map.reserve(nu);
        for (int64_t j = 0; j < nu; ++j) {
            if (removed_using[j]) continue;
            for (int v = 0; v < 4; ++v) {
                uint64_t key = (static_cast<uint64_t>(using_rows[j][4]) << 32) | using_rows[j][v];
                index[v][key].push_back(j);
            }
        }
        auto indexed = Clock::now();
        nthreads = std::max(1, std::min(nthreads, static_cast<int>(std::max<int64_t>(1, nm))));
        std::vector<std::vector<Pair>> results(nthreads);
        std::vector<int64_t> candidate_counts(nthreads, 0);
        std::vector<std::exception_ptr> failures(nthreads);
        const auto& const_index = index;
        auto search = [&](int thread) {
            try {
                std::vector<int64_t> seen(nu, -1), candidates, winners;
                for (int64_t i = nm*thread/nthreads; i < nm*(thread+1)/nthreads; ++i) {
                    if (removed_master[i]) continue;
                    candidates.clear();
                    for (int v = 0; v < 4; ++v) {
                        if (fields[v].values[master[i][v]].text.empty()) continue;
                        uint64_t key = (static_cast<uint64_t>(master[i][4]) << 32) | master[i][v];
                        auto found = const_index[v].find(key);
                        if (found == const_index[v].end()) continue;
                        for (auto j : found->second) if (seen[j] != i) {
                            seen[j] = i; candidates.push_back(j);
                        }
                    }
                    candidate_counts[thread] += candidates.size();
                    double best = -std::numeric_limits<double>::infinity();
                    winners.clear();
                    for (auto j : candidates) {
                        double match = 0., nonmatch = 0.;
                        for (int v = 0; v < 5; ++v) {
                            const auto& m = fields[v].values[master[i][v]];
                            const auto& u = fields[v].values[using_rows[j][v]];
                            bool me = m.text.empty(), ue = u.text.empty();
                            if (v == 4) {
                                if (!me) match = f32(match + wm[v]);
                            } else if (!me) {
                                double bi = bigram(m, u);
                                double bonus = bi > minbigram
                                    ? biscale * std::pow(std::max(bi-minbigram, 0.), 1./3.) : 0.;
                                match = f32(match + wm[v] * (.5 * (bi*bi) + bonus));
                                if (!ue) {
                                    double pen = bi < minbigram-.2 ? 1.
                                        : bi < minbigram ? 1.-bi*bi : (1.-bi)*(1.-bi);
                                    nonmatch = f32(nonmatch + wnm[v] * pen);
                                }
                            }
                            if (me != ue) nonmatch = f32(nonmatch + wnm[v]*.3);
                        }
                        double score = f32(match / (match + nonmatch));
                        if (score > best) { best = score; winners.clear(); winners.push_back(j); }
                        else if (score == best) winners.push_back(j);
                    }
                    if (best >= minscore)
                        for (auto j : winners) results[thread].push_back({i, j, best});
                }
            } catch (...) { failures[thread] = std::current_exception(); }
        };
        if (nthreads == 1) search(0);
        else {
            std::vector<std::thread> pool;
            for (int t = 0; t < nthreads; ++t) pool.emplace_back(search, t);
            for (auto& t : pool) t.join();
        }
        for (auto& failure : failures) if (failure) std::rethrow_exception(failure);
        auto searched = Clock::now();
        for (int t = 0; t < nthreads; ++t) {
            out->candidates += candidate_counts[t];
            perfect.insert(perfect.end(), results[t].begin(), results[t].end());
        }
        std::sort(perfect.begin(), perfect.end(), [](const Pair& a, const Pair& b) {
            return a.i < b.i || (a.i == b.i && a.j < b.j);
        });
        out->count = perfect.size();
        out->i = static_cast<int64_t*>(std::malloc(perfect.size()*sizeof(int64_t)));
        out->j = static_cast<int64_t*>(std::malloc(perfect.size()*sizeof(int64_t)));
        out->score = static_cast<double*>(std::malloc(perfect.size()*sizeof(double)));
        if (out->count && (!out->i || !out->j || !out->score)) throw std::bad_alloc();
        for (size_t k = 0; k < perfect.size(); ++k) {
            out->i[k] = perfect[k].i; out->j[k] = perfect[k].j; out->score[k] = perfect[k].score;
        }
        auto ended = Clock::now();
        out->feature_s = seconds(start, featured);
        out->index_s = seconds(featured, indexed);
        out->search_s = seconds(indexed, searched);
        out->output_s = seconds(searched, ended);
    } catch (const std::exception& e) { out->error = strdup(e.what()); }
    return out;
}
double linkage_bigram(const char* master, const char* using_value) {
    Field f;
    auto m = f.add(master, std::strlen(master));
    auto u = f.add(using_value, std::strlen(using_value));
    return bigram(f.values[m], f.values[u]);
}
}
