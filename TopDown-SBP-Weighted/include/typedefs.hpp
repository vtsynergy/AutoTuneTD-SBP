/**
 * Useful type definitions and such.
 */
#ifndef SBP_TYPEDEFS_HPP
#define SBP_TYPEDEFS_HPP

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tsl/robin_map.h"

/// Stores a list of edges for a given structure (vertex or block) and their weights. Only stores the second portion
/// of the edge, so additional information is needed to reconstruct the edge. i.e.: for a list of edges (1-->2), (1--4),
/// (1-->6), only [2, 4, 6] and the corresponding weights will be stored.
struct EdgeWeights {
    std::vector<long> indices;
    std::vector<long> values;

    void print() {
        if (this->indices.empty()) {
            std::cout << "[]" << std::endl;
            return;
        }
        std::cout << "[" << this->indices[0] << ": " << this->values[0] << ", ";
        for (size_t num_printed = 1; num_printed < this->indices.size() - 1; num_printed++) {
            if (num_printed % 25 == 0) {
                std::cout << std::endl << " ";
            }
            std::cout << this->indices[num_printed] << ": " << this->values[num_printed] << ", ";
        }
        std::cout << this->indices[this->indices.size() - 1] << ": " << this->values[this->indices.size() - 1] << "]" << std::endl;
    }
};

/// Stores intermediate evaluation info & timers for later printing.
struct PartialProfile {
    double iteration = -1;
    double mdl = -1;
    double normalized_mdl_v1 = -1;
    double modularity = -1;
    long mcmc_iterations = -1;
    double mcmc_time = 0.0;
    double mcmc_sequential_time = 0.0;
    double mcmc_parallel_time = 0.0;
    double mcmc_vertex_move_time = 0.0;
    ulong mcmc_moves = 0;
    double block_merge_time = 0.0;
    double block_merge_loop_time = 0.0;
    double block_split_time = 0.0;
    double block_split_loop_time = 0.0;
    double blockmodel_build_time = 0.0;
    double finetune_time = 0.0;
    double load_balancing_time = 0.0;
    double sort_time = 0.0;
    double access_time = 0.0;
    double update_assignment = 0.0;
    double total_time = 0.0;
    long num_blocks = 0;
};

/// Used to hash a pair of integers. Source: https://codeforces.com/blog/entry/21853
struct longPairHash {
    size_t operator() (const std::pair<long, long> &pair) const {
        return std::hash<long long>() (((long long) pair.first) ^ (((long long) pair.second) << 32));
    }
};

typedef std::vector<std::vector<long>> NeighborList;

/**
 * Non-owning view into a contiguous range of neighbor ids (a row of a CSR matrix).
 * Supports range-for, .size(), operator[], .empty(), and .to_vector().
 * Lifetime is bound to the CSR object that owns the underlying array.
 */
struct NeighborView {
    NeighborView() : ptr(nullptr), len(0) {}
    NeighborView(const long* ptr, long len) : ptr(ptr), len(len) {}

    const long* begin() const { return ptr; }
    const long* end()   const { return ptr + len; }
    long size()         const { return len; }
    bool empty()        const { return len == 0; }
    const long& operator[](long i) const { return ptr[i]; }

    /// Copy into a new std::vector<long>.
    std::vector<long> to_vector() const {
        return std::vector<long>(ptr, ptr + len);
    }

    const long* ptr;
    long len;
};

template <typename T>
struct SparseVector {
    std::vector<long>    idx;   // The index of the corresponding element in data
    std::vector<T>      data;  // The non-zero values of the vector
    // /// Returns the sum of all elements in data.
    // inline T sum() {
    //     T result;
    //     for (const T &value: this->data) {
    //         result += value;
    //     }
    //     return result;
    // }
    inline SparseVector<T> operator/(const double &rhs) {
        SparseVector<T> result;
        for (long i = 0; i < this->idx.size(); ++i) {
            result.idx.push_back(this->idx[i]);
            result.data.push_back(this->data[i] / rhs); 
        }
        return result;
    }
};

//template <typename T>
//using MapVector = std::unordered_map<long, T>;

//template <typename T>
//using MapVector = absl::flat_hash_map<long, T>;

template <typename T>
using MapVector = tsl::robin_map<long, T>;
using LongEntry = std::pair<long, long>;

struct Merge {
    long block = -1;
    long proposal = -1;
    double delta_entropy = std::numeric_limits<double>::max();
};

struct Membership {
    long vertex = -1;
    long block = -1;
};

struct Vertex {
    long id;
    long out_degree;
    long in_degree;  // maybe add self-edge? that way, degree = out_degree + in_degree - self_edge..., but I don't think that we need total degree
};

const Vertex InvalidVertex { -1, 0, 0 };

//const Membership NullMembership { -1, -1 };

struct VertexMove {
    double delta_entropy;
    bool did_move;
    long vertex;
    long proposed_block;
};

struct VertexMove_v2 {
    double delta_entropy;
    bool did_move;
    long vertex;
    long proposed_block;
    EdgeWeights out_edges;
    EdgeWeights in_edges;
};

struct VertexMove_v3 {
    double delta_entropy;
    bool did_move;
    Vertex vertex;
    long proposed_block;
    EdgeWeights out_edges;
    EdgeWeights in_edges;
};

typedef std::unordered_map<std::pair<long, long>, long, longPairHash> PairIndexVector;

namespace map_vector {
/// Returns either 0, or the value stored in `vector[key]` if it exists.
    inline long get(const MapVector<long> &vector, long key) {
        const auto iterator = vector.find(key);
        if (iterator == vector.end())
            return 0;
        return iterator->second;
    }
}  // namespace map_vector

/// Returns either 0, or the value stored in `vector[key]` if it exists.
inline long get(const PairIndexVector &vector, const std::pair<long, long> &key) {
    const auto iterator = vector.find(key);
    if (iterator == vector.end())
        return 0;
    return iterator->second;
}

// template<class T>
// using SparseVector = std::vector
// typedef struct proposal_evaluation_t {
//     long proposed_block;
//     double delta_entropy;
// } ProposalEvaluation;

#endif // SBP_TYPEDEFS_HPP
