/***
 * Sparse adjacency matrix in CSR format, for graph (not blockmodel) storage.
 * Designed for future GPU offloading via OpenMP target map on MI300A.
 */
#ifndef SBP_MATRIX_CSR_HPP
#define SBP_MATRIX_CSR_HPP

#include <vector>
#include "typedefs.hpp"

/**
 * CSR-format adjacency matrix for a graph.
 *
 * Three contiguous arrays of equal or related length:
 *   row_ptrs   : size V+1  — row_ptrs[v]..row_ptrs[v+1] is the range for vertex v
 *   col_indices: size E    — destination vertices in row order
 *   vals       : size E    — edge weights, parallel to col_indices (filled with 1
 *                            for unweighted graphs; ready for weighted edges later)
 *
 * All three expose .data() for direct use with `#pragma omp target map`.
 */
class CSR {
  public:
    CSR() = default;

    /// Build from an adjacency list. Edges are stored in the order they appear
    /// in each row of neighbor_list.
    CSR(const NeighborList &neighbor_list, long num_vertices, long num_edges) {
        row_ptrs.resize(num_vertices + 1, 0);
        col_indices.reserve(num_edges);
        vals.reserve(num_edges);

        // Count edges per row (row_ptrs will hold counts temporarily)
        for (long v = 0; v < num_vertices; ++v) {
            row_ptrs[v + 1] = static_cast<long>(neighbor_list[v].size());
        }
        // Prefix-sum to produce final row pointers
        for (long v = 0; v < num_vertices; ++v) {
            row_ptrs[v + 1] += row_ptrs[v];
        }
        // Fill col_indices and vals
        for (long v = 0; v < num_vertices; ++v) {
            for (const long neighbor : neighbor_list[v]) {
                col_indices.push_back(neighbor);
                vals.push_back(1);
            }
        }
    }

    /// Number of rows (vertices).
    long num_rows() const { return static_cast<long>(row_ptrs.size()) - 1; }

    /// Number of stored entries (edges).
    long nnz() const { return static_cast<long>(col_indices.size()); }

    /// Out-degree of vertex v.
    long degree(long v) const {
        return row_ptrs[v + 1] - row_ptrs[v];
    }

    /// View of the neighbors of vertex v (col-index array slice).
    NeighborView neighbors(long v) const {
        return NeighborView(col_indices.data() + row_ptrs[v],
                            row_ptrs[v + 1] - row_ptrs[v]);
    }

    /// Raw pointer to row_ptrs array (for omp target map).
    const long* row_ptrs_data() const { return row_ptrs.data(); }
    /// Raw pointer to col_indices array (for omp target map).
    const long* col_indices_data() const { return col_indices.data(); }
    /// Raw pointer to vals array (for omp target map).
    const long* vals_data() const { return vals.data(); }

    std::vector<long> row_ptrs;
    std::vector<long> col_indices;
    std::vector<long> vals;
};

#endif // SBP_MATRIX_CSR_HPP
