/***
 * Dense Matrix that uses a 2D vector to store the blockmodel.
 */
#ifndef CPPSBP_PARTITION_DENSE_MATRIX_HPP
#define CPPSBP_PARTITION_DENSE_MATRIX_HPP

#include <omp.h>
#include <vector>

#include "csparse_matrix.hpp"
#include "delta.hpp"
#include "../utils.hpp"

#pragma omp begin declare target
/**
 * Used to access a DenseMatrix on the GPU.
 */
struct DenseMatrixView {
  const long* data;
  long nrows;
  long ncols;
  long get(long row, long col) const {
    return data[row * ncols + col];
  };
  /// Populates the values in `result` with the weighted neighbors of `block`.
  /// Assumptions: result is already initialized to 0 and is of size max(nrows, ncols),
  /// and blocks < max(nrows, ncols).
  void neighbors_weights(long* result, long block) const {
    // Outgoing edges
    for (long col = 0; col < this->ncols; ++col) {
        long value = this->data[block * this->ncols + col];
        result[col] += value;
    }
    // Incoming edges
    for (long row = 0; row < this->nrows; ++row) {
        long value = this->data[row * this->ncols + block];
        result[row] += value * (long)(row != block);  // skips diagonals by multiplying by 0 if row == block
    }
  }
};
#pragma omp end declare target

/**
 * Dense matrix implementation for blockmodel storage.
 * Uses std::vector<std::vector<long>> for simple, cache-friendly storage.
 * Best for blockmodels with many inter-block edges.
 */
class DenseMatrix : public ISparseMatrix {
  public:
    DenseMatrix() = default;
    DenseMatrix(long nrows, long ncols) {
        this->ncols = ncols;
        this->nrows = nrows;
        this->matrix = std::vector<long>(this->nrows * this->ncols, 0);
        this->shape = std::make_pair(this->nrows, this->ncols);
    }
    void add(long row, long col, long val) override;
    void clearrow(long row) override;
    void clearcol(long col) override;
    ISparseMatrix* copy() const override;
    long distinct_edges(long block) const override;
    std::vector<std::tuple<long, long, long>> entries() const override;
    long get(long row, long col) const override;
    std::vector<long> getcol(long col) const override;
    MapVector<long> getcol_sparse(long col) const override;
    const MapVector<long>& getcol_sparseref(long col) const override;
    void getcol_sparse(long col, MapVector<long> &col_vector) const override;
    std::vector<long> getrow(long row) const override;
    MapVector<long> getrow_sparse(long row) const override;
    void getrow_sparse(long row, MapVector<long> &row_vector) const override;
    const MapVector<long>& getrow_sparseref(long row) const override;
    /// Exposes a GPU-friendly view of the DenseMatrix.
    DenseMatrixView gpu_view() const {
      return DenseMatrixView{this->matrix.data(), this->nrows, this->ncols};
    }
    EdgeWeights incoming_edges(long block) const override;
    std::set<long> neighbors(long block) const override;
    MapVector<long> neighbors_weights(long block) const override;
    Indices nonzero() const override;
    EdgeWeights outgoing_edges(long block) const override;
    void setrow(long row, const MapVector<long> &vector) override;
    void setcol(long col, const MapVector<long> &vector) override;
    void sub(long row, long col, long val) override;
    long edges() const override;
    void print() const override;
    std::vector<long> sum(long axis) const override;
    long trace() const override;
    void update_edge_counts(long current_block, long proposed_block, std::vector<long> current_row,
                            std::vector<long> proposed_row, std::vector<long> current_col,
                            std::vector<long> proposed_col) override;
    void update_edge_counts(long current_block, long proposed_block, MapVector<long> current_row,
                            MapVector<long> proposed_row, MapVector<long> current_col,
                            MapVector<long> proposed_col) override;
    void update_edge_counts(const Delta &delta) override;
    bool validate(long row, long col, long val) const override;
    std::vector<long> values() const override;

  private:
    std::vector<long> matrix;
    // Ring buffer of 4 slots so concurrent sparseref callers (e.g. the 4 in
    // delta_mdl) each get a stable reference rather than all sharing one vector.
    static constexpr size_t SPARSEREF_SLOTS = 4;
    mutable std::array<MapVector<long>, SPARSEREF_SLOTS> temp_sparse_vectors;
    mutable size_t temp_vector_idx = 0;
};

#endif // CPPSBP_PARTITION_DENSE_MATRIX_HPP

