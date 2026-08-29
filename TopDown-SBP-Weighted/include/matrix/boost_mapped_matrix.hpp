/***
 * Sparse Matrix based on the Boost mapped_matrix
 */
#ifndef CPPSBP_PARTITION_SPARSE_BOOST_MAPPED_MATRIX_HPP
#define CPPSBP_PARTITION_SPARSE_BOOST_MAPPED_MATRIX_HPP

#include <boost/numeric/ublas/matrix_sparse.hpp>

#include "csparse_matrix.hpp"
// TODO: figure out where to put utils.hpp so this never happens
#include "../utils.hpp"

// #include <Eigen/Core>

/**
 * C++ interface of the dictionary (map of maps) sparse matrix
 */
// class BoostMappedMatrix : public CSparseMatrix {
class BoostMappedMatrix {
  public:
    BoostMappedMatrix() {}
    BoostMappedMatrix(long nrows, long ncols) : ncols(ncols), nrows(nrows) {
        // this->matrix = boost::numeric::ublas::coordinate_matrix<long>(this->nrows, this->ncols);
        this->matrix = boost::numeric::ublas::compressed_matrix<long>(this->nrows, this->ncols);
        // this->matrix = boost::numeric::ublas::mapped_matrix<long>(this->nrows, this->ncols);
        // long shape_array[2] = {this->nrows, this->ncols};
        this->shape = std::make_pair(this->nrows, this->ncols);
    }
    void add(long row, long col, long val);
    void add(long row, std::vector<long> cols, std::vector<long> values);
    BoostMappedMatrix copy();
    long get(long row, long col);
    std::vector<long> getcol(long col);
    std::vector<long> getrow(long row);
    EdgeWeights incoming_edges(long block);
    Indices nonzero();
    EdgeWeights outgoing_edges(long block);
    void sub(long row, long col, long val);
    long sum();
    std::vector<long> sum(long axis = 0);
    long trace();
    void update_edge_counts(long current_block, long proposed_block, std::vector<long> current_row,
                            std::vector<long> proposed_row, std::vector<long> current_col, std::vector<long> proposed_col);
    std::vector<long> values();
    std::pair<long, long> shape;

  private:
    void check_row_bounds(long row);
    void check_col_bounds(long col);
    long ncols;
    long nrows;
    boost::numeric::ublas::compressed_matrix<long> matrix;
    // boost::numeric::ublas::mapped_matrix<long> matrix;
    // boost::numeric::ublas::coordinate_matrix<long> matrix;
};

#endif // CPPSBP_PARTITION_SPARSE_BOOST_MAPPED_MATRIX_HPP
