#include "dense_matrix.hpp"

void DenseMatrix::add(long row, long col, long val) {
    check_row_bounds(row);
    check_col_bounds(col);
    this->matrix[row * this->ncols + col] += val;
}

void DenseMatrix::clearrow(long row) {
    check_row_bounds(row);
    for (long col = 0; col < this->ncols; ++col) {
        this->matrix[row * this->ncols + col] = 0;
    }
}

void DenseMatrix::clearcol(long col) {
    check_col_bounds(col);
    for (long row = 0; row < this->nrows; ++row) {
        this->matrix[row * this->ncols + col] = 0;
    }
}

ISparseMatrix* DenseMatrix::copy() const {
    DenseMatrix* dense_matrix = new DenseMatrix(this->nrows, this->ncols);
    for (long row = 0; row < this->nrows; ++row) {
        for (long col = 0; col < this->ncols; ++col) {
            dense_matrix->matrix[row * this->ncols + col] = this->matrix[row * this->ncols + col];
        }
    }
    return dense_matrix;
}

long DenseMatrix::distinct_edges(long block) const {
    check_row_bounds(block);
    check_col_bounds(block);
    long result = 0;
    // Count non-zero entries in row (outgoing edges)
    for (long col = 0; col < this->ncols; ++col) {
        if (this->matrix[block * this->ncols + col] != 0) {
            result++;
        }
    }
    // Count non-zero entries in column (incoming edges), excluding self-edge
    for (long row = 0; row < this->nrows; ++row) {
        if (row != block && this->matrix[row * this->ncols + block] != 0) {
            result++;
        }
    }
    return result;
}

std::vector<std::tuple<long, long, long>> DenseMatrix::entries() const {
    std::vector<std::tuple<long, long, long>> result;
    for (long row = 0; row < this->nrows; ++row) {
        for (long col = 0; col < this->ncols; ++col) {
            long value = this->matrix[row * this->ncols + col];
            if (value != 0) {
                result.emplace_back(row, col, value);
            }
        }
    }
    return result;
}

long DenseMatrix::get(long row, long col) const {
    check_row_bounds(row);
    check_col_bounds(col);
    return this->matrix[row * this->ncols + col];
}

std::vector<long> DenseMatrix::getcol(long col) const {
    check_col_bounds(col);
    std::vector<long> col_values(this->nrows);
    for (long row = 0; row < this->nrows; ++row) {
        col_values[row] = this->matrix[row * this->ncols + col];
    }
    return col_values;
}

MapVector<long> DenseMatrix::getcol_sparse(long col) const {
    check_col_bounds(col);
    MapVector<long> col_vector;
    for (long row = 0; row < this->nrows; ++row) {
        long value = this->matrix[row * this->ncols + col];
        if (value != 0) {
            col_vector[row] = value;
        }
    }
    return col_vector;
}

const MapVector<long>& DenseMatrix::getcol_sparseref(long col) const {
    check_col_bounds(col);
    MapVector<long> &slot = temp_sparse_vectors[temp_vector_idx % SPARSEREF_SLOTS];
    ++temp_vector_idx;
    slot.clear();
    for (long row = 0; row < this->nrows; ++row) {
        long value = this->matrix[row * this->ncols + col];
        if (value != 0) {
            slot[row] = value;
        }
    }
    return slot;
}

void DenseMatrix::getcol_sparse(long col, MapVector<long> &col_vector) const {
    check_col_bounds(col);
    col_vector.clear();
    for (long row = 0; row < this->nrows; ++row) {
        long value = this->matrix[row * this->ncols + col];
        if (value != 0) {
            col_vector[row] = value;
        }
    }
}

std::vector<long> DenseMatrix::getrow(long row) const {
    check_row_bounds(row);
    std::vector<long> row_values(this->ncols);
    for (long col = 0; col < this->ncols; ++col) {
        row_values[col] = this->matrix[row * this->ncols + col];
    }
    return row_values;
}

MapVector<long> DenseMatrix::getrow_sparse(long row) const {
    check_row_bounds(row);
    MapVector<long> row_vector;
    for (long col = 0; col < this->ncols; ++col) {
        long value = this->matrix[row * this->ncols + col];
        if (value != 0) {
            row_vector[col] = value;
        }
    }
    return row_vector;
}

void DenseMatrix::getrow_sparse(long row, MapVector<long> &row_vector) const {
    check_row_bounds(row);
    row_vector.clear();
    for (long col = 0; col < this->ncols; ++col) {
        long value = this->matrix[row * this->ncols + col];
        if (value != 0) {
            row_vector[col] = value;
        }
    }
}

const MapVector<long>& DenseMatrix::getrow_sparseref(long row) const {
    check_row_bounds(row);
    MapVector<long> &slot = temp_sparse_vectors[temp_vector_idx % SPARSEREF_SLOTS];
    ++temp_vector_idx;
    slot.clear();
    for (long col = 0; col < this->ncols; ++col) {
        long value = this->matrix[row * this->ncols + col];
        if (value != 0) {
            slot[col] = value;
        }
    }
    return slot;
}

EdgeWeights DenseMatrix::incoming_edges(long block) const {
    check_col_bounds(block);
    std::vector<long> indices;
    std::vector<long> values;
    for (long row = 0; row < this->nrows; ++row) {
        long value = this->matrix[row * this->ncols + block];
        if (value != 0) {
            indices.push_back(row);
            values.push_back(value);
        }
    }
    return EdgeWeights{indices, values};
}

std::set<long> DenseMatrix::neighbors(long block) const {
    check_row_bounds(block);
    check_col_bounds(block);
    std::set<long> result;
    // Outgoing edges
    for (long col = 0; col < this->ncols; ++col) {
        if (this->matrix[block * this->ncols + col] != 0) {
            result.insert(col);
        }
    }
    // Incoming edges
    for (long row = 0; row < this->nrows; ++row) {
        if (this->matrix[row * this->ncols + block] != 0) {
            result.insert(row);
        }
    }
    return result;
}

MapVector<long> DenseMatrix::neighbors_weights(long block) const {
    check_row_bounds(block);
    check_col_bounds(block);
    MapVector<long> result;
    // Outgoing edges
    for (long col = 0; col < this->ncols; ++col) {
        long value = this->matrix[block * this->ncols + col];
        if (value != 0) {
            result[col] += value;
        }
    }
    // Incoming edges
    for (long row = 0; row < this->nrows; ++row) {
        if (row != block) {
            long value = this->matrix[row * this->ncols + block];
            if (value != 0) {
                result[row] += value;
            }
        }
    }
    return result;
}

Indices DenseMatrix::nonzero() const {
    std::vector<long> row_vector;
    std::vector<long> col_vector;
    for (long row = 0; row < this->nrows; ++row) {
        for (long col = 0; col < this->ncols; ++col) {
            if (this->matrix[row * this->ncols + col] != 0) {
                row_vector.push_back(row);
                col_vector.push_back(col);
            }
        }
    }
    return Indices{row_vector, col_vector};
}

EdgeWeights DenseMatrix::outgoing_edges(long block) const {
    check_row_bounds(block);
    std::vector<long> indices;
    std::vector<long> values;
    for (long col = 0; col < this->ncols; ++col) {
        long value = this->matrix[block * this->ncols + col];
        if (value != 0) {
            indices.push_back(col);
            values.push_back(value);
        }
    }
    return EdgeWeights{indices, values};
}

void DenseMatrix::setrow(long row, const MapVector<long> &vector) {
    check_row_bounds(row);
    // Clear the row first
    for (long col = 0; col < this->ncols; ++col) {
        this->matrix[row * this->ncols + col] = 0;
    }
    // Set values from sparse vector
    for (const auto &entry : vector) {
        this->matrix[row * this->ncols + entry.first] = entry.second;
    }
}

void DenseMatrix::setcol(long col, const MapVector<long> &vector) {
    check_col_bounds(col);
    // Clear the column first
    for (long row = 0; row < this->nrows; ++row) {
        this->matrix[row * this->ncols + col] = 0;
    }
    // Set values from sparse vector
    for (const auto &entry : vector) {
        this->matrix[entry.first * this->ncols + col] = entry.second;
    }
}

void DenseMatrix::sub(long row, long col, long val) {
    check_row_bounds(row);
    check_col_bounds(col);
    this->matrix[row * this->ncols + col] -= val;
}

long DenseMatrix::edges() const {
    long total = 0;
    for (long row = 0; row < this->nrows; ++row) {
        for (long col = 0; col < this->ncols; ++col) {
            total += this->matrix[row * this->ncols + col];
        }
    }
    return total;
}

void DenseMatrix::print() const {
    for (long row = 0; row < this->nrows; ++row) {
        for (long col = 0; col < this->ncols; ++col) {
            std::cout << this->matrix[row * this->ncols + col] << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<long> DenseMatrix::sum(long axis) const {
    if (axis < 0 || axis > 1) {
        throw IndexOutOfBoundsException(axis, 2);
    }
    if (axis == 0) {  // sum across columns
        std::vector<long> totals(this->ncols, 0);
        for (long row = 0; row < this->nrows; ++row) {
            for (long col = 0; col < this->ncols; ++col) {
                totals[col] += this->matrix[row * this->ncols + col];
            }
        }
        return totals;
    } else {  // (axis == 1) sum across rows
        std::vector<long> totals(this->nrows, 0);
        for (long row = 0; row < this->nrows; ++row) {
            for (long col = 0; col < this->ncols; ++col) {
                totals[row] += this->matrix[row * this->ncols + col];
            }
        }
        return totals;
    }
}

long DenseMatrix::trace() const {
    long total = 0;
    for (long index = 0; index < this->nrows && index < this->ncols; ++index) {
        total += this->matrix[index * this->ncols + index];
    }
    return total;
}

void DenseMatrix::update_edge_counts(long current_block, long proposed_block, 
                                     std::vector<long> current_row,
                                     std::vector<long> proposed_row, 
                                     std::vector<long> current_col,
                                     std::vector<long> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    
    // Update rows
    for (long col = 0; col < this->ncols; ++col) {
        this->matrix[current_block * this->ncols + col] = current_row[col];
        this->matrix[proposed_block * this->ncols + col] = proposed_row[col];
    }
    
    // Update columns
    for (long row = 0; row < this->nrows; ++row) {
        this->matrix[row * this->ncols + current_block] = current_col[row];
        this->matrix[row * this->ncols + proposed_block] = proposed_col[row];
    }
}

void DenseMatrix::update_edge_counts(long current_block, long proposed_block, 
                                     MapVector<long> current_row,
                                     MapVector<long> proposed_row, 
                                     MapVector<long> current_col,
                                     MapVector<long> proposed_col) {
    check_row_bounds(current_block);
    check_col_bounds(current_block);
    check_row_bounds(proposed_block);
    check_col_bounds(proposed_block);
    
    // Clear and update current_block row
    for (long col = 0; col < this->ncols; ++col) {
        this->matrix[current_block * this->ncols + col] = 0;
    }
    for (const auto &entry : current_row) {
        this->matrix[current_block * this->ncols + entry.first] = entry.second;
    }
    
    // Clear and update proposed_block row
    for (long col = 0; col < this->ncols; ++col) {
        this->matrix[proposed_block * this->ncols + col] = 0;
    }
    for (const auto &entry : proposed_row) {
        this->matrix[proposed_block * this->ncols + entry.first] = entry.second;
    }
    
    // Update columns
    for (long row = 0; row < this->nrows; ++row) {
        this->matrix[row * this->ncols + current_block] = 0;
        this->matrix[row * this->ncols + proposed_block] = 0;
    }
    for (const auto &entry : current_col) {
        this->matrix[entry.first * this->ncols + current_block] = entry.second;
    }
    for (const auto &entry : proposed_col) {
        this->matrix[entry.first * this->ncols + proposed_block] = entry.second;
    }
}

void DenseMatrix::update_edge_counts(const Delta &delta) {
    for (const std::tuple<long, long, long> &entry : delta.entries()) {
        long row = std::get<0>(entry);
        long col = std::get<1>(entry);
        long change = std::get<2>(entry);
        this->matrix[row * this->ncols + col] += change;
    }
}

bool DenseMatrix::validate(long row, long col, long val) const {
    return this->get(row, col) == val;
}

std::vector<long> DenseMatrix::values() const {
    std::vector<long> values;
    for (long row = 0; row < this->nrows; ++row) {
        for (long col = 0; col < this->ncols; ++col) {
            long value = this->matrix[row * this->ncols + col];
            if (value != 0) {
                values.push_back(value);
            }
        }
    }
    return values;
}

