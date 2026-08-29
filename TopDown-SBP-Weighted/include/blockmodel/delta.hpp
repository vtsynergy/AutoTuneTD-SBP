/***
 * Stores the incremental changes to the blockmodel adjacency matrix when a vertex
 * moves from current_block to proposed_block, or when two blocks are merged.
 *
 * Two backing representations are available, selected at construction time via the
 * `coo` parameter (set from args.coodelta by the factory functions):
 *
 *   Map mode (default, coo=false):
 *     Four tsl::robin_map slices, one per affected row/column slice. Fast O(1)
 *     amortized add/get; iteration order is unspecified.
 *
 *   COO mode (coo=true):
 *     Three parallel std::vector<long> arrays (_coo_rows, _coo_cols, _coo_vals)
 *     storing (row, col, val) triplets. Entries are appended during add()/sub()
 *     and sorted+coalesced lazily in finalize() on the first const read (get/entries).
 *     Arrays expose .data() for use with `#pragma omp target map`.
 *
 * All public methods (add, sub, get, entries, self_edge_weight, current_block,
 * proposed_block) behave identically in both modes; callers do not need to know
 * which representation is active.
 *
 * TODO (future GPU perf): explore storing Delta as 
 * four sparse 1-D vectors (current/proposed x row/col); represent each line
 * as a sorted (index, change) array and merge-walk it against the corresponding
 * sorted matrix row/col inside the entropy pass. The merge position classifies each
 * cell (changed / unchanged-but-degree-shifted / new), eliminating get(), entries(),
 * and the separate coalesce phase entirely. This requires getrow_sparse/getcol_sparse
 * to return sorted flat arrays on the GPU and a rewrite of the entropy consumers.
 * See cursor_files/CHANGELOG.md (2026-06-10 "fold Delta into the matrix-row pass").
 */
#ifndef SBP_BLOCKMODEL_DELTA_HPP
#define SBP_BLOCKMODEL_DELTA_HPP

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

#include "typedefs.hpp"
#include "utils.hpp"

class Delta {
private:
    // --- Map-mode storage (active when _coo == false) ---
    MapVector<long> _current_block_row;
    MapVector<long> _proposed_block_row;
    MapVector<long> _current_block_col;
    MapVector<long> _proposed_block_col;

    // --- COO-mode storage (active when _coo == true) ---
    // Three parallel arrays forming unsorted (row, col, val) triplets during
    // construction; sorted row-major and coalesced by finalize().
    // TODO: figure out how to get around this whole mutable headache.
    mutable std::vector<long> _coo_rows;
    mutable std::vector<long> _coo_cols;
    mutable std::vector<long> _coo_vals;
    // Guards re-running finalize(): set to false on every mutation, true after sort+coalesce.
    mutable bool _finalized;

    // --- Shared ---
    long _current_block;
    long _proposed_block;
    long _self_edge_weight;
    bool _coo;

    /// Sorts _coo_rows/_coo_cols/_coo_vals by (row, col) row-major and coalesces
    /// duplicate (row,col) pairs by summing their values. Entries whose coalesced
    /// value is zero are removed. Sets _finalized = true.
    /// Called lazily by get() and entries() the first time after any mutation.
    void finalize() const {
        if (_finalized) return;
        if (this->nnz() == 0) {
            _finalized = true;
            return;
        }
        std::vector<long> indices = utils::range<long>(0, this->nnz());
        // TODO: explore changing this to a radix sort. Have to think through whether or not this makes sense
        // on GPU.
        /// Sort the entries based on row,col
        std::stable_sort(indices.begin(), indices.end(), [this](long row, long col) {
            // TODO: branches inside GPU code are expensive. Replace with a single comparison.
            if (_coo_rows[row] != _coo_rows[col]) return _coo_rows[row] < _coo_rows[col];
            return _coo_cols[row] < _coo_cols[col];
        });
        for (long i = 0; i < this->nnz(); ++i) {
            // we're trying to place indices[i] in the correct place, AND place indices[]
            if (indices[i] < 0) continue;  // index has already been visited
            long unsorted_idx = i;
            while (indices[unsorted_idx] != i) {
                long swap_idx = indices[unsorted_idx]; // what are we swapping unsorted_idx with?
                std::swap(_coo_rows[unsorted_idx], _coo_rows[swap_idx]);
                std::swap(_coo_cols[unsorted_idx], _coo_cols[swap_idx]);
                std::swap(_coo_vals[unsorted_idx], _coo_vals[swap_idx]);
                indices[unsorted_idx] = -1; // 
                unsorted_idx = swap_idx; // now we sort swap_idx
            }
            indices[unsorted_idx] = -1; // all entries in cycle are sorted now
        }
        /// Coalesce the internal vectors
        long coalesced_index = 0;
        long row = this->_coo_rows[0];
        long col = this->_coo_cols[0];
        for (long i = 1; i < this->nnz(); ++i) {
            if (this->_coo_rows[i] == row && this->_coo_cols[i] == col) { // need to coalesce
                this->_coo_vals[coalesced_index] += this->_coo_vals[i];
            } else { // need to restart row & col counter
                row = this->_coo_rows[i];
                col = this->_coo_cols[i];
                long val = this->_coo_vals[i];
                coalesced_index++;
                this->_coo_rows[coalesced_index] = row;
                this->_coo_cols[coalesced_index] = col;
                this->_coo_vals[coalesced_index] = val;
            }
        }
        this->_coo_rows.resize(coalesced_index + 1);
        this->_coo_cols.resize(coalesced_index + 1);
        this->_coo_vals.resize(coalesced_index + 1);
        _finalized = true;
    }

    /// An iterator over the COO-mode Delta structure. Used for searching the sorted entries
    /// by (row,col) key.
    struct COOIter {
        const long* rows;
        const long* cols;
        long i;
        using difference_type   = long;
        using value_type        = std::pair<long, long>;
        using reference         = std::pair<long, long>;  // returned by value — no true reference
        using pointer           = void;
        using iterator_category = std::random_access_iterator_tag;
        std::pair<long,long> operator*()        const { return {rows[i], cols[i]}; }
        COOIter& operator++()                         { ++i; return *this; }
        COOIter& operator--()                         { --i; return *this; }
        COOIter& operator+=(long n)                   { i += n; return *this; }
        COOIter& operator-=(long n)                   { i -= n; return *this; }
        COOIter  operator+(long n)              const { return {rows, cols, i + n}; }
        COOIter  operator-(long n)              const { return {rows, cols, i - n}; }
        long     operator-(const COOIter& o)    const { return i - o.i; }
        bool     operator< (const COOIter& o)   const { return i < o.i; }
        bool     operator==(const COOIter& o)   const { return i == o.i; }
        bool     operator!=(const COOIter& o)   const { return i != o.i; }
    };

public:
    Delta() : _finalized(false), _current_block(-1), _proposed_block(-1),
              _self_edge_weight(0), _coo(false) {}

    /// Primary constructor. `buckets` is a size hint for the map-mode hash maps (ignored
    /// in COO mode). `coo` selects the storage representation; pass args.coodelta from
    /// the factory functions.
    Delta(long current_block, long proposed_block, long buckets = 10, bool coo = false)
        : _finalized(false), _current_block(current_block), _proposed_block(proposed_block),
          _self_edge_weight(0), _coo(coo) {
        if (!_coo) {
            _current_block_row  = MapVector<long>(buckets);
            _proposed_block_row = MapVector<long>(buckets);
            _current_block_col  = MapVector<long>(buckets);
            _proposed_block_col = MapVector<long>(buckets);
        } else {
            _coo_rows.reserve(buckets * 4);
            _coo_cols.reserve(buckets * 4);
            _coo_vals.reserve(buckets * 4);
        }
    }

    /// Pre-seeded constructor (map mode only). Initialises the map entries with zeros
    /// for every key present in the four reference rows/cols so that subsequent get()
    /// calls on cells that happen to be zero still return 0 rather than throwing.
    /// In COO mode the pre-seeding is skipped; add()/sub() start appending from empty.
    Delta(long current_block, long proposed_block,
          const MapVector<long> &block_row, const MapVector<long> &block_col,
          const MapVector<long> &proposed_row, const MapVector<long> &proposed_col,
          bool coo = false)
        : _finalized(false), _current_block(current_block), _proposed_block(proposed_block),
          _self_edge_weight(0), _coo(coo) {
        if (!_coo) {
            zero_init(block_row, block_col, proposed_row, proposed_col);
        }
        // COO mode: no pre-seeding needed; entries are built purely by add()/sub().
    }

    /// Adds `value` as the delta to cell matrix[`row`,`col`].
    void add(long row, long col, long value) {
        if (_coo) {
            _coo_rows.push_back(row);
            _coo_cols.push_back(col);
            _coo_vals.push_back(value);
            _finalized = false;
        } else {
            if (row == _current_block)
                _current_block_row[col] += value;
            else if (row == _proposed_block)
                _proposed_block_row[col] += value;
            else if (col == _current_block)
                _current_block_col[row] += value;
            else if (col == _proposed_block)
                _proposed_block_col[row] += value;
            else
                throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
        }
    }

    /// Adds -`value` (negative `value`) as the delta to cell matrix[`row`,`col`].
    void sub(long row, long col, long value) {
        if (_coo) {
            _coo_rows.push_back(row);
            _coo_cols.push_back(col);
            _coo_vals.push_back(-value);
            _finalized = false;
        } else {
            if (row == _current_block)
                _current_block_row[col] -= value;
            else if (row == _proposed_block)
                _proposed_block_row[col] -= value;
            else if (col == _current_block)
                _current_block_col[row] -= value;
            else if (col == _proposed_block)
                _proposed_block_col[row] -= value;
            else
                throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
        }
    }

    /// Returns the delta for matrix[`row`,`col`] without modifying the underlying data structure.
    long get(long row, long col) const {
        if (_coo) {
            finalize();
            auto target = std::make_pair(row, col);
            COOIter begin{_coo_rows.data(), _coo_cols.data(), 0};
            COOIter end{_coo_rows.data(), _coo_cols.data(), this->nnz()};
            COOIter result = std::lower_bound(begin, end, target);
            if (result != end && *result == target)
                return _coo_vals[result.i];
            return 0;  // cell not in delta → net change is zero
        }
        if (row == _current_block)
            return map_vector::get(_current_block_row, col);
        else if (row == _proposed_block)
            return map_vector::get(_proposed_block_row, col);
        else if (col == _current_block)
            return map_vector::get(_current_block_col, row);
        else if (col == _proposed_block)
            return map_vector::get(_proposed_block_col, row);
        throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
    }

    /// Returns all stored deltas as a list of tuples (row, col, delta).
    /// In COO mode the list is sorted row-major. In map mode the order is unspecified.
    std::vector<std::tuple<long, long, long>> entries() const {
        std::vector<std::tuple<long, long, long>> result;
        if (_coo) {
            finalize();
            result.reserve(_coo_vals.size());
            for (long i = 0; i < this->nnz(); ++i) {
                result.emplace_back(_coo_rows[i], _coo_cols[i], _coo_vals[i]);
            }
            return result;
        }
        for (const LongEntry &entry : _current_block_row)
            result.emplace_back(_current_block, entry.first, entry.second);
        for (const LongEntry &entry : _proposed_block_row)
            result.emplace_back(_proposed_block, entry.first, entry.second);
        for (const LongEntry &entry : _current_block_col)
            result.emplace_back(entry.first, _current_block, entry.second);
        for (const LongEntry &entry : _proposed_block_col)
            result.emplace_back(entry.first, _proposed_block, entry.second);
        return result;
    }

    /// Returns the weight of the self edge for this move, if any.
    long self_edge_weight() const { return _self_edge_weight; }
    /// Sets the weight of the self edge for this move, if any.
    void self_edge_weight(long weight) { _self_edge_weight = weight; }

    long current_block()  const { return _current_block; }
    long proposed_block() const { return _proposed_block; }
    /// Returns true if this Delta is in COO mode, false if in map mode.
    bool is_coo() const { return _coo; }

    /// Initiates map-mode deltas with 0s for all non-zero elements currently present
    /// in `block_row`, `block_col`, `proposed_row`, and `proposed_col`.
    /// Map mode only; no-op (and not called) in COO mode.
    void zero_init(const MapVector<long> &block_row, const MapVector<long> &block_col,
                   const MapVector<long> &proposed_row, const MapVector<long> &proposed_col) {
        _current_block_row = MapVector<long>(block_row.bucket_count());
        for (const LongEntry &entry : block_row)
            add(_current_block, entry.first, 0);

        _current_block_col = MapVector<long>(block_col.bucket_count());
        for (const LongEntry &entry : block_col)
            add(entry.first, _current_block, 0);

        _proposed_block_row = MapVector<long>(proposed_row.bucket_count());
        for (const LongEntry &entry : proposed_row)
            add(_proposed_block, entry.first, 0);

        _proposed_block_col = MapVector<long>(proposed_col.bucket_count());
        for (const LongEntry &entry : proposed_col)
            add(entry.first, _proposed_block, 0);
    }

    // -------------------------------------------------------------------------
    // GPU accessors (COO mode only)
    // These expose the finalized contiguous arrays for `#pragma omp target map`.
    // Call get() or entries() first to trigger finalize(), then use these pointers.
    // -------------------------------------------------------------------------

    /// Number of non-zero entries in the finalized COO delta. COO mode only.
    long nnz() const { return static_cast<long>(_coo_vals.size()); }
    /// Raw pointer to the row-index array (for omp target map). COO mode only.
    const long* rows_data() const { return _coo_rows.data(); }
    /// Raw pointer to the col-index array (for omp target map). COO mode only.
    const long* cols_data() const { return _coo_cols.data(); }
    /// Raw pointer to the values array (for omp target map). COO mode only.
    const long* vals_data() const { return _coo_vals.data(); }
};

#endif // SBP_BLOCKMODEL_DELTA_HPP
