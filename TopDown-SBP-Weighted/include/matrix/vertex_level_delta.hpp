//
// Created by Frank on 9/9/2022.
//

#ifndef SBP_POlongER_DELTA_HPP
#define SBP_POlongER_DELTA_HPP

#include "typedefs.hpp"

class VertexLevelDelta {
private:
//    MapVector<long> _current_block_row;
//    MapVector<long> _proposed_block_row;
//    MapVector<long> _current_block_col;
//    MapVector<long> _proposed_block_col;
    MapVector<long> _outgoing_edges;
    MapVector<long> _incoming_edges;
    long _current_vertex;
    long _current_block;
    long _proposed_block;
    long _self_edge_weight;
public:
    VertexLevelDelta() {
        this->_current_vertex = -1;
        this->_current_block = -1;
        this->_proposed_block = -1;
        this->_self_edge_weight = 0;
    }
    VertexLevelDelta(long current_vertex, long current_block, long proposed_block, long buckets = 10) {
        this->_current_vertex = current_vertex;
        this->_current_block = current_block;
        this->_proposed_block = proposed_block;
        this->_self_edge_weight = 0;
        this->_outgoing_edges = MapVector<long>(buckets);
        this->_incoming_edges = MapVector<long>(buckets);
    }
    /// Adds `value` as the delta to edge `from` -> `to`.
    void add(long from, long to, long value) {
        if (from == this->_current_vertex)
            this->_outgoing_edges[to] += value;
        else if (to == this->_current_vertex)
            this->_incoming_edges[from] += value;
        else
            throw std::logic_error("Neither the from nor the to vertex is current_vertex.");
    }

    /// Returns all stores deltas as a list of tuples storing `from`, `to`, `delta`.
    [[nodiscard]] std::vector<std::tuple<long, long, long>> entries() const {
        std::vector<std::tuple<long, long, long>> result;
        for (const std::pair<long, long> &entry : this->_outgoing_edges) {
            result.emplace_back(this->_current_vertex, entry.first, entry.second);
        }
        for (const std::pair<long, long> &entry : this->_incoming_edges) {
            result.emplace_back(entry.first, this->_current_vertex, entry.second);
        }
        return result;
    }

    /// Returns the delta for the edge `from`, `to` without modifying the underlying data structure.
    [[nodiscard]] long get(long from, long to) const {
        if (from == this->_current_vertex)
            return map_vector::get(this->_outgoing_edges, to);
        else if (to == this->_current_vertex)
            return map_vector::get(this->_incoming_edges, from);
        throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
    }

    /// Returns the weight of the self edge for this move, if any.
    [[nodiscard]] long self_edge_weight() const {
        return this->_self_edge_weight;
    }

    /// Sets the weight of the self edge for this move, if any.
    void self_edge_weight(long weight) {
        this->_self_edge_weight = weight;
    }

//    /// Adds -`value` (negative `value`) as the delta to edge `from` -> `to`.
//    void sub(long row, long col, long value) {
//        if (row == this->_current_block)
//            this->_current_block_row[col] -= value;
//        else if (row == this->_proposed_block)
//            this->_proposed_block_row[col] -= value;
//        else if (col == this->_current_block)
//            this->_current_block_col[row] -= value;
//        else if (col == this->_proposed_block)
//            this->_proposed_block_col[row] -= value;
//        else
//            throw std::logic_error("Neither the row nor column are current_block or proposed_block.");
//    }
};

#endif //SBP_POlongER_DELTA_HPP
