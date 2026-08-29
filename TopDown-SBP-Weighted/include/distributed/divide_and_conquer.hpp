//
// Created by Frank on 3/23/2023.
//

#ifndef SBP_DIVIDE_AND_CONQUER_HPP
#define SBP_DIVIDE_AND_CONQUER_HPP
#include <vector>

#include "blockmodel.hpp"
#include "evaluate.hpp"
#include "graph.hpp"
#include "sample.hpp"

const int NUM_VERTICES_TAG = 0;
const int VERTICES_TAG = 1;
const int BLOCKS_TAG = 2;

extern double sample_time;
extern double sample_extend_time;
extern double finetune_time;

namespace dnc {

std::vector<long> combine_partitions(const Graph &graph, long &offset, std::vector<std::vector<long>> &vertex_lists,
                                     std::vector<std::vector<long>> &assignment_lists);

std::vector<long> combine_two_blockmodels(const std::vector<long> &combined_vertices,
                                          const std::vector<long> &assignment_a,
                                          const std::vector<long> &assignment_b, const Graph &original_graph);

void evaluate_partition(const Graph &graph, Blockmodel &blockmodel, double runtime);

Blockmodel finetune_partition(Blockmodel &blockmodel, const Graph &graph);

Blockmodel merge_blocks(const Blockmodel &blockmodel, const sample::Sample &subgraph, long my_num_blocks,
                        long combined_num_blocks);

void receive_partition(int src, std::vector<std::vector<long>> &src_vertices,
                       std::vector<std::vector<long>> &src_assignments);

void translate_local_partition(std::vector<long> &local_vertices, std::vector<long> &local_assignment,
                               const sample::Sample &subgraph, long num_vertices,
                               const std::vector<long> &partition_assignment);

//void write_results(const Graph &graph, const evaluate::Eval &eval, double runtime);

}

#endif //SBP_DIVIDE_AND_CONQUER_HPP
