/**
 * The distributed divisive stochastic block blockmodeling module.
 */
#ifndef SBP_DIST_DIVISIVE_SBP_HPP
#define SBP_DIST_DIVISIVE_SBP_HPP

#include "blockmodel.hpp"
#include "distributed/dist_blockmodel_triplet.hpp"
#include "distributed/two_hop_blockmodel.hpp"
#include "top_down_sbp.hpp"

#include <vector>

namespace divisive::dist {

/// Applies the best cluster splits.
void apply_best_splits(const Blockmodel &blockmodel, const std::vector<double> &split_entropy,
                       const std::vector<long> &comm_assignment, int target_num_communities);

/// Use agglomerative SBP to continue execution.
/// TODO: add a way to start SBP from a given blockmodel or triplet
TwoHopBlockmodel continue_agglomerative(Graph &graph, DistTopDownBlockmodelTriplet &blockmodel_triplet,
                                        float iteration);

/// Returns true if end condition has not been reached. If args.mix is True, then the end condition is reaching the
/// golden ratio. Otherwise, the end condition is identifying the optimal blockmodel.
bool end_condition_not_reached(TwoHopBlockmodel &blockmodel, DistTopDownBlockmodelTriplet &triplet);

/// Communicates the best split information across all nodes using MPI_Allreduce calls
void mpi_get_best_splits(std::vector<double> &delta_entropy_for_each_block, std::vector<long> &comm_assignment);

/// Runs the divisive SBP algorithm using MPI.
Blockmodel run(Graph &graph);

/// The reverse of block_merge::merge_blocks. Proposes several cluster splits, and applies the best ones until the
/// number of communities reaches `target_num_communities`.
TwoHopBlockmodel split_communities(TwoHopBlockmodel &blockmodel, const Graph &graph, int target_num_communities);

}

#endif  // SBP_DIST_DIVISIVE_SBP_HPP
