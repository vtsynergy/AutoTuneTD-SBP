/**
 * The block merge phase of the stochastic block blockmodeling module.
 */
#ifndef CPPSBP_BLOCK_MERGE_HPP
#define CPPSBP_BLOCK_MERGE_HPP

#include <limits>
#include <numeric>
#include <random>

// #include <omp.h>
#include "common.hpp"
#include "blockmodel/blockmodel.hpp"
// #include "matrix/boost_mapped_matrix.hpp"
#include "matrix/csparse_matrix.hpp"
#include "typedefs.hpp"
#include "graph.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

namespace block_merge {

static const long NUM_AGG_PROPOSALS_PER_BLOCK = 10;  // Proposals per block

typedef struct proposal_evaluation_t {
    long proposed_block;
    double delta_entropy;
} ProposalEvaluation;

/// Performs the block merges with the highest change in entropy/MDL, recalculating change in entropy before each
/// merge to account for dependencies between merges. This function modified the blockmodel.
void carry_out_best_merges_advanced(Blockmodel &blockmodel, const std::vector<double> &delta_entropy_for_each_block,
                                    const std::vector<long> &best_merge_for_each_block, const Graph &graph);

/// Returns the potential changes to the blockmodel if `current_block` was merged into `proposed_block`.
Delta blockmodel_delta(long current_block, long proposed_block, const Blockmodel &blockmodel);

/// Computes the new edge counts for the affected blocks (communities) under a proposed block merge.
EdgeCountUpdates edge_count_updates(std::shared_ptr<ISparseMatrix> blockmodel, long current_block, long proposed_block,
                                    EdgeWeights &out_blocks, EdgeWeights &in_blocks);

/// Fills the new edge counts for the affected blocks (communities) under a proposed block merge.
/// Results are stored as sparse vectors (unordered_maps)
void edge_count_updates_sparse(ISparseMatrix *blockmodel, long current_block, long proposed_block,
                               EdgeWeights &out_blocks, EdgeWeights &in_blocks, SparseEdgeCountUpdates &updates);

/// Merges entire blocks (communities) in blockmodel together
Blockmodel &merge_blocks(Blockmodel &blockmodel, const Graph &graph, long num_edges);

/// Proposes a merge for current_block based on the current blockmodel state
ProposalEvaluation propose_merge(long current_block, long num_edges, Blockmodel &blockmodel,
                                 std::vector<long> &block_assignment);

/// Proposes a merge for current_block based on the current blockmodel state, using sparse PartialProfile structures
ProposalEvaluation propose_merge_sparse(long current_block, const Blockmodel &blockmodel, const Graph &graph,
                                        std::unordered_map<long, bool> &past_proposals);

} // namespace block_merge

#endif // CPPSBP_BLOCK_MERGE_HPP
