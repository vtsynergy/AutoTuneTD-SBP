/**
 * The town-down alternative approach to stochastic block blockmodeling.
 */
#ifndef SBP_TOP_DOWN_HPP
#define SBP_TOP_DOWN_HPP

#include <memory>
#include <vector>

#include "blockmodel/blockmodel.hpp"
#include "blockmodel/blockmodel_triplet.hpp"
#include "graph.hpp"

namespace divisive {

/// Stores information about a cluster split.
struct Split {
    /// The blockmodel for the two split communities.
    std::shared_ptr<Blockmodel> blockmodel;
    /// The number of vertices involved in the split.
    long num_vertices = std::numeric_limits<long>::max();
    /// The number of edges involved in the split.
    long num_edges;
    /// Translates full graph vertex IDs to subgraph vertex IDs
    MapVector<long> translator;
};

static const long NUM_AGG_PROPOSALS_PER_BLOCK = 10;  // Proposals per block

/// Accepts or rejects a blockmodel split.
bool accept(const Split &split, const Blockmodel &blockmodel);

/// Applies the best cluster splits.
void apply_best_splits(const Blockmodel &blockmodel, const std::vector<Split> &best_splits,
                       const std::vector<double> &split_entropy, int target_num_communities);

/// Applies the cluster split to `blockmodel`.
//void apply_split(const Split &split, Blockmodel &blockmodel);

/// Returns true if end condition has not been reached. If args.mix is True, then the end condition is reaching the
/// golden ratio. Otherwise, the end condition is idenitfying the optimal blockmodel.
bool end_condition_not_reached(Blockmodel &blockmodel, TopDownBlockmodelTriplet &triplet);

/// Extracts a subgraph from `graph` by selecting vertices that belong to cluster `cluster`. `subgraph` and
/// `translator` are modified by this method, and serve as the two "return values".
void extract_subgraph(const Graph &graph, const Blockmodel &blockmodel, Graph &subgraph, MapVector<long> &translator,
                      int cluster);

/// Splits a single cluster into two. Returns a blockmodel containing just 2 communities that resulted from
/// splitting cluster `cluster`.
Split propose_split(long cluster, const Graph &graph, const Blockmodel &blockmodel);

/// Splits a single cluster into two. Returns a blockmodel containing just 2 communities that resulted from
/// splitting cluster `cluster`. This preferred version of the function uses a pre-made subgraph.
Split propose_split(const Graph &subgraph, const MapVector<long> &translator);

/// Proposes a split by randomly partitioning vertices across two blocks.
std::vector<long> propose_random_split(const Graph &subgraph);

/// Proposes a split using two snowball samples, where at each step the vertex with the highest connectivity is
/// selected.
std::vector<long> propose_connectivity_snowball_split(const Graph &subgraph);

/// Proposes a split using two unweighted snowball samples.
std::vector<long> propose_snowball_split(const Graph &subgraph);

/// Proposes a split using one unweighted snowball sample.
std::vector<long> propose_single_snowball_split(const Graph &subgraph);

/// Runs the top-down cluster detection algorithm.
Blockmodel run(const Graph &graph);

/// Runs the top-down cluster detection algorithm until golden ratio is reached, then reverts to block merges.
//Blockmodel run_mix(const Graph &graph);

/// The reverse of block_merge::merge_blocks. Proposes several cluster splits, and applies the best ones until the
/// number of communities reaches `target_num_communities`.
Blockmodel split_communities(Blockmodel &blockmodel, const Graph &graph, int target_num_communities);

/// Selects the two vertices that initialize the split
std::pair<long, long> split_init(const Graph &subgraph, const std::vector<long> &vertex_degrees);

/// Selects two vertices uniformly at random to initialize the split.
std::pair<long, long> split_init_random(const Graph &subgraph);

/// Selects two vertices to initialize the split randomly, with weight = vertex degree.
std::pair<long, long> split_init_degree_weighted(const Graph &subgraph, const std::vector<long> &vertex_degrees);

/// Selects the two vertices with the highest degree to initialize the split.
std::pair<long, long> split_init_high_degree(const Graph &subgraph, const std::vector<long> &vertex_degrees);

}

#endif // SBP_TOP_DOWN_HPP