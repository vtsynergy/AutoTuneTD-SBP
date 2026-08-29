//
// The town-down alternative approach to stochastic block blockmodeling.
// Created by wanye on 5/11/2022.
//
#include "top_down_sbp.hpp"

#include <iostream>
#include <limits>
#include <memory>
#include <mpi.h>
#include <random>
#include <vector>
#include <utility>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "blockmodel/blockmodel_triplet.hpp"
#include "block_merge.hpp"
#include "common.hpp"
#include "entropy.hpp"
#include "finetune.hpp"
#include "globals.hpp"
#include "graph.hpp"
#include "sbp.hpp"
#include "rng.hpp"
#include "utils.hpp"

namespace divisive {

void apply_best_splits(Blockmodel &blockmodel, const std::vector<Split> &best_splits,
                       const std::vector<double> &split_entropy, int target_num_communities) {
    // Sort entropies in ascending order
    std::vector<long> sorted_splits = utils::argsort<double>(split_entropy);
//    std::cout << "the argsort result = " << std::endl;
//    utils::print<long>(sorted_splits);
    // Modify assignment, increasing blockmodel.blockNum() until reaching target
    long num_blocks = blockmodel.num_blocks();
//    long index_of_split = 0;
//    while (_num_blocks < target_num_communities) {
    for (int index_of_split = best_splits.size() - 1; index_of_split > -1; --index_of_split) {  // < best_splits.size(); ++index_of_split) {
        long block = sorted_splits[index_of_split];
//        std::cout << "block = " << block << std::endl;
        const Split& split = best_splits[block];
//        std::cout << "split V = " << split.num_vertices << " split subgraph V = " << split.extract_subgraph.num_vertices() << std::endl;
//        if (split.extract_subgraph.num_vertices() == 1) continue;  // do not try to split block with only one vertex
        // TODO: fix: split.subgrpah.num_vertices() doesn't match split.num_vertices. May be as simple as initializing extract_subgraph to an empty graph?
//        std::cout << "Looking at split with index = " << index_of_split << " and num vertices " << split.num_vertices << " and entropy = " << split_entropy[index_of_split] << std::endl;
        if (split.num_vertices > (long) blockmodel.block_assignment().size()) continue;
//        std::cout << "Applying split with index = " << index_of_split << " and num vertices " << split.num_vertices << " and entropy = " << split_entropy[index_of_split] << std::endl;
        MapVector<long> reverse_translator;
        for (const auto &entry : split.translator) {
            reverse_translator[entry.second] = entry.first;
        }
        for (long index = 0; index < split.num_vertices; ++index) {
            long assignment = split.blockmodel->block_assignment(index);
            if (assignment == 1) {
                long vertex = reverse_translator[index];
                blockmodel.set_block_assignment(vertex, num_blocks);
            }
        }
        num_blocks++;
        if (num_blocks >= target_num_communities) break;
    }
    // Re-form blockmodel
    blockmodel.num_blocks(num_blocks);
}

Blockmodel continue_agglomerative(const Graph &graph, TopDownBlockmodelTriplet &blockmodel_triplet, float iteration) {
    double start_t = MPI_Wtime();
    Blockmodel blockmodel;
    BlockmodelTriplet gr_blockmodel_triplet = BlockmodelTriplet();
    blockmodel = gr_blockmodel_triplet.get_next_blockmodel(blockmodel_triplet.get(0));
    blockmodel = gr_blockmodel_triplet.get_next_blockmodel(blockmodel_triplet.get(1));
    blockmodel = gr_blockmodel_triplet.get_next_blockmodel(blockmodel_triplet.get(2));
    while (!sbp::done_blockmodeling(blockmodel, gr_blockmodel_triplet)) {
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Merging blocks down from " << blockmodel.num_blocks() << " to "
                      << blockmodel.num_blocks() - blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        double start_bm = MPI_Wtime();
        blockmodel = block_merge::merge_blocks(blockmodel, graph, graph.num_edges());
        timers::BlockMerge_time += MPI_Wtime() - start_bm;
        std::cout << "Starting MCMC vertex moves" << std::endl;
        double start_mcmc = MPI_Wtime();
        blockmodel = finetune::mcmc(iteration, graph, blockmodel, blockmodel_triplet);
        timers::MCMC_time += MPI_Wtime() - start_mcmc;
        double mdl = blockmodel.getOverall_entropy();
        long num_blocks = blockmodel.num_blocks();
        blockmodel = gr_blockmodel_triplet.get_next_blockmodel(blockmodel);
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.num_blocks() - 2);
        timers::total_time += MPI_Wtime() - start_t;
        start_t = MPI_Wtime();
        utils::save_partial_profile(++iteration, -1, mdl, entropy::normalize_mdl_v1(mdl, graph), num_blocks);
    }
    return blockmodel;
}

void extract_subgraph(const Graph &graph, const Blockmodel &blockmodel, Graph &subgraph, MapVector<long> &translator,
                      int cluster) {
//    Split split;
    std::vector<bool> cluster_flag = utils::constant<bool>(graph.num_vertices(), false);
    std::vector<long> cluster_vertices;
    long index = 0;
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        if (blockmodel.block_assignment(vertex) != cluster) continue;
        cluster_flag[vertex] = true;
        cluster_vertices.push_back(vertex);
//        split.translator[vertex] = index;
        translator[vertex] = index;
        index++;
    }
//    split.num_vertices = long(cluster_vertices.size());
    long num_vertices = long(cluster_vertices.size());
    std::vector<long> split_assignment;
//    Graph extract_subgraph(split.num_vertices, graph.num_edges() / graph.num_vertices());
    subgraph = Graph(num_vertices);  // , graph.num_edges() / graph.num_vertices());
    if (subgraph.num_vertices() < 2) {
        return;
//        split.subgraph = extract_subgraph;
//        split_assignment = utils::constant<long>(1, 0);
//        split.blockmodel = std::make_shared<Blockmodel>(1, extract_subgraph, 0.5, split_assignment);
//        return split;
    }
//    std::cout << "subgraph numvertices: " << extract_subgraph.num_vertices() << std::endl;
    for (long vertex: cluster_vertices) {
        for (long neighbor: graph.out_neighbors(vertex)) {
            if (!cluster_flag[neighbor]) continue;
//            extract_subgraph.add_edge(split.translator[vertex], split.translator[neighbor]);
            subgraph.add_edge(translator[vertex], translator[neighbor]);
        }
    }
    subgraph.build_csr();
//    return split;
}

Split propose_split(long cluster, const Graph &graph, const Blockmodel &blockmodel) {
    Split split;
    std::vector<bool> cluster_flag = utils::constant<bool>(graph.num_vertices(), false);
    std::vector<long> cluster_vertices;
    long index = 0;
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        if (blockmodel.block_assignment(vertex) != cluster) continue;
        cluster_flag[vertex] = true;
        cluster_vertices.push_back(vertex);
        split.translator[vertex] = index;
        index++;
    }
    split.num_vertices = long(cluster_vertices.size());
    std::vector<long> split_assignment;
    Graph subgraph(split.num_vertices, graph.num_edges() / graph.num_vertices());
    if (subgraph.num_vertices() < 2) {
//        split.subgraph = subgraph;
        split_assignment = utils::constant<long>(1, 0);
        split.blockmodel = std::make_shared<Blockmodel>(1, subgraph, 0.5, split_assignment);
        return split;
    }
//    std::cout << "subgraph numvertices: " << extract_subgraph.num_vertices() << std::endl;
    for (long vertex: cluster_vertices) {
        for (long neighbor: graph.out_neighbors(vertex)) {
            if (!cluster_flag[neighbor]) continue;
            subgraph.add_edge(split.translator[vertex], split.translator[neighbor]);
        }
    }
    subgraph.build_csr();
    if (args.split == "random")
        split_assignment = propose_random_split(subgraph);
    else if (args.split == "connectivity-snowball")
        split_assignment = propose_connectivity_snowball_split(subgraph);
    else if (args.split == "snowball")
        split_assignment = propose_snowball_split(subgraph);
    else if (args.split == "single-snowball")
        split_assignment = propose_single_snowball_split(subgraph);
    else {
        std::cerr << "Invalid split type provided." << std::endl;
        exit(-2);
    }
    split.blockmodel = std::make_shared<Blockmodel>(2, subgraph, 0.5, split_assignment);
    split.num_edges = subgraph.num_edges();
//    split.subgraph = subgraph;
    return split;
}

Split propose_split(const Graph &subgraph, const MapVector<long> &translator) {
    Split split;
    split.translator = translator;
//    std::vector<bool> cluster_flag = utils::constant<bool>(graph.num_vertices(), false);
//    std::vector<long> cluster_vertices;
//    long index = 0;
//    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
//        if (blockmodel.block_assignment(vertex) != cluster) continue;
//        cluster_flag[vertex] = true;
//        cluster_vertices.push_back(vertex);
//        split.translator[vertex] = index;
//        index++;
//    }
    split.num_vertices = subgraph.num_vertices();
    std::vector<long> split_assignment;
//    Graph extract_subgraph(split.num_vertices, graph.num_edges() / graph.num_vertices());
    if (subgraph.num_vertices() < 2) {
//        split.subgraph = extract_subgraph;
        split_assignment = utils::constant<long>(1, 0);
        split.blockmodel = std::make_shared<Blockmodel>(1, subgraph, 0.5, split_assignment);
        return split;
    }
//    std::cout << "extract_subgraph numvertices: " << subgraph.num_vertices() << std::endl;
//    for (long vertex: cluster_vertices) {
//        for (long neighbor: graph.out_neighbors(vertex)) {
//            if (!cluster_flag[neighbor]) continue;
//            extract_subgraph.add_edge(split.translator[vertex], split.translator[neighbor]);
//        }
//    }
    if (args.split == "random")
        split_assignment = propose_random_split(subgraph);
    else if (args.split == "connectivity-snowball")
        split_assignment = propose_connectivity_snowball_split(subgraph);
    else if (args.split == "snowball")
        split_assignment = propose_snowball_split(subgraph);
    else if (args.split == "single-snowball")
        split_assignment = propose_single_snowball_split(subgraph);
    else {
        std::cerr << "Invalid split type provided." << std::endl;
        exit(-2);
    }
    split.blockmodel = std::make_shared<Blockmodel>(2, subgraph, 0.5, split_assignment);
    split.num_edges = subgraph.num_edges();
//    split.extract_subgraph = subgraph;
    return split;
}

std::vector<long> propose_random_split(const Graph &subgraph) {
    // TODO: -1s in assignment may screw with blockmodel formation
    std::vector<long> split_assignment = utils::constant<long>(subgraph.num_vertices(), -1);
//    std::uniform_int_distribution<long> distribution(0, 1);
    auto split_size = double(common::random_integer(subgraph.num_vertices()*0.2, subgraph.num_vertices()*0.8));
//    std::cout << "split size: " << split_size << std::endl;
    std::discrete_distribution<> distribution({split_size, (double) subgraph.num_vertices() - split_size});
    // TODO: implement translator for vertex IDs, and store it in Split
    for (long vertex = 0; vertex < subgraph.num_vertices(); ++vertex) {
        split_assignment[vertex] = distribution(rng::generator());
    }
    return split_assignment;
}

std::vector<long> propose_connectivity_snowball_split(const Graph &subgraph) {
    std::vector<long> split_assignment = utils::constant<long>(subgraph.num_vertices(), -1);
    MapVector<bool> unsampled(subgraph.num_vertices());
    for (long vertex = 0; vertex < subgraph.num_vertices(); ++vertex) {
        unsampled[vertex] = true;
    }
    MapVector<bool> sampled;
    MapVector<bool> frontier;
    MapVector<bool> next_frontier;
    std::vector<long> vertex_degrees = subgraph.degrees();
    // ============= SETUP ================
    std::pair<long, long> init_vertices = split_init(subgraph, vertex_degrees);
    long start_one = init_vertices.first;
    long start_two = init_vertices.second;

    int subgraph_id = 0;
    for (long start_vertex : { start_one, start_two }) {  // int subgraph_id = 0; subgraph_id < 2; ++subgraph_id) {
//        std::cout << "split_assignment size: " << split_assignment.size() << " start: " << start_vertex << std::endl;
        split_assignment[start_vertex] = subgraph_id;
        sampled[start_vertex] = true;
        unsampled.erase(start_vertex);
        subgraph_id++;
    }
    // Fill in frontiers
    for (long start_vertex : { start_one, start_two }) {
        std::vector<long> neighbors = subgraph.neighbors(start_vertex);
        for (const long &vertex : neighbors) {
            if (split_assignment[vertex] != -1) continue;  // vertices that were already sampled shouldn't be in the frontier
            frontier[vertex] = true;
        }
    }
    // ============= END OF SETUP ===============
    // ============= SNOWBALL ==============
    auto sample_vertex = [&subgraph, &split_assignment, &frontier, &sampled, &unsampled, &next_frontier](long vertex) {
        int edges_to_0 = 0;
        int edges_to_1 = 0;
        for (const long &neighbor : subgraph.neighbors(vertex)) {
            long block = split_assignment[neighbor];
            if (block == 0) edges_to_0++;
            if (block == 1) edges_to_1++;
            if (split_assignment[neighbor] != -1) continue;  // If already sampled, don't add to frontier
            next_frontier[neighbor] = true;
        }
        int subgraph_id = edges_to_1 > edges_to_0 ? 1 : 0;
        split_assignment[vertex] = subgraph_id;
        sampled[vertex] = true;
        unsampled.erase(vertex);
        frontier[vertex] = false;
    };
    while (!unsampled.empty()) {
        // TODO: sample highest degree vertex in frontier
        if (unsampled.empty()) break;
        long selected;
        if (frontier.empty()) {
            selected = unsampled.begin()->first;
            sample_vertex(selected);
            continue;
        }
        for (const auto &entry : frontier) {
            long vertex = entry.first;
            sample_vertex(vertex);
        }
        frontier = std::move(next_frontier);
        next_frontier = MapVector<bool>();
    }
    return split_assignment;
}

std::vector<long> propose_snowball_split(const Graph &subgraph) {
    std::vector<long> split_assignment = utils::constant<long>(subgraph.num_vertices(), -1);
    MapVector<bool> unsampled(subgraph.num_vertices());
    for (long vertex = 0; vertex < subgraph.num_vertices(); ++vertex) {
        unsampled[vertex] = true;
    }
    std::vector<MapVector<bool>> sampled(2);
    std::vector<MapVector<bool>> frontiers(2);
    std::vector<long> vertex_degrees = subgraph.degrees();
    // ============= SETUP ================
    std::pair<long, long> init_vertices = split_init(subgraph, vertex_degrees);
    // Mark vertices as sampled
    for (int subgraph_id = 0; subgraph_id < 2; ++subgraph_id) {
        long nth_vertex = subgraph_id == 0 ? init_vertices.first : init_vertices.second;
        split_assignment[nth_vertex] = subgraph_id;
        sampled[subgraph_id][nth_vertex] = true;
        unsampled.erase(nth_vertex);
    }
    // Fill in frontiers
    for (int subgraph_id = 0; subgraph_id < 2; ++subgraph_id) {
        long nth_vertex = subgraph_id == 0 ? init_vertices.first : init_vertices.second;
        std::vector<long> neighbors = subgraph.neighbors(nth_vertex);
        for (const long &vertex : neighbors) {
            if (split_assignment[vertex] != -1) continue;  // vertices that were already sampled shouldn't be in the frontier
            frontiers[subgraph_id][vertex] = true;
        }
    }
    // ============= END OF SETUP ===============
    // ============= SNOWBALL ==============
    auto split_size = double(common::random_integer(subgraph.num_vertices()*0.2, subgraph.num_vertices()*0.8));
    std::discrete_distribution<int> distribution({split_size, (double) subgraph.num_vertices() - split_size});
    while (!unsampled.empty()) {
//        auto subgraph_id = (int) common::random_integer(0, 1);
        int subgraph_id = distribution(rng::generator());
//        for (int subgraph_id = 0; subgraph_id < 2; ++subgraph_id) {  // Iterate through the subgraphs
        // TODO: sample highest degree vertex in frontier
        if (unsampled.empty()) break;
        long selected;
        if (frontiers[subgraph_id].empty()) { // Select a random (first) unsampled vertex.
            selected = unsampled.begin()->first;
        } else {
            selected = frontiers[subgraph_id].begin()->first;
        }
        split_assignment[selected] = subgraph_id;
        sampled[subgraph_id][selected] = true;
        unsampled.erase(selected);
        for (MapVector<bool> &frontier : frontiers) {
            frontier.erase(selected);
        }
        for (const long &neighbor : subgraph.neighbors(selected)) {
            if (split_assignment[neighbor] != -1) continue;  // If already sampled, don't add to frontier
            frontiers[subgraph_id][neighbor] = true;
        }
    }
    return split_assignment;
}

void _sample_vertex(long vertex, const Graph &subgraph, MapVector<bool> &unsampled, MapVector<bool> &sampled,
                    MapVector<bool> &frontier, MapVector<bool> &next_frontier, std::vector<long> &split_assignment,
                    long block) {
    if (sampled[vertex]) return;
    split_assignment[vertex] = block;
    sampled[vertex] = true;
    unsampled.erase(vertex);
//    frontier.erase(vertex);
    // Fill in frontiers
    std::vector<long> neighbors = subgraph.neighbors(vertex);
    for (const long &neighbor : neighbors) {
        if (split_assignment[neighbor] == 1) continue;  // vertices that were already sampled shouldn't be in the frontier
        next_frontier[neighbor] = true;
    }
}

std::vector<long> propose_single_snowball_split(const Graph &subgraph) {
    // vertices start in one cluster
    std::vector<long> split_assignment = utils::constant<long>(subgraph.num_vertices(), 0);
    MapVector<bool> unsampled(subgraph.num_vertices());
    for (long vertex = 0; vertex < subgraph.num_vertices(); ++vertex) {
        unsampled[vertex] = true;
    }
    MapVector<bool> sampled;
    MapVector<bool> frontier;
    MapVector<bool> next_frontier;
//    std::vector<long> vertex_degrees = extract_subgraph.degrees();
    // ============= SETUP ================
//    long start = utils::argmax<long>(vertex_degrees);
    std::vector<long> vertex_degrees = subgraph.degrees();
    std::pair<long, long> init_vertices = split_init(subgraph, vertex_degrees);
    long start = init_vertices.first;
//    long start = common::random_integer(0, extract_subgraph.num_vertices() - 1);
//    std::cout << "vertex " << start << " has max degree of: " << vertex_degrees[start] << std::endl;
    split_assignment[start] = 1;
    sampled[start] = true;
    unsampled.erase(start);
    std::vector<int> indices = utils::range<int>(0, subgraph.num_vertices());
    // Fill in frontiers
    std::vector<long> neighbors = subgraph.neighbors(start);
    for (const long &vertex : neighbors) {
        if (split_assignment[vertex] == 1) continue;  // vertices that were already sampled shouldn't be in the frontier
        frontier[vertex] = true;
    }
    // ============= END OF SETUP ===============
    // ============= SNOWBALL ==============
    auto target_sample_size = size_t(common::random_integer(subgraph.num_vertices()*0.2, subgraph.num_vertices()*0.8));
//    long target_sample_size = extract_subgraph.num_vertices() / 2;
    while (sampled.size() < target_sample_size) {
        if (frontier.empty() && next_frontier.empty()) break;  // do not try to reach target_sample_size if ran out of vertices
        for (const std::pair<long, bool> &entry : frontier) {
            long vertex = entry.first;
            _sample_vertex(vertex, subgraph, unsampled, sampled, frontier, next_frontier, split_assignment, 1);
            if (sampled.size() >= target_sample_size) break;
        }
        frontier = std::move(next_frontier);
        next_frontier = MapVector<bool>();
    }
    return split_assignment;
}

bool end_condition_not_reached(Blockmodel &blockmodel, TopDownBlockmodelTriplet &triplet) {
    if (args.mix) {
        return triplet.golden_ratio_not_reached();
    }
    return !sbp::done_blockmodeling(blockmodel, triplet);
}

Blockmodel run(const Graph &graph) {
    if (args.threads > 0)
        omp_set_num_threads(args.threads);
    else
        omp_set_num_threads(omp_get_num_procs());
    double start_t = MPI_Wtime();
    std::cout << "num threads: " << omp_get_max_threads() << std::endl;
    std::vector<long> initial_memberships = utils::constant<long>(graph.num_vertices(), 0);
//    Blockmodel blockmodel(1, graph, 1.0 / float(BLOCK_REDUCTION_RATE), initial_memberships);
    Blockmodel blockmodel(1, graph, 1.5, initial_memberships);
    common::candidates = std::uniform_int_distribution<long>(0, blockmodel.num_blocks() - 2);
    double initial_mdl = entropy::nonparametric::mdl(blockmodel, graph);
//    double initial_mdl = entropy::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    utils::save_partial_profile(0, -1, initial_mdl, entropy::normalize_mdl_v1(initial_mdl, graph),
                                blockmodel.num_blocks());
    TopDownBlockmodelTriplet blockmodel_triplet = TopDownBlockmodelTriplet();
    blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
    int iteration = 0;
    while (end_condition_not_reached(blockmodel, blockmodel_triplet)) {
//        std::cout << "============= Block sizes ============" << std::endl;
//        utils::print<long>(blockmodel.block_sizes());
//        std::cout << "Next iteration, we're gonna split the communities in blockmodel with B = " << blockmodel.num_blocks() << std::endl;
        if (blockmodel.getNum_blocks_to_merge() != 0) {
            std::cout << "Splitting blocks up from " << blockmodel.num_blocks() << " to "
                      << blockmodel.getNum_blocks_to_merge() << std::endl;
        }
        double block_split_t = MPI_Wtime();
        blockmodel = split_communities(blockmodel, graph, blockmodel.getNum_blocks_to_merge());
        timers::BlockSplit_time = MPI_Wtime() - block_split_t;
//        std::cout << "============== Block sizes after split" << std::endl;
//        utils::print<long>(blockmodel.block_sizes());
//        std::cout << "============== num blocks after split = " << blockmodel.num_blocks() << std::endl;
        if (iteration < 1) {
            double mdl = entropy::nonparametric::mdl(blockmodel, graph);
            utils::save_partial_profile(0.5, -1, mdl, entropy::normalize_mdl_v1(mdl, graph), blockmodel.num_blocks());
        }
        double start = MPI_Wtime();
        blockmodel = finetune::mcmc(iteration, graph, blockmodel, blockmodel_triplet);
        timers::MCMC_time = MPI_Wtime() - start;
        double mdl = blockmodel.getOverall_entropy();
        long num_blocks = blockmodel.num_blocks();
        blockmodel = blockmodel_triplet.get_next_blockmodel(blockmodel);
        common::candidates = std::uniform_int_distribution<long>(0, blockmodel.num_blocks() - 2);
        timers::total_time = MPI_Wtime() - start_t;
        start_t = MPI_Wtime();
        utils::save_partial_profile(++iteration, -1, mdl, entropy::normalize_mdl_v1(mdl, graph),
                                    num_blocks);
    }
    if (args.mix)
        blockmodel = continue_agglomerative(graph, blockmodel_triplet, iteration);
    return blockmodel;
}

Blockmodel split_communities(Blockmodel &blockmodel, const Graph &graph, int target_num_communities) {
    std::string user_arg = args.matrix_type; // args.no_transpose;
    int num_blocks = blockmodel.num_blocks();
    std::vector<Split> best_split_for_each_block(num_blocks);
    std::vector<double> delta_entropy_for_each_block =
            utils::constant<double>(num_blocks, std::numeric_limits<double>::max());
    std::vector<omp_lock_t> locks(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        omp_init_lock(&locks[i]);
    }
    // For dense runs, keep dense so the split-proposal blockmodels use DenseMatrix
    // and the dense compute path. For all other matrix types, force sparse to avoid
    // the overhead of transpose maintenance on thousands of throwaway 2-block models.
    if (user_arg != "dense") args.matrix_type = "sparse";
    double loop_start_t = MPI_Wtime();
    std::vector<Graph> subgraphs(blockmodel.num_blocks());
    std::vector<MapVector<long>> translators(blockmodel.num_blocks());
    #pragma omp parallel for schedule(dynamic) default(none) shared(graph, blockmodel, subgraphs, translators)
    for (int block = 0; block < blockmodel.num_blocks(); ++block) {
        extract_subgraph(graph, blockmodel, subgraphs[block], translators[block], block);
    }
    #pragma omp parallel for schedule(dynamic) collapse(2) default(none) \
    shared(num_blocks, NUM_AGG_PROPOSALS_PER_BLOCK, blockmodel, graph, best_split_for_each_block, delta_entropy_for_each_block, locks, subgraphs, translators)
    for (int current_block = 0; current_block < num_blocks; ++current_block) {
        for (int i = 0; i < NUM_AGG_PROPOSALS_PER_BLOCK; ++i) {
            // Do not attempt to split small clusters
            if ((double) blockmodel.block_size(current_block) < 0.001 * (double) graph.num_vertices()) {
                Split split;
                omp_set_lock(&locks[current_block]);
                best_split_for_each_block[current_block] = split;
                delta_entropy_for_each_block[current_block] = std::numeric_limits<double>::max();
                omp_unset_lock(&locks[current_block]);
                continue;
            }
//            Split split = propose_split(current_block, graph, blockmodel);
            Split split = propose_split(subgraphs[current_block], translators[current_block]);
            // TODO: currently computing delta entropy for the split ONLY. Can we compute dE for entire blockmodel?
//            double new_entropy = entropy::nonparametric::mdl(*(split.blockmodel), split.extract_subgraph);
            double new_entropy = entropy::nonparametric::mdl(*(split.blockmodel), subgraphs[current_block]);
            double old_entropy = entropy::null_mdl_v1(subgraphs[current_block]);
//            double old_entropy = entropy::null_mdl_v1(split.extract_subgraph);
            double delta_entropy = new_entropy - old_entropy;
            omp_set_lock(&locks[current_block]);
            if (delta_entropy < delta_entropy_for_each_block[current_block]) {
                delta_entropy_for_each_block[current_block] = delta_entropy;
                best_split_for_each_block[current_block] = split;
            }
            omp_unset_lock(&locks[current_block]);
        }
    }
    timers::BlockSplit_loop_time += MPI_Wtime() - loop_start_t;
    args.matrix_type = user_arg;
    for (int i = 0; i < num_blocks; ++i) {
        omp_destroy_lock(&locks[i]);
    }
//    utils::print<double>(delta_entropy_for_each_block);
//    std::cout << "splits =================" << std::endl;
//    for (size_t i = 0; i < best_split_for_each_block.size(); ++i) {
//        std::cout << "V = " << best_split_for_each_block[i].num_vertices << " dE = " << delta_entropy_for_each_block[i] << std::endl;
//    }
//    std::cout << "Applying best splits" << std::endl;
    apply_best_splits(blockmodel, best_split_for_each_block, delta_entropy_for_each_block, target_num_communities);
    blockmodel.initialize_edge_counts(graph);
    return blockmodel;
}

std::pair<long, long> split_init(const Graph &subgraph, const std::vector<long> &vertex_degrees) {
    if (args.splitinit == "random" || subgraph.num_vertices() < 10) {
        return split_init_random(subgraph);
    } else if (args.splitinit == "degree-weighted") {
        return split_init_degree_weighted(subgraph, vertex_degrees);
    } else if (args.splitinit == "high-degree") {
        return split_init_high_degree(subgraph, vertex_degrees);
    }
    throw std::logic_error("This init split type has not been implemented yet.");
}

std::pair<long, long> split_init_random(const Graph &subgraph) {
    std::uniform_int_distribution<long> distribution(0, subgraph.num_vertices() - 1);
//    std::uniform_int_distribution<long> distribution(0, (extract_subgraph.num_vertices() / 10) - 1);
    long start_one = distribution(rng::generator());
    long start_two = distribution(rng::generator());
    while (start_two == start_one) start_two = distribution(rng::generator());
    if (start_one >= subgraph.num_vertices() || start_two >= subgraph.num_vertices()) {
        std::cerr << "ERROR: random init gives too large a vertex ID! V = " << subgraph.num_vertices() << " start_one: " << start_one << " start_two: " << start_two << std::endl;
        exit(-55);
    }
    return std::make_pair(start_one, start_two);
}

std::pair<long, long> split_init_degree_weighted(const Graph &subgraph, const std::vector<long> &vertex_degrees) {
    std::vector<int> indices = utils::range<int>(0, subgraph.num_vertices());
    std::nth_element(indices.data(), indices.data() + (subgraph.num_vertices() / 10),
                     indices.data() + indices.size(), [&vertex_degrees](size_t i1, size_t i2) {
                return vertex_degrees[i1] > vertex_degrees[i2];
            });
    std::uniform_int_distribution<long> distribution(0, (subgraph.num_vertices() / 10) - 1);
    long start_one = distribution(rng::generator());
    long start_two = distribution(rng::generator());
    while (start_two == start_one) start_two = distribution(rng::generator());
    if (start_one >= subgraph.num_vertices() || start_two >= subgraph.num_vertices()) {
        std::cerr << "ERROR: degree weighted init gives too large a vertex ID! V = " << subgraph.num_vertices() << " start_one: " << start_one << " start_two: " << start_two << std::endl;
        exit(-55);
    }
    return std::make_pair(indices[start_one], indices[start_two]);
}

std::pair<long, long> split_init_high_degree(const Graph &subgraph, const std::vector<long> &vertex_degrees) {
    std::vector<int> indices = utils::range<int>(0, subgraph.num_vertices());
    std::nth_element(indices.data(), indices.data() + 3,
                     indices.data() + indices.size(), [&vertex_degrees](size_t i1, size_t i2) {
                return vertex_degrees[i1] > vertex_degrees[i2];
            });
    if (indices[0] >= subgraph.num_vertices() || indices[1] >= subgraph.num_vertices()) {
        std::cerr << "ERROR: high degree init gives too large a vertex ID! V = " << subgraph.num_vertices() << " start_one: " << indices[0] << " start_two: " << indices[1] << std::endl;
        exit(-55);
    }
    return std::make_pair(indices[0], indices[1]);
}

}

