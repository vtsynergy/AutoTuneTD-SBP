#include "finetune.hpp"

#include "args.hpp"
#include "entropy.hpp"
#include "mpi_data.hpp"
#include "rng.hpp"
#include "utils.hpp"
#include "typedefs.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
//#include <fenv.h>
#include <iostream>
#include <random>

namespace finetune {

long num_surrounded = 0;
//std::ofstream my_file;

bool accept(double delta_entropy, double hastings_correction) {
    if (!args.hastings_correction) {
        return delta_entropy < 0.0;
    }
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double random_probability = distribution(rng::generator());
    // NOTE: 3.0 can be a user parameter (beta) -- higher value favors exploitation
    double accept_probability = exp(-3.0 * delta_entropy) * hastings_correction;
    accept_probability = (accept_probability >= 1.0) ? 1.0 : accept_probability;
    return random_probability <= accept_probability;
}

Blockmodel &asynchronous_gibbs(Blockmodel &blockmodel, const Graph &graph, bool golden_ratio_not_reached) {
    std::cout << "Asynchronous Gibbs iteration" << std::endl;
    if (blockmodel.num_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    std::vector<long> vertex_moves;
    std::vector<long> vertices = utils::range<long>(0, graph.num_vertices());
    long total_vertex_moves = 0;
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    double initial_entropy = blockmodel.getOverall_entropy();
    double last_entropy = initial_entropy;
    std::vector<long> shuffled_vertices;
    if (!args.ordered) {
        shuffled_vertices = utils::range<long>(0, graph.num_vertices());
    }
    // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
    // asynchronous Gibbs sampling. Kept alive across iterations so that the per-vertex edge lists retain
    // their capacity: re-creating it per batch dominated runtime in freeing the O(V) nested vectors.
    // Note #2: static is used to avoid re-allocation of the vector across iterations. This may break in some
    // compilers because it is passed into #pragma omp parallel for as a shared variable. In that case, either
    // pull it out into the sbp() function or use a static buffer + a local std::vector<VectorMove_v3> &moves
    // reference.
    static std::vector<VertexMove_v3> moves(graph.num_vertices());
    for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        if (!args.ordered) {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(shuffled_vertices.begin(), shuffled_vertices.end(), std::mt19937_64(seed));
        }
        long _vertex_moves = 0;
        double num_batches = args.batches;
        long batch_size = long(ceil(double(graph.num_vertices()) / num_batches));
        for (long batch = 0; batch < graph.num_vertices() / batch_size; ++batch) {
            long start = batch * batch_size;
            long end = std::min(graph.num_vertices(), (batch + 1) * batch_size);
            // Only did_move is reset; the edge lists are left intact because a move is applied solely when
            // moves[vertex] has been overwritten this batch.
            for (VertexMove_v3 &move : moves) {
                move.did_move = false;
            }
            double start_t = MPI_Wtime();
            #pragma omp parallel for schedule(dynamic) default(none) \
            shared(start, end, blockmodel, graph, _vertex_moves, moves, args, shuffled_vertices)
            for (long index = start; index < end; ++index) {
                long vertex = index;
                if (!args.ordered) {
                    vertex = shuffled_vertices[index];
                }
                VertexMove_v3 proposal = propose_gibbs_move_v3(blockmodel, vertex, graph);
                moves[vertex] = proposal;
            }
            double parallel_t = MPI_Wtime();
            timers::MCMC_parallel_time += parallel_t - start_t;
            for (const VertexMove_v3 &move : moves) {
                if (!move.did_move) continue;
                if (blockmodel.move_vertex(move)) {
                    _vertex_moves++;
                }
            }
            timers::MCMC_vertex_move_time += MPI_Wtime() - parallel_t;
        }
        double entropy = entropy::mdl(blockmodel, graph);
        double delta_entropy = entropy - last_entropy;
        delta_entropies.push_back(delta_entropy);
        last_entropy = entropy;
        vertex_moves.push_back(_vertex_moves);
        timers::MCMC_moves += _vertex_moves;
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << _vertex_moves << ", delta S: ";
        std::cout << delta_entropy / initial_entropy << std::endl;
        total_vertex_moves += _vertex_moves;
        timers::MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, golden_ratio_not_reached, initial_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

Blockmodel &asynchronous_gibbs_load_balanced(Blockmodel &blockmodel, const Graph &graph,
                                             bool golden_ratio_not_reached) {
    std::cout << "Load Balanced Asynchronous Gibbs iteration" << std::endl;
    if (blockmodel.num_blocks() == 1) {
        return blockmodel;
    }
//    std::vector<std::vector<long>> thread_vertices = load_balance(graph);
    std::vector<double> delta_entropies;
    std::vector<long> vertex_moves;
    std::vector<long> all_vertices = utils::range<long>(0, graph.num_vertices());
    long total_vertex_moves = 0;
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    double initial_entropy = blockmodel.getOverall_entropy();
    double last_entropy = initial_entropy;
    for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        std::shuffle(all_vertices.begin(), all_vertices.end(), rng::generator());
        long _vertex_moves = 0;
        size_t num_batches = args.batches;
        auto batch_size = size_t(ceil(double(graph.num_vertices()) / double(num_batches)));
        long num_processed_vertices = 0;
        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t start = batch * batch_size;
            size_t end = std::min(all_vertices.size(), (batch + 1) * batch_size);
            std::vector<VertexMove_v3> moves(graph.num_vertices());
            std::vector<std::vector<long>> thread_vertices = load_balance(graph, all_vertices, start, end);
            double parallel_start_t = MPI_Wtime();
            #pragma omp parallel default(none) shared(args, batch, blockmodel, std::cout, graph, moves, \
            num_batches, thread_vertices, num_processed_vertices)
            {
                assert(omp_get_num_threads() == (int) thread_vertices.size());
                std::vector<long> &vertices = thread_vertices[omp_get_thread_num()];
//                auto batch_size = size_t(ceil(double(vertices.size()) / double(num_batches)));
//                size_t start = batch * batch_size;
//                size_t end = std::min(vertices.size(), (batch + 1) * batch_size);
//                std::shuffle(vertices.begin() + long(start), vertices.begin() + long(end), rng::generator());
                for (long vertex : vertices) { // size_t vertex_index = start; vertex_index < end; ++vertex_index) {
//                    long vertex = vertices[vertex_index];
                    VertexMove_v3 proposal = propose_gibbs_move_v3(blockmodel, vertex, graph);
                    moves[vertex] = proposal;
                    #pragma omp atomic
                    num_processed_vertices++;
                }
            }
            double end_parallel_t = MPI_Wtime();
            timers::MCMC_parallel_time += end_parallel_t - parallel_start_t;
            for (const VertexMove_v3 &move: moves) {
                if (!move.did_move) continue;
                if (blockmodel.move_vertex(move)) {
                    _vertex_moves++;
                }
            }
            timers::MCMC_vertex_move_time += MPI_Wtime() - end_parallel_t;
        }
        assert(num_processed_vertices == graph.num_vertices());
        double entropy = entropy::mdl(blockmodel, graph);
        double delta_entropy = entropy - last_entropy;
        delta_entropies.push_back(delta_entropy);
        last_entropy = entropy;
        vertex_moves.push_back(_vertex_moves);
        timers::MCMC_moves += _vertex_moves;
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << _vertex_moves << ", delta S: ";
        std::cout << delta_entropy / initial_entropy << std::endl;
        total_vertex_moves += _vertex_moves;
        timers::MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, golden_ratio_not_reached, initial_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

EdgeWeights block_edge_weights(const std::vector<long> &block_assignment, const EdgeWeights &neighbor_weights) {
    std::map<long, long> block_counts;
    for (ulong i = 0; i < neighbor_weights.indices.size(); ++i) {
        long neighbor = neighbor_weights.indices[i];
        long block = block_assignment[neighbor];
        long weight = neighbor_weights.values[i];
        block_counts[block] += weight; // block_count[new block] should initialize to 0
    }
    std::vector<long> blocks;
    std::vector<long> weights;
    for (auto const &entry: block_counts) {
        blocks.push_back(entry.first);
        weights.push_back(entry.second);
    }
    return EdgeWeights{blocks, weights};
}

Delta blockmodel_delta(long vertex, long current_block, long proposed_block, const EdgeWeights &out_edges,
                       const EdgeWeights &in_edges, const Blockmodel &blockmodel) {
    Delta delta(current_block, proposed_block, long(std::max(out_edges.indices.size(), in_edges.indices.size())), args.coodelta);

    // current_block -> current_block == proposed_block --> proposed_block  (this includes self edges)
    // current_block --> other_block == proposed_block --> other_block
    // other_block --> current_block == other_block --> proposed_block
    // current_block --> proposed_block == proposed_block --> proposed_block
    // proposed_block --> current_block == proposed_block --> proposed_block
    for (size_t i = 0; i < out_edges.indices.size(); ++i) {
        long out_vertex = out_edges.indices[i];
        long out_block = blockmodel.block_assignment(out_vertex);
        long edge_weight = out_edges.values[i];
        if (vertex == out_vertex) {
            delta.add(proposed_block, proposed_block, edge_weight);
            delta.self_edge_weight(1);
        } else {
            delta.add(proposed_block, out_block, edge_weight);
        }
        delta.sub(current_block, out_block, edge_weight);
    }
    for (size_t i = 0; i < in_edges.indices.size(); ++i) {
        long in_vertex = in_edges.indices[i];
        long in_block = blockmodel.block_assignment(in_vertex);
        long edge_weight = in_edges.values[i];
        if (vertex == in_vertex) {
            delta.add(proposed_block, proposed_block, edge_weight);
            delta.self_edge_weight(1);
        } else {
            delta.add(in_block, proposed_block, edge_weight);
        }
        delta.sub(in_block, current_block, edge_weight);
    }
    return delta;
}

std::pair<std::vector<long>, long> count_low_degree_block_neighbors(const Graph &graph, const Blockmodel &blockmodel) {
    const std::vector<long> &low_degree_vertices = graph.low_degree_vertices();
    std::vector<long> result = utils::constant<long>(blockmodel.num_blocks(), 0);
    long total = 0;
    for (long vertex : low_degree_vertices) {
        long block = blockmodel.block_assignment(vertex);
        long neighbors = blockmodel.blockmatrix()->distinct_edges(block);
        result[block] = std::max(result[block], neighbors);
        total += neighbors;
    }
    std::vector<long> ownership = utils::constant<long>((long) low_degree_vertices.size(), -1);
    long thread = 0;
    long current_neighbors = 0;
    long average_neighbors = total / omp_get_max_threads();
    for (long index = 0; index < (long) low_degree_vertices.size(); ++index) {
        long vertex = low_degree_vertices[index];
        long block = blockmodel.block_assignment(vertex);
        long neighbors = result[block];
        current_neighbors += neighbors;
        ownership[index] = thread;
        if (current_neighbors >= average_neighbors && thread < (omp_get_max_threads() - 1)) {
            current_neighbors = 0;
            thread++;
        }
    }
    return std::make_pair(ownership, total);
}

bool early_stop(long iteration, bool golden_ratio_not_reached, double initial_entropy,
                std::vector<double> &delta_entropies) {
    size_t last_index = delta_entropies.size() - 1;
    if (delta_entropies[last_index] == 0.0) {
        return true;
    }
    if (iteration < 3) {
        return false;
    }
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold;
    if (golden_ratio_not_reached) { // Golden ratio bracket not yet established
        threshold = 5e-4 * initial_entropy;
    } else {
        threshold = 1e-4 * initial_entropy;
    }
    return average < threshold;
}

bool early_stop(long iteration, double initial_entropy, std::vector<double> &delta_entropies) {
    if (iteration < 3) {
        return false;
    }
    size_t last_index = delta_entropies.size() - 1;
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold = 1e-4 * initial_entropy;
    return average < threshold;
}

bool early_stop_parallel(long iteration, bool golden_ratio_not_reached, double initial_entropy,
                         std::vector<double> &delta_entropies, std::vector<long> &vertex_moves) {
    size_t last_index = delta_entropies.size() - 1;
    if (vertex_moves[last_index] == 0) {
        return true;
    }
    if (iteration < 4) {
        return false;
    }
    double average = delta_entropies[last_index] + delta_entropies[last_index - 1] + delta_entropies[last_index - 2];
    average /= -3.0;
    double threshold;
    if (golden_ratio_not_reached) { // Golden ratio bracket not yet established
        threshold = 5e-4 * initial_entropy;
    } else {
        threshold = 1e-4 * initial_entropy;
    }
    if (average < threshold) return true;
    long max = std::max(vertex_moves[last_index - 1],
                       std::max(vertex_moves[last_index - 2], vertex_moves[last_index - 3]));
    return vertex_moves[last_index] > max;
}

[[maybe_unused]] EdgeCountUpdates edge_count_updates(ISparseMatrix *blockmodel, long current_block, long proposed_block,
                                                     EdgeWeights &out_blocks, EdgeWeights &in_blocks,
                                                     long self_edge_weight) {
    std::vector<long> block_row = blockmodel->getrow(current_block);
    std::vector<long> block_col = blockmodel->getcol(current_block);
    std::vector<long> proposal_row = blockmodel->getrow(proposed_block);
    std::vector<long> proposal_col = blockmodel->getcol(proposed_block);

    long count_in_block = 0, count_out_block = 0;
    long count_in_proposal = self_edge_weight, count_out_proposal = self_edge_weight;

    for (ulong i = 0; i < in_blocks.indices.size(); ++i) {
        long index = in_blocks.indices[i];
        long value = in_blocks.values[i];
        if (index == current_block) {
            count_in_block += value;
        }
        if (index == proposed_block) {
            count_in_proposal += value;
        }
        block_col[index] -= value;
        proposal_col[index] += value;
    }
    for (ulong i = 0; i < out_blocks.indices.size(); ++i) {
        long index = out_blocks.indices[i];
        long value = out_blocks.values[i];
        if (index == current_block) {
            count_out_block += value;
        }
        if (index == proposed_block) {
            count_out_proposal += value;
        }
        block_row[index] -= value;
        proposal_row[index] += value;
    }

    proposal_row[current_block] -= count_in_proposal;
    proposal_row[proposed_block] += count_in_proposal;
    proposal_col[current_block] -= count_out_proposal;
    proposal_col[proposed_block] += count_out_proposal;

    block_row[current_block] -= count_in_block;
    block_row[proposed_block] += count_in_block;
    block_col[current_block] -= count_out_block;
    block_col[proposed_block] += count_out_block;

    return EdgeCountUpdates{block_row, proposal_row, block_col, proposal_col};
}

// TODO: remove double counts from the edge count updates? But then we'll have to figure out how to correctly update
// the blockmodel since we won't be able to do bm.column[block/proposal] = updates.block/proposal_col
void edge_count_updates_sparse(const Blockmodel &blockmodel, long vertex, long current_block, long proposed_block,
                               EdgeWeights &out_edges, EdgeWeights &in_edges, SparseEdgeCountUpdates &updates) {
    updates.block_row = blockmodel.blockmatrix()->getrow_sparse(current_block);
    updates.block_col = blockmodel.blockmatrix()->getcol_sparse(current_block);
    updates.proposal_row = blockmodel.blockmatrix()->getrow_sparse(proposed_block);
    updates.proposal_col = blockmodel.blockmatrix()->getcol_sparse(proposed_block);

    for (size_t i = 0; i < out_edges.indices.size(); ++i) {
        long out_vertex = out_edges.indices[i];
        long out_block = blockmodel.block_assignment(out_vertex);
        long edge_weight = out_edges.values[i];
        if (vertex == out_vertex) {
            updates.proposal_row[proposed_block] += edge_weight;
            updates.proposal_col[proposed_block] += edge_weight;
        } else {
            updates.proposal_row[out_block] += edge_weight;
            if (out_block == proposed_block)
                updates.proposal_col[proposed_block] += edge_weight;
            if (out_block == current_block)
                updates.block_col[proposed_block] += edge_weight;
        }
        updates.block_row[out_block] -= edge_weight;
        if (out_block == current_block)
            updates.block_col[current_block] -= edge_weight;
        if (out_block == proposed_block)
            updates.proposal_col[current_block] -= edge_weight;
    }
    for (size_t i = 0; i < in_edges.indices.size(); ++i) {
        long in_vertex = in_edges.indices[i];
        long in_block = blockmodel.block_assignment(in_vertex);
        long edge_weight = in_edges.values[i];
        if (vertex == in_vertex) {
            updates.proposal_col[proposed_block] += edge_weight;
            updates.proposal_row[proposed_block] += edge_weight;
        } else {
            updates.proposal_col[in_block] += edge_weight;
            if (in_block == proposed_block)
                updates.proposal_row[proposed_block] += edge_weight;
            if (in_block == current_block)
                updates.block_row[proposed_block] += edge_weight;
        }
        updates.block_col[in_block] -= edge_weight;
        if (in_block == current_block)
            updates.block_row[current_block] -= edge_weight;
        if (in_block == proposed_block)
            updates.proposal_row[current_block] -= edge_weight;
    }
}

EdgeWeights edge_weights(const NeighborView &neighbors, long vertex, bool ignore_self) {
    std::vector<long> indices;
    std::vector<long> values;
    // Assumes graph is unweighted
    for (const long neighbor : neighbors) {
        if (ignore_self && neighbor == vertex) continue;
        indices.push_back(neighbor);
        values.push_back(1);
    }
    return EdgeWeights{indices, values};
}

VertexMove eval_vertex_move(long vertex, long current_block, utils::ProposalAndEdgeCounts proposal,
                            const Blockmodel &blockmodel, const Graph &graph, EdgeWeights &out_edges,
                            EdgeWeights &in_edges) {
//    if (args.nodelta)
//        return eval_vertex_move_nodelta(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
    const Delta delta = blockmodel_delta(vertex, current_block, proposal.proposal, out_edges, in_edges, blockmodel);
    double hastings = entropy::hastings_correction(vertex, graph, blockmodel, delta, current_block, proposal);
    double delta_entropy = !args.parametric ?
            entropy::nonparametric::delta_mdl(blockmodel, graph, vertex, delta, proposal) :
            entropy::delta_mdl(blockmodel, delta, proposal);
    if (accept(delta_entropy, hastings))
        return VertexMove{delta_entropy, true, vertex, proposal.proposal};
    return VertexMove{delta_entropy, false, -1, -1};
}

VertexMove_v3 eval_vertex_move_v3(long vertex, long current_block, utils::ProposalAndEdgeCounts proposal,
                                 const Blockmodel &blockmodel, const Graph &graph, EdgeWeights &out_edges,
                                 EdgeWeights &in_edges) {
    // TODO: things like size and such should be ulong/size_t, not long.
    Vertex v = { vertex, long(graph.out_neighbors(vertex).size()), long(graph.in_neighbors(vertex).size()) };
    const Delta delta = blockmodel_delta(vertex, current_block, proposal.proposal, out_edges, in_edges, blockmodel);
    double hastings = entropy::hastings_correction(vertex, graph, blockmodel, delta, current_block, proposal);
    double delta_entropy = !args.parametric ?
                           entropy::nonparametric::delta_mdl(blockmodel, graph, vertex, delta, proposal) :
                           entropy::delta_mdl(blockmodel, delta, proposal);
    if (accept(delta_entropy, hastings))
        return VertexMove_v3{delta_entropy, true, v, proposal.proposal, out_edges, in_edges};
//        return VertexMove_v2{delta_entropy, true, vertex, proposal.proposal, out_edges, in_edges};
    return VertexMove_v3{delta_entropy, false, InvalidVertex, -1, out_edges, in_edges};
//    return VertexMove_v2{delta_entropy, false, -1, -1, out_edges, in_edges};
}

Blockmodel &hybrid_mcmc_load_balanced(Blockmodel &blockmodel, const Graph &graph, bool golden_ratio_not_reached) {
        std::cout << "Hybrid MCMC iteration" << std::endl;
        if (blockmodel.num_blocks() == 1) {
            return blockmodel;
        }
        std::vector<double> delta_entropies;
        long total_vertex_moves = 0;
        blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
        double initial_entropy = blockmodel.getOverall_entropy();
        double last_entropy = initial_entropy;
        double num_batches = args.batches;
        long num_low_degree_vertices = long(graph.low_degree_vertices().size());
        long batch_size = long(ceil(num_low_degree_vertices / num_batches));
        std::vector<unsigned long> thread_degrees(omp_get_max_threads());
        static std::vector<VertexMove_v3> moves(graph.num_vertices());
//        std::vector<std::pair<long,long>> vertex_properties = sort_vertices_by_degree(graph);
        for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
//            std::vector<std::pair<long,long>> block_neighbors = sort_vertices_by_degree(graph);
            for (long i = 0; i < omp_get_max_threads(); ++i) {
                thread_degrees[i] = 0;
            }
            num_surrounded = 0;
            long vertex_moves = 0;
            double start_t = MPI_Wtime();
            for (long vertex : graph.high_degree_vertices()) {  // Only run Metropolis-Hastings on high-degree vertices
                VertexMove proposal = propose_move(blockmodel, vertex, graph);
                if (proposal.did_move) {
                    vertex_moves++;
                }
            }
            double sequential_t = MPI_Wtime();
            timers::MCMC_sequential_time += sequential_t - start_t;
            std::pair<std::vector<long>, long> block_neighbors = count_low_degree_block_neighbors(graph, blockmodel);
            // TODO: make sure that with batches, we still go over every vertex in the graph
            for (long batch = 0; batch < num_low_degree_vertices / batch_size; ++batch) {
                long start = batch * batch_size;
                long end = std::min(num_low_degree_vertices, (batch + 1) * batch_size);
                // Only did_move is reset; the edge lists are left intact because a move is applied solely when
                // moves[vertex] has been overwritten this batch.
                for (VertexMove_v3 &move : moves) {
                    move.did_move = false;
                }
                // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
                // asynchronous Gibbs sampling
                std::vector<long> block_assignment(blockmodel.block_assignment());
//                omp_set_dynamic(0);
                start_t = MPI_Wtime();
                #pragma omp parallel default(none) shared(start, end, blockmodel, graph, block_assignment, moves, thread_degrees, block_neighbors, std::cout)
                {
                    long thread_id = omp_get_thread_num();
                    if (thread_id == 0)
                        std::cout << "Using " << omp_get_num_threads() << "/" << omp_get_max_threads() << " threads!" << std::endl;
                    // TODO: figure out how to make this happen once per iteration
//                    std::vector<bool> my_blocks = load_balance(blockmodel, block_neighbors);
//                    std::vector<bool> my_vertices = load_balance_vertices(graph, vertex_properties);
//                    std::vector<bool> my_vertices = load_balance_block_neighbors(graph, blockmodel, block_neighbors);
//                    long num_processed = 0;
                    for (long index = start; index < end; ++index) {
                        if (block_neighbors.first[index] != thread_id) continue;
                        long vertex = graph.low_degree_vertices()[index];
//                        if (!my_vertices[vertex]) continue;
                        long block = blockmodel.block_assignment(vertex);
//                        if (!my_blocks[block]) continue;  // Only process the vertices this thread is responsible for
                        unsigned long num_neighbors = blockmodel.blockmatrix()->distinct_edges(block);
                        thread_degrees[thread_id] += num_neighbors;
                        VertexMove_v3 proposal = propose_gibbs_move_v3(blockmodel, vertex, graph);
                        if (proposal.did_move) {
                            block_assignment[vertex] = proposal.proposed_block;
                        }
                        moves[vertex] = proposal;
//                        num_processed++;
                    }
//                    std::cout << thread_id << ": " << num_processed << std::endl;
                }
                double parallel_t = MPI_Wtime();
                timers::MCMC_parallel_time += parallel_t - start_t;
                for (const VertexMove_v3 &move : moves) {
                    if (!move.did_move)
                        continue;
                    if (blockmodel.move_vertex(move)) {
                        vertex_moves++;
                    }
                }
                timers::MCMC_vertex_move_time += MPI_Wtime() - parallel_t;
            }
            double entropy = entropy::mdl(blockmodel, graph);
            double delta_entropy = entropy - last_entropy;
            delta_entropies.push_back(delta_entropy);
            last_entropy = entropy;
            std::cout << "Itr: " << iteration << ", number of vertex moves: " << vertex_moves << ", delta S: ";
            std::cout << delta_entropy / initial_entropy << ", num surrounded vertices: " << num_surrounded << std::endl;
            total_vertex_moves += vertex_moves;
            timers::MCMC_iterations++;
            // Early stopping
            if (early_stop(iteration, golden_ratio_not_reached, initial_entropy, delta_entropies)) {
                break;
            }
        }
        blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
        std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
        std::cout << blockmodel.getOverall_entropy() << std::endl;
        timers::MCMC_moves += total_vertex_moves;
        return blockmodel;
    }

Blockmodel &hybrid_mcmc(Blockmodel &blockmodel, const Graph &graph, bool golden_ratio_not_reached) {
    std::cout << "Hybrid MCMC iteration" << std::endl;
    if (blockmodel.num_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    std::vector<long> vertex_moves;
    long total_vertex_moves = 0;
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    double initial_entropy = blockmodel.getOverall_entropy();
    double last_entropy = initial_entropy;
    double num_batches = args.batches;
    long num_low_degree_vertices = long(graph.low_degree_vertices().size());
    long batch_size = long(ceil(num_low_degree_vertices / num_batches));
    std::vector<long> hdv = graph.high_degree_vertices();
    std::vector<long> ldv = graph.low_degree_vertices();
    static std::vector<VertexMove_v3> moves(graph.num_vertices());
    for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
//        std::cout << "thread_limit: " << omp_get_max_threads() << std::endl;
        num_surrounded = 0;
        long _vertex_moves = 0;
        double start_t = MPI_Wtime();
        if (!args.ordered) {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(hdv.begin(), hdv.end(), std::mt19937_64(seed));
            std::shuffle(ldv.begin(), ldv.end(), std::mt19937_64(seed));
        }
        for (long vertex : hdv) {  // Only run Metropolis-Hastings on high-degree vertices
            VertexMove proposal = propose_move(blockmodel, vertex, graph);
            if (proposal.did_move) {
                _vertex_moves++;
            }
        }
        double sequential_t = MPI_Wtime();
        timers::MCMC_sequential_time += sequential_t - start_t;
//        assert(blockmodel.validate(graph));
        for (long batch = 0; batch < num_low_degree_vertices / batch_size; ++batch) {
            start_t = MPI_Wtime();
            long start = batch * batch_size;
            long end = std::min(num_low_degree_vertices, (batch + 1) * batch_size);
            // Only did_move is reset; the edge lists are left intact because a move is applied solely when
            // moves[vertex] has been overwritten this batch.
            for (VertexMove_v3 &move : moves) {
                move.did_move = false;
            }
            // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
            // asynchronous Gibbs sampling
            #pragma omp parallel for schedule(dynamic) default(none) \
            shared(start, end, blockmodel, graph, _vertex_moves, moves, ldv)
            for (long index = start; index < end; ++index) {
                long vertex = ldv[index];
                VertexMove_v3 proposal = propose_gibbs_move_v3(blockmodel, vertex, graph);
                moves[vertex] = proposal;
            }
            double parallel_t = MPI_Wtime();
            timers::MCMC_parallel_time += parallel_t - start_t;
            for (const VertexMove_v3 &move : moves) {
                if (!move.did_move) continue;
                if (blockmodel.move_vertex(move)) {
                    _vertex_moves++;
                }
            }
            timers::MCMC_vertex_move_time += MPI_Wtime() - parallel_t;
//            assert(blockmodel.validate(graph));
        }
        double entropy = entropy::mdl(blockmodel, graph);
        double delta_entropy = entropy - last_entropy;
        delta_entropies.push_back(delta_entropy);
        last_entropy = entropy;
        vertex_moves.push_back(_vertex_moves);
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << _vertex_moves << ", delta S: ";
        std::cout << delta_entropy / initial_entropy << ", num surrounded vertices: " << num_surrounded << std::endl;
        total_vertex_moves += _vertex_moves;
        timers::MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, golden_ratio_not_reached, initial_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    timers::MCMC_moves += total_vertex_moves;
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

std::vector<std::vector<long>> load_balance(const Graph &graph) {
    std::vector<std::vector<long>> thread_vertices(args.threads);
    std::vector<long> vertex_degrees = graph.degrees();
    std::vector<long> sorted_indices = utils::argsort<long>(vertex_degrees);
    for (size_t index = 0; index < size_t(graph.num_vertices()); ++index) {
        long vertex = sorted_indices[index];
        size_t thread_id = index % (2 * args.threads);
        if (thread_id >= size_t(args.threads))
            thread_id = ((2 * args.threads) - 1) - thread_id;
        thread_vertices[thread_id].push_back(vertex);
    }
    return thread_vertices;
}

std::vector<std::vector<long>> load_balance(const Graph &graph, const std::vector<long> &all_vertices,
                                            size_t start_index, size_t end_index) {
    std::vector<std::vector<long>> thread_vertices(args.threads);
    std::vector<long> vertex_degrees;
    for (size_t i = start_index; i < end_index; ++i) {
        long vertex = all_vertices[i];
        vertex_degrees.push_back(graph.degree(vertex));
    }
    std::vector<long> sorted_indices = utils::argsort<long>(vertex_degrees);
    for (size_t index = 0; index < vertex_degrees.size(); ++index) {
        size_t sorted_index = sorted_indices[index];
        size_t vertex_index = sorted_index + start_index;
        long vertex = all_vertices[vertex_index];
        size_t thread_id = sorted_index % (2 * args.threads);
        if (thread_id >= size_t(args.threads))
            thread_id = ((2 * args.threads) - 1) - thread_id;
        thread_vertices[thread_id].push_back(vertex);
    }
    return thread_vertices;
}


std::vector<bool> load_balance(const Blockmodel &blockmodel, const std::vector<std::pair<long, long>> &block_neighbors) {
    // Decide which blocks each thread is responsible for
    long thread_id = omp_get_thread_num();
    std::vector<bool> my_blocks = utils::constant<bool>(blockmodel.num_blocks(), false);
    for (long i = thread_id; i < blockmodel.num_blocks(); i += 2 * omp_get_max_threads()) {
        long block = block_neighbors[i].first;
        my_blocks[block] = true;
    }
    for (long i = 2 * omp_get_max_threads() - 1 - thread_id; i < blockmodel.num_blocks(); i += 2 * omp_get_max_threads()) {
        long block = block_neighbors[i].first;
        my_blocks[block] = true;
    }
    return my_blocks;
}

std::vector<bool> load_balance_block_neighbors(const Graph &graph, const Blockmodel &blockmodel,
                                               const std::pair<std::vector<long>, long> &block_neighbors) {
    // Decide which blocks each thread is responsible for
    long thread_id = omp_get_thread_num();
    std::vector<bool> my_vertices = utils::constant<bool>(graph.num_vertices(), false);
    long total_neighbors = block_neighbors.second;
//    for (long num_neighbors : block_neighbors.first) {
//        total_neighbors += num_neighbors;
//    }
    long average_neighbors = (long) total_neighbors / omp_get_max_threads();
    if (thread_id == 4)
        std::cout << "average neighbors: " << average_neighbors << " total neighbors: " << total_neighbors << std::endl << " threads: " << omp_get_max_threads();
    long thread = 0;
    long current_neighbors = 0;
    for (long vertex : graph.low_degree_vertices()) {
        long block = blockmodel.block_assignment(vertex);
        current_neighbors += block_neighbors.first[block];
    }
    assert(current_neighbors == total_neighbors);
    current_neighbors = 0;
    for (long vertex : graph.low_degree_vertices()) {
        long block = blockmodel.block_assignment(vertex);
        long neighbors = block_neighbors.first[block];
        if (thread == thread_id) {
            my_vertices[vertex] = true;
        }
        current_neighbors += neighbors;
        if (current_neighbors >= average_neighbors && thread < (omp_get_max_threads() - 1)) {
            current_neighbors = 0;
            thread++;
        }
        if (thread_id == 4) {
            std::cout << "block: " << block << " neighbors: " << neighbors << " thread: " << thread << " current_neighbors: " << current_neighbors << std::endl;
        }
    }
    if (thread_id == 4)
        std::cout << "average neighbors: " << average_neighbors << " total neighbors: " << total_neighbors << std::endl << " threads: " << omp_get_max_threads();
    return my_vertices;
}

std::vector<bool> load_balance_vertices(const Graph &graph, const std::vector<std::pair<long, long>> &vertex_properties) {
    // Decide which blocks each thread is responsible for
    long thread_id = omp_get_thread_num();
    std::vector<bool> my_vertices = utils::constant<bool>(graph.num_vertices(), false);
    for (long i = thread_id; i < graph.num_vertices(); i += 2 * omp_get_max_threads()) {
        long vertex = vertex_properties[i].first;
        my_vertices[vertex] = true;
    }
    for (long i = 2 * omp_get_max_threads() - 1 - thread_id; i < graph.num_vertices(); i += 2 * omp_get_max_threads()) {
        long vertex = vertex_properties[i].first;
        my_vertices[vertex] = true;
    }
    return my_vertices;
}

Blockmodel &mcmc(int iteration, const Graph &graph, Blockmodel &blockmodel, BlockmodelTriplet &blockmodel_triplet) {
//    timers::MCMC_moves = 0;
//    timers::MCMC_iterations = 0;
//    timers::MCMC_vertex_move_time = 0;
//    timers::MCMC_parallel_time = 0;
//    timers::MCMC_sequential_time = 0;
    common::candidates = std::uniform_int_distribution<long>(0, blockmodel.num_blocks() - 2);
//    std::cout << "Starting MCMC vertex moves" << std::endl;
    if (args.algorithm == "async_gibbs" && iteration < args.asynciterations)
        blockmodel = finetune::asynchronous_gibbs(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
    else if (args.algorithm == "async_gibbs_load_balanced" && iteration < args.asynciterations)
        blockmodel = finetune::asynchronous_gibbs_load_balanced(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
    else if (args.algorithm == "hybrid_mcmc")
        blockmodel = finetune::hybrid_mcmc(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
    else // args.algorithm == "metropolis_hastings"
        blockmodel = finetune::metropolis_hastings(blockmodel, graph, blockmodel_triplet.golden_ratio_not_reached());
    return blockmodel;
}

Blockmodel &metropolis_hastings(Blockmodel &blockmodel, const Graph &graph, bool golden_ratio_not_reached) {
    std::cout << "Metropolis hastings iteration" << std::endl;
    if (blockmodel.num_blocks() == 1) {
        return blockmodel;
    }
    std::vector<double> delta_entropies;
    long total_vertex_moves = 0;
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    std::vector<long> shuffled_vertices;
    if (!args.ordered) {
        shuffled_vertices = utils::range<long>(0, graph.num_vertices());
    }
    for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        if (!args.ordered) {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(shuffled_vertices.begin(), shuffled_vertices.end(), std::mt19937_64(seed));
        }
        long vertex_moves = 0;
        double delta_entropy = 0.0;
        double start_t = MPI_Wtime();
        for (long index = 0; index < graph.num_vertices(); ++index) {
            long vertex = index;
            if (!args.ordered) {
                vertex = shuffled_vertices[index];
            }
            VertexMove proposal = propose_move(blockmodel, vertex, graph);
            if (proposal.did_move) {
                vertex_moves++;
                delta_entropy += proposal.delta_entropy;
            }
        }
        timers::MCMC_sequential_time += MPI_Wtime() - start_t;
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of vertex moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy << std::endl;
        total_vertex_moves += vertex_moves;
        timers::MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, golden_ratio_not_reached, blockmodel.getOverall_entropy(), delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    timers::MCMC_moves += total_vertex_moves;
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

VertexMove move_vertex(long vertex, long current_block, utils::ProposalAndEdgeCounts proposal, Blockmodel &blockmodel,
                       const Graph &graph, EdgeWeights &out_edges, EdgeWeights &in_edges) {
//    if (args.nodelta)
//        return move_vertex_nodelta(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
    Delta delta = blockmodel_delta(vertex, current_block, proposal.proposal, out_edges, in_edges,
                                   blockmodel);
    double hastings = entropy::hastings_correction(vertex, graph, blockmodel, delta, current_block, proposal);
    double delta_entropy = !args.parametric ?
                           entropy::nonparametric::delta_mdl(blockmodel, graph, vertex, delta, proposal) :
                           entropy::delta_mdl(blockmodel, delta, proposal);
//    double delta_entropy = entropy::delta_mdl(blockmodel, delta, proposal);

    if (accept(delta_entropy, hastings)) {
        Vertex v = { vertex, (long) graph.out_neighbors(vertex).size(), (long) graph.in_neighbors(vertex).size() };
        blockmodel.move_vertex(v, delta, proposal);
        return VertexMove{delta_entropy, true, vertex, proposal.proposal};
    }
    return VertexMove{delta_entropy, false, vertex, proposal.proposal};
}

VertexMove propose_move(Blockmodel &blockmodel, long vertex, const Graph &graph) {
    bool did_move = false;
    long current_block = blockmodel.block_assignment(vertex);
    if (blockmodel.block_size(current_block) == 1) {
        return VertexMove{std::numeric_limits<double>::max(), did_move, -1, -1 };
    }
    EdgeWeights out_edges = edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(vertex), vertex, true);

    MapVector<long> neighbor_blocks;
    for (long neighbor : out_edges.indices) {
        neighbor_blocks[blockmodel.block_assignment(neighbor)] += 1;
    }
    for (long neighbor : in_edges.indices) {
        neighbor_blocks[blockmodel.block_assignment(neighbor)] += 1;
    }
    if (neighbor_blocks.size() == 1) {
        num_surrounded += 1;
    }

    utils::ProposalAndEdgeCounts proposal = common::propose_new_block(
            current_block, out_edges, in_edges, blockmodel.block_assignment(), blockmodel, false);
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1 };
    }

    return move_vertex(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
}

VertexMove_v3 propose_gibbs_move_v3(const Blockmodel &blockmodel, long vertex, const Graph &graph) {
    bool did_move = false;
    long current_block = blockmodel.block_assignment(vertex);
    // TODO: need to do this more intelligently. Instead of preventing moves here, prevent them in the code that
    // actually does the moves.
//    if (blockmodel.block_size(current_block) <= args.threads) {
//        return VertexMove_v3{ 0.0, did_move, InvalidVertex, -1 };
//    }

    EdgeWeights out_edges = edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(vertex), vertex, true);

    MapVector<long> neighbor_blocks;
    for (long neighbor : out_edges.indices) {
        neighbor_blocks[blockmodel.block_assignment(neighbor)] += 1;
    }
    for (long neighbor : in_edges.indices) {
        neighbor_blocks[blockmodel.block_assignment(neighbor)] += 1;
    }
    if (neighbor_blocks.size() == 1) {
        num_surrounded += 1;
    }

    utils::ProposalAndEdgeCounts proposal = common::propose_new_block(current_block, out_edges, in_edges,
                                                                      blockmodel.block_assignment(), blockmodel,
                                                                      false);
    if (proposal.proposal == current_block) {
        return VertexMove_v3{0.0, did_move, {-1, -1, -1 }, -1, out_edges, in_edges};
    }
    return eval_vertex_move_v3(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
}

[[maybe_unused]] Blockmodel &finetune_assignment(Blockmodel &blockmodel, Graph &graph) {
    std::cout << "Fine-tuning partition results after sample results have been extended to full graph" << std::endl;
    std::vector<double> delta_entropies;
    // TODO: Add number of finetuning iterations to evaluation
    long total_vertex_moves = 0;
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        long vertex_moves = 0;
        double delta_entropy = 0.0;
        for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            VertexMove proposal = propose_move(blockmodel, vertex, graph);
            if (proposal.did_move) {
                vertex_moves++;
                delta_entropy += proposal.delta_entropy;
            }
        }
        delta_entropies.push_back(delta_entropy);
        std::cout << "Itr: " << iteration << ", number of finetuning moves: " << vertex_moves << ", delta S: ";
        std::cout << delta_entropy / blockmodel.getOverall_entropy() << std::endl;
        total_vertex_moves += vertex_moves;
        // Early stopping
        if (early_stop(iteration, blockmodel.getOverall_entropy(), delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    std::cout << blockmodel.getOverall_entropy() << std::endl;
    return blockmodel;
}

std::vector<std::pair<long, long>> sort_blocks_by_neighbors(const Blockmodel &blockmodel) {
    std::vector<std::pair<long, long>> block_neighbors;
    for (long i = 0; i < blockmodel.num_blocks(); ++i) {
        block_neighbors.emplace_back(std::make_pair(i, blockmodel.blockmatrix()->distinct_edges(i)));
    }
    utils::radix_sort(block_neighbors);
//    std::sort(block_neighbors.begin(), block_neighbors.end(),
//              [](const std::pair<long, long> &a, const std::pair<long, long> &b) { return a.second > b.second; });
//    std::cout << "thread_limit: " << omp_get_max_threads() << std::endl;
    return block_neighbors;
}

std::vector<std::pair<long, long>> sort_blocks_by_size(const Blockmodel &blockmodel) {
    std::vector<std::pair<long,long>> block_sizes;
    for (long i = 0; i < blockmodel.num_blocks(); ++i) {
        block_sizes.emplace_back(std::make_pair(i, 0));
    }
    for (const long &block : blockmodel.block_assignment()) {
        block_sizes[block].second++;
    }
    utils::radix_sort(block_sizes);
//    std::sort(block_sizes.begin(), block_sizes.end(),
//              [](const std::pair<long, long> &a, const std::pair<long, long> &b) { return a.second > b.second; });
//    std::cout << "thread_limit: " << omp_get_max_threads() << std::endl;
    return block_sizes;
}

std::vector<std::pair<long,long>> sort_vertices_by_degree(const Graph &graph) {
    std::vector<std::pair<long,long>> vertex_degrees;
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        long degree = (long)(graph.out_neighbors(vertex).size() + graph.in_neighbors(vertex).size());
        vertex_degrees.emplace_back(std::make_pair(vertex, degree));
    }
    utils::radix_sort(vertex_degrees);
//    std::sort(vertex_degrees.begin(), vertex_degrees.end(),
//              [](const std::pair<long, long> &a, const std::pair<long, long> &b) { return a.second > b.second; });
//    std::cout << "thread_limit: " << omp_get_max_threads() << std::endl;
    return vertex_degrees;
}

//namespace undirected {
//
//double mdl(const Blockmodel &blockmodel, long num_vertices, long num_edges) {
//    std::cout << "undirected!" << std::endl;
//    double log_posterior_p = blockmodel.log_posterior_probability(num_edges);
//    if (std::isnan(log_posterior_p)) {
//        std::cout << "nan in log posterior" << std::endl;
//        exit(-5000);
//    }
//    double x = blockmodel.getNum_blocks() * (blockmodel.num_blocks() + 1.0) / (2.0 * num_edges);
//    if (std::isnan(x)) {
//        std::cout << "nan in X" << std::endl;
//        exit(-5000);
//    }
//    double h = ((1 + x) * log(1 + x)) - (x * log(x));
//    if (std::isnan(h)) {
//        std::cout << "nan in h()" << std::endl;
//    }
//        // std::cout << "X: " << x << std::endl;
//        // std::cout << "log(X): " << log(x) << std::endl;
//    if (std::isnan(h)) {
//        exit(-5000);
//    }
//    double first = (num_edges * h) + (num_vertices * log(blockmodel.num_blocks()));
//    std::cout << "first: " << first << " log_posterior: " << log_posterior_p << std::endl;
//    double result = (num_edges * h) + (num_vertices * log(blockmodel.num_blocks())) - log_posterior_p;
//    if (std::isnan(result)) {
//        std::cout << "nan in result" << std::endl;
//        exit(-5000);
//    }
//    return result;
//}
//
//}  // namespace undirected

}  // namespace finetune
