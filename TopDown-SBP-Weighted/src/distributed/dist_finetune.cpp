/**
 * The distributed finetuning phase of the stochastic block blockmodeling algorithm.
 */
#ifndef SBP_DIST_FINETUNE_HPP
#define SBP_DIST_FINETUNE_HPP

#include "distributed/dist_finetune.hpp"

#include <mpi.h>

#include "distributed/dist_common.hpp"
#include "entropy.hpp"
#include "finetune.hpp"

namespace finetune::dist {

std::vector<double> MCMC_RUNTIMES;
std::vector<unsigned long> MCMC_VERTEX_EDGES;
std::vector<long> MCMC_NUM_BLOCKS;
std::vector<unsigned long> MCMC_BLOCK_DEGREES;
std::vector<unsigned long long> MCMC_AGGREGATE_BLOCK_DEGREES;

const int MEMBERSHIP_T_BLOCK_LENGTHS[2] = {1, 1};
const MPI_Aint MEMBERSHIP_T_DISPLACEMENTS[2] = {0, sizeof(long)};
const MPI_Datatype MEMBERSHIP_T_TYPES[2] = {MPI_LONG, MPI_LONG};

MPI_Datatype Membership_t;

std::vector<Membership> mpi_get_assignment_updates(const std::vector<Membership> &membership_updates) {
//    for (const auto &m : membership_updates) {
//        if (m.vertex < 0 || m.vertex >= 50000) {
//            std::cout << mpi.rank << " ERROR: invalid vertex " << m.vertex << std::endl;
//            exit(-111);
//        }
//    }
    int num_moves = static_cast<int>(membership_updates.size());
    std::vector<int> rank_moves(mpi.num_processes);
    utils::MPI(MPI_Allgather(&num_moves, 1, MPI_INT, rank_moves.data(), 1, MPI_INT, mpi.comm));

    std::vector<int> offsets(mpi.num_processes);
    offsets[0] = 0;
    for (int i = 1; i < mpi.num_processes; ++i) {
        offsets[i] = offsets[i - 1] + rank_moves[i - 1];
    }

    int total_moves = offsets[mpi.num_processes - 1] + rank_moves[mpi.num_processes - 1];
    std::vector<Membership> collected_membership_updates(total_moves, {-1, -1});

//    std::cout << mpi.rank << " | size of cmu: " << collected_membership_updates.size() << std::endl;
//    for (int i = 0; i < mpi.num_processes; ++i) {
//        std::cout << "(offset: " << offsets[i] << ", rank_moves: " << rank_moves[i] << "), ";
//    }
//    std::cout << std::endl;

    utils::MPI(MPI_Allgatherv(
            membership_updates.data(), num_moves, Membership_t,collected_membership_updates.data(),
            rank_moves.data(), offsets.data(), Membership_t, mpi.comm));

    return collected_membership_updates;
}


bool async_move(const Membership &membership, const Graph &graph, TwoHopBlockmodel &blockmodel) {
    EdgeWeights out_edges = edge_weights(graph.out_neighbors(membership.vertex), membership.vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(membership.vertex), membership.vertex, true);
    Vertex v = { membership.vertex,
                 (long) graph.out_neighbors(membership.vertex).size(),
                 (long) graph.in_neighbors(membership.vertex).size() };
    VertexMove_v3 move {
            0.0, true, v, membership.block, out_edges, in_edges
    };
    return blockmodel.move_vertex(move);
}

std::vector<Membership> asynchronous_gibbs_iteration(TwoHopBlockmodel &blockmodel, const Graph &graph,
                                                     std::vector<long> *next_assignment, MPI_Win mcmc_window,
                                                     const std::vector<long> &active_set, int batch) {
    std::vector<Membership> membership_updates;
    std::vector<long> vertices;
    if (active_set.empty())
        vertices = utils::range<long>(0, graph.num_vertices());
    else
        vertices = active_set;
    auto batch_size = size_t(ceil(double(vertices.size()) / args.batches));
    size_t start = batch * batch_size;
    size_t end = std::min(vertices.size(), (batch + 1) * batch_size);
    #pragma omp parallel for schedule(dynamic) default(none) \
    shared(args, start, end, vertices, blockmodel, graph, membership_updates, mcmc_window, next_assignment)
    for (size_t index = start; index < end; ++index) {
        long vertex = vertices[index];
        if (!blockmodel.owns_vertex(vertex)) continue;
        VertexMove proposal = dist::propose_gibbs_move(blockmodel, vertex, graph);
        if (proposal.did_move) {
            remote_update_membership(vertex, proposal.proposed_block, membership_updates, next_assignment, mcmc_window);
        }
    }
    return membership_updates;
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

Blockmodel &finetune_assignment(TwoHopBlockmodel &blockmodel, Graph &graph) {
    return mcmc(graph, blockmodel, false);
    MPI_Win mcmc_window;
    std::vector<long> next_assignment;
    if (args.nonblocking) {
//        if (mpi.rank == 0) std::cout << "nonblocking call" << std::endl;
        std::cout << mpi.rank << " | initializing the MPI single-sided comm window" << std::endl;
        next_assignment = blockmodel.block_assignment();
        auto win_size = long(next_assignment.size() * sizeof(long));
        MPI_Win_create(next_assignment.data(), win_size, sizeof(long), MPI_INFO_NULL,
                       mpi.comm, &mcmc_window);
        assert(next_assignment.size() == (size_t) graph.num_vertices());
    } else {
        MPI_Type_create_struct(2, MEMBERSHIP_T_BLOCK_LENGTHS, MEMBERSHIP_T_DISPLACEMENTS, MEMBERSHIP_T_TYPES,
                               &Membership_t);
        MPI_Type_commit(&Membership_t);
    }
    if (mpi.rank == 0)
        std::cout << "Fine-tuning partition results after sample results have been extended to full graph" << std::endl;
    std::vector<double> delta_entropies;
    long total_vertex_moves = 0;
    double old_entropy = !args.parametric ?
                  entropy::nonparametric::mdl(blockmodel, graph) :
                  entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    blockmodel.setOverall_entropy(old_entropy);
    for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        double start_t = MPI_Wtime();
        if (args.nonblocking) {
//            if (mpi.rank == 0) std::cout << "nonblocking call" << std::endl;
            next_assignment.assign(blockmodel.block_assignment().begin(), blockmodel.block_assignment().end());
        }
//        for (const long &b: block_assignment) {
//            assert(b < blockmodel.num_blocks());
//        }
        std::vector<long> vertices = utils::range<long>(0, graph.num_vertices());
        shuffle_active_set(vertices);
        size_t vertex_moves = 0;
        for (int batch = 0; batch < args.batches; ++batch) {
            std::vector<Membership> membership_updates = metropolis_hastings_iteration(
                    blockmodel, graph, &next_assignment, mcmc_window, vertices, batch);
            vertex_moves += update_blockmodel(graph, blockmodel, membership_updates, &next_assignment, mcmc_window);
        }
        MCMC_RUNTIMES.push_back(MPI_Wtime() - start_t);
        double new_entropy = !args.parametric ?
                             entropy::nonparametric::mdl(blockmodel, graph) :
                             entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        double delta_entropy = new_entropy - old_entropy;
        old_entropy = new_entropy;
        delta_entropies.push_back(delta_entropy);
        if (mpi.rank == 0) {
            std::cout << "Itr: " << iteration << " vertex moves: " << vertex_moves << " delta S: "
                      << delta_entropy / new_entropy << std::endl;
        }
        total_vertex_moves += vertex_moves;
        timers::MCMC_iterations++;
        // Early stopping
        if (finetune::early_stop(iteration, blockmodel.getOverall_entropy(), delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(entropy::mdl(blockmodel, graph));
    if (mpi.rank == 0) std::cout << "Total number of vertex moves: " << total_vertex_moves << ", overall entropy: ";
    if (mpi.rank == 0) std::cout << blockmodel.getOverall_entropy() << std::endl;
    if (args.nonblocking) {
//        if (mpi.rank == 0) std::cout << "nonblocking call" << std::endl;
        utils::MPI(MPI_Win_free(&mcmc_window));
    } else {
        MPI_Type_free(&Membership_t);
    }
    return blockmodel;
}

void measure_imbalance_metrics(const TwoHopBlockmodel &blockmodel, const Graph &graph) {
    std::vector<long> degrees = graph.degrees();
    MapVector<bool> block_count;
    unsigned long num_degrees = 0;
    unsigned long long num_aggregate_block_degrees = 0;
    for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
        if (!blockmodel.owns_vertex(vertex)) continue;
        num_degrees += degrees[vertex];
        long block = blockmodel.block_assignment(vertex);
        block_count[block] = true;
        num_aggregate_block_degrees += blockmodel.degrees(block);
    }
    MCMC_VERTEX_EDGES.push_back(num_degrees);
    MCMC_NUM_BLOCKS.push_back(block_count.size());
    unsigned long block_degrees = 0;
    for (const std::pair<long, bool> &entry : block_count) {
        long block = entry.first;
        block_degrees += blockmodel.degrees(block);
    }
    MCMC_BLOCK_DEGREES.push_back(block_degrees);
    MCMC_AGGREGATE_BLOCK_DEGREES.push_back(num_aggregate_block_degrees);
}

TwoHopBlockmodel &mcmc(Graph &graph, TwoHopBlockmodel &blockmodel, bool golden_ratio_not_reached) {
    if (blockmodel.num_blocks() == 1) {
        std::cout << mpi.rank << " | number of blocks is 1 for some reason..." << std::endl;
        return blockmodel;
    }
    common::candidates = std::uniform_int_distribution<long>(0, blockmodel.num_blocks() - 2);
    if (mpi.rank == 0) std::cout << "Starting MCMC vertex moves using " << args.algorithm << std::endl;
    // MPI Datatype init
    MPI_Win mcmc_window;
    std::vector<long> next_assignment;
    if (args.nonblocking) {
//        if (mpi.rank == 0) std::cout << "nonblocking call" << std::endl;
//        std::cout << mpi.rank << " | initializing the MPI single-sided comm window" << std::endl;
        next_assignment = blockmodel.block_assignment();
        auto win_size = long(next_assignment.size() * sizeof(long));
        MPI_Win_create(next_assignment.data(), win_size, sizeof(long), MPI_INFO_NULL,
                       mpi.comm, &mcmc_window);
        assert(next_assignment.size() == (size_t) graph.num_vertices());
        utils::MPI(MPI_Win_fence(0, mcmc_window));
    } else {
        MPI_Type_create_struct(2, MEMBERSHIP_T_BLOCK_LENGTHS, MEMBERSHIP_T_DISPLACEMENTS, MEMBERSHIP_T_TYPES,
                               &Membership_t);
        MPI_Type_commit(&Membership_t);
    }
    std::vector<double> delta_entropies;
//    size_t total_vertex_moves = 0;
    double old_entropy = !args.parametric ?
                         entropy::nonparametric::mdl(blockmodel, graph) :
                         entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
    blockmodel.setOverall_entropy(old_entropy);
    double new_entropy = 0;
    for (long iteration = 0; iteration < MAX_NUM_ITERATIONS; ++iteration) {
        measure_imbalance_metrics(blockmodel, graph);
        double start_t = MPI_Wtime();
        // Block assignment used to re-create the Blockmodel after each batch to improve mixing time of
        // asynchronous Gibbs sampling
        if (args.nonblocking) {
//            if (mpi.rank == 0) std::cout << "nonblocking call" << std::endl;
            next_assignment.assign(blockmodel.block_assignment().begin(), blockmodel.block_assignment().end());
        }
        size_t vertex_moves = 0;
        if (args.algorithm == "hybrid_mcmc") {
            std::vector<long> active_set = graph.high_degree_vertices();
            shuffle_active_set(active_set);
//            std::shuffle(active_set.begin(), active_set.end(), rng::generator());
            if (mpi.rank == 0) std::cout << mpi.rank << " | starting MH portion of hybrid MCMC =========" << std::endl;
            std::vector<Membership> membership_updates = metropolis_hastings_iteration(blockmodel, graph, &next_assignment, mcmc_window, graph.high_degree_vertices(), -1);
            vertex_moves = update_blockmodel(graph, blockmodel, membership_updates, &next_assignment, mcmc_window);
            if (mpi.rank == 0) std::cout << mpi.rank << " | starting AG portion of hybrid MCMC =========" << std::endl;
            active_set = graph.low_degree_vertices();
            shuffle_active_set(active_set);
//            std::shuffle(active_set.begin(), active_set.end(), rng::generator());
            for (int batch = 0; batch < args.batches; ++batch) {
                if (mpi.rank == 0) std::cout << mpi.rank << " | starting AG batch " << batch << " portion of hybrid MCMC =========" << std::endl;
                std::vector<Membership> async_updates = asynchronous_gibbs_iteration(blockmodel, graph, &next_assignment, mcmc_window, active_set, batch);
                vertex_moves += update_blockmodel(graph, blockmodel, async_updates, &next_assignment, mcmc_window);
            }
        } else {
            std::vector<long> active_set = utils::range<long>(0, graph.num_vertices());
            shuffle_active_set(active_set);
//            std::shuffle(active_set.begin(), active_set.end(), rng::generator());
            for (int batch = 0; batch < args.batches; ++batch) {
//                if (mpi.rank == 0) std::cout << "processing batch = " << batch << std::endl;
                std::vector<Membership> membership_updates;
                if (args.algorithm == "async_gibbs")
                    membership_updates = asynchronous_gibbs_iteration(blockmodel, graph, &next_assignment, mcmc_window,
                                                                      active_set, batch);
                else
                    membership_updates = metropolis_hastings_iteration(blockmodel, graph, &next_assignment, mcmc_window, active_set, batch);
                vertex_moves += update_blockmodel(graph, blockmodel, membership_updates, &next_assignment, mcmc_window);
            }
        }
        MCMC_RUNTIMES.push_back(MPI_Wtime() - start_t);
        new_entropy = !args.parametric ?
                      entropy::nonparametric::mdl(blockmodel, graph) :
                      entropy::dist::mdl(blockmodel, graph.num_vertices(), graph.num_edges());
        double delta_entropy = new_entropy - old_entropy;
        old_entropy = new_entropy;
        delta_entropies.push_back(delta_entropy);
        if (mpi.rank == 0) {
            std::cout << "Itr: " << iteration << " vertex moves: " << vertex_moves << " delta S: "
                      << delta_entropy / new_entropy << std::endl;
        }
//        total_vertex_moves += vertex_moves;
        timers::MCMC_iterations++;
        // Early stopping
        if (early_stop(iteration, golden_ratio_not_reached, new_entropy, delta_entropies)) {
            break;
        }
    }
    blockmodel.setOverall_entropy(new_entropy);
    if (args.nonblocking) {
//        if (mpi.rank == 0) std::cout << "nonblocking call" << std::endl;
        utils::MPI(MPI_Win_free(&mcmc_window));
    } else {
        MPI_Type_free(&Membership_t);
    }
    return blockmodel;
}

std::vector<Membership> metropolis_hastings_iteration(TwoHopBlockmodel &blockmodel, Graph &graph,
                                                      std::vector<long> *next_assignment, MPI_Win mcmc_window,
                                                      const std::vector<long> &active_set, int batch) {
    std::vector<Membership> membership_updates;
    std::vector<long> vertices;
    if (active_set.empty())
        vertices = utils::range<long>(0, graph.num_vertices());
    else
        vertices = active_set;
    long batch_size = long(ceil(double(vertices.size()) / args.batches));
    long start = batch * batch_size;
    size_t end = std::min(long(vertices.size()), (batch + 1) * batch_size);
    // for hybrid_mcmc, we want to go through entire active_set in one go, regardless of number of batches
    if (batch == -1) {
        start = 0;
        end = vertices.size();
    }
    for (size_t index = start; index < end; ++index) {  // long vertex : vertices) {
        long vertex = vertices[index];
        if (!blockmodel.owns_vertex(vertex)) continue;
        VertexMove proposal = dist::propose_mh_move(blockmodel, vertex, graph);
        if (proposal.did_move) {
//            assert(proposal.proposed_block < blockmodel.num_blocks());
            assert(blockmodel.stores(proposal.proposed_block));
            remote_update_membership(vertex, proposal.proposed_block, membership_updates, next_assignment, mcmc_window);
//            membership_updates.push_back(Membership{vertex, proposal.proposed_block});
        }
    }
    return membership_updates;
}

VertexMove propose_gibbs_move(const TwoHopBlockmodel &blockmodel, long vertex, const Graph &graph) {
    bool did_move = false;
    long current_block = blockmodel.block_assignment(vertex);
//    if (blockmodel.block_size(current_block) <= args.threads) {
//        return VertexMove{0.0, did_move, -1, -1 };
//    }

    EdgeWeights out_edges = edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(vertex), vertex, true);

    utils::ProposalAndEdgeCounts proposal = common::dist::propose_new_block(
            current_block, out_edges, in_edges, blockmodel.block_assignment(), blockmodel, false);
    if (!blockmodel.stores(proposal.proposal)) {
        std::cerr << "ERROR " << "blockmodel doesn't own proposed block!!!!!" << std::endl;
        exit(-1000000000);
    }
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1 };
    }

    return eval_vertex_move(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
}

VertexMove propose_mh_move(TwoHopBlockmodel &blockmodel, long vertex, const Graph &graph) {
    bool did_move = false;
    long current_block = blockmodel.block_assignment(vertex);  // getBlock_assignment()[vertex];
    if (blockmodel.block_size(current_block) == 1) {
        return VertexMove{0.0, did_move, -1, -1 };
    }
    EdgeWeights out_edges = edge_weights(graph.out_neighbors(vertex), vertex, false);
    EdgeWeights in_edges = edge_weights(graph.in_neighbors(vertex), vertex, true);

    utils::ProposalAndEdgeCounts proposal = common::dist::propose_new_block(
            current_block, out_edges, in_edges, blockmodel.block_assignment(), blockmodel, false);
    if (!blockmodel.stores(proposal.proposal)) {
        std::cerr << "ERROR " << "blockmodel doesn't own proposed block!!!!!" << std::endl;
        exit(-1000000000);
    }
    if (proposal.proposal == current_block) {
        return VertexMove{0.0, did_move, -1, -1 };
    }

    return move_vertex(vertex, current_block, proposal, blockmodel, graph, out_edges, in_edges);
}

void remote_update_membership(long vertex, long new_block, std::vector<Membership> &membership_updates,
                              std::vector<long> *next_assignment, MPI_Win mcmc_window) {
//    std::cout << mpi.rank << " | attempting to move " << vertex << " to " << new_block << std::endl;
    if (args.nonblocking) {
//        if (mpi.rank == 0) std::cout << "nonblocking call" << std::endl;
        (*next_assignment)[vertex] = new_block;
        for (int rank = 0; rank < mpi.num_processes; ++rank) {
            if (rank == mpi.rank) continue;
//            std::cout << mpi.rank << " | attempting to update index " << vertex << " to " << (*next_assignment)[vertex] << " on rank " << rank << std::endl;
//            std::cout << "origin_addr = " << &new_block << std::endl;
//            assert(mcmc_window != nullptr);
//            std::cout << "window addr = " << &mcmc_window << std::endl;
//            assert(next_assignment != nullptr);
//            assert(next_assignment->size() > vertex);
//            MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, mcmc_window);
            utils::MPI(MPI_Put(&new_block, 1, MPI_LONG, rank, vertex, 1, MPI_LONG, mcmc_window));
//            MPI_Win_unlock(rank, mcmc_window);
        }
        return;
    }
    #pragma omp critical (updates)
    {
        membership_updates.push_back(Membership{vertex, new_block});
    }
}

void shuffle_active_set(std::vector<long> &active_set) {
    if (!args.ordered) {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(active_set.begin(), active_set.end(), std::mt19937_64(seed));
    }
}

size_t update_blockmodel(const Graph &graph, TwoHopBlockmodel &blockmodel,
                         const std::vector<Membership> &membership_updates,
                         std::vector<long> *next_assignment, MPI_Win mcmc_window) {
    if (args.nonblocking) {
//        if (mpi.rank == 0) std::cout << "nonblocking call" << std::endl;
//        std::cout << mpi.rank << " | updating blockmodel!" << std::endl;
//        std::cout << mpi.rank << " | next_assignment_addr = " << next_assignment << std::endl;
//        assert(next_assignment != nullptr);
//        assert(next_assignment->size() == graph.num_vertices());
//        assert(mcmc_window != nullptr);
//        std::cout << mpi.rank << " | window_addr = " << mcmc_window << std::endl;
        size_t vertex_moves = 0;
//        std::cout << mpi.rank << " | arrived at win_fence" << std::endl;
        utils::MPI(MPI_Win_fence(0, mcmc_window));
//        std::cout << mpi.rank << " | got through win_fence" << std::endl;
        for (long vertex = 0; vertex < graph.num_vertices(); ++vertex) {
            if ((*next_assignment)[vertex] == blockmodel.block_assignment(vertex)) continue;
//            if ((*next_assignment)[vertex] >= blockmodel.num_blocks()) {
//                std::cout << mpi.rank << " | updated assignment for " << vertex << " is " << (*next_assignment)[vertex]
//                          << " when old assignment is " << blockmodel.block_assignment(vertex) << std::endl;
//                assert((*next_assignment)[vertex] < blockmodel.num_blocks());
//            }
            if (async_move({vertex, (*next_assignment)[vertex]}, graph, blockmodel)) {
                vertex_moves++;
            }
        }
        return vertex_moves;
    }
    std::vector<Membership> collected_membership_updates = mpi_get_assignment_updates(membership_updates);
    size_t vertex_moves = 0;
    for (const Membership &membership: collected_membership_updates) {
        if (membership.vertex < 0 || membership.vertex >= graph.num_vertices()) {
            std::cout << mpi.rank << " | ERROR moving " << membership.vertex << " to block " << membership.block << std::endl;
            exit(-114);
        }
        if (membership.block == blockmodel.block_assignment(membership.vertex)) continue;
        if (async_move(membership, graph, blockmodel)) {
            vertex_moves++;
        }
    }
//    size_t vertex_moves = collected_membership_updates.size();
    return vertex_moves;
}

}
// namespace finetune::dist

#endif // SBP_DIST_FINETUNE_HPP
