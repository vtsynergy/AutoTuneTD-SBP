
#include <chrono>
//#include <execinfo.h>
//#include <fenv.h>  // break on nans or infs
#include <iostream>
#include <mpi.h>
//#include <signal.h>
#include <string>

#include "args.hpp"
#include "block_merge.hpp"
#include "blockmodel/blockmodel.hpp"
#include "distributed/dist_sbp.hpp"
#include "distributed/divide_and_conquer.hpp"
#include "entropy.hpp"
#include "evaluate.hpp"
#include "finetune.hpp"
#include "graph.hpp"
#include "mpi_data.hpp"
#include "partition.hpp"
#include "rng.hpp"
#include "sample.hpp"
#include "sbp.hpp"


//MPI_t mpi;
//Args args;

//const int NUM_VERTICES_TAG = 0;
//const int VERTICES_TAG = 1;
//const int BLOCKS_TAG = 2;

struct Partition {
    Graph graph;
    Blockmodel blockmodel;
};

void run(Partition &partition) {
    if (mpi.num_processes > 1) {
//        MPI_Barrier(mpi.comm);  // keep start - end as close as possible for all processes
//        double start = MPI_Wtime();
        partition.blockmodel = sbp::dist::stochastic_block_partition(partition.graph, args);
//        double end = MPI_Wtime();
//        if (mpi.rank == 0)
//        evaluate_partition(partition.graph, partition.blockmodel, end - start);
    } else {
//        auto start = std::chrono::steady_clock::now();
        partition.blockmodel = sbp::stochastic_block_partition(partition.graph, args);
//        auto end = std::chrono::steady_clock::now();
//        evaluate_partition(partition.graph, partition.blockmodel, std::chrono::duration<double>(end - start).count());
    }
}

int main(int argc, char* argv[]) {
    // signal(SIGABRT, handler);
    // long rank, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.num_processes);
    // std::cout << "rank: " << mpi.rank << " np: " << mpi.num_processes << std::endl;

    args = Args(argc, argv);
    rng::init_generators();

    if (mpi.rank == 0) {
        std::cout << "Number of processes = " << mpi.num_processes << std::endl;
        // std::cout << "Parsed out the arguments" << std::endl;
    }
    // TODO: figure out how to distribute the graph if it doesn't fit in memory
    Graph graph = Graph::load();
    sample::Sample subgraph;
    if (args.subgraphpartition == "snowball") {
        std::cout << "Running snowball partitioning" << std::endl;
        subgraph = sample::snowball(graph, mpi.rank, mpi.num_processes);
    } else {
        std::cout << "Running round_robin partitioning" << std::endl;
        subgraph = sample::round_robin(graph, mpi.rank, mpi.num_processes);
    }
    long num_islands = subgraph.graph.num_islands();
    std::cout << "Rank " << mpi.rank << "'s graph has " << num_islands << " island vertices." << std::endl;
    MPI_Reduce(&num_islands, &(timers::total_num_islands), 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (mpi.rank == 0) {
        std::cout << "====== Total island vertices = " << timers::total_num_islands << std::endl;
    }
    Partition partition;
    double start = MPI_Wtime();
    std::cout << "Rank " << mpi.rank << "'s graph has V = " << subgraph.graph.num_vertices() << " E = " << subgraph.graph.num_edges() << std::endl << std::flush;
    partition.graph = std::move(subgraph.graph);
    // TODO: add stopping at golden ratio
//    std::cout << "Rank " << mpi.rank << "'s graph has V = " << partition.graph.num_vertices() << " E = " << partition.graph.num_edges() << std::endl << std::flush;
//    MPI_Barrier(MPI_COMM_WORLD);
    partition.blockmodel = sbp::stochastic_block_partition(partition.graph, args, true);
    double end_blockmodeling = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    double finetune_start_t = MPI_Wtime();
    std::cout << "Rank " << mpi.rank << " took " << end_blockmodeling - start << "s to finish runtime | final B = "
              << partition.blockmodel.num_blocks() << std::endl;
    std::vector<std::vector<long>> rank_vertices;
    std::vector<std::vector<long>> rank_assignment;
    std::vector<long> local_vertices, local_assignment;
    dnc::translate_local_partition(local_vertices, local_assignment, subgraph, graph.num_vertices(),
                                   partition.blockmodel.block_assignment());
    int local_num_vertices = subgraph.graph.num_vertices();
//    MPI_Barrier(MPI_COMM_WORLD);
//    exit(-10);
    std::cout << "Rank " << mpi.rank << " done computing local information" << std::endl;

    if (mpi.rank == 0) {
        rank_vertices.push_back(local_vertices);
        rank_assignment.push_back(local_assignment);
        // Receive data from all processes
        for (int rank = 1; rank < mpi.num_processes; ++rank) {
            dnc::receive_partition(rank, rank_vertices, rank_assignment);
        }
    } else {
        // The sender
        // Send partition information to root
        std::cout << "rank " << mpi.rank << " sending info to root..." << std::endl;
        MPI_Send(&local_num_vertices, 1, MPI_INT, 0, NUM_VERTICES_TAG, MPI_COMM_WORLD);
        MPI_Send(local_vertices.data(), local_num_vertices, MPI_LONG, 0, VERTICES_TAG, MPI_COMM_WORLD);
        MPI_Send(local_assignment.data(), local_num_vertices, MPI_LONG, 0, BLOCKS_TAG, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << mpi.rank << " done sending/receiving local information" << std::endl;

    if (mpi.rank == 0) {
        long offset = 0;
        std::vector<long> combined_assignment = dnc::combine_partitions(graph, offset, rank_vertices, rank_assignment);
//        std::cout << "======= Combined Assignment ==========" << std::endl;
//        utils::print<long>(combined_assignment);
        Blockmodel blockmodel(offset, graph, 0.25, combined_assignment);
        // Make this distributed?
        blockmodel = dnc::finetune_partition(blockmodel, graph);
        double finetune_end_t = MPI_Wtime();
        timers::finetune_time = finetune_end_t - finetune_start_t;
        // only last iteration result will calculate expensive modularity
        double modularity = -1;
        if (args.modularity)
            modularity = graph.modularity(blockmodel.block_assignment());
        double mdl = blockmodel.getOverall_entropy();
        utils::save_partial_profile(-1, modularity, mdl, entropy::normalize_mdl_v1(mdl, graph),
                                    blockmodel.num_blocks());
        // Evaluate finetuned assignment
        double end = MPI_Wtime();
        dnc::evaluate_partition(graph, blockmodel, end - start);
    }
    MPI_Finalize();
}
