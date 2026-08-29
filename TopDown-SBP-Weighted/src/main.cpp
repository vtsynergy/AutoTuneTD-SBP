
//#include <chrono>
//#include <execinfo.h>
//#include <fenv.h>  // break on nan or infinity
#include <iostream>
#include <mpi.h>
//#include <signal.h>
#include <string>

//#include <nlohmann/json.hpp>

#include "args.hpp"
#include "blockmodel/blockmodel.hpp"
#include "distributed/dist_sbp.hpp"
#include "distributed/dist_finetune.hpp"
#include "entropy.hpp"
#include "evaluate.hpp"
#include "finetune.hpp"
#include "globals.hpp"
#include "graph.hpp"
//#include "mpi_data.hpp"
#include "partition.hpp"
#include "rng.hpp"
#include "sample.hpp"
#include "sbp.hpp"
//#include "typedefs.hpp"
#include "utils.hpp"

//MPI_t mpi;
//Args args;

//double sample_time = 0.0;
//double sample_extend_time = 0.0;
//double finetune_time = 0.0;

struct Partition {
    Graph graph;
    Blockmodel blockmodel;
};

void evaluate_partition(Graph &graph, Blockmodel &blockmodel, double runtime) {
    if (mpi.rank != 0) return;
    utils::write_json(blockmodel.block_assignment(), blockmodel.getOverall_entropy(),
                      timers::MCMC_moves,timers::MCMC_iterations, runtime);
    if (!args.evaluate) return;
    evaluate::Eval result = evaluate::evaluate_blockmodel(graph, blockmodel);
    std::cout << "Final F1 score = " << result.f1_score << std::endl;
    std::cout << "Community detection runtime = " << runtime << "s" << std::endl;
    if (std::isnan(result.nmi) || std::isinf(result.nmi)) {
        result.nmi = 0.00;
    }
    evaluate::write_results(graph, result, runtime);
}

void run(Partition &partition) {
    timers::total_num_islands = partition.graph.num_islands();
    if (mpi.num_processes > 1) {
        partition.blockmodel = sbp::dist::stochastic_block_partition(partition.graph, args);
    } else {
        partition.blockmodel = sbp::stochastic_block_partition(partition.graph, args);
    }
}

int main(int argc, char* argv[]) {
    // signal(SIGABRT, handler);
    // long rank, num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Comm_rank(mpi.comm, &mpi.rank);
    MPI_Comm_size(mpi.comm, &mpi.num_processes);

    // std::cout << "rank: " << mpi.rank << " np: " << mpi.num_processes << std::endl;

    args = Args(argc, argv);
    rng::init_generators();

    if (mpi.rank == 0) {
        std::cout << "Number of processes = " << mpi.num_processes << std::endl;
        // std::cout << "Parsed out the arguments" << std::endl;
    }
    // TODO: figure out how to distribute the graph if it doesn't fit in memory
    Graph graph = Graph::load();
    sample::Sample detached;
    Partition partition;
    double start = MPI_Wtime();
    if (args.detach) {  // if we're getting rid of vertices with degree < 2
        detached = sample::detach(graph);
        partition.graph = std::move(detached.graph);
        std::cout << "detached num vertices: " << partition.graph.num_vertices() << " E: "
                  << partition.graph.num_edges() << std::endl;
    } else {
        partition.graph = std::move(graph);
    }
    if (args.samplesize <= 0.0) {
        std::cerr << "ERROR " << "Sample size of " << args.samplesize << " is too low. Must be greater than 0.0" << std::endl;
        exit(-5);
    }
    if (args.samplesize < 1.0) {
        double sample_start_t = MPI_Wtime();
        std::cout << "Running sampling with size: " << args.samplesize << std::endl;
//        sample::Sample s = sample::max_degree(partition.graph);
        sample::Sample s = sample::sample(partition.graph);
        if (mpi.num_processes > 1) {
            MPI_Bcast(s.mapping.data(), (int) partition.graph.num_vertices(), MPI_LONG, 0, mpi.comm);
            if (mpi.rank > 0) {
                std::vector<long> vertices;
                for (const long &mapped_id : s.mapping) {
                    if (mapped_id >= 0)
                        vertices.push_back(mapped_id);
                }
                s = sample::from_vertices(partition.graph, vertices, s.mapping);
            }
            MPI_Barrier(mpi.comm);
        }
        Partition sample_partition;
        sample_partition.graph = std::move(s.graph);  // s.graph may be empty now
        // add timer
        double sample_end_t = MPI_Wtime();
        timers::sample_time = sample_end_t - sample_start_t;
        run(sample_partition);
        double extend_start_t = MPI_Wtime();
        s.graph = std::move(sample_partition.graph);  // refill s.graph
        // extend sample to full graph
        // TODO: this seems deterministic...
        std::vector<long> assignment = sample::extend(partition.graph, sample_partition.blockmodel, s);
        // fine-tune full graph
        double finetune_start_t = MPI_Wtime();
        if (mpi.num_processes > 1) {
            MPI_Bcast(assignment.data(), (int) assignment.size(), MPI_LONG, 0, mpi.comm);
            // if (mpi.rank > 0) {
            //     std::vector<long> vertices;
            //    for (const long &mapped_id : s.mapping) {
            //        if (mapped_id >= 0)
            //            vertices.push_back(mapped_id);
            //    }
            //    s = sample::from_vertices(partition.graph, vertices, s.mapping);
            //}
            // MPI_Barrier(mpi.comm);
            Rank_indices = std::vector<long>();  // reset the rank_indices
            auto blockmodel = TwoHopBlockmodel(sample_partition.blockmodel.num_blocks(), partition.graph, 0.5, assignment);
            partition.blockmodel = finetune::dist::finetune_assignment(blockmodel, partition.graph);
        } else {
            partition.blockmodel = Blockmodel(sample_partition.blockmodel.num_blocks(), partition.graph, 0.5, assignment);
            partition.blockmodel = finetune::finetune_assignment(partition.blockmodel, partition.graph);
        }
        double finetune_end_t = MPI_Wtime();
        timers::sample_extend_time = finetune_start_t - extend_start_t;
        timers::sample_finetune_time = finetune_end_t - finetune_start_t;
    } else {
        std::cout << "Running without sampling." << std::endl;
        run(partition);
    }
    if (args.detach) {
        std::cout << "Reattaching island and 1-degree vertices" << std::endl;
        partition.blockmodel = sample::reattach(graph, partition.blockmodel, detached);
    } else {
        graph = std::move(partition.graph);
    }
    // evaluate
    double end = MPI_Wtime();
    evaluate_partition(graph, partition.blockmodel, end - start);
//    finetune::dist::free_mpi_window();
    MPI_Finalize();
}
