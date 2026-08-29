/**
* Globally accessible variables
*/
#ifndef SBP_GLOBALS_HPP
#define SBP_GLOBALS_HPP

#include "args.hpp"
#include "mpi_data.hpp"
#include "typedefs.hpp"

//namespace globals {

/// MPI cluster information
extern MPI_t mpi;

/// User-passed arguments
extern Args args;

/// Returns true when the blockmodel is currently using the dense matrix backend, in which case
/// consumer code should take the dense compute path (operate on dense getrow/getcol vectors
/// directly instead of materializing MapVector slices). Reads args.matrix_type live so it tracks
/// the temporary backend override during split scoring (see split_communities).
inline bool dense_compute() { return args.matrix_type == "dense"; }

namespace timers {

/// The total amount of time spent performing block merges, to be dynamically updated during execution.
extern double BlockMerge_time;
/// The total amount of time spent in the main parallelizable loop of the block merge iterations, to be dynamically
/// updated during execution.
extern double BlockMerge_loop_time;
/// The time taken to sort potential block merges.
extern double BlockMerge_sort_time;
/// The time taken to split blocks.
extern double BlockSplit_time;
/// The total amount of time spent in the main parallelizable loop of the block merge iterations, to be dynamically
/// updated during execution.
extern double BlockSplit_loop_time;
/// The time taken to build the blockmodel.
extern double BLOCKMODEL_BUILD_TIME;
/// The time taken to sort the vertices in the blockmodel.
extern double Blockmodel_sort_time;
/// The time taken to access the blockmodel.
extern double Blockmodel_access_time;
/// The time taken to update the blockmodel assignment.
extern double Blockmodel_update_assignment;
/// The time taken to finetune the partition.
extern double finetune_time;
/// The time taken to perform load balancing.
extern double Load_balancing_time;
/// The total number of MCMC iterations completed, to be dynamically updated during execution.
extern long MCMC_iterations;
/// The total amount of time spent performing MCMC iterations, to be dynamically updated during execution.
extern double MCMC_time;
/// The total amount of time spent in the main parallelizable loop of the MCMC iterations, to by dynamically
/// updated during execution.
extern double MCMC_sequential_time, MCMC_parallel_time, MCMC_vertex_move_time;
/// The number of MCMC moves performed.
extern ulong MCMC_moves;
/// The time taken to extend partial results from the sample to the full graph.
extern double sample_extend_time;
/// The time taken to finetune the clustering results with sampling.
extern double sample_finetune_time;
/// The time taken to perform sampling.
extern double sample_time;
/// The total amount of time spent community detection, to be dynamically updated during execution.
extern double total_time;
/// The total number of island vertices (across all MPI ranks, if applicable)
extern long total_num_islands;

extern std::vector<PartialProfile> partial_profiles;

}  // namespace timers
//}

#endif // SBP_GLOBALS_HPP