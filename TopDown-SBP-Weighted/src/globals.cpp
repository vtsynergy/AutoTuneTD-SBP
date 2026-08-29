#include "globals.hpp"

MPI_t mpi;
Args args;

namespace timers {

double BlockMerge_time = 0.0;
double BlockMerge_loop_time = 0.0;
double BlockSplit_time = 0.0;
double BlockSplit_loop_time = 0.0;
double BLOCKMODEL_BUILD_TIME = 0.0;
double Blockmodel_sort_time = 0.0;
double Blockmodel_access_time = 0.0;
double Blockmodel_update_assignment = 0.0;
double finetune_time = 0.0;
double Load_balancing_time = 0.0;
long MCMC_iterations = 0;
double MCMC_time = 0.0;
double MCMC_sequential_time = 0.0;
double MCMC_parallel_time = 0.0;
double MCMC_vertex_move_time = 0.0;
ulong MCMC_moves = 0;
double sample_extend_time = 0.0;
double sample_finetune_time = 0.0;
double sample_time = 0.0;
double total_time = 0.0;
long total_num_islands = 0;

std::vector<PartialProfile> partial_profiles;

}  // namespace timers