#include "distributed/dist_blockmodel_triplet.hpp"
#include "distributed/two_hop_blockmodel.hpp"

namespace finetune::dist {

/// Stores individual MCMC runtimes per MCMC iteration for checking runtime imbalance
extern std::vector<double> MCMC_RUNTIMES;
extern std::vector<unsigned long> MCMC_VERTEX_EDGES;
extern std::vector<long> MCMC_NUM_BLOCKS;
extern std::vector<unsigned long> MCMC_BLOCK_DEGREES;
extern std::vector<unsigned long long> MCMC_AGGREGATE_BLOCK_DEGREES;

//extern MPI_Win win;
//extern std::vector<long> block_assignment;

/// Updates `blockmodel` for one membership update contained in `membership`.
bool async_move(const Membership &membership, const Graph &graph, TwoHopBlockmodel &blockmodel);

/// Runs the Asynchronous Gibbs algorithm in a distributed fashion using MPI.
//TwoHopBlockmodel &asynchronous_gibbs(TwoHopBlockmodel &blockmodel, Graph &graph, bool golden_ratio_not_reached,
//                                     std::vector<long> *next_assignment = nullptr, MPI_Win mcmc_window = nullptr);

/// Runs one iteration of the asynchronous Gibbs algorithm in a distributed fashion using MPI.
std::vector<Membership> asynchronous_gibbs_iteration(TwoHopBlockmodel &blockmodel, const Graph &graph,
                                                     std::vector<long> *next_assignment = nullptr,
                                                     MPI_Win mcmc_window = MPI_WIN_NULL,
                                                     const std::vector<long> &active_set = std::vector<long>(),
                                                     int batch = 0);

/// If the average of the last 3 delta entropies is < threshold * initial_entropy, stop the algorithm.
bool early_stop(long iteration, bool golden_ratio_not_reached, double initial_entropy,
                std::vector<double> &delta_entropies);

/// Finetunes the partial results on a given graph.
Blockmodel &finetune_assignment(TwoHopBlockmodel &blockmodel, Graph &graph);

/// Frees the MPI window.
//inline void free_mpi_window() {
//    std::cout << mpi.rank << " freeing one-sided comm window." << std::endl;
//    MPI_Win_free(&win);
//}

/// Runs the hybrid MCMC algorithm in a distributed fashion using MPI.
//TwoHopBlockmodel &hybrid_mcmc(TwoHopBlockmodel &blockmodel, Graph &graph, bool golden_ratio_not_reached);

/// Records metrics that may be causing imbalance.
void measure_imbalance_metrics(const TwoHopBlockmodel &blockmodel, const Graph &graph);

/// Runs one of the available distributed MCMC algorithms.
//TwoHopBlockmodel &mcmc(int iteration, Graph &graph, TwoHopBlockmodel &blockmodel,
//                       DistBlockmodelTriplet &blockmodel_triplet);

/// Runs one of the available distributed MCMC algorithms.
TwoHopBlockmodel &mcmc(Graph &graph, TwoHopBlockmodel &blockmodel, bool golden_ratio_not_reached);  // DistBlockmodelTriplet &blockmodel_triplet);

/// Runs the Metropolis Hastings algorithm in a distributed fashion using MPI.
//TwoHopBlockmodel &metropolis_hastings(TwoHopBlockmodel &blockmodel, Graph &graph, bool golden_ratio_not_reached);

/// Runs one iteration of the Metropolis-Hastings algorithm. Returns the accepted vertex moves.
std::vector<Membership> metropolis_hastings_iteration(TwoHopBlockmodel &blockmodel, Graph &graph,
                                                      std::vector<long> *next_assignment = nullptr,
                                                      MPI_Win mcmc_window = MPI_WIN_NULL,
                                                      const std::vector<long> &active_set = std::vector<long>(),
                                                      int batch = -1);

/// Proposes an asynchronous Gibbs move in a distributed setting.
VertexMove propose_gibbs_move(const TwoHopBlockmodel &blockmodel, long vertex, const Graph &graph);

/// Proposes a metropolis hastings move in a distributed setting.
VertexMove propose_mh_move(TwoHopBlockmodel &blockmodel, long vertex, const Graph &graph);

void remote_update_membership(long vertex, long new_block, std::vector<Membership> &membership_updates,
                              std::vector<long> *next_assignment = nullptr, MPI_Win mcmc_window = MPI_WIN_NULL);

void shuffle_active_set(std::vector<long> &active_set);

size_t update_blockmodel(const Graph &graph, TwoHopBlockmodel &blockmodel,
                         const std::vector<Membership> &membership_updates,
                         std::vector<long> *next_assignment = nullptr, MPI_Win mcmc_window = MPI_WIN_NULL);

}  // namespace finetune::dist
