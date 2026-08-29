/// A wrapper around tclap to make the use of command-line arguments bearable.
#ifndef TCLAP_WRAPPER_ARGS
#define TCLAP_WRAPPER_ARGS

#include <iostream>
#include <limits>
#include <omp.h>
#include <string>
#include <unistd.h>

#include "tclap/CmdLine.h"

// TODO: add a verbose argument
class Args {
public:  // Everything in here is public, because why not?
    /** Define all the arguments here for easy access */
    std::string algorithm;
    bool approximate;
    int asynciterations;
    int batches;
    std::string blocksizevar;
    size_t cachesize;
    std::string csv;  // TODO: get rid of this - results now saved to json
    bool degreecorrected;
    bool vertex_degree_sort;
    std::string delimiter;
    bool detach;
    std::string distribute;
    std::string directory;
    bool evaluate;
    std::string filepath;
    bool hastings_correction;
    std::string json;
    float mh_percent;
    bool mix;
    bool modularity;
    bool nodelta;  // TODO: if delta is much faster, get rid of this and associated methods.
    bool noduplicates;
    bool nonblocking;
    bool parametric;
    int numvertices;
    bool ordered;
    std::string output_file;
    std::string overlap;
    double samplesize;
    std::string samplingalg;
    std::string split;
    std::string splitinit;
    float splitrate;
    int subgraphs;
    std::string subgraphpartition;
    std::string tag;
    int threads;
    bool coodelta;
    bool csrgraph;
    std::string matrix_type;
    std::string type;
    bool undirected;

    /// Default Args constructor.
    Args() {}

    /// Parses the command-line options and stores the results in an easy-to-retrieve manner.
    Args(int argc, char** argv) {
        /** Use tclap to retrieve the arguments here */
        try {
            TCLAP::CmdLine parser("Stochastic block blockmodeling algorithm", ' ', "alpha.1.1");
            TCLAP::ValueArg<std::string> _algorithm("a", "algorithm", "The algorithm to use for the finetuning/MCMC "
                                                    "step of stochastic block blockmodeling. Note: there is currently no "
                                                    "parallel implementation of metropolis hastings", false,
                                                    "hybrid_mcmc", "async_gibbs|metropolis_hastings|hybrid_mcmc", parser);
            TCLAP::SwitchArg _approximate("", "approximate", "If set, an approximate version of the block merge "
                                          "step will be used. It's slightly faster, but less accurate for complex "
                                          "graphs.", parser, false);
            TCLAP::ValueArg<int> _async_iterations("", "asynciterations", "The number of asynchronous iterations to "
                                                   "run before switching to metropolis hastings.", false,
                                                   std::numeric_limits<int>::max(), "[1, infinity]", parser);
            TCLAP::ValueArg<int> _batches("", "batches", "The number of batches to use for the asynchronous_gibbs "
                                          "algorithm. Too many batches will lead to many updates and little parallelism,"
                                          " but too few will lead to poor results or more iterations", false, 2, "int",
                                          parser);
            TCLAP::ValueArg<std::string> _blocksizevar("b", "blocksizevar", "The variation between the sizes of "
                                                       "communities", false, "low", "low|high|unk", parser);
            TCLAP::ValueArg<size_t> _cachesize("", "cachesize", "The size of the log cache", false, 20000, ">= 1", parser);
            TCLAP::ValueArg<std::string> _csv("c", "csv",
                                              "The path to the csv file in which the results will be stored, "
                                              "without the suffix, e.g.:\n"
                                              "if --csv=eval/test, results will be stored in eval/test.csv.",
                                              false, "./eval/test", "path", parser);
            TCLAP::SwitchArg _degreecorrected("", "degreecorrected", "If set, will compute the degree-corrected description length.",
                                              parser, false);
            TCLAP::SwitchArg _vertex_degree_sort("", "vertex-degree-sort", "If set, will use vertex degree instead of edge degree product "
                                                "to identify high-influence vertices (less accurate but faster).", parser, false);
            TCLAP::ValueArg<std::string> _delimiter("", "delimiter", "The delimiter used in the file storing the graph",
                                                    false, "\t", "string, usually `\\t` or `,`", parser);
            TCLAP::SwitchArg _detach("", "detach", "If set, will detach 1-degree vertices before running"
                                     "community detection.", parser, false);
            TCLAP::ValueArg<std::string> _distribute("", "distribute", "The distribution scheme to use. Default = "
                                                     "none", false, "none", "none | 2hop-round-robin "
                                                     "| 2hop-size-balanced | 2hop-snowball", parser);
            TCLAP::ValueArg<std::string> _directory("d", "directory",
                                "The directory in which the graph is stored. The following structure is assumed:\n"
                                "filename for graph:"
                                "<type>_<overlap>Overlap_<blocksizevar>BlockSizeVar_<numvertices>_nodes.tsv\n"
                                "filename for truth:"
                                "<type>_<overlap>Overlap_<blocksizevar>BlockSizeVar_<numvertices>_trueBlockmodel.tsv\n"
                                "directory structure:"
                                "<directory>/<type>/<overlap>Overlap_<blocksizevar>BlockSizeVar/<filename>\n",
                                                   false, "./data", "path", parser);
            TCLAP::SwitchArg _evaluate("", "evaluate", "If set, will evaluate the results before exiting",
                                       parser, false);
            TCLAP::ValueArg<std::string> _filepath("f", "filepath", "The filepath for the graph, minus the extension.",
                                                   true, "./data/default_graph", "path", parser);
            TCLAP::SwitchArg _hastings_correction("", "hastings-correction", "If set, will compute Hastings correction for more accurate MCMC (slower, only useful in parametric mode)",
                                     parser, false);
            TCLAP::ValueArg<std::string> _json("j", "json", "The path to the directory containing json output",
                                               false, "output", "path", parser);
            TCLAP::ValueArg<float> _mh_percent("m", "mh_percent", "The percentage of vertices to process sequentially if alg==hybrid_mcmc",
                                               false, 0.1, "float", parser);
            TCLAP::SwitchArg _mix("", "mix", "If set, will run block merges after golden ratio is found", parser, false);
            TCLAP::SwitchArg _modularity("", "modularity", "If set, will compute modularity at the end of execution.",
                                         parser, false);
            TCLAP::SwitchArg _nodelta("", "nodelta", "If set, do not use the blockmodel deltas for "
                                      "entropy calculations.", parser, false);
            TCLAP::SwitchArg _noduplicates("", "noduplicates", "If set, will check for duplicate edges in the graph "
                                           "and ensure that they're not inserted twice. Otherwise, the graph needs to "
                                           "be manually checked to ensure that it's not a multigraph.", parser, false);
            TCLAP::SwitchArg _nonblocking("", "nonblocking", "If set, will use MPI nonblocking single-sided communication in the MCMC phase.", parser, false);
            TCLAP::SwitchArg _parametric("", "parametric", "If set, will use the parametric blockmodel entropy computations instead of nonparametric (disables default behavior).",
                                            parser, false);
            TCLAP::ValueArg<int> _numvertices("n", "numvertices", "The number of vertices in the graph", false, 1000,
                                              "int", parser);
            TCLAP::SwitchArg _ordered("", "ordered", "If set, will loop through vertices in an ordered fashion", parser, false);
            TCLAP::ValueArg<std::string> _output_file("", "output_file", "The filename of the json output. Will be stored in <json>/<output_file>",
                                                      false, "", "string that ends in .json", parser);
            TCLAP::ValueArg<std::string> _overlap("o", "overlap", "The degree of overlap between communities", false,
                                                  "low", "low|high|unk", parser);
            TCLAP::ValueArg<double> _samplesize("", "samplesize", "The percentage of vertices to include in the sample",
                                               false, 1.0, "0 < x <= 1.0", parser);
            TCLAP::ValueArg<std::string> _samplingalg("", "samplingalg", "The sampling algorithm to use, if --samplesize < 1.0",
                                                      false, "random", "random|max_degree|expansion_snowball", parser);
            TCLAP::ValueArg<std::string> _split("", "split", "The type of split to use in TopDownSBP", false, "connectivity-snowball",
                                                "random|snowball|single-snowball|connectivity-snowball", parser);
            TCLAP::ValueArg<std::string> _splitinit("", "splitinit", "The type of split initialization to use", false, "random",
                                                    "random|degree-weighted|high-degree", parser);
            TCLAP::ValueArg<float> _splitrate("", "splitrate", "Growth factor for the target block count during the "
                                              "TopDownSBP initial expansion phase. Each iteration targets "
                                              "ceil(current_blocks * splitrate) blocks. Smaller values (e.g. 1.1) "
                                              "make finer-grained splits; larger values (e.g. 1.5) are coarser.",
                                              false, 1.25, "> 1.0", parser);
            TCLAP::ValueArg<int> _subgraphs("", "subgraphs", "If running divide and conquer SBP, the number of subgraphs"
                                            "to partition the data into. Must be <= number of MPI ranks. If <= 1, set to number of MPI ranks",
                                            false, 0, "<= number of MPI ranks>", parser);
            TCLAP::ValueArg<std::string> _subgraphpartition("", "subgraphpartition", "The algorithm used to create subgraphs when using the divide-and-conquer scheme",
                                                            false, "round_robin", "round_robin|snowball", parser);
            TCLAP::ValueArg<std::string> _tag("", "tag", "The tag value for this run, for differentiating different "
                                              "runs or adding custom parameters to the save file", false, "default tag",
                                              "string or param1=value1;param2=value2", parser);
            TCLAP::ValueArg<int> _threads("", "threads", "The number of OpenMP threads to use. If less than 1, will set "
                                          "number of threads to number of logical CPU cores", false, 1, "int", parser);
            TCLAP::ValueArg<std::string> _matrix_type("", "matrix_type", "Matrix storage type for blockmodel",
                                                      false, "sparse_transpose", "dense|sparse|sparse_transpose", parser);
            TCLAP::ValueArg<std::string> _type("t", "type", "The type of streaming/name of the graph", false, "static",
                                               "string", parser);
            TCLAP::SwitchArg _coodelta("", "coodelta", "If set, use sorted COO (coordinate) storage for blockmodel "
                                      "deltas instead of the default hash-map storage. The COO arrays are GPU-mappable "
                                      "via omp target map.", parser, false);
            TCLAP::SwitchArg _csrgraph("", "csrgraph", "If set, use CSR adjacency for the graph. "
                                      "Otherwise use the neighbor-list (vector-of-vectors) adjacency.", parser, false);
            TCLAP::SwitchArg _undirected("", "undirected", "If set, graph will be treated as undirected", parser,
                                         false);
            parser.parse(argc, argv);
            this->algorithm = _algorithm.getValue();
            this->approximate = _approximate.getValue();
            this->asynciterations = _async_iterations.getValue();
            this->batches = _batches.getValue();
            this->blocksizevar = _blocksizevar.getValue();
            this->cachesize = _cachesize.getValue();
            this->csv = _csv.getValue();
            this->degreecorrected = _degreecorrected.getValue();
            this->vertex_degree_sort = _vertex_degree_sort.getValue();
            this->delimiter = _delimiter.getValue();
            this->detach = _detach.getValue();
            this->distribute = _distribute.getValue();
            this->directory = _directory.getValue();
            this->evaluate = _evaluate.getValue();
            this->filepath = _filepath.getValue();
            this->hastings_correction = _hastings_correction.getValue();
            this->json = _json.getValue();
            this->mh_percent = _mh_percent.getValue();
            this->mix = _mix.getValue();
            this->modularity = _modularity.getValue();
            this->nodelta = _nodelta.getValue();
            this->noduplicates = _noduplicates.getValue();
            this->nonblocking = _nonblocking.getValue();
            this->parametric = _parametric.getValue();
            this->numvertices = _numvertices.getValue();
            this->ordered = _ordered.getValue();
            this->output_file = _output_file.getValue();
            if (this->output_file.empty()) {
                std::ostringstream output_file_stream;
                output_file_stream << getpid() << "_" << time(nullptr) << ".json";
                this->output_file = output_file_stream.str();
            }
            this->overlap = _overlap.getValue();
            this->samplesize = _samplesize.getValue();
            this->samplingalg = _samplingalg.getValue();
            this->split = _split.getValue();
            this->splitinit = _splitinit.getValue();
            this->splitrate = _splitrate.getValue();
            this->subgraphs = _subgraphs.getValue();
            this->subgraphpartition = _subgraphpartition.getValue();
            this->tag = _tag.getValue();
            this->threads = _threads.getValue();
            this->coodelta = _coodelta.getValue();
            this->csrgraph = _csrgraph.getValue();
            this->matrix_type = _matrix_type.getValue();
            this->type = _type.getValue();
            this->undirected = _undirected.getValue();
            if (!this->parametric) {
                std::cout << "NOTE: using nonparametric entropy, Hastings correction disabled (greedy mode)" << std::endl;
                this->hastings_correction = false;
            }
        } catch (TCLAP::ArgException &exception) {
            std::cerr << "ERROR " << "ERROR: " << exception.error() << " for argument " << exception.argId() << std::endl;
            exit(-1);
        }
    }
};

//extern Args args;

#endif // TCLAP_WRAPPER_ARGS
