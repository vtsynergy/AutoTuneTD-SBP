/***
 * Stores a Graph.
 */
#ifndef SBP_GRAPH_HPP
#define SBP_GRAPH_HPP

#include <iostream>
#include <string>

#include "matrix/csr.hpp"
#include "typedefs.hpp"
#include "fs.hpp"
#include "globals.hpp"
#include "utils.hpp"

class Graph {
public:
    explicit Graph(long num_vertices) {
        this->_num_vertices = num_vertices;
        this->_num_edges = 0;
        this->_self_edges = utils::constant<bool>(num_vertices, false);
        this->_assignment = utils::constant<long>(num_vertices, -1);
        this->_out_staging.resize(num_vertices);
        this->_in_staging.resize(num_vertices);
    }
    Graph(long num_vertices, size_t reserve) {
        this->_num_vertices = num_vertices;
        this->_num_edges = 0;
        this->_self_edges = utils::constant<bool>(num_vertices, false);
        this->_assignment = utils::constant<long>(num_vertices, -1);
        this->_out_staging.resize(num_vertices);
        this->_in_staging.resize(num_vertices);
        for (long v = 0; v < num_vertices; ++v) {
            this->_out_staging[v].reserve(reserve);
            this->_in_staging[v].reserve(reserve);
        }
    }
    Graph(NeighborList &out_neighbors, NeighborList &in_neighbors, long num_vertices, long num_edges,
          const std::vector<bool> &self_edges = std::vector<bool>(),
          const std::vector<long> &assignment = std::vector<long>()) {
        this->_out_staging = out_neighbors;
        this->_in_staging = in_neighbors;
        this->_num_vertices = num_vertices;
        this->_num_edges = num_edges;
        this->_self_edges = self_edges;
        this->_assignment = assignment;
        this->build_csr();
        this->sort_vertices();
    }
    Graph() = default;
    virtual ~Graph() = default;
    /// Loads the graph from file (TSV or MTX format).
    static Graph load();
    /// Loads the graph if it's in a matrix market format.
    static Graph load_matrix_market(std::vector<std::vector<std::string>> &csv_contents);
    /// Loads the graph if it's in a text format: a list of "from to" string pairs.
    static Graph load_text(std::vector<std::vector<std::string>> &csv_contents);
    //============================================
    // GETTERS & SETTERS
    //============================================
    /// Adds an edge to the staging adjacency lists. Call build_csr() when done adding edges.
    void add_edge(long from, long to);
    /// Adds an integer-weighted edge. A weight of W is stored as W parallel unit edges.
    void add_edge(long from, long to, long weight);
    /// Builds CSR from the staging adjacency lists and frees staging memory.
    void build_csr();
    /// Returns a const reference to the assignment
    const std::vector<long> &assignment() const { return this->_assignment; }
    /// Sets the assignment vector for the given graph
    void assignment(const std::vector<long> &assignment_vector) { this->_assignment = assignment_vector; }
    /// Returns the block/community assignment of vertex `v`
    long assignment(long v) const { return this->_assignment[v]; }
    /// Sets the assignment of vertex `v` to block `b`
    void assign(long v, long b) { this->_assignment[v] = b; }
    /// Returns the degree of a given vertex `v`
    long degree(size_t v) const;
    /// Returns a vector containing the vertex degrees for every vertex in the graph
    std::vector<long> degrees() const;
    /// Returns a NeighborView of the in-neighbors of vertex `v`
    NeighborView in_neighbors(long v) const {
        return args.csrgraph ? this->_in_csr.neighbors(v)
                             : NeighborView(this->_in_staging[v].data(), (long)this->_in_staging[v].size());
    }
    /// Returns the list of high degree vertices
    const std::vector<long> &high_degree_vertices() const { return this->_high_degree_vertices; }
    /// Returns the list of low degree vertices
    const std::vector<long> &low_degree_vertices() const { return this->_low_degree_vertices; }
    /// Calculates the modularity of this graph given a particular vertex-to-block `assignment`
    double modularity(const std::vector<long> &assignment) const;
    /// Returns all the neighbors of a given vertex. Note that vertices that are both in- and out- neighbors are
    /// repeated.
    [[nodiscard]] std::vector<long> neighbors(long vertex) const;
    /// Returns the number of edges in this graph
    virtual long num_edges() const { return this->_num_edges; }
    /// Counts the number of island vertices in this graph
    long num_islands() const;
    /// Returns the number of vertices in this graph
    long num_vertices() const { return this->_num_vertices; }
    /// Returns a NeighborView of the out-neighbors of vertex `v`
    NeighborView out_neighbors(long v) const {
        return args.csrgraph ? this->_out_csr.neighbors(v)
                             : NeighborView(this->_out_staging[v].data(), (long)this->_out_staging[v].size());
    }
    /// Returns a const reference to the out-adjacency CSR (GPU-mappable).
    const CSR& out_csr() const { return this->_out_csr; }
    /// Returns a const reference to the in-adjacency CSR (GPU-mappable).
    const CSR& in_csr() const { return this->_in_csr; }
    /// Sorts vertices into low/high degree lists. build_csr() must be called first.
    void sort_vertices();
    /// Returns a list of edges, sorted by degree product
    [[nodiscard]] std::vector<std::pair<std::pair<long, long>, long>> sorted_edge_list() const;
    /// Sorts vertices into low and high influence vertices via vertex degree products.
    void degree_product_sort();
protected:
    /// For every vertex, stores the community they belong to.
    /// If assignment[v] = -1, then the community of v is not known
    std::vector<long> _assignment;
    /// Stores a list of the high degree vertices
    std::vector<long> _high_degree_vertices;
    /// Stores a list of the low degree vertices
    std::vector<long> _low_degree_vertices;
    /// Temporary adjacency lists used during incremental construction (add_edge).
    /// Freed after build_csr() is called.
    NeighborList _out_staging;
    NeighborList _in_staging;
    /// CSR adjacency (out- and in-edges). Built by build_csr(). GPU-mappable.
    CSR _out_csr;
    CSR _in_csr;
    /// The number of vertices in the graph
    long _num_vertices = 0;
    /// The number of edges in the graph
    long _num_edges = 0;
    /// Stores true if a vertex has self edges, false otherwise
    std::vector<bool> _self_edges;
    /// Parses a directed graph from csv contents
    static void parse_directed(NeighborList &in_neighbors, NeighborList &out_neighbors, long &num_vertices,
                               std::vector<bool> &self_edges, std::vector<std::vector<std::string>> &contents);
    /// Parses an undirected graph from csv contents
    static void parse_undirected(NeighborList &in_neighbors, NeighborList &out_neighbors, long &num_vertices,
                                 std::vector<bool> &self_edges, std::vector<std::vector<std::string>> &contents);
};

#endif // SBP_GRAPH_HPP
