#include "graph.hpp"

#include <stdexcept>

#include "mpi.h"

#include "globals.hpp"
#include "utils.hpp"
#include "mpi_data.hpp"

namespace {

long parse_edge_weight(const std::vector<std::string> &edge) {
    if (edge.size() < 3) {
        return 1;
    }

    long weight = 0;
    try {
        size_t parsed_chars = 0;
        weight = std::stol(edge[2], &parsed_chars);
        if (parsed_chars != edge[2].size()) {
            throw std::invalid_argument("trailing characters");
        }
    } catch (const std::exception &) {
        std::cerr << "ERROR invalid integer edge weight: " << edge[2] << std::endl;
        exit(-1);
    }
    if (weight <= 0) {
        std::cerr << "ERROR edge weight must be a positive integer: " << edge[2] << std::endl;
        exit(-1);
    }
    return weight;
}

void insert_weighted(NeighborList &neighbors, long from, long to, long weight) {
    if (from >= (long)neighbors.size()) {
        std::vector<std::vector<long>> padding(from - neighbors.size() + 1, std::vector<long>());
        neighbors.insert(neighbors.end(), padding.begin(), padding.end());
    }
    auto &row = neighbors[from];
    row.insert(row.end(), size_t(weight), to);
}

long count_self_edges(NeighborView neighbors, long vertex) {
    long self_edges = 0;
    for (long neighbor : neighbors) {
        if (neighbor == vertex) {
            self_edges++;
        }
    }
    return self_edges;
}

long count_edges(const NeighborList &out_neighbors, bool undirected) {
    long num_edges = 0;
    for (size_t source = 0; source < out_neighbors.size(); ++source) {
        for (long dest : out_neighbors[source]) {
            if (!undirected || source <= (size_t)dest) {
                num_edges++;
            }
        }
    }
    return num_edges;
}

}  // namespace

void Graph::add_edge(long from, long to) {
    utils::insert(this->_out_staging, from, to);
    utils::insert(this->_in_staging, to, from);
    this->_num_edges++;
    if (from == to) {
        this->_self_edges[from] = true;
    }
    // TODO: undirected version?
}

void Graph::add_edge(long from, long to, long weight) {
    if (weight <= 0) {
        std::cerr << "ERROR edge weight must be a positive integer: " << weight << std::endl;
        exit(-1);
    }
    insert_weighted(this->_out_staging, from, to, weight);
    insert_weighted(this->_in_staging, to, from, weight);
    this->_num_edges += weight;
    if (from == to) {
        this->_self_edges[from] = true;
    }
}

void Graph::build_csr() {
    if (!args.csrgraph) return;  // NL mode: keep staging as the permanent store
    this->_out_csr = CSR(this->_out_staging, this->_num_vertices, this->_num_edges);
    this->_in_csr  = CSR(this->_in_staging,  this->_num_vertices, this->_num_edges);
    this->_out_staging.clear();
    this->_out_staging.shrink_to_fit();
    this->_in_staging.clear();
    this->_in_staging.shrink_to_fit();
}

long Graph::degree(size_t v) const {
    NeighborView out = this->out_neighbors((long)v);
    return out.size() + this->in_neighbors((long)v).size() - count_self_edges(out, (long)v);
}

std::vector<long> Graph::degrees() const {
    std::vector<long> vertex_degrees;
    vertex_degrees.reserve(this->_num_vertices);
    for (long v = 0; v < this->_num_vertices; ++v) {
        vertex_degrees.push_back(this->degree(v));
    }
    return vertex_degrees;
}

Graph Graph::load() {
    // TODO: Add capability to process multiple "streaming" graph parts
    std::string base_path = utils::build_filepath();
    fs::path graph_path = base_path + ".tsv";
    fs::path truth_path = base_path + "_truePartition.tsv";
    std::vector<std::vector<std::string>> csv_contents = utils::read_csv(graph_path);
    if (csv_contents.empty()) {
        graph_path = base_path + ".mtx";
        csv_contents = utils::read_csv(graph_path);
    }
    Graph graph;
    if (csv_contents[0][0] == "%%MatrixMarket") {
        graph = Graph::load_matrix_market(csv_contents);
    } else {
        graph = Graph::load_text(csv_contents);
    }
    if (mpi.rank == 0)
        std::cout << "V: " << graph.num_vertices() << " E: " << graph.num_edges() << std::endl;

    csv_contents = utils::read_csv(truth_path);
    std::vector<long> assignment;
    // TODO: vertices, communities should be size_t or ulong. Will need to make sure -1 returns are properly handled
    // elsewhere.
    if (!csv_contents.empty()) {
        for (std::vector<std::string> &assign: csv_contents) {
            long vertex = std::stoi(assign[0]) - 1;
            long community = std::stoi(assign[1]) - 1;
            if (vertex >= (long)assignment.size()) {
                std::vector<long> padding(vertex - assignment.size() + 1, -1);
                assignment.insert(assignment.end(), padding.begin(), padding.end());
            }
            assignment[vertex] = community;
        }
    } else {
        assignment = utils::constant<long>(graph.num_vertices(), 0);
    }
    graph.assignment(assignment);
    return graph;
}

/// Loads the graph if it's in a matrix market format.
Graph Graph::load_matrix_market(std::vector<std::vector<std::string>> &csv_contents) {
    if (csv_contents[0][2] != "coordinate") {
        std::cerr << "ERROR " << "Dense matrices are not supported!" << std::endl;
        exit(-1);
    }
    if (csv_contents[0][4] == "symmetric") {
        std::cout << "Graph is symmetric" << std::endl;
        args.undirected = true;
    }
    // Find index at which edges start
    ulong index = 0;
    long num_vertices = 0;
    long declared_entries = 0;
    for (ulong i = 0; i < csv_contents.size(); ++i) {
        const std::vector<std::string> &line = csv_contents[i];
        if (line[0][0] == '%') continue;
        num_vertices = std::stoi(line[0]);
        if (num_vertices != std::stoi(line[1])) {
            std::cerr << "ERROR " << "Rectangular matrices are not supported!" << std::endl;
            exit(-1);
        }
        declared_entries = std::stoi(line[2]);
        index = i + 1;
        break;
    }
    NeighborList out_neighbors;
    NeighborList in_neighbors;
    std::vector<bool> self_edges = utils::constant<bool>(num_vertices, false);
    long num_edges = 0;
    for (ulong i = index; i < csv_contents.size(); ++i) {
        const std::vector<std::string> &edge = csv_contents[i];
        long from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        long to = std::stoi(edge[1]) - 1;
        long weight = parse_edge_weight(edge);
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        while (self_edges.size() < (size_t)num_vertices) {
            self_edges.push_back(false);
        }
        insert_weighted(out_neighbors, from, to, weight);
        insert_weighted(in_neighbors, to, from, weight);
        num_edges += weight;
        if (args.undirected && from != to) {  // Force symmetric graph to be directed by including reverse edges.
            insert_weighted(out_neighbors, to, from, weight);
            insert_weighted(in_neighbors, from, to, weight);
            num_edges += weight;
        }
        if (from == to) {
            self_edges[from] = true;
        }
    }
    if (mpi.rank == 0 && declared_entries != (long)(csv_contents.size() - index)) {
        std::cout << "WARNING MatrixMarket declared " << declared_entries << " entries but read "
                  << csv_contents.size() - index << " entries." << std::endl;
    }
    // Pad the neighbors lists
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<long>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<long>());
    }
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges);
}

/// Loads the graph if it's in a text format: a list of "from to" string pairs.
Graph Graph::load_text(std::vector<std::vector<std::string>> &csv_contents) {
    NeighborList out_neighbors;
    NeighborList in_neighbors;
    std::vector<bool> self_edges;
    long num_vertices = 0;
    if (args.undirected)
        Graph::parse_undirected(in_neighbors, out_neighbors, num_vertices, self_edges, csv_contents);
    else
        Graph::parse_directed(in_neighbors, out_neighbors, num_vertices, self_edges, csv_contents);
    long num_edges = count_edges(out_neighbors, args.undirected);
    return Graph(out_neighbors, in_neighbors, num_vertices, num_edges, self_edges);
}

double Graph::modularity(const std::vector<long> &assignment) const {
    // See equation for Q_d in: https://hal.archives-ouvertes.fr/hal-01231784/document
    double result = 0.0;
    for (long vertex_i = 0; vertex_i < this->_num_vertices; ++vertex_i) {
        for (long vertex_j = 0; vertex_j < this->_num_vertices; ++vertex_j) {
            if (assignment[vertex_i] != assignment[vertex_j]) continue;
            long edge_weight = 0;
            for (const long neighbor : this->out_neighbors(vertex_i)) {
                if (neighbor == vertex_j) {
                    edge_weight = 1;
                    break;
                }
            }
            long deg_out_i = (long)this->out_neighbors(vertex_i).size();
            long deg_in_j  = (long)this->in_neighbors(vertex_j).size();
            double temp = edge_weight - (double(deg_out_i * deg_in_j) / double(this->_num_edges));
            result += temp;
        }
    }
    result /= double(this->_num_edges);
    return result;
}

std::vector<long> Graph::neighbors(long vertex) const {
    std::vector<long> all_neighbors;
    for (const long out_neighbor : this->out_neighbors(vertex)) {
        all_neighbors.push_back(out_neighbor);
    }
    for (const long in_neighbor : this->in_neighbors(vertex)) {
        all_neighbors.push_back(in_neighbor);
    }
    return all_neighbors;
}

void Graph::parse_directed(NeighborList &in_neighbors, NeighborList &out_neighbors, long &num_vertices,
                           std::vector<bool> &self_edges, std::vector<std::vector<std::string>> &contents) {
    for (std::vector<std::string> &edge : contents) {
        long from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        long to = std::stoi(edge[1]) - 1;
        long weight = parse_edge_weight(edge);
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        insert_weighted(out_neighbors, from, to, weight);
        insert_weighted(in_neighbors, to, from, weight);
        while (self_edges.size() < (size_t) num_vertices) {
            self_edges.push_back(false);
        }
        if (from == to) {
            self_edges[from] = true;
        }
    }
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<long>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<long>());
    }
}

void Graph::parse_undirected(NeighborList &in_neighbors, NeighborList &out_neighbors, long &num_vertices,
                             std::vector<bool> &self_edges, std::vector<std::vector<std::string>> &contents) {
    for (std::vector<std::string> &edge : contents) {
        long from = std::stoi(edge[0]) - 1;  // Graph storage format indices vertices from 1, not 0
        long to = std::stoi(edge[1]) - 1;
        long weight = parse_edge_weight(edge);
        num_vertices = (from + 1 > num_vertices) ? from + 1 : num_vertices;
        num_vertices = (to + 1 > num_vertices) ? to + 1 : num_vertices;
        insert_weighted(out_neighbors, from, to, weight);
        if (from != to)
            insert_weighted(out_neighbors, to, from, weight);
        while (self_edges.size() < (size_t) num_vertices) {
            self_edges.push_back(false);
        }
        if (from == to) {
            self_edges[from] = true;
        }
    }
    in_neighbors = NeighborList(out_neighbors);
    while (out_neighbors.size() < size_t(num_vertices)) {
        out_neighbors.push_back(std::vector<long>());
    }
    while (in_neighbors.size() < size_t(num_vertices)) {
        in_neighbors.push_back(std::vector<long>());
    }
}

void Graph::sort_vertices() {
    if (!args.vertex_degree_sort) {
        // Default: use edge degree product for more accurate high-influence vertex identification
        this->degree_product_sort();
    } else {
        // Alternative: use vertex degree (faster but less accurate)
        std::vector<long> vertex_degrees = this->degrees();
        std::vector<int> indices = utils::range<int>(0, this->_num_vertices);
        std::stable_sort(indices.data(),
                  indices.data() + indices.size(), [&vertex_degrees](size_t i1, size_t i2) {
                  return vertex_degrees[i1] > vertex_degrees[i2];
        });
        for (int index = 0; index < this->_num_vertices; ++index) {
            int vertex = indices[index];
            if (index < (args.mh_percent * this->_num_vertices)) {
                this->_high_degree_vertices.push_back(vertex);
            } else {
                this->_low_degree_vertices.push_back(vertex);
            }
        }
        int num_islands = 0;
        for (int deg : vertex_degrees) {
            if (deg == 0) num_islands++;
        }
        std::cout << "Num island vertices = " << num_islands << std::endl;
    }
}

void Graph::degree_product_sort() {
    std::vector<std::pair<std::pair<long, long>, long>> edge_info = this->sorted_edge_list();
    MapVector<bool> selected;
    auto num_to_select = size_t(args.mh_percent * this->_num_vertices);
    int edge_index = 0;
    while (selected.size() < num_to_select) {
        const std::pair<std::pair<long, long>, long> &edge = edge_info[edge_index];
        selected[edge.first.first] = true;
        selected[edge.first.second] = true;
        edge_index++;
    }
    for (const std::pair<long, bool> &entry : selected) {
        this->_high_degree_vertices.push_back(entry.first);
    }
    for (long vertex = 0; vertex < this->_num_vertices; ++vertex) {
        if (selected[vertex]) continue;
        this->_low_degree_vertices.push_back(vertex);
    }
}

long Graph::num_islands() const {
    std::vector<long> vertex_degrees = this->degrees();
    long num_islands = 0;
    for (const long &degree : vertex_degrees) {
        if (degree == 0) num_islands++;
    }
    return num_islands;
}

std::vector<std::pair<std::pair<long, long>, long>> Graph::sorted_edge_list() const {
    std::vector<long> vertex_degrees = this->degrees();
    std::vector<std::pair<std::pair<long, long>, long>> edge_info;
    for (long source = 0; source < this->_num_vertices; ++source) {
        for (const long dest : this->out_neighbors(source)) {
            long information = vertex_degrees[source] * vertex_degrees[dest];
            edge_info.emplace_back(std::make_pair(source, dest), information);
        }
    }
    std::stable_sort(edge_info.begin(), edge_info.end(), [](const auto &i1, const auto &i2) {
        return i1.second > i2.second;
    });
    return edge_info;
}
