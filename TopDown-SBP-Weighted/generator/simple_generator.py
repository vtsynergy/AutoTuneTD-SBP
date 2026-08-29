import argparse
import os
# from partition_baseline_support import *
import graph_tool.all as gt
import pandas as pd # for writing output graph TSV files
import math
import networkx as nx
import numpy as np
import random
import scipy.stats as stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--numvertices", type=int, default=40, help="Number of vertices in graph")
    parser.add_argument("-c", "--communities", type=int, default=4, help="Number of communities in graph. If -1, uses communityexponent argument instead.")
    parser.add_argument("-o", "--overlap", type=float, default=0.05, help="0.0 < o < 1.0; o% of the intra-block edges will be added between blocks.")
    parser.add_argument("--biasblocks", type=int, default=2, help="biasblocks < communities; the number of blocks that have a higher chance of forming inter-block edges.")
    parser.add_argument("-b", "--blockbias", type=float, default=0.5, help="0.0 < b < 1.0; the chance that an interblock edge is formed between the biased blocks.")
    parser.add_argument("--biasvertices", type=int, default=2, help="The number of vertices in each block that have a higher chance of forming inter-block edges.")
    parser.add_argument("--vertexbias", type=float, default=1.0, help="0.0 <= vertexbias <= 1.0; the chance that an interblock edge is formed between the biased blocks.")
    parser.add_argument("--explicitoverlap", type=float, default=-1, help="If > 0, will use this value as the explicit number of edges b/n blocks.")
    parser.add_argument("-s", "--blocksizevariation", type=float, default=1.0, help="Power law exponent for block sizes. s > 1")
    parser.add_argument("-e", "--powerlawexponent", type=float, default=2.1, help="The power law exponent. e > 1")
    parser.add_argument("-d", "--density", type=float, default=1.0, help="1 - d fraction of edges will be removed")
    parser.add_argument("--directory", type=str, default="/mnt/d/workspace/graph_data")
    return parser.parse_args()
# Generate the graph according to the blockmodel and parameters

args = parse_args()
N = args.numvertices
N_adjusted = int(args.numvertices * 1.13)
C = args.communities
ratio_within_over_between = args.overlap
block_size_heterogeneity = args.blocksizevariation
powerlaw_exponent = args.powerlawexponent
density = args.density
num_blocks = C
print('Number of blocks: {}'.format(num_blocks))

save_graph = True
tag = "test_{0}_{1}_{2}_{3}_{4}".format(num_blocks, args.biasblocks, args.blockbias, args.biasvertices, args.vertexbias)
overlap = "low"
if args.overlap < 5:
    overlap = "high"
block_size_variation = "low"
if args.blocksizevariation > 1.0:
    block_size_variation = "high"
file_name = "{4}/{3}/{0}Overlap_{1}BlockSizeVar/{3}_{0}Overlap_{1}BlockSizeVar_{2}_nodes".format(
    args.overlap, args.blocksizevariation, N, tag, args.directory)
os.makedirs(os.path.dirname(file_name), exist_ok=True)

bias_vertices = np.random.choice(np.arange(N), args.biasvertices, False)
vertex_degrees = np.zeros(N)

graph = gt.Graph(directed=True)
start = 0
end = math.ceil(N / num_blocks)
blocks = list()
while start < N:
    print("start: {0} end: {1}".format(start, end))
    # build a fully connected component
    blocks.append(list(range(start, end)))
    for i in range(start, end):

        # if i in bias_vertices and random.random() < 0.5
        if i in bias_vertices and vertex_degrees[i] > len(blocks[-1])/2:
            continue
        for j in range(i+1, end):
            if i == j:
                continue
            # if j in bias_vertices and random.random() < 0.5:
            if j in bias_vertices and vertex_degrees[j] > len(blocks[-1])/2:
                continue
            if random.random() < 0.5:
                graph.add_edge(i, j)
            else:
                graph.add_edge(j, i)
            vertex_degrees[i] += 1
            vertex_degrees[j] += 1
    start = end
    end += int(N / num_blocks)
    end = min(end, N)

# TODO: deterministic bias_vertex selection (find all community combinations, then loop over those)

# add edges between components
block_indices = np.arange(num_blocks)
true_assignment = list()
for b in block_indices:
    true_assignment.extend([b] * len(blocks[b]))
true_assignment = np.asarray(true_assignment)

for vertex in bias_vertices:
    vertex_block = true_assignment[vertex]
    # expected_edges = 2
    # expected_edges = len(blocks[vertex_block]) - 1
    print("vertex degrees: ", graph.get_total_degrees([vertex]))
    # missing_edges = int(expected_edges) - int(graph.get_total_degrees([vertex])[0])
    missing_edges = int(graph.get_total_degrees([vertex])[0] - 1)
    b = np.random.choice(block_indices)
    while b == true_assignment[vertex]:
        b = np.random.choice(block_indices)
    for i in range(missing_edges):
        # b = np.random.choice(block_indices)
        # while b == true_assignment[vertex]:
        #     b = np.random.choice(block_indices)
        v_b = np.random.choice(blocks[b])
        if random.random() < 0.5:
            while v_b in graph.get_out_neighbors(vertex):
                v_b = np.random.choice(blocks[b])
            graph.add_edge(vertex, v_b)
        else:
            while v_b in graph.get_in_neighbors(vertex):
                v_b = np.random.choice(blocks[b])
            graph.add_edge(v_b, vertex)

    # out_edges = graph.get_out_edges(vertex)
    # for edge in out_edges:
    #     if random.random() < 0.5:
    #         graph.remove_edge(edge)
    #         b = np.random.choice(block_indices)
    #         while b == true_assignment[vertex]:
    #             b = np.random.choice(block_indices)
    #         v_b = np.random.choice(blocks[b])
    #         while v_b in graph.get_out_neighbors(vertex):
    #             v_b = np.random.choice(blocks[b])
    #         graph.add_edge(vertex, v_b)
    # in_edges = graph.get_out_edges(vertex)
    # for edge in in_edges:
    #     if random.random() < 0.5:
    #         graph.remove_edge(edge)
    #         b = np.random.choice(block_indices)
    #         while b == true_assignment[vertex]:
    #             b = np.random.choice(block_indices)
    #         v_b = np.random.choice(blocks[b])
    #         while v_b in graph.get_in_neighbors(vertex):
    #             v_b = np.random.choice(blocks[b])
    #         graph.add_edge(v_b, vertex)

for i in range(int(args.overlap * graph.num_edges())):
    if random.random() < args.blockbias:
        a, b = np.random.choice(block_indices[:args.biasblocks], 2, False)
    else:
        # block_probs = np.ones(num_blocks)
        # block_probs[0] = args.blockbias
        # block_probs[1] = args.blockbias
        # block_probs /= block_probs.sum()
        a, b = np.random.choice(block_indices, 2, False)  # , block_probs)
    # if random.random() < args.vertexbias:
    #     v_a = np.random.choice(blocks[a][:args.biasvertices])
    # else:
    v_a = np.random.choice(blocks[a])
    v_b = np.random.choice(blocks[b])
    while v_b in graph.get_out_neighbors(v_a):
        # if random.random() < args.vertexbias:
        #     v_a = np.random.choice(blocks[a][:args.biasvertices])
        # else:
        v_a = np.random.choice(blocks[a])
        v_b = np.random.choice(blocks[b])
    graph.add_edge(v_a, v_b)

# gt.remove_self_loops(g_sample)
# gt.remove_parallel_edges(g_sample)
#
# # remove (1-density) percent of the edges
# edge_filter = g_sample.new_edge_property('bool')
# edge_filter.a = stats.bernoulli.rvs(density, size=edge_filter.a.shape)
# # print("edge filter: ", edge_filter.a)
# g_sample.set_edge_filter(edge_filter)
# g_sample.purge_edges()
#
# # remove all island vertices
# print('Filtering out zero vertices...')
# degrees = g_sample.get_total_degrees(np.arange(g_sample.num_vertices()))
# vertex_filter = g_sample.new_vertex_property('bool', vals=degrees > 0.0)
# g_sample.set_vertex_filter(vertex_filter)
# g_sample.purge_vertices()
#
# # store the nodal block memberships in a vertex property
# block_membership_vector = block_membership_vector[degrees > 0.0]
# true_partition = block_membership_vector
# assert block_membership_vector.size == g_sample.num_vertices()
# block_membership = g_sample.new_vertex_property("int",
#                                                 vals=block_membership_vector)
#
# # compute and report basic statistics on the generated graph
# bg, bb, vcount, ecount, avp, aep = gt.condensation_graph(g_sample, block_membership, self_loops=True)
# edge_count_between_blocks = np.zeros((num_blocks, num_blocks))
# for e in bg.edges():
#     edge_count_between_blocks[bg.vertex_index[e.source()], bg.vertex_index[e.target()]] = ecount[e]
# num_within_block_edges = sum(edge_count_between_blocks.diagonal())
# num_between_block_edges = g_sample.num_edges() - num_within_block_edges
# print count statistics
# print('Number of nodes: {} expected {} filtered % {}'.format(
#       graph.num_vertices(), N, (N_adjusted - .num_vertices()) / N_adjusted))
# print('Number of edges: {} expected number of edges: {}'.format(
#     g_sample.num_edges(), expected_e))
print('Number of vertices: {0} number of edges: {1}'.format(graph.num_vertices(), graph.num_edges()))
degrees = graph.get_total_degrees(np.arange(graph.num_vertices()))
print('Vertex degrees: [{},{},{}]'.format(
    np.min(degrees), np.mean(degrees), np.max(degrees)))
unique_degrees, counts = np.unique(degrees, return_counts=True)
print("degrees: {}\ncounts: {}".format(unique_degrees[:20], counts[:20]))
print('Avg. Number of nodes per block: {}'.format(graph.num_vertices() / num_blocks))
# print('# Within-block edges / # Between-blocks edges: {}'.format(num_within_block_edges / num_between_block_edges))

if save_graph:  # output the graph and truth partition to TSV files with standard format
    # graph.save('{}.gt.bz2'.format(file_name))  # save graph-tool graph object
    # store edge list
    edge_list = np.zeros((graph.num_edges(), 3), dtype=int)
    # populate the edge list.
    counter = 0
    for e in graph.edges():  # iterate through all edges (edge list access is not available in all versions of graph-tool)
        edge_list[counter, 0] = int(e.source()) + 1;  # nodes are indexed starting at 1 in the standard format
        edge_list[counter, 1] = int(e.target()) + 1;  # nodes are indexed starting at 1 in the standard format
        edge_list[counter, 2] = 1;  # all edges are weighted equally at 1 in this generator
        counter += 1
    # write graph TSV file using pandas DataFrame
    df_graph = pd.DataFrame(edge_list)
    df_graph.to_csv('{}.tsv'.format(file_name), sep='\t', header=False, index=False)
    # write truth partition TSV file using pandas DataFrame; nodes and blocks are indexed starting at 1 in the standard format
    df_partition = pd.DataFrame(np.column_stack((np.arange(graph.num_vertices()) + 1, true_assignment + 1)))
    if graph.num_vertices() < 500:
        gt.graph_draw(graph, vertex_fill_color=graph.new_vertex_property("int", true_assignment), output="{}_view.png".format(file_name))
    df_partition.to_csv('{}_truePartition.tsv'.format(file_name), sep='\t', header=False, index=False)
    print("Graph saved to {}".format(file_name))

