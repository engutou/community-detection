#! python

import networkx as nx


def set_node_attributes(G, attr_name):
    """Set attribute for each vertex of graph G.
    The name of attribute is given by >attr_name<.
    :param G: graph
    :param attr_name: string
    :return: None
    """
    if attr_name == 'k-index':
        core_number = nx.core_number(G)
        nx.set_node_attributes(G, core_number, name=attr_name)
    else:
        print('Unknown attribute name:', attr_name)

    
def get_neighbors(G, nodes_in_community):
    """
    Return the neighbor nodes and edges of a set of nodes.
    =======
    example:
        G: x -- 1 -- 2 -- y
                 \  /
                  3
                  |
                  z
        nodes_in_community = {1, 2, 3}
        输出：{x, y, z} & {(1, x); (2, y); (3, z)}

    :param G: networkx.Graph
            The original graph.
    :param nodes_in_community: list
            The list contains a set of nodes in the graph.
    :return:
    """
    neighbor_nodes = []
    for n in nodes_in_community:
        neighbor_nodes.extend([n for n in nx.neighbors(G, n) if n not in nodes_in_community])
    # 移除重复的邻居节点
    neighbor_nodes = list(set(neighbor_nodes))

    # 边的一端是邻居节点
    neighbor_edges = [e for e in G.edges(neighbor_nodes)]
    # 边的另一端是给定的输入节点
    neighbor_edges = list(filter(lambda e: e[0] in nodes_in_community or e[1] in nodes_in_community, neighbor_edges))
    return neighbor_nodes, neighbor_edges


def modularity(G, nodes_in_community, weight):
    """ Compute the modularity based on weighted edges.
    modularity = sum_{e in community}w_e / (sum_{e in community}w_e + sum_{e' in neighbor edges}w_e')
    :param G:
    :param nodes_in_community:
    :return:
    """
    edges_in_community = list(G.subgraph(nodes_in_community).edges)
    weight_sum_ce = sum([G[e[0]][e[1]][weight] for e in edges_in_community])

    _, neighbor_edges = get_neighbors(G, nodes_in_community=nodes_in_community)
    weight_sum_ne = sum([G[e[0]][e[1]][weight] for e in neighbor_edges])

    return float(weight_sum_ce) / (weight_sum_ce + weight_sum_ne)


def find_local_community(G, seed_node, weight, verbose=0):
    """ Find the local community for the seed node.
    :param G: NetworkX.Graph
        待分析的加权无向图

    :param seed_node: one node of the graph G, or a list of nodes in the graph
        如果seed_node是单个节点，则以该节点为种子节点发现对应社团
        如果seed_node是多个节点，则表示初始社团已经包含这些节点

    :param weight: string
        用于社团发现的边权重名称

    :return nodes_in_community: a list of nodes in the graph
        The list consists of the nodes in the community.

    :return mod: float
        The corresponding modularity of the community.
    """
    nodes_in_community = seed_node if isinstance(seed_node, list) else [seed_node]
    mod = modularity(G, nodes_in_community=nodes_in_community, weight=weight)
    neighbor_nodes, neighbor_edges = get_neighbors(G, nodes_in_community=nodes_in_community)
    if verbose >= 1:
        print('==========\nInitial Community:', nodes_in_community)
        print('Modularity = %.4f' % mod)
        print('Neighbor:\n\t\tNodes:', neighbor_nodes, '\n\t\tEdges:', neighbor_edges)
    while neighbor_nodes:
        # Compute the new modularity for each neighbor node,
        # suppose the neighbor node is added to the community
        mod_max, c_max, n_max = 0, None, None
        for n in neighbor_nodes:
            nodes_in_temp_community = nodes_in_community.copy()
            nodes_in_temp_community.append(n)
            mod_temp = modularity(G, nodes_in_community=nodes_in_temp_community, weight=weight)
            if mod_temp > mod_max:
                mod_max, c_max, n_max = mod_temp, nodes_in_temp_community, n
        if mod_max > mod:
            if verbose >= 1:
                print('==========\nNode', n_max,
                      'and edge', set(G.edges(n_max)).intersection(neighbor_edges),
                      'are added to the community')

            # Update the community and the corresponding neighbor edges
            nodes_in_community, mod = c_max, mod_max
            neighbor_nodes, neighbor_edges = get_neighbors(G, nodes_in_community=nodes_in_community)

            if verbose >= 1:
                print('Updated Community:', nodes_in_community)
                print('Modularity = %.4f' % mod_max)
                print('Neighbor:\n\t\tNodes:', neighbor_nodes, '\n\t\tEdges:', neighbor_edges)
        else:
            if verbose >= 1:
                print('Found The Community:', nodes_in_community)
            break
    return nodes_in_community, mod


def pick_seed_node(G, edge_weight='weight', mode='k-index'):
    from operator import itemgetter
    if mode == 'k-index':
        return max(list(G.nodes(data='k-index')), key=itemgetter(1))[0]
    elif mode == 'edge_weight':
        return list(max(list(G.edges(data=edge_weight)), key=itemgetter(2))[0:2])
    else:
        return None


def detection_algorithm(G, edge_weight='weight', seed_node_mode='k-index'):
    """
    对整个加权无向图执行社区发现算法
    :param G: networkx.Graph
        The weighted undirected graph.

    :param edge_weight: string
        加权无向图的权重名

    :param seed_node_mode: string
        为各个社区选取种子节点的方法
    """
    Gc = G.copy()
    communities = []

    if seed_node_mode == 'k-index':
        # 优先将k-index值大的节点作为种子节点
        set_node_attributes(Gc, attr_name='k-index')

    elif seed_node_mode == 'edge_weight':
        # 优先将权重大的边关联的两个节点作为种子节点
        # 这种方式以边为导向，初始化各个社区时就加入了一条边
        # 查看整个图中的边权重均值，目前没有使用，后续可能有用
        mean_weight = sum([edge[-1] for edge in list(Gc.edges(data=edge_weight))]) / float(Gc.number_of_edges())
    else:
        print('Unknown seed node picking mode:', seed_node_mode)

    while Gc.number_of_nodes() > 0:
        seed_node = pick_seed_node(Gc, edge_weight=edge_weight, mode=seed_node_mode)
        nodes_in_community, mod = find_local_community(Gc, seed_node=seed_node, weight=edge_weight, verbose=1)
        communities.append((nodes_in_community, mod))
        Gc.remove_nodes_from(nodes_in_community)
    return communities


if __name__ == "__main__":
    def sample_1():
        edge_list = [
                    ('1', '2', 1),
                    ('1', '3', 1),
                    ('2', '3', 30),
                    ('2', '4', 5),
                    ('2', '5', 5),
                    ('2', '6', 2),
                    ('3', '6', 4),
                    ('3', '7', 3),
                    ('4', '5', 20),
                    ('4', '8', 1),
                    ('6', '7', 20),
                    ('6', '8', 30),
                    ('7', '8', 30)]
        G = nx.Graph()
        G.add_weighted_edges_from(edge_list, weight='call-num')
        # print(G.nodes(data=True))
        # print(G.edges(data=True))
        print('\n*******1*********\n', find_local_community(G, seed_node=['1', '2', '3'], weight='call-num'))
        print('\n********2********\n', find_local_community(G, seed_node='1', weight='call-num'))
        print('\n*********3*******\n', find_local_community(G, seed_node='8', weight='call-num'))
        print('\n**********4******\n', find_local_community(G, seed_node='6', weight='call-num'))
        print('\n***********5*****\n', find_local_community(G, seed_node='4', weight='call-num'))
        # print(detection_algorithm(G, edge_weight='call-num', seed_node_mode='edge_weight'))
        # print(detection_algorithm(G, edge_weight='call-num', seed_node_mode='k-index'))

    sample_1()




