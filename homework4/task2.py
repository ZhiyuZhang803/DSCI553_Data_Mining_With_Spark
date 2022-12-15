import sys
import time
from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict, deque
import os
import copy

# environment setting
# os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"

def assign_credit(root, nodes, tree, parent_node, path):
    # assign the default credit
    score_numb = {}
    for v in nodes:
        if v == root:
            score_numb[v] = 0
        else:
            score_numb[v] = 1

    each_level = list(tree.values())[::-1][:-1]
    edge_score = {}

    # from bottom to top, calculate the final result
    for this_level in each_level:
        for n in this_level:
            for par in parent_node[n]:
                weight = path[par] / path[n]
                score = weight * score_numb[n]
                score_numb[par] += score
                edge_score[tuple(sorted((n, par)))] = score

    return [(k, v) for k, v in edge_score.items()]


def bfs(root, node_relationship):
    traverse = set()
    traverse.add(root)
    cur_level = deque([root])
    depth = 0

    # keep track of parent node
    parent_node = defaultdict(set)
    # keep track of the path
    path = defaultdict(int)
    # summarize level of the nodes
    node_level = defaultdict(int)
    # store the tree
    tree = defaultdict(list)

    while cur_level:
        depth += 1
        temp = []
        # each loop is a different level
        for i in range(len(cur_level)):
            node = cur_level.popleft()
            node_level[node] = depth
            temp.append(node)
            related_nodes = node_relationship[node]
            for d in related_nodes:
                # only keep those nodes that do not appear in the past path
                if d not in traverse:
                    traverse.add(d)
                    cur_level.append(d)
                # if it connects with a node in the previous level
                if node_level.get(d) == (depth - 1):
                    parent_node[node].add(d)

            # compute the num of shortest path of the current node (num of parent from previous level)
            if node == root:
                path[node] = 1
            else:
                plist = parent_node[node]
                path[node] = sum([path[single] for single in plist])
        tree[depth] = temp

    return root, tree, parent_node, path


def Girvan_Newman(root, adjacency, nodes):
    result = bfs(root, adjacency)
    return assign_credit(result[0], nodes, result[1], result[2], result[3])

def find_new_community(nodes, new_relationship):
    info = set()
    community = []
    for n in nodes:
        if n in info:
            continue
        else:
            temp_c = set()
            temp_c.add(n)
            temp_comm = deque([n])

            while temp_comm:
                tt = temp_comm.popleft()
                dst = new_relationship[tt]
                for r_n in dst:
                    if r_n not in temp_c:
                        temp_comm.append(r_n)
                        info.add(r_n)
                        temp_c.add(r_n)
            community.append(temp_c)
    return community

def modularity_calculator(community):
    new_modularity = 0
    for each_community in community:
        for p1 in each_community:
            for p2 in each_community:
                if p1 != p2:
                    if p2 in A_info[p1]:
                        A = 1
                    else:
                        A = 0
                else:
                    continue
                new_modularity += (A - (degree_each_point[p1]*degree_each_point[p2])/(2*m))
    new_modularity = new_modularity/(2*m)
    return new_modularity


if __name__ == '__main__':
    # set the path for reading and outputting files
    threshold = int(sys.argv[1])
    input_filepath = sys.argv[2]
    bet_output_filepath = sys.argv[3]
    com_output_filepath = sys.argv[4]

    # Uncommon when run at local machine
    # threshold = 7
    # input_filepath = "ub_sample_data.csv"
    # bet_output_filepath = "taks2.1bet.txt"
    # com_output_filepath = "../homework4/task2.1com.txt"

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'task1').getOrCreate()
    sc.setLogLevel("ERROR")

    start = time.time()
    # start read the data
    rdd = sc.textFile(input_filepath)
    head = rdd.first()
    uid_bid = rdd.filter(lambda x: x != head).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
    # extract each user with a set of business: uid:[bid, bid....]
    uid_bid_dict = uid_bid.groupByKey().map(lambda x: (x[0], list(set(x[1])))).filter(
        lambda x: len(x[1]) >= threshold).collectAsMap()

    # create pairs for all pairs of user
    nodes = set()
    edges = set()
    uid_pairs = combinations(list(uid_bid_dict.keys()), 2)

    edge_rdd = sc.parallelize(list(uid_pairs), 10) \
        .map(lambda x: (x, len(set(uid_bid_dict[x[0]]).intersection(set(uid_bid_dict[x[1]]))))) \
        .filter(lambda x: x[1] >= threshold).sortBy(lambda x: x[0])

    # construct some useful entities related to the edges
    edge_single = edge_rdd.flatMap(lambda x: (x[0][0], x[0][1])).distinct().collect()   # list contains all single entity
    edge_dict = edge_rdd.collectAsMap()   # {(uid1, uid2): number they commonly have, (uid1, uid2): numbers they commonly have}
    edge_pairs = list(set(list(edge_dict.keys())))  # [(uid1, uid2), ......]
    connect_each_single = defaultdict(set)  # {uid1: {uid2, uid3, uid4 ...}, ...} connected users: bi directional
    for e in edge_pairs:
        connect_each_single[e[0]].add(e[1])
        connect_each_single[e[1]].add(e[0])
    # print(edge_rdd.collect())
    # exit()

    # calculate the betweeness for each user
    cal_rdd = sc.parallelize(edge_single,10).map(lambda x: Girvan_Newman(x, connect_each_single, edge_single)). \
        flatMap(lambda pair: pair).reduceByKey(lambda x, y: x + y).map(lambda x: (sorted(x[0]), x[1] / 2)).sortBy(lambda x: (-x[1], x[0]))

    # prepare for the output
    betweeness = cal_rdd.map(lambda x: (sorted(x[0]), round(x[1],5))).collect()
    with open(bet_output_filepath, 'w') as f:
        for single in betweeness:
            # f.write(str(single)[1:-1].replace("[","(").replace("]",")") + "\n")
            f.write("('"+ single[0][0]+ "', '"+ single[0][1]+ "'),"+ str(single[1]) + "\n")
    print('Duration_Betweeness', time.time() - start)

    ######################################### Calculate Community ###############################################
    # degree of each single node, that is, k_i and k_j
    degree_each_point = {}
    for single_node in connect_each_single.keys():
        degree_each_point[single_node] = len(connect_each_single[single_node])
    # calculate total number of edge
    m = len(edge_pairs)
    # if exist an edge
    A_info = copy.deepcopy(connect_each_single)

    # start to divide the group
    total_modularity = 0
    flag = m

    betweeness_number = cal_rdd.collect()

    # start split
    while True:
        highest_betweeness_number = betweeness_number[0][1]
        for i in betweeness_number:
            if i[1] == highest_betweeness_number:
                highest_betweeness_pair = i[0]
                p1 = highest_betweeness_pair[0]
                p2 = highest_betweeness_pair[1]
                # print(highest_betweeness_number,p1,p2)
                # update the original relationship
                connect_each_single[p1].remove(p2)
                connect_each_single[p2].remove(p1)
                flag -= 1


        # detect new community
        # print(len(edge_single))
        new_community = find_new_community(edge_single, connect_each_single)
        # print(len(new_community))
        new_modularity = modularity_calculator(new_community)

        # find the highest
        if new_modularity >= total_modularity:
            final_community = new_community
            total_modularity = new_modularity
            # update the bewtweeness

        if flag ==0:
            break

        # print("total_modularity", total_modularity, "new_modularity", new_modularity)
        betweeness_number = sc.parallelize(edge_single, 10).map(lambda x: Girvan_Newman(x, connect_each_single, edge_single)). \
            flatMap(lambda pair: pair).reduceByKey(lambda x, y: x + y).map(lambda x: (sorted(x[0]), x[1] / 2)).sortBy(lambda x: (-x[1], x[0])).collect()
        # print(len(betweeness_number))

    # end up with outpt file
    output = sc.parallelize(final_community,10).map(lambda x: sorted(x)).sortBy(lambda x: (len(x), x)).collect()
    # print(final_community)
    # print("output",output)
    print('Duration2:', time.time() - start)
    with open(com_output_filepath, 'w') as p:
        for s in output:
            p.write("'" + "', '".join(s) + "'" + "\n")
    # print("final modularity", total_modularity)


