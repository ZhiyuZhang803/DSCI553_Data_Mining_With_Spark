import sys
import time
from pyspark.sql import SQLContext
from pyspark import SparkContext
from graphframes import GraphFrame
from itertools import combinations, permutations
import os

# environment setting
# os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"

if __name__ == '__main__':
    # set the path for reading and outputting files
    threshold = int(sys.argv[1])
    input_filepath = sys.argv[2]
    output_filepath = sys.argv[3]

    # Uncommon when run at local machine
    # threshold = 7
    # input_filepath = "ub_sample_data.csv"
    # output_filepath = "taks1result.txt"

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'task1').getOrCreate()
    sc.setLogLevel("ERROR")
    sqlContext = SQLContext(sc)

    start = time.time()
    # start read the data
    rdd = sc.textFile(input_filepath)
    head = rdd.first()
    uid_bid = rdd.filter(lambda x: x != head).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
    # extract each user with a set of business: uid:[bid, bid....]
    uid_bid_dict = uid_bid.groupByKey().map(lambda x: (x[0], list(set(x[1])))).filter(lambda x: len(x[1]) >= threshold).collectAsMap()

    # create pairs for all pairs of user
    nodes = set()
    edges = set()
    uid_pairs = combinations(list(uid_bid_dict.keys()),2)
    # reconstruct uid_pairs
    # print(list(uid_pairs)[:10])
    # exit()
    '''
    for i in uid_pairs:
        user1 = i[0]
        user2 = i[1]
        intersect = list(set(uid_bid_dict[user1]).intersection(uid_bid_dict[user2]))
        if len(intersect) >= threshold:
            nodes.add((user1,))
            nodes.add((user2,))
            edges.add((user1,user2))
    '''
    edge_rdd = sc.parallelize(list(uid_pairs), 10) \
        .map(lambda x: (x, len(set(uid_bid_dict[x[0]]).intersection(set(uid_bid_dict[x[1]]))))) \
        .filter(lambda x: x[1] >= threshold).sortBy(lambda x: x[0])

    user_list = list(edge_rdd.map(lambda x: (1, x[0])).reduceByKey(lambda a, b: set(a).union(set(b))).collect()[0][1])

    nodes = [(user,) for user in sorted(user_list)]
    edges = [tup[0] for tup in edge_rdd.collect()] + [tup[0][::-1] for tup in edge_rdd.collect()]

    # build the target graph
    nodes = sqlContext.createDataFrame(nodes, ['id'])
    edges = sqlContext.createDataFrame(edges, ['src', 'dst'])
    graph = GraphFrame(nodes, edges)

    res = graph.labelPropagation(maxIter=5)
    result = res.rdd.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y).map(lambda x: (sorted(x[1]))).sortBy(lambda x: (len(x), x)).collect()

    duration = time.time() - start
    print('Duration:', duration)

    with open(output_filepath, 'w') as out:
        for c in result:
            out.write("'" + "', '".join(c) + "'" + "\n")



