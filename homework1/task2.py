import sys
import time
import json
from pyspark import SparkContext
# import os

# environment setting
# os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"

if __name__ == '__main__':
    # set the path for reading and outputting files
    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]
    n_partition = int(sys.argv[3])

    # Uncommon when run at local machine
    # review_filepath = "test_review.json"
    # output_filepath = "output1.json"
    # n_partition = 5

    # dict to take the final result
    final_result = {}

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'task2').getOrCreate()
    sc.setLogLevel("ERROR")

    # read in the file and keep it on the memory use persist
    rdd = sc.textFile(review_filepath).map(lambda line: json.loads(line))


    # before improving
    default = {}
    rdd1 = rdd.map(lambda single: [single["business_id"],1])
    start_time = time.time()
    # executing default situation
    result1 = rdd1.reduceByKey(lambda x, y: x + y).takeOrdered(10,key=lambda x: [-x[1],x[0]])
    end_time = time.time()

    default['n_partition'] = rdd1.getNumPartitions()
    default['n_items'] = rdd1.glom().map(lambda x: len(x)).collect()
    default['exe_time'] = end_time - start_time
    final_result["default"] = default


    # after improving
    customized = {}
    rdd2 = rdd.map(lambda single: [single["business_id"], 1]).partitionBy(n_partition,lambda x: ord(x[0][0]))
    start_time2 = time.time()
    # executing default situation
    result2 = rdd2.reduceByKey(lambda x, y: x + y).takeOrdered(10, key=lambda x: [-x[1], x[0]])
    end_time2 = time.time()

    customized['n_partition'] = rdd2.getNumPartitions()
    customized['n_items'] = rdd2.glom().map(lambda x: len(x)).collect()
    customized['exe_time'] = end_time2 - start_time2
    final_result["customized"] = customized

    # release the connection
    sc.stop()

    # write in a json file
    with open(output_filepath, 'w') as f:
        json.dump(final_result, f)