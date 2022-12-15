import sys
import json
from pyspark import SparkContext
import os

# environment setting
# os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"

if __name__ == '__main__':
    # set the path for reading and outputting files
    review_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    # just use for test
    # review_filepath = "test_review.json"
    # output_filepath = "zzyoutput1.json"

    result = {}

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'task1').getOrCreate()
    sc.setLogLevel("ERROR")

    # read in the original file
    rdd = sc.textFile(review_filepath).map(lambda line: json.loads(line))

    # A. The total number of reviews (0.5 point)
    result["n_review"] = rdd.count()

    # B. The number of reviews in 2018 (0.5 point)
    result["n_review_2018"] = rdd.map(lambda single: single['date']).filter(lambda x: "2018" in x).count()

    # C. The number of distinct users who wrote reviews (0.5 point)
    result["n_user"] = rdd.map(lambda single: single["user_id"]).distinct().count()

    # D. The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
    result["top10_user"] = rdd.map(lambda single: [single["user_id"], 1]).reduceByKey(lambda x, y: x+y).takeOrdered(10, key=lambda x: [-x[1],x[0]])

    # E. The number of distinct businesses that have been reviewed (0.5 point)
    result["n_business"] = rdd.map(lambda single: single["business_id"]).distinct().count()

    # F. The top 10 businesses that had the largest numbers of reviews and the number of reviews they had
    result["top10_business"] = rdd.map(lambda single: [single["business_id"],1]).reduceByKey(lambda x, y: x+y).takeOrdered(10,key=lambda x: [-x[1],x[0]])

    sc.stop()

    # write in a json file
    with open(output_filepath, 'w') as f:
        json.dump(result, f)
