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
    business_filepath = sys.argv[2]
    output_filepath_question_a = sys.argv[3]
    output_filepath_question_b = sys.argv[4]

    # Uncommon when run at local machine
    # review_filepath = "test_review.json"
    # business_filepath = "business.json"
    # output_filepath_question_a = "output3a.json"
    # output_filepath_question_b = "output3b.json"

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'task3').getOrCreate()
    sc.setLogLevel("ERROR")

    # read in the file and keep it on the memory use persist
    rdd_review = sc.textFile(review_filepath).map(lambda line: json.loads(line))
    rdd_business = sc.textFile(business_filepath).map(lambda line: json.loads(line))

    # question A: What are the average stars for each city? (1 point)
    # Use the review table star information
    business_star = rdd_review.map(lambda single: (single["business_id"],single["stars"]))
    business_city = rdd_business.map(lambda single: (single["business_id"],single["city"]))
    # join two tables
    star_city = business_star.leftOuterJoin(business_city)
    # aggregate and calculate
    new = star_city.map(lambda single: (single[1][1],single[1][0])).filter(lambda single: single[1] is not None).\
        groupByKey().map(lambda single: (single[0],(sum(single[1]),len(single[1])))).map(lambda single: (single[0],float(single[1][0]/single[1][1]))).\
        sortBy(lambda single:[-single[1],single[0]]).collect()

    with open(output_filepath_question_a, 'w', encoding="utf-8") as f:
        f.write("city,stars\n")
        for each_city in new:
            f.write(f'{each_city[0]},{each_city[1]}' + '\n')

    # question B: compare the execution time of using two methods to print top 10 cities with highest stars.
    result_final = {}
    # method 1: normal python sort
    start1 = time.time()
    # read in the file and keep it on the memory use persist
    rdd_review_b1 = sc.textFile(review_filepath).map(lambda line: json.loads(line))
    rdd_business_b1 = sc.textFile(business_filepath).map(lambda line: json.loads(line))
    # Use the review table star information
    business_star_b1 = rdd_review_b1.map(lambda single: (single["business_id"], single["stars"]))
    business_city_b1 = rdd_business_b1.map(lambda single: (single["business_id"], single["city"]))
    # join two tables
    star_city_b1 = business_star_b1.leftOuterJoin(business_city)
    # aggregate and calculate
    new_b1 = star_city_b1.map(lambda single: (single[1][1], single[1][0])).filter(lambda single: single[1] is not None). \
        groupByKey().map(lambda single: (single[0], (sum(single[1]), len(single[1])))).map(lambda single: (single[0], float(single[1][0] / single[1][1]))).collect()
    result_b1 = sorted(new_b1,key=lambda single: (-single[1], single[0]))
    if len(result_b1)>=10:
        for i in result_b1[:10]:
            print(i[0])
    else:
        for i in result_b1:
            print(i[0])
    end1 = time.time()
    m1 = end1-start1
    result_final["m1"] = m1

    # method 2: spark sort
    start2 = time.time()
    # read in the file and keep it on the memory use persist
    rdd_review_b2 = sc.textFile(review_filepath).map(lambda line: json.loads(line))
    rdd_business_b2 = sc.textFile(business_filepath).map(lambda line: json.loads(line))
    # Use the review table star information
    business_star_b2 = rdd_review_b2.map(lambda single: (single["business_id"], single["stars"]))
    business_city_b2 = rdd_business_b2.map(lambda single: (single["business_id"], single["city"]))
    # join two tables
    star_city_b2 = business_star_b2.leftOuterJoin(business_city)
    # aggregate and calculate
    result_b2 = star_city_b2.map(lambda single: (single[1][1], single[1][0])).filter(lambda single: single[1] is not None). \
        groupByKey().map(lambda single: (single[0], (sum(single[1]), len(single[1])))).map(lambda single: (single[0], float(single[1][0] / single[1][1]))).\
        takeOrdered(10,key=lambda single: (-single[1], single[0]))
    if len(result_b2) >= 10:
        for i in result_b2[:10]:
            print(i[0])
    else:
        for i in result_b2:
            print(i[0])
    end2 = time.time()
    m2 = end2-end1
    result_final["m2"] = m2

    sc.stop()

    # my spark will work faster
    result_final["reason"] = "In this problem, spark can finish the work in several partitions but python can only use computer sole memory, so spark will be faster."

    with open(output_filepath_question_b, 'w') as f:
        json.dump(result_final, f)
