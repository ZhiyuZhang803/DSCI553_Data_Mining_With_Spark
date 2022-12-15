import sys
import time
import csv
from pyspark import SparkContext
import os

# environment setting

# os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"

p_similar = {}
def PartHash(string):
    seed = 131
    hash = 0
    for ch in string:
        hash = hash * seed + ord(ch)
    return hash & 0x7FFFFFFF

def PearsonSimilarity(bid1, bid2):
    avg1 = bid_avg[bid1]
    avg2 = bid_avg[bid2]
    rate_users_bid1 = bid_uid_dict[bid1]
    rate_users_bid2 = bid_uid_dict[bid2]
    co_rated_users = set(rate_users_bid1).intersection(set(rate_users_bid2))
    # find scores for each co_rated user
    co_rated_scores = []
    for co_user in co_rated_users:
        co_rated_scores.append([bid_uid_star_dict[tuple([bid1,co_user])]-avg1,bid_uid_star_dict[tuple([bid2,co_user])]-avg2])

    # calculate
    numerator = sum([p[0]*p[1] for p in co_rated_scores])
    denominator1 = sum([p[0]**2 for p in co_rated_scores])**0.5
    denominator2 = sum([p[1]**2 for p in co_rated_scores])**0.5
    denominator = denominator1 * denominator2
    if denominator == 0:
        p_similar[tuple([bid1,bid2])] = 0
        return 0
    else:
        p_similar[tuple([bid1, bid2])] = numerator/denominator
        return numerator/denominator

"""def calculatePearsonAllpairs():
    res = {}
    for i1 in range(len(bid)):
        bid1 = bid[i1]
        for bid2 in bid[i1+1:]:
            a = set(bid1).intersection(set(bid2))
            if len(a) == 0:
                continue
            else:
                p = PearsonSimilarity(bid1,bid2)
                res[tuple([bid1,bid2])] = p
        # print("finish")
    return res"""

def FillInScore(targetBusiness):
    # first find which one is missing
    if targetBusiness not in bid:
        return [[user,targetBusiness,3.0] for user in bid_uid_target_dict[targetBusiness]]
    missing_rater = bid_uid_target_dict[targetBusiness]
    res = []
    # then for each missing user, find all the candidates have that user rate
    for rater in missing_rater:
        if rater not in uid:
            res.append(rater,targetBusiness,3.0)
            continue
        temp_info = []
        # find all other business with same rater rated
        # rater_list = []
        """for each_bid in uid_bid_dict[rater]:
            if each_bid == targetBusiness:
                continue
            else:
                rater_list.append(each_bid)"""
        rater_list = set(uid_bid_dict[rater])-set(targetBusiness)
        if not rater_list:
            res.append(rater, targetBusiness, 3.0)
            continue
        # NOTICE!!! 这里很可能出现空的位置RATER_LIS！！！
        # calculate similarity and store the information (score, similarity)
        for valid_bid in rater_list:
            co_rated_users = set(bid_uid_dict[targetBusiness]).intersection(set(bid_uid_dict[valid_bid]))
            if len(co_rated_users) == 0:
                continue
            else:
                if tuple([targetBusiness,valid_bid]) in p_similar:
                    p_similarity = p_similar[tuple([targetBusiness,valid_bid])]
                elif tuple([valid_bid,targetBusiness]) in p_similar:
                    p_similarity = p_similar[tuple([valid_bid,targetBusiness])]
                else:
                    p_similarity = PearsonSimilarity(valid_bid, targetBusiness)
                temp_info.append([bid_uid_star_dict[tuple([valid_bid,rater])],p_similarity])
        n = min(len(temp_info), 10)
        temp_info.sort(key=lambda x: x[1], reverse=True)
        temp_info = temp_info[:n]
        rate_numerator = sum([p[0]*p[1] for p in temp_info])
        rate_denominator = sum([abs(p[1]) for p in temp_info])
        if rate_denominator == 0:
            res.append([rater, targetBusiness, 3.0])
        else:
            rate = rate_numerator/rate_denominator
            res.append([rater, targetBusiness, 0.1*rate + 0.5*bid_avg[targetBusiness] + 0.4*uid_avg[rater]])
    # sort res
    # res = sorted(res)
    # print("finish1")
    return res


if __name__ == '__main__':
    # set the path for reading and outputting files
    train_filepath = sys.argv[1]
    test_filepath = sys.argv[2]
    output_filepath = sys.argv[3]

    # Uncommon when run at local machine
    # train_filepath = "yelp_train.csv"
    # test_filepath = "yelp_val.csv"
    # output_filepath = "output_task2_1.csv"

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'task1').getOrCreate()
    sc.setLogLevel("ERROR")

    start_time = time.time()
    # read in the file and keep it on the memory use persist
    rdd = sc.textFile(train_filepath)
    head = rdd.first()
    # only retain the main data (uid,bid,star)
    uid_bid_star = rdd.filter(lambda x: x != head).map(lambda x: x.split(",")).map(lambda x : (x[0],x[1],float(x[2])))
    # uid_bid information
    uid_bid = uid_bid_star.map(lambda x: (x[0], x[1])).groupByKey().map(lambda x: (x[0], list(set(x[1])))).collect()
    uid_bid_dict = {}
    for i in range(len(uid_bid)):
        uid_bid_dict[uid_bid[i][0]] = uid_bid[i][1]
    # score information sep (business,uid):score
    bid_uid_star = uid_bid_star.map(lambda x: (x[1],x[0],x[2])).collect()
    bid_uid_star_dict = {}
    for i in range(len(bid_uid_star)):
        bid_uid_star_dict[tuple([bid_uid_star[i][0],bid_uid_star[i][1]])] = bid_uid_star[i][2]
    # avg star for each restaurant
    bid_avg_rdd = uid_bid_star.map(lambda x: (x[1], x[2])).groupByKey().map(lambda x: (x[0], sum(list(x[1]))/len(list(x[1])))).collect()
    bid_avg = {}
    for i in range(len(bid_avg_rdd)):
        bid_avg[bid_avg_rdd[i][0]] = bid_avg_rdd[i][1]
    # avg star for each user
    uid_avg_rdd = uid_bid_star.map(lambda x: (x[0], x[2])).groupByKey().map(lambda x: (x[0], sum(list(x[1])) / len(list(x[1])))).collect()
    uid_avg = {}
    for i in range(len(uid_avg_rdd)):
        uid_avg[uid_avg_rdd[i][0]] = uid_avg_rdd[i][1]
    # bid information (no prob)
    bid = uid_bid_star.map(lambda x: x[1]).distinct().collect()
    # uid_index info (no prob)
    uid = uid_bid_star.map(lambda x: x[0]).distinct().collect()
    uid_index = {}
    for i in range(len(uid)):
        uid_index[uid[i]] = i
    # bid_uid related info (no prob)
    bid_uid = uid_bid_star.map(lambda x: (x[1],x[0])).groupByKey().map(lambda x: (x[0],list(set(x[1]))))
    bid_uid_list = bid_uid.collect()
    bid_uid_dict = {}
    for i in range(len(bid_uid_list)):
        bid_uid_dict[bid_uid_list[i][0]] = bid_uid_list[i][1]

    # use some information from validation dataset
    rdd2 = sc.textFile(test_filepath)
    head2 = rdd2.first()
    # only retain the main data (uid,bid,star)
    uid_bid_star2 = rdd2.filter(lambda x: x != head2).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
    # uid_bid_star2 = rdd2.filter(lambda x: x != head2).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    bid_uid_target = uid_bid_star2.map(lambda x: (x[1],x[0])).groupByKey().map(lambda x: (x[0],list(set(x[1]))))
    bid_uid_target_list = bid_uid_target.collect()
    bid_uid_target_dict = {}
    for i in range(len(bid_uid_target_list)):
        bid_uid_target_dict[bid_uid_target_list[i][0]] = bid_uid_target_list[i][1]
    after_fill_score = bid_uid_target.partitionBy(10, PartHash).map(lambda x: FillInScore(x[0])).flatMap(lambda x: x)
    """
    # compare RSME
    after_fill_score_compare = after_fill_score.map(lambda x: ((x[0],x[1]),x[2]))
    uid_bid_compare = uid_bid_star2.map(lambda x: ((x[0],x[1]),x[2]))
    RMSE = after_fill_score_compare.join(uid_bid_compare).map(lambda x: (1, (x[1][0] - x[1][1]) ** 2)).reduceByKey(lambda x, y: x + y).collect()
    # print("RMSE:{}".format((RMSE[0][1] / len(after_fill_score_compare.collect()))**0.5))
    """
    # write time
    end_time = time.time()
    duration = end_time - start_time
    print("Duration:", duration)

    with open(output_filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        for i in after_fill_score.collect():
            writer.writerow(i)

