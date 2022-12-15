import sys
import time
import csv
import random
from itertools import combinations
from pyspark import SparkContext
import os

# environment setting

# os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"

def buildHashMap(nHashRequired, indexNeedHash):
    res = []
    a = random.sample(range(1,60000),nHashRequired)
    b = random.sample(range(1,60000), nHashRequired)
    p = 17377
    m = len(indexNeedHash)+1
    for nHash in range(nHashRequired):
        info = []
        for nNum in range(len(indexNeedHash)):
            # f(x) = ((ax + b) % p) % m
            info.append(((a[nHash] * nNum + b[nHash]) % p) % m)
        res.append(info)
    return res

def MinHash(itemList):
    # project the item to the index
    index_list = [uid_index[i] for i in itemList]
    res = []
    # for each HashMap
    for num_Hash in range(len(ordered_HashMap)):
        hash_value_list = [ordered_HashMap[num_Hash][ele] for ele in index_list]
        res.append(min(hash_value_list))
    return res

def CreatePair(hashSet):
    res = set()
    for band_numb in range(band):
        start = int(band_numb * 2)
        end = int((band_numb+1) * 2)
        temp_dict = {}
        for set_numb in range(len(hashSet)):
            single_set = hashSet[set_numb]
            # print(single_set)
            sub_hash = tuple(single_set[start:end])
            if sub_hash in temp_dict:
                temp_dict[sub_hash].append(set_numb)
            else:
                temp_dict[sub_hash]= [set_numb]
        for value in list(temp_dict.values()):
            if len(value) == 1:
                continue
            else:
                comb = combinations(value,2)
                for single_comb in comb:
                    res.add(tuple(sorted(single_comb)))
    return res

def CheckSimilarity(CandidatePair):
    res = []
    for ele in CandidatePair:
        ele1 = index_bid[ele[0]]
        ele2 = index_bid[ele[1]]
        review1 = set(bid_list_uid_dict[ele1])
        review2 = set(bid_list_uid_dict[ele2])
        j_similarity = len(review1.intersection(review2))/len(review1.union(review2))
        if j_similarity>=threshold:
            res.append(sorted([ele1,ele2])+[j_similarity])
    res = sorted(res)
    return res

if __name__ == '__main__':
    # set the path for reading and outputting files
    input_filepath = sys.argv[1]
    output_filepath = sys.argv[2]

    # Uncommon when run at local machine
    # input_filepath = "yelp_train.csv"
    # output_filepath = "output_task1.csv"

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'task1').getOrCreate()
    sc.setLogLevel("ERROR")

    start_time = time.time()
    # read in the file and keep it on the memory use persist
    rdd = sc.textFile(input_filepath)
    head = rdd.first()
    # only retain the main data (uid,bid)
    uid_bid = rdd.filter(lambda x: x != head).map(lambda x: x.split(",")).map(lambda x: (x[0],x[1]))
    # finish the first step, build (bid,[uid1,uid2...])
    bid_list_uid = uid_bid.map(lambda x: (x[1],x[0])).groupByKey().map(lambda x: (x[0],list(set(x[1]))))
    bid_list_uid_list = bid_list_uid.collect()
    bid_list_uid_dict = {}
    for i in range(len(bid_list_uid_list)):
        bid_list_uid_dict[bid_list_uid_list[i][0]] = bid_list_uid_list[i][1]
    # print(bid_list_uid_dict)
    # exit()
    # build uid_index
    uid = uid_bid.map(lambda x: x[0]).distinct().collect()
    uid_index = {}
    for i in range(len(uid)):
        uid_index[uid[i]] = i
    # build 100 hash map
    n_hash = 100
    ordered_HashMap = buildHashMap(n_hash,uid)
    # 100 11270
    # print(len(ordered_HashMap),len(ordered_HashMap[1]))
    # prepare for Mini hash, placeholder
    # bid_placeholder = bid_list_uid.map(lambda x: (x[1],[float("inf")]*n_hash))
    # iterate each bid and map the concrete number to each index, and use hash to find minimum number
    bid_hash = bid_list_uid.map(lambda x: (x[0],MinHash(x[1])))
    # conduct the LSH
    threshold = 0.5
    band = 50
    n_rows = 2
    # create a bid list in order for reference
    bid = bid_hash.map(lambda x: x[0]).collect()
    bid_index = {}
    index_bid = {}
    for i in range(len(bid)):
        bid_index[bid[i]] = i
        index_bid[i] = bid[i]
    # build collect of hash function
    hash_collect = bid_hash.map(lambda x: (1,x[1])).groupByKey().map(lambda x: list(x[1]))
    # find pairs that might be similar
    create_pairs = hash_collect.map(lambda x: CreatePair(x))
    # calculate similarity score
    check_similarity = create_pairs.map(lambda x: CheckSimilarity(x)).collect()[0]
    # print(check_similarity[0])

    end_time = time.time()
    duration = end_time - start_time
    print("Duration:", duration)

    with open(output_filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["business_id_1", "business_id_2", "similarity"])
        for i in check_similarity:
            writer.writerow(i)


    # print(create_pairs.take(1))