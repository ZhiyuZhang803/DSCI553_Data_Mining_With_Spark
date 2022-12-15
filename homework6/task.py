import sys
import os
import time
import numpy as np
import math
from copy import deepcopy
from sklearn.cluster import KMeans
from collections import defaultdict
from pyspark import SparkContext

# environment setting
# os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"


def hash_function(x):
    value = ((133333 * x + 13) % 181081) % 5
    return int(value)


def generate_ds_rs(sample_set, sample_labels, current_DS, current_RS, first_step=False):
    new_rs = {}
    # generate cluster
    current_cluster = defaultdict(list)
    for i in range(len(sample_labels)):
        current_cluster[sample_labels[i]].append(list(sample_set.keys())[i])

    # distinguish the ds and rs set
    for single_cluster in current_cluster:
        cluster = current_cluster[single_cluster]
        if len(cluster) <= 1:
            new_rs[cluster[0]] = sample_set[cluster[0]]
            current_RS[cluster[0]] = sample_set[cluster[0]]
        else:
            temp = [sample_set[i] for i in cluster]
            temp = np.array(temp)

            length_cluster = len(temp)
            n_sum = temp.sum(axis=0)
            sum_sq = (temp ** 2).sum(axis=0)
            current_DS[single_cluster] = [cluster, length_cluster, n_sum, sum_sq]

    # make judgement
    if not first_step:
        return current_DS, current_RS
    else:
        return current_DS, new_rs


def genearte_cs_rs(current_CS, current_RS):
    current_CS = deepcopy(current_CS)
    current_RS = deepcopy(current_RS)

    # if number of data points is less than number of clusters, no need to proceed
    if len(current_RS) <= 5 * n_cluster:
        return current_CS, current_RS
    else:
        temp_l = list(current_RS.values())
        model_temp = KMeans(n_clusters=5 * n_cluster).fit(temp_l)
        current_CS, current_RS = generate_ds_rs(current_RS, model_temp.labels_, current_CS, current_RS)
        return current_CS, current_RS


def intermidiate_result(current_DS, current_CS, current_RS):
    ds_set = [s[0] for s in current_DS.values()]
    ds_number = sum([len(y) for y in ds_set])

    num_cs = len(current_CS)
    csid = [a[0] for a in current_CS.values()]
    cs_number = sum([len(y) for y in csid])

    rs_number = len(current_RS)
    return (ds_number, num_cs, cs_number, rs_number)


def assign_points(current_DS, current_CS, current_RS, coming_points):
    #iterate through each datapoints
    for id in coming_points:
        point_info = np.array(coming_points[id])
        dimension = len(point_info)
        min_distance = math.inf
        target_cluster = None
        for cluster in current_DS: # note: the cluster here is cluster id
            ma_distance = calculate_ma_distance(point_info, current_DS[cluster])
            if ma_distance <= min_distance:
                min_distance = ma_distance
                target_cluster = cluster
        # step8: assign to nearest ds
        if min_distance <= 2 * math.sqrt(dimension):
            cluster_deal = current_DS[target_cluster]
            new_id = cluster_deal[0] + [id]
            new_num = cluster_deal[1] + 1
            new_sum = cluster_deal[2] + point_info
            new_sum_sq = cluster_deal[3] + (point_info ** 2)
            current_DS[target_cluster] = [new_id, new_num, new_sum, new_sum_sq]
        # step9: assign to nearest cs
        else:
            min_distance_cs = math.inf
            target_cluster_cs = None
            for cs_cluster in current_CS:
                ma_distance_cs = calculate_ma_distance(point_info, current_CS[cs_cluster])
                if ma_distance_cs <= min_distance_cs:
                    target_cluster_cs = cs_cluster
                    min_distance_cs = ma_distance_cs
            if min_distance_cs <= 2 * math.sqrt(dimension):
                cluster_deal_cs = current_CS[target_cluster_cs]
                new_id = cluster_deal_cs[0] + [id]
                new_num = cluster_deal_cs[1] + 1
                new_sum = cluster_deal_cs[2] + point_info
                new_sum_sq = cluster_deal_cs[3] + (point_info ** 2)
                current_CS[target_cluster_cs] = [new_id, new_num, new_sum, new_sum_sq]
            # step10: assign to nearest ds
            else:
                current_RS[id] = coming_points[id]

    # step11: run k-means with large K on rs
    current_CS, current_RS = genearte_cs_rs(current_CS, current_RS)
    # step12: merge cs which have man_distance smaller than threshold
    current_CS = merge_cs(current_CS)
    return current_DS, current_CS, current_RS


def merge_cs(current_CS):
    CS = deepcopy(current_CS)
    # keep record for those cs deleted
    delete_cs = set()
    for cs1 in CS: # this is the index of cs1
        # basic info of cs1
        cs1_info = CS[cs1]
        length_cs1 = cs1_info[1]
        sum_cs1 = cs1_info[2]
        sum_sq_cs1 = cs1_info[3]
        centroid_cs1 = sum_cs1/length_cs1
        dimension_cs1 = len(centroid_cs1)
        # try to find closest cs2
        min_cs1_cs2 = math.inf
        final_cs2 = None
        for cs2 in CS:
            if cs2 != cs1 and cs2 not in delete_cs:
                cs1_cs2_distance = calculate_ma_distance(centroid_cs1, CS[cs2])
                if cs1_cs2_distance <= min_cs1_cs2:
                    min_cs1_cs2 = cs1_cs2_distance
                    final_cs2 = cs2
        if min_cs1_cs2 <= 2 * math.sqrt(dimension_cs1):
            cluster_deal_cs2 = current_CS[final_cs2]
            new_id = cluster_deal_cs2[0] + cs1_info[0]
            new_num = cluster_deal_cs2[1] + length_cs1
            new_sum = cluster_deal_cs2[2] + sum_cs1
            new_sum_sq = cluster_deal_cs2[3] + sum_sq_cs1
            current_CS[cs2] = [new_id, new_num, new_sum, new_sum_sq]
            delete_cs.add(cs1)
    # finally merged all groups need to be deleted
    for single_cs in delete_cs:
        CS.pop(single_cs)
    return CS


def final_round_merge(current_DS, current_CS):
    for cs_index in current_CS:
        cs_info = current_CS[cs_index]
        length_cs = cs_info[1]
        sum_cs = cs_info[2]
        sum_sq_cs = cs_info[3]
        centroid_cs = sum_cs/length_cs
        dimension_cs = len(centroid_cs)
        # try to find nearest ds
        min_cs_ds = math.inf
        final_ds = None
        for ds_index in current_DS:
            cs_ds_distance = calculate_ma_distance(centroid_cs, current_DS[ds_index])
            if cs_ds_distance <= min_cs_ds:
                min_cs_ds = cs_ds_distance
                final_ds = ds_index
        if min_cs_ds <= 2* math.sqrt(dimension_cs):
            cluster_deal_ds = current_DS[final_ds]
            new_id = cluster_deal_ds[0] + cs_info[0]
            new_num = cluster_deal_ds[1] + length_cs
            new_sum = cluster_deal_ds[2] + sum_cs
            new_sum_sq = cluster_deal_ds[3] + sum_sq_cs
            current_DS[final_ds] = [new_id, new_sum, new_sum, new_sum_sq]
            current_CS.pop(cs_index)
    return current_DS, current_CS



def calculate_ma_distance(current_point, current_cluster):
    cluster_num = current_cluster[1]
    cluster_sum = np.array(current_cluster[2])
    cluster_sum_sq = np.array(current_cluster[3])

    mu = cluster_sum / cluster_num
    sig = np.sqrt((cluster_sum_sq / cluster_num) - mu ** 2)
    dist = np.sqrt((((current_point - mu) / sig) ** 2).sum())
    return dist



if __name__ == '__main__':
    # set the path for reading and outputting files
    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file = sys.argv[3]

    # Uncommon when run at local machine
    # input_file = "hw6_clustering.txt"
    # n_cluster = 50
    # output_file = "homework6task.txt"

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'task').getOrCreate()
    sc.setLogLevel("ERROR")

    start_time = time.time()

    DS = {}
    CS = {}
    RS = {}
    final_result = []

    # readfile and find a proper hash function to generate sample
    rdd = sc.textFile(input_file)
    # use hash function to give 5 cats
    population = rdd.map(lambda x: (int(x.split(",")[0]), [float(single_num) for single_num in x.split(",")[2:]])) \
        .map(lambda x: (x[0], hash_function(x[0]), x[1]))

    # step1: get first group of sample: hash=0
    first_sample = population.filter(lambda x: x[1] == 0).map(lambda x: (x[0], x[2])).collectAsMap()

    # step2: run k-means with 5n
    k_means_model1 = KMeans(n_clusters=int(n_cluster * 5)).fit(list(first_sample.values()))

    # step3: move all the clusters that contain only one point to RS
    label = k_means_model1.labels_
    label_dict = {}
    label_count = defaultdict(int)
    for i in range(len(list(first_sample.keys()))):
        label_dict[list(first_sample.keys())[i]] = label[i]
        label_count[label[i]] += 1

    single_ele = []
    for ele in label_count:
        if label_count[ele] == 1:
            single_ele.append(ele)

    for i in label_dict:
        if label_dict[i] in single_ele:
            RS[i] = first_sample[i]
            first_sample.pop(i)

    # step4: Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
    k_means_model2 = KMeans(n_clusters=int(n_cluster)).fit(list(first_sample.values()))

    # step5: Use the K-Means result from Step 4 to generate the DS clusters
    DS, RS = generate_ds_rs(first_sample, k_means_model2.labels_, DS, RS, first_step=True)

    # step6: Run K-Means on the points in the RS with a large K to generate CS and RS .
    RS, CS = genearte_cs_rs(CS, RS)

    info = intermidiate_result(DS, CS, RS)
    final_result.append(info)
    # print(info)
    # print(CS, RS, DS)

    # step7-12: continue recursive work
    for flag in [1, 2, 3, 4]:
        # step7: load data from original dataset
        current_sample = population.filter(lambda x: x[1] == flag).map(lambda x: (x[0], x[2])).collectAsMap()
        # step9-12: new points, compare them to each of the DS using the Mahalanobis Distance and assign
        # them to the nearest DS clusters if the distanceis <2 𝑑.
        DS, CS, RS = assign_points(DS, CS, RS, current_sample)

        # last round, needed additional merge
        if flag == 4:
            DS, CS = final_round_merge(DS, CS)

        info = intermidiate_result(DS, CS, RS)
        final_result.append(info)

    print('Percentage of discard points', final_result[-1][0] / len(population.collect()))

    final_result2 = []
    for label1 in DS:
        # ds[0] stores the list of point ids
        for d1 in DS[label1][0]:
            final_result2.append((d1, label1))

    # cs all merged into ds?
    for label2 in CS:
        for d2 in CS[label2][0]:
            final_result2.append((d2, -1))

    for rs in RS:
        final_result2.append((rs, -1))

    # sort by data point index ascending
    final_result2 = sorted(final_result2, key=lambda a: a[0])

    with open(output_file, 'w') as f:
        f.write('The intermediate results:\n')
        for i in range(len(final_result)):
            x = final_result[i]
            line = f'Round {i+1}: {x[0]},{x[1]},{x[2]},{x[3]}\n'
            f.write(line)
        f.write('\n')

        f.write('The clustering results:\n')
        for x in final_result2:
            f.write(f'{x[0]},{x[1]}\n')

    print("Duration:",time.time()-start_time)