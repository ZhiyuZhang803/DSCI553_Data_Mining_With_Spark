import sys
from pyspark import SparkContext
import os
import time
from itertools import combinations
import math

# environment setting
# os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"

def get_single_items(basket_l, threshold):
    info = {}
    final_candidate = []
    # iterate through all the single item and count
    for single_basket in basket_l:
        for item in single_basket:
            info[item] = info.get(item, 0) + 1
    # judge if it occurs over the threshold
    for candidate in info.keys():
        if info[candidate] >= threshold:
            final_candidate.append(candidate)
        else:
            continue
    return final_candidate

'''
# make a try for two items
def get_pair_items(basket_l, threshold):
    info = {}
    final_candidate = []
    frequent_single = get_single_items(basket_l,threshold)
    # find all possible frequent pairs
    possible_items = combinations(frequent_single,2)
    for pair in possible_items:
        for single_basket in basket_l:
            if pair[0] in single_basket and pair[1] in single_basket:
                info[pair] = info.get(pair, 0) + 1
    # make the judgement with the threshold
    for candidate in info.keys():
        if info[candidate] >= threshold:
            final_candidate.append(candidate)
        else:
            continue
    return final_candidate
'''
# check and count
def intersection_check(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    if len(lst3) == len(lst1):
        return True
    else:
        return False

def get_any_num_items(basket_l, threshold, target_number):
    while target_number == 1:
        return sorted(get_single_items(basket_l, threshold))
    prev_frequent = get_single_items(basket_l,threshold)
    huge_info = sorted([(i,) for i in prev_frequent])
    flag = target_number
    while flag > 1:
        info = {}
        final_candidate = []
        combo_number = target_number-flag+2
        # get the frequent items in  the previous stage
        if combo_number == 2:
            stage_freq_single = prev_frequent
        else:
            stage_freq_single = set()
            for combo in prev_frequent:
                for item in combo:
                    stage_freq_single.add(item)
            stage_freq_single = sorted(stage_freq_single)
        # construct all possible item pairs
        new_comb = combinations(stage_freq_single,combo_number)

        for combo_long in new_comb:
            for single_basket in basket_l:
                if intersection_check(combo_long,single_basket):
                    info[combo_long] = info.get(combo_long, 0) + 1
        # check the threshold
        for candidate in info.keys():
            if info[candidate] >= threshold:
                # candidate = sorted(candidate)
                final_candidate.append(candidate)
        final_candidate = [tuple(sorted(i)) for i in final_candidate]
        # print(type(final_candidate))
        final_candidate = sorted(list(set(final_candidate)))
        if final_candidate != []:
            huge_info = huge_info + final_candidate
        # update the iteration
        flag = flag-1
        prev_frequent = final_candidate
    return huge_info

def implement_son(single_market_basket,total_length,support_num):
    max_length = max([len(i) for i in single_market_basket])
    current_threshold = math.ceil(len(single_market_basket) * support_num / total_length)
    return get_any_num_items(single_market_basket,current_threshold,max_length)

def format_output(items):
    info = {}
    for i in items:
        length = len(i)
        new_item = "('" + "', '".join(list(i)) + "'),"
        if length in info:
            info[length] = info[length]+new_item
        else:
            info[length] = new_item
    max_length = list(info.keys())[-1]
    final_output = ""
    for ele in info.keys():
        if ele != max_length:
            final_output = final_output+info[ele][:-1]+"\n\n"
        else:
            final_output = final_output+info[ele][:-1]
    return final_output

def check_in_total(total_market_basket, frequent_items):
    info = {}
    for i in frequent_items:
        for single_basket in total_market_basket:
            if intersection_check(i,single_basket):
                info[i] = info.get(i, 0) + 1
    final_result = [(i,info[i]) for i in info.keys()]
    return final_result


if __name__ == '__main__':
    # set the path for reading and outputting files
    case = int(sys.argv[1])
    support = int(sys.argv[2])
    input_filepath = sys.argv[3]
    output_filepath = sys.argv[4]

    # only for the test use
    # case = 1
    # support = 4
    # input_filepath = "small2.csv"
    # output_filepath = "outputcase1.txt"

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'task1').getOrCreate()
    sc.setLogLevel("ERROR")

    time_start = time.time()
    # read in the original file
    rdd = sc.textFile(input_filepath).filter(lambda x: x != "user_id,business_id")
    if case == 1:
        # separate the infomation
        rdd_sep = rdd.map(lambda x: (x.split(",")[0],[x.split(",")[1]]))
        rdd_question1 = rdd_sep.reduceByKey(lambda x, y: x+y).mapValues(lambda x: list(set(x)))
        rdd_question1.collect()
    elif case == 2:
        rdd_sep2 = rdd.map(lambda x: (x.split(",")[1], [x.split(",")[0]]))
        rdd_question2 = rdd_sep2.reduceByKey(lambda x, y: x + y).mapValues(lambda x: list(set(x)))
        rdd_question2.collect()

    if case == 1:
        rdd_next = rdd_question1
    elif case == 2:
        rdd_next = rdd_question2

    # for the phase one
    # calculate the length of total rdd in order to change the threshold
    length_rdd = rdd_next.count()
    # construct the sub_threshold
    # weight_rdd = [math.ceil(len(i)*support/length_rdd) for i in rdd_next.glom().collect()]
    basket_list = rdd_next.map(lambda x: x[1]).glom().map(lambda x: implement_son(x,length_rdd,support)).flatMap(lambda x: x).collect()
    inter_out = sorted(list(set(basket_list)), key=lambda x: (len(x), x))

    # for the phase two
    basket_list2 = rdd_next.map(lambda x: x[1]).glom().map(lambda x: check_in_total(x,inter_out)).flatMap(lambda x: x).\
        reduceByKey(lambda x,y: x+y).filter(lambda x: x[1]>=support).map(lambda x: x[0]).collect()
    final_out = sorted(list(set(basket_list2)), key=lambda x: (len(x), x))

    # write to the outer files
    with open(output_filepath, 'w') as output_file:
        out = 'Candidates:\n' + format_output(inter_out) + '\n\n' + 'Frequent Itemsets:\n' + format_output(final_out)
        output_file.write(out)

    end_time = time.time()
    duration = round(end_time - time_start, 0)
    print('Duration:', duration)





