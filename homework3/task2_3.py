import sys
import time
import csv
import math
import numpy as np
from pyspark import SparkContext
import json
from operator import add
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import pickle
import os

# environment setting

os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"

def dealWithNaN(x,pos,num):
    if x[pos] is None:
        return num
    else:
        return x[pos]

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
        return [[user,targetBusiness,-1] for user in bid_uid_target_dict[targetBusiness]]
    missing_rater = bid_uid_target_dict[targetBusiness]
    res = []
    # then for each missing user, find all the candidates have that user rate
    for rater in missing_rater:
        if rater not in uid:
            res.append(rater,targetBusiness,-1)
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
            res.append(rater, targetBusiness, -1)
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
        n = min(len(temp_info), 8)
        if n<=2:
            res.append([rater, targetBusiness, 3])
        else:
            temp_info.sort(key=lambda x: x[1], reverse=True)
            temp_info = temp_info[:n]
            rate_numerator = sum([p[0]*p[1] for p in temp_info])
            rate_denominator = sum([abs(p[1]) for p in temp_info])
            if rate_denominator == 0:
                res.append([rater, targetBusiness, 3])
            else:
                rate = rate_numerator/rate_denominator
                res.append([rater, targetBusiness, 0.1*rate + 0.5*bid_avg[targetBusiness] + 0.4*uid_avg[rater]])
                # 0.5*rate + 0.3*bid_avg[targetBusiness] + 0.2*uid_avg[rater]
    # sort res
    # res = sorted(res)
    # print("finish1")
    return res

if __name__ == '__main__':
    # set the path for reading and outputting files
    folder_path = sys.argv[1]
    test_filepath = sys.argv[2]
    output_filepath = sys.argv[3]

    # Uncommon when run at local machine
    # folder_path = "HW3StudentData/"
    # test_filepath = "yelp_val.csv"
    # output_filepath = "output_task2_3_add.csv"

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'task2_3').getOrCreate()
    sc.setLogLevel("ERROR")

    # prepare datasets
    yelp_train = folder_path + "yelp_train.csv"
    # yelp_valid = folder_path + "yelp_val.csv"
    user = folder_path + "user.json"
    business = folder_path + "business.json"
    photo = folder_path + "photo.json"
    # checkin = folder_path + "checkin.json"
    review_train = folder_path + "review_train.json"
    # tip = folder_path + "tip.json"
    vec_features = "VectorizedFeatures.csv"

    start_time = time.time()
    # import dataset!
    # train_dataset
    rdd1 = sc.textFile(yelp_train)
    head = rdd1.first()
    uid_bid_star = rdd1.filter(lambda x: x != head).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    bid = uid_bid_star.map(lambda x: (x[1],1)).distinct()
    uid = uid_bid_star.map(lambda x: (x[0],1)).distinct()
    # test_dataset
    rdd2 = sc.textFile(test_filepath)
    head2 = rdd2.first()
    uid_bid_need = rdd2.filter(lambda x: x != head2).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
    # uid_bid_need = rdd2.filter(lambda x: x != head2).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    bid_test = uid_bid_need.map(lambda x: (x[1], 1)).distinct()
    uid_test = uid_bid_need.map(lambda x: (x[0], 1)).distinct()
    # business_dataset
    rdd3 = sc.textFile(business).map(lambda line: json.loads(line)).map(lambda x: (x['business_id'], x['review_count'], x['stars']))
    # photo dataset
    rdd4 = sc.textFile(photo).map(lambda line: json.loads(line)).map(lambda x: (x['business_id'], x['photo_id'], x['label']))
    # review_train dataset
    rdd5 = sc.textFile(review_train).map(lambda line: json.loads(line)).map(lambda x: (x['user_id'], x['stars'], x['useful'], x['funny'], x['cool'], x["business_id"]))
    # user dataset
    rdd6 = sc.textFile(user).map(lambda line: json.loads(line)).map(lambda x: (x["user_id"], x["review_count"], x["average_stars"], x['fans'], x["friends"], x["useful"], x["funny"], x["cool"],\
        x["compliment_hot"],x["compliment_more"],x["compliment_profile"],x["compliment_cute"],x["compliment_list"],x["compliment_note"],x["compliment_plain"],x["compliment_cool"],\
        x["compliment_funny"],x["compliment_writer"],x["compliment_photos"],x["yelping_since"]))
    # vec features import
    df_features = pd.read_csv(vec_features)

    # build the vars needed
    # from business perspective
    # types of photos business have
    restaurant_photo = rdd4.filter(lambda x: x[2] in ["inside","outside","food","drink"]).map(lambda x: (x[0], 1)).reduceByKey(add)
    # number of reviews business have
    # use the whole dataset
    bid_num_review = rdd3.map(lambda x: (x[0], x[1])).reduceByKey(add)
    # only use the review_train dataset
    # bid_num_review = rdd5.map(lambda x: (x[5],1)).reduceByKey(add)
    # average stars business have
    # use whole dataset
    bid_avg_stars = rdd3.map(lambda x: (x[0], x[2]))
    # only use review_train dataset
    # bid_avg_stars = rdd5.map(lambda x: (x[5], x[1])).groupByKey().mapValues(list).mapValues(lambda x: np.mean(x))
    # variance of stars for a business
    bid_var_stars = rdd5.map(lambda x: (x[5],x[1])).groupByKey().mapValues(list).mapValues(lambda x: np.var(x))

    # print(bid_avg_stars.take(3))
    # from user perspectives
    # number of creditability
    # num_fans_user = rdd5.map(lambda x: (x[0], x[2]+x[3]+x[4])).reduceByKey(add)
    num_fans_user = rdd6.map(lambda x: (x[0], x[3]))
    # number of reviews user write
    # uid_num_review = rdd5.map(lambda x: (x[0], 1)).reduceByKey(add)
    uid_num_review = rdd6.map(lambda x: (x[0], x[1]))
    # average stars of users
    # uid_avg_stars = rdd5.map(lambda x: (x[0],x[1])).reduceByKey(add).leftOuterJoin(uid_num_review).map(lambda x: (x[0],x[1][0]/x[1][1]))
    uid_avg_stars = rdd6.map(lambda x: (x[0],x[2]))
    # variance of stars for a user
    uid_var_stars = rdd5.map(lambda x: (x[0],x[1])).groupByKey().mapValues(list).mapValues(lambda x: np.var(x))
    # number of friends a user have
    uid_num_friends = rdd6.map(lambda x: (x[0],len(x[4].split(","))))
    # number of interations for a user
    uid_num_iteractions = rdd6.map(lambda x: (x[0], x[5]+x[6]+x[7]))
    # number of compliments recieved
    uid_num_compliment = rdd6.map(lambda x: (x[0],x[8]+x[9]+x[10]+x[11]+x[12]+x[13]+x[14]+x[15]+x[16]+x[17]+x[18]))
    # yelp since
    uid_yelp_since = rdd6.map(lambda x: (x[0],(2022-int(x[19][:4]))))
    # print(uid_avg_stars.take(3))
    # print(rdd3.collect())
    '''
    # join the data to create the train dataset
    # for bid (treat all null restaurant as 3)
    bid_final_train = bid.leftOuterJoin(restaurant_photo).map(lambda x: (x[0],x[1][1])).map(lambda x: (x[0],dealWithNaN(x,1,0)))\
        .leftOuterJoin(bid_num_review).map(lambda x: (x[0],x[1][0], x[1][1])).map(lambda x: (x[0],(x[1],dealWithNaN(x,2,0))))\
        .leftOuterJoin(bid_avg_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][1]))\
        .map(lambda x: (x[0],(x[1],x[2],dealWithNaN(x,3,3))))\
        .leftOuterJoin(bid_var_stars).map(lambda x: (x[0],x[1][0][0],x[1][0][1],x[1][0][2],x[1][1]))\
        .map(lambda x: (x[0], (x[1],x[2],x[3],dealWithNaN(x,4,0))))
    # print(bid_final_train.collect())
    # exit()
    # for uid (treat all null user as 3)
    uid_final_train = uid.leftOuterJoin(num_fans_user).map(lambda x: (x[0],x[1][1])).map(lambda x: (x[0],dealWithNaN(x,1,0)))\
        .leftOuterJoin(uid_num_review).map(lambda x: (x[0],x[1][0], x[1][1])).map(lambda x: (x[0],(x[1],dealWithNaN(x,2,0))))\
        .leftOuterJoin(uid_avg_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][1]))\
        .map(lambda x: (x[0],(x[1],x[2],dealWithNaN(x,3,3)))) \
        .leftOuterJoin(uid_var_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], dealWithNaN(x, 4, 0))))\
        .leftOuterJoin(uid_num_friends).map(lambda x: (x[0], x[1][0][0],x[1][0][1],x[1][0][2],x[1][0][3],x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], dealWithNaN(x, 5, 1))))\
        .leftOuterJoin(uid_num_iteractions).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], x[5], dealWithNaN(x, 6, 1))))\
        .leftOuterJoin(uid_num_compliment).map(lambda x: (x[0], x[1][0][0],x[1][0][1],x[1][0][2],x[1][0][3], x[1][0][4], x[1][0][5],x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], x[5], x[6],dealWithNaN(x, 7, 1))))\
        .leftOuterJoin(uid_yelp_since).map(lambda x: (x[0], x[1][0][0],x[1][0][1],x[1][0][2],x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6],x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], x[5], x[6], x[7], dealWithNaN(x, 8, 0))))

    # print(uid_final_train.take(3))
    # exit()
    # merge them together
    # final format: uid, bid, num_photo, bid_num_review, bid_avg_stars, bid_var, uid_num_fans, uid_num_reviews, uid_avg_stars, uid_var, num_friends, num_interact, actual_stars
    uid_bid_star_final = uid_bid_star.map(lambda x: (x[1],(x[0],x[2]))).leftOuterJoin(bid_final_train).\
        map(lambda x: (x[1][0][0], (x[0],x[1][1][0],x[1][1][1],x[1][1][2], x[1][1][3], x[1][0][1]))).leftOuterJoin(uid_final_train) .\
        map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3], x[1][1][4], x[1][1][5], x[1][1][6], x[1][1][7], x[1][0][5])).collect()
    # print(uid_bid_star_final)
    # exit()
    # prepare for the machine learning train data
    df_train_features = pd.DataFrame({"num_photo":[p[2] for p in uid_bid_star_final], "bid_num_review":[p[3] for p in uid_bid_star_final],
                             "bid_avg_stars":[p[4] for p in uid_bid_star_final], "bid_var": [p[5] for p in uid_bid_star_final],"uid_num_fans":[p[6] for p in uid_bid_star_final],
                             "uid_num_reviews":[p[7] for p in uid_bid_star_final],"uid_avg_stars":[p[8] for p in uid_bid_star_final], "uid_var":[p[9] for p in uid_bid_star_final],
                             "uid_num_friends":[p[10] for p in uid_bid_star_final], "uid_num_interactions": [p[11] for p in uid_bid_star_final],
                            "uid_num_compliment": [p[12] for p in uid_bid_star_final], "uid_yelp_since":[p[13] for p in uid_bid_star_final]})

    df_train_target = pd.DataFrame({"score":[p[14] for p in uid_bid_star_final]})
    scaler = preprocessing.StandardScaler()
    scaler.fit(df_train_features)
    df_train_features_scaled = scaler.transform(df_train_features)

    # set the model
    model = xgb.XGBRegressor()
    
    clf = GridSearchCV(model, {'max_depth': [8,9], 'learning_rate': [0.08,0.1,0.12],
                               "colsample_bytree": [0.3,0.4,0.5],
                               "subsample": [0.5,0.6], "alpha": [0], "random_state": [0]})
    
    clf = xgb.XGBRegressor(colsample_bytree=0.4, learning_rate=0.1, \
                              max_depth= 7 , alpha=0, n_estimators=110, subsample=0.6, random_state=0)
    clf.fit(df_train_features_scaled, df_train_target)
    # print("\n The best parameters across ALL searched params:\n", clf.best_params_)
    # print("training_RMSE:", mean_squared_error(y_true = df_train_target['score'], y_pred = clf.predict(df_train_features_scaled), squared=False))
    # exit()
    # join the data to create the test dataset
    # for bid (treat all null restaurant as 3)
    '''

    bid_final_test = bid_test.leftOuterJoin(restaurant_photo).map(lambda x: (x[0], x[1][1])).map(lambda x: (x[0], dealWithNaN(x, 1, 0))) \
        .leftOuterJoin(bid_num_review).map(lambda x: (x[0], x[1][0], x[1][1])).map(lambda x: (x[0], (x[1], dealWithNaN(x, 2, 0)))) \
        .leftOuterJoin(bid_avg_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], dealWithNaN(x, 3, 3)))) \
        .leftOuterJoin(bid_var_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], dealWithNaN(x, 4, 0))))
    # for uid (treat all null user as 3)
    uid_final_test = uid_test.leftOuterJoin(num_fans_user).map(lambda x: (x[0], x[1][1])).map(lambda x: (x[0], dealWithNaN(x, 1, 0))) \
        .leftOuterJoin(uid_num_review).map(lambda x: (x[0], x[1][0], x[1][1])).map(lambda x: (x[0], (x[1], dealWithNaN(x, 2, 0)))) \
        .leftOuterJoin(uid_avg_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], dealWithNaN(x, 3, 3)))) \
        .leftOuterJoin(uid_var_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], dealWithNaN(x, 4, 0))))\
        .leftOuterJoin(uid_num_friends).map(lambda x: (x[0], x[1][0][0],x[1][0][1],x[1][0][2],x[1][0][3],x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], dealWithNaN(x, 5, 1))))\
        .leftOuterJoin(uid_num_iteractions).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], x[5], dealWithNaN(x, 6, 1)))) \
        .leftOuterJoin(uid_num_compliment).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], x[5], x[6], dealWithNaN(x, 7, 1)))) \
        .leftOuterJoin(uid_yelp_since).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], x[5], x[6], x[7], dealWithNaN(x, 8, 0))))
    # merge them together
    '''
    # final format: uid, bid, num_photo, bid_num_review, bid_avg_stars, bid_var, uid_num_fans, uid_num_reviews, uid_avg_stars, uid_var, actual_stars
    uid_bid_star_final_test = uid_bid_need.map(lambda x: (x[1], (x[0], x[2]))).leftOuterJoin(bid_final_test). \
        map(lambda x: (x[1][0][0], (x[0], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3], x[1][0][1]))).leftOuterJoin(
        uid_final_test). \
        map(lambda x: (
    x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3],
    x[1][0][5])).collect()
    # print(uid_bid_star_final)
    # exit()
    # prepare for the machine learning train data
    df_test_features = pd.DataFrame(
        {"num_photo": [p[2] for p in uid_bid_star_final_test], "bid_num_review": [p[3] for p in uid_bid_star_final_test],
         "bid_avg_stars": [p[4] for p in uid_bid_star_final_test], "bid_var": [p[5] for p in uid_bid_star_final_test],
         "uid_num_fans": [p[6] for p in uid_bid_star_final_test],
         "uid_num_reviews": [p[7] for p in uid_bid_star_final_test], "uid_avg_stars": [p[8] for p in uid_bid_star_final_test],
         "uid_var": [p[9] for p in uid_bid_star_final_test]})
    df_test_target = pd.DataFrame({"score": [p[10] for p in uid_bid_star_final_test]})
    df_test_features_scaled = scaler.transform(df_test_features)
    # clf.fit(df_test_features_scaled, df_test_target)
    # print("\n The best parameters across ALL searched params:\n", clf.best_params_)
    print("testing_RMSE:",mean_squared_error(y_true=df_test_target['score'], y_pred=clf.predict(df_test_features_scaled),
                             squared=False))
    '''

    # final format: uid, bid, num_photo, bid_num_review, bid_avg_stars, uid_num_fans, uid_num_reviews, uid_avg_stars, actual_stars
    uid_bid_star_final_test = uid_bid_need.map(lambda x: (x[1], (x[0]))).leftOuterJoin(bid_final_test). \
        map(lambda x: (x[1][0], (x[0], x[1][1][0], x[1][1][1], x[1][1][2],x[1][1][3]))).leftOuterJoin(uid_final_test).\
        map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3],x[1][1][4],x[1][1][5],x[1][1][6], x[1][1][7])).collect()
    # print(uid_bid_star_final_test)
    # exit()
    # prepare for the machine learning train data
    df_test_features_add = pd.DataFrame({"num_photo": [p[2] for p in uid_bid_star_final_test], "bid_num_review": [p[3] for p in uid_bid_star_final_test],
         "bid_avg_stars": [p[4] for p in uid_bid_star_final_test], "bid_var": [p[5] for p in uid_bid_star_final_test],"uid_num_fans": [p[6] for p in uid_bid_star_final_test],
         "uid_num_reviews": [p[7] for p in uid_bid_star_final_test], "uid_avg_stars": [p[8] for p in uid_bid_star_final_test],"uid_var":[p[9] for p in uid_bid_star_final_test],
         "uid_num_friends":[p[10] for p in uid_bid_star_final_test], "uid_num_interactions": [p[11] for p in uid_bid_star_final_test],
        "uid_num_compliment": [p[12] for p in uid_bid_star_final_test],  "uid_yelp_since":[p[13] for p in uid_bid_star_final_test]})
    # df_test_target = pd.DataFrame({"score": [p[8] for p in uid_bid_star_final_test]})
    df_train_features = df_train_features_add.merge(df_features, left_on="uid", right_on="id", how="left").merge(
        df_features, left_on="bid", right_on="id", how="left", suffixes=("_1", "_2")).drop(
        ['uid', 'bid', 'id_1', 'id_2'], axis=1).fillna(0)
    scaler = pickle.load()
    df_test_features_scaled = scaler.transform(df_test_features)
    y_pred = clf.predict(df_test_features_scaled)
    '''
    df_final = pd.DataFrame(
        {"user_id": [p[0] for p in uid_bid_star_final_test], "business_id": [p[1] for p in uid_bid_star_final_test], "num_photo": [p[2] for p in uid_bid_star_final_test], "bid_num_review": [p[3] for p in uid_bid_star_final_test],
         "bid_avg_stars": [p[4] for p in uid_bid_star_final_test], "bid_var": [p[5] for p in uid_bid_star_final_test],"uid_num_fans": [p[6] for p in uid_bid_star_final_test],
         "uid_num_reviews": [p[7] for p in uid_bid_star_final_test], "uid_avg_stars": [p[8] for p in uid_bid_star_final_test],"uid_var":[p[9] for p in uid_bid_star_final_test],
         "uid_num_friends":[p[10] for p in uid_bid_star_final_test], "uid_num_interactions": [p[11] for p in uid_bid_star_final_test],"uid_num_compliment": [p[12] for p in uid_bid_star_final_test], "uid_yelp_since":[p[13] for p in uid_bid_star_final_test],
         "prediction": y_pred})
    '''
    df_final = pd.DataFrame({"user_id":[p[0] for p in uid_bid_star_final_test],"business_id":[p[1] for p in uid_bid_star_final_test],"prediction": y_pred})
    # print("testing_RMSE:",mean_squared_error(y_true=df_test_target['score'], y_pred=clf.predict(df_test_features_scaled),squared=False))
    end_time = time.time()
    duration = end_time - start_time
    print("Duration:", duration)


    start_time = time.time()
    # read in the file and keep it on the memory use persist
    rdd = sc.textFile(yelp_train)
    head = rdd.first()
    # only retain the main data (uid,bid,star)
    uid_bid_star = rdd.filter(lambda x: x != head).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    # uid_bid information
    uid_bid = uid_bid_star.map(lambda x: (x[0], x[1])).groupByKey().map(lambda x: (x[0], list(set(x[1])))).collect()
    uid_bid_dict = {}
    for i in range(len(uid_bid)):
        uid_bid_dict[uid_bid[i][0]] = uid_bid[i][1]
    # score information sep (business,uid):score
    bid_uid_star = uid_bid_star.map(lambda x: (x[1], x[0], x[2])).collect()
    bid_uid_star_dict = {}
    for i in range(len(bid_uid_star)):
        bid_uid_star_dict[tuple([bid_uid_star[i][0], bid_uid_star[i][1]])] = bid_uid_star[i][2]
    # avg star for each restaurant
    bid_avg_rdd = uid_bid_star.map(lambda x: (x[1], x[2])).groupByKey().map(
        lambda x: (x[0], sum(list(x[1])) / len(list(x[1])))).collect()
    bid_avg = {}
    for i in range(len(bid_avg_rdd)):
        bid_avg[bid_avg_rdd[i][0]] = bid_avg_rdd[i][1]
    # avg star for each user
    uid_avg_rdd = uid_bid_star.map(lambda x: (x[0], x[2])).groupByKey().map(
        lambda x: (x[0], sum(list(x[1])) / len(list(x[1])))).collect()
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
    bid_uid = uid_bid_star.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(set(x[1]))))
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
    bid_uid_target = uid_bid_star2.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], list(set(x[1]))))
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
    # print("Duration2:", duration)
    col1 = []
    col2 = []
    col3 = []
    for i in after_fill_score.collect():
        col1.append(i[0])
        col2.append(i[1])
        col3.append(i[2])
    df_final2 = pd.DataFrame({"user_id":col1, "business_id":col2, "prediction":col3})

    df = df_final.merge(df_final2, on=["user_id", "business_id"])

    info = []
    for i in range(len(df["prediction_x"])):
        if df["prediction_y"][i] == -1:
            info.append(df["prediction_x"][i])
        else:
            info.append(df["prediction_y"][i] * 0.1 + df["prediction_x"][i] * 0.9)

    df["prediction"] = info
    df_final = df[["user_id", "business_id", "prediction"]]
    df_final.to_csv(output_filepath,index=False)
    '''
    df2 = df_final
    df3 = pd.read_csv(test_filepath)

    df = df3.merge(df2, on=["user_id", "business_id"])
    df["diff"] = abs(df["stars"] - df["prediction"])
    df["diff_2"] = [i ** 2 for i in df["diff"]]
    print((sum(list(df["diff_2"])) / len(list(df["diff_2"]))) ** 0.5)
    '''



