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
import os

# environment setting

# os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"

def dealWithNaN(x,pos,num):
    if x[pos] is None:
        return num
    else:
        return x[pos]

if __name__ == '__main__':
    # set the path for reading and outputting files
    folder_path = sys.argv[1]
    test_filepath = sys.argv[2]
    output_filepath = sys.argv[3]

    # Uncommon when run at local machine
    # folder_path = "HW3StudentData/"
    # test_filepath = "yelp_val.csv"
    # output_filepath = "output_task2_2.csv"

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'task2_2').getOrCreate()
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
    rdd6 = sc.textFile(user).map(lambda line: json.loads(line)).map(lambda x: (x["user_id"], x["review_count"], x["average_stars"], x['fans'], x["friends"], x["useful"], x["funny"], x["cool"]))

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
    # print(uid_avg_stars.take(3))
    # print(rdd3.collect())

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
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], x[5], dealWithNaN(x, 6, 1))))

    # print(uid_final_train.take(3))
    # exit()
    # merge them together
    # final format: uid, bid, num_photo, bid_num_review, bid_avg_stars, bid_var, uid_num_fans, uid_num_reviews, uid_avg_stars, uid_var, num_friends, num_interact, actual_stars
    uid_bid_star_final = uid_bid_star.map(lambda x: (x[1],(x[0],x[2]))).leftOuterJoin(bid_final_train).\
        map(lambda x: (x[1][0][0], (x[0],x[1][1][0],x[1][1][1],x[1][1][2], x[1][1][3], x[1][0][1]))).leftOuterJoin(uid_final_train) .\
        map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3], x[1][1][4], x[1][1][5], x[1][0][5])).collect()
    # print(uid_bid_star_final)
    # exit()
    # prepare for the machine learning train data
    df_train_features = pd.DataFrame({"num_photo":[p[2] for p in uid_bid_star_final], "bid_num_review":[p[3] for p in uid_bid_star_final],
                             "bid_avg_stars":[p[4] for p in uid_bid_star_final], "bid_var": [p[5] for p in uid_bid_star_final],"uid_num_fans":[p[6] for p in uid_bid_star_final],
                             "uid_num_reviews":[p[7] for p in uid_bid_star_final],"uid_avg_stars":[p[8] for p in uid_bid_star_final], "uid_var":[p[9] for p in uid_bid_star_final],
                             "uid_num_friends":[p[10] for p in uid_bid_star_final], "uid_num_interactions": [p[11] for p in uid_bid_star_final]})
    df_train_target = pd.DataFrame({"score":[p[12] for p in uid_bid_star_final]})
    scaler = preprocessing.StandardScaler()
    scaler.fit(df_train_features)
    df_train_features_scaled = scaler.transform(df_train_features)

    # set the model
    model = xgb.XGBRegressor()
    '''
    clf = GridSearchCV(model, {'max_depth': [8,9], 'learning_rate': [0.08,0.1,0.15],
                               "colsample_bytree": [0.2,0.3,0.4],
                               "subsample": [0.5,0.6], "alpha": [0], "random_state": [0]})
    '''
    clf = xgb.XGBRegressor(colsample_bytree=0.4, learning_rate=0.1, \
                              max_depth=8, alpha=0, n_estimators=80, subsample=0.6, random_state=0)
    clf.fit(df_train_features_scaled, df_train_target)
    # print("\n The best parameters across ALL searched params:\n", clf.best_params_)
    # print("training_RMSE:", mean_squared_error(y_true = df_train_target['score'], y_pred = clf.predict(df_train_features_scaled), squared=False))
    # exit()
    # join the data to create the test dataset
    # for bid (treat all null restaurant as 3)
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
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], x[5], dealWithNaN(x, 6, 1))))
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
        map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3],x[1][1][4],x[1][1][5])).collect()
    # print(uid_bid_star_final_test)
    # exit()
    # prepare for the machine learning train data
    df_test_features = pd.DataFrame({"num_photo": [p[2] for p in uid_bid_star_final_test], "bid_num_review": [p[3] for p in uid_bid_star_final_test],
         "bid_avg_stars": [p[4] for p in uid_bid_star_final_test], "bid_var": [p[5] for p in uid_bid_star_final_test],"uid_num_fans": [p[6] for p in uid_bid_star_final_test],
         "uid_num_reviews": [p[7] for p in uid_bid_star_final_test], "uid_avg_stars": [p[8] for p in uid_bid_star_final_test],"uid_var":[p[9] for p in uid_bid_star_final_test],
         "uid_num_friends":[p[10] for p in uid_bid_star_final_test], "uid_num_interactions": [p[11] for p in uid_bid_star_final_test]})
    # df_test_target = pd.DataFrame({"score": [p[8] for p in uid_bid_star_final_test]})
    df_test_features_scaled = scaler.transform(df_test_features)
    y_pred = clf.predict(df_test_features_scaled)
    df_final = pd.DataFrame({"user_id":[p[0] for p in uid_bid_star_final_test],"business_id":[p[1] for p in uid_bid_star_final_test],"prediction": y_pred})
    # print("testing_RMSE:",mean_squared_error(y_true=df_test_target['score'], y_pred=clf.predict(df_test_features_scaled),squared=False))
    end_time = time.time()
    duration = end_time - start_time
    print("Duration:", duration)

    df_final.to_csv(output_filepath,index=False)


