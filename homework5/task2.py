from blackbox import BlackBox
import binascii
import random
import sys
import time
import csv


def myhashs(user_id):
    result = []
    # user_int = int(binascii.hexlify(user_id.encode("utf8")),16)
    for f in hash_function_list:
        result.append(f(user_id))
    return result

def generate_hash(n):
    hash_function_list = []
    for k in range(n):
        def hashFunc(x):
            uid_int = int(binascii.hexlify(x.encode('utf8')), 16)
            a = A_list[k]
            b = B_list[k]
            p = 1993
            value = ((a * uid_int + b) % p) % 69997
            return value
        hash_function_list.append(hashFunc)
    return hash_function_list

def calculate_estimation(stream_users):
    # check each number in the stream
    # visited_candidates = set(stream_users)
    record = [0] * 5
    for i in range(len(stream_users)):
        hash_number = myhashs(stream_users[i])

        for single_num in range(len(hash_number)):
            # convert to bin
            bin_num = bin(hash_number[single_num])
            num_zero = len(bin_num.split("1")[-1])
            if num_zero >= record[single_num]:
                record[single_num] = num_zero
    record_corr = [2**i for i in record]
    estimation = round(sum(record_corr)/5)
    return estimation


if __name__ == '__main__':

    random.seed(553)

    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file = sys.argv[4]

    # input_file = "users.txt"
    # stream_size = 300
    # num_of_asks = 30
    # output_file = "task2Result.csv"

    start_time = time.time()
    # create a set to store users
    previous_users = set()
    # number of hash
    n = 5
    # generate hash functions
    A_list = random.sample(range(1, 70000), n)
    B_list = random.sample(range(0, 70000), n)

    # start hash
    bx = BlackBox()
    result = []
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_file, stream_size)
        hash_function_list = generate_hash(5)
        est_num = calculate_estimation(stream_users)
        result.append((_, stream_size, est_num))

    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Ground Truth', "Estimation"])
        for i in result:
            writer.writerow(i)
    print("Duration:", time.time() - start_time)