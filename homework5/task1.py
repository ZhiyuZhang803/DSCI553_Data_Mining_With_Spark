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
            p = 181081
            value = ((a * uid_int + b) % p) % 69997
            return value
        hash_function_list.append(hashFunc)
    return hash_function_list

def calculate_flase_pos(stream_users):
    # check each number in the stream
    count = 0
    record = [0] * stream_size
    for i in range(len(stream_users)):
        temp_list = []
        hash_number = myhashs(stream_users[i])
        for pos in hash_number:
            if bit_array[pos] == 0:
                break
            else:
                temp_list.append(1)
        if len(temp_list) == 5:
            record[i] = 1
        if stream_users[i] not in previous_users and record[i] == 1:
            count += 1

    temp_sum = sum(record)
    sum2 = stream_size - temp_sum
    rate = count / (count + sum2)
    return rate

if __name__ == '__main__':

    random.seed(553)

    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file = sys.argv[4]

    # input_file = "users.txt"
    # stream_size = 100
    # num_of_asks = 30
    # output_file = "task1Result.csv"

    start_time = time.time()
    # create bit array
    bit_array = [0] * 69997
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

        rate = calculate_flase_pos(stream_users)
        result.append((_, rate))

        for user in stream_users:
            previous_users.add(user)
            for pos in myhashs(user):
                bit_array[pos] = 1

    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'FPR'])
        for i in result:
            writer.writerow(i)
    print("Duration:", time.time() - start_time)