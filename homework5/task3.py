from blackbox import BlackBox
import random
import sys
import time
import csv


if __name__ == '__main__':

    random.seed(553)

    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_file = sys.argv[4]

    # input_file = "users.txt"
    # stream_size = 100
    # num_of_asks = 30
    # output_file = "task3Result.csv"

    start_time = time.time()
    bx = BlackBox()
    result = []
    final = []
    count = 0

    for _ in range(num_of_asks):
        stream_users = bx.ask(input_file, stream_size)
        if _ == 0:
            result = result + stream_users
            count += stream_size
        else:
            for single in stream_users:
                count += 1
                prob = random.random()
                if prob < 100/count:
                    idx = random.randint(0, 99)
                    result[idx] = single

        final.append((count, result[0], result[20], result[40], result[60], result[80]))

    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['seqnum', '0_id', "20_id", '40_id', '60_id', '80_id'])
        for i in final:
            writer.writerow(i)
    print("Duration:", time.time() - start_time)