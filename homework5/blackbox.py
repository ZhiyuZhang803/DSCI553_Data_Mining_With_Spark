import random

class BlackBox:

    def ask(self, file_name, stream_size):
        place_holder = [0 for i in range(stream_size)]
        # read file
        users = open(file_name, "r").readlines()
        for ele in range(stream_size):
            index = random.randint(0, len(users)-1)
            place_holder[ele] = users[index].rstrip("\n")
        return place_holder