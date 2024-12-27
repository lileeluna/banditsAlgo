def which_comes_first(a, b, list):
    if(a == b):
        print("Invalid arm inputs. The two arms cannot be the same.")
        return
    if a in list and b in list:
        for i in list:
            if(i == a):
                return 0
            if(i == b):
                return 1
    if a in list and b not in list:
        return 0
    elif a not in list and b in list:
        return 1
    else:
        return -1

def print_three():
    list = []
    for i in range(100):
        list.append(i)
    combinations = []
    group_len = 10
    a = 2
    b = 3
    count_a = 0
    count_b = 0
    a_ahead = 0
    b_ahead = 0
    a_obs = 0
    b_obs = 0
    print("Combinations:")
    length = len(list)
    for i in range(length):
        temp = []
        for l in range(group_len):
            temp.append(list[(i + l) % length])
        combinations.append(temp)
        val = which_comes_first(a, b, temp)
        if(val == 0):
            count_a += 1
        elif(val == 1):
            count_b += 1
        else:
            continue

    for i in range(len(combinations)):
        print(*combinations[i], sep='\t')

    print()
    print("First Arm: " + str(count_a))
    print("Second Arm: " + str(count_b))

print_three()