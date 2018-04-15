result = []


def a_side(val1, val2):
    if ((val1 - val2) <= 1) & ((val1 - val2) >= -1):
        return True
    else:
        return False


def is_answer(x):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10=x
    # 第二题
    if x5 > 1:
        if not x2 == x5 - 2:
            return False
    else:
        if not x2 == x5 + 2:
            return False
    # 第三题
    if x3 == x6 == x2:
        if not x3 == 3:
            return False
    elif x6 == x2 == x4:
        if not x3 == 0:
            return False
    elif x3 == x2 == x4:
        if not x3 == 1:
            return False
    elif x6 == x3 == x4:
        if not x3 == 2:
            return False
    else:
        return False
    # 第四题
    if x1 == x5:
        if not x4 == 0:
            return False
    elif x2 == x7:
        if not x4 == 1:
            return False
    elif x1 == x9:
        if not x4 == 2:
            return False
    elif x6 == x10:
        if not x4 == 3:
            return False
    else:
        return False
    # 第五题
    if x5 == 0:
        if not x5 == x8:
            return False
    elif x5 == 1:
        if not x5 == x4:
            return False
    elif x5 == 2:
        if not x5 == x9:
            return False
    elif x5 == 3:
        if not x5 == x7:
            return False
    # 第六题
    if x6 == 0:
        if not x2 == x4 == x8:
            return False
    elif x6 == 1:
        if not x1 == x6 == x8:
            return False
    elif x6 == 2:
        if not x3 == x10 == x8:
            return False
    elif x6 == 3:
        if not x5 == x9 == x8:
            return False
    # 第七题
    n_list = [x.count(2), x.count(1), x.count(0), x.count(3), ]
    n_min = min(n_list)
    n_max = max(n_list)
    n_diff = n_max - n_min
    n_index = n_list.index(n_min)
    if not x7 == n_index:
        return False

    # 第八题
    if x8 == 0:
        if a_side(x7, x1):
            return False
    elif x8 == 1:
        if a_side(x5, x1):
            return False
    elif x8 == 2:
        if a_side(x2, x1):
            return False
    elif x8 == 3:
        if a_side(x10, x1):
            return False
    # 第九题
    if x1 == x6:
        if x9 == 0:
            if x6 == x5:
                return False
        if x9 == 1:
            if x10 == x5:
                return False
        if x9 == 2:
            if x2 == x5:
                return False
        if x9 == 3:
            if x9 == x5:
                return False
    else:
        if x9 == 0:
            if not x6 == x5:
                return False
        if not x9 == 1:
            if x10 == x5:
                return False
        if not x9 == 2:
            if x2 == x5:
                return False
        if not x9 == 3:
            if x9 == x5:
                return False
    # 第十题
    if x10 == 0:
        if not n_diff == 3:
            return False
    elif x10 == 1:
        if not n_diff == 2:
            return False
    elif x10 == 2:
        if not n_diff == 4:
            return False
    elif x10 == 3:
        if not n_diff == 1:
            return False

    return True


for x1 in range(4):
    for x2 in range(4):
        for x3 in range(4):
            for x4 in range(4):
                for x5 in range(4):
                    for x6 in range(4):
                        for x7 in range(4):
                            for x8 in range(4):
                                for x9 in range(4):
                                    for x10 in range(4):
                                        x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
                                        if is_answer(x):
                                            result.append(x)

for i, val in enumerate(result):
    print(i, val)
