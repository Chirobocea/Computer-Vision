def check_all_letters_used(matrix):
    count = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != '-':
                count = count + 1
    if count == 7:
        return 50
    else:
        return 0

def get_all_words_created(diff, matrix):
    all_words = diff.copy()

    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):

            if diff[i][j] != '-':

                k = 1
                while i-k >= 0:
                    if matrix[i-k][j] != '-':
                        all_words[i-k][j] = matrix[i-k][j]
                    else:
                        break
                    k = k + 1
                k = 1
                while i+k < 15:
                    if matrix[i+k][j] != '-':
                        all_words[i+k][j] = matrix[i+k][j]
                    else:
                        break
                    k = k + 1
                k = 1
                while j-k >= 0:
                    if matrix[i][j-k] != '-':
                        all_words[i][j-k] = matrix[i][j-k]
                    else:
                        break
                    k = k + 1
                k = 1
                while j+k < 15:
                    if matrix[i][j+k] != '-':
                        all_words[i][j+k] = matrix[i][j+k]
                    else:
                        break
                    k = k + 1
                k = 1
    
    return all_words

def calculate_points(diff, matrix):

    letter_points = {
        "A": 1,
        "B": 9,
        "C": 1,
        "D": 2,
        "E": 1,
        "F": 8,
        "G": 9,
        "H": 10,
        "I": 1,
        "J": 10,
        "L": 1,
        "M": 4,
        "N": 1,
        "O": 1,
        "P": 2,
        "R": 1,
        "S": 1,
        "T": 1,
        "U": 1,
        "V": 8,
        "X": 10,
        "Z": 10,
        "?": 0
    }

    # 6 -> tripleaza valoarea cuvantului
    # 3 -> tripleaza valoarea literei
    # 4 -> dubleaza valoarea cuvantului
    # 2 -> dubleaza valoarea literei
    special_spot=[
        [6, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 6],
        [0, 4, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 4, 0],
        [0, 0, 4, 0, 0, 0, 2, 0, 2, 0, 0, 0, 4, 0, 0],
        [2, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 2],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0],
        [0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0],
        [6, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 6],
        [0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0],
        [0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
        [2, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 2],
        [0, 0, 4, 0, 0, 0, 2, 0, 2, 0, 0, 0, 4, 0, 0],
        [0, 4, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 4, 0],
        [6, 0, 0, 2, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 6]
    ]


    points = check_all_letters_used(diff)

    all_words = get_all_words_created(diff, matrix)

    for i in range(all_words.shape[0]):
        word_points = 0
        double = False
        triple = False
        len = 0
        is_new_word = False
        for j in range(all_words.shape[1]):
            if all_words[i][j] != '-':
                len = len + 1
                if diff[i][j] != '-':
                    is_new_word = True
                    if special_spot[i][j] == 0:
                        word_points = word_points + letter_points[all_words[i][j]]
                    elif special_spot[i][j] == 2:
                        word_points = word_points + 2*letter_points[all_words[i][j]]
                    elif special_spot[i][j] == 3:
                        word_points = word_points + 3*letter_points[all_words[i][j]]
                    elif special_spot[i][j] == 4:
                        word_points = word_points + letter_points[all_words[i][j]]
                        double = True
                    elif special_spot[i][j] == 6:
                        word_points = word_points + letter_points[all_words[i][j]]
                        triple = True
                else:
                    word_points = word_points + letter_points[all_words[i][j]]

        if double:
            word_points = 2*word_points
        if triple:
            word_points = 3*word_points

        if len > 1 and is_new_word:
            points = points + word_points

    for j in range(all_words.shape[0]):
        word_points = 0
        double = False
        triple = False
        len = 0
        is_new_word = False
        for i in range(all_words.shape[1]):
            if all_words[i][j] != '-':
                len = len + 1
                if diff[i][j] != '-':
                    is_new_word = True
                    if special_spot[i][j] == 0:
                        word_points = word_points + letter_points[all_words[i][j]]
                    elif special_spot[i][j] == 2:
                        word_points = word_points + 2*letter_points[all_words[i][j]]
                    elif special_spot[i][j] == 3:
                        word_points = word_points + 3*letter_points[all_words[i][j]]
                    elif special_spot[i][j] == 4:
                        word_points = word_points + letter_points[all_words[i][j]]
                        double = True
                    elif special_spot[i][j] == 6:
                        word_points = word_points + letter_points[all_words[i][j]]
                        triple = True
                else:
                    word_points = word_points + letter_points[all_words[i][j]]
        if double:
            word_points = 2*word_points
        if triple:
            word_points = 3*word_points

        if len > 1 and is_new_word:
            points = points + word_points

    return str(points)