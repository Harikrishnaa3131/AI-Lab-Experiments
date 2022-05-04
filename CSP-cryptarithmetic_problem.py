import itertools

def get_value_012(word, substitution):
    s = 0
    factor = 1
    for letter in reversed(word):
        s += factor * substitution[letter]
        factor *= 10
    return s

def solve_012(equation):
    left, right = equation.lower().replace(' ', '').split('=')
    left = left.split('+')
    letters = set(right)
    for word in left:
        for letter in word:
            letters.add(letter)
    letters = list(letters)

    digits = range(10)
    for perm in itertools.permutations(digits, len(letters)):
        sol = dict(zip(letters, perm))

        if sum(get_value_012(word, sol) for word in left) == get_value_012(right, sol):
            print(' + '.join(str(get_value_012(word, sol)) for word in left) + " = {} (mapping: {})".format(get_value_012, sol))


solve_012('ODD+ODD=EVEN')