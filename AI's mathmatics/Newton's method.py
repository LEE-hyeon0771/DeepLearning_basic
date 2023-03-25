def f(x):
    return pow(x, 4) - 6*x + 1

def fprime(x):
    return 4 * pow(x, 3) - 6

guess = 0

for i in range(1, 5):
    nextGuess = guess - f(guess)/fprime(guess)
    print(nextGuess)
    guess = nextGuess
