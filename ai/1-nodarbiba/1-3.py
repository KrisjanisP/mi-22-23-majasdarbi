def f(x, b):
    result = 1
    if b > 0:
        result = (x ** b + b) * f(x, b - 1)
    return result


print(f(4, 3))
