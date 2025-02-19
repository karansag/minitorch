import minitorch
import random


# y = 2x + 3
def generate_data(n: int = 10) -> list[tuple[int, int]]:
    randoms = [random.random() * 5 - 2.5 for _ in range(10)]
    return [(x, 2 * x + 3) for x in randoms]


m = minitorch.Scalar(0.5)
b = minitorch.Scalar(0)


def run_step(m, b, lr=0.01):
    print(f"m prev: {m}")
    print(f"b prev: {b}")
    process_data(m, b)
    print(f"m deriv: {m.derivative}")
    print(f"b deriv: {b.derivative}")
    m.data = m.data - lr * m.derivative
    b.data = b.data - lr * b.derivative
    print(f"m next: {m}")
    print(f"b next: {b}")
    m.derivative = 0
    b.derivative = 0


def process_data(m, b):
    data_pts = generate_data(10)

    loss = 0
    for x, y in data_pts:
        x = minitorch.Scalar(x)
        y_pred = m * x + b
        diff = (y_pred - y) ** 2
        loss += diff.data
        diff.backward()

    print(f"total loss: {loss}")
    return (m, b)
