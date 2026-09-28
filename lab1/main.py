import numpy as np
import matplotlib.pyplot as plt


def solve_tridiagonal(alpha, beta, gamma, delta):
    """
    Розв'язання тридіагональної СЛАР методом прогонки:
    alpha[i] * x[i-1] + beta[i] * x[i] + gamma[i] * x[i+1] = delta[i]
    """
    n = len(delta)
    A = np.zeros(n)
    B = np.zeros(n)

    # Пряма прогонка
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]
    for i in range(1, n - 1):
        denom = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denom

    # Зворотна прогонка
    x = np.zeros(n)
    denom_n = alpha[-1] * A[-2] + beta[-1]
    x[-1] = (delta[-1] - alpha[-1] * B[-2]) / denom_n

    for i in range(n - 2, -1, -1):
        x[i] = A[i] * x[i + 1] + B[i]

    return x


def compute_spline_coefficients(x_nodes, y_nodes):
    """
    Обчислення коефіцієнтів a, b, c, d кубічного сплайна:
    S_i(x) = a_i + b_i*(x - x_{i-1}) + c_i*(x - x_{i-1})^2 + d_i*(x - x_{i-1})^3,  i = 1..m
    де m = len(x_nodes) - 1.
    """
    m = len(x_nodes) - 1
    h = np.diff(x_nodes)

    # Формування тридіагональної системи для c_1..c_m (розмірність m)
    alpha = np.zeros(m)
    beta = np.zeros(m)
    gamma = np.zeros(m)
    delta = np.zeros(m)

    # Крайова умова: c_1 = 0
    beta[0] = 1.0
    gamma[0] = 0.0
    delta[0] = 0.0

    # Внутрішні рівняння для c_2..c_{m-1} (індекси k = 1..m-2)
    for k in range(1, m - 1):
        alpha[k] = h[k - 1]
        beta[k] = 2.0 * (h[k - 1] + h[k])
        gamma[k] = h[k]
        delta[k] = 3.0 * ((y_nodes[k + 1] - y_nodes[k]) / h[k] - (y_nodes[k] - y_nodes[k - 1]) / h[k - 1])

    # Крайова умова для c_m на останньому сегменті:
    # h_{m-1}*c_{m-1} + 2*(h_{m-1} + h_m)*c_m = 3*(((y_m - y_{m-1})/h_m) - ((y_{m-1} - y_{m-2})/h_{m-1}))
    if m > 1:
        alpha[-1] = h[-2]
        beta[-1] = 2.0 * (h[-2] + h[-1])
        gamma[-1] = 0.0
        delta[-1] = 3.0 * ((y_nodes[m] - y_nodes[m - 1]) / h[-1] - (y_nodes[m - 1] - y_nodes[m - 2]) / h[-2])

    c = solve_tridiagonal(alpha, beta, gamma, delta)

    # Розрахунок коефіцієнтів a_i, b_i, d_i
    a = y_nodes[:-1].copy()
    b = np.zeros(m)
    d = np.zeros(m)

    for i in range(m - 1):
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])
        b[i] = (y_nodes[i + 1] - y_nodes[i]) / h[i] - (h[i] / 3.0) * (c[i + 1] + 2.0 * c[i])

    # Останній відрізок
    d[-1] = -c[-1] / (3.0 * h[-1])
    b[-1] = (y_nodes[-1] - y_nodes[-2]) / h[-1] - (2.0 / 3.0) * h[-1] * c[-1]

    return a, b, c, d, (alpha, beta, gamma, delta)


def evaluate_spline(x_eval, x_nodes, a, b, c, d):
    """Обчислення значень сплайна у заданих точках x_eval"""
    y_eval = np.zeros_like(x_eval)
    for idx, x in enumerate(x_eval):
        # Пошук підінтервалу
        i = np.searchsorted(x_nodes, x) - 1
        i = max(0, min(i, len(a) - 1))
        dx = x - x_nodes[i]
        y_eval[idx] = a[i] + b[i] * dx + c[i] * (dx ** 2) + d[i] * (dx ** 3)
    return y_eval


# Вхідні дані табуляції
distances = np.array([
    0.00, 123.85, 213.45, 323.50, 418.87, 517.41, 587.78, 771.72,
    932.77, 1118.27, 1370.81, 1610.05, 1821.16, 1997.39, 2149.88,
    2320.63, 2508.86, 2713.03, 2904.98, 3020.67, 3069.34
])

elevations = np.array([
    1264.00, 1285.00, 1285.00, 1333.00, 1310.00, 1318.00, 1318.00,
    1339.00, 1375.00, 1417.00, 1486.00, 1524.00, 1553.00, 1630.00,
    1757.00, 1794.00, 1828.00, 1887.00, 1975.00, 1975.00, 2031.00
])

# Розрахунок повного сплайна (21 вузол)
a_full, b_full, c_full, d_full, tridiag = compute_spline_coefficients(distances, elevations)
alpha, beta, gamma, delta = tridiag

print("Коефіцієнти СЛАР для c_i (alpha, beta, gamma, delta):")
print(" i |    alpha    |    beta     |    gamma    |    delta")
print("-" * 55)
for i in range(len(delta)):
    print(f"{i+1:2d} | {alpha[i]:11.4f} | {beta[i]:11.4f} | {gamma[i]:11.4f} | {delta[i]:11.4f}")

print("\nКоефіцієнти кубічних сплайнів:")
print(" i |      a_i      |      b_i      |      c_i      |      d_i")
print("-" * 58)
for i in range(len(a_full)):
    print(f"{i+1:2d} | {a_full[i]:13.4f} | {b_full[i]:13.6f} | {c_full[i]:13.6e} | {d_full[i]:13.6e}")

# Додаткові фізичні метрики
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, len(elevations)))
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, len(elevations)))
mass = 80.0
g = 9.81
energy_j = mass * g * total_ascent

print("\nХарактеристики маршруту:")
print(f"Загальна довжина: {distances[-1]:.2f} м")
print(f"Сумарний набір висоти: {total_ascent:.2f} м")
print(f"Сумарний спуск: {total_descent:.2f} м")
print(f"Механічна робота: {energy_j / 1000:.2f} кДж ({energy_j / 4184:.2f} ккал)")

# Порівняння для 10, 15 та 21 вузла
xx = np.linspace(distances[0], distances[-1], 500)
yy_full = evaluate_spline(xx, distances, a_full, b_full, c_full, d_full)

# Вибірка вузлів
idx_10 = np.round(np.linspace(0, len(distances) - 1, 10)).astype(int)
idx_15 = np.round(np.linspace(0, len(distances) - 1, 15)).astype(int)

a_10, b_10, c_10, d_10, _ = compute_spline_coefficients(distances[idx_10], elevations[idx_10])
yy_10 = evaluate_spline(xx, distances[idx_10], a_10, b_10, c_10, d_10)

a_15, b_15, c_15, d_15, _ = compute_spline_coefficients(distances[idx_15], elevations[idx_15])
yy_15 = evaluate_spline(xx, distances[idx_15], a_15, b_15, c_15, d_15)

# Градієнт
grad_full = np.gradient(yy_full, xx) * 100.0
print(f"Максимальний підйом: {np.max(grad_full):.2f}%")
print(f"Максимальний спуск: {np.min(grad_full):.2f}%")
print(f"Середній градієнт: {np.mean(np.abs(grad_full)):.2f}%")

# Побудова графіків
plt.figure(figsize=(12, 9))

# 1. Сплайни для 10, 15, 21 вузлів
plt.subplot(2, 1, 1)
plt.plot(distances, elevations, 'ko', label='Вихідні вузли GPS (21)')
plt.plot(xx, yy_10, '--', label='Сплайн (10 вузлів)')
plt.plot(xx, yy_15, '-.', label='Сплайн (15 вузлів)')
plt.plot(xx, yy_full, 'b-', label='Сплайн (21 вузол)')
plt.title('Інтерполяція висотного профілю маршруту Заросляк — Говерла')
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Висота (м)')
plt.grid(True)
plt.legend()

# 2. Абсолютна похибка відносно повного набору (21 вузол)
plt.subplot(2, 1, 2)
plt.plot(xx, np.abs(yy_full - yy_10), 'r--', label='Похибка ε для 10 вузлів')
plt.plot(xx, np.abs(yy_full - yy_15), 'g-.', label='Похибка ε для 15 вузлів')
plt.title('Абсолютна похибка інтерполяції ε = |y_full - y_approx|')
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Похибка (м)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()