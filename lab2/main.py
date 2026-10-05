import csv
import math
import numpy as np
import matplotlib.pyplot as plt


def read_data(filename="data_var4.csv"):
    """Зчитує вхідні масиви x та y із вказаного CSV-файлу."""
    x, y = [], []
    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['tasks']))
            y.append(float(row['cost']))
    return np.array(x, dtype=float), np.array(y, dtype=float)


def omega_k(x_val, x_nodes, k):
    """
    Обчислює значення базисного добутку w_k(x) = prod_{i=0}^{k-1} (x - x_i).
    w_0(x) = 1 за означенням.
    """
    prod = 1.0
    for i in range(k):
        prod *= (x_val - x_nodes[i])
    return prod


def divided_differences_table(x_nodes, y_nodes):
    """Будує повну таблицю розділених різниць для довільної сітки."""
    n = len(x_nodes)
    table = np.zeros((n, n), dtype=float)
    table[:, 0] = y_nodes

    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x_nodes[i + j] - x_nodes[i])
    return table


def newton_polynomial(x_val, x_nodes, table):
    """Обчислює значення многочлена Ньютона у точці x_val."""
    n = len(x_nodes)
    result = table[0, 0]
    for k in range(1, n):
        result += table[0, k] * omega_k(x_val, x_nodes, k)
    return result


def forward_diff_table(y_nodes):
    """Таблиця скінченних різниць Delta^k y_0 (для рівновіддалених вузлів)."""
    n = len(y_nodes)
    diff = np.zeros((n, n), dtype=float)
    diff[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            diff[i, j] = diff[i + 1, j - 1] - diff[i, j - 1]
    return diff


def factorial_polynomial(x_val, x_nodes, y_nodes):
    """
    Обчислення через факторіальний многочлен за скінченними різницями:
    t = (x - x0) / h;  t^(k) = t(t-1)...(t-k+1)
    """
    h = x_nodes[1] - x_nodes[0]
    t = (x_val - x_nodes[0]) / h
    diffs = forward_diff_table(y_nodes)

    res = diffs[0, 0]
    t_factorial = 1.0
    for k in range(1, len(x_nodes)):
        t_factorial *= (t - (k - 1))
        res += (diffs[0, k] / math.factorial(k)) * t_factorial
    return res


def main():

    x_nodes, y_nodes = read_data("data_var4.csv")

    table_diff = divided_differences_table(x_nodes, y_nodes)

    x_target = 15000.0
    pred_newton = newton_polynomial(x_target, x_nodes, table_diff)

    x_uniform_5 = np.linspace(1000, 20000, 5)
    y_uniform_5 = np.array([newton_polynomial(xi, x_nodes, table_diff) for xi in x_uniform_5])
    pred_factorial = factorial_polynomial(x_target, x_uniform_5, y_uniform_5)

    print("\n--- РЕЗУЛЬТАТИ ВАРІАНТА 4 ---")
    print(f"Цільова кількість задач (Tasks): {x_target}")
    print(f"Прогнозована вартість (Ньютон, нерівномірна сітка): {pred_newton:.4f} $")
    print(f"Прогнозована вартість (Факторіальний многочлен):   {pred_factorial:.4f} $")
    print(f"Абсолютна різниця між методами:                   {abs(pred_newton - pred_factorial):.6e}")

    # f(x) = 0.2 + 0.00035 * x + 0.05 * sin(x / 1500)
    def cost_analytical(x):
        return 0.2 + 0.00035 * x + 0.05 * np.sin(x / 1500.0)

    a, b = 1000.0, 20000.0
    x_dense = np.linspace(a, b, 500)
    y_analytical = cost_analytical(x_dense)

    node_counts = [5, 10, 20]
    results_nodes = {}

    for n in node_counts:
        # Рівномірне розбиття
        x_n = np.linspace(a, b, n)
        y_n = cost_analytical(x_n)
        tbl_n = divided_differences_table(x_n, y_n)

        y_interp = np.array([newton_polynomial(xi, x_n, tbl_n) for xi in x_dense])
        error = np.abs(y_analytical - y_interp)
        results_nodes[n] = (x_n, y_n, y_interp, error)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    x_plot = np.linspace(1000, 20000, 250)
    y_plot = [newton_polynomial(xi, x_nodes, table_diff) for xi in x_plot]
    plt.plot(x_plot, y_plot, 'b-', label="Многочлен Ньютона N4(x)")
    plt.scatter(x_nodes, y_nodes, color='red', zorder=5, label="Експериментальні точки")
    plt.plot(x_target, pred_newton, 'g*', markersize=14, label=f"Прогноз Tasks=15000 ({pred_newton:.2f}$)")
    plt.title("Варіант 4: Прогноз вартості Cost(Tasks)")
    plt.xlabel("Tasks")
    plt.ylabel("Cost ($)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(x_dense, y_analytical, 'k--', linewidth=2, label="Еталонна функція f(x)")
    colors = ['orange', 'purple', 'teal']
    for idx, n in enumerate(node_counts):
        _, _, y_interp, _ = results_nodes[n]
        plt.plot(x_dense, y_interp, color=colors[idx], label=f"Інтерполяція n = {n}")
    plt.title("Поведінка полінома Ньютона при n = 5, 10, 20")
    plt.xlabel("Tasks")
    plt.ylabel("Cost ($)")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    for idx, n in enumerate(node_counts):
        _, _, _, err = results_nodes[n]
        plt.semilogy(x_dense, err + 1e-16, color=colors[idx], label=f"Похибка n = {n}")
    plt.title("Похибка |f(x) - N_n(x)| (логарифмічна шкала)")
    plt.xlabel("Tasks")
    plt.ylabel("Похибка")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    w_5 = [omega_k(xi, results_nodes[5][0], 5) for xi in x_dense]
    plt.plot(x_dense, w_5, 'm-', label="w_5(x)")
    plt.title("Базисний поліном вузлів w_5(x)")
    plt.xlabel("Tasks")
    plt.ylabel("w_5(x)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("variant4_analysis.png", dpi=300)
    print("Графіки збережено у файл 'variant4_analysis.png'.")
    plt.show()


if __name__ == "__main__":
    main()