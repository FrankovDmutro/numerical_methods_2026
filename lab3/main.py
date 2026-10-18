import csv
import math
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# 1. Функція зчитування вхідних даних з текстового/CSV файлу
# -------------------------------------------------------------
def read_data(filename: str):
    """Зчитує масиви x (місяці) та f (температури) з CSV файлу."""
    x = []
    f = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Пропуск заголовка (Month, Temp)
        for row in reader:
            if row:
                x.append(float(row[0].strip()))
                f.append(float(row[1].strip()))
    return x, f


# -------------------------------------------------------------
# 2. Функції формування матриці B та вектора C для МНК
# -------------------------------------------------------------
def form_matrix(x: list, m: int):
    """
    Формує симетричну матрицю B розміром (m+1) x (m+1).
    b_{k, l} = sum_{i=0}^n (x_i^(k + l))
    """
    size = m + 1
    B = [[0.0] * size for _ in range(size)]
    for k in range(size):
        for l in range(size):
            power = k + l
            B[k][l] = sum(xi ** power for xi in x)
    return B


def form_vector(x: list, f: list, m: int):
    """
    Формує вектор вільних членів C розміром (m+1).
    c_k = sum_{i=0}^n (f_i * x_i^k)
    """
    size = m + 1
    C = [0.0] * size
    for k in range(size):
        C[k] = sum(fi * (xi ** k) for xi, fi in zip(x, f))
    return C


# -------------------------------------------------------------
# 3. Розв'язування СЛАР методом Гауса з вибором головного елемента
# -------------------------------------------------------------
def gauss_solve(A: list, b: list):
    """
    Прямий та зворотний хід методу Гауса з вибором 
    найбільшого по модулю елемента в стовпці.
    """
    n = len(b)
    # Створюємо локальні копії
    M = [row[:] for row in A]
    rhs = b[:]

    # Прямий хід
    for k in range(n):
        # Пошук головного елемента по k-му стовпцю
        max_row = k
        max_val = abs(M[k][k])
        for i in range(k + 1, n):
            if abs(M[i][k]) > max_val:
                max_val = abs(M[i][k])
                max_row = i

        if abs(M[max_row][k]) < 1e-14:
            raise ValueError(f"Матриця є виродженою або чисельно нестійкою на кроці {k}.")

        # Перестановка поточного та знайденого рядків
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
            rhs[k], rhs[max_row] = rhs[max_row], rhs[k]

        # Занулення елементів під головним
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n):
                M[i][j] -= factor * M[k][j]
            rhs[i] -= factor * rhs[k]

    # Зворотний хід
    x_sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_row = sum(M[i][j] * x_sol[j] for j in range(i + 1, n))
        x_sol[i] = (rhs[i] - sum_row) / M[i][i]

    return x_sol


# -------------------------------------------------------------
# 4. Обчислення значення многочлена та дисперсії
# -------------------------------------------------------------
def polynomial(x_val: float, coef: list):
    """Обчислює phi(x) = sum_{j=0}^m (a_j * x^j)."""
    return sum(a * (x_val ** j) for j, a in enumerate(coef))


def evaluate_polynomial_list(x_list: list, coef: list):
    """Обчислює значення многочлена для списку точок."""
    return [polynomial(xi, coef) for xi in x_list]


def compute_variance(f_true: list, f_approx: list):
    """
    Обчислює дисперсію / середньоквадратичну похибку delta:
    sqrt( sum((phi(x_i) - f_i)^2) / (n + 1) )
    """
    n_points = len(f_true)
    sum_sq_diff = sum((fa - ft) ** 2 for ft, fa in zip(f_true, f_approx))
    return math.sqrt(sum_sq_diff / n_points)


# -------------------------------------------------------------
# Основний цикл виконання
# -------------------------------------------------------------
def main():
    csv_file = "temperature_data.csv"
    x, f = read_data(csv_file)
    n = len(x)  # Кількість вузлів (24)
    x0, xn = x[0], x[-1]

    print(f"Зчитано {n} вузлів сітки на відрізку [{int(x0)}, {int(xn)}].")

    # Перебір степенів m = 1..10
    max_m = 10
    variances = []
    models = {}

    print("\n" + "-" * 50)
    print(f"{'Степінь m':^12} | {'Дисперсія delta':^25}")
    print("-" * 50)

    for m in range(1, max_m + 1):
        try:
            B = form_matrix(x, m)
            C = form_vector(x, f, m)
            coef = gauss_solve(B, C)
            models[m] = coef
            
            f_approx = evaluate_polynomial_list(x, coef)
            var = compute_variance(f, f_approx)
            variances.append((m, var))
            print(f"{m:^12} | {var:^25.4f}")
        except ValueError as err:
            print(f"{m:^12} | Помилка: {err}")

    # Вибір оптимального m
    optimal_m, min_var = min(variances, key=lambda item: item[1])
    print("-" * 50)
    print(f"Оптимальне значення степеня: m = {optimal_m} (з мінімумом дисперсії delta = {min_var:.4f})")

    opt_coef = models[optimal_m]
    print(f"\nКоефіцієнти оптимального многочлена m = {optimal_m}:")
    for j, a in enumerate(opt_coef):
        print(f"  a_{j} = {a:14.6e}")

    # Прогноз на наступні 3 місяці (місяці 25, 26, 27)
    x_future = [25.0, 26.0, 27.0]
    f_future = evaluate_polynomial_list(x_future, opt_coef)
    print("\nПрогноз середньомісячної температури на наступні 3 місяці:")
    for m_f, t_f in zip(x_future, f_future):
        print(f"  Місяць {int(m_f)}: {t_f:.2f} °C")

    # ---------------------------------------------------------
    # Табуляція похибок з кроком h1 = (xn - x0) / (20 * n)
    # Оскільки точні значення f(x) між вузлами спостережень не задані,
    # оцінюємо похибку відносно лінійно інтерпольованих табличних даних.
    # ---------------------------------------------------------
    h1 = (xn - x0) / (20.0 * (n - 1))
    num_steps = int(round((xn - x0) / h1))
    x_tab = [x0 + i * h1 for i in range(num_steps + 1)]

    def get_interpolated_truth(t):
        """Кусково-лінійна інтерполяція табличних даних для табуляції похибки."""
        if t <= x[0]:
            return f[0]
        if t >= x[-1]:
            return f[-1]
        for i in range(len(x) - 1):
            if x[i] <= t <= x[i + 1]:
                ratio = (t - x[i]) / (x[i + 1] - x[i])
                return f[i] + ratio * (f[i + 1] - f[i])
        return f[-1]

    f_interpolated = [get_interpolated_truth(t) for t in x_tab]

    # ---------------------------------------------------------
    # Візуалізація результатів
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(14, 12))

    # 1. Графік залежності величини дисперсії від степеня m
    ax1 = fig.add_subplot(2, 2, 1)
    degrees = [item[0] for item in variances]
    vars_val = [item[1] for item in variances]
    ax1.plot(degrees, vars_val, 'o-', color='tab:red', lw=2, markersize=6)
    ax1.scatter([optimal_m], [min_var], color='blue', s=120, zorder=5, 
                label=f'Мінімум: m={optimal_m} (delta={min_var:.2f})')
    ax1.set_title("1. Залежність дисперсії від степеня m", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Степінь m")
    ax1.set_ylabel("Дисперсія delta")
    ax1.set_xticks(degrees)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 2. Графік фактичних даних, апроксимації та прогнозу
    ax2 = fig.add_subplot(2, 2, 2)
    # Щільна сітка для гладкого відображення кривої разом із прогнозом
    x_smooth = [x0 + i * 0.05 for i in range(int((x_future[-1] - x0) / 0.05) + 1)]
    f_smooth = evaluate_polynomial_list(x_smooth, opt_coef)

    ax2.scatter(x, f, color='black', s=35, zorder=4, label='Фактичні дані (1..24 міс.)')
    ax2.plot(x_smooth, f_smooth, color='royalblue', lw=2, label=f'Апроксимація МНК (m={optimal_m})')
    ax2.scatter(x_future, f_future, color='crimson', marker='s', s=70, zorder=5, 
                label='Прогноз (25..27 міс.)')
    ax2.set_title("2. Фактичні дані, апроксимація та прогноз", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Місяць")
    ax2.set_ylabel("Температура, °C")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    # 3. Графіки похибок epsilon(x) для m = 1..10 (вимога пункту 4 «Ходу роботи»)
    ax3 = fig.add_subplot(2, 1, 2)
    # Відображаємо характерні степені або всі 10
    for m in degrees:
        coef = models[m]
        poly_vals = evaluate_polynomial_list(x_tab, coef)
        err = [abs(ft - fa) for ft, fa in zip(f_interpolated, poly_vals)]
        if m in [1, 2, 4, optimal_m, 10]:
            lw = 2.5 if m == optimal_m else 1.2
            ax3.plot(x_tab, err, label=f'm = {m}', lw=lw)

    ax3.set_title("3. Протабульована похибка epsilon(x) = |f(x) - phi(x)| уздовж відрізка [x0, xn]", 
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel("Місяць")
    ax3.set_ylabel("Похибка, °C")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(ncol=3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()