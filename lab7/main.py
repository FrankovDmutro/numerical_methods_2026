import math
import os
import matplotlib.pyplot as plt
import numpy as np


def generate_diagonally_dominant_matrix(
    n: int = 100,
    low: float = -5.0,
    high: float = 5.0,
    margin: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Генерація випадкової квадратної матриці розмірності n x n
    зі строгим діагональним переважанням:
        |a_ii| > sum_{j != i} |a_ij|
    Для гарантування збіжності методу простої ітерації діагональні елементи
    задаються додатними: a_ii = sum_{j != i} |a_ij| + margin.
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(low, high, (n, n))

    # Забезпечуємо строге діагональне переважання
    for i in range(n):
        row_sum = math.fsum(abs(A[i, j]) for j in range(n) if j != i)
        A[i, i] = row_sum + margin

    return A


def write_matrix(matrix: np.ndarray, filename: str) -> None:
    """
    Запис матриці у текстовий файл із повною точністю (.17e).
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Розмірність: {matrix.shape[0]}x{matrix.shape[1]}\n")
        for row in matrix:
            f.write(" ".join(f"{val:24.17e}" for val in row) + "\n")


def read_matrix(filename: str) -> np.ndarray:
    """
    Зчитування матриці з текстового файлу.
    """
    rows = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows, dtype=np.float64)


def write_vector(vector: np.ndarray, filename: str) -> None:
    """
    Запис вектора у текстовий файл із повною точністю (.17e).
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Розмірність: {len(vector)}\n")
        for val in vector:
            f.write(f"{val:24.17e}\n")


def read_vector(filename: str) -> np.ndarray:
    """
    Зчитування вектора з текстового файлу.
    """
    values = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.append(float(line))
    return np.array(values, dtype=np.float64)


def matrix_vector_mult(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Обчислення добутку матриці на вектор: b_i = sum_{j=1}^n A[i, j] * x[j].
    Для мінімізації похибки округлення застосовано math.fsum.
    """
    n, m = A.shape
    if m != len(x):
        raise ValueError(f"Невідповідність розмірностей: матриця {A.shape}, вектор {len(x)}")
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        result[i] = math.fsum(A[i, j] * x[j] for j in range(m))
    return result


def vector_norm(v: np.ndarray, ord=np.inf) -> float:
    """
    Обчислення норми вектора.
    За замовчуванням обчислюється нескінченна (максимальна) норма ||v||_inf = max_i |v_i|.
    """
    if ord == np.inf:
        return float(np.max(np.abs(v)))
    elif ord == 1:
        return float(np.sum(np.abs(v)))
    elif ord == 2:
        return float(np.sqrt(np.sum(v ** 2)))
    else:
        return float(np.linalg.norm(v, ord=ord))


def matrix_norm(A: np.ndarray, ord=np.inf) -> float:
    """
    Обчислення норми матриці:
      - ord = np.inf: нескінченна норма (максимум сум модулів по рядках);
      - ord = 1: перша норма (максимум сум модулів по стовпцях);
      - ord = 'fro': норма Фробеніуса (евклідова норма).
    """
    if ord == np.inf:
        return float(np.max(np.sum(np.abs(A), axis=1)))
    elif ord == 1:
        return float(np.max(np.sum(np.abs(A), axis=0)))
    elif ord in ("fro", 2):
        return float(np.sqrt(np.sum(A ** 2)))
    else:
        return float(np.linalg.norm(A, ord=ord))


def solve_simple_iteration(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    eps0: float = 1e-14,
    max_iter: int = 2000,
    tau: float | None = None,
    x_exact: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Розв'язання СЛАР методом простої ітерації:
        X^(k+1) = (E - tau * A) * X^(k) + tau * b = X^(k) - tau * (A * X^(k) - b).
    Якщо параметр tau не вказано, обчислюється оптимальне значення:
        tau_opt = 2 / (lambda_min + lambda_max).
    """
    n = len(b)
    if tau is None:
        # Для матриці з додатними власними значеннями
        eigs = np.linalg.eigvals(A)
        real_eigs = np.real(eigs)
        tau = float(2.0 / (np.min(real_eigs) + np.max(real_eigs)))

    x = x0.copy()
    history = []

    # Початковий стан
    r0 = b - matrix_vector_mult(A, x)
    res0 = vector_norm(r0)
    err0 = vector_norm(x - x_exact) if x_exact is not None else None
    history.append({
        "iteration": 0,
        "delta_norm": float("nan"),
        "residual_norm": res0,
        "error_norm": err0,
        "converged": False,
    })

    for k in range(1, max_iter + 1):
        r = b - matrix_vector_mult(A, x)
        x_next = x + tau * r
        delta = vector_norm(x_next - x)
        x = x_next

        r_new = b - matrix_vector_mult(A, x)
        res_norm = vector_norm(r_new)
        err_norm = vector_norm(x - x_exact) if x_exact is not None else None

        converged = (delta < eps0) or (res_norm <= eps0)

        history.append({
            "iteration": k,
            "delta_norm": delta,
            "residual_norm": res_norm,
            "error_norm": err_norm,
            "converged": converged,
        })

        if converged:
            break

    return x, history


def solve_jacobi(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    eps0: float = 1e-14,
    max_iter: int = 2000,
    x_exact: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Розв'язання СЛАР методом Якобі:
        x_i^(k+1) = (b_i - sum_{j != i} a_ij * x_j^(k)) / a_ii.
    """
    n = len(b)
    x = x0.copy()
    history = []

    r0 = b - matrix_vector_mult(A, x)
    res0 = vector_norm(r0)
    err0 = vector_norm(x - x_exact) if x_exact is not None else None
    history.append({
        "iteration": 0,
        "delta_norm": float("nan"),
        "residual_norm": res0,
        "error_norm": err0,
        "converged": False,
    })

    for k in range(1, max_iter + 1):
        x_next = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = math.fsum(A[i, j] * x[j] for j in range(n) if j != i)
            x_next[i] = (b[i] - s) / A[i, i]

        delta = vector_norm(x_next - x)
        x = x_next

        r_new = b - matrix_vector_mult(A, x)
        res_norm = vector_norm(r_new)
        err_norm = vector_norm(x - x_exact) if x_exact is not None else None

        converged = (delta < eps0) or (res_norm <= eps0)

        history.append({
            "iteration": k,
            "delta_norm": delta,
            "residual_norm": res_norm,
            "error_norm": err_norm,
            "converged": converged,
        })

        if converged:
            break

    return x, history


def solve_seidel(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    eps0: float = 1e-14,
    max_iter: int = 2000,
    x_exact: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Розв'язання СЛАР методом Гауса-Зейделя:
        x_i^(k+1) = (b_i - sum_{j < i} a_ij * x_j^(k+1) - sum_{j > i} a_ij * x_j^(k)) / a_ii.
    """
    n = len(b)
    x = x0.copy()
    history = []

    r0 = b - matrix_vector_mult(A, x)
    res0 = vector_norm(r0)
    err0 = vector_norm(x - x_exact) if x_exact is not None else None
    history.append({
        "iteration": 0,
        "delta_norm": float("nan"),
        "residual_norm": res0,
        "error_norm": err0,
        "converged": False,
    })

    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s1 = math.fsum(A[i, j] * x[j] for j in range(i))
            s2 = math.fsum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - s1 - s2) / A[i, i]

        delta = vector_norm(x - x_old)
        r_new = b - matrix_vector_mult(A, x)
        res_norm = vector_norm(r_new)
        err_norm = vector_norm(x - x_exact) if x_exact is not None else None

        converged = (delta < eps0) or (res_norm <= eps0)

        history.append({
            "iteration": k,
            "delta_norm": delta,
            "residual_norm": res_norm,
            "error_norm": err_norm,
            "converged": converged,
        })

        if converged:
            break

    return x, history


def plot_convergence(
    history_sim: list[dict],
    history_jac: list[dict],
    history_sei: list[dict],
    x_exact: np.ndarray,
    x_sim: np.ndarray,
    x_jac: np.ndarray,
    x_sei: np.ndarray,
    eps0: float,
    plot_file: str,
) -> None:
    """
    Побудова графіків збіжності для трьох методів та покомпонентних похибок.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Графік збіжності похибки відносно точного розв'язку (||X^(k) - X*||_inf)
    it_sim = [r["iteration"] for r in history_sim]
    err_sim = [r["error_norm"] for r in history_sim]

    it_jac = [r["iteration"] for r in history_jac]
    err_jac = [r["error_norm"] for r in history_jac]

    it_sei = [r["iteration"] for r in history_sei]
    err_sei = [r["error_norm"] for r in history_sei]

    ax1.semilogy(it_sim, err_sim, "o-", color="#d9534f", lw=2, ms=5, label=f"Проста ітерація ({len(it_sim)-1} іт.)")
    ax1.semilogy(it_jac, err_jac, "s-", color="#0275d8", lw=2, ms=5, label=f"Якобі ({len(it_jac)-1} іт.)")
    ax1.semilogy(it_sei, err_sei, "^-", color="#5cb85c", lw=2, ms=6, label=f"Зейдель ({len(it_sei)-1} іт.)")
    ax1.axhline(eps0, color="gray", linestyle=":", lw=1.5, label=r"Цільова точність $\varepsilon_0 = 10^{-14}$")

    ax1.set_title("Швидкість збіжності ітераційних методів СЛАР", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Номер ітерації $k$", fontsize=11)
    ax1.set_ylabel(r"Похибка $\|X^{(k)} - X^*\|_\infty$ (log scale)", fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=10.5, loc="upper right")

    # 2. Покомпонентна похибка для всіх 100 невідомих (порівняння з початковою)
    indices = np.arange(1, len(x_exact) + 1)
    err_comp_init = np.full(len(x_exact), 1.5)
    # Додаємо мінімальний поріг 1e-16 для коректного відображення нульових машинних похибок на логарифмічній шкалі
    err_comp_sim = np.maximum(np.abs(x_sim - x_exact), 1e-16)
    err_comp_jac = np.maximum(np.abs(x_jac - x_exact), 1e-16)
    err_comp_sei = np.maximum(np.abs(x_sei - x_exact), 1e-16)

    ax2.semilogy(indices, err_comp_init, "--", color="gray", lw=1.5, alpha=0.7, label=r"Початкова похибка $|x_i^{(0)} - 2.5| = 1.5$")
    ax2.semilogy(indices, err_comp_sim, "o-", color="#d9534f", ms=3.5, alpha=0.6, label="Проста ітерація")
    ax2.semilogy(indices, err_comp_jac, "s-", color="#0275d8", ms=3.5, alpha=0.6, label="Якобі")
    ax2.semilogy(indices, err_comp_sei, "^-", color="#5cb85c", ms=4, alpha=0.6, label="Зейдель")

    ax2.set_title("Покомпонентна абсолютна похибка після збіжності ($n = 100$)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Індекс компоненти $i$", fontsize=11)
    ax2.set_ylabel(r"Абсолютна похибка $|x_i - 2.5|$ (log scale)", fontsize=11)
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(fontsize=10, loc="center right")

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    matrix_file = os.path.join(base_dir, "matrix_A.txt")
    vector_file = os.path.join(base_dir, "vector_b.txt")
    plot_file = os.path.join(base_dir, "lab7_convergence.png")

    print("=" * 80)
    print("Лабораторна робота №7: Ітераційні методи розв'язку СЛАР (проста ітерація, Якобі, Зейдель)")
    print("=" * 80)

    # 1. Генерація матриці A розмірності 100x100 з діагональним переважанням
    n = 100
    exact_val = 2.5
    print(f"\n[Крок 1] Генерація матриці A ({n}x{n}) з діагональним переважанням...")
    A_gen = generate_diagonally_dominant_matrix(n=n, low=-5.0, high=5.0, margin=10.0, seed=42)
    write_matrix(A_gen, matrix_file)
    print(f"  -> Матрицю A успішно збережено у: {os.path.basename(matrix_file)}")

    x_exact = np.full(n, exact_val, dtype=np.float64)
    b_gen = matrix_vector_mult(A_gen, x_exact)
    write_vector(b_gen, vector_file)
    print(f"  -> Вектор вільних членів B успішно збережено у: {os.path.basename(vector_file)}")

    # 2. Зчитування даних із текстових файлів
    print(f"\n[Крок 2] Зчитування даних з текстових файлів...")
    A = read_matrix(matrix_file)
    b = read_vector(vector_file)
    print(f"  -> Зчитано матрицю A {A.shape} та вектор b розмірністю {len(b)}")

    # Обчислення норм матриці
    norm_inf = matrix_norm(A, ord=np.inf)
    norm_1 = matrix_norm(A, ord=1)
    norm_fro = matrix_norm(A, ord="fro")
    cond_inf = float(np.linalg.cond(A, p=np.inf))
    print(f"  -> Норми матриці A: ||A||_inf = {norm_inf:.2f}, ||A||_1 = {norm_1:.2f}, ||A||_F = {norm_fro:.2f}")
    print(f"  -> Число обумовленості kappa_inf(A) = {cond_inf:.2f}")

    # 3. Початкове наближення
    x0 = np.full(n, 1.0, dtype=np.float64)
    print(f"\n[Крок 3] Задано початкове наближення x_i^(0) = 1.0 (для всіх i = 1..{n})")

    eps0 = 1e-14

    # 4. Розв'язання методом простої ітерації
    print(f"\n[Крок 4.1] Розв'язання методом простої ітерації (eps0 = {eps0})...")
    x_sim, hist_sim = solve_simple_iteration(A, b, x0, eps0=eps0, x_exact=x_exact)
    k_sim = len(hist_sim) - 1
    err_sim = vector_norm(x_sim - x_exact)
    res_sim = vector_norm(matrix_vector_mult(A, x_sim) - b)
    print(f"  -> Кількість ітерацій: {k_sim}")
    print(f"  -> Фінальна похибка ||X - X*||_inf: {err_sim:.4e}")
    print(f"  -> Фінальна нев'язка ||AX - B||_inf: {res_sim:.4e}")

    # 4.2. Розв'язання методом Якобі
    print(f"\n[Крок 4.2] Розв'язання методом Якобі (eps0 = {eps0})...")
    x_jac, hist_jac = solve_jacobi(A, b, x0, eps0=eps0, x_exact=x_exact)
    k_jac = len(hist_jac) - 1
    err_jac = vector_norm(x_jac - x_exact)
    res_jac = vector_norm(matrix_vector_mult(A, x_jac) - b)
    print(f"  -> Кількість ітерацій: {k_jac}")
    print(f"  -> Фінальна похибка ||X - X*||_inf: {err_jac:.4e}")
    print(f"  -> Фінальна нев'язка ||AX - B||_inf: {res_jac:.4e}")

    # 4.3. Розв'язання методом Зейделя
    print(f"\n[Крок 4.3] Розв'язання методом Зейделя (eps0 = {eps0})...")
    x_sei, hist_sei = solve_seidel(A, b, x0, eps0=eps0, x_exact=x_exact)
    k_sei = len(hist_sei) - 1
    err_sei = vector_norm(x_sei - x_exact)
    res_sei = vector_norm(matrix_vector_mult(A, x_sei) - b)
    print(f"  -> Кількість ітерацій: {k_sei}")
    print(f"  -> Фінальна похибка ||X - X*||_inf: {err_sei:.4e}")
    print(f"  -> Фінальна нев'язка ||AX - B||_inf: {res_sei:.4e}")

    # Порівняльна таблиця результатів
    print("\n" + "=" * 80)
    print("ПОРІВНЯЛЬНА ТАБЛИЦЯ ІТЕРАЦІЙНИХ МЕТОДІВ")
    print("=" * 80)
    print(f"{'Метод':<20} | {'К-сть ітерацій':<15} | {'Похибка ||X - X*||_inf':<24} | {'Нев\'язка ||AX - B||_inf':<24}")
    print("-" * 88)
    print(f"{'Проста ітерація':<20} | {k_sim:<15} | {err_sim:<24.4e} | {res_sim:<24.4e}")
    print(f"{'Якобі':<20} | {k_jac:<15} | {err_jac:<24.4e} | {res_jac:<24.4e}")
    print(f"{'Гауса-Зейделя':<20} | {k_sei:<15} | {err_sei:<24.4e} | {res_sei:<24.4e}")
    print("-" * 88)

    # 5. Побудова графіків
    print(f"\n[Крок 5] Генерація візуалізації (графіків збіжності)...")
    plot_convergence(hist_sim, hist_jac, hist_sei, x_exact, x_sim, x_jac, x_sei, eps0, plot_file)
    print(f"  -> Графік збережено у: {os.path.basename(plot_file)}")
    print("\nЛабораторну роботу успішно виконано.")


if __name__ == "__main__":
    main()
