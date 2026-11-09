"""
Лабораторна робота №8
Чисельні методи розв’язування нелінійних рівнянь з одним невідомим

Виконав: студент групи ФеП-33 Франков Д. О.
Перевірив: асистент Каськун О. Д.
Кафедра системного проектування
Львівський національний університет імені Івана Франка
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Налаштування стилю графіків
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# ==============================================================================
# ЧАСТИНА 1. ТРАНСЦЕНДЕНТНЕ НЕЛІНІЙНЕ РІВНЯННЯ
# ==============================================================================

def f_transcendental(x: float) -> float:
    """Трансцендентна функція f(x) = x * sin(x) - 1."""
    return x * math.sin(x) - 1.0


def df_transcendental(x: float) -> float:
    """Перша похідна f'(x) = sin(x) + x * cos(x)."""
    return math.sin(x) + x * math.cos(x)


def d2f_transcendental(x: float) -> float:
    """Друга похідна f''(x) = 2 * cos(x) - x * sin(x)."""
    return 2.0 * math.cos(x) - x * math.sin(x)


def tabulate_function(func, a: float, b: float, h: float, filename: str):
    """
    1. Табулює функцію на відрізку [a, b] з кроком h та записує результати у файл.
    Повертає список точок (x_i, y_i) та знайдені інтервали ізоляції коренів.
    """
    nodes = []
    x = a
    while x <= b + 1e-9:
        nodes.append((round(x, 6), func(x)))
        x += h

    # Запис у текстовий файл
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("# Результати табуляції функції f(x) = x * sin(x) - 1\n")
        file.write(f"# Відрізок: [{a}, {b}], крок: {h}\n")
        file.write(f"{'i':>4} | {'x_i':>10} | {'f(x_i)':>14}\n")
        file.write("-" * 34 + "\n")
        for i, (xi, yi) in enumerate(nodes):
            file.write(f"{i:4d} | {xi:10.4f} | {yi:14.6f}\n")

    # Пошук інтервалів зміни знаку
    brackets = []
    for i in range(len(nodes) - 1):
        x1, y1 = nodes[i]
        x2, y2 = nodes[i + 1]
        if y1 * y2 <= 0.0:
            behavior = "зростання" if y1 < y2 else "спадання"
            brackets.append({
                'interval': (x1, x2),
                'y_vals': (y1, y2),
                'behavior': behavior,
                'x_init': x1
            })

    return nodes, brackets


# --- 6 ІТЕРАЦІЙНИХ МЕТОДІВ ДЛЯ ТРАНСЦЕНДЕНТНОГО РІВНЯННЯ ---

def solve_simple_iteration(func, dfunc, x0: float, eps: float = 1e-10, max_iter: int = 100):
    """
    1. Метод простої ітерації (релаксації):
    x_{k+1} = x_k + tau * f(x_k), де tau = -1 / f'(x0).
    """
    tau = -1.0 / dfunc(x0)
    history = []
    x_curr = x0

    for k in range(1, max_iter + 1):
        fx = func(x_curr)
        x_next = x_curr + tau * fx
        diff = abs(x_next - x_curr)
        res = abs(func(x_next))
        history.append({
            'iter': k,
            'x': x_next,
            'diff': diff,
            'res': res
        })
        if diff < eps and res < eps:
            return x_next, history
        x_curr = x_next

    return x_curr, history


def solve_newton(func, dfunc, x0: float, eps: float = 1e-10, max_iter: int = 100):
    """
    2. Метод Ньютона (дотичних):
    x_{k+1} = x_k - f(x_k) / f'(x_k). Порядок збіжності: 2.
    """
    history = []
    x_curr = x0

    for k in range(1, max_iter + 1):
        fx = func(x_curr)
        dfx = dfunc(x_curr)
        if abs(dfx) < 1e-15:
            break
        x_next = x_curr - fx / dfx
        diff = abs(x_next - x_curr)
        res = abs(func(x_next))
        history.append({
            'iter': k,
            'x': x_next,
            'diff': diff,
            'res': res
        })
        if diff < eps and res < eps:
            return x_next, history
        x_curr = x_next

    return x_curr, history


def solve_chebyshev(func, dfunc, d2func, x0: float, eps: float = 1e-10, max_iter: int = 100):
    """
    3. Метод Чебишева (3-го порядку):
    x_{k+1} = x_k - f/f' - (f'' * f^2) / (2 * (f')^3). Порядок збіжності: 3.
    """
    history = []
    x_curr = x0

    for k in range(1, max_iter + 1):
        fx = func(x_curr)
        dfx = dfunc(x_curr)
        d2fx = d2func(x_curr)
        if abs(dfx) < 1e-15:
            break
        step1 = fx / dfx
        step2 = (d2fx * (fx ** 2)) / (2.0 * (dfx ** 3))
        x_next = x_curr - step1 - step2
        diff = abs(x_next - x_curr)
        res = abs(func(x_next))
        history.append({
            'iter': k,
            'x': x_next,
            'diff': diff,
            'res': res
        })
        if diff < eps and res < eps:
            return x_next, history
        x_curr = x_next

    return x_curr, history


def solve_secant(func, x0: float, x1: float, eps: float = 1e-10, max_iter: int = 100):
    """
    4. Метод хорд (січних):
    x_{k+1} = x_k - f(x_k) * (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1})). Порядок: 1.618.
    """
    history = []
    x_prev = x0
    x_curr = x1

    for k in range(1, max_iter + 1):
        f_curr = func(x_curr)
        f_prev = func(x_prev)
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            break
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        diff = abs(x_next - x_curr)
        res = abs(func(x_next))
        history.append({
            'iter': k,
            'x': x_next,
            'diff': diff,
            'res': res
        })
        if diff < eps and res < eps:
            return x_next, history
        x_prev, x_curr = x_curr, x_next

    return x_curr, history


def solve_muller(func, x0: float, x1: float, x2: float, eps: float = 1e-10, max_iter: int = 100):
    """
    5. Метод парабол (Мюллера):
    Квадратична парабола за трьома вузлами (x0, x1, x2). Порядок: 1.839.
    """
    history = []
    p0, p1, p2 = x0, x1, x2

    for k in range(1, max_iter + 1):
        f0, f1, f2 = func(p0), func(p1), func(p2)
        h1 = p1 - p0
        h2 = p2 - p1
        if abs(h1) < 1e-15 or abs(h2) < 1e-15:
            break
        del1 = (f1 - f0) / h1
        del2 = (f2 - f1) / h2
        d = (del2 - del1) / (h2 + h1)
        b = del2 + h2 * d
        c = f2

        discriminant = b ** 2 - 4.0 * d * c
        rad = np.sqrt(complex(discriminant))
        denom1 = b + rad
        denom2 = b - rad
        denom = denom1 if abs(denom1) >= abs(denom2) else denom2
        if abs(denom) < 1e-15:
            break
        dx = -2.0 * c / denom
        x_next = p2 + dx.real

        diff = abs(x_next - p2)
        res = abs(func(x_next))
        history.append({
            'iter': k,
            'x': x_next,
            'diff': diff,
            'res': res
        })
        if diff < eps and res < eps:
            return x_next, history
        p0, p1, p2 = p1, p2, x_next

    return p2, history


def solve_inverse_interpolation(func, x0: float, x1: float, x2: float, eps: float = 1e-10, max_iter: int = 100):
    """
    6. Метод зворотної квадратичної інтерполяції (Лагранжа за 3 вузлами в точці y=0):
    x_{k+1} = L(0). Порядок: ~1.839.
    """
    history = []
    q0, q1, q2 = x0, x1, x2

    for k in range(1, max_iter + 1):
        y0, y1, y2 = func(q0), func(q1), func(q2)
        denom0 = (y0 - y1) * (y0 - y2)
        denom1 = (y1 - y0) * (y1 - y2)
        denom2 = (y2 - y0) * (y2 - y1)
        if abs(denom0) < 1e-15 or abs(denom1) < 1e-15 or abs(denom2) < 1e-15:
            break

        term0 = q0 * (y1 * y2) / denom0
        term1 = q1 * (y0 * y2) / denom1
        term2 = q2 * (y0 * y1) / denom2
        x_next = term0 + term1 + term2

        diff = abs(x_next - q2)
        res = abs(func(x_next))
        history.append({
            'iter': k,
            'x': x_next,
            'diff': diff,
            'res': res
        })
        if diff < eps and res < eps:
            return x_next, history
        q0, q1, q2 = q1, q2, x_next

    return q2, history


# ==============================================================================
# ЧАСТИНА 2. АЛГЕБРАЇЧНІ МНОГОЧЛЕНИ (ГОРНЕР ТА ЛІН)
# ==============================================================================

def write_polynomial_coeffs(coeffs: list, filename: str):
    """6. Записує коефіцієнти многочлена a_n, a_{n-1}, ..., a_0 у файл."""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("# Коефіцієнти алгебраїчного многочлена P3(x) = a3*x^3 + a2*x^2 + a1*x + a0\n")
        file.write("# Задано многочлен з 1 дійсним коренем і 2 комплексно-спряженими коренями\n")
        for i, c in enumerate(coeffs):
            power = len(coeffs) - 1 - i
            file.write(f"a_{power} = {c}\n")


def read_polynomial_coeffs(filename: str) -> list:
    """7а. Зчитує коефіцієнти многочлена з текстового файлу у список [a_n, ..., a_0]."""
    coeffs = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                val = float(line.split('=')[1].strip())
                coeffs.append(val)
    return coeffs


def horner_eval(coeffs: list, x: float):
    """
    Обчислює значення многочлена P(x) та його першої похідної P'(x)
    за узагальненою схемою Горнера.
    P(x) = (x - x_k)*Q(x) + b0  => P(x) = b0
    Q(x) = (x - x_k)*R(x) + c1  => P'(x) = c1
    """
    n = len(coeffs) - 1
    # Обчислення b_k для P(x)
    b = [0.0] * (n + 1)
    b[0] = coeffs[0]
    for i in range(1, n + 1):
        b[i] = coeffs[i] + b[i - 1] * x
    p_val = b[-1]

    # Обчислення c_k для P'(x) з коефіцієнтів Q(x) = b[0..n-1]
    c = [0.0] * n
    c[0] = b[0]
    for i in range(1, n):
        c[i] = b[i] + c[i - 1] * x
    dp_val = c[-1]

    return p_val, dp_val


def poly_eval(coeffs: list, x: float) -> float:
    """
    7б. Скласти функцію, яка з допомогою отриманого масиву коефіцієнтів
    довільного алгебраїчного рівняння повертає для деякого x значення цього многочлена.
    """
    p_val, _ = horner_eval(coeffs, x)
    return p_val


def solve_newton_horner(coeffs: list, x0: float, eps: float = 1e-10, max_iter: int = 100):
    """
    8. Метод Ньютона за схемою Горнера для знаходження дійсного кореня полінома:
    x_{k+1} = x_k - P(x_k) / P'(x_k) = x_k - b0 / c1.
    """
    history = []
    x_curr = x0

    for k in range(1, max_iter + 1):
        p_val, dp_val = horner_eval(coeffs, x_curr)
        if abs(dp_val) < 1e-15:
            break
        x_next = x_curr - p_val / dp_val
        diff = abs(x_next - x_curr)
        res = abs(poly_eval(coeffs, x_next))
        history.append({
            'iter': k,
            'x': x_next,
            'diff': diff,
            'res': res
        })
        if diff < eps and res < eps:
            return x_next, history
        x_curr = x_next

    return x_curr, history


def solve_lin_method(coeffs: list, p0: float, q0: float, eps: float = 1e-10, max_iter: int = 100):
    """
    9. Метод Ліна для виділення квадратичного множника x^2 + p*x + q кубічного полінома:
    P3(x) = (x^2 + p*x + q)*(b3*x + b2) + R1*x + R0.
    Уточнений множник: q_{k+1} = a0 / b2, p_{k+1} = (a1 - q_{k+1}*b3) / b2.
    Комплексні корені: alpha = -p/2, beta = sqrt(q - alpha^2).
    """
    a3, a2, a1, a0 = coeffs
    p_curr, q_curr = p0, q0
    history = []

    for k in range(1, max_iter + 1):
        b3 = a3
        b2 = a2 - p_curr * b3
        if abs(b2) < 1e-15:
            break
        q_next = a0 / b2
        p_next = (a1 - q_next * b3) / b2

        # Обчислення коренів
        alpha = -p_next / 2.0
        disc = q_next - alpha ** 2
        beta = math.sqrt(disc) if disc >= 0 else 0.0

        p_diff = abs(p_next - p_curr)
        q_diff = abs(q_next - q_curr)
        history.append({
            'iter': k,
            'p': p_next,
            'q': q_next,
            'alpha': alpha,
            'beta': beta,
            'p_diff': p_diff,
            'q_diff': q_diff
        })

        if p_diff < eps and q_diff < eps:
            return p_next, q_next, alpha, beta, history

        p_curr, q_curr = p_next, q_next

    alpha = -p_curr / 2.0
    beta = math.sqrt(max(0.0, q_curr - alpha ** 2))
    return p_curr, q_curr, alpha, beta, history


# ==============================================================================
# ПОБУДОВА ГРАФІКІВ ТА ГОЛОВНИЙ ЦИКЛ
# ==============================================================================

def generate_plots(results_root1, results_root2, poly_coeffs, root_real, poly_history, lin_history, output_dir):
    """Генерує якісні графіки для звіту."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Графік трансцендентної функції f(x)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=300)
    xs = np.linspace(0.0, 4.0, 500)
    ys = [f_transcendental(x) for x in xs]
    ax.plot(xs, ys, label=r'$f(x) = x \sin(x) - 1$', color='#1f77b4', lw=2)
    ax.axhline(0, color='black', lw=0.8, linestyle='--')

    # Інтервали локалізації за кроком h=0.1
    ax.axvspan(1.1, 1.2, color='#2ca02c', alpha=0.22, label='Інтервал 1: [1.1, 1.2] (зростання)')
    ax.axvspan(2.7, 2.8, color='#d62728', alpha=0.22, label='Інтервал 2: [2.7, 2.8] (спадання)')

    # Відмітка точних коренів
    r1 = results_root1['Newton']['x']
    r2 = results_root2['Newton']['x']
    ax.scatter([r1], [0], color='#2ca02c', s=70, zorder=5, label=f'Корінь 1: $x^*_1 \\approx {r1:.4f}$')
    ax.scatter([r2], [0], color='#d62728', s=70, zorder=5, label=f'Корінь 2: $x^*_2 \\approx {r2:.4f}$')

    ax.set_title('Графік трансцендентної функції $f(x) = x \\sin(x) - 1$ на $[0, 4]$ ($h = 0.1$)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$f(x)$', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    f_path = os.path.join(output_dir, 'f_transcendental.png')
    plt.savefig(f_path)
    plt.close()
    print(f"[+] Збережено графік: {f_path}")

    # 2. Графік кубічного полінома P3(x)
    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=300)
    xs_p = np.linspace(0.0, 3.5, 400)
    ys_p = [poly_coeffs[0]*x**3 + poly_coeffs[1]*x**2 + poly_coeffs[2]*x + poly_coeffs[3] for x in xs_p]
    ax.plot(xs_p, ys_p, label=r'$P_3(x) = x^3 - 4x^2 + 9x - 10$', color='#9467bd', lw=2)
    ax.axhline(0, color='black', lw=0.8, linestyle='--')
    ax.scatter([root_real], [0], color='#d62728', s=80, zorder=5, label=f'Дійсний корінь: $x_1 = {root_real:.4f}$')

    # Дотична у дійсному корені
    p_val, dp_val = horner_eval(poly_coeffs, root_real)
    tangent_xs = np.linspace(1.2, 2.8, 100)
    tangent_ys = dp_val * (tangent_xs - root_real)
    ax.plot(tangent_xs, tangent_ys, color='#ff7f0e', linestyle=':', lw=1.5, label=f'Дотична у $x=2$ ($P\'(2)={dp_val:.1f}$)')

    ax.set_title('Графік алгебраїчного многочлена $P_3(x) = x^3 - 4x^2 + 9x - 10$', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$P_3(x)$', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    p_path = os.path.join(output_dir, 'p3_polynomial.png')
    plt.savefig(p_path)
    plt.close()
    print(f"[+] Збережено графік: {p_path}")

    # 3. Графіки збіжності методів (динаміка похибки log10|x_k - x*|)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
    methods = ['Simple Iteration', 'Newton', 'Chebyshev', 'Chords (Secant)', 'Muller (Parabolas)', 'Inverse Interp']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'p']

    # Корінь 1
    exact_r1 = results_root1['Newton']['x']
    for idx, m in enumerate(methods):
        hist = results_root1[m]['history']
        iters = [h['iter'] for h in hist]
        errs = [max(1e-16, abs(h['x'] - exact_r1)) for h in hist]
        ax1.plot(iters, np.log10(errs), label=m, color=colors[idx], marker=markers[idx], markersize=5, lw=1.5)
    ax1.set_title('Корінь 1 ($x^*_1 \\approx 1.114157$, зростання $f\' > 0$)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Номер ітерації $k$', fontsize=10)
    ax1.set_ylabel('$\\log_{10}|x_k - x^*|$', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=8, frameon=True)

    # Корінь 2
    exact_r2 = results_root2['Newton']['x']
    for idx, m in enumerate(methods):
        hist = results_root2[m]['history']
        iters = [h['iter'] for h in hist]
        errs = [max(1e-16, abs(h['x'] - exact_r2)) for h in hist]
        ax2.plot(iters, np.log10(errs), label=m, color=colors[idx], marker=markers[idx], markersize=5, lw=1.5)
    ax2.set_title('Корінь 2 ($x^*_2 \\approx 2.772605$, спадання $f\' < 0$)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Номер ітерації $k$', fontsize=10)
    ax2.set_ylabel('$\\log_{10}|x_k - x^*|$', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(fontsize=8, frameon=True)

    plt.suptitle('Порівняння швидкості збіжності 6 чисельних методів ($\\varepsilon = 10^{-10}$)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    c_path = os.path.join(output_dir, 'convergence.png')
    plt.savefig(c_path)
    plt.close()
    print(f"[+] Збережено графік: {c_path}")


def main():
    print("=" * 70)
    print("Лабораторна робота №8: Чисельні методи розв’язування нелінійних рівнянь")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tab_file = os.path.join(base_dir, 'tabulation.txt')
    poly_file = os.path.join(base_dir, 'poly_coeffs.txt')
    plots_dir = os.path.join(base_dir, 'plots')

    # Крок 1. Табуляція трансцендентної функції f(x) на [0, 4] з кроком h = 0.1
    print("\n--- КРОК 1. Табуляція f(x) = x * sin(x) - 1 на [0, 4], h = 0.1 ---")
    nodes, brackets = tabulate_function(f_transcendental, 0.0, 4.0, 0.1, tab_file)
    print(f"Збережено {len(nodes)} вузлів у {tab_file}")
    for b in brackets:
        print(f"  Знайдено інтервал локалізації: {b['interval']}, f(x) є {b['behavior']}")

    # Крок 2, 3, 4. Розв'язок 6 методами з критерієм |f| < eps та |x_{k+1}-x_k| < eps (eps = 1e-10)
    eps = 1e-10

    # Дослідження кореня 1 (зростання f'(x) > 0, інтервал [1.1, 1.2])
    results_root1 = {}
    x1_final, hist1 = solve_simple_iteration(f_transcendental, df_transcendental, 1.1, eps)
    results_root1['Simple Iteration'] = {'x': x1_final, 'iters': len(hist1), 'history': hist1}

    x1_final, hist2 = solve_newton(f_transcendental, df_transcendental, 1.1, eps)
    results_root1['Newton'] = {'x': x1_final, 'iters': len(hist2), 'history': hist2}

    x1_final, hist3 = solve_chebyshev(f_transcendental, df_transcendental, d2f_transcendental, 1.1, eps)
    results_root1['Chebyshev'] = {'x': x1_final, 'iters': len(hist3), 'history': hist3}

    x1_final, hist4 = solve_secant(f_transcendental, 1.0, 1.1, eps)
    results_root1['Chords (Secant)'] = {'x': x1_final, 'iters': len(hist4), 'history': hist4}

    x1_final, hist5 = solve_muller(f_transcendental, 0.9, 1.0, 1.1, eps)
    results_root1['Muller (Parabolas)'] = {'x': x1_final, 'iters': len(hist5), 'history': hist5}

    x1_final, hist6 = solve_inverse_interpolation(f_transcendental, 0.9, 1.0, 1.1, eps)
    results_root1['Inverse Interp'] = {'x': x1_final, 'iters': len(hist6), 'history': hist6}

    # Дослідження кореня 2 (спадання f'(x) < 0, інтервал [2.7, 2.8])
    results_root2 = {}
    x2_final, hist1 = solve_simple_iteration(f_transcendental, df_transcendental, 2.7, eps)
    results_root2['Simple Iteration'] = {'x': x2_final, 'iters': len(hist1), 'history': hist1}

    x2_final, hist2 = solve_newton(f_transcendental, df_transcendental, 2.7, eps)
    results_root2['Newton'] = {'x': x2_final, 'iters': len(hist2), 'history': hist2}

    x2_final, hist3 = solve_chebyshev(f_transcendental, df_transcendental, d2f_transcendental, 2.7, eps)
    results_root2['Chebyshev'] = {'x': x2_final, 'iters': len(hist3), 'history': hist3}

    x2_final, hist4 = solve_secant(f_transcendental, 2.6, 2.7, eps)
    results_root2['Chords (Secant)'] = {'x': x2_final, 'iters': len(hist4), 'history': hist4}

    x2_final, hist5 = solve_muller(f_transcendental, 2.5, 2.6, 2.7, eps)
    results_root2['Muller (Parabolas)'] = {'x': x2_final, 'iters': len(hist5), 'history': hist5}

    x2_final, hist6 = solve_inverse_interpolation(f_transcendental, 2.5, 2.6, 2.7, eps)
    results_root2['Inverse Interp'] = {'x': x2_final, 'iters': len(hist6), 'history': hist6}

    # Вивід зведеної таблиці результатів
    print("\n" + "=" * 78)
    print(f"ЗВЕДЕНА ТАБЛИЦЯ РОЗВ'ЯЗКІВ f(x) = 0 (ТОЧНІСТЬ eps = {eps:.0e})")
    print("=" * 78)
    print(f"{'Метод':<20} | {'Корінь 1 (зростає, [1.1,1.2])':<26} | {'Корінь 2 (спадає, [2.7,2.8])':<26}")
    print(f"{'':<20} | {'x*':<12} {'Iter':<5} {'|f|':<8} | {'x*':<12} {'Iter':<5} {'|f|':<8}")
    print("-" * 78)
    for m in results_root1:
        r1 = results_root1[m]
        r2 = results_root2[m]
        f1_res = abs(f_transcendental(r1['x']))
        f2_res = abs(f_transcendental(r2['x']))
        print(f"{m:<20} | {r1['x']:<12.10f} {r1['iters']:<5d} {f1_res:<8.1e} | {r2['x']:<12.10f} {r2['iters']:<5d} {f2_res:<8.1e}")
    print("-" * 78)

    # Крок 5, 6. Алгебраїчний поліном P3(x) = x^3 - 4x^2 + 9x - 10
    print("\n--- КРОК 5, 6. Поліном 3-го порядку: P3(x) = x^3 - 4x^2 + 9x - 10 ---")
    poly_coeffs = [1.0, -4.0, 9.0, -10.0]
    write_polynomial_coeffs(poly_coeffs, poly_file)
    print(f"Коефіцієнти полінома записано у {poly_file}")

    # Крок 7. Зчитування, окрема функція poly_eval та перевірка схеми Горнера
    read_coeffs = read_polynomial_coeffs(poly_file)
    p_val_test = poly_eval(read_coeffs, 2.0)
    p_test, dp_test = horner_eval(read_coeffs, 2.0)
    print(f"Зчитані коефіцієнти: {read_coeffs}")
    print(f"Окрема функція poly_eval(coeffs, 2.0) = {p_val_test:.6f}")
    print(f"Схема Горнера horner_eval(coeffs, 2.0): P(2) = {p_test:.6f}, P'(2) = {dp_test:.6f}")

    # Крок 8. Метод Ньютона-Горнера для дійсного кореня (eps = 1e-10)
    print("\n--- КРОК 8. Метод Ньютона-Горнера (початкове x0 = 3.5, eps = 1e-10) ---")
    root_real, poly_history = solve_newton_horner(read_coeffs, 3.5, eps)
    print(f"Знайдено дійсний корінь: x = {root_real:.10f} за {len(poly_history)} ітерацій")
    for step in poly_history:
        print(f"  Ітерація {step['iter']:2d}: x = {step['x']:.10f}, різниця = {step['diff']:.2e}, |P(x)| = {step['res']:.2e}")

    # Крок 9. Метод Ліна для виділення квадратичного множника (eps = 1e-10)
    print("\n--- КРОК 9. Метод Ліна (початкові p0 = -1.0, q0 = 3.0, eps = 1e-10) ---")
    p_opt, q_opt, alpha, beta, lin_history = solve_lin_method(read_coeffs, -1.0, 3.0, eps)
    print(f"Знайдено множник: x^2 + ({p_opt:.10f})*x + ({q_opt:.10f}) = 0 за {len(lin_history)} ітерацій")
    print(f"Комплексно-спряжені корені: x = {alpha:.8f} +/- {beta:.8f} i")
    for step in lin_history:
        print(f"  Ітерація {step['iter']:2d}: p = {step['p']:.8f}, q = {step['q']:.8f}, "
              f"корені = {step['alpha']:.6f} +/- {step['beta']:.6f}i, dp = {step['p_diff']:.2e}, dq = {step['q_diff']:.2e}")

    # Генерація графіків
    print("\n--- Генерація графіків ---")
    generate_plots(results_root1, results_root2, read_coeffs, root_real, poly_history, lin_history, plots_dir)
    print("\n[V] Всі обчислення лабораторної роботи №8 успішно оновлено!")


if __name__ == '__main__':
    main()
