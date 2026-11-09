"""
Лабораторна робота №9
Метод Хука-Дживса багатовимірної оптимізації

Виконав: студент 3 курсу групи ФеП-33 Франков Д. О.
Перевірив: асистент Каськун О. Д.
Факультет електроніки та комп'ютерних технологій
Львівський національний університет імені Івана Франка, 2026
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Стиль оформлення графіків
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# ==============================================================================
# 1. ЦІЛЬОВІ ФУНКЦІЇ ТА СИСТЕМА НЕЛІНІЙНИХ РІВНЯНЬ
# ==============================================================================

def f_rosenbrock(x: np.ndarray) -> float:
    """
    1. Класична тестова функція Розенброка (цільова функція №1 з методички):
    f(x1, x2) = 100 * (x2 - x1^2)^2 + (1 - x1)^2.
    Глобальний мінімум у точці (1.0, 1.0), де f(1.0, 1.0) = 0.
    """
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


def system_f1(x1: float, x2: float) -> float:
    """Рівняння 1: коло x1^2 + x2^2 - 5 = 0 (радіус sqrt(5) ≈ 2.236)."""
    return x1 ** 2 + x2 ** 2 - 5.0


def system_f2(x1: float, x2: float) -> float:
    """Рівняння 2: рівнобічна гіпербола x1 * x2 - 2 = 0."""
    return x1 * x2 - 2.0


def phi_system(x: np.ndarray) -> float:
    """
    Цільова функція нев'язок для системи нелінійних рівнянь (m = 2):
    Phi(x1, x2) = [f1(x1, x2)]^2 + [f2(x1, x2)]^2 >= 0.
    Мінімум досягається у точках розв'язку системи, де Phi = 0.
    """
    r1 = system_f1(x[0], x[1])
    r2 = system_f2(x[0], x[1])
    return r1 ** 2 + r2 ** 2


# ==============================================================================
# 2. АЛГОРИТМ МЕТОДУ ХУКА-ДЖИВСА
# ==============================================================================

def hooke_jeeves(func, x0, step, beta=2.0, alpha=1.0, eps_step=1e-6, eps_f=1e-10, max_iter=2000):
    """
    Чисельний метод нульового порядку Хука-Дживса (пошук за зразком та досліджуючий пошук).
    
    Параметри:
      func       - оптимізована цільова функція
      x0         - початкова базисна точка (масив/список)
      step       - початковий вектор кроків (масив/список)
      beta       - коефіцієнт редукції кроку (beta > 1, за замовчуванням 2.0)
      alpha      - коефіцієнт кроку за зразком (за замовчуванням 1.0)
      eps_step   - критерій зупинки за кроком (||delta|| < eps_step)
      eps_f      - критерій зупинки за значенням функції (Phi < eps_f)
      max_iter   - максимальна кількість ітерацій
    
    Повертає:
      x_opt, f_opt, trajectory, iters, f_evals
    """
    x_base = np.array(x0, dtype=float)
    delta = np.array(step, dtype=float)
    n = len(x_base)
    trajectory = []
    
    f_base = func(x_base)
    f_evals = 1

    trajectory.append({
        'k': 0,
        'x': x_base.copy(),
        'f': f_base,
        'delta': delta.copy(),
        'type': 'Початкова точка'
    })

    def explore(x_start, d):
        """Досліджуючий пошук навколо точки x_start за координатними напрямками."""
        nonlocal f_evals
        x_curr = x_start.copy()
        f_curr = func(x_curr)
        for i in range(n):
            # Пробний крок у додатному напрямку +d_i
            x_try = x_curr.copy()
            x_try[i] += d[i]
            f_try = func(x_try)
            f_evals += 1
            if f_try < f_curr:
                x_curr = x_try
                f_curr = f_try
            else:
                # Пробний крок у від'ємному напрямку -d_i
                x_try = x_curr.copy()
                x_try[i] -= d[i]
                f_try = func(x_try)
                f_evals += 1
                if f_try < f_curr:
                    x_curr = x_try
                    f_curr = f_try
        return x_curr, f_curr

    k = 0
    while k < max_iter:
        k += 1
        x_new, f_new = explore(x_base, delta)

        if f_new < f_base:
            # Досліджуючий пошук успішний
            trajectory.append({
                'k': k,
                'x': x_new.copy(),
                'f': f_new,
                'delta': delta.copy(),
                'type': 'Досліджуючий пошук'
            })

            # Серія рухів за зразком
            while f_new < f_base and k < max_iter:
                # Точка зразка: xp = x_new + alpha * (x_new - x_base)
                x_p = x_new + alpha * (x_new - x_base)
                x_base = x_new.copy()
                f_base = f_new

                # Досліджуючий пошук із точки зразка xp
                x_p_new, f_p_new = explore(x_p, delta)
                if f_p_new < f_base:
                    k += 1
                    x_new = x_p_new
                    f_new = f_p_new
                    trajectory.append({
                        'k': k,
                        'x': x_new.copy(),
                        'f': f_new,
                        'delta': delta.copy(),
                        'type': 'Пошук по зразку'
                    })
                else:
                    break
        else:
            # Досліджуючий пошук не дав покращення: редукція кроку
            delta = delta / beta
            trajectory.append({
                'k': k,
                'x': x_base.copy(),
                'f': f_base,
                'delta': delta.copy(),
                'type': f'Редукція кроку (h/{beta:.0f})'
            })

            if np.max(delta) < eps_step or f_base < eps_f:
                break

    return x_base, f_base, trajectory, k, f_evals


# ==============================================================================
# 3. ЕКСПОРТ ТРАЄКТОРІЇ ТА ПОБУДОВА ГРАФІКІВ
# ==============================================================================

def save_trajectory(trajectory: list, filename: str, task_title: str):
    """Експортує покрокові координати траєкторії спуску у текстовий файл."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Траєкторія спуску методу Хука-Дживса\n")
        f.write(f"# Завдання: {task_title}\n")
        f.write(f"# Всього кроків: {len(trajectory)}\n")
        f.write(f"{'Крок':>5} | {'x_1':>12} | {'x_2':>12} | {'f(x)':>14} | {'Крок h_1':>10} | {'Тип операції'}\n")
        f.write("-" * 80 + "\n")
        for step in trajectory:
            f.write(f"{step['k']:5d} | {step['x'][0]:12.8f} | {step['x'][1]:12.8f} | {step['f']:14.6e} | "
                    f"{step['delta'][0]:10.6f} | {step['type']}\n")


def generate_all_plots(traj_ros, traj_sys1, traj_sys2, plots_dir):
    """Генерує набір якісних ілюстрацій для звіту."""
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Графік кривих системи нелінійних рівнянь
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=300)
    theta = np.linspace(0, 2 * np.pi, 400)
    r = math.sqrt(5.0)
    circle_x = r * np.cos(theta)
    circle_y = r * np.sin(theta)
    ax.plot(circle_x, circle_y, label=r'Коло: $x_1^2 + x_2^2 = 5$', color='#1f77b4', lw=2)

    # Гіпербола x2 = 2 / x1
    x_pos = np.linspace(0.4, 3.5, 300)
    x_neg = np.linspace(-3.5, -0.4, 300)
    ax.plot(x_pos, 2.0 / x_pos, label=r'Гіпербола: $x_1 x_2 = 2$', color='#d62728', lw=2)
    ax.plot(x_neg, 2.0 / x_neg, color='#d62728', lw=2)

    # Точки перетину (розв'язки системи)
    roots = [(2.0, 1.0), (1.0, 2.0), (-2.0, -1.0), (-1.0, -2.0)]
    for idx, (rx, ry) in enumerate(roots):
        label = f'Розв\'язок $x^* = ({rx:.0f}, {ry:.0f})$' if idx == 0 else None
        ax.scatter([rx], [ry], color='#2ca02c', s=80, zorder=5, label=label)

    # Початкові точки спуску
    ax.scatter([3.0], [2.5], color='#ff7f0e', s=75, marker='s', zorder=5, label=r'Початкова точка 1: $(3.0, 2.5)$')
    ax.scatter([0.5], [2.5], color='#9467bd', s=75, marker='^', zorder=5, label=r'Початкова точка 2: $(0.5, 2.5)$')

    ax.axhline(0, color='black', lw=0.6, linestyle='--')
    ax.axvline(0, color='black', lw=0.6, linestyle='--')
    ax.set_xlim(-3.5, 4.0)
    ax.set_ylim(-3.5, 3.5)
    ax.set_title(r'Геометрична інтерпретація системи: $x_1^2 + x_2^2 = 5$, $x_1 x_2 = 2$', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$', fontsize=10)
    ax.set_ylabel('$x_2$', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, 'system_curves.png')
    plt.savefig(p1)
    plt.close()
    print(f"[+] Збережено: {p1}")

    # 2. Траєкторія на функції Розенброка
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=300)
    X = np.linspace(-1.5, 1.5, 350)
    Y = np.linspace(-0.5, 1.5, 350)
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z = 100.0 * (Y_grid - X_grid ** 2) ** 2 + (1.0 - X_grid) ** 2
    levels = np.logspace(-1, 3.5, 25)
    cs = ax.contour(X_grid, Y_grid, Z, levels=levels, cmap='viridis', alpha=0.7, linewidths=0.8)
    ax.clabel(cs, inline=1, fontsize=7, fmt='%.1f')

    # Траєкторія
    traj_x = [step['x'][0] for step in traj_ros]
    traj_y = [step['x'][1] for step in traj_ros]
    ax.plot(traj_x, traj_y, color='#d62728', lw=1.5, marker='o', markersize=3, label='Траєкторія Хука-Дживса')
    ax.scatter([traj_x[0]], [traj_y[0]], color='#ff7f0e', s=80, marker='s', zorder=5, label=f'Старт: $(-1.2, 1.0)$')
    ax.scatter([1.0], [1.0], color='#2ca02c', s=100, marker='*', zorder=5, label=f'Мінімум: $(1.0, 1.0)$')

    ax.set_title(r'Траєкторія спуску Хука-Дживса на функції Розенброка', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$', fontsize=10)
    ax.set_ylabel('$x_2$', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, 'rosenbrock_descent.png')
    plt.savefig(p2)
    plt.close()
    print(f"[+] Збережено: {p2}")

    # 3. Траєкторія спуску для системи нелінійних рівнянь
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=300)
    X = np.linspace(0.0, 3.8, 350)
    Y = np.linspace(0.0, 3.2, 350)
    X_grid, Y_grid = np.meshgrid(X, Y)
    R1 = X_grid ** 2 + Y_grid ** 2 - 5.0
    R2 = X_grid * Y_grid - 2.0
    Z_sys = R1 ** 2 + R2 ** 2

    levels_sys = np.logspace(-3, 2.5, 25)
    cs = ax.contour(X_grid, Y_grid, Z_sys, levels=levels_sys, cmap='magma', alpha=0.65, linewidths=0.8)
    ax.clabel(cs, inline=1, fontsize=7, fmt='%.1e')

    # Траєкторія 1: спуск до (2, 1)
    t1_x = [step['x'][0] for step in traj_sys1]
    t1_y = [step['x'][1] for step in traj_sys1]
    ax.plot(t1_x, t1_y, color='#1f77b4', lw=1.8, marker='o', markersize=3.5, label='Траєкторія 1: старт $(3.0, 2.5) \\to (2, 1)$')
    ax.scatter([t1_x[0]], [t1_y[0]], color='#1f77b4', s=70, marker='s', zorder=5)

    # Траєкторія 2: спуск до (1, 2)
    t2_x = [step['x'][0] for step in traj_sys2]
    t2_y = [step['x'][1] for step in traj_sys2]
    ax.plot(t2_x, t2_y, color='#2ca02c', lw=1.8, marker='^', markersize=3.5, label='Траєкторія 2: старт $(0.5, 2.5) \\to (1, 2)$')
    ax.scatter([t2_x[0]], [t2_y[0]], color='#2ca02c', s=70, marker='^', zorder=5)

    # Відмітка точних розв'язків
    ax.scatter([2.0], [1.0], color='#d62728', s=100, marker='*', zorder=6, label='Розв\'язок $(2, 1)$')
    ax.scatter([1.0], [2.0], color='#d62728', s=100, marker='*', zorder=6, label='Розв\'язок $(1, 2)$')

    ax.set_title(r'Лінії рівня цільової функції $\Phi(x_1, x_2)$ та траєкторії спуску', fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$', fontsize=10)
    ax.set_ylabel('$x_2$', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8.5)
    plt.tight_layout()
    p3 = os.path.join(plots_dir, 'system_descent.png')
    plt.savefig(p3)
    plt.close()
    print(f"[+] Збережено: {p3}")


# ==============================================================================
# 4. ГОЛОВНИЙ ЦИКЛ ОБЧИСЛЕНЬ
# ==============================================================================

def main():
    print("=" * 70)
    print("Лабораторна робота №9: Метод Хука-Дживса багатовимірної оптимізації")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    traj_file = os.path.join(base_dir, 'trajectory.txt')
    plots_dir = os.path.join(base_dir, 'plots')

    # Параметри оптимізації
    eps_step = 1e-6
    eps_f = 1e-10
    step_init = [0.5, 0.5]
    beta = 2.0
    alpha = 1.0

    # 1. Тестування на функції Розенброка (Пункт 3 «Хід роботи»)
    print("\n--- 1. Оптимізація функції Розенброка з точки (-1.2, 1.0) ---")
    x_opt_ros, f_opt_ros, traj_ros, iters_ros, fev_ros = hooke_jeeves(
        f_rosenbrock, [-1.2, 1.0], step_init, beta=beta, alpha=alpha, eps_step=eps_step, eps_f=eps_f
    )
    print(f"Знайдений мінімум: x = [{x_opt_ros[0]:.8f}, {x_opt_ros[1]:.8f}]")
    print(f"Значення функції:  f(x) = {f_opt_ros:.4e}")
    print(f"Кількість кроків:  {iters_ros}, обчислень функції: {fev_ros}, точок траєкторії: {len(traj_ros)}")

    # 2. Розв'язання системи нелінійних рівнянь (Пункт 4 «Хід роботи»)
    # Траєкторія 1: старт з точки (3.0, 2.5) -> розв'язок (2, 1)
    print("\n--- 2. Мінімізація цільової функції системи: Старт (3.0, 2.5) ---")
    x_opt_s1, f_opt_s1, traj_sys1, iters_s1, fev_s1 = hooke_jeeves(
        phi_system, [3.0, 2.5], step_init, beta=beta, alpha=alpha, eps_step=eps_step, eps_f=eps_f
    )
    print(f"Знайдений розв'язок: x = [{x_opt_s1[0]:.8f}, {x_opt_s1[1]:.8f}]")
    print(f"Нев'язка системи:    Phi(x) = {f_opt_s1:.4e}")
    print(f"Нев'язки рівнянь:    f1 = {system_f1(x_opt_s1[0], x_opt_s1[1]):.2e}, f2 = {system_f2(x_opt_s1[0], x_opt_s1[1]):.2e}")
    print(f"Кількість кроків:    {iters_s1}, обчислень функції: {fev_s1}, точок траєкторії: {len(traj_sys1)}")

    # Траєкторія 2: старт з точки (0.5, 2.5) -> розв'язок (1, 2)
    print("\n--- 3. Мінімізація цільової функції системи: Старт (0.5, 2.5) ---")
    x_opt_s2, f_opt_s2, traj_sys2, iters_s2, fev_s2 = hooke_jeeves(
        phi_system, [0.5, 2.5], step_init, beta=beta, alpha=alpha, eps_step=eps_step, eps_f=eps_f
    )
    print(f"Знайдений розв'язок: x = [{x_opt_s2[0]:.8f}, {x_opt_s2[1]:.8f}]")
    print(f"Нев'язка системи:    Phi(x) = {f_opt_s2:.4e}")
    print(f"Кількість кроків:    {iters_s2}, обчислень функції: {fev_s2}")

    # Збереження повної траєкторії спуску у файл trajectory.txt (Пункт 5 «Хід роботи»)
    save_trajectory(traj_sys1, traj_file, "Розв'язок системи нелінійних рівнянь з точки (3.0, 2.5)")
    print(f"\n[+] Траєкторію спуску збережено у {traj_file}")

    # Генерація графіків
    print("\n--- 4. Генерація графіків ---")
    generate_all_plots(traj_ros, traj_sys1, traj_sys2, plots_dir)
    print("\n[V] Обчислювальну частину лабораторної роботи №9 успішно завершено!")


if __name__ == '__main__':
    main()
