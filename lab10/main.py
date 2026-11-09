"""
Лабораторна робота №10
Тема: Методи Рунге-Кутта та Адамса розв’язання задачі Коші для звичайних
      диференціальних рівнянь першого порядку
Виконав: студент 3 курсу групи ФеП-33 Франков Д. О.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. ПОСТАНОВКА ЗАДАЧІ КОШІ ТА ТОЧНИЙ РОЗВ'ЯЗОК
# ----------------------------------------------------------------------
# Диференціальне рівняння: dy/dx = f(x, y) = y - x^2 + 1
# Відрізок: [0.0, 1.0], початкова умова: y(0) = 0.5
# Точний розв'язок: y(x) = (x + 1)^2 - 0.5 * exp(x)

def f_ode(x: float, y: float) -> float:
    """Права частина диференціального рівняння dy/dx = f(x, y)."""
    return y - x**2 + 1.0


def y_analytical(x: float | np.ndarray) -> float | np.ndarray:
    """Точний аналітичний розв'язок задачі Коші."""
    return (x + 1.0)**2 - 0.5 * np.exp(x)


# ----------------------------------------------------------------------
# 2. МЕТОД РУНГЕ-КУТТА 4-ГО ПОРЯДКУ
# ----------------------------------------------------------------------

def rk4_step(x: float, y: float, h: float) -> float:
    """Один крок класичного методу Рунге-Кутта 4-го порядку."""
    k1 = f_ode(x, y)
    k2 = f_ode(x + 0.5 * h, y + 0.5 * h * k1)
    k3 = f_ode(x + 0.5 * h, y + 0.5 * h * k2)
    k4 = f_ode(x + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_rk4_fixed(x0: float, y0: float, x_end: float, h: float):
    """
    Чисельний розв'язок методом Рунге-Кутта 4-го порядку з фіксованим кроком h.
    Також обчислює глобальну оцінку похибки за правилом Рунге на подвійній сітці:
    Phi_n = (16/15) * |y_n(h) - y_{2n}(h/2)|.
    """
    n_steps = int(np.round((x_end - x0) / h))
    xs = np.linspace(x0, x_end, n_steps + 1)
    ys = np.zeros(n_steps + 1)
    ys[0] = y0

    # 1. Траєкторія з кроком h
    for i in range(n_steps):
        ys[i + 1] = rk4_step(xs[i], ys[i], h)

    # 2. Незалежна повна траєкторія з кроком h/2
    h_half = 0.5 * h
    n_half = 2 * n_steps
    xs_half = np.linspace(x0, x_end, n_half + 1)
    ys_half = np.zeros(n_half + 1)
    ys_half[0] = y0
    for i in range(n_half):
        ys_half[i + 1] = rk4_step(xs_half[i], ys_half[i], h_half)

    exact_ys = y_analytical(xs)
    errors = np.abs(ys - exact_ys)
    # Глобальна оцінка за правилом Рунге у спільних вузлах сітки:
    runge_errors = (16.0 / 15.0) * np.abs(ys - ys_half[::2])

    return xs, ys, exact_ys, errors, runge_errors


def investigate_rk4_convergence(x0: float, y0: float, x_end: float, steps: list[float]):
    """Дослідження залежності глобальної похибки RK4 від величини кроку h."""
    max_errors = []
    for h in steps:
        _, ys, exact_ys, _, _ = solve_rk4_fixed(x0, y0, x_end, h)
        max_err = np.max(np.abs(ys - exact_ys))
        max_errors.append(max_err)
    return np.array(steps), np.array(max_errors)


def solve_rk4_adaptive(x0: float, y0: float, x_end: float, h_start: float, eps: float):
    """
    Автоматичний вибір кроку для методу Рунге-Кутта 4-го порядку.
    Правило зміни кроку за методичкою (стор. 7):
    s = 4 -> k = 2^(s+1) = 32.
    Якщо локальна похибка delta > eps: крок зменшується вдвічі (h = h / 2).
    Якщо delta < eps / 32: крок збільшується вдвічі (h = 2 * h).
    Інакше крок зберігається.
    """
    cur_x = x0
    cur_y = y0
    h = h_start

    xs = [cur_x]
    ys = [cur_y]
    hs = [h]
    errors = [0.0]

    while cur_x < x_end - 1e-12:
        if cur_x + h > x_end:
            h = x_end - cur_x

        # Крок h та два кроки h/2
        y_h = rk4_step(cur_x, cur_y, h)
        h_half = 0.5 * h
        y_half = rk4_step(cur_x, cur_y, h_half)
        y_two_half = rk4_step(cur_x + h_half, y_half, h_half)

        # Локальна похибка за методом Рунге
        delta = (16.0 / 15.0) * abs(y_h - y_two_half)

        if delta > eps and h > 1e-6:
            # Похибка занадто велика -> зменшуємо крок удвічі та повторюємо
            h = 0.5 * h
            continue

        # Приймаємо крок (беремо точніше значення y_two_half або y_h)
        cur_x += h
        cur_y = y_two_half
        err = abs(cur_y - y_analytical(cur_x))

        xs.append(cur_x)
        ys.append(cur_y)
        hs.append(h)
        errors.append(err)

        # Перевірка умови збільшення кроку
        if delta < (eps / 32.0):
            h = min(2.0 * h, 0.2)

    return np.array(xs), np.array(ys), np.array(hs), np.array(errors)


# ----------------------------------------------------------------------
# 3. МЕТОД ПРОГНОЗУ ТА КОРЕКЦІЇ АДАМСА 2-ГО ПОРЯДКУ
# ----------------------------------------------------------------------

def solve_adams_fixed(x0: float, y0: float, x_end: float, h: float):
    """
    Чисельний розв'язок методом прогнозу і корекції Адамса 2-го порядку.
    - Старт: точка y_1 обчислюється методом Рунге-Кутта 4-го порядку.
    - Прогноз: y_{n+1}^{np} = y_n + (h/2) * (3*f_n - f_{n-1})
    - Корекція: y_{n+1}^{kor} = y_n + (h/2) * (f(x_{n+1}, y_{n+1}^{np}) + f_n)
    - Уточнене значення: y_{n+1} = y_{n+1}^{kor} - (1/6) * (y_{n+1}^{kor} - y_{n+1}^{np})
    - Оцінка похибки: Delta_PC = y_{n+1}^{kor} - y_{n+1}^{np}
    """
    n_steps = int(np.round((x_end - x0) / h))
    xs = [x0]
    ys = [y0]
    y_pred_list = [y0]
    y_corr_list = [y0]
    errors = [0.0]
    est_errors = [0.0]

    # Перший вузол x_1 через Рунге-Кутта 4-го порядку
    x1 = x0 + h
    y1 = rk4_step(x0, y0, h)
    xs.append(x1)
    ys.append(y1)
    y_pred_list.append(y1)
    y_corr_list.append(y1)
    errors.append(abs(y1 - y_analytical(x1)))
    est_errors.append(0.0)

    # Ітераційний процес прогнозу-корекції для n = 1, 2, ...
    for n in range(1, n_steps):
        xn = xs[n]
        yn = ys[n]
        fn = f_ode(xn, yn)
        fn_prev = f_ode(xs[n - 1], ys[n - 1])

        xn1 = xn + h

        # 1. Прогноз (екстраполяція)
        y_pred = yn + 0.5 * h * (3.0 * fn - fn_prev)

        # 2. Корекція (інтерполяція)
        fn1_pred = f_ode(xn1, y_pred)
        y_corr = yn + 0.5 * h * (fn1_pred + fn)

        # 3. Оцінка локальної похибки
        diff_pc = y_corr - y_pred
        # Кінцеве значення з урахуванням виправлення похибки (стор. 11 методички)
        y_final = y_corr - (1.0 / 6.0) * diff_pc

        err_exact = abs(y_final - y_analytical(xn1))

        xs.append(xn1)
        ys.append(y_final)
        y_pred_list.append(y_pred)
        y_corr_list.append(y_corr)
        errors.append(err_exact)
        est_errors.append(abs(diff_pc))

    return (np.array(xs), np.array(ys), np.array(y_pred_list),
            np.array(y_corr_list), np.array(errors), np.array(est_errors))


def solve_adams_adaptive(x0: float, y0: float, x_end: float, h_start: float, eps: float):
    """
    Автоматичний вибір кроку для методу Адамса 2-го порядку.
    s = 2 -> k = 2^(s+1) = 8.
    Оцінка похибки: R = (1/6) * |y_kor - y_pred|.
    Якщо R > eps: зменшуємо крок h = h / 2.
    Якщо R < eps / 8: збільшуємо крок h = 2 * h.
    """
    cur_x = x0
    cur_y = y0
    h = h_start

    xs = [cur_x]
    ys = [cur_y]
    hs = [h]
    errors = [0.0]

    # Для Адамса потрібні дві точки. Зробимо перший крок через RK4:
    y_next = rk4_step(cur_x, cur_y, h)
    cur_x += h
    cur_y = y_next
    xs.append(cur_x)
    ys.append(cur_y)
    hs.append(h)
    errors.append(abs(cur_y - y_analytical(cur_x)))

    while cur_x < x_end - 1e-12:
        if cur_x + h > x_end:
            h = x_end - cur_x

        xn = xs[-1]
        yn = ys[-1]
        xn_prev = xs[-2]
        yn_prev = ys[-2]

        # Якщо попередній крок відрізнявся за довжиною, адаптуємо xn_prev через RK4
        fn = f_ode(xn, yn)
        fn_prev = f_ode(xn - h, rk4_step(xn, yn, -h))

        xn1 = xn + h
        y_pred = yn + 0.5 * h * (3.0 * fn - fn_prev)
        fn1_pred = f_ode(xn1, y_pred)
        y_corr = yn + 0.5 * h * (fn1_pred + fn)

        diff_pc = abs(y_corr - y_pred)
        r_err = (1.0 / 6.0) * diff_pc

        if r_err > eps and h > 1e-6:
            h = 0.5 * h
            continue

        y_final = y_corr - (1.0 / 6.0) * (y_corr - y_pred)
        cur_x = xn1
        cur_y = y_final

        xs.append(cur_x)
        ys.append(cur_y)
        hs.append(h)
        errors.append(abs(cur_y - y_analytical(cur_x)))

        if r_err < (eps / 8.0):
            h = min(2.0 * h, 0.2)

    return np.array(xs), np.array(ys), np.array(hs), np.array(errors)


# ----------------------------------------------------------------------
# 4. ПОБУДОВА ГРАФІКІВ ТА ЕКСПОРТ РЕЗУЛЬТАТІВ
# ----------------------------------------------------------------------

def generate_plots(plots_dir: str):
    """Генерація 7 обов'язкових графіків згідно з вимогами лабораторної роботи."""
    os.makedirs(plots_dir, exist_ok=True)
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

    x0, y0, x_end = 0.0, 0.5, 1.0

    # 1. Адамс: числовий розв'язок vs точний
    xs_a, ys_a, _, _, err_a, est_err_a = solve_adams_fixed(x0, y0, x_end, h=0.05)
    xx = np.linspace(x0, x_end, 200)
    yy_exact = y_analytical(xx)

    plt.figure(figsize=(7, 4.5))
    plt.plot(xx, yy_exact, 'b-', label='Точний розв’язок $y(x) = (x+1)^2 - 0.5 e^x$', linewidth=2)
    plt.plot(xs_a, ys_a, 'ro--', label='Метод Адамса (прогноз-корекція, $h=0.05$)', markersize=5, linewidth=1.2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Порівняння розв’язку методу Адамса 2-го порядку з точним')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'adams_solution.png'), dpi=300)
    plt.close()

    # 2. Адамс: графік похибки (дійсна vs оцінка y_кор - y_пр)
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs_a[1:], err_a[1:], 'r-o', label='Дійсна локальна похибка $\\phi_n = |y_n - y(x_n)|$', linewidth=1.5, markersize=5)
    plt.plot(xs_a[1:], est_err_a[1:], 'b--s', label='Оцінка похибки $|y_{n+1}^{кор} - y_{n+1}^{пр}|$', linewidth=1.5, markersize=4)
    plt.xlabel('x')
    plt.ylabel('Локальна похибка')
    plt.title('Дійсна похибка та оцінка прогнозу-корекції Адамса')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'adams_error.png'), dpi=300)
    plt.close()

    # 3. Адамс: графік автоматичного вибору кроку h(x)
    xs_ad_adapt, _, hs_ad_adapt, _ = solve_adams_adaptive(x0, y0, x_end, h_start=0.05, eps=1e-4)
    plt.figure(figsize=(7, 4.2))
    plt.step(xs_ad_adapt, hs_ad_adapt, where='post', color='darkgreen', linewidth=1.8, label='Адаптивний крок $h(x)$ ($\\varepsilon = 10^{-4}$)')
    plt.plot(xs_ad_adapt, hs_ad_adapt, 'o', color='darkgreen', markersize=4)
    plt.xlabel('x')
    plt.ylabel('Крок інтегрування h')
    plt.title('Залежність величини адаптивного кроку $h(x)$ (метод Адамса)')
    plt.ylim(0, max(hs_ad_adapt) * 1.25)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'adams_step_size.png'), dpi=300)
    plt.close()

    # 4. Рунге-Кутта 4: числовий розв'язок vs точний
    xs_rk, ys_rk, exact_rk, err_rk, runge_err_rk = solve_rk4_fixed(x0, y0, x_end, h=0.01)
    plt.figure(figsize=(7, 4.5))
    plt.plot(xx, yy_exact, 'b-', label='Точний розв’язок $y(x)$', linewidth=2)
    plt.plot(xs_rk[::5], ys_rk[::5], 'm^', label='Метод Рунге-Кутта 4-го порядку ($h=0.01$)', markersize=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Порівняння розв’язку методу Рунге-Кутта 4-го порядку з точним')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rk4_solution.png'), dpi=300)
    plt.close()

    # 5. Рунге-Кутта 4: дійсна похибка vs оцінка за правилом Рунге (глобальна)
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs_rk, err_rk, 'r-', label='Дійсна похибка $\\phi_n = |y_n - y(x_n)|$', linewidth=2.0)
    plt.plot(xs_rk, runge_err_rk, 'k--', label='Оцінка за правилом Рунге $\\Phi_n = \\frac{16}{15}|y_n^{(h)} - y_{2n}^{(h/2)}|$', linewidth=1.5)
    plt.xlabel('x')
    plt.ylabel('Похибка')
    plt.title('Порівняння дійсної похибки та оцінки за правилом Рунге (RK4, $h=0.01$)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rk4_error.png'), dpi=300)
    plt.close()

    # 6. Рунге-Кутта 4: графік адаптивного кроку h(x)
    xs_rk_adapt, _, hs_rk_adapt, _ = solve_rk4_adaptive(x0, y0, x_end, h_start=0.1, eps=1e-8)
    plt.figure(figsize=(7, 4.2))
    plt.step(xs_rk_adapt, hs_rk_adapt, where='post', color='indigo', linewidth=1.8, label='Адаптивний крок $h(x)$ ($\\varepsilon = 10^{-8}$)')
    plt.plot(xs_rk_adapt, hs_rk_adapt, 'd', color='indigo', markersize=4)
    plt.xlabel('x')
    plt.ylabel('Крок інтегрування h')
    plt.title('Залежність величини адаптивного кроку $h(x)$ (метод Рунге-Кутта)')
    plt.ylim(0, max(hs_rk_adapt) * 1.25)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rk4_step_size.png'), dpi=300)
    plt.close()

    # 7. Рунге-Кутта 4: дослідження порядку збіжності O(h^4)
    steps_test = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    h_arr, max_errors = investigate_rk4_convergence(x0, y0, x_end, steps_test)

    # Розрахунок емпіричного порядку збіжності між сусідніми кроками
    p_empirical = np.log(max_errors[:-1] / max_errors[1:]) / np.log(h_arr[:-1] / h_arr[1:])

    plt.figure(figsize=(7, 4.5))
    plt.loglog(h_arr, max_errors, 'rs-', linewidth=1.8, markersize=6, label='Експериментальна похибка $\\max |y_n - y(x_n)|$')
    # Еталонна пряма O(h^4)
    c_ref = max_errors[-1] / (h_arr[-1]**4)
    plt.loglog(h_arr, c_ref * (h_arr**4), 'k--', linewidth=1.5, label='Теоретичний нахил $\\mathcal{O}(h^4)$ (4-й порядок)')
    plt.xlabel('Крок інтегрування h (лог. шкала)')
    plt.ylabel('Максимальна похибка (лог. шкала)')
    plt.title('Дослідження порядку збіжності методу Рунге-Кутта ($p \\approx 4$)')
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'rk4_convergence.png'), dpi=300)
    plt.close()

    print("[+] Усі 7 графіків успішно збережено у директорію:", plots_dir)
    return p_empirical


def export_results(base_dir: str):
    """Експорт результатів розрахунків у текстові файли."""
    x0, y0, x_end = 0.0, 0.5, 1.0

    # 1. Адамс (h = 0.05)
    xs_a, ys_a, y_pred, y_corr, err_a, est_err_a = solve_adams_fixed(x0, y0, x_end, h=0.05)
    file_adams = os.path.join(base_dir, 'results_adams.txt')
    with open(file_adams, 'w', encoding='utf-8') as f:
        f.write("# Протокол чисельного розв'язку методом прогнозу-корекції Адамса 2-го порядку\n")
        f.write(f"# Рівняння: dy/dx = y - x^2 + 1, x in [{x0}, {x_end}], y(0) = {y0}, крок h = 0.05\n")
        f.write("# " + "-" * 88 + "\n")
        f.write(f"# {'n':>3} | {'x_n':>7} | {'y_pred':>13} | {'y_corr':>13} | {'y_final':>13} | {'y_exact':>13} | {'err_exact':>11} | {'|y_c - y_p|':>11}\n")
        f.write("# " + "-" * 88 + "\n")
        for i in range(len(xs_a)):
            f.write(f"{i:4d} | {xs_a[i]:7.4f} | {y_pred[i]:13.8f} | {y_corr[i]:13.8f} | {ys_a[i]:13.8f} | {y_analytical(xs_a[i]):13.8f} | {err_a[i]:11.3e} | {est_err_a[i]:11.3e}\n")
    print(f"[+] Протокол Адамса збережено: {file_adams}")

    # 2. Рунге-Кутта 4 (h = 0.01)
    xs_rk, ys_rk, exact_rk, err_rk, runge_err_rk = solve_rk4_fixed(x0, y0, x_end, h=0.01)
    file_rk4 = os.path.join(base_dir, 'results_rk4.txt')
    with open(file_rk4, 'w', encoding='utf-8') as f:
        f.write("# Протокол чисельного розв'язку методом Рунге-Кутта 4-го порядку\n")
        f.write(f"# Рівняння: dy/dx = y - x^2 + 1, x in [{x0}, {x_end}], y(0) = {y0}, крок h = 0.01\n")
        f.write("# " + "-" * 76 + "\n")
        f.write(f"# {'n':>4} | {'x_n':>7} | {'y_num':>15} | {'y_exact':>15} | {'err_exact':>13} | {'runge_est':>13}\n")
        f.write("# " + "-" * 76 + "\n")
        for i in range(len(xs_rk)):
            f.write(f"{i:4d} | {xs_rk[i]:7.4f} | {ys_rk[i]:15.10f} | {exact_rk[i]:15.10f} | {err_rk[i]:13.4e} | {runge_err_rk[i]:13.4e}\n")
    print(f"[+] Протокол Рунге-Кутта збережено: {file_rk4}")


def main():
    print("=" * 70)
    print("Лабораторна робота №10: Методи Рунге-Кутта та Адамса для задачі Коші")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(base_dir, 'plots')

    # Генерація графіків
    p_emp = generate_plots(plots_dir)
    print(f"[*] Емпіричний порядок збіжності RK4: {p_emp.round(2)}")

    # Експорт табличних результатів
    export_results(base_dir)

    print("=" * 70)
    print("[V] Розрахунки лабораторної роботи №10 завершено успішно!")
    print("=" * 70)


if __name__ == '__main__':
    main()
