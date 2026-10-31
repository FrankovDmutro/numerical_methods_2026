import math
import os
import subprocess
import numpy as np
import pygments
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer


def get_html_report(data: dict) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    main_py_path = os.path.join(base_dir, "main.py")
    with open(main_py_path, "r", encoding="utf-8") as f:
        code_text = f.read()

    formatter = HtmlFormatter(noclasses=True, style="friendly", nowrap=False)
    highlighted_code = pygments.highlight(code_text, PythonLexer(), formatter)

    plot_rel = "lab6_convergence.png"

    # Build iteration table rows
    iter_rows = ""
    for rec in data["history"]:
        k = rec["iteration"]
        d_val = f"{rec['delta_norm']:.4e}" if not math.isnan(rec["delta_norm"]) else "— (початковий)"
        err_val = f"{rec['error_norm']:.4e}"
        res_val = f"{rec['residual_norm']:.4e}"
        status = "<span class='badge-success'>Досягнуто</span>" if rec["converged"] else (
            "<span class='badge-info'>Початок</span>" if k == 0 else "<span class='badge-warning'>В процесі</span>"
        )
        iter_rows += f"""
        <tr>
            <td style="text-align: center; font-weight: bold;">{k}</td>
            <td style="text-align: right; font-family: monospace;">{d_val}</td>
            <td style="text-align: right; font-family: monospace;">{err_val}</td>
            <td style="text-align: right; font-family: monospace;">{res_val}</td>
            <td style="text-align: center;">{status}</td>
        </tr>
        """

    # Build sample coordinates table (10 points across the vector)
    sample_indices = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    sample_rows = ""
    for idx in sample_indices:
        i_display = idx + 1
        x_exact_val = data["x_exact"][idx]
        x0_val = data["x0"][idx]
        x_ref_val = data["x_refined"][idx]
        err_init = abs(x0_val - x_exact_val)
        err_ref = abs(x_ref_val - x_exact_val)
        sample_rows += f"""
        <tr>
            <td style="text-align: center; font-weight: bold;">{i_display}</td>
            <td style="text-align: right; font-family: monospace;">{x_exact_val:.1f}</td>
            <td style="text-align: right; font-family: monospace;">{x0_val:.14f}</td>
            <td style="text-align: right; font-family: monospace; color: #a94442;">{err_init:.4e}</td>
            <td style="text-align: right; font-family: monospace;">{x_ref_val:.16f}</td>
            <td style="text-align: right; font-family: monospace; color: #3c763d; font-weight: bold;">{err_ref:.4e}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <title>МТО_ЛАБ6_ФРАНКОВ</title>
    <style>
        @page {{
            size: A4;
            margin: 18mm 15mm 18mm 20mm;
            @bottom-right {{
                content: counter(page);
                font-family: 'Times New Roman', serif;
                font-size: 10pt;
            }}
        }}

        @page :first {{
            @bottom-right {{
                content: normal;
            }}
        }}

        body {{
            font-family: 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.3;
            color: #111;
            margin: 0;
            padding: 0;
            background-color: #fff;
        }}

        .page-break {{
            page-break-after: always;
            break-after: page;
        }}

        /* Title page styling */
        .title-page {{
            height: 94vh;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            text-align: center;
            box-sizing: border-box;
            padding: 10mm 5mm;
        }}

        .title-header {{
            font-size: 13pt;
            font-weight: bold;
            line-height: 1.4;
            text-transform: uppercase;
        }}

        .title-center {{
            margin-top: 14vh;
            margin-bottom: 14vh;
        }}

        .title-lab-num {{
            font-size: 21pt;
            font-weight: bold;
            letter-spacing: 1px;
            margin-bottom: 12px;
        }}

        .title-subject {{
            font-size: 14pt;
            margin-bottom: 14px;
        }}

        .title-topic {{
            font-size: 16pt;
            font-weight: bold;
            line-height: 1.35;
            color: #1a2a4a;
        }}

        .title-author {{
            width: 100%;
            display: flex;
            justify-content: flex-end;
            text-align: left;
            margin-bottom: 7vh;
        }}

        .title-author-block {{
            display: inline-block;
            text-align: left;
            font-size: 12pt;
            line-height: 1.5;
            margin-left: auto;
            margin-right: 25px;
        }}

        .title-footer {{
            font-size: 12pt;
            font-weight: bold;
        }}

        /* Headings */
        h2 {{
            font-size: 13.5pt;
            color: #1a2a4a;
            border-bottom: 1.5px solid #2a3b5c;
            padding-bottom: 3px;
            margin-top: 14px;
            margin-bottom: 8px;
        }}

        h3 {{
            font-size: 11.5pt;
            color: #222;
            margin-top: 10px;
            margin-bottom: 5px;
        }}

        p {{
            text-align: justify;
            text-indent: 1.25cm;
            margin-top: 3px;
            margin-bottom: 5px;
        }}

        p.no-indent {{
            text-indent: 0;
        }}

        ul, ol {{
            margin-top: 3px;
            margin-bottom: 6px;
            padding-left: 1.5cm;
        }}

        li {{
            margin-bottom: 3px;
            text-align: justify;
        }}

        .formula-box {{
            background-color: #f7f9fc;
            border-left: 3px solid #2a3b5c;
            padding: 6px 14px;
            margin: 7px 0;
            font-style: italic;
            text-align: center;
            font-size: 11pt;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 9.5pt;
        }}

        table, th, td {{
            border: 1px solid #c8d0da;
        }}

        th {{
            background-color: #f0f4f8;
            color: #1a2a4a;
            font-weight: bold;
            padding: 5px 6px;
            text-align: center;
        }}

        td {{
            padding: 4px 6px;
        }}

        .badge-success {{
            background-color: #dff0d8;
            color: #3c763d;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 8.5pt;
            font-weight: bold;
            border: 1px solid #d6e9c6;
        }}

        .badge-info {{
            background-color: #d9edf7;
            color: #31708f;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 8.5pt;
            border: 1px solid #bce8f1;
        }}

        .badge-warning {{
            background-color: #fcf8e3;
            color: #8a6d3b;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 8.5pt;
            border: 1px solid #faebcc;
        }}

        .img-container {{
            text-align: center;
            margin: 12px 0;
        }}

        .img-container img {{
            max-width: 98%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }}

        .img-caption {{
            font-size: 9.5pt;
            color: #555;
            margin-top: 5px;
            font-style: italic;
            text-align: center;
        }}

        .highlight-container {{
            background-color: #fcfcfc;
            border: 1px solid #e1e4e8;
            border-radius: 4px;
            padding: 6px 10px;
            margin: 8px 0;
            overflow-x: auto;
            font-size: 8pt;
            line-height: 1.2;
            font-family: Consolas, 'Courier New', monospace;
        }}

        .metric-card-row {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            gap: 8px;
        }}

        .metric-card {{
            flex: 1;
            background-color: #f8fafc;
            border: 1px solid #dbe2ea;
            border-radius: 5px;
            padding: 8px 6px;
            text-align: center;
        }}

        .metric-title {{
            font-size: 9pt;
            color: #555;
            margin-bottom: 3px;
        }}

        .metric-value {{
            font-size: 12pt;
            font-weight: bold;
            color: #1a2a4a;
            font-family: monospace;
        }}
    </style>
</head>
<body>

    <!-- 1. ТИТУЛЬНА СТОРІНКА -->
    <div class="title-page page-break">
        <div class="title-header">
            Міністерство освіти і науки України<br>
            Львівський національний університет імені Івана Франка<br>
            Факультет електроніки та комп'ютерних технологій<br>
            Кафедра системного проектування
        </div>

        <div class="title-center">
            <div class="title-lab-num">ЗВІТ</div>
            <div class="title-subject">з лабораторної роботи №6<br>з дисципліни «Методи та технології обчислень»</div>
            <div class="title-topic">«LU-розклад. Ітераційні методи уточнення розв’язку СЛАР»</div>
        </div>

        <div>
            <div class="title-author">
                <div class="title-author-block">
                    <b>Виконав:</b><br>
                    студент 3 курсу групи ФеП-33<br>
                    <b>Франков Дмитро Олександрович</b><br><br>
                    <b>Перевірив:</b><br>
                    асистент<br>
                    <b>Каськун Олег Данилович</b>
                </div>
            </div>

            <div class="title-footer">
                Львів — 2026
            </div>
        </div>
    </div>

    <!-- 2. МЕТА ТА ТЕОРІЯ -->
    <div>
        <h2>Мета роботи</h2>
        <p>
            Вивчити метод розв’язку системи лінійних алгебраїчних рівнянь за допомогою <i>LU</i>-розкладу (алгоритм Дулітла) та ітераційний метод уточнення розв’язку СЛАР. Дослідити алгоритмічну реалізацію розкладу матриці <i>A</i> на добуток нижньої трикутної матриці <i>L</i> та верхньої унітрикутної матриці <i>U</i> (з одиничною головною діагоналлю), розв'язання двох трикутних систем підстановками, оцінку точності знайденого наближення та збіжність ітераційного уточнення до заданого порогу похибки <i>eps</i><sub>0</sub> = 10<sup>&minus;14</sup>.
        </p>

        <h2>Опис виконання роботи</h2>
        
        <h3>1. Теоретичні основи LU-розкладу (алгоритм Дулітла)</h3>
        <p>
            <i>LU</i>-розклад полягає у представленні квадратної матриці <i>A</i> розмірності <i>n</i> &times; <i>n</i> у вигляді добутку двох трикутних матриць:
        </p>
        <div class="formula-box">
            <i>A</i> = <i>L</i> &middot; <i>U</i>
        </div>
        <p>
            де <i>L</i> = (<i>l<sub>ik</sub></i>) — нижня трикутна матриця (<i>l<sub>ik</sub></i> = 0 при <i>k</i> &gt; <i>i</i>), а <i>U</i> = (<i>u<sub>ki</sub></i>) — верхня трикутна матриця з одиницями на головній діагоналі: <i>u<sub>kk</sub></i> = 1 для всіх <i>k</i> = 1, &hellip;, <i>n</i>, та <i>u<sub>ki</sub></i> = 0 при <i>k</i> &gt; <i>i</i>.
        </p>
        <p>
            Згідно з означенням матричного множення елемент <i>a<sub>ik</sub></i> визначається сумою:
        </p>
        <div class="formula-box">
            <i>a<sub>ik</sub></i> = &sum;<sub><i>j</i>=1</sub><sup><i>n</i></sup> <i>l<sub>ij</sub></i> <i>u<sub>jk</sub></i> = &sum;<sub><i>j</i>=1</sub><sup>min(<i>i</i>, <i>k</i>)</sup> <i>l<sub>ij</sub></i> <i>u<sub>jk</sub></i>
        </div>
        <p>
            Звідси отримуємо рекурентний алгоритм послідовного знаходження стовпців <i>L</i> та рядків <i>U</i>:
        </p>
        <ol>
            <li>Ініціалізуємо матриці <i>L</i> та <i>U</i> розмірності <i>n</i> &times; <i>n</i> нулями.</li>
            <li>Задаємо значення діагональних елементів матриці <i>U</i>: <i>u<sub>ii</sub></i> = 1 (<i>i</i> = 1, &hellip;, <i>n</i>).</li>
            <li>
                Почергово для кожного <i>k</i> = 1, &hellip;, <i>n</i> обчислюємо:
                <ul>
                    <li>
                        Елементи <i>k</i>-го стовпця матриці <i>L</i> (для <i>i</i> &ge; <i>k</i>):
                        <div class="formula-box" style="margin: 3px 0;">
                            <i>l<sub>ik</sub></i> = <i>a<sub>ik</sub></i> &minus; &sum;<sub><i>j</i>=1</sub><sup><i>k</i>&minus;1</sup> <i>l<sub>ij</sub></i> <i>u<sub>jk</sub></i>, &emsp; <i>i</i> &ge; <i>k</i>
                        </div>
                    </li>
                    <li>
                        Елементи <i>k</i>-го рядка матриці <i>U</i> (для <i>i</i> &gt; <i>k</i>):
                        <div class="formula-box" style="margin: 3px 0;">
                            <i>u<sub>ki</sub></i> = (1 / <i>l<sub>kk</sub></i>) &middot; ( <i>a<sub>ki</sub></i> &minus; &sum;<sub><i>j</i>=1</sub><sup><i>k</i>&minus;1</sup> <i>l<sub>kj</sub></i> <i>u<sub>ji</sub></i> ), &emsp; <i>i</i> &gt; <i>k</i>
                        </div>
                    </li>
                </ul>
            </li>
        </ol>

        <h3>2. Розв'язання СЛАР за допомогою LU-розкладу</h3>
        <p>
            Вихідна система лінійних алгебраїчних рівнянь <i>A</i> <i>X</i> = <i>B</i> з урахуванням розкладу <i>A</i> = <i>L</i> <i>U</i> перетворюється на систему <i>L</i> <i>U</i> <i>X</i> = <i>B</i>. Її розв'язок знаходиться послідовним розв'язанням двох трикутних систем:
        </p>
        <ol>
            <li>
                <b>Прямий хід (розв'язання системи <i>L</i> <i>Z</i> = <i>B</i>):</b>
                <div class="formula-box">
                    <i>z</i><sub>1</sub> = <i>b</i><sub>1</sub> / <i>l</i><sub>11</sub>,<br>
                    <i>z<sub>k</sub></i> = (1 / <i>l<sub>kk</sub></i>) &middot; ( <i>b<sub>k</sub></i> &minus; &sum;<sub><i>j</i>=1</sub><sup><i>k</i>&minus;1</sup> <i>l<sub>kj</sub></i> <i>z<sub>j</sub></i> ), &emsp; <i>k</i> = 2, &hellip;, <i>n</i>
                </div>
            </li>
            <li>
                <b>Зворотний хід (розв'язання системи <i>U</i> <i>X</i> = <i>Z</i>, враховуючи <i>u<sub>ii</sub></i> = 1):</b>
                <div class="formula-box">
                    <i>x<sub>n</sub></i> = <i>z<sub>n</sub></i>,<br>
                    <i>x<sub>k</sub></i> = <i>z<sub>k</sub></i> &minus; &sum;<sub><i>j</i>=<i>k</i>+1</sub><sup><i>n</i></sup> <i>u<sub>kj</sub></i> <i>x<sub>j</sub></i>, &emsp; <i>k</i> = <i>n</i>&minus;1, &hellip;, 1
                </div>
            </li>
        </ol>
    </div>

    <div class="page-break"></div>

    <!-- 3. ТЕОРІЯ УТОЧНЕННЯ ТА ОПИС МОДУЛІВ -->
    <div>
        <h3>3. Ітераційний метод уточнення розв'язку СЛАР</h3>
        <p>
            Нехай для системи лінійних алгебраїчних рівнянь <i>A</i> <i>X</i> = <i>B</i> знайдено деякий початковий розв'язок <i>X</i><sup>(0)</sup>. Введемо позначення:
        </p>
        <ul>
            <li>&Delta;<i>X</i><sup>(<i>k</i>)</sup> = <i>X</i> &minus; <i>X</i><sup>(<i>k</i>)</sup> — вектор похибки значень невідомих на <i>k</i>-й ітерації;</li>
            <li><i>R</i><sup>(<i>k</i>)</sup> = <i>B</i> &minus; <i>B</i><sup>(<i>k</i>)</sup> = <i>B</i> &minus; <i>A</i> <i>X</i><sup>(<i>k</i>)</sup> — вектор нев'язки (де <i>B</i><sup>(<i>k</i>)</sup> = <i>A</i> <i>X</i><sup>(<i>k</i>)</sup> — знайдений вектор вільних членів).</li>
        </ul>
        <p>
            Підставивши точний розв'язок <i>X</i> = <i>X</i><sup>(<i>k</i>)</sup> + &Delta;<i>X</i><sup>(<i>k</i>)</sup> у вихідну систему, маємо:
        </p>
        <div class="formula-box">
            <i>A</i> ( <i>X</i><sup>(<i>k</i>)</sup> + &Delta;<i>X</i><sup>(<i>k</i>)</sup> ) = <i>B</i> &emsp;&rArr;&emsp; <i>A</i> <i>X</i><sup>(<i>k</i>)</sup> + <i>A</i> &Delta;<i>X</i><sup>(<i>k</i>)</sup> = <i>B</i> &emsp;&rArr;&emsp; <i>A</i> &Delta;<i>X</i><sup>(<i>k</i>)</sup> = <i>R</i><sup>(<i>k</i>)</sup>
        </div>
        <p>
            Розв'язавши цю систему рівнянь, знаходимо поправку &Delta;<i>X</i><sup>(<i>k</i>)</sup>. Оскільки матриця системи <i>A</i> залишається тією ж самою, розв'язання виконується за допомогою раніше знайденого <i>LU</i>-розкладу (послідовно <i>L</i> <i>Z</i> = <i>R</i><sup>(<i>k</i>)</sup> та <i>U</i> &Delta;<i>X</i><sup>(<i>k</i>)</sup> = <i>Z</i>), що вимагає лише <i>O</i>(<i>n</i><sup>2</sup>) дій замість <i>O</i>(<i>n</i><sup>3</sup>).
        </p>
        <p>
            Обчислюємо уточнене значення розв'язку:
        </p>
        <div class="formula-box">
            <i>X</i><sup>(<i>k</i>+1)</sup> = <i>X</i><sup>(<i>k</i>)</sup> + &Delta;<i>X</i><sup>(<i>k</i>)</sup>
        </div>
        <p>
            Умовами закінчення ітераційної процедури є виконання співвідношень:
        </p>
        <div class="formula-box">
            ||&Delta;<i>X</i><sup>(<i>k</i>)</sup>||<sub>&infin;</sub> &le; <i>eps</i><sub>0</sub> &emsp; або &emsp; ||<i>A</i> <i>X</i><sup>(<i>k</i>+1)</sup> &minus; <i>B</i>||<sub>&infin;</sub> &le; <i>eps</i><sub>0</sub>, &emsp; <i>eps</i><sub>0</sub> = 10<sup>&minus;14</sup>
        </div>
        <p>
            Якщо умови виконуються, <i>X</i><sup>(<i>k</i>+1)</sup> приймається як розв'язок із заданою точністю. Якщо ні, приймаємо <i>X</i><sup>(<i>k</i>)</sup> = <i>X</i><sup>(<i>k</i>+1)</sup> і повторюємо ітераційний крок.
        </p>

        <h3>4. Опис розроблених модулів програми</h3>
        <p class="no-indent">
            У програмі <code>main.py</code> реалізовано повний набір функцій згідно з вимогами пункту 2:
        </p>
        <ul>
            <li><code>generate_matrix(n, low, high, seed)</code>: генерація випадкової матриці <i>A</i> розмірності <i>n</i> &times; <i>n</i> (<i>n</i> = 100);</li>
            <li><code>write_matrix(A, filename)</code> / <code>read_matrix(filename)</code>: збереження та зчитування матриці <i>A</i> у/з текстового файлу;</li>
            <li><code>write_vector(b, filename)</code> / <code>read_vector(filename)</code>: збереження та зчитування вектора вільних членів <i>B</i> у/з текстового файлу;</li>
            <li><code>lu_decomposition(A)</code>: обчислення <i>LU</i>-розкладу квадратної матриці методом Дулітла;</li>
            <li><code>write_lu(L, U, filename)</code>: запис знайдених матриць <i>L</i> та <i>U</i> у текстовий файл;</li>
            <li><code>solve_lu(L, U, b)</code>: розв'язання системи лінійних рівнянь за допомогою факторів <i>L</i> та <i>U</i>;</li>
            <li><code>matrix_vector_mult(A, x)</code>: обчислення добутку матриці на вектор з використанням <code>math.fsum</code> для мінімізації похибки додавання;</li>
            <li><code>vector_norm(v, ord)</code>: обчислення векторних норм (зокрема рівномірної ||<i>v</i>||<sub>&infin;</sub>);</li>
            <li><code>iterative_refinement(A, L, U, b, x0, ...)</code>: реалізація ітераційного уточнення розв'язку.</li>
        </ul>
    </div>

    <div class="page-break"></div>

    <!-- 4. РЕЗУЛЬТАТИ ВИКОНАННЯ РОБОТИ -->
    <div>
        <h2>Результати виконання роботи</h2>
        <p>
            Згенеровано матрицю <i>A</i> розмірності 100 &times; 100. Задано точний розв'язок <i>x<sub>i</sub></i><sup>*</sup> = 2.5 (<i>i</i> = 1, &hellip;, 100). Обчислено вектор вільних членів <i>b<sub>i</sub></i> = &sum;<sub><i>j</i>=1</sub><sup><i>n</i></sup> <i>a<sub>ij</sub></i> <i>x<sub>j</sub></i><sup>*</sup>.
        </p>

        <div class="metric-card-row">
            <div class="metric-card">
                <div class="metric-title">Розмірність системи</div>
                <div class="metric-value">100 &times; 100</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Число обумовленості &kappa;<sub>&infin;</sub>(A)</div>
                <div class="metric-value">{data['cond_number']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Точність розкладу ||A &minus; LU||</div>
                <div class="metric-value">{data['decomp_error']:.4e}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Кількість ітерацій</div>
                <div class="metric-value">{data['iterations_count']}</div>
            </div>
        </div>

        <p class="no-indent"><b>Характеристики початкового розв'язку X<sup>(0)</sup>:</b></p>
        <ul style="margin-top: 2px; margin-bottom: 6px;">
            <li>Норма початкової нев'язки: <i>eps</i> = ||<i>A</i> <i>X</i><sup>(0)</sup> &minus; <i>B</i>||<sub>&infin;</sub> = <b>{data['eps_init_res']:.4e}</b></li>
            <li>Абсолютна похибка відносно точного вектора: ||<i>X</i><sup>(0)</sup> &minus; <i>X</i><sup>*</sup>||<sub>&infin;</sub> = <b>{data['eps_init_err']:.4e}</b></li>
            <li>Відносна похибка: ||<i>X</i><sup>(0)</sup> &minus; <i>X</i><sup>*</sup>||<sub>&infin;</sub> / ||<i>X</i><sup>*</sup>||<sub>&infin;</sub> = <b>{data['rel_init_err']:.4e}</b></li>
        </ul>

        <p class="no-indent"><b>Динаміка ітераційного уточнення розв'язку СЛАР (до порогу eps<sub>0</sub> = 10<sup>&minus;14</sup>):</b></p>
        <table>
            <thead>
                <tr>
                    <th style="width: 10%;">Ітерація <i>k</i></th>
                    <th style="width: 25%;">Норма поправки ||&Delta;<i>X</i><sup>(<i>k</i>)</sup>||<sub>&infin;</sub></th>
                    <th style="width: 25%;">Похибка ||<i>X</i><sup>(<i>k</i>)</sup> &minus; <i>X</i><sup>*</sup>||<sub>&infin;</sub></th>
                    <th style="width: 25%;">Нев'язка ||<i>A</i><i>X</i><sup>(<i>k</i>)</sup> &minus; <i>B</i>||<sub>&infin;</sub></th>
                    <th style="width: 15%;">Статус</th>
                </tr>
            </thead>
            <tbody>
                {iter_rows}
            </tbody>
        </table>

        <p class="no-indent"><b>Порівняння значень розв'язку для вибіркових компонент вектора (n = 100):</b></p>
        <table>
            <thead>
                <tr>
                    <th style="width: 8%;">Індекс <i>i</i></th>
                    <th style="width: 14%;">Точне <i>x<sub>i</sub></i><sup>*</sup></th>
                    <th style="width: 22%;">Початкове <i>x<sub>i</sub></i><sup>(0)</sup></th>
                    <th style="width: 18%;">Похибка поч.</th>
                    <th style="width: 22%;">Уточнене <i>x<sub>i</sub></i><sup>(ref)</sup></th>
                    <th style="width: 16%;">Похибка уточн.</th>
                </tr>
            </thead>
            <tbody>
                {sample_rows}
            </tbody>
        </table>
    </div>

    <div class="page-break"></div>

    <!-- 5. ГРАФІЧНА ІЛЮСТРАЦІЯ -->
    <div>
        <h2>Графічна ілюстрація результатів</h2>
        <p>
            На рис. 1 наведено результати експериментального дослідження збіжності ітераційного методу уточнення розв'язку СЛАР та покомпонентний профіль розподілу похибок для всіх 100 координат.
        </p>

        <div class="img-container">
            <img src="{plot_rel}" alt="Графіки збіжності та похибок">
            <div class="img-caption">
                Рис. 1. Ліворуч: динаміка зменшення норми поправки ||&Delta;X<sup>(k)</sup>||<sub>&infin;</sub>, похибки ||X<sup>(k)</sup> &minus; X<sup>*</sup>||<sub>&infin;</sub> та нев'язки ||AX<sup>(k)</sup> &minus; B||<sub>&infin;</sub>. Праворуч: покомпонентна похибка наближення на всіх 100 змінних до та після уточнення.
            </div>
        </div>

        <p>
            <b>Аналіз графічних матеріалів:</b>
        </p>
        <ul>
            <li>
                <b>Лівий графік:</b> демонструє зменшення величини поправки ||&Delta;<i>X</i><sup>(<i>k</i>)</sup>||<sub>&infin;</sub> за 2 ітерації: від 8.10 &times; 10<sup>&minus;13</sup> до 8.94 &times; 10<sup>&minus;15</sup>, що менше встановленого порогу <i>eps</i><sub>0</sub> = 10<sup>&minus;14</sup>. При цьому норма нев'язки зменшилася з 1.59 &times; 10<sup>&minus;11</sup> до 5.68 &times; 10<sup>&minus;14</sup>, що відповідає граничному рівню похибки машинної арифметики подвійної точності для матриць даної розмірності.
            </li>
            <li>
                <b>Правий графік:</b> демонструє покомпонентний розподіл похибки |<i>x<sub>i</sub></i> &minus; 2.5| на всіх 100 компонентах. Жовта лінія (початкове наближення <i>x<sub>i</sub></i><sup>(0)</sup>) має рівень похибок у діапазоні 10<sup>&minus;13</sup> &hellip; 10<sup>&minus;12</sup>, тоді як синя лінія (після ітераційного уточнення) стабільно опускається в діапазон 10<sup>&minus;15</sup> &hellip; 10<sup>&minus;14</sup> для абсолютно всіх компонент системи.
            </li>
        </ul>
    </div>

    <div class="page-break"></div>

    <!-- 6. ЛІСТИНГ ПРОГРАМИ -->
    <div>
        <h2>Лістинг програми</h2>
        <p>
            Нижче наведено повний вихідний код програми (файл <code>main.py</code>), розроблений мовою Python з використанням бібліотек <code>numpy</code> та <code>matplotlib</code>:
        </p>

        <div class="highlight-container">
            {highlighted_code}
        </div>
    </div>

    <div class="page-break"></div>

    <!-- 7. ВИСНОВКИ -->
    <div>
        <h2>Висновки</h2>
        <ol>
            <li>
                <b>Опанування теоретичного та практичного базису:</b> У ході виконання лабораторної роботи №6 детально вивчено та реалізовано прямий метод розв’язку систем лінійних алгебраїчних рівнянь за допомогою <i>LU</i>-розкладу квадратної матриці (алгоритм Дулітла з одиничною діагоналлю матриці <i>U</i>) та ітераційний метод уточнення розв’язку.
            </li>
            <li>
                <b>Обчислювальна ефективність та перевага LU-розкладу:</b> Знаходження <i>LU</i>-розкладу матриці <i>A</i> вимагає <i>O</i>(<i>n</i><sup>3</sup> / 3) арифметичних операцій. Проте після виконання факторизації розв'язання двох трикутних систем <i>L</i> <i>Z</i> = <i>B</i> та <i>U</i> <i>X</i> = <i>Z</i> виконується прямим і зворотним ходом лише за <i>O</i>(<i>n</i><sup>2</sup>) операцій. Це забезпечує вирішальну перевагу в ітераційних процесах уточнення: оскільки вихідна матриця <i>A</i> не змінюється від ітерації до ітерації, <i>LU</i>-розклад виконується <b>лише один раз</b>, а кожен крок уточнення обчислюється практично миттєво.
            </li>
            <li>
                <b>Швидкість збіжності ітераційного уточнення:</b> Початкове наближення <i>X</i><sup>(0)</sup>, знайдене прямим <i>LU</i>-розв'язком, мало похибку ||<i>X</i><sup>(0)</sup> &minus; <i>X</i><sup>*</sup>||<sub>&infin;</sub> = {data['eps_init_err']:.4e} та нев'язку {data['eps_init_res']:.4e}. Застосування ітераційного уточнення дозволило вже на 2-й ітерації досягти норми поправки ||&Delta;<i>X</i><sup>(2)</sup>||<sub>&infin;</sub> = {data['history'][2]['delta_norm']:.4e} &le; 10<sup>&minus;14</sup>, що успішно виконало заданий критерій зупинки <i>eps</i><sub>0</sub> = 10<sup>&minus;14</sup>.
            </li>
            <li>
                <b>Аналіз чисельної точності та обумовленості:</b> Для згенерованої випадкової матриці 100 &times; 100 число обумовленості склало &kappa;<sub>&infin;</sub>(<i>A</i>) &approx; {data['cond_number']:.2f}. Теоретична границя машинної похибки становить &kappa;(<i>A</i>) &middot; &epsilon;<sub>mach</sub> &approx; 3.5 &times; 10<sup>&minus;14</sup>. У результаті ітераційного уточнення похибка зменшилася до {data['final_err']:.4e}, а норма нев'язки склала {data['final_res']:.4e}, що відповідає граничній точності обчислень з плаваючою комою стандарту IEEE 754 (подвійна точність <code>float64</code>).
            </li>
            <li>
                <b>Повнота виконання вимог завдання:</b> Створено всі необхідні програмні модулі: генерацію матриці <i>A</i> та вектора <i>B</i>, збереження та зчитування матриць і векторів у текстові файли (<code>matrix_A.txt</code>, <code>vector_b.txt</code>, <code>lu_decomposition.txt</code>), функції множення матриці на вектор та розрахунку векторних норм. Правильність результатів підтверджено аналітично заданим тестовим вектором <i>x<sub>i</sub></i><sup>*</sup> = 2.5.
            </li>
        </ol>
    </div>

</body>
</html>
    """
    return html


def run_pipeline_and_generate_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    matrix_file = os.path.join(base_dir, "matrix_A.txt")
    vector_file = os.path.join(base_dir, "vector_b.txt")
    lu_file = os.path.join(base_dir, "lu_decomposition.txt")
    plot_file = os.path.join(base_dir, "lab6_convergence.png")
    html_file = os.path.join(base_dir, "report.html")
    pdf_file = os.path.join(base_dir, "report.pdf")

    import main as m

    n = 100
    exact_val = 2.5
    A = m.generate_matrix(n=n, low=-10.0, high=10.0, seed=42)
    m.write_matrix(A, matrix_file)

    x_exact = np.full(n, exact_val, dtype=np.float64)
    b = m.matrix_vector_mult(A, x_exact)
    m.write_vector(b, vector_file)

    L, U = m.lu_decomposition(A)
    m.write_lu(L, U, lu_file)
    decomp_error = m.vector_norm(A - L @ U)

    x0 = m.solve_lu(L, U, b)
    eps_init_res = m.vector_norm(m.matrix_vector_mult(A, x0) - b)
    eps_init_err = m.vector_norm(x0 - x_exact)
    rel_init_err = eps_init_err / m.vector_norm(x_exact)

    eps0 = 1e-14
    x_refined, history = m.iterative_refinement(
        A, L, U, b, x0, x_exact=x_exact, eps0=eps0, max_iter=20
    )

    cond_number = float(np.linalg.cond(A, p=np.inf))
    final_err = m.vector_norm(x_refined - x_exact)
    final_res = m.vector_norm(m.matrix_vector_mult(A, x_refined) - b)
    iterations_count = len(history) - 1

    data = {
        "n": n,
        "exact_val": exact_val,
        "x_exact": x_exact,
        "A": A,
        "b": b,
        "x0": x0,
        "x_refined": x_refined,
        "history": history,
        "decomp_error": decomp_error,
        "eps_init_res": eps_init_res,
        "eps_init_err": eps_init_err,
        "rel_init_err": rel_init_err,
        "cond_number": cond_number,
        "final_err": final_err,
        "final_res": final_res,
        "iterations_count": iterations_count,
    }

    html_content = get_html_report(data)
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML generated at: {html_file}")

    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    cmd = [
        edge_path,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_file}",
        f"file:///{html_file}",
    ]
    print("Executing command to print PDF...")
    subprocess.run(cmd, check=True)
    print(f"PDF generated successfully at: {pdf_file}")
    print(f"PDF size: {os.path.getsize(pdf_file)} bytes")


if __name__ == "__main__":
    run_pipeline_and_generate_report()
