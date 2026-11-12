"""Марковские цепи: аналитика, Монте-Карло, графики и GIF-анимация."""
from pathlib import Path
import argparse
import os
import sys
import subprocess
import numpy as np
import matplotlib
# Не блокируем интерактивный backend: Agg используем только без дисплея.
if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

P = np.array([[.70, .20, .10], [.15, .70, .15], [.10, .25, .65]])
INITIAL = np.array([1., 0., 0.])
NAMES = ['Состояние 1', 'Состояние 2', 'Состояние 3']


def stationary(p):
    a = p.T - np.eye(len(p))
    a[-1, :] = 1
    b = np.zeros(len(p)); b[-1] = 1
    return np.linalg.solve(a, b)


def simulate(p, n, steps, seed):
    rng = np.random.default_rng(seed)
    states = np.zeros(n, dtype=np.int64)
    freq = np.empty((steps + 1, len(p)))
    freq[0] = np.bincount(states, minlength=len(p)) / n
    cumulative = np.cumsum(p, axis=1)
    for t in range(1, steps + 1):
        u = rng.random(n)
        states = (u[:, None] > cumulative[states, :-1]).sum(axis=1)
        freq[t] = np.bincount(states, minlength=len(p)) / n
    return freq


def open_file(path):
    """Открыть сохранённый файл в системном приложении, если возможно."""
    path = Path(path).resolve()
    try:
        if sys.platform == 'win32':
            os.startfile(str(path))
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            subprocess.Popen(['xdg-open', str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            print(f'Нет графического окружения. Файл сохранён: {path}')
    except (OSError, FileNotFoundError) as exc:
        print(f'Не удалось открыть {path.name}: {exc}. Файл сохранён.')


def make_animation(out, times, theory, empirical, pi, fps=12):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
    x = np.arange(3)
    bars_theory = ax1.bar(x - .18, theory[0], .35, color=colors, alpha=.9, label='Теория')
    bars_mc = ax1.bar(x + .18, empirical[0], .35, color=colors, alpha=.4, hatch='//', label='Монте-Карло')
    ax1.set_xticks(x, NAMES)
    ax1.set_ylim(0, 1.04)
    ax1.set_ylabel('Вероятность')
    ax1.set_title('Распределение по состояниям')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=.2)
    lines_t, lines_m = [], []
    for j in range(3):
        (lt,) = ax2.plot([], [], color=colors[j], lw=2, label=f'Теория {j+1}')
        (lm,) = ax2.plot([], [], color=colors[j], lw=1.5, ls='--', alpha=.75, label=f'МК {j+1}')
        ax2.axhline(pi[j], color=colors[j], ls=':', lw=1, alpha=.65)
        lines_t.append(lt); lines_m.append(lm)
    ax2.set(xlim=(0, max(1, times[-1])), ylim=(0, 1.04), xlabel='Шаг', ylabel='Вероятность', title='Эволюция вероятностей')
    ax2.grid(alpha=.2)
    ax2.legend(fontsize=8, ncol=2)
    title = fig.suptitle('Шаг 0')
    def update(frame):
        for j in range(3):
            bars_theory[j].set_height(theory[frame, j])
            bars_mc[j].set_height(empirical[frame, j])
            lines_t[j].set_data(times[:frame+1], theory[:frame+1, j])
            lines_m[j].set_data(times[:frame+1], empirical[:frame+1, j])
        title.set_text(f'Марковская цепь — шаг {frame} из {len(times)-1}')
        return [*bars_theory, *bars_mc, *lines_t, *lines_m, title]
    # 0-й и последний кадры включены, анимация не зависит от окна plt.show().
    animation = FuncAnimation(fig, update, frames=len(times), interval=1000/fps, blit=False, repeat=True)
    gif = out / 'markov_animation.gif'
    animation.save(str(gif), writer=PillowWriter(fps=fps), dpi=95)
    plt.close(fig)
    return gif


def experiment(out, steps=70, trajectories=12000, seed=42, fps=12, show=True, animate=True):
    if steps < 1 or trajectories < 1 or fps < 1:
        raise ValueError('steps, trajectories и fps должны быть положительными')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if not np.all(P >= 0) or not np.allclose(P.sum(axis=1), 1):
        raise ValueError('Некорректная матрица переходов')
    pi = stationary(P)
    theory = np.empty((steps + 1, 3)); theory[0] = INITIAL
    for t in range(steps):
        theory[t+1] = theory[t] @ P
    empirical = simulate(P, trajectories, steps, seed)
    times = np.arange(steps + 1)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for j in range(3):
        axs[0].plot(times, theory[:, j], label=f'Теория: {j+1}')
        axs[0].plot(times, empirical[:, j], ls='--', alpha=.75, label=f'Монте-Карло: {j+1}')
    axs[0].set(xlabel='Шаг', ylabel='Вероятность', title='Распределение состояний')
    axs[0].legend(fontsize=7, ncol=2)
    axs[1].plot(times, np.max(np.abs(theory-pi), axis=1), label='Теория')
    axs[1].plot(times, np.max(np.abs(empirical-pi), axis=1), label='Монте-Карло', alpha=.8)
    axs[1].set(xlabel='Шаг', ylabel='Максимальное отклонение', title='Сходимость к стационарному распределению')
    axs[1].legend()
    for ax in axs: ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out/'dynamics.png', dpi=180); plt.close(fig)
    sample_sizes = [100, 500, 2000, 12000, 50000]
    reps = 30; errors = []; stds = []
    for n in sample_sizes:
        err = [np.max(np.abs(simulate(P, n, steps, seed+1000+r)[-1]-theory[-1])) for r in range(reps)]
        errors.append(np.mean(err)); stds.append(np.std(err, ddof=1))
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.errorbar(sample_sizes, errors, yerr=stds, fmt='o-', capsize=3)
    ax.set(xscale='log', xlabel='Количество траекторий', ylabel='Средняя максимальная ошибка', title='Точность Монте-Карло (30 повторов)')
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out/'accuracy.png', dpi=180); plt.close(fig)
    np.savetxt(out/'dynamics.csv', np.column_stack([times, theory, empirical]), delimiter=',', header='step,theory_1,theory_2,theory_3,mc_1,mc_2,mc_3', comments='')
    np.savetxt(out/'accuracy.csv', np.column_stack([sample_sizes, errors, stds]), delimiter=',', header='trajectories,mean_max_error,std_max_error', comments='')
    gif = make_animation(out, times, theory, empirical, pi, fps) if animate else None
    print('Матрица переходов:\n', P)
    print('Стационарное распределение:', pi)
    print('Теория, последний шаг:', theory[-1])
    print('Монте-Карло, последний шаг:', empirical[-1])
    print('Максимальная ошибка:', np.max(np.abs(theory[-1]-empirical[-1])))
    print('Файлы результатов:', out.resolve())
    if show:
        for filename in ['dynamics.png', 'accuracy.png']:
            open_file(out / filename)
        if gif is not None:
            open_file(gif)
    return pi, theory, empirical, sample_sizes, errors, stds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=70)
    parser.add_argument('--trajectories', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results')
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--no-show', action='store_true', help='Не открывать изображения и GIF автоматически')
    parser.add_argument('--no-animation', action='store_true', help='Не создавать GIF')
    args = parser.parse_args()
    experiment(args.output, args.steps, args.trajectories, args.seed, args.fps, not args.no_show, not args.no_animation)
