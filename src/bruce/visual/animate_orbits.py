import bruce_c
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation




def animate_orbit(planets_params, t=np.linspace(-0.1, 0.1, 1000),
                  line_alpha=0.3, zoom_primary=False,
                  save_gif=False, gif_path='orbit.gif', fps=30):
    trajectories = []
    for planet_params in planets_params:
        trajectory = bruce_c.xyzz(
            t,
            planet_params['t_zero'],
            planet_params['period'],
            planet_params['incl'],
            planet_params['e'],
            planet_params['w'],
            1,
        )
        radius_1 = planet_params['radius_1']
        trajectory['x'] = trajectory['x'] / radius_1
        trajectory['y'] = trajectory['y'] / radius_1
        trajectory['z'] = trajectory['z'] / radius_1
        trajectory['zz'] = trajectory['zz'] * 0.1 / radius_1
        trajectories.append(trajectory)

    mosaic = '''
    AABB
    AABB
    CCCC
    CCCC'''
    fig, ax = plt.subplot_mosaic(mosaic=mosaic)
    for trajectory in trajectories:
        ax['A'].plot(trajectory['x'], trajectory['y'], 'k', zorder=1,
                     lw=1, alpha=line_alpha)
        ax['B'].plot(trajectory['x'], trajectory['z'], 'k', lw=1,
                     alpha=line_alpha)
        ax['C'].plot(t, trajectory['zz'], 'k', lw=1, alpha=line_alpha)
    if zoom_primary : ax['A'].set(xlim=(-1.5,1.5), ylim=(-1.5,1.5))
    star_xy = Circle((0, 0), 1, facecolor='gold', edgecolor='tab:orange', alpha=1.0,
                    zorder=2)
    star_xz = Circle((0, 0), 1, facecolor='gold', edgecolor='tab:orange', alpha=1.0,
                    zorder=0)
    ax['A'].add_patch(star_xy)
    ax['B'].add_patch(star_xz)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    planet_xy_front = []
    planet_xy_back = []
    planet_xz = []
    planet_zz = []
    legend_handles = []
    for index, planet_params in enumerate(planets_params):
        color = colors[index % len(colors)]
        label = planet_params.get('label', f'Planet {chr(ord("b") + index)}')
        front = Circle((0, 0), planet_params['k'], facecolor=color,
                       edgecolor='black', zorder=10)
        back = Circle((0, 0), planet_params['k'], facecolor=color,
                      edgecolor='black', zorder=1)
        xz = Circle((0, 0), planet_params['k'], facecolor=color,
                    edgecolor='black', zorder=10)
        ax['A'].add_patch(front)
        ax['A'].add_patch(back)
        ax['B'].add_patch(xz)
        zz, = ax['C'].plot([], [], 'o', color=color, zorder=10)
        planet_xy_front.append(front)
        planet_xy_back.append(back)
        planet_xz.append(xz)
        planet_zz.append(zz)
        legend_handles.append(Line2D([], [], marker='o', linestyle='None',
                         color=color, markeredgecolor='black',
                         label=label))
    time_text = ax['C'].text(0.02, 0.95, '', transform=ax['C'].transAxes,
                            va='top')

    ax['A'].set(xlabel='x', ylabel='y')
    ax['B'].set(xlabel='x', ylabel='z')
    ax['C'].set(xlabel='Time', ylabel='|z|')


    for name in ('A', 'B', 'C'):
        ax[name].grid(True, alpha=0.25)

    ax['A'].set_aspect('equal', adjustable='datalim')
    ax['B'].set_aspect('equal', adjustable='datalim')
    ax['A'].legend(handles=legend_handles)


    def init():
        for front, back, xz, zz in zip(planet_xy_front, planet_xy_back,
                                       planet_xz, planet_zz):
            front.set_visible(False)
            back.set_visible(False)
            xz.set_visible(False)
            zz.set_data([], [])
        time_text.set_text('')
        return (*planet_xy_front, *planet_xy_back, *planet_xz, *planet_zz,
                time_text)


    def update(frame):
        for index, trajectory in enumerate(trajectories):
            planet_xy_position = (trajectory['x'][frame], trajectory['y'][frame])
            planet_xz_position = (trajectory['x'][frame], trajectory['z'][frame])
            planet_xy_front[index].set_center(planet_xy_position)
            planet_xy_back[index].set_center(planet_xy_position)
            planet_xz[index].set_center(planet_xz_position)
            if trajectory['f'][frame] > 0:
                planet_xy_front[index].set_visible(True)
                planet_xy_back[index].set_visible(False)
            else:
                planet_xy_front[index].set_visible(False)
                planet_xy_back[index].set_visible(True)
            planet_xz[index].set_visible(True)
            planet_zz[index].set_data([t[frame]], [trajectory['zz'][frame]])
        time_text.set_text(f't = {t[frame]:.4f}')
        return (*planet_xy_front, *planet_xy_back, *planet_xz, *planet_zz,
                time_text)


    animation = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=20,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()

    if save_gif:
        animation.save(gif_path, writer='pillow', fps=fps)
        print(f'Saved animation to {gif_path}')

    plt.show()