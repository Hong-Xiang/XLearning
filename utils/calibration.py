import numpy as np



def plot_event(evt_map, inc_pos=None, sci_poss=None, cmap='jet'):
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt
    if not isinstance(evt_map, np.ndarray):
        raise TypeError("Invalid evt_map type.")
    if evt_map.shape == (100, ):
        shape = [10, 10]
        evt_map = np.reshape(evt_map, shape)
    plt.imshow(evt_map, cmap=cmap)
    plt.xlabel('x position (mm)')
    plt.ylabel('y position (mm)')
    plt.colorbar()
    ax = plt.gca()
    x = (inc_pos[0] + 30.0) / 6.0 - 0.5
    y = (inc_pos[1] + 30.0) / 6.0 - 0.5
    if inc_pos.size == 3:
        inci_point = Circle(
            (x, y), radius=((-inc_pos[2]+12.5)/5) / 5, linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(inci_point)
    inci_circle = Circle((x, y), radius= 0.2 , linewidth=1,
                         edgecolor='k', facecolor='k')
    ax.add_patch(inci_circle)
    ax.set_xticks(np.linspace(-0.5, 9.5, 11))
    ax.set_yticks(np.linspace(-0.5, 9.5, 11))
    ax.set_xticklabels(range(-30, 30, 6))
    ax.set_yticklabels(range(-30, 30, 6))

