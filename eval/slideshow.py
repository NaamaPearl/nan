import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def slider_show(data, norm_by=None, title=''):
    if norm_by is None:
        vmin = data.min()
        vmax = data.max()
        norm_by = (vmin, vmax)
    N = data.shape[0]

    # Define initial parameters
    init_frequency = 0

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    image = plt.imshow(data[0], vmin=norm_by[0], vmax=norm_by[1])
    plt.title(title)
    plt.colorbar()

    axcolor = 'lightgoldenrodyellow'
    ax.margins(x=0)

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the frames.
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    frames_slider = Slider(
        ax=axfreq,
        label='frame ',
        valmin=0,
        valmax=N-1,
        valstep=1,
        valinit=init_frequency,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        image.set_data(data[int(frames_slider.val)])
        fig.canvas.draw_idle()

    # register the update function with each slider
    frames_slider.on_changed(update)

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        frames_slider.reset()
    button.on_clicked(reset)

    plt.show()
    return fig, frames_slider


def slider_show_rgb_ray(w_rgb, rgb_in, show=True):
    N = w_rgb.shape[0]

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots(ncols=2, figsize=(3, 3))
    ax[0].set_xticks([])
    ax[0].set_ylabel('source views')
    ax[0].set_title(r'$\mathregular{RGB}_{\mathregular{weights}}$')
    w_image = ax[0].imshow(w_rgb[0], vmin=0, vmax=1)
    fig.colorbar(w_image, ax=ax[0])
    ax[1].set_xticks([])
    ax[1].set_title(r'$\mathregular{RGB}_{\mathregular{in}}$')
    rgb_image = ax[1].imshow(rgb_in[0], vmin=0, vmax=1)

    plt.subplots_adjust(top=0.91,
                        bottom=0.11,
                        left=0.0,
                        right=1.0,
                        hspace=0.305,
                        wspace=0.2)

    axcolor = 'lightgoldenrodyellow'
    # ax.margins(x=0)

    # adjust the main plot to make room for the sliders
    # plt.subplots_adjust(left=0.25, bottom=0.25)

    # Make a horizontal slider to control the frames.
    axfreq = plt.axes([0.23, 0.05, 0.65, 0.03], facecolor=axcolor)
    frames_slider = Slider(
        ax=axfreq,
        label='sample ',
        valmin=0,
        valmax=N-1,
        valstep=1,
        valinit=0,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        w_image.set_data(w_rgb[int(val)])
        rgb_image.set_data(rgb_in[int(val)])
        fig.canvas.draw_idle()

    # register the update function with each slider
    frames_slider.on_changed(update)

    if show:
        plt.show()
    return fig, frames_slider


if __name__ == '__main__':
    rgb_in = np.arange(128*10*3).reshape((128, 10, 1, 3))
    w_rgb  = np.arange(128*10*1).reshape((128, 10, 1, 1))
    rgb_in = rgb_in / rgb_in.max()
    slider_show_rgb_ray(w_rgb, rgb_in)

    N = 100

    x_ = np.linspace(0, 10, 200)
    y_ = np.linspace(0, 10, 200)
    z_ = np.linspace(0, 10, N)

    x, y, z = np.meshgrid(x_, y_, z_, indexing='xy')

    data = np.sin(x + z) * np.cos(y)
    slider_show(data)

