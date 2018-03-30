import pystencils.plot2d as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tempfile import NamedTemporaryFile
import base64
import sympy as sp

__all__ = ['log_progress', 'makeImshowAnimation', 'makeSurfacePlotAnimation',
           'disp', 'setDisplayMode']


def log_progress(sequence, every=None, size=None, name='Items'):
    """Copied from https://github.com/alexanderkuk/log-progress"""
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


VIDEO_TAG = """<video controls width="80%">
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""


def __anim_to_html(anim, fps):
    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264', '-pix_fmt',
                                                   'yuv420p', '-profile:v', 'baseline', '-level', '3.0'])
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video).decode('ascii')

    return VIDEO_TAG.format(anim._encoded_video)


def makeImshowAnimation(grid, gridUpdateFunction, frames=90, **kwargs):
    from functools import partial
    fig = plt.figure()
    im = plt.imshow(grid, interpolation='none')

    def updatefig(*args, **kwargs):
        image = kwargs['image']
        image = gridUpdateFunction(image)
        im.set_array(image)
        return im,

    return animation.FuncAnimation(fig, partial(updatefig, image=grid), frames=frames)


def makeSurfacePlotAnimation(runFunction, frames=90, interval=30):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, data = runFunction(1)
    ax.plot_surface(X, Y, data, rstride=2, cstride=2, color='b', cmap=cm.coolwarm,)
    ax.set_zlim(-1.0, 1.0)

    def updatefig(*args):
        X, Y, data = runFunction(1)
        ax.clear()
        plot = ax.plot_surface(X, Y, data, rstride=2, cstride=2, color='b', cmap=cm.coolwarm,)
        ax.set_zlim(-1.0, 1.0)
        return plot,

    return animation.FuncAnimation(fig, updatefig, interval=interval, frames=frames, blit=False)


# -------   Version 1: Embed the animation as HTML5 video --------- ----------------------------------

def displayAsHtmlVideo(anim, fps=30, show=True, **kwargs):
    try:
        plt.close(anim._fig)
        res = __anim_to_html(anim, fps)
        if show:
            return HTML(res)
        else:
            return HTML("")
    except KeyboardInterrupt:
        pass


# -------   Version 2: Animation is shown in extra matplotlib window ----------------------------------


def displayInExtraWindow(animation, *args, **kwargs):
    fig = plt.gcf()
    try:
      fig.canvas.manager.window.raise_()
    except Exception:
      pass
    plt.show()


# -------   Version 3: Animation is shown in images that are updated directly in website --------------

def displayAsHtmlImage(animation, show=True, iterations=10000,  *args, **kwargs):
    from IPython import display

    try:
        if show:
            fig = plt.gcf()
        if show:
            animation._init_draw()
        for i in range(iterations):
            if show:
                display.display(fig)
            animation._step()
            if show:
                display.clear_output(wait=True)
    except KeyboardInterrupt:
        display.clear_output(wait=False)


# Dispatcher

animation_display_mode = 'imageupdate'
display_animation_func = None


def disp(*args, **kwargs):
    from IPython import get_ipython
    ipython = get_ipython()
    if not ipython:
        return

    if not display_animation_func:
        raise Exception("Call set_display_mode first")
    return display_animation_func(*args, **kwargs)


def setDisplayMode(mode):
    from IPython import get_ipython
    ipython = get_ipython()
    if not ipython:
        return
    global animation_display_mode
    global display_animation_func
    animation_display_mode = mode
    if animation_display_mode == 'video':
        ipython.magic("matplotlib inline")
        display_animation_func = displayAsHtmlVideo
    elif animation_display_mode == 'window':
        ipython.magic("matplotlib qt")
        display_animation_func = displayInExtraWindow
    elif animation_display_mode == 'imageupdate':
        ipython.magic("matplotlib inline")
        display_animation_func = displayAsHtmlImage
    else:
        raise Exception("Unknown mode. Available modes 'imageupdate', 'video' and 'window' ")


def activateIPython():
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython:
        setDisplayMode('imageupdate')
        ipython.magic("config InlineBackend.rc = { }")
        ipython.magic("matplotlib inline")
        plt.rc('figure', figsize=(16, 6))
        sp.init_printing()

activateIPython()
