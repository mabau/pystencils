

def toDot(expr, graphStyle={}):
    """Show a sympy or pystencils AST as dot graph"""
    from pystencils.ast import Node
    import graphviz
    if isinstance(expr, Node):
        from pystencils.backends.dot import dotprint
        return graphviz.Source(dotprint(expr, short=True, graph_attr=graphStyle))
    else:
        from sympy.printing.dot import dotprint
        return graphviz.Source(dotprint(expr, graph_attr=graphStyle))


def highlightCpp(code):
    """Highlight the given C/C++ source code with Pygments"""
    from IPython.display import HTML, display
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import CppLexer

    display(HTML("""
            <style>
            {pygments_css}
            </style>
            """.format(pygments_css=HtmlFormatter().get_style_defs('.highlight'))))
    return HTML(highlight(code, CppLexer(), HtmlFormatter()))


# ----------------- Embedding of animations as videos in IPython notebooks ---------------------------------------------


# -------   Version 1: Animation is embedded as an HTML5 Video tag  ---------------------------------------

VIDEO_TAG = """<video controls width="100%">
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""


def __anim_to_html(anim, fps):
    from tempfile import NamedTemporaryFile
    import base64

    if not hasattr(anim, '_encoded_video'):
        with NamedTemporaryFile(suffix='.mp4') as f:
            anim.save(f.name, fps=fps, extra_args=['-vcodec', 'libx264', '-pix_fmt',
                                                   'yuv420p', '-profile:v', 'baseline', '-level', '3.0'])
            video = open(f.name, "rb").read()
        anim._encoded_video = base64.b64encode(video).decode('ascii')

    return VIDEO_TAG.format(anim._encoded_video)


def disp_as_video(anim, fps=30, show=True, **kwargs):
    import matplotlib.pyplot as plt
    from IPython.display import HTML

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


def disp_extra_window(animation, *args,**kwargs):
    import matplotlib.pyplot as plt

    fig = plt.gcf()
    try:
      fig.canvas.manager.window.raise_()
    except Exception:
      pass
    plt.show()


# -------   Version 3: Animation is shown in images that are updated directly in website --------------

def disp_image_update(animation, iterations=10000, *args, **kwargs):
    from IPython import display
    import matplotlib.pyplot as plt

    try:
        fig = plt.gcf()
        animation._init_draw()
        for i in range(iterations):
            display.display(fig)
            animation._step()
            display.clear_output(wait=True)
    except KeyboardInterrupt:
        pass


# Dispatcher

animation_display_mode = 'imageupdate'
display_animation_func = None


def disp(*args, **kwargs):
    if not display_animation_func:
        raise Exception("Call set_display_mode first")
    return display_animation_func(*args, **kwargs)


def set_display_mode(mode):
    from IPython import get_ipython
    ipython = get_ipython()
    global animation_display_mode
    global display_animation_func
    animation_display_mode = mode
    if animation_display_mode == 'video':
        ipython.magic("matplotlib inline")
        display_animation_func = disp_as_video
    elif animation_display_mode == 'window':
        ipython.magic("matplotlib qt")
        display_animation_func = disp_extra_window
    elif animation_display_mode == 'imageupdate':
        ipython.magic("matplotlib inline")
        display_animation_func = disp_image_update
    else:
        raise Exception("Unknown mode. Available modes 'imageupdate', 'video' and 'window' ")


set_display_mode('video')


# --------------------- Convenience functions --------------------------------------------------------------------------


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
