"""
This module extends the pyplot module with functions to show scalar and vector fields in the usual
simulation coordinate system (y-axis goes up), instead of the "image coordinate system" (y axis goes down) that
matplotlib normally uses.
"""
import warnings
from itertools import cycle

from matplotlib.pyplot import *


def vector_field(array, step=2, **kwargs):
    """Plots given vector field as quiver (arrow) plot.

    Args:
        array: numpy array with 3 dimensions, first two are spatial x,y coordinate, the last
               coordinate should have shape 2 and stores the 2 velocity components
        step: plots only every steps's cell, increase the step for high resolution arrays
        kwargs: keyword arguments passed to :func:`matplotlib.pyplot.quiver`

    Returns:
        quiver plot object
    """
    assert len(array.shape) == 3, "Wrong shape of array - did you forget to slice your 3D domain first?"
    assert array.shape[2] == 2, "Last array dimension is expected to store 2D vectors"
    vel_n = array.swapaxes(0, 1)
    res = quiver(vel_n[::step, ::step, 0], vel_n[::step, ::step, 1], **kwargs)
    axis('equal')
    return res


def vector_field_magnitude(array, **kwargs):
    """Plots the magnitude of a vector field as colormap.

    Args:
        array: numpy array with 3 dimensions, first two are spatial x,y coordinate, the last
               coordinate should have shape 2 and stores the 2 velocity components
        kwargs: keyword arguments passed to :func:`matplotlib.pyplot.imshow`

    Returns:
        imshow object
    """
    assert len(array.shape) == 3, "Wrong shape of array - did you forget to slice your 3D domain first?"
    assert array.shape[2] in (2, 3), "Wrong size of the last coordinate. Has to be a 2D or 3D vector field."
    from numpy.linalg import norm
    norm = norm(array, axis=2, ord=2)
    if hasattr(array, 'mask'):
        norm = np.ma.masked_array(norm, mask=array.mask[:, :, 0])
    return scalar_field(norm, **kwargs)


def scalar_field(array, **kwargs):
    """Plots field values as colormap.

    Works just as imshow, but uses coordinate system where second coordinate (y) points upwards.

    Args:
        array: two dimensional numpy array
        kwargs: keyword arguments passed to :func:`matplotlib.pyplot.imshow`

    Returns:
        imshow object
    """
    import numpy
    array = numpy.swapaxes(array, 0, 1)
    res = imshow(array, origin='lower', **kwargs)
    axis('equal')
    return res


def scalar_field_surface(array, **kwargs):
    """Plots scalar field as 3D surface

    Args:
        array: the two dimensional numpy array to plot
        kwargs: keyword arguments passed to :func:`mpl_toolkits.mplot3d.Axes3D.plot_surface`
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    fig = gcf()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(np.arange(array.shape[0]), np.arange(array.shape[1]), indexing='ij')
    kwargs.setdefault('rstride', 2)
    kwargs.setdefault('cstride', 2)
    kwargs.setdefault('color', 'b')
    kwargs.setdefault('cmap', cm.coolwarm)
    return ax.plot_surface(x, y, array, **kwargs)


def scalar_field_alpha_value(array, color, clip=False, **kwargs):
    """Plots an image with same color everywhere, using the array values as transparency.

    Array is supposed to have values between 0 and 1 (if this is not the case it is normalized).
    An image is plotted that has the same color everywhere, the passed array determines the transparency.
    Regions where the array is 1  are fully opaque, areas with 0 are fully transparent.

    Args:
        array: 2D array with alpha values
        color: fill color
        clip: if True, all values in the array larger than 1 are set to 1, all values smaller than 0 are set to zero
              if False, the array is linearly scaled to the [0, 1] interval
        **kwargs: arguments passed to imshow

    Returns:
        imshow object
    """
    import numpy
    import matplotlib
    assert len(array.shape) == 2, "Wrong shape of array - did you forget to slice your 3D domain first?"
    array = numpy.swapaxes(array, 0, 1)

    if clip:
        normalized_field = array.copy()
        normalized_field[normalized_field < 0] = 0
        normalized_field[normalized_field > 1] = 1
    else:
        minimum, maximum = numpy.min(array), numpy.max(array)
        normalized_field = (array - minimum) / (maximum - minimum)

    color = matplotlib.colors.to_rgba(color)
    field_to_plot = numpy.empty(array.shape + (4,))
    # set the complete array to the color
    for i in range(3):
        field_to_plot[:, :, i] = color[i]
    # only the alpha channel varies using the array values
    field_to_plot[:, :, 3] = normalized_field

    res = imshow(field_to_plot, origin='lower', **kwargs)
    axis('equal')
    return res


def scalar_field_contour(array, **kwargs):
    """Small wrapper around contour to transform the coordinate system.

    For details see  :func:`matplotlib.pyplot.imshow`
    """
    array = np.swapaxes(array, 0, 1)
    res = contour(array, **kwargs)
    axis('equal')
    return res


def multiple_scalar_fields(array, **kwargs):
    """Plots a 3D array by slicing the last dimension and creates on plot for each entry of the last dimension.

    Args:
        array: 3D array to plot.
        **kwargs: passed along to imshow
    """
    assert len(array.shape) == 3
    sub_plots = array.shape[-1]
    for i in range(sub_plots):
        subplot(1, sub_plots, i + 1)
        title(str(i))
        scalar_field(array[..., i], **kwargs)
        colorbar()


def phase_plot(phase_field: np.ndarray, linewidth=1.0, clip=True) -> None:
    """Plots a phase field array using the phase variables as alpha channel.

    Args:
        phase_field: array with len(shape) == 3, first two dimensions are spatial, the last one indexes the phase
                     components.
        linewidth: line width of the 0.5 contour lines that are drawn over the alpha blended phase images
        clip: see scalar_field_alpha_value function
    """
    color_cycle = cycle(['#fe0002', '#00fe00', '#0000ff', '#ffa800', '#f600ff'])

    assert len(phase_field.shape) == 3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(phase_field.shape[-1]):
            scalar_field_alpha_value(phase_field[..., i], next(color_cycle), clip=clip, interpolation='bilinear')
        if linewidth:
            for i in range(phase_field.shape[-1]):
                scalar_field_contour(phase_field[..., i], levels=[0.5], colors='k', linewidths=[linewidth])


def sympy_function(expr, x_values=None, **kwargs):
    """Plots the graph of a sympy term that depends on one symbol only.

    Args:
        expr: sympy term that depends on one symbol only, which is plotted on the x axis
        x_values: describes sampling of x axis. Possible values are:
                     * tuple of (start, stop) or (start, stop, nr_of_steps)
                     * None, then start=0, stop=1, nr_of_steps=100
                     * 1D numpy array with x values
        **kwargs: passed on to :func:`matplotlib.pyplot.plot`

    Returns:
        plot object
    """
    import sympy as sp
    if x_values is None:
        x_arr = np.linspace(0, 1, 100)
    elif type(x_values) is tuple:
        x_arr = np.linspace(*x_values)
    elif isinstance(x_values, np.ndarray):
        assert len(x_values.shape) == 1
        x_arr = x_values
    else:
        raise ValueError("Invalid value for parameter x_values")
    symbols = expr.atoms(sp.Symbol)
    assert len(symbols) == 1, "Sympy expression may only depend on one variable only. Depends on " + str(symbols)
    y_arr = sp.lambdify(symbols.pop(), expr)(x_arr)
    return plot(x_arr, y_arr, **kwargs)


# ------------------------------------------- Animations ---------------------------------------------------------------

def __scale_array(arr):
    from numpy.linalg import norm
    norm_arr = norm(arr, axis=2, ord=2)
    if isinstance(arr, np.ma.MaskedArray):
        norm_arr = np.ma.masked_array(norm_arr, arr.mask[..., 0])
    return arr / norm_arr.max()


def vector_field_animation(run_function, step=2, rescale=True, plot_setup_function=lambda *_: None,
                           plot_update_function=lambda *_: None, interval=200, frames=180, **kwargs):
    """Creates a matplotlib animation of a vector field using a quiver plot.

    Args:
        run_function: callable without arguments, returning a 2D vector field i.e. numpy array with len(shape)==3
        step: see documentation of vector_field function
        rescale: if True, the length of the arrows is rescaled in every time step
        plot_setup_function: optional callable with the quiver object as argument,
                             that can be used to set up the plot (title, legend,..)
        plot_update_function: optional callable with the quiver object as argument
                              that is called of the quiver object was updated
        interval: delay between frames in milliseconds (see matplotlib.FuncAnimation)
        frames: how many frames should be generated, see matplotlib.FuncAnimation
        **kwargs: passed to quiver plot

    Returns:
        matplotlib animation object
    """
    import matplotlib.animation as animation

    fig = gcf()
    im = None
    field = run_function()
    if rescale:
        field = __scale_array(field)
        kwargs.setdefault('scale', 0.6)
        kwargs.setdefault('angles', 'xy')
        kwargs.setdefault('scale_units', 'xy')

    quiver_plot = vector_field(field, step=step, **kwargs)
    plot_setup_function(quiver_plot)

    def update_figure(*_):
        f = run_function()
        f = np.swapaxes(f, 0, 1)
        if rescale:
            f = __scale_array(f)
        u, v = f[::step, ::step, 0], f[::step, ::step, 1]
        quiver_plot.set_UVC(u, v)
        plot_update_function(quiver_plot)
        return im,

    return animation.FuncAnimation(fig, update_figure, interval=interval, frames=frames)


def vector_field_magnitude_animation(run_function, plot_setup_function=lambda *_: None, rescale=False,
                                     plot_update_function=lambda *_: None, interval=30, frames=180, **kwargs):
    """Animation of a vector field, showing the magnitude as colormap.

    For arguments, see vector_field_animation
    """
    import matplotlib.animation as animation
    from numpy.linalg import norm

    fig = gcf()
    im = None
    field = run_function()
    if rescale:
        field = __scale_array(field)
    im = vector_field_magnitude(field, **kwargs)
    plot_setup_function(im)

    def update_figure(*_):
        f = run_function()
        if rescale:
            f = __scale_array(f)
        normed = norm(f, axis=2, ord=2)
        if hasattr(f, 'mask'):
            normed = np.ma.masked_array(normed, mask=f.mask[:, :, 0])
        normed = np.swapaxes(normed, 0, 1)
        im.set_array(normed)
        plot_update_function(im)
        return im,

    return animation.FuncAnimation(fig, update_figure, interval=interval, frames=frames)


def scalar_field_animation(run_function, plot_setup_function=lambda *_: None, rescale=True,
                           plot_update_function=lambda *_: None, interval=30, frames=180, **kwargs):
    """Animation of scalar field as colored image, see `scalar_field`."""
    import matplotlib.animation as animation

    fig = gcf()
    im = None
    field = run_function()
    if rescale:
        f_min, f_max = np.min(field), np.max(field)
        field = (field - f_min) / (f_max - f_min)
        im = scalar_field(field, vmin=0.0, vmax=1.0, **kwargs)
    else:
        im = scalar_field(field, **kwargs)
    plot_setup_function(im)

    def update_figure(*_):
        f = run_function()
        if rescale:
            f_min, f_max = np.min(f), np.max(f)
            f = (f - f_min) / (f_max - f_min)
        if hasattr(f, 'mask'):
            f = np.ma.masked_array(f, mask=f.mask[:, :])
        f = np.swapaxes(f, 0, 1)
        im.set_array(f)
        plot_update_function(im)
        return im,

    return animation.FuncAnimation(fig, update_figure, interval=interval, frames=frames)


def surface_plot_animation(run_function, frames=90, interval=30, zlim=None,  **kwargs):
    """Animation of scalar field as 3D plot."""
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    from matplotlib import cm
    fig = gcf()
    ax = fig.add_subplot(111, projection='3d')
    data = run_function()
    x, y = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing='ij')
    kwargs.setdefault('rstride', 2)
    kwargs.setdefault('cstride', 2)
    kwargs.setdefault('color', 'b')
    kwargs.setdefault('cmap', cm.coolwarm)
    ax.plot_surface(x, y, data, **kwargs)
    if zlim is not None:
        ax.set_zlim(*zlim)

    def update_figure(*_):
        d = run_function()
        ax.clear()
        plot = ax.plot_surface(x, y, d, **kwargs)
        if zlim is not None:
            ax.set_zlim(*zlim)
        return plot,

    return animation.FuncAnimation(fig, update_figure, interval=interval, frames=frames, blit=False)
