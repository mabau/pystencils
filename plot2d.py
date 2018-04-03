from matplotlib.pyplot import *


def vector_field(field, step=2, **kwargs):
    """
    Plot given vector field as quiver (arrow) plot.

    :param field: numpy array with 3 dimensions, first two are spatial x,y coordinate, the last
                  coordinate should have shape 2 and stores the 2 velocity components
    :param step: plots only every steps's cell
    :param kwargs: keyword arguments passed to :func:`matplotlib.pyplot.quiver`
    """
    vel_n = field.swapaxes(0, 1)
    res = quiver(vel_n[::step, ::step, 0], vel_n[::step, ::step, 1], **kwargs)
    axis('equal')
    return res


def vector_field_magnitude(field, **kwargs):
    """
    Plots the magnitude of a vector field as colormap
    :param field: numpy array with 3 dimensions, first two are spatial x,y coordinate, the last
                  coordinate should have shape 2 and stores the 2 velocity components
    :param kwargs: keyword arguments passed to :func:`matplotlib.pyplot.imshow`
    """
    from numpy.linalg import norm
    norm = norm(field, axis=2, ord=2)
    if hasattr(field, 'mask'):
        norm = np.ma.masked_array(norm, mask=field.mask[:, :, 0])
    return scalar_field(norm, **kwargs)


def scalar_field(field, **kwargs):
    """
    Plots field values as colormap

    :param field: two dimensional numpy array
    :param kwargs: keyword arguments passed to :func:`matplotlib.pyplot.imshow`
    """
    import numpy
    field = numpy.swapaxes(field, 0, 1)
    res = imshow(field, origin='lower', **kwargs)
    axis('equal')
    return res


def scalar_field_alpha_value(field, color, clip=False, **kwargs):
    import numpy
    import matplotlib
    field = numpy.swapaxes(field, 0, 1)
    color = matplotlib.colors.to_rgba(color)

    field_to_plot = numpy.empty(field.shape + (4,))
    for i in range(3):
        field_to_plot[:, :, i] = color[i]

    if clip:
        normalized_field = field.copy()
        normalized_field[normalized_field < 0] = 0
        normalized_field[normalized_field > 1] = 1
    else:
        minimum, maximum = numpy.min(field), numpy.max(field)
        normalized_field = (field - minimum) / (maximum - minimum)
    field_to_plot[:, :, 3] = normalized_field

    res = imshow(field_to_plot, origin='lower', **kwargs)
    axis('equal')
    return res


def scalar_field_contour(field, **kwargs):
    field = np.swapaxes(field, 0, 1)
    res = contour(field, **kwargs)
    axis('equal')
    return res


def multiple_scalar_fields(field, **_):
    sub_plots = field.shape[-1]
    for i in range(sub_plots):
        subplot(1, sub_plots, i + 1)
        title(str(i))
        scalar_field(field[..., i])
        colorbar()


def sympy_function(f, var, bounds, **kwargs):
    import sympy as sp
    x_arr = np.linspace(bounds[0], bounds[1], 101)
    y_arr = sp.lambdify(var, f)(x_arr)
    plot(x_arr, y_arr, **kwargs)

# ------------------------------------------- Animations ---------------------------------------------------------------


def vector_field_animation(run_function, step=2, rescale=True, plot_setup_function=lambda: None,
                           plot_update_function=lambda: None, interval=30, frames=180, **kwargs):
    import matplotlib.animation as animation
    from numpy.linalg import norm

    fig = gcf()
    im = None
    field = run_function()
    if rescale:
        max_norm = np.max(norm(field, axis=2, ord=2))
        field = field / max_norm
        if 'scale' not in kwargs:
            kwargs['scale'] = 1.0

    quiver_plot = vector_field(field, step=step, **kwargs)
    plot_setup_function()

    def update_figure(*_):
        f = run_function()
        f = np.swapaxes(f, 0, 1)
        if rescale:
            f = f / np.max(norm(f, axis=2, ord=2))
        u, v = f[::step, ::step, 0], f[::step, ::step, 1]
        quiver_plot.set_UVC(u, v)
        plot_update_function()
        return im,

    return animation.FuncAnimation(fig, update_figure, interval=interval, frames=frames)


def vector_field_magnitude_animation(run_function, plot_setup_function=lambda: None,
                                     plot_update_function=lambda: None, interval=30, frames=180, **kwargs):
    import matplotlib.animation as animation
    from numpy.linalg import norm

    fig = gcf()
    im = None
    field = run_function()
    im = vector_field_magnitude(field, **kwargs)
    plot_setup_function()

    def update_figure(*_):
        f = run_function()
        normed = norm(f, axis=2, ord=2)
        if hasattr(f, 'mask'):
            normed = np.ma.masked_array(normed, mask=f.mask[:, :, 0])
        normed = np.swapaxes(normed, 0, 1)
        im.set_array(normed)
        plot_update_function()
        return im,

    return animation.FuncAnimation(fig, update_figure, interval=interval, frames=frames)
