from matplotlib.pyplot import *


def vectorField(field, step=2, **kwargs):
    """
    Plot given vector field as quiver (arrow) plot.

    :param field: numpy array with 3 dimensions, first two are spatial x,y coordinate, the last
                  coordinate should have shape 2 and stores the 2 velocity components
    :param step: plots only every steps's cell
    :param kwargs: keyword arguments passed to :func:`matplotlib.pyplot.quiver`
    """
    veln = field.swapaxes(0, 1)
    res = quiver(veln[::step, ::step, 0], veln[::step, ::step, 1], **kwargs)
    axis('equal')
    return res


def vectorFieldMagnitude(field, **kwargs):
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
    return scalarField(norm, **kwargs)


def scalarField(field, **kwargs):
    """
    Plots field values as colormap

    :param field: two dimensional numpy array
    :param kwargs: keyword arguments passed to :func:`matplotlib.pyplot.imshow`
    """
    import numpy as np
    field = np.swapaxes(field, 0, 1)
    res = imshow(field, origin='lower', **kwargs)
    axis('equal')
    return res


def scalarFieldAlphaValue(field, color, clip=False, **kwargs):
    import numpy as np
    import matplotlib
    field = np.swapaxes(field, 0, 1)
    color = matplotlib.colors.to_rgba(color)

    fieldToPlot = np.empty(field.shape + (4,))
    for i in range(3):
        fieldToPlot[:, :, i] = color[i]

    if clip:
        normalizedField = field.copy()
        normalizedField[normalizedField<0] = 0
        normalizedField[normalizedField>1] = 1
    else:
        min, max = np.min(field), np.max(field)
        normalizedField = (field - min) / (max - min)
    fieldToPlot[:, :, 3] = normalizedField

    res = imshow(fieldToPlot, origin='lower', **kwargs)
    axis('equal')
    return res


def scalarFieldContour(field, **kwargs):
    field = np.swapaxes(field, 0, 1)
    res = contour(field, **kwargs)
    axis('equal')
    return res


def multipleScalarFields(field, **kwargs):
    subPlots = field.shape[-1]
    for i in range(subPlots):
        subplot(1, subPlots, i + 1)
        title(str(i))
        scalarField(field[..., i])
        colorbar()


# ------------------------------------------- Animations ---------------------------------------------------------------


def vectorFieldAnimation(runFunction, step=2, rescale=True, plotSetupFunction=lambda: None,
                         plotUpdateFunction=lambda: None, interval=30, frames=180, **kwargs):
    import matplotlib.animation as animation
    from numpy.linalg import norm

    fig = gcf()
    im = None
    field = runFunction()
    if rescale:
        maxNorm = np.max(norm(field, axis=2, ord=2))
        field = field / maxNorm
        if 'scale' not in kwargs:
            kwargs['scale'] = 1.0

    quiverPlot = vectorField(field, step=step, **kwargs)
    plotSetupFunction()

    def updatefig(*args):
        f = runFunction()
        f = np.swapaxes(f, 0, 1)
        if rescale:
            maxNorm = np.max(norm(f, axis=2, ord=2))
            f = f / maxNorm
        u, v = f[::step, ::step, 0], f[::step, ::step, 1]
        quiverPlot.set_UVC(u, v)
        plotUpdateFunction()
        return im,

    return animation.FuncAnimation(fig, updatefig, interval=interval, frames=frames)


def vectorFieldMagnitudeAnimation(runFunction, plotSetupFunction=lambda: None,
                                  plotUpdateFunction=lambda: None, interval=30, frames=180, **kwargs):
    import matplotlib.animation as animation
    from numpy.linalg import norm

    fig = gcf()
    im = None
    field = runFunction()
    im = vectorFieldMagnitude(field, **kwargs)
    plotSetupFunction()

    def updatefig(*args):
        f = runFunction()
        normed = norm(f, axis=2, ord=2)
        if hasattr(f, 'mask'):
            normed = np.ma.masked_array(normed, mask=f.mask[:, :, 0])
        normed = np.swapaxes(normed, 0, 1)
        im.set_array(normed)
        plotUpdateFunction()
        return im,

    return animation.FuncAnimation(fig, updatefig, interval=interval, frames=frames)