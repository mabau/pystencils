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


def multipleScalarFields(field, **kwargs):
    subPlots = field.shape[-1]
    for i in range(subPlots):
        subplot(1, subPlots, i + 1)
        title(str(i))
        scalarField(field[..., i])
        colorbar()


def plotBoundaryHandling(boundaryHandling, boundaryNameToColor=None):
    """
    Shows boundary cells

    :param boundaryHandling: instance of :class:`lbmpy.boundaries.BoundaryHandling`
    :param boundaryNameToColor: optional dictionary mapping boundary names to colors
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if len(boundaryHandling.flagField.shape) != 2:
        raise NotImplementedError("Only implemented for 2D boundary handlings")

    if boundaryNameToColor:
        fixedColors = boundaryNameToColor
    else:
        fixedColors = {
            'fluid': '#1f77ff11',
            'noSlip': '#000000'
        }

    boundaryNames = []
    flagValues = []
    for name, flag in sorted(boundaryHandling.getBoundaryNameToFlagDict().items(), key=lambda l: l[1]):
        boundaryNames.append(name)
        flagValues.append(flag)
    defaultCycler = matplotlib.rcParams['axes.prop_cycle']
    colorValues = [fixedColors[name] if name in fixedColors else cycle['color']
                   for cycle, name in zip(defaultCycler, boundaryNames)]

    cmap = matplotlib.colors.ListedColormap(colorValues)
    bounds = np.array(flagValues, dtype=float) - 0.5
    bounds = list(bounds) + [bounds[-1] + 1]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    flagField = boundaryHandling.flagField.swapaxes(0, 1)
    plt.imshow(flagField, interpolation='none', origin='lower',
               cmap=cmap, norm=norm)

    patches = [matplotlib.patches.Patch(color=color, label=name) for color, name in zip(colorValues, boundaryNames)]
    plt.axis('equal')
    plt.legend(handles=patches, bbox_to_anchor=(1.02, 0.5), loc=2, borderaxespad=0.)

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
        field /= maxNorm
        if 'scale' not in kwargs:
            kwargs['scale'] = 1.0

    quiverPlot = vectorField(field, step=step, **kwargs)
    plotSetupFunction()

    def updatefig(*args):
        f = runFunction()
        f = np.swapaxes(f, 0, 1)
        if rescale:
            maxNorm = np.max(norm(f, axis=2, ord=2))
            f /= maxNorm
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