# Python 3 compatability
# from __future__ import division, print_function
# from six.moves import range

import dynesty
import pickle
#import _pickle as pickle
from dynesty import utils as dyfunc
import dynamical_model as dm

from pathlib import Path

# basic numeric setup
import numpy as np

# plotting
import matplotlib
from matplotlib import pyplot as plt

# re-defining plotting defaults
from matplotlib import rcParams
import types
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
from scipy.ndimage import gaussian_filter as norm_kde
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import logging

try:
    str_type = types.StringTypes
    float_type = types.FloatType
    int_type = types.IntType
except:
    str_type = str
    float_type = float
    int_type = int

rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'font.size': 35})
rcParams.update({'xtick.labelsize': 35})
rcParams.update({'ytick.labelsize': 35})


def _hist2d(x, y, smooth=0.02, span=None, weights=None, levels=None,
            ax=None, color='gray', plot_datapoints=False, plot_density=True,
            plot_contours=True, no_fill_contours=False, fill_contours=True,
            contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
            **kwargs):
    """
    Internal function called by :meth:`cornerplot` used to generate a
    a 2-D histogram/contour of samples.

    Parameters
    ----------
    x : interable with shape (nsamps,)
       Sample positions in the first dimension.

    y : iterable with shape (nsamps,)
       Sample positions in the second dimension.

    span : iterable with shape (ndim,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `0.999999426697` (5-sigma credible interval).

    weights : iterable with shape (nsamps,)
        Weights associated with the samples. Default is `None` (no weights).

    levels : iterable, optional
        The contour levels to draw. Default are `[0.5, 1, 1.5, 2]`-sigma.

    ax : `~matplotlib.axes.Axes`, optional
        An `~matplotlib.axes.axes` instance on which to add the 2-D histogram.
        If not provided, a figure will be generated.

    color : str, optional
        The `~matplotlib`-style color used to draw lines and color cells
        and contours. Default is `'gray'`.

    plot_datapoints : bool, optional
        Whether to plot the individual data points. Default is `False`.

    plot_density : bool, optional
        Whether to draw the density colormap. Default is `True`.

    plot_contours : bool, optional
        Whether to draw the contours. Default is `True`.

    no_fill_contours : bool, optional
        Whether to add absolutely no filling to the contours. This differs
        from `fill_contours=False`, which still adds a white fill at the
        densest points. Default is `False`.

    fill_contours : bool, optional
        Whether to fill the contours. Default is `True`.

    contour_kwargs : dict
        Any additional keyword arguments to pass to the `contour` method.

    contourf_kwargs : dict
        Any additional keyword arguments to pass to the `contourf` method.

    data_kwargs : dict
        Any additional keyword arguments to pass to the `plot` method when
        adding the individual data points.

    """

    if ax is None:
        ax = plt.gca()

    # Determine plotting bounds.
    data = [x, y]
    if span is None:
        span = [0.999999426697 for i in range(2)]
    span = list(span)
    if len(span) != 2:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = dyfunc.quantile(data[i], q, weights=weights)

    # The default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # Color map for the density plot, over-plotted to indicate the
    # density of the points near the center.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_cmap", [color, (1, 1, 1, 0)])

    # Color map used to hide the points at the high density areas.
    white_cmap = LinearSegmentedColormap.from_list(
        "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    # This "color map" is the list of colors for the contour levels if the
    # contours are filled.
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for l in levels] + [rgba_color]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] *= float(i) / (len(levels)+1)

    # Initialize smoothing.
    if (isinstance(smooth, int_type) or isinstance(smooth, float_type)):
        smooth = [smooth, smooth]
    bins = []
    svalues = []
    for s in smooth:
        if isinstance(s, int_type):
            # If `s` is an integer, the weighted histogram has
            # `s` bins within the provided bounds.
            bins.append(s)
            svalues.append(0.)
        else:
            # If `s` is a float, oversample the data relative to the
            # smoothing filter by a factor of 2, then use a Gaussian
            # filter to smooth the results.
            bins.append(int(round(2. / s)))
            svalues.append(2.)

    # We'll make the 2D histogram to directly estimate the density.
    try:
        H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins,
                                 range=list(map(np.sort, span)),
                                 weights=weights)
    except ValueError:
        raise ValueError("It looks like at least one of your sample columns "
                         "have no dynamic range.")

    # Smooth the results.
    if not np.all(svalues == 0.):
        H = norm_kde(H, svalues)

    # Compute the density levels.
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except:
            V[i] = Hflat[0]
    V.sort()
    m = (np.diff(V) == 0)
    if np.any(m) and plot_contours:
        logging.warning("Too few points to create valid contours.")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = (np.diff(V) == 0)
    V.sort()

    # Compute the bin centers.
    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    # Extend the array for the sake of the contours at the plot edges.
    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    X2 = np.concatenate([X1[0] + np.array([-2, -1]) * np.diff(X1[:2]), X1,
                         X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    Y2 = np.concatenate([Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]), Y1,
                         Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])

    # Plot the data points.
    if plot_datapoints:
        if data_kwargs is None:
            data_kwargs = {}
        data_kwargs["color"] = data_kwargs.get("color", color)
        data_kwargs["ms"] = data_kwargs.get("ms", 2.0)
        data_kwargs["mec"] = data_kwargs.get("mec", "none")
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.1)
        ax.plot(x, y, "o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the base fill to hide the densest data points.
    if (plot_contours or plot_density) and not no_fill_contours:
        ax.contourf(X2, Y2, H2.T, [V.min(), H.max()],
                    cmap=white_cmap, antialiased=False)

    if plot_contours and fill_contours:
        if contourf_kwargs is None:
            contourf_kwargs = {}
        contourf_kwargs["colors"] = contourf_kwargs.get("colors", contour_cmap)
        contourf_kwargs["antialiased"] = contourf_kwargs.get("antialiased", False)
        ax.contourf(X2, Y2, H2.T, np.concatenate([[0], V, [H.max()*(1+1e-4)]]), **contourf_kwargs)

    # Plot the density map. This can't be plotted at the same time as the
    # contour fills.
    elif plot_density:
        ax.pcolor(X, Y, H.max() - H.T, cmap=density_cmap)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None:
            contour_kwargs = {}
        contour_kwargs["colors"] = contour_kwargs.get("colors", color)
        ax.contour(X2, Y2, H2.T, V, **contour_kwargs)

    ax.set_xlim(span[0])
    ax.set_ylim(span[1])


def mycorn(results, span=None, quantiles=[0.025, 0.5, 0.975], color='black', smooth=0.02, hist_kwargs=None,
           hist2d_kwargs=None, labels=None, label_kwargs=None, show_titles=False, title_fmt=".2f", title_kwargs=None,
           truths=None, truth_color='red', truth_kwargs=None, max_n_ticks=5, top_ticks=False, use_math_text=False,
           verbose=False, fig=None, errcol='b', medcol='k', labelunits=None, tickfs=40, shortlabs=None, raf=None,
           decf=None):
    """
    Generate a corner plot of the 1-D and 2-D marginalized posteriors.

    Parameters
    ----------
    results : :class:`~dynesty.results.Results` instance
        A :class:`~dynesty.results.Results` instance from a nested
        sampling run. **Compatible with results derived from**
        `nestle <http://kylebarbary.com/nestle/>`_.

    span : iterable with shape (ndim,), optional
        A list where each element is either a length-2 tuple containing
        lower and upper bounds or a float from `(0., 1.]` giving the
        fraction of (weighted) samples to include. If a fraction is provided,
        the bounds are chosen to be equal-tailed. An example would be::

            span = [(0., 10.), 0.95, (5., 6.)]

        Default is `0.999999426697` (5-sigma credible interval).

    quantiles : iterable, optional
        A list of fractional quantiles to overplot on the 1-D marginalized
        posteriors as vertical dashed lines. Default is `[0.025, 0.5, 0.975]`
        (spanning the 95%/2-sigma credible interval).

    color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting the histograms.
        Default is `'black'`.

    smooth : float or iterable with shape (ndim,), optional
        The standard deviation (either a single value or a different value for
        each subplot) for the Gaussian kernel used to smooth the 1-D and 2-D
        marginalized posteriors, expressed as a fraction of the span.
        Default is `0.02` (2% smoothing). If an integer is provided instead,
        this will instead default to a simple (weighted) histogram with
        `bins=smooth`.

    hist_kwargs : dict, optional
        Extra keyword arguments to send to the 1-D (smoothed) histograms.

    hist2d_kwargs : dict, optional
        Extra keyword arguments to send to the 2-D (smoothed) histograms.

    labels : iterable with shape (ndim,), optional
        A list of names for each parameter. If not provided, the default name
        used when plotting will follow :math:`x_i` style.

    label_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_xlabel` and
        `~matplotlib.axes.Axes.set_ylabel` methods.

    show_titles : bool, optional
        Whether to display a title above each 1-D marginalized posterior
        showing the 0.5 quantile along with the upper/lower bounds associated
        with the 0.025 and 0.975 (95%/2-sigma credible interval) quantiles.
        Default is `True`.

    title_fmt : str, optional
        The format string for the quantiles provided in the title. Default is
        `'.2f'`.

    title_kwargs : dict, optional
        Extra keyword arguments that will be sent to the
        `~matplotlib.axes.Axes.set_title` command.

    truths : iterable with shape (ndim,), optional
        A list of reference values that will be overplotted on the traces and
        marginalized 1-D posteriors as solid horizontal/vertical lines.
        Individual values can be exempt using `None`. Default is `None`.

    truth_color : str or iterable with shape (ndim,), optional
        A `~matplotlib`-style color (either a single color or a different
        value for each subplot) used when plotting `truths`.
        Default is `'red'`.

    truth_kwargs : dict, optional
        Extra keyword arguments that will be used for plotting the vertical
        and horizontal lines with `truths`.

    max_n_ticks : int, optional
        Maximum number of ticks allowed. Default is `5`.

    top_ticks : bool, optional
        Whether to label the top (rather than bottom) ticks. Default is
        `False`.

    use_math_text : bool, optional
        Whether the axis tick labels for very large/small exponents should be
        displayed as powers of 10 rather than using `e`. Default is `False`.

    verbose : bool, optional
        Whether to print the values of the computed quantiles associated with
        each parameter. Default is `False`.

    fig : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`), optional
        If provided, overplot the traces and marginalized 1-D posteriors
        onto the provided figure. Otherwise, by default an
        internal figure is generated.

    Returns
    -------
    cornerplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axes`)
        Output corner plot.

    """

    # rcParams['axes.titlesize'] = tickfs

    # Initialize values.
    if quantiles is None:
        quantiles = []

    # Extract weighted samples.
    samples = results['samples']
    try:
        weights = np.exp(results['logwt'] - results['logz'][-1])
    except:
        weights = results['weights']

    print(samples.shape)
    print(np.median(samples[:,3]))
    samples[:,3] += 90.  # MODIFY PA!
    # samples[:, 6] = raf(samples[:, 6])  # MODIFY RA
    # samples[:, 7] = decf(samples[:, 7])  # MODIFY DEC
    samples[:, 6] = (samples[:, 6] - 125.565) * 0.02  # MODIFY RA
    samples[:, 7] = (samples[:, 7] - 149.912) * 0.02  # MODIFY DEC
    #print(oop)

    # Deal with 1D results. A number of extra catches are also here
    # in case users are trying to plot other results besides the `Results`
    # instance generated by `dynesty`.
    samples = np.atleast_1d(samples)
    if len(samples.shape) == 1:
        samples = np.atleast_2d(samples)
    else:
        assert len(samples.shape) == 2, "Samples must be 1- or 2-D."
        samples = samples.T
    assert samples.shape[0] <= samples.shape[1], "There are more " \
                                                 "dimensions than samples!"
    ndim, nsamps = samples.shape

    # Check weights.
    if weights.ndim != 1:
        raise ValueError("Weights must be 1-D.")
    if nsamps != weights.shape[0]:
        raise ValueError("The number of weights and samples disagree!")

    # Determine plotting bounds.
    if span is None:
        span = [0.999999426697 for i in range(ndim)]
    span = list(span)
    if len(span) != ndim:
        raise ValueError("Dimension mismatch between samples and span.")
    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except:
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            span[i] = dyfunc.quantile(samples[i], q, weights=weights)

    # Set labels
    if labels is None:
        labels = [r"$x_{"+str(i+1)+"}$" for i in range(ndim)]

    # Setting up smoothing.
    if (isinstance(smooth, int_type) or isinstance(smooth, float_type)):
        smooth = [smooth for i in range(ndim)]

    # Setup axis layout (from `corner.py`).
    factor = 2.0  # size of side of one panel
    lbdim = 0.5 * factor  # size of left/bottom margin
    trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # size of width/height margin
    plotdim = factor * ndim + factor * (ndim - 1.) * whspace  # plot size
    dim = lbdim + plotdim + trdim  # total size

    # Initialize figure.
    if fig is None:
        fig, axes = plt.subplots(ndim, ndim, figsize=(dim, dim))
    else:
        try:
            fig, axes = fig
            axes = np.array(axes).reshape((ndim, ndim))
        except:
            raise ValueError("Mismatch between axes and dimension.")

    # Format figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    # Plotting.
    for i, x in enumerate(samples):
        if np.shape(samples)[0] == 1:
            ax = axes
        else:
            ax = axes[i, i]

        # Plot the 1-D marginalized posteriors.

        # Setup axes
        ax.set_xlim(span[i])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                   prune="lower"))
            ax.yaxis.set_major_locator(NullLocator())
        # Label axes.
        sf = ScalarFormatter(useMathText=use_math_text)
        ax.xaxis.set_major_formatter(sf)
        if i < ndim - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                #[l.set_rotation(45) for l in ax.get_xticklabels()]
                # plt.xticks(fontsize=tickfs)
            else:
                ax.set_xticklabels([])#, labelsize=tickfs)
        else:
            # [l.set_rotation(45) for l in ax.get_xticklabels()]
            ax.set_xlabel(labels[i], **label_kwargs)
            ax.xaxis.set_label_coords(0.5, -0.3)
            # plt.xticks(fontsize=tickfs)
        # Generate distribution.
        sx = smooth[i]
        if isinstance(sx, int_type):
            # If `sx` is an integer, plot a weighted histogram with
            # `sx` bins within the provided bounds.
            n, b, _ = ax.hist(x, bins=sx, weights=weights, color=color,
                              range=np.sort(span[i]), **hist_kwargs)
        else:
            # If `sx` is a float, oversample the data relative to the
            # smoothing filter by a factor of 10, then use a Gaussian
            # filter to smooth the results.
            bins = int(round(10. / sx))
            n, b = np.histogram(x, bins=bins, weights=weights,
                                range=np.sort(span[i]))
            n = norm_kde(n, 10.)
            b0 = 0.5 * (b[1:] + b[:-1])
            n, b, _ = ax.hist(b0, bins=b, weights=n,
                              range=np.sort(span[i]), color=color, alpha=0.5)  #, **hist_kwargs)
        ax.set_ylim([0., max(n) * 1.05])
        # Plot quantiles.
        if quantiles is not None and len(quantiles) > 0:
            qs = dyfunc.quantile(x, quantiles, weights=weights)
            # for q in qs:
                # ax.axvline(q, lw=2, ls="dashed", color=color)
            qcols = [errcol, medcol, errcol]
            lss = ['--', '-', '--']
            for q in range(len(qs)):
                ax.axvline(qs[q], lw=2, ls=lss[q], color=qcols[q])
            if verbose:
                print("Quantiles:")
                print(labels[i], [blob for blob in zip(quantiles, qs)])
        # Add truth value(s).
        if truths is not None and truths[i] is not None:
            try:
                [ax.axvline(t, color=truth_color, **truth_kwargs)
                 for t in truths[i]]
            except:
                ax.axvline(truths[i], color=truth_color, **truth_kwargs)
        # Set titles.

        if show_titles:
            title = None
            if title_fmt is not None:
                ql, qm, qh = dyfunc.quantile(x, quantiles, weights=weights)  # [0.025, 0.5, 0.975]
                # INSTEAD EDIT: /Users/jonathancohn/anaconda3/envs/three/lib/python3.6/site-packages/dynesty/plotting.py
                q_minus, q_plus = qm - ql, qh - qm
                fmt = "{{0:{0}}}".format(title_fmt).format
                fmt1 = "{{0:{0}}}".format('.1f').format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"  #  {{{3}}}"
                if i == 2 or i == 3 or i == 4 or i == 5:
                    title = title.format(fmt1(qm), fmt1(q_minus), fmt1(q_plus))  # , labelunits[i])
                else:
                    title = title.format(fmt(qm), fmt(q_minus), fmt(q_plus))  # , labelunits[i])
                # title = "{0} = {1}".format(labels[i], title)
                ax.set_title(title, **title_kwargs)

        for j, y in enumerate(samples):
            if np.shape(samples)[0] == 1:
                ax = axes
            else:
                ax = axes[i, j]

            # Plot the 2-D marginalized posteriors.

            # Setup axes.
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue

            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
            # Label axes.
            sf = ScalarFormatter(useMathText=use_math_text)
            ax.xaxis.set_major_formatter(sf)
            ax.yaxis.set_major_formatter(sf)
            if i < ndim - 1:
                ax.set_xticklabels([])#, labelsize=tickfs)
            else:
                # [l.set_rotation(45) for l in ax.get_xticklabels()]
                ax.set_xlabel(labels[j], **label_kwargs)
                ax.xaxis.set_label_coords(0.5, -0.3)
                # plt.xticks(fontsize=tickfs)
                # l = ax.get_xticklabels()
                # ax.set_xticklabels(l, fontsize=tickfs)
                # plt.xticks(fontsize=tickfs)
            if j > 0:
                ax.set_yticklabels([])#, labelsize=tickfs)
            else:
                # [l.set_rotation(45) for l in ax.get_yticklabels()]
                ax.set_ylabel(shortlabs[i], **label_kwargs)
                ax.yaxis.set_label_coords(-0.3, 0.5)
                # plt.yticks(fontsize=tickfs)
                # l = ax.get_yticklabels()
                # ax.set_yticklabels(l, fontsize=tickfs)
                # plt.yticks(fontsize=tickfs)
            # Generate distribution.
            sy = smooth[j]
            check_ix = isinstance(sx, int_type)
            check_iy = isinstance(sy, int_type)
            if check_ix and check_iy:
                fill_contours = False
                plot_contours = False
            else:
                fill_contours = True
                plot_contours = True

            _hist2d(y, x, ax=ax, span=[span[j], span[i]],
                    weights=weights, color=color, smooth=[sy, sx])  # , **hist2d_kwargs)
            # Add truth values
            if truths is not None:
                if truths[j] is not None:
                    try:
                        [ax.axvline(t, color=truth_color, **truth_kwargs)
                         for t in truths[j]]
                    except:
                        ax.axvline(truths[j], color=truth_color,
                                   **truth_kwargs)
                if truths[i] is not None:
                    try:
                        [ax.axhline(t, color=truth_color, **truth_kwargs)
                         for t in truths[i]]
                    except:
                        ax.axhline(truths[i], color=truth_color,
                                   **truth_kwargs)

    return (fig, axes)



def big_table(output_dicts, logged=False, avg=True):

    texlines = '\\begin{table}\n    \\begin{center}\n    \\textbf{Table 3} \\\\\n    Model test results \\\\\n    ' +\
               '\\begin{tabular}{ |l|r|r|r| }\n    \hline\n    Model & \multicolumn{1}{|l|}{$\chi^2_\\nu$}' +\
               ' & \multicolumn{1}{|l|}{\mbh [$\\times10^9M_\odot$]} & $\Delta$\mbh \\\\\n    \hline\n    \hline\n'

    direc = '/Users/jonathancohn/Documents/dyn_mod/nest_out/'
    mbh_fiducial = 2461189947.064265 / 1e9

    table_names = {'ahe': 'dust-corrected', 'dlogz0.001': 'dlogz = 0.001', 'ds48': 'ds $=4\\times8$',
                   'ds510': 'ds $=5\\times10$', 'exp': 'expsig', 'fiducial': 'fiducial', 'fullpriors': 'wide priors',
                   'gas': 'gas', 'kappa': '$\kappa$', 'lucyn5': 'Lucy n = 5', 'lucyn15': 'Lucy n = 15',
                   'nlive1000': 'nlive = 1000', 'os1': '$s=1$', 'os2': '$s=2$', 'os3': '$s=3$', 'os6': '$s=6$',
                   'os8': '$s=8$', 'os10': '$s=10$', 'os12': '$s=12$', 'rfit0.3': 'r$_{\\text{ell}}=0\\farcs{3}$',
                   'rfit0.4': 'r$_{\\text{ell}}=0\\farcs{4}$', 'rfit0.5': 'r$_{\\text{ell}}=0\\farcs{5}$',
                   'rfit0.6': 'r$_{\\text{ell}}=0\\farcs{6}$', 'rfit0.8': 'r$_{\\text{ell}}=0\\farcs{8}$',
                   'rre': 'original $H$-band', 'vrad': 'v$_{\\text{rad}}$'}

    for od in output_dicts:
        thing = direc + output_dicts[od]['pkl']
        parfile = output_dicts[od]['outpf']
        model = output_dicts[od]['mod']
        print(model)

        params, priors, nfree, qobs = dm.par_dicts(parfile, q=True)  # get params and file names from output parfile

        if 'ds2' not in params:
            params['ds2'] = params['ds']

        mod_ins = dm.model_prep(data=params['data'], ds=params['ds'], ds2=params['ds2'], lucy_out=params['lucy'],
                                lucy_b=params['lucy_b'], lucy_mask=params['lucy_mask'], lucy_in=params['lucy_in'],
                                lucy_it=params['lucy_it'], data_mask=params['mask'], grid_size=params['gsize'],
                                res=params['resolution'], x_std=params['x_fwhm'], y_std=params['y_fwhm'], avg=avg,
                                xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                                zrange=[params['zi'], params['zf']], theta_ell=np.deg2rad(params['theta_ell']),
                                xell=params['xell'], yell=params['yell'], q_ell=params['q_ell'], pa=params['PAbeam'])

        lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise, co_rad, co_sb = mod_ins
        vrad = None
        kappa = None
        omega = None
        if model == 'vrad':
            vrad = params['vrad']
        elif model == 'omega':
            kappa = params['kappa']
            omega = params['omega']
        elif model == 'kappa':
            kappa = params['kappa']

        inc_fixed = np.deg2rad(67.7)  # based on fiducial model (67.68 deg)
        vcg_in = None
        if params['incl_gas'] == 'True':
            vcg_in = dm.gas_vel(params['resolution'], co_rad, co_sb, params['dist'], f_0, inc_fixed, zfixed=0.02152)

        mg = dm.ModelGrid(x_loc=params['xloc'], y_loc=params['yloc'], mbh=params['mbh'], ml_ratio=params['ml_ratio'],
            inc=np.deg2rad(params['inc']), vsys=params['vsys'], theta=np.deg2rad(params['PAdisk']), vrad=vrad,
            kappa=kappa, omega=omega, f_w=params['f'], os=params['os'], enclosed_mass=params['mass'],
            sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], resolution=params['resolution'],
            lucy_out=lucy_out, out_name=None, beam=beam, rfit=params['rfit'], zrange=[params['zi'], params['zf']],
            dist=params['dist'], input_data=input_data, sig_type=params['s_type'], menc_type=params['mtype'],
            theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'],yell=params['yell'], q_ell=params['q_ell'],
            ds=params['ds'], ds2=params['ds2'], reduced=True, f_0=f_0, freq_ax=freq_ax, noise=noise, bl=params['bl'],
            fstep=fstep, xyrange=[params['xi'], params['xf'], params['yi'], params['yf']], n_params=nfree,
            data_mask=params['mask'], incl_gas=params['incl_gas']=='True', co_rad=co_rad, co_sb=co_sb, vcg_func=vcg_in,
            pvd_width=params['x_fwhm'] / params['resolution'], avg=avg, quiet=True)

        mg.grids()
        mg.convolution()
        chi2, chi2_nu = mg.chi2()
        # print(model)
        print(chi2, chi2_nu)

        fmt = "{{0:{0}}}".format('.2f').format
        fmt3 = "{{0:{0}}}".format('.3f').format
        chititle = r"${{{0}}}$".format(fmt3(chi2))
        chinutitle = r"${{{0}}}$".format(fmt3(chi2_nu))
        # texlines += '    ' + table_names[model] + ' & ' + chititle + ' (' + chinutitle + ') & '
        texlines += '    ' + table_names[model] + ' & ' + chinutitle + ' & '

        with open(thing, 'rb') as pk:
            u = pickle._Unpickler(pk)
            u.encoding = 'latin1'
            dyn_res = u.load()  #
            # dyn_res = pickle.load(pk)  #

        weights = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])  # normalized weights

        quants = [0.0015, 0.5, 0.9985]
        if sig == 1 or sig == 'mod':
            quants = [0.16, 0.5, 0.84]
        elif sig == 2:
            quants = [0.025, 0.5, 0.975]
        elif sig == 3:
            quants = [0.0015, 0.5, 0.9985]

        mbh_q = dyfunc.quantile(dyn_res['samples'][:, 0], quants, weights=weights)
        if logged:
            q = np.log10(mbh_q)
        else:
            q = np.asarray(mbh_q) / 1e9

        dmbh = 100 * (q[1] - mbh_fiducial) / mbh_fiducial

        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        if sig == 'mod':
            mod = (2*4597) ** (1/4)  # BUCKET 4606 changes if the fitting region changes
            texlines += title.format(fmt(q[1]), fmt((q[1] - q[0]) * mod), fmt((q[2] - q[1]) * mod)) + ' & '
        else:
            texlines += title.format(fmt(q[1]), fmt(q[1] - q[0]), fmt(q[2] - q[1])) + ' & ' + fmt(dmbh) + '\% \\\\\n' +\
                        '    \hline\n'

    texlines += '    \end{tabular}\n    \end{center}\n    \caption{\\textbf{to do}}\n    \label{tab_compare}\n' +\
                '\end{table}'

    return texlines


def table_it(things, parfiles, models, parlabels, sig=3, avg=True, logged=False, percent_diff=False, pris=None,
             tabpars=None, raf=None, decf=None):

    hdr = '| model | '
    hdrl = '| --- |'
    for la in range(len(parlabels)):
        hdrl += ' --- |'
        hdr += parlabels[la] + ' | '

    texlines = '& '
    lines = '| '
    newlines = ''

    for t in range(len(things)):
        print(t, len(things), len(parfiles), len(models))
        params, priors, nfree, qobs = dm.par_dicts(parfiles[t], q=True)  # get params and file names from output parfile

        if 'ds2' not in params:
            params['ds2'] = params['ds']

        mod_ins = dm.model_prep(data=params['data'], ds=params['ds'], ds2=params['ds2'], lucy_out=params['lucy'],
                                lucy_b=params['lucy_b'], lucy_mask=params['lucy_mask'], lucy_in=params['lucy_in'],
                                lucy_it=params['lucy_it'], data_mask=params['mask'], grid_size=params['gsize'],
                                res=params['resolution'], x_std=params['x_fwhm'], y_std=params['y_fwhm'], avg=avg,
                                xyerr=[params['xerr0'], params['xerr1'], params['yerr0'], params['yerr1']],
                                zrange=[params['zi'], params['zf']], theta_ell=np.deg2rad(params['theta_ell']),
                                xell=params['xell'], yell=params['yell'], q_ell=params['q_ell'], pa=params['PAbeam'])

        lucy_mask, lucy_out, beam, fluxes, freq_ax, f_0, fstep, input_data, noise, co_rad, co_sb = mod_ins
        vrad = None
        kappa = None
        omega = None
        if 'vrad' in parlabels:
            vrad = params['vrad']
        elif 'omega' in parlabels:
            kappa = params['kappa']
            omega = params['omega']
        elif 'kappa' in parlabels:
            kappa = params['kappa']

        inc_fixed = np.deg2rad(67.7)  # based on fiducial model (67.68 deg)
        vcg_in = None
        if params['incl_gas'] == 'True':
            vcg_in = dm.gas_vel(params['resolution'], co_rad, co_sb, params['dist'], f_0, inc_fixed, zfixed=0.02152)

        mg = dm.ModelGrid(x_loc=params['xloc'], y_loc=params['yloc'], mbh=params['mbh'], ml_ratio=params['ml_ratio'],
            inc=np.deg2rad(params['inc']), vsys=params['vsys'], theta=np.deg2rad(params['PAdisk']), vrad=vrad,
            kappa=kappa, omega=omega, f_w=params['f'], os=params['os'], enclosed_mass=params['mass'],
            sig_params=[params['sig0'], params['r0'], params['mu'], params['sig1']], resolution=params['resolution'],
            lucy_out=lucy_out, out_name=None, beam=beam, rfit=params['rfit'], zrange=[params['zi'], params['zf']],
            dist=params['dist'], input_data=input_data, sig_type=params['s_type'], menc_type=params['mtype'],
            theta_ell=np.deg2rad(params['theta_ell']), xell=params['xell'],yell=params['yell'], q_ell=params['q_ell'],
            ds=params['ds'], ds2=params['ds2'], reduced=True, f_0=f_0, freq_ax=freq_ax, noise=noise, bl=params['bl'],
            fstep=fstep, xyrange=[params['xi'], params['xf'], params['yi'], params['yf']], n_params=nfree,
            data_mask=params['mask'], incl_gas=params['incl_gas']=='True', co_rad=co_rad, co_sb=co_sb, vcg_func=vcg_in,
            pvd_width=(params['x_fwhm'] + params['y_fwhm']) / params['resolution'] / 2., avg=avg)

        mg.grids()
        mg.convolution()
        chi2, chi2_nu = mg.chi2()
        print(models[t])
        print(chi2, chi2_nu)

        fmt1 = "{{0:{0}}}".format('.1f').format
        fmt2 = "{{0:{0}}}".format('.2f').format
        fmt3 = "{{0:{0}}}".format('.3f').format
        chititle = r"${{0}}$".format(fmt3(chi2))
        chinutitle = r"${{0}}$".format(fmt3(chi2_nu))
        altchititle = r"{0}".format(fmt3(chi2))
        altchinutitle = r"{0}".format(fmt3(chi2_nu))
        texlines += models[t] + ' & ' + chititle + ' & ' + chinutitle + ' & '
        lines += models[t] + ' | ' + altchititle + ' | ' + altchinutitle + ' | '

        with open(things[t], 'rb') as pk:
            u = pickle._Unpickler(pk)
            u.encoding = 'latin1'
            dyn_res = u.load()  #
            # dyn_res = pickle.load(pk)  #

        weights = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])  # normalized weights

        quants = [0.0015, 0.5, 0.9985]
        if sig == 1 or sig == 'mod':
            quants = [0.16, 0.5, 0.84]
        elif sig == 2:
            quants = [0.025, 0.5, 0.975]
        elif sig == 3:
            quants = [0.0015, 0.5, 0.9985]

        orig_order = np.array(['mbh', 'xloc', 'yloc', 'sig', 'inc', 'pa', 'vsys', 'ml', 'f0'])
        re_order = np.array([0, 7, 4, 5, 6, 3, 1, 2, 8])
        dyn_res['samples'] = dyn_res['samples'][:, re_order]
        tabpars = tabpars[re_order]
        pris = pris[re_order, :]
        reordered_pars = orig_order[re_order]

        for i in range(dyn_res['samples'].shape[1]):  # for each parameter
            q = dyfunc.quantile(dyn_res['samples'][:, i], quants, weights=weights)
            q1 = dyfunc.quantile(dyn_res['samples'][:, i], [0.16, 0.5, 0.84], weights=weights)
            q3 = dyfunc.quantile(dyn_res['samples'][:, i], [0.0015, 0.5, 0.9985], weights=weights)
            print(q)
            if i == 0 and 'nobh' not in parfiles[t]:
                q = np.asarray(q) / 1e9
                q1 = np.asarray(q1) / 1e9
                q3 = np.asarray(q3) / 1e9
            if tabpars[i].startswith(r'\xloc\ '):
                # q1 = raf(np.asarray(q1))[::-1]
                # q3 = raf(np.asarray(q3))[::-1]
                q1 = (np.asarray(q1)[::-1] - 125.565) * 0.02
                q3 = (np.asarray(q3)[::-1] - 125.565) * 0.02
            elif tabpars[i].startswith(r'\yloc\ '):
                # q1 = decf(np.asarray(q1))
                # q3 = decf(np.asarray(q3))
                q1 = (np.asarray(q1) - 149.912) * 0.02
                q3 = (np.asarray(q3) - 149.912) * 0.02
            if tabpars[i] == r'\pa\ [$^\circ$]':
                q1 = np.asarray(q1) + 90
                q3 = np.asarray(q3) + 90
                pris[i][0] += 90
                pris[i][1] += 90
                #if logged:
                #    q = np.log10(q)
                #else:
                #    q = np.asarray(q) / 1e9
            newlines += tabpars[i] + ' & '
            title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            title_new = r"${{{0}}}$ & $_{{-{1}}}^{{+{2}}}$ & $_{{-{3}}}^{{+{4}}}$ & ${{{5}}} \rightarrow {{{6}}}$ \\"
            alttitle = r"{0} -{1}/+{2}"
            if sig == 'mod':
                mod = (2*4597) ** (1/4)  # BUCKET 4606 changes if the fitting region changes
                texlines += title.format(fmt2(q[1]), fmt2((q[1] - q[0]) * mod), fmt2((q[2] - q[1]) * mod)) + ' & '
                lines += alttitle.format(fmt2(q[1]), fmt2((q[1] - q[0]) * mod), fmt2((q[2] - q[1]) * mod)) + ' | '
            else:
                texlines += title.format(fmt2(q[1]), fmt2(q[1] - q[0]), fmt2(q[2] - q[1])) + ' & '
                lines += alttitle.format(fmt2(q[1]), fmt2(q[1] - q[0]), fmt2(q[2] - q[1])) + ' | '

                if reordered_pars[i] == 'inc' or reordered_pars[i] == 'pa' or reordered_pars[i] == 'sig' or\
                    reordered_pars[i] == 'vsys':
                    newlines += title_new.format(fmt1(q1[1]), fmt1(q1[1] - q1[0]), fmt1(q1[2] - q1[1]),
                                                 fmt1(q3[1] - q3[0]), fmt1(q3[2] - q3[1]), str(fmt1(pris[i][0])),
                                                 str(fmt1(pris[i][1]))) + '\n'
                else:
                    newlines += title_new.format(fmt2(q1[1]), fmt2(q1[1] - q1[0]), fmt2(q1[2] - q1[1]),
                                                 fmt2(q3[1] - q3[0]), fmt2(q3[2] - q3[1]), str(fmt2(pris[i][0])),
                                                 str(fmt2(pris[i][1]))) + '\n'

            #lines += str(q[1].format('.2f').format) + ' +' + str((q[2]-q[1]).format('.2f').format) + ' -' +\
            #         str((q[1] - q[0]).format('.2f').format) + ' | '

        if percent_diff:
            hdr += 'MBH % difference |'
            fid_bh = 2461189947.064265
            mod_bh = dyfunc.quantile(dyn_res['samples'][:, 0], quants, weights=weights)[1]
            lines += str(fmt2(100 * (mod_bh - fid_bh) / fid_bh)) + '% |'
            texlines += str(fmt2(100 * (mod_bh - fid_bh) / fid_bh)) + '%'

        lines += '\n| '
        texlines += '\n| '
        # newlines += '\\\\ \n'

    return hdr, hdrl, lines, texlines, newlines


def my_own_thing(results, par_labels, ax_labels, quantiles, ax_lims=None, fs=20, savefig=None):
    # results should be dyn_res['samples']
    roundto = 3  # 2  # 4
    npar = len(par_labels)
    if npar == 10:
        fig, axes = plt.subplots(2, 5, figsize=(20, 12))  # 2 rows, 5 cols of subplots; because there are 10 free params
        # labels = np.array(['mbh', 'xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f',]) vrad, kappa, etc
        axes_order = [[0, 0], [1, 0], [1, 1], [1, 2], [0, 3], [0, 4], [1, 3], [0, 1], [0, 2], [1, 4]]
    elif npar == 11:
        fig, axes = plt.subplots(3, 4, figsize=(20, 12))  # 3 rows, 4 cols of subplots; because there are 11 free params
        # labels =   ['mbh', 'xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f', kappa, omega], etc
        axes_order = [[0, 0], [1, 1], [1, 2], [1, 3], [0, 3], [1, 0], [2, 0], [0, 1], [0, 2], [2, 1], [2, 2]]
    elif npar == 9:
        fig, axes = plt.subplots(3, 3, figsize=(20, 12))  # 3 rows, 3 cols of subplots; because there are 9 free params
        # labels = np.array(['mbh', 'xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f'])
        axes_order = [[0, 0], [2, 0], [2, 1], [2, 2], [1, 0], [1, 1], [1, 2], [0, 1], [0, 2]]
    elif npar == 8:
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))  # 2 rows, 4 cols of subplots; because there are 8 free params
        # labels = np.array(['xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f'])
        axes_order = [[0, 0], [0, 1], [1, 0], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3]]
    for i in range(len(results[0])):
        row, col = axes_order[i]
        if par_labels[i] == r'$\log_{10}(M_{\text{BH}}/$M$_{\odot})$':  # or par_labels[i] == 'PAdisk':
            bins = 400  # 400 #1000
        else:
            bins = 100
        chain = results[:, i]
        weight = np.ones_like(chain) * 2e-3
        axes[row, col].hist(chain, bins=bins, color="b", histtype="step", weights=weight)  # axes[i]
        # print(quantiles[i], 'look')
        # percs = np.percentile(chain, [.15, 50., 99.85])  # 3sigma
        # axes[row, col].axvline(percs[1], color='b', ls='--')  # axes[i]
        # axes[row, col].axvspan(percs[0], percs[2], color='b', alpha=0.25)
        axes[row, col].axvline(quantiles[i][1], color='b', ls='--')  # axes[i]
        axes[row, col].axvspan(quantiles[i][0], quantiles[i][2], color='b', alpha=0.25)
        axes[row, col].tick_params('both', labelsize=fs)
        axes[row, col].set_title(par_labels[i] + ': ' + str(round(quantiles[i][1], roundto)) + ' (+'
                                 + str(round(quantiles[i][2] - quantiles[i][1], roundto)) + ', -'
                                 + str(round(quantiles[i][1] - quantiles[i][0], roundto)) + ')', fontsize=fs)
        axes[row, col].set_xlabel(ax_labels[i], fontsize=fs)

        if ax_lims is not None:
            axes[row, col].set_xlim(ax_lims[i][0], ax_lims[i][1])
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()


def output_dictionaries(err):
    fid_dict = {'pkl': 'ugc_2698_finaltests_fiducial_10000000_8_0.02_1598991563.9127946_end.pkl',
                'name': 'finaltests/u2698_finaltests_fiducial_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_fiducial_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_fiducial.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_fiducial_out.txt',
                'mod': 'fiducial', 'extra_params': None}

    ahe_dict = {'pkl': 'ugc_2698_finaltests_ahe_10000000_8_0.02_1598996196.6371024_end.pkl',
                'name': 'finaltests/u2698_finaltests_ahe_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_ahe_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_ahe.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_ahe_out.txt',
                'mod': 'ahe', 'extra_params': None}

    rre_dict = {'pkl': 'ugc_2698_finaltests_rre_10000000_8_0.02_1599004482.6025503_end.pkl',
                'name': 'finaltests/u2698_finaltests_rre_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_rre_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_rre.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_rre_out.txt',
                'mod': 'rre', 'extra_params': None}

    dlz_dict = {'pkl': 'ugc_2698_finaltests_dlogz0.001_10000000_8_0.001_1598998076.8872557_end.pkl',
                'name': 'finaltests/u2698_finaltests_dlogz0.001_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_dlogz0.001_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_dlogz0.001.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_dlogz0.001_out.txt',
                'mod': 'dlogz0.001', 'extra_params': None}

    nlv_dict = {'pkl': 'ugc_2698_finaltests_nlive1000_10000000_8_0.02_1599014686.3287234_end.pkl',
                'name': 'finaltests/u2698_finaltests_nlive1000_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_nlive1000_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_nlive1000.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_nlive1000_out.txt',
                'mod': 'nlive1000', 'extra_params': None}

    d48_dict = {'pkl': 'ugc_2698_finaltests_ds48_10000000_8_0.02_1598991134.1821778_end.pkl',
                'name': 'finaltests/u2698_finaltests_ds48_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_ds48_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_ds48.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_ds48_out.txt',
                'mod': 'ds48', 'extra_params': None}

    d51_dict = {'pkl': 'ugc_2698_finaltests_ds510_10000000_8_0.02_1598996125.0661013_end.pkl',
                'name': 'finaltests/u2698_finaltests_ds510_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_ds510_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_ds510.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_ds510_out.txt',
                'mod': 'ds510', 'extra_params': None}

    exp_dict = {'pkl': 'ugc_2698_finaltests_exp_10000000_8_0.02_1599221245.840947_end.pkl',  # ~297000 func calls; ~638013 (batch stage)
                'name': 'finaltests/u2698_finaltests_exp_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_exp_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_exp.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_exp_out.txt',
                'mod': 'exp', 'extra_params': [['r0', 'pc'], ['sig1', 'km/s']]}

    ful_dict = {'pkl': 'ugc_2698_finaltests_fullpriors_10000000_8_0.02_1599057523.4734955_end.pkl',
                'name': 'finaltests/u2698_finaltests_fullpriors_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_fullpriors_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_fullpriors.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_fullpriors_out.txt',
                'mod': 'fullpriors', 'extra_params': None}

    gas_dict = {'pkl': 'ugc_2698_finaltests_gas_10000000_8_0.02_1598991110.5280113_end.pkl',
                'name': 'finaltests/u2698_finaltests_gas_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_gas_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_gas.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_gas_out.txt',
                'mod': 'gas', 'extra_params': None}

    kap_dict = {'pkl': 'ugc_2698_finaltests_kappa_10000000_8_0.02_1599130684.4075377_end.pkl',  # ~327116 func calls temp, ~413422 temp2 (prelim cvg)
                'name': 'finaltests/u2698_finaltests_kappa_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_kappa_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_kappa.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_kappa_out.txt',
                'mod': 'kappa', 'extra_params': [['kappa', 'unitless']]}

    s01_dict = {'pkl': 'ugc_2698_finaltests_os1_10000000_8_0.02_1598993131.9236333_end.pkl',
                'name': 'finaltests/u2698_finaltests_os1_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os1_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os1.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os1_out.txt',
                'mod': 'os1', 'extra_params': None}

    s02_dict = {'pkl': 'ugc_2698_finaltests_os2_10000000_8_0.02_1598993294.4051502_end.pkl',
                'name': 'finaltests/u2698_finaltests_os2_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os2_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os2.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os2_out.txt',
                'mod': 'os2', 'extra_params': None}

    s03_dict = {'pkl': 'ugc_2698_finaltests_os3_10000000_8_0.02_1598987576.2494218_end.pkl',
                'name': 'finaltests/u2698_finaltests_os3_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os3_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os3.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os3_out.txt',
                'mod': 'os3', 'extra_params': None}

    s06_dict = {'pkl': 'ugc_2698_finaltests_os6_10000000_8_0.02_1599013569.0664136_end.pkl',
                'name': 'finaltests/u2698_finaltests_os6_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os6_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os6.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os6_out.txt',
                'mod': 'os6', 'extra_params': None}

    s08_dict = {'pkl': 'ugc_2698_finaltests_os8_10000000_8_0.02_1599021614.7866514_end.pkl',
                'name': 'finaltests/u2698_finaltests_os8_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os8_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os8.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os8_out.txt',
                'mod': 'os8', 'extra_params': None}

    s10_dict = {'pkl': 'ugc_2698_finaltests_os10_10000000_8_0.02_1599027485.286418_end.pkl',
                'name': 'finaltests/u2698_finaltests_os10_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os10_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os10.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os10_out.txt',
                'mod': 'os10', 'extra_params': None}

    s12_dict = {'pkl': 'ugc_2698_finaltests_os12_10000000_8_0.02_1599039316.859327_end.pkl',
                'name': 'finaltests/u2698_finaltests_os12_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_os12_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_os12.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_os12_out.txt',
                'mod': 'os12', 'extra_params': None}

    r03_dict = {'pkl': 'ugc_2698_finaltests_rfit0.3_10000000_8_0.02_1598973877.4783337_tempsave.pkl',  # ~325800 calls; ~649210 temp2 (unchanged from temp); ~958458 temp3
                'name': 'finaltests/u2698_finaltests_rfit0.3_' + str(err) + 'sig_temp3.png',
                'cornername': 'finaltests/u2698_finaltests_rfit0.3_corner_' + str(err) + 'sig_temp3.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_rfit0.3.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_rfit0.3_out_temp3.txt',
                'mod': 'rfit0.3', 'extra_params': None}

    r04_dict = {'pkl': 'ugc_2698_finaltests_rfit0.4_10000000_8_0.02_1598973877.4808373_tempsave.pkl',  # ~293900 calls; ~627888 temp2 (identical to temp); ~918764 temp3
                'name': 'finaltests/u2698_finaltests_rfit0.4_' + str(err) + 'sig_temp3.png',
                'cornername': 'finaltests/u2698_finaltests_rfit0.4_corner_' + str(err) + 'sig_temp3.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_rfit0.4.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_rfit0.4_out_temp3.txt',
                'mod': 'rfit0.4', 'extra_params': None}

    r05_dict = {'pkl': 'ugc_2698_finaltests_rfit0.5_10000000_8_0.02_1599019066.0246398_end.pkl',
                'name': 'finaltests/u2698_finaltests_rfit0.5_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_rfit0.5_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_rfit0.5.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_rfit0.5_out.txt',
                'mod': 'rfit0.5', 'extra_params': None}

    r06_dict = {'pkl': 'ugc_2698_finaltests_rfit0.6_10000000_8_0.02_1599003992.1003041_end.pkl',
                'name': 'finaltests/u2698_finaltests_rfit0.6_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_rfit0.6_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_rfit0.6.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_rfit0.6_out.txt',
                'mod': 'rfit0.6', 'extra_params': None}

    r08_dict = {'pkl': 'ugc_2698_finaltests_rfit0.8_10000000_8_0.02_1598994597.4476626_end.pkl',
                'name': 'finaltests/u2698_finaltests_rfit0.8_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_rfit0.8_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_rfit0.8.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_rfit0.8_out.txt',
                'mod': 'rfit0.8', 'extra_params': None}

    vra_dict = {'pkl': 'ugc_2698_finaltests_vrad_10000000_8_0.02_1599163402.2385688_end.pkl',  # ~233582 calls, temp2 299242 (after prelim cvg), temp3 ~603270
                'name': 'finaltests/u2698_finaltests_vrad_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_vrad_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_vrad.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_vrad_out.txt',
                'mod': 'vrad', 'extra_params': [['vrad', 'km/s']]}

    l05_dict = {'pkl': 'ugc_2698_finaltests_lucyn5_10000000_8_0.02_1599618047.3316298_end.pkl',
                'name': 'finaltests/u2698_finaltests_lucyn5_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_lucyn5_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_lucyn5.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_lucyn5_out.txt',
                'mod': 'lucyn5', 'extra_params': None}

    l15_dict = {'pkl': 'ugc_2698_finaltests_lucyn15_10000000_8_0.02_1599615320.6710668_end.pkl',
                'name': 'finaltests/u2698_finaltests_lucyn15_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_lucyn15_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_lucyn15.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_lucyn15_out.txt',
                'mod': 'lucyn15', 'extra_params': None}

    lvb_dict = {'pkl': 'ugc_2698_finaltests_lucyvb_10000000_8_0.02_1600323488.5804818_end.pkl',
                'name': 'finaltests/u2698_finaltests_lucyvb_' + str(err) + 'sig.png',
                'cornername': 'finaltests/u2698_finaltests_lucyvb_corner_' + str(err) + 'sig.png',
                'inpf': 'ugc_2698/ugc_2698_finaltests_lucyvb.txt',
                'outpf': 'ugc_2698/ugc_2698_finaltests_lucyvb_out.txt',
                'mod': 'lucyvb', 'extra_params': None}

    results_dict = {
                    'fiducial': fid_dict,#
                    'ahe': ahe_dict,#
                    'rre': rre_dict,#
                    'os1': s01_dict,  #
                    'os2': s02_dict,  #
                    'os3': s03_dict,  #
                    'os6': s06_dict,  #
                    'os8': s08_dict,  #
                    'os10': s10_dict,  #
                    'os12': s12_dict,  #
                    'ds48': d48_dict,#
                    'ds510': d51_dict,#
                    'rfit0.3': r03_dict,  #
                    'rfit0.4': r04_dict,  #
                    'rfit0.5': r05_dict,  #
                    'rfit0.6': r06_dict,  #
                    'rfit0.8': r08_dict,  #
                    'gas': gas_dict,#
                    'kappa': kap_dict,#
                    'vrad': vra_dict,
                    'exp': exp_dict,#
                    'lucyn5': l05_dict,#
                    'lucyn15': l15_dict,#
                    'lucyvb': lvb_dict,#
                    'nlive': nlv_dict,#
                    'dlogz': dlz_dict,#
                    'fullpriors': ful_dict,#
                    }

    return results_dict


# DEFINE DIRECTORIES
direc = '/Users/jonathancohn/Documents/dyn_mod/nest_out/'
grp = '/Users/jonathancohn/Documents/dyn_mod/groupmtg/'
sig = 1  # 1 # 3  # show 1sigma errors or 3sigma errors

# CHOOSE DICTIONARY
dict = output_dictionaries(sig)['fiducial']

if 'nobh' in dict['pkl']:
    # labels = np.array(['xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f'])
    labels = np.array([r'$x_0$', r'$y_0$', r'$\sigma_0$', r'$i$', r'$\Gamma$', r'v$_{\text{sys}}$', r'$M/L$',
                       r'$f_0$'])
    ax_lab = np.array(['pixels', 'pixels', 'km $s^{-1}$', 'deg', 'deg', 'km $s^{-1}$', r'M$_{\odot}$/L$_{\odot}$',
                       'unitless'])
    tablabs = np.array(['reduced chi^2', 'xloc [pix]', 'yloc [pix]', 'sig0 [km/s]', 'inc [deg]',
                        'PAdisk [deg]', 'vsys [km/s]', 'ml_ratio [Msol/Lsol]', 'f [unitless]'])
else:
    # labels = np.array(['mbh', 'xloc', 'yloc', 'sig0', 'inc', 'PAdisk', 'vsys', 'ml_ratio', 'f'])
    # labels = np.array([r'M$_{\text{BH}}$', r'$x_0$', r'$y_0$', r'$\sigma_0$', r'$\iota$', r'$\Gamma$',
    labels = np.array([r'$\log_{10}(M_{\text{BH}}/$M$_{\odot})$', r'$x_0$', r'$y_0$', r'$\sigma_0$', r'$i$',
                       r'$\Gamma$', r'$v_{\text{sys}}$', r'$M/L_{\text{H}}$', r'$f_0$'])
    # ax_lab = np.array([r'$\log_{10}$(M$_{\odot}$)', 'pixels', 'pixels', 'km/s', 'deg', 'deg', 'km/s',
    #                    r'M$_{\odot}$/L$_{\odot}$', 'unitless'])
    ax_lab = np.array([r'$\log_{10}$(M$_{\text{BH}}/$M$_{\odot}$)', 'pixels', 'pixels', 'km $s^{-1}$', 'deg', 'deg',
                       'km $s^{-1}$', r'M$_{\odot}$/L$_{\odot}$', 'unitless'])
    # cornunits = np.array(['', 'pixels', 'pixels', 'km $s^{-1}$', 'deg', 'deg', 'km $s^{-1}$',
    #                       r'M$_{\odot}$/L$_{\odot}$', ''])
    cornunits = np.array(['', 'pixels', 'pixels', 'km $s^{-1}$', 'deg', 'deg', 'km $s^{-1}$',
                          r'M$_{\odot}$/L$_{\odot}$', ''])
    tablabs = np.array(['chi^2' 'reduced chi^2', 'log10(mbh) [Msol]', 'xloc [pix]', 'yloc [pix]', 'sig0 [km/s]',
                        'inc [deg]', 'PAdisk [deg]', 'vsys [km/s]', 'ml_ratio [Msol/Lsol]', 'f [unitless]'])
    tabpars = np.array([r'\mbh\ [$10^9\ M_\odot$]', r'\xloc\ [$\Delta\arcsec$ RA]', r'\yloc\ [$\Delta\arcsec$ Dec]',
                        r'\sig\ [km s$^{-1}$]', r'\inc\ [$^\circ$]', r'\pa\ [$^\circ$]', r'\vsys\ [km s$^{-1}$]',
                        r'\ml\ [$M_\odot$/$L_\odot$]', '\\fw\\'])
    cornerax = np.array([r'$\log_{10}(M_{\text{BH}}/$M$_{\odot})$', r'$x_0$ [$\Delta^{\prime\prime}$ RA]',
                         r'$y_0$ [$\Delta^{\prime\prime}$ Dec]', r'$\sigma_0$ [km s$^{-1}$]', r'$i$ [deg]', r'$\Gamma$ [deg]',
                         r'$v_{\text{sys}}$ [km s$^{-1}$]', r'$M/L_{\text{H}}$ [M$_{\odot}$/L$_{\odot}$]', r'$f_0$'])
                        # r'$x_0$ [pixels]', r'$y_0$ [pixels]',

    pris = np.array([[0.1,10], [128, 124], [148, 152], [0, 40], [52.38, 89], [5, 35], [6405, 6505], [0.3, 3],
                     [0.5, 1.5]])

if dict['extra_params'] is not None:
    for par in dict['extra_params']:
        labels = np.append(labels, par[0])
        ax_lab = np.append(ax_lab, par[1])
        tablabs = np.append(tablabs, par[0] + ' [' + par[1] + ']')

print(labels)


### CHANGE TO RA, DEC ARCSEC OFFSET
from astropy.io import fits
hdua = fits.open('/Users/jonathancohn/Documents/dyn_mod/ugc_2698/UGC2698_C4_CO21_bri_20.3kms.pbcor.fits')
hdra = hdua[0].header
hdua.close()
ras = []
decs = []
rasfull = []
decsfull = []
scale = 3600.  # 3600 if want arcsec units, 1 if want deg units
for xx in range(hdra['NAXIS1']):
    rasfull.append(((xx - (hdra['CRPIX1']-1)) * hdra['CDELT1'] + hdra['CRVAL1']) * scale)  # CRPIX-1 bc python starts at 0
    ras.append(((xx - (hdra['CRPIX1']-1)) * hdra['CDELT1']) * scale)  # CRPIX-1 bc python starts at 0; OFFSET from 150
for yy in range(hdra['NAXIS2']):
    decsfull.append(((yy - (hdra['CRPIX2']-1)) * hdra['CDELT2'] + hdra['CRVAL2']) * scale)  # CRPIX-1 bc python starts at 0
    decs.append(((yy - (hdra['CRPIX2']-1)) * hdra['CDELT2']) * scale)  # CRPIX-1 bc python starts at 0; OFFSET from 150
from scipy import interpolate
ra_func = interpolate.interp1d(range(len(ras)), ras, kind='quadratic', fill_value='extrapolate')
dec_func = interpolate.interp1d(range(len(decs)), decs, kind='quadratic', fill_value='extrapolate')
rafull_func = interpolate.interp1d(range(len(rasfull)), rasfull, kind='quadratic', fill_value='extrapolate')
decfull_func = interpolate.interp1d(range(len(decsfull)), decsfull, kind='quadratic', fill_value='extrapolate')
print(ra_func(126.85457), ra_func(150))
print(dec_func(150.96257), dec_func(150))
# pris[1] = ra_func(pris[1])
# pris[2] = dec_func(pris[2])
pris[1] = (pris[1] - 125.565) * 0.02  # MODIFY RA
pris[2] = (pris[2] - 149.912) * 0.02  # MODIFY DEC
###

# '''  #
# ONLY table_it *AFTER* OUT FILE CREATED
#tl = big_table(output_dictionaries(sig))
#print(tl)
#print(oop)

dictr0 = output_dictionaries(sig)['lucyn5']  # rfit0.5  # ds48  # nlive  # lucyn5  # lucyn15
dictr1 = output_dictionaries(sig)['lucyvb']  # rfit0.8  # ds510  # dlogz  # fullpriors  # lucyvb

hd, hl, li, tx, nl = table_it([direc + dict['pkl'], direc+dictr0['pkl'], direc+dictr1['pkl']],
                              [dict['outpf'], dictr0['outpf'], dictr1['outpf']],
                              [dict['mod'], dictr0['mod'], dictr1['mod']], tablabs, sig=sig, logged=True,
                              percent_diff=True, pris=pris, tabpars=tabpars, raf=ra_func, decf=dec_func)
print(li)
print(oop)

hd, hl, li, tx, nl = table_it([direc + dict['pkl']], [dict['outpf']], [dict['mod']], tablabs, sig=sig, logged=True,
                              percent_diff=True, pris=pris, tabpars=tabpars, raf=ra_func, decf=dec_func)
print(nl)
#print(hd)
#print(hl)
#print(li)
#print(tx)
print(oop)
# '''  #

out_name = direc + dict['pkl']

# https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
with open(out_name, 'rb') as pk:
    u = pickle._Unpickler(pk)
    u.encoding = 'latin1'
    dyn_res = u.load()  #
    # dyn_res = pickle.load(pk)  #
print(dyn_res['samples'].shape)

# PREPARE QUANTILES!
weights = np.exp(dyn_res['logwt'] - dyn_res['logz'][-1])  # normalized weights

# WRITE OUTFILE!
three_sigs = []
one_sigs = []
with open(dict['inpf'], 'r') as inpff:
    # outpf = dict['inpf'][:-4] + '_out.txt'
    outpf = dict['outpf']
    print(dict['inpf'], outpf)
    if not Path(outpf).exists():
        with open(outpf, 'w+') as outpff:
            idx = 0
            for line in inpff:
                if line.startswith('free'):
                    insert = str(dyfunc.quantile(dyn_res['samples'][:, idx], [0.0015, 0.5, 0.9985],
                                                 weights=weights)[1])
                    idx += 1
                    cols = line.split()
                    cols[2] = insert
                    line = ' '.join(cols) + '\n'
                outpff.write(line)

# CALCULATE QUANTILES!
for i in range(dyn_res['samples'].shape[1]):  # for each parameter
    quantiles_3 = dyfunc.quantile(dyn_res['samples'][:, i], [0.0015, 0.5, 0.9985], weights=weights)
    quantiles_2 = dyfunc.quantile(dyn_res['samples'][:, i], [0.025, 0.5, 0.975], weights=weights)
    quantiles_1 = dyfunc.quantile(dyn_res['samples'][:, i], [0.16, 0.5, 0.84], weights=weights)
    print(labels[i])
    if i == 0 and 'nobh' not in dict['pkl']:
        print(np.log10(quantiles_3), quantiles_3)
        print(np.log10(quantiles_2), quantiles_2)
        print(np.log10(quantiles_1), quantiles_1)
        three_sigs.append(np.log10(quantiles_3))
        one_sigs.append(np.log10(quantiles_1))
    else:
        print(quantiles_3)
        print(quantiles_2)
        print(quantiles_1)
        three_sigs.append(quantiles_3)
        one_sigs.append(quantiles_1)
from dynesty import plotting as dyplot

sig1 = [0.16, 0.5, 0.84]
sig2 = [0.025, 0.5, 0.975]
sig3 = [0.0015, 0.5, 0.9985]

sigs = three_sigs  # set default to 3sig
qts = sig3  # set default to 3sig

if sig == 1:
    qts = sig1
    sigs = one_sigs
elif sig == 3:
    qts = sig3
    sigs = three_sigs

logm = True
if logm and 'nobh' not in dict['pkl']:
    dyn_res['samples'][:, 0] = np.log10(dyn_res['samples'][:, 0])
    # labels[0] = 'log mbh'  # r'log$_{10}$mbh'
    # labels[0] = r'log$_{{10}}($M$_{\text{BH}})$'
    # labels[0] = '$\\log_{10}(M_{\\text{BH}})$'

ax_lims = None

if 'fullpriors' in out_name:
    ax_lims = [[6., 12.], [116., 140.], [140., 160.], [0., 200.], [52.4, 89.], [0., 89.], [5000., 8100.], [0.1, 10.],
               [0.1, 2.5]]
elif 'exp' in out_name:
    ax_lims = [[8., 10.], [124., 128.], [148, 152], [0., 100.], [52.4, 89], [5., 35.], [6405, 6505], [0.3, 3.],
               [0.5, 1.5], [0., 100.], [0., 100.]]  # [19.13, 19.23]
elif 'vrad' in out_name:
    ax_lims = [[8., 10.], [124., 128.], [148, 152], [0., 40.], [52.4, 89], [5., 35.], [6405, 6505], [0.3, 3.],
               [0.5, 1.5], [-50., 50.]]
elif 'kappa' in out_name:
    ax_lims = [[8., 10.], [124., 128.], [148, 152], [0., 40.], [52.4, 89], [5., 35.], [6405, 6505], [0.3, 3.],
               [0.5, 1.5], [-1., 1.]]
else:
    ax_lims = [[9.15, 9.65], [124., 128.], [148, 152], [0., 40.], [52.4, 89], [5., 35.], [6405, 6505], [0.3, 3.],
               [0.5, 1.5]]  # 8., 10.

import matplotlib as mpl
# mpl.rcParams['font.size'] = 20
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']  # for \text command
#mpl.rcParams['xtick.labelsize'] = 20
#mpl.rcParams['ytick.labelsize'] = 20

# ELSE USE THIS
# my_own_thing(dyn_res['samples'], labels, ax_lab, one_sigs, ax_lims=ax_lims, savefig=grp + dict['name'])
# print(oop)

# plot initial run (res1; left)

# TO EDIT SOURCE CODE: open /Users/jonathancohn/anaconda3/envs/three/lib/python3.6/site-packages/dynesty/plotting.py

# '''  # UNCOMMENT FOR CORNERFIG!
# MAKE CORNER PLOT
ndim = len(labels)
factor = 2.0  # size of side of one panel
lbdim = 0.5 * factor  # size of left/bottom margin
trdim = 0.2 * factor  # size of top/right margin
whspace = 0.05  # size of width/height margin
plotdim = factor * ndim + factor * (ndim - 1.) * whspace  # plot size
dim = lbdim + plotdim + trdim  # total size
fig, axes = plt.subplots(ndim, ndim, figsize=(1.7*dim, dim))
# fg, ax = dyplot.cornerplot(dyn_res, color='blue', show_titles=True, title_kwargs={'fontsize': 30}, max_n_ticks=3,
#                            quantiles=qts, labels=labels, label_kwargs={'fontsize': 30}, fig=(fig, axes))

orig_order = np.array(['mbh', 'xloc', 'yloc', 'sig', 'inc', 'pa', 'vsys', 'ml', 'f0'])
re_order = np.array([0, 7, 4, 5, 6, 3, 1, 2, 8])
# print(dyn_res['samples'].shape, dyn_res['logwt'].shape, dyn_res['logz'].shape)
#print(oop)
dyn_res['samples'] = dyn_res['samples'][:, re_order]
# print(dyn_res['samples'].shape)
#dyn_res['weights'] = dyn_res['weights'][re_order]
#dyn_res['logwt'] = dyn_res['logwt'][re_order]
#dyn_res['logz'] = dyn_res['logz'][re_order]
labels = labels[re_order]
cornunits = cornunits[re_order]
cornerax = cornerax[re_order]
#mpl.rcParams['xtick.labelsize'] = 50
#mpl.rcParams['ytick.labelsize'] = 50

fg, ax = mycorn(dyn_res, color='blue', show_titles=True, title_kwargs={'fontsize': 35}, max_n_ticks=3, tickfs=35,
                quantiles=qts, labels=cornerax, label_kwargs={'fontsize': 35}, fig=(fig, axes), shortlabs=labels,
                raf=ra_func, decf=dec_func)
# plt.savefig(grp + dict['cornername'])
plt.savefig(grp + 'finaltests/fiducial_corner_radec_rc35_xycentroidoffset.png')
plt.show()
# '''  #

'''  #
# ONLY table_it *AFTER* OUT FILE CREATED
hd, hl, li, tx = table_it([direc + dict['pkl']], [dict['outpf']], [dict['mod']], tablabs, sig=sig, logged=True,
                          percent_diff=True)
print(hd)
print(hl)
print(li)
print(tx)
# '''  #
