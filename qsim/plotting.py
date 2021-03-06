import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import gridspec
from matplotlib import cm
from qsim.exact.methods import projection


def plot_projections(state_array, x_axis=None, y_axis=None, qubit_index=None,
                     projection_axes=None, axes_labels=None, title=None):
    """
    Function which plots the projection of a state vector onto x, and z axes

    Args:
        state_array (max shape (m, n, 2**l), min shape (n, 2**l):
            m = inputs
            n = times
            l = qubits

        x_axis array (optional) (shape (n,))
        y_axis array (optional) (shape (m,))
        qubit_index (optional) qubit to plot for, if none specified will plot
            all qubits
        projection_axes (optional), which axes to plot along, default [X, Y, Z]
        axes_labels (optional), what to label the xy axes, default
            ['time', 'input', 'qubit']

    Returns:
        fig with subplots (2d if 2 of m, n and l are > 1, )
    """
    projection_axes = projection_axes or ['X', 'Y', 'Z']
    projection_axes = [p.upper() for p in projection_axes]
    state_array = np.array(state_array)

    fig = plt.figure()
    gs = gridspec.GridSpec(len(projection_axes), 2, width_ratios=(9.5, 0.5))

    for i, p in enumerate(projection_axes):
        if p.upper() not in ['X', 'Y', 'Z']:
            raise RuntimeError('projection_axes must be in [X, Y, Z], '
                               'got {}'.format(projection_axes))

    axes_labels = axes_labels or ['time', 'input', 'qubit']
    if len(state_array.shape) > 2 and 1 in state_array.shape:
        redundant_index = state_array.shape.index(1)
        new_shape = tuple([s for s in state_array.shape if s != 1])
        state_array = state_array.reshape(new_shape)
        if redundant_index == 0:
            try:
                axes_labels.remove('input')
            except ValueError:
                pass
        if redundant_index == 1:
            try:
                axes_labels.remove('time')
            except ValueError:
                pass
    elif len(state_array.shape) == 2:
        try:
            axes_labels.remove('input')
        except ValueError:
                pass
    if state_array.shape[-1] == 2 or qubit_index is not None:
        try:
            axes_labels.remove('qubit')
        except ValueError:
            pass
        if state_array.shape[-1] == 2 and qubit_index is None:
            qubit_index = 0

    if len(axes_labels) > 2:
        raise RuntimeError(
            'cannot plot 3d, array provided has multiple qubits, inputs'
            ' and times, shape {}. Specify qubit ot slice array'
            ''.format(state_array.shape))
    elif len(axes_labels) == 0:
        raise RuntimeError(
            'why are you here {}'.format(state_array.shape))

    projection_values = [[]] * len(projection_axes)
    if 'qubit' not in axes_labels and len(axes_labels) == 2:  # [time, input]
        if x_axis is None:
            x_axis = np.arange(state_array.shape[1])
        if y_axis is None:
            y_axis = np.arange(state_array.shape[0] + 1)
        elif len(y_axis) == state_array.shape[0]:
            y_axis = np.append(y_axis, y_axis[-1] + (y_axis[-1] - y_axis[-2]))
        for i in range(len(projection_axes)):
            projection_values[i] = np.zeros(
                (state_array.shape[1], state_array.shape[0]), dtype=complex)
        for i, state in enumerate(state_array):
            for j, p in enumerate(projection_axes):
                projection_values[j][:, i] = projection(
                    state, axis=p)[:, qubit_index]
    elif len(axes_labels) == 2:  # [time, qubit], [input, qubit]
        if x_axis is None:
            x_axis = np.arange(state_array.shape[0])
        if y_axis is None:
            y_axis = np.arange(np.log2(state_array.shape[1]) + 1)
        elif len(y_axis) == state_array.shape[0]:
            y_axis = np.append(y_axis, y_axis[-1] + (y_axis[-1] - y_axis[-2]))
        for j, p in enumerate(projection_axes):
            projection_values[j] = projection(state_array, axis=p)
    elif 'qubit' not in axes_labels:  # [input], [time]
        if x_axis is None:
            x_axis = np.arange(state_array.shape[0])
        for j, p in enumerate(projection_axes):
            projection_values[j] = projection(
                state_array, axis=p)[:, qubit_index]
    else:  # [qubit]
        if x_axis is None:
            x_axis = np.arange(np.log2(len(state_array)))
        state_array = [state_array]
        for j, p in enumerate(projection_axes):
            projection_values[j] = projection(state_array, axis=p)

    for i, p in enumerate(projection_axes):
        proj = projection_values[i]
        ax = plt.subplot(gs[i, 0])
        if len(proj.shape) == 1:
            plot1d(x_axis, np.real(proj), ax)
            ax.set_ylim([-1.1, 1.1])
            ax.set_ylabel('{}_proj'.format(p))
            ax.set_xlabel(axes_labels[0])
        else:
            x_axis_2d, y_axis_2d = np.meshgrid(x_axis, y_axis)
            cax = plt.subplot(gs[i, 1])
            plot2d(x_axis_2d, y_axis_2d, np.real(proj.T), ax, cax,
                   cbarlimits='{}_proj'.format(p))
            ax.yaxis.set_major_formatter(
                ticker.ScalarFormatter(useMathText=True))
            ax.set_xlabel(axes_labels[0])
            ax.set_ylabel(axes_labels[1])
        ax.set_title((title or '') + '{} projection'.format(p))

    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    return fig


def build_title(**kwargs):
    t = ''
    if 'datanum' in kwargs:
        t += str(kwargs['datanum']) + '   '
    if 'amp' in kwargs:
        t += 'Amp: {0:5.3g}'.format(kwargs['amp'])
    if 'mod_freq' in kwargs:
        t += ',  Mod Freq {0:5.3g}'.format(kwargs['mod_freq'])
    return t


def plot2d(x, y, z, ax, cax, **kwargs):

    # second graph
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    pc = ax.pcolor(x, y, z, cmap=cm.hot)

    # colorbar
    clb = plt.colorbar(pc, cax=cax)
    if 'cbarlimits' in kwargs:
        if type(kwargs['cbarlimits']) is bool:
            clb.set_label('P(1)', labelpad=-40, y=1.05, rotation=0)
            pc.set_clim([-0.1, 1.1])
            clb.set_ticks([0, 0.5, 1])
        else:
            clb.set_label(kwargs['cbarlimits'],
                          labelpad=-40, y=1.05, rotation=0)
            pc.set_clim([-1.1, 1.1])
            clb.set_ticks([-1, 0, 1])

    # titles
    t = build_title(**kwargs)
    plt.setp(ax, title=t)

    # axes labelling
    if 'detuning' in kwargs:
        ax.set_ylabel('detuning (Hz)')
    if 'xticklabels' in kwargs:
        if not kwargs['xticklabels']:
            plt.setp(ax.get_xticklabels(), visible=False)


def plot1d(x, y, ax, **kwargs):

    ax.plot(x, y)
    if 'drive_mod' in kwargs:
        plt.setp(ax, title='qubit_drive_modulation')
        ax.set_ylabel('drive modulation')
    if 'xticklabels' in kwargs:
        if not kwargs['xticklabels']:
            plt.setp(ax.get_xticklabels(), visible=False)


def plot_data(datax, datay, dataz, **kwargs):
    """
    kwarg options:
        datanum
        mod_freq
        amp
        zero_mag
        one_mag
        title
    """
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=(9.5, 0.5))
    ax1 = plt.subplot(gs[0, 0])
    cax1 = plt.subplot(gs[0, 1])
    if all(k in kwargs for k in ['zero_mag', 'one_mag']):
        data_cbar_lim = True
    elif 'sim' in kwargs:
        data_cbar_lim = kwargs['sim']
    else:
        data_cbar_lim = False
    plot2d(datax, datay, dataz, ax1, cax1,
           xticklabels=True, cbarlimits=data_cbar_lim,
           detuning=True, **kwargs)
    ax1.set_xlabel('drive duration (s)')
    if 'title' in kwargs:
        plt.suptitle(kwargs['title'], fontsize=15)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    return fig


def plot_sim(simx, simy, simz, **kwargs):
    """
    kwarg options:
        datanum
        mod_freq
        amp
        title
    """
    kwargs['sim'] = True
    return plot_data(simx, simy, simz, **kwargs)


def plot_data_mod(datax, datay, dataz, mod_array, **kwargs):
    """
    kwarg options:
        datanum
        mod_freq
        amp
        zero_mag
        one_mag
        title
    """
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, height_ratios=(3, 1), width_ratios=(9.5, 0.5))
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)
    cax1 = plt.subplot(gs[0, 1])
    if all(k in kwargs for k in ['zero_mag', 'one_mag']):
        data_cbar_lim = True
    elif 'sim' in kwargs:
        data_cbar_lim = kwargs['sim']
    else:
        data_cbar_lim = False
    plot2d(datax, datay, dataz, ax1, cax1, xticklabels=False,
           cbarlimits=data_cbar_lim, detuning=True, **kwargs)
    plot1d(datax, mod_array, ax2, drive_mod=True, xticklabels=True)
    ax2.set_xlabel('drive duration (s)')

    if 'title' in kwargs:
        fig.suptitle(kwargs['title'], fontsize=15)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    return fig


def plot_sim_mod(simx, simy, simz, mod_array, **kwargs):
    """
    kwarg options:
        datanum
        mod_freq
        amp
        title
    """
    kwargs['sim'] = True
    return plot_data_mod(simx, simy, simz, mod_array, **kwargs)


def plot_data_sim_mod(datax, datay, dataz, simx, simy, simz, mod_array,
                      **kwargs):
    """
    kwarg options:
        title
        awg_amp
        sim_amp
        mod_freq
        zero_mag
        one_mag
        datanum
    """
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(3, 2, height_ratios=(3, 3, 1),
                           width_ratios=(9.5, 0.5))
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0], sharex=ax1)
    ax3 = plt.subplot(gs[2, 0], sharex=ax1)
    cax1 = plt.subplot(gs[0, 1])
    cax2 = plt.subplot(gs[1, 1])

    if all(k in kwargs for k in ['zero_mag', 'one_mag']):
        data_cbar_lim = True
    else:
        data_cbar_lim = False
    if 'amp' in kwargs:
        raise Exception('set sim_amp and awg_amp separately')
    if 'awg_amp' in kwargs:
        kwargs['amp'] = kwargs.pop('awg_amp')
    plot2d(datax, datay, dataz, ax1, cax1,
           xticklabels=False, cbarlimits=data_cbar_lim,
           detuning=True, **kwargs)
    if 'sim_amp' in kwargs:
        kwargs['amp'] = kwargs.pop('sim_amp')
    else:
        try:
            kwargs.pop('amp')
        except KeyError:
            pass
    plot2d(simx, simy, simz, ax2, cax2,
           cbarlimits=True, xticklabels=False,
           detuning=True, **kwargs)
    plot1d(datax, mod_array, ax3, drive_mod=True, xticklabels=True)
    ax3.set_xlabel('drive duration (s)')

    if 'title' in kwargs:
        fig.suptitle(kwargs['title'], fontsize=13)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    return fig


def plot_data_stroboscopic(x, y, z, sr, **kwargs):
    if 'mod_freq' not in kwargs:
        raise Exception('must specify mod_freq in kwargs to make '
                        'stroboscopic plot')
    spacing = int(sr / kwargs['mod_freq'])
    x_points = x[::spacing]
    x_points = np.append(x_points, x_points[-1] + 1 / kwargs['mod_freq'])
    z_points = z[:, ::spacing]
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=(9.5, 0.5))

    ax = plt.subplot(gs[0, 0])
    cax = plt.subplot(gs[0, 1])
    if all(k in kwargs for k in ['zero_mag', 'one_mag']):
        data_cbar_lim = True
    elif 'sim' in kwargs:
        data_cbar_lim = kwargs['sim']
    else:
        data_cbar_lim = False
    plot2d(x_points, y, z_points, ax, cax,
           cbarlimits=data_cbar_lim,
           detuning=True, **kwargs)
    ax.set_xlabel('drive duration (s)')

    if 'title' in kwargs:
        fig.suptitle(kwargs['title'], fontsize=13)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    return fig


def plot_sim_stroboscopic(x, y, z, sr, **kwargs):
    kwargs['sim'] = True
    return plot_data_stroboscopic(x, y, z, sr, **kwargs)


def plot_data_sim_stroboscopic(datax, datay, dataz, simx, simy, simz, sr,
                               **kwargs):
    if 'mod_freq' not in kwargs:
        raise Exception('must specify mod_freq in kwargs to make '
                        'stroboscopic plot')
    spacing = int(sr / kwargs['mod_freq'])
    x_datapoints = datax[::spacing]
    x_datapoints = np.append(
        x_datapoints, x_datapoints[-1] + 1 / kwargs['mod_freq'])
    z_datapoints = dataz[:, ::spacing]
    x_simpoints = simx[::spacing]
    x_simpoints = np.arange(len(x_datapoints))
    z_simpoints = simz[:, ::spacing]
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=(9.5, 0.5))

    ax1 = plt.subplot(gs[0, 0])
    cax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 0])
    cax2 = plt.subplot(gs[1, 1])
    if all(k in kwargs for k in ['zero_mag', 'one_mag']):
        data_cbar_lim = True
    else:
        data_cbar_lim = False
    if 'amp' in kwargs:
        raise Exception('set sim_amp and awg_amp separately')
    if 'awg_amp' in kwargs:
        kwargs['amp'] = kwargs.pop('awg_amp')
    plot2d(x_datapoints, datay, z_datapoints, ax1, cax1,
           cbarlimits=data_cbar_lim, xticklabels=True,
           detuning=True, **kwargs)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    ax1.set_xlabel('drive duration (s)')
    if 'sim_amp' in kwargs:
        kwargs['amp'] = kwargs.pop('sim_amp')
    else:
        try:
            kwargs.pop('amp')
        except KeyError:
            pass
    plot2d(x_simpoints, simy, z_simpoints, ax2, cax2,
           cbarlimits=data_cbar_lim, xticklabels=True,
           detuning=True, **kwargs)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
    ax2.set_xlabel('stroboscopic periods')
    if len(x_datapoints) < 14:
        ax1.set_xticks(x_datapoints)
        ax2.set_xticks(x_simpoints)
    else:
        ax1.set_xticks(x_datapoints[::4])
        ax2.set_xticks(x_simpoints[::2])

    if 'title' in kwargs:
        fig.suptitle(kwargs['title'], fontsize=13)
    gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
    return fig
