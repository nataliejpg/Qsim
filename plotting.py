import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
from matplotlib import cm
from .state_vectors.helpers import projection


def plot_projections(state_array, t=None):
    """
    Function which plots the projection of a state vector onto x, and z axes

    Args:
        state_array (shape (m, 2**n): state vector(s) at each time where n is
            qubit_num
        time array (optional) (shape (m,))

    Returns:
        fig with subplots (2d if m and n are both > 1, )
    """
    state_array = np.array(state_array)
    if t is None:
        t = np.arange(state_array.shape[0])

    fig = plt.figure()
    gs = gridspec.GridSpec(3, 2, width_ratios=(9.5, 0.5))

    ax1 = plt.subplot(gs[0, 0])
    cax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 0])
    cax2 = plt.subplot(gs[1, 1])
    ax3 = plt.subplot(gs[2, 0])
    cax3 = plt.subplot(gs[2, 1])
    ax1.set_title('x projection')
    ax1.set_ylim([-1.1, 1.1])
    ax2.set_title('y projection')
    ax2.set_ylim([-1.1, 1.1])
    ax3.set_title('z projection')
    ax3.set_ylim([-1.1, 1.1])

    x_projections = projection(state_array, axis='X')
    y_projections = projection(state_array, axis='Y')
    z_projections = projection(state_array, axis='Z')

    if state_array.shape[1] > 2:
        qubit_nums = np.arange(int(np.log2(state_array.shape[1])))
        plot2d(t, qubit_nums, x_projections, ax1,
               cax1, cbarlimits='x_projection')
        plot2d(t, qubit_nums, y_projections, ax2,
               cax2, cbarlimits='y_projection')
        plot2d(t, qubit_nums, z_projections, ax3,
               cax3, cbarlimits='z_projection')
        for ax in [ax1, ax2, ax3]:
            ax.set_ylabel('qubit index')
    else:
        plot1d(t, x_projections, ax1)
        plot1d(t, y_projections, ax2)
        plot1d(t, z_projections, ax3)
        ax1.set_ylabel('x projection')
        ax2.set_ylabel('y projection')
        ax3.set_ylabel('z projection')

    ax1.set_xlabel('t')
    ax2.set_xlabel('t')
    ax3.set_xlabel('t')

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
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    pc = ax.pcolor(x, y, z, cmap=cm.hot)

    # colorbar
    clb = plt.colorbar(pc, cax=cax)
    if 'cbarlimits' in kwargs:
        if type(kwargs['cbarlimits']) is bool:
            clb.set_label('P(1)', labelpad=-40, y=1.05, rotation=0)
            pc.set_clim([-0.1, 1.1])
        else:
            clb.set_label(kwargs['cbarlimits'],
                          labelpad=-40, y=1.05, rotation=0)
            pc.set_clim([-1.1, 1.1])

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
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
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
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
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
