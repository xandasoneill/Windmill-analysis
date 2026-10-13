"""
Generic diagrams for beam-like parts (shafts, axles, ...). Nothing here depends on a specific part:
every function takes plain arrays, in SI units (N, N·m, m), and plots them in kN / kN·m.
"""
import matplotlib

# Qt backend (matplotlib default here) crashes on this machine (GLib/GNOME settings error),
# so use Tk. Falls back to the default backend if Tk is not available.
try:
    matplotlib.use("TkAgg")
except ImportError:
    pass
import matplotlib.pyplot as plt

COMPONENT_COLORS = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
MARKER_COLORS = ['red', 'orange', 'purple', 'green', 'brown']


def _mark_positions(ax, markers, legend):
    """Dashed zero line, grid and vertical dotted lines at the marked positions {label: z}."""
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    for (label, z_pos), color in zip((markers or {}).items(), MARKER_COLORS):
        ax.axvline(z_pos, color=color, linestyle=':', label=label if legend else None)
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_diagram(ax, z, components, resultant=None, ylabel='', title='',
                 markers=None, mark_legend=False, resultant_color='#1f77b4',
                 max_point=None, unit='kN', legend_loc='best'):
    """
    Plot one diagram (shear or moment) on `ax`.

    z          : positions along the part [m]
    components : {label: array} of the components (e.g. {"V_x": ..., "V_y": ...}) [SI units]
    resultant  : optional array with the resultant of the components
    markers    : {label: z position} to mark (supports, gears, ...)
    max_point  : optional (z, value) to highlight, e.g. the maximum moment
    unit       : unit shown in the max_point label ('kN' or 'kN·m')
    """
    _mark_positions(ax, markers, mark_legend)
    for (label, values), color in zip(components.items(), COMPONENT_COLORS):
        ax.plot(z, values / 1000, color=color, linewidth=1.5, label=f'{label}(z)')
    if resultant is not None:
        ax.plot(z, resultant / 1000, color=resultant_color, linewidth=2, label='Resultant (z)')
    if max_point is not None:
        z_max, value_max = max_point
        ax.plot(z_max, value_max / 1000, 'ko', markersize=8,
                label=f'max = {value_max/1000:.2f} {unit}')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12)
    ax.legend(loc=legend_loc)


def plot_shear_moment(z, shear, moment, shear_res=None, moment_res=None,
                      markers=None, max_point=None, title='', show=True):
    """
    Shear force and bending moment diagrams, stacked and sharing the z axis.

    z          : positions along the part [m]
    shear      : {label: array} shear components [N]
    moment     : {label: array} bending moment components [N·m]
    shear_res, moment_res : optional resultant arrays
    markers    : {label: z position} to mark in both diagrams
    max_point  : optional (z, M) of the maximum bending moment to highlight
    Returns the figure and the two axes.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plot_diagram(ax1, z, shear, shear_res, ylabel='Shear Force V [kN]',
                 title='Shear Force Diagram (SFD)', markers=markers, mark_legend=True,
                 legend_loc='upper right')
    plot_diagram(ax2, z, moment, moment_res, ylabel='Bending Moment M [kN·m]',
                 title='Bending Moment Diagram (BMD)', markers=markers,
                 resultant_color='#d62728', max_point=max_point, unit='kN·m',
                 legend_loc='lower right')
    ax2.set_xlabel('Position along shaft z-axis [m]', fontsize=11, fontweight='bold')

    plt.tight_layout()
    if show:
        plt.show()
    return fig, (ax1, ax2)
