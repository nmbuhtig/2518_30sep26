#!/usr/bin/env python3


import json
import numpy as np
import matplotlib.pyplot as plt


OUTPUT_PREFIX = "figure1_single_event_standalone_v5"


EMBEDDED_DATA = {
    "input_state": {"p_in": 0, "l_in": 0, "sigma_in": 1},
    "collection": {
        "eta_col": 0.1,
        "NA_col": 0.1,
        "theta_col_deg": 5.739170477266787,
        "sigma_col_direct_um2": 6.9079496335332955,
        "sigma_col_fft_um2": 6.907949633533293,
        "fft_parseval_relative_error": 3.857201341141673e-16,
        "sigma_col_selected_l_um2": 6.9079492759885355,
        "selected_l_capture_fraction": 0.9999999482415508,
        "absolute_collected_fraction": 0.26857842145758537,
    },
    "branching": {
        "jbreak_um2": 4.173202817608378,
        "spinflip_um2": 0.013075592545905432,
        "compensated_so_um2": 0.0014877421212514124,
        "jbreak_fraction_all_scattering": 0.1622525184223614,
        "spinflip_fraction_all_scattering": 0.0005083740026931284,
        "compensated_so_fraction_all_scattering": 5.78428407355495e-05,
        "jbreak_fraction_collected": 0.6041159879554387,
        "spinflip_fraction_collected": 0.0018928326405902723,
        "compensated_so_fraction_collected": 0.00021536667175878904,
        "centered_selected_probability": 0.15323013795942508,
        "centered_jbreak_probability": 2.5578247250554053e-33,
        "centered_jbreak_fraction": 1.669270000744051e-32,
    },
    "l_values": list(range(-12, 13)),
    "sigma_plus_conditional_collected": [
        1.2381679540113156e-07,
        6.770180613892906e-07,
        3.428643673825096e-06,
        1.6082907239319392e-05,
        6.972600531409445e-05,
        0.00027852287997237303,
        0.0010217661304727755,
        0.003432663707139605,
        0.01054071767679904,
        0.029576448846529643,
        0.07600279261291487,
        0.17987804754168502,
        0.39566859361435347,
        0.1801708062353963,
        0.07625047677402251,
        0.02972118504237577,
        0.010609565003995016,
        0.0034607175388888635,
        0.0010317965529714453,
        0.0002817158175672592,
        7.064036901100998e-05,
        1.6320393355062342e-05,
        3.484950422433136e-06,
        6.892602787995981e-07,
        1.262617254762026e-07,
    ],
    "sigma_minus_conditional_collected": [
        3.728776517073579e-09,
        1.6893846370520083e-08,
        7.067595950573309e-08,
        2.726828233967361e-07,
        9.668213365721213e-07,
        3.133354362047342e-06,
        9.219701469315249e-06,
        2.4430748046780973e-05,
        5.7716833273117195e-05,
        0.00011996655780952265,
        0.0002152284456188981,
        0.000323099799351772,
        0.00038339315307628927,
        0.0003233276460514337,
        0.00021536667175878904,
        0.00012030826648464727,
        5.795564352362021e-05,
        2.4565593792989147e-05,
        9.284031411249893e-06,
        3.160015328150713e-06,
        9.766185429523348e-07,
        2.759563886362189e-07,
        7.171982286149123e-08,
        1.7234706377695754e-08,
        3.847028459298471e-09,
    ],
}


FIGURE_SETTINGS = {
    "figsize": (7.4, 3.7),
    "dpi_png": 450,
    "tight_layout_pad": 0.35,
    "wspace": 0.42,
    "hspace": 0.18,
}


PANEL_A = {
    "x_label": r"Output OAM index $l′$",
    "y_label": r"Collected probability",
    "x_label_fontsize": 9.0,
    "y_label_fontsize": 9.0,
    "x_tick_fontsize": 8.0,
    "y_tick_fontsize": 8.0,
    "legend_fontsize": 7.5,
    "line_width": 1.0,
    "marker_size": 3.8,
    "bar_width": 0.42,
    "legend_loc": "lower center",
    "legend_bbox_to_anchor": (0.5, 1.0),
    "panel_label": "(a)",
    "panel_label_fontsize": 10.0,
    "panel_label_color": "black",
    "panel_label_position": (0.02, 0.98),
    "x_limits": (-12.8, 12.8),
    "y_limits": (1.0e-9, 1.0),
    "y_scale": "log",
    "show_grid": False,
    "same_helicity_label": r"$\sigma′=+1$",
    "flip_helicity_label": r"$\sigma′=-1$",
}

PANEL_B = {
    "x_label": r"Output OAM index $l′$",
    "y_label": r"$10^3 P(l′,\sigma′=-1\mid\mathrm{col})$",
    "x_label_fontsize": 9.0,
    "y_label_fontsize": 9.0,
    "x_tick_fontsize": 8.0,
    "y_tick_fontsize": 8.0,
    "legend_fontsize": 7.5,
    "line_width": 1.0,
    "marker_size": 4.0,
    "bar_width": 0.70,
    "legend_loc": "lower center",
    "legend_bbox_to_anchor": (0.5, 1.0),
    "panel_label": "(b)",
    "panel_label_fontsize": 10.0,
    "panel_label_color": "black",
    "panel_label_position": (0.02, 0.98),
    "x_limits": (-12.8, 12.8),
    "y_limits": (0.0, 0.43),
    "show_grid": False,
    "scale_factor": 1.0e3,
    "compensated_l": 2,
    "all_flipped_label": r"$l′\neq2$",
    "compensated_label": r"$l′=2$",
    "annotation_text": "$|0,+1\\rangle\\rightarrow$\n$|2,-1\\rangle$",
    "annotation_fontsize": 6.0,
    "annotation_position": (4.3, 0.31),
}


def _apply_common_axis_style(ax, style):
    ax.set_xlabel(style["x_label"], fontsize=style["x_label_fontsize"])
    ax.set_ylabel(style["y_label"], fontsize=style["y_label_fontsize"])
    ax.tick_params(axis="x", labelsize=style["x_tick_fontsize"])
    ax.tick_params(axis="y", labelsize=style["y_tick_fontsize"])
    if style.get("x_limits") is not None:
        ax.set_xlim(*style["x_limits"])
    if style.get("y_limits") is not None:
        ax.set_ylim(*style["y_limits"])
    if style.get("y_scale") is not None:
        ax.set_yscale(style["y_scale"])
    if style.get("show_grid", False):
        ax.grid(True, alpha=0.3)
    else:
        ax.grid(False)
    ax.text(
        style["panel_label_position"][0],
        style["panel_label_position"][1],
        style["panel_label"],
        transform=ax.transAxes,
        fontsize=style["panel_label_fontsize"],
        color=style["panel_label_color"],
        ha="left",
        va="top",
    )


def _panel_a(ax, style):
    l_values = np.asarray(EMBEDDED_DATA["l_values"], dtype=float)
    y_plus = np.asarray(
        EMBEDDED_DATA["sigma_plus_conditional_collected"], dtype=float
    )
    y_minus = np.asarray(
        EMBEDDED_DATA["sigma_minus_conditional_collected"], dtype=float
    )
    w = style["bar_width"]

    ax.bar(
        l_values - 0.5 * w,
        y_plus,
        width=w,
        linewidth=style["line_width"],
        label=style["same_helicity_label"],
    )
    ax.bar(
        l_values + 0.5 * w,
        y_minus,
        width=w,
        linewidth=style["line_width"],
        label=style["flip_helicity_label"],
    )

    _apply_common_axis_style(ax, style)

    if style["legend_bbox_to_anchor"] is None:
        ax.legend(
            frameon=False,
            fontsize=style["legend_fontsize"],
            loc=style["legend_loc"],
        )
    else:
        ax.legend(
            frameon=False,
            fontsize=style["legend_fontsize"],
            loc=style["legend_loc"],
            bbox_to_anchor=style["legend_bbox_to_anchor"],
            ncol=2,
            handlelength=1.2,
            columnspacing=1.0,
        )


def _panel_b(ax, style):
    l_values = np.asarray(EMBEDDED_DATA["l_values"], dtype=float)
    y_minus = (
        np.asarray(
            EMBEDDED_DATA["sigma_minus_conditional_collected"], dtype=float
        )
        * style["scale_factor"]
    )

    compensated_l = int(style["compensated_l"])
    compensated_index = int(np.where(l_values == compensated_l)[0][0])

    mask = np.ones(l_values.size, dtype=bool)
    mask[compensated_index] = False

    ax.bar(
        l_values[mask],
        y_minus[mask],
        width=style["bar_width"],
        linewidth=style["line_width"],
        label=style["all_flipped_label"],
    )
    ax.bar(
        [l_values[compensated_index]],
        [y_minus[compensated_index]],
        width=style["bar_width"],
        linewidth=style["line_width"],
        label=style["compensated_label"],
    )

    _apply_common_axis_style(ax, style)

    ax.annotate(
        style["annotation_text"],
        xy=(l_values[compensated_index] + 0.35, y_minus[compensated_index]),
        xytext=style["annotation_position"],
        textcoords="data",
        fontsize=style["annotation_fontsize"],
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "->", "linewidth": 0.7},
    )

    if style["legend_bbox_to_anchor"] is None:
        ax.legend(
            frameon=False,
            fontsize=style["legend_fontsize"],
            loc=style["legend_loc"],
        )
    else:
        ax.legend(
            frameon=False,
            fontsize=style["legend_fontsize"],
            loc=style["legend_loc"],
            bbox_to_anchor=style["legend_bbox_to_anchor"],
            ncol=2,
            handlelength=1.2,
            columnspacing=1.0,
        )


FONT_SCALE = 1.5
plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "mathtext.fontset": "custom",
    "mathtext.rm": "DejaVu Sans:bold",
    "mathtext.it": "DejaVu Sans:italic:bold",
    "mathtext.bf": "DejaVu Sans:bold",
    "mathtext.cal": "DejaVu Sans:bold",
    "mathtext.sf": "DejaVu Sans:bold",
})


def _scale_fonts(style):
    for key in list(style):
        if key.endswith("fontsize"):
            style[key] = style[key] * FONT_SCALE


def main():
    _scale_fonts(PANEL_A)
    _scale_fonts(PANEL_B)
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SETTINGS["figsize"])
    fig.subplots_adjust(
        wspace=FIGURE_SETTINGS["wspace"],
        hspace=FIGURE_SETTINGS["hspace"],
    )

    _panel_a(axes[0], PANEL_A)
    _panel_b(axes[1], PANEL_B)

    out_png = f"{OUTPUT_PREFIX}.png"
    out_pdf = f"{OUTPUT_PREFIX}.pdf"
    out_json = f"{OUTPUT_PREFIX}.json"

    fig.savefig(
        out_png,
        dpi=FIGURE_SETTINGS["dpi_png"],
        bbox_inches="tight",
        pad_inches=FIGURE_SETTINGS["tight_layout_pad"],
    )
    fig.savefig(
        out_pdf,
        bbox_inches="tight",
        pad_inches=FIGURE_SETTINGS["tight_layout_pad"],
    )
    plt.close(fig)

    export = {
        "figure_name": "Figure 1",
        "standalone": True,
        "revision": "v5 presentation only; numerical data unchanged",
        "frozen_content": {
            "panel_a": (
                "single-event collected spin-resolved OAM distribution "
                "on logarithmic probability scale"
            ),
            "panel_b": (
                "magnified helicity-flipped OAM distribution with "
                "Jz-compensated l'=2 channel highlighted"
            ),
        },
        "output_files": {
            "png": out_png,
            "pdf": out_pdf,
            "json": out_json,
        },
        "figure_settings": FIGURE_SETTINGS,
        "panel_styles": {
            "panel_a": PANEL_A,
            "panel_b": PANEL_B,
        },
        "embedded_data": EMBEDDED_DATA,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
