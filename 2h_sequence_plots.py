# python 2h_sequence_plots.py

# ========= IMPORTS =========
import io
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cartopy.crs as ccrs
from cartopy import geodesic
from seismostats.analysis import (ClassicBValueEstimator,
                                  BPositiveBValueEstimator,
                                  estimate_mc_maxc)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from seismostats.plots import plot_cum_fmd, plot_mags_in_time, plot_in_space

from functions.main_functions import find_sequences, load_catalog

# ======== SPECIFY PARAMETERS ===
RESULT_DIR = Path("results/sequence_images")

# ========= LOAD PARAMETERS =========
DIR = Path("data")
variables_df = pd.read_csv(DIR / "variables.csv")
variables = variables_df.to_dict(orient="records")[0]

SHAPE_DIR = Path(variables["SHAPE_DIR"])
CAT_DIR = Path(variables["CAT_DIR"])

# b-val estimation
MC_FIXED = variables["MC_FIXED"]
CORRECTION_FACTOR = variables["CORRECTION_FACTOR"]
DELTA_M = variables["DELTA_M"]
DMC = variables["DMC"]
MIN_N_M = variables["MIN_N_M"]

# transformation to local
EPSG_JAPAN_M = variables["EPSG_JAPAN_M"]

# sequences
DIMENSION = variables["DIMENSION"]
BUFFER_M = variables["BUFFER_M"]
MAGNITUDE_THRESHOLD = variables["MAGNITUDE_THRESHOLD"]
RUPTURE_RELATION = variables["RUPTURE_RELATION"]
DAYS_AFTER = variables["DAYS_AFTER"]
DAYS_BEFORE = variables["DAYS_BEFORE"]
RADIUS_FAR = variables["RADIUS_FAR"]
RADIUS_CLOSE = variables["RADIUS_CLOSE"]
EXCLUDE_AFTERSHOCK_DAYS = variables["EXCLUDE_AFTERSHOCK_DAYS"]
MIN_N_SEQ = variables["MIN_N_SEQ"]

# ========= HELPERS =========


def add_min_max_mag_rows(df, min_mag, max_mag):
    # Create copies of a single empty row so dtypes are preserved
    empty_row = df.iloc[0:0].copy()

    # Create new rows
    min_row = empty_row.copy()
    min_row.loc[0, :] = np.nan
    min_row.loc[0, 'magnitude'] = min_mag

    max_row = empty_row.copy()
    max_row.loc[0, :] = np.nan
    max_row.loc[0, 'magnitude'] = max_mag

    # Concatenate
    return pd.concat([df, min_row, max_row], ignore_index=True)


def define_relative_seqs(sequence,
                         sequence_main_idx,
                         cat_all,
                         exclude_aftershock_days):

    # get events before and after main event
    min_mag = sequence.magnitude.min()
    max_mag = sequence.magnitude.max()
    main = cat_all[cat_all.index == sequence_main_idx].copy()

    before = sequence[sequence.time.values < main.time.values[0]]
    after = sequence[sequence.time >=
                     main.time.values[0] +
                     pd.Timedelta(days=exclude_aftershock_days)]
    after_immediate = sequence[sequence.time > main.time.values[0]]
    after_immediate = after_immediate[
        after_immediate.time.values < main.time.values[0]
        + pd.Timedelta(days=exclude_aftershock_days)]

    # scale correctly
    main_x = add_min_max_mag_rows(main, min_mag, max_mag)
    before = add_min_max_mag_rows(before, min_mag, max_mag)
    after = add_min_max_mag_rows(after, min_mag, max_mag)
    after_immediate = add_min_max_mag_rows(after_immediate, min_mag, max_mag)

    return before, after, after_immediate, main_x, main


def get_farclose_beforeafter(times,
                             mags,
                             distances,
                             main_event,
                             exclude_aftershock_days,
                             radius_close):
    # estimate indexes
    idx_close = distances <= radius_close * main_event.rupture_length.values[0]
    idx_main = times == main_event.time.values[0]

    idx_before = times < main_event.time.values[0]
    idx_after = times > main_event.time.values[0] + \
        pd.Timedelta(days=exclude_aftershock_days)

    mags_close = mags[idx_close & ~idx_main]
    mags_far = mags[~idx_close & ~idx_main]
    mags_before = mags[idx_before]
    mags_after = mags[idx_after]

    return (mags_close, mags_far, mags_before,  mags_after)


# ========= PLOTTING FUNCTIONS =========

# Space
def create_map(sequence, sequence_main_idx, cat_all, exclude_aftershock_days):
    fig_map = plt.figure(figsize=(8, 8))

    before, after, after_immediate, main_x, main = define_relative_seqs(
        sequence, sequence_main_idx, cat_all, exclude_aftershock_days)
    # plot mags
    ax = plot_in_space(before.longitude, before.latitude,
                       before.magnitude, color_dots='cornflowerblue')
    plot_in_space(after.longitude, after.latitude,
                  after.magnitude, ax=ax, color_dots='darkseagreen')
    plot_in_space(after_immediate.longitude, after_immediate.latitude,
                  after_immediate.magnitude, ax=ax, color_dots='indianred')
    plot_in_space(main_x.longitude, main_x.latitude,
                  main_x.magnitude, ax=ax, color_dots='yellow',
                  dot_labels=[2, 4, 6])

    # Plot the data points
    # Coastline
    ax.plot(df_coast['longitude'], df_coast['latitude'], 'k',
            linewidth=0.5, transform=ccrs.PlateCarree(), color='grey')

    # Geodesic circles
    r1 = main.rupture_length.values / 2 * 1000
    r2 = main.rupture_length.values * 2 * 1000
    circle1 = geodesic.Geodesic().circle(lon=main.longitude.values,
                                         lat=main.latitude.values,
                                         radius=r1,
                                         n_samples=100)
    circle2 = geodesic.Geodesic().circle(lon=main.longitude.values,
                                         lat=main.latitude.values,
                                         radius=r2,
                                         n_samples=100)
    ax.fill(*circle1.T, color='grey', alpha=0.8,
            transform=ccrs.PlateCarree(), linewidth=0, zorder=-1)
    ax.fill(*circle2.T, color='lightgrey', alpha=0.4,
            transform=ccrs.PlateCarree(), linewidth=0)

    # plot line of circle
    circle_coords = np.array([circle1.T[0], circle1.T[1]]).T
    ax.plot(circle_coords[:, 0],
            circle_coords[:, 1],
            color='black',
            linewidth=0.5,
            transform=ccrs.PlateCarree(),
            linestyle='-',
            zorder=1000)
    circle_coords = np.array([circle2.T[0], circle2.T[1]]).T
    ax.plot(circle_coords[:, 0],
            circle_coords[:, 1],
            color='black',
            linewidth=0.5,
            transform=ccrs.PlateCarree(),
            linestyle='--',
            zorder=1000)

    # Compute circle coordinates
    all_lons = np.concatenate([circle_coords[:, 0], sequence.longitude.values])
    all_lats = np.concatenate([circle_coords[:, 1], sequence.latitude.values])
    lon_min = all_lons.min()
    lon_max = all_lons.max()
    lat_min = all_lats.min()
    lat_max = all_lats.max()
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Save map figure to buffer
    fig_map.tight_layout(pad=0.1)
    fig_map.set_dpi(600)
    canvas = FigureCanvas(fig_map)
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)
    img_map = mpimg.imread(buf)
    plt.close(fig_map)

    return img_map


# Time
def create_time_plot(sequence,
                     sequence_main_idx,
                     cat_all,
                     exclude_aftershock_days):
    before, after, after_immediate, main_x, _ = define_relative_seqs(
        sequence, sequence_main_idx, cat_all, exclude_aftershock_days)

    fig_time, ax = plt.subplots(figsize=(8, 4), linewidth=1)

    # plot mags
    plot_mags_in_time(before.time, before.magnitude,
                      ax=ax, color_dots='cornflowerblue')
    plot_mags_in_time(after.time, after.magnitude,
                      ax=ax, color_dots='darkseagreen')
    plot_mags_in_time(after_immediate.time,
                      after_immediate.magnitude, ax=ax, color_dots='indianred')
    plot_mags_in_time(main_x.time, main_x.magnitude,
                      ax=ax, color_dots='yellow')
    ax.set_xlim(sequence.time.min(), sequence.time.max())

    # rotate the x ticks
    _ = plt.xticks(rotation=45)

    # Save time figure to buffer
    fig_time.set_dpi(600)
    fig_time.tight_layout(pad=0.1)
    canvas2 = FigureCanvas(fig_time)
    buf2 = io.BytesIO()
    canvas2.print_png(buf2)
    buf2.seek(0)
    img_time = mpimg.imread(buf2)
    plt.close(fig_time)

    return img_time


# FMD
def create_fmd(sequence,
               sequence_main_idx,
               cat_all,
               delta_m,
               correction_factor,
               dmc,
               exclude_aftershock_days,
               radius_close,
               n_check):

    fig_fmd, ax = plt.subplots(figsize=(8, 8), linewidth=1)

    # estimate mc for the sequence
    mc, _ = estimate_mc_maxc(sequence.magnitude,
                             fmd_bin=delta_m,
                             correction_factor=correction_factor)

    # estimate differences
    estimator = BPositiveBValueEstimator()
    estimator.calculate(sequence.magnitude, mc=mc,
                        delta_m=delta_m, times=sequence.time, dmc=dmc)
    mags = estimator.magnitudes
    times = estimator.times
    distances = sequence['distance_to_main'].values[estimator.idx]

    # get relative magnitudes
    main_event = cat_all[cat_all.index == sequence_main_idx].copy()
    (mags_close, mags_far, mags_before, mags_after) = get_farclose_beforeafter(
        times,
        mags,
        distances,
        main_event,
        exclude_aftershock_days,
        radius_close)

    # overall b-value
    estimator = ClassicBValueEstimator()
    estimator.calculate(mags, mc=dmc, delta_m=delta_m)
    plot_cum_fmd(
        estimator.magnitudes,
        fmd_bin=0.1,
        mc=dmc,
        b_value=estimator.b_value,
        ax=ax,
        color='k',
        label='all events')

    # close
    if len(mags_close) > n_check:
        estimator.calculate(mags_close, mc=dmc, delta_m=delta_m)
        plot_cum_fmd(
            estimator.magnitudes,
            ax=ax,
            fmd_bin=0.1,
            color='darkgrey',
            label='close to main event')
    # far
    if len(mags_far) > n_check:
        estimator.calculate(mags_far, mc=dmc, delta_m=delta_m)
        plot_cum_fmd(
            estimator.magnitudes,
            ax=ax,
            fmd_bin=0.1,
            color='lightgrey',
            label='far from main event')

    # before
    if len(mags_before) > n_check:
        estimator.calculate(mags_before, mc=dmc, delta_m=delta_m)
        plot_cum_fmd(
            estimator.magnitudes,
            ax=ax,
            fmd_bin=0.1,
            color='cornflowerblue',
            label='before main event')

    # after
    if len(mags_after) > n_check:
        estimator.calculate(mags_after, mc=dmc, delta_m=delta_m)
        plot_cum_fmd(
            estimator.magnitudes,
            ax=ax,
            fmd_bin=0.1,
            color='darkseagreen',
            label='after main event')

    # save fmd figure to buffer
    fig_fmd.set_dpi(600)
    fig_fmd.tight_layout(pad=0.1)
    canvas3 = FigureCanvas(fig_fmd)
    buf3 = io.BytesIO()
    canvas3.print_png(buf3)
    buf3.seek(0)
    img_fmd = mpimg.imread(buf3)
    plt.close(fig_fmd)

    return img_fmd


def save_joint_plot_fmd(img_map, img_time, img_fmd, sequence_no, result_dir):
    fig_final, (ax_top, ax_middle, ax_bottom) = plt.subplots(
        3, 1,
        figsize=(24, 20),
        gridspec_kw={'height_ratios': [2, 1, 2]}
    )

    # Get image dimensions
    h_map, w_map = img_map.shape[:2]
    h_time, w_time = img_time.shape[:2]
    h_fmd, w_fmd = img_fmd.shape[:2]

    # Plot map image in top Axes
    ax_top.imshow(img_map, aspect='equal', extent=(0, w_map, 0, h_map))
    ax_top.set_xlim(0, w_map)
    ax_top.set_ylim(0, h_map)
    ax_top.axis('off')

    # Plot time image in bottom Axes
    ax_middle.imshow(img_time, aspect='equal', extent=(0, w_time, 0, h_time))
    ax_middle.set_xlim(0, w_time)
    ax_middle.set_ylim(0, h_time)
    ax_middle.axis('off')

    # Plot FMD image in middle Axes
    ax_bottom.imshow(img_fmd, aspect='equal', extent=(0, w_fmd, 0, h_fmd))
    ax_bottom.set_xlim(0, w_fmd)
    ax_bottom.set_ylim(0, h_fmd)
    ax_bottom.axis('off')

    # save figure
    fig_final.savefig(
        result_dir / f"sequence{sequence_no}_overview.png",
        bbox_inches='tight', dpi=600)

    plt.close(fig_final)

# ========= MAIN =========


if __name__ == "__main__":
    # load catalog
    print('Loading catalogs...')
    fname_close = "df_japan_buffered_catalog_" + \
        str(int(BUFFER_M/1000))+"km_" + str(DIMENSION) + "D.csv"
    fname_far = "df_japan_buffered_catalog_400km_" + str(DIMENSION) + "D.csv"
    cat_close = load_catalog(fname_close, MC_FIXED -
                             CORRECTION_FACTOR, DELTA_M, CAT_DIR)
    cat_400km = load_catalog(fname_far, MC_FIXED -
                             CORRECTION_FACTOR, DELTA_M, CAT_DIR)

    # load coast
    column_names = ["longitude", "latitude"]
    file_path = SHAPE_DIR / "coast_japan_only_borders.m"
    df_coast = pd.read_csv(
        file_path, delim_whitespace=True, names=column_names)

    # find sequences
    print('Finding sequences...')
    # find sequences (with and without immediate aftershocks)
    seqs, main_idx, cat_close = find_sequences(
        cat_close, cat_400km,
        magnitude_threshold=MAGNITUDE_THRESHOLD,
        relation=RUPTURE_RELATION,
        days_after=pd.Timedelta(days=DAYS_AFTER),
        days_before=pd.Timedelta(days=DAYS_BEFORE),
        exclude_aftershocks=pd.Timedelta(days=EXCLUDE_AFTERSHOCK_DAYS),
        dimension=DIMENSION,
        radius_far=RADIUS_FAR,
        min_n_seq=MIN_N_SEQ,
        post_include_aftershocks=True,
    )

    seqs_noaftershocks, _, _ = find_sequences(
        cat_close, cat_400km,
        magnitude_threshold=MAGNITUDE_THRESHOLD,
        relation=RUPTURE_RELATION,
        days_after=pd.Timedelta(days=DAYS_AFTER),
        days_before=pd.Timedelta(days=DAYS_BEFORE),
        exclude_aftershocks=pd.Timedelta(days=EXCLUDE_AFTERSHOCK_DAYS),
        dimension=DIMENSION,
        radius_far=RADIUS_FAR,
        min_n_seq=MIN_N_SEQ,
        post_include_aftershocks=False,
    )

    print(f"Found {len(seqs)} sequences.")

    # make plots for each sequence
    print('Making plots...')

    for sequence_no in range(len(seqs)):
        print(f"Making plots for sequence no {sequence_no}...")
        sequence = seqs[sequence_no]
        sequence_noaftershock = seqs_noaftershocks[sequence_no]
        img_map = create_map(
            sequence, main_idx[sequence_no], cat_close,
            EXCLUDE_AFTERSHOCK_DAYS)
        img_time = create_time_plot(
            sequence, main_idx[sequence_no], cat_close,
            EXCLUDE_AFTERSHOCK_DAYS)
        img_fmd = create_fmd(sequence_noaftershock,
                             main_idx[sequence_no],
                             cat_close,
                             DELTA_M,
                             CORRECTION_FACTOR,
                             DMC,
                             EXCLUDE_AFTERSHOCK_DAYS,
                             RADIUS_CLOSE,
                             MIN_N_M)
        save_joint_plot_fmd(img_map, img_time, img_fmd,
                            sequence_no, RESULT_DIR)
