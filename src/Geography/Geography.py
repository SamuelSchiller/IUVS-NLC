# CODE DEVELOPED BY SAMUEL SCHILLER
# EDITED 05/11/2026

# WORKING VERSION OF LAT/LON TOOL, WITH NIGHTSIDE

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from datetime import datetime
import os, glob, re
import matplotlib.colors as colors
from scipy.ndimage import zoom

# -----------------------------
# USER CONFIG
# -----------------------------
BLOCK_FOLDER = "19400"        # Folder containing a block of orbits
ORBIT_TO_OPEN = "19489"       # The specific orbit number to load
MAIN_FOLDER = "/Volumes/MARS/spectral_tool_2/ALL_NLC_L1B/"
OUTPUT_DIR = MAIN_FOLDER
TOLERANCE_DEG = 1.0           # max jump in mirror angles within a swath
MIRROR_MATCH_TOLERANCE = 25.0  # tolerance for matching nightside to dayside mirror angles
REVERSE_SWATHS = True        # True to flip rightmost swath to the left

# If nightside becomes saturated/dark, try different parameters!

# vmi_night_SS = 0.5  
# vma_night_SS = 3.0

vmi_night_SS = 5.0  
vma_night_SS = 30.0  

# vmi_night_SS = 50.0  
# vma_night_SS = 300.0  

# vmi_night_SS = 500.0  
# vma_night_SS = 3000.0  

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def extract_orbit_number(filename):
    """Extract the 5-digit orbit number from the FITS filename."""
    basename = os.path.basename(filename)
    match = re.search(r"orbit(\d{5})", basename)
    if match:
        return match.group(1)
    return None

def normalize_power_law(x, vmin, vmax, gamma=0.2):
    x = (x - vmin) / (vmax - vmin)
    x = np.clip(x, 0, 1)
    return x ** gamma
 
def group_by_mirror_angles(hduls, tolerance_deg):
    grouped_swaths = []
    current_swath = []
    for i, entry in enumerate(hduls):
        if not current_swath:
            current_swath.append(i)
            continue
        prev_last = hduls[current_swath[-1]]["last_angle"]
        if abs(entry["first_angle"] - prev_last) < tolerance_deg:
            current_swath.append(i)
        else:
            grouped_swaths.append(current_swath)
            current_swath = [i]
    if current_swath:
        grouped_swaths.append(current_swath)
    return grouped_swaths

def load_orbit_files_by_voltage(fits_files):
    """Load FITS files separated by MCP voltage, skipping 2D PRIMARY data."""
    hduls_dayside = []  # MCP < 750V
    hduls_nightside = []  # MCP >= 750V
    
    for file in fits_files:
        basename = os.path.basename(file)
        try:
            utc_str = basename.split("_")[4]
            utc_time = datetime.strptime(utc_str, "%Y%m%dT%H%M%S")
        except Exception:
            utc_time = datetime.fromtimestamp(os.path.getmtime(file))

        hdul = fits.open(file)

        # Skip if PRIMARY is 2D
        primary_data = hdul["PRIMARY"].data
        if primary_data.ndim == 2:
            print(f"Skipping file {basename}: PRIMARY is 2D")
            hdul.close()
            continue

        if "Observation" in hdul:
            mcp = float(hdul["Observation"].data["MCP_volt"][0])
            mirror_angles = hdul["Integration"].data["MIRROR_DEG"]
            
            if len(mirror_angles) > 0:
                entry = {
                    "file": file,
                    "hdul": hdul,
                    "utc": utc_time,
                    "first_angle": mirror_angles[0],
                    "last_angle": mirror_angles[-1],
                    "mcp": mcp
                }
                
                if mcp < 750:
                    hduls_dayside.append(entry)
                else:
                    hduls_nightside.append(entry)
        else:
            hdul.close()

    hduls_dayside.sort(key=lambda x: x["utc"])
    hduls_nightside.sort(key=lambda x: x["utc"])
    return hduls_dayside, hduls_nightside

# -----------------------------
# DISCOVER FILES IN BLOCK FOLDER
# -----------------------------
search_folder = os.path.join(MAIN_FOLDER, f"orbit{BLOCK_FOLDER}")
all_files = glob.glob(os.path.join(search_folder, "mvn_iuv_l1b_apoapse-orbit*-muv_*.fits*"))

# Filter by the desired orbit number
fits_files = [f for f in all_files if extract_orbit_number(f) == ORBIT_TO_OPEN]

if not fits_files:
    print(f"No files found for orbit {ORBIT_TO_OPEN} in folder {BLOCK_FOLDER}.")
else:
    print(f"Found {len(fits_files)} files for orbit {ORBIT_TO_OPEN} in folder {BLOCK_FOLDER}.")

# -----------------------------
# Colorization helper functions (Option B pipeline)
# -----------------------------
def make_equidistant_spectral_cutoff_indices(n_spectral_bins: int) -> tuple[int, int]:
    """Make indices so the spectral bins are evenly split into 3 channels."""
    blue_green_cutoff = round(n_spectral_bins / 3)
    green_red_cutoff = round(n_spectral_bins * 2 / 3)
    return blue_green_cutoff, green_red_cutoff

def turn_detector_image_to_3_channels(image: np.ndarray) -> np.ndarray:
    """
    Convert detector image (integrations, spatial, spectral_bins) -> RGB by coadding
    spectral ranges into blue, green, red channels.
    """
    n_spectral_bins = image.shape[2]
    blue_green_cutoff, green_red_cutoff = make_equidistant_spectral_cutoff_indices(n_spectral_bins)
    red = np.sum(image[..., green_red_cutoff:], axis=-1)
    green = np.sum(image[..., blue_green_cutoff:green_red_cutoff], axis=-1)
    blue = np.sum(image[..., :blue_green_cutoff], axis=-1)
    return np.dstack([red, green, blue])

def histogram_equalize_grayscale_image_with_reference(image: np.ndarray, reference_pixels: np.ndarray) -> np.ndarray:
    """
    Equalize an image channel using a reference pixel array (1D).
    reference_pixels: 1D array of pixel values used to derive mapping.
    """
    if reference_pixels.size == 0:
        mn, mx = np.nanmin(image), np.nanmax(image)
        if mx <= mn:
            return np.zeros_like(image)
        return (image - mn) / (mx - mn) * 255.0

    sorted_values = np.sort(reference_pixels, axis=None)
    left_cutoffs = np.array([sorted_values[int(i / 256 * (len(sorted_values)-1))] for i in range(256)])
    rgb_vals = np.linspace(0, 255, num=256)
    out = np.interp(image, left_cutoffs, rgb_vals, left=0, right=255)
    return out

# -----------------------------
# NEW GLOBALS / SELECTION DATAFRAME
# -----------------------------
current_cloud_id = 1
all_selected_pixels_df = pd.DataFrame(
    columns=[
        "cloud_id","sel_type","orbit","fits_file","integration","spatial","pixel",
        "lat","lon","sza","phase_angle","emission_angle",
        "solar_lon","local_time","utc_time"
    ]
)
marker_artists = []
next_sel_type = None


# -----------------------------
# REPLACEMENT: process_and_plot_orbit
# -----------------------------
def process_and_plot_orbit(orbit_number, fits_files):
    global current_cloud_id, all_selected_pixels_df, marker_artists, next_sel_type

    print(f"\n=== Processing orbit {orbit_number} ({len(fits_files)} files) ===")
    hduls_dayside, hduls_nightside = load_orbit_files_by_voltage(fits_files)
    
    print(f"Found {len(hduls_dayside)} dayside files (MCP<750V)")
    print(f"Found {len(hduls_nightside)} nightside files (MCP>=750V)")
    
    if not hduls_dayside:
        print(f"No usable dayside files for orbit {orbit_number}")
        return

    grouped_dayside = group_by_mirror_angles(hduls_dayside, TOLERANCE_DEG)
    grouped_nightside = group_by_mirror_angles(hduls_nightside, TOLERANCE_DEG) if hduls_nightside else []
    
    print(f"Dayside swaths: {len(grouped_dayside)}")
    print(f"Nightside swaths: {len(grouped_nightside)}")

    # -----------------------------
    # Match nightside swaths to dayside swaths using MIRROR ANGLE continuity.
    #
    # Algorithm:
    # 1. Determine if nightside swath is 'above' or 'below' based on UTC comparison to global dayside midpoint
    # 2. For 'above' nightside: find dayside where nightside.last_angle ≈ dayside.first_angle
    # 3. For 'below' nightside: find dayside where nightside.first_angle ≈ dayside.last_angle
    # 4. If no match within tolerance, that nightside swath is skipped (gap in data)
    # -----------------------------
    nightside_to_dayside_map = {}   # nightside_swath_idx -> dayside_swath_idx
    nightside_position_map   = {}   # nightside_swath_idx -> 'above' | 'below'

    # Global dayside midpoint (sorted list, pick middle entry)
    all_day_utcs = [hduls_dayside[i]["utc"]
                    for swath in grouped_dayside for i in swath]
    all_day_utcs.sort()
    global_day_mid_utc = all_day_utcs[len(all_day_utcs) // 2]

    # -----------------------------
    # Match nightside to dayside using global best-first assignment.
    #
    # Score ALL possible (nightside, dayside) pairs first, then assign
    # from best score down. This avoids the greedy trap where early
    # assignments lock in sub-optimal matches and leave bad leftovers.
    #
    # Position ('above'/'below') is determined per-pair by comparing the
    # nightside swath's UTC to the dayside swath's UTC range.
    # Mirror angle continuity is used as the primary matching quality
    # metric, with UTC proximity as a secondary factor.
    # -----------------------------

    # Pre-compute per-swath mirror angles and UTC ranges
    day_info = []
    for day_idx, day_swath in enumerate(grouped_dayside):
        day_utcs = [hduls_dayside[i]["utc"] for i in day_swath]
        day_info.append({
            "first_angle": hduls_dayside[day_swath[0]]["first_angle"],
            "last_angle":  hduls_dayside[day_swath[-1]]["last_angle"],
            "first_utc":   day_utcs[0],
            "last_utc":    day_utcs[-1],
            "mid_utc":     day_utcs[len(day_utcs) // 2],
        })

    night_info = []
    for night_idx, night_swath in enumerate(grouped_nightside):
        night_utcs = [hduls_nightside[i]["utc"] for i in night_swath]
        night_info.append({
            "first_angle": hduls_nightside[night_swath[0]]["first_angle"],
            "last_angle":  hduls_nightside[night_swath[-1]]["last_angle"],
            "mid_utc":     night_utcs[len(night_utcs) // 2],
        })

    # Score every (nightside, dayside) pair
    all_candidates = []   # (combined_score, angle_diff, night_idx, day_idx, position)

    for night_idx, ni in enumerate(night_info):
        for day_idx, di in enumerate(day_info):

            # Determine position relative to this dayside swath via UTC
            if ni["mid_utc"] < di["first_utc"]:
                position = 'above'
            elif ni["mid_utc"] > di["last_utc"]:
                position = 'below'
            else:
                dist_to_start = abs((ni["mid_utc"] - di["first_utc"]).total_seconds())
                dist_to_end   = abs((ni["mid_utc"] - di["last_utc"]).total_seconds())
                position = 'above' if dist_to_start < dist_to_end else 'below'

            # Mirror angle continuity based on position
            if position == 'above':
                angle_diff = abs(ni["last_angle"] - di["first_angle"])
            else:
                angle_diff = abs(di["last_angle"] - ni["first_angle"])

            # Skip pairs that exceed the mirror angle tolerance
            if angle_diff > MIRROR_MATCH_TOLERANCE:
                continue

            time_diff_sec = abs((ni["mid_utc"] - di["mid_utc"]).total_seconds())
            time_score    = time_diff_sec * 0.01
            combined_score = angle_diff + time_score

            all_candidates.append((combined_score, angle_diff, night_idx, day_idx, position))

    # Sort all candidates by combined score (best first)
    all_candidates.sort(key=lambda x: x[0])

    night_used  = set()
    dayside_used = set()

    for combined_score, angle_diff, night_idx, day_idx, position in all_candidates:
        if night_idx in night_used or day_idx in dayside_used:
            continue
        nightside_to_dayside_map[night_idx] = day_idx
        nightside_position_map[night_idx]   = position
        night_used.add(night_idx)
        dayside_used.add(day_idx)
        print(f"Nightside swath {night_idx} -> Dayside swath {day_idx} "
              f"[{position}]  (angle diff: {angle_diff:.2f}°, combined score: {combined_score:.2f})")

    # Fallback: any nightside swath not yet matched gets the closest unused dayside by UTC
    unmatched_nightside = [i for i in range(len(grouped_nightside)) if i not in night_used]
    if unmatched_nightside:
        print(f"\nFallback matching for {len(unmatched_nightside)} unmatched nightside swaths:")

        unused_day_info = [(i, day_info[i]) for i in range(len(grouped_dayside)) if i not in dayside_used]

        for night_idx in unmatched_nightside:
            if not unused_day_info:
                print(f"  WARNING: No unused dayside swaths left for nightside {night_idx}")
                continue

            ni = night_info[night_idx]
            best_day_idx = min(unused_day_info,
                               key=lambda x: abs((ni["mid_utc"] - x[1]["mid_utc"]).total_seconds()))
            day_idx, di = best_day_idx

            # Determine position relative to matched dayside
            if ni["mid_utc"] < di["first_utc"]:
                position = 'above'
            elif ni["mid_utc"] > di["last_utc"]:
                position = 'below'
            else:
                dist_to_start = abs((ni["mid_utc"] - di["first_utc"]).total_seconds())
                dist_to_end   = abs((ni["mid_utc"] - di["last_utc"]).total_seconds())
                position = 'above' if dist_to_start < dist_to_end else 'below'

            nightside_to_dayside_map[night_idx] = day_idx
            nightside_position_map[night_idx]   = position
            night_used.add(night_idx)
            dayside_used.add(day_idx)
            unused_day_info = [(i, d) for i, d in unused_day_info if i != day_idx]

            time_diff = abs((ni["mid_utc"] - di["mid_utc"]).total_seconds())
            print(f"  Fallback: Nightside {night_idx} -> Dayside {day_idx} [{position}] "
                  f"(UTC diff: {time_diff:.1f}s)")

    # -----------------------------
    # Build DAYSIDE RGB tiles
    # -----------------------------
    swath_widths, swath_heights = [], []
    swath_file_heights = []
    for swath in grouped_dayside:
        file_widths = [hduls_dayside[i]["hdul"]["PRIMARY"].data.shape[1] for i in swath]
        file_heights = [hduls_dayside[i]["hdul"]["PRIMARY"].data.shape[0] for i in swath]
        swath_widths.append(max(file_widths))
        swath_heights.append(sum(file_heights))
        swath_file_heights.append(file_heights)

    # Calculate nightside dimensions and integer zoom factors
    nightside_widths = {}
    nightside_heights = {}
    nightside_file_heights = {}
    nightside_zoom_factors = {}
    nightside_scaled_heights = {}
    nightside_scaled_widths = {}
    
    for night_idx, night_swath in enumerate(grouped_nightside):
        file_widths = [hduls_nightside[i]["hdul"]["PRIMARY"].data.shape[1] for i in night_swath]
        file_heights = [hduls_nightside[i]["hdul"]["PRIMARY"].data.shape[0] for i in night_swath]
        nightside_widths[night_idx] = max(file_widths)
        nightside_heights[night_idx] = sum(file_heights)
        nightside_file_heights[night_idx] = file_heights
        
        # Calculate NEAREST INTEGER zoom factor
        if night_idx in nightside_to_dayside_map:
            day_idx = nightside_to_dayside_map[night_idx]
            target_cols = swath_widths[day_idx]
            zoom_factor_float = target_cols / nightside_widths[night_idx]
            zoom_factor_int = max(1, round(zoom_factor_float))
            nightside_zoom_factors[night_idx] = zoom_factor_int
            nightside_scaled_heights[night_idx] = nightside_heights[night_idx] * zoom_factor_int
            nightside_scaled_widths[night_idx] = nightside_widths[night_idx] * zoom_factor_int
            print(f"  Nightside {night_idx}: zoom {zoom_factor_float:.2f} -> {zoom_factor_int}x (blocks)")

    # Calculate total dimensions
    dayside_base_width = sum(swath_widths)
    
    max_overhang = 0
    for night_idx, day_idx in nightside_to_dayside_map.items():
        night_width = nightside_scaled_widths[night_idx]
        day_width = swath_widths[day_idx]
        overhang = max(0, night_width - day_width)
        max_overhang = max(max_overhang, overhang)
    
    total_width = dayside_base_width + max_overhang
    dayside_max_height = max(swath_heights) if swath_heights else 0

    max_nightside_height_above = 0
    max_nightside_height_below = 0

    for night_idx, position in nightside_position_map.items():
        if night_idx not in nightside_scaled_heights:
            continue
        if position == 'above':
            max_nightside_height_above = max(
                max_nightside_height_above, nightside_scaled_heights[night_idx]
            )
        else:
            max_nightside_height_below = max(
                max_nightside_height_below, nightside_scaled_heights[night_idx]
            )

    nightside_padding_above = max_nightside_height_above
    nightside_padding_below = max_nightside_height_below
    total_height = nightside_padding_above + dayside_max_height + nightside_padding_below
    
    rgb_all = np.zeros((total_height, total_width, 3), dtype=float)

    all_reference_pixels = [np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)]

    cum_file_heights_per_swath = [np.cumsum([0] + fh) for fh in swath_file_heights]
    cum_file_heights_nightside = {night_idx: np.cumsum([0] + nightside_file_heights[night_idx]) 
                                  for night_idx in nightside_file_heights}

    # Collect reference pixels from dayside only
    for swath in grouped_dayside:
        for entry_idx in swath:
            entry = hduls_dayside[entry_idx]
            hdul = entry["hdul"]
            try:
                PIXEL_GEOM = hdul["PixelGeometry"].data
                tangent_alt = PIXEL_GEOM["PIXEL_CORNER_MRH_ALT"][..., 4]
                sza = PIXEL_GEOM["PIXEL_SOLAR_ZENITH_ANGLE"]
                mask = (tangent_alt == 0) & (~np.isnan(sza))
            except Exception:
                mask = None

            PRIMARY = hdul["PRIMARY"].data
            rgb_image = turn_detector_image_to_3_channels(PRIMARY)

            for ch in range(3):
                channel_vals = rgb_image[..., ch]
                if mask is None:
                    vals = channel_vals.ravel()
                else:
                    vals = channel_vals[mask]
                if vals.size:
                    all_reference_pixels[ch] = np.concatenate([all_reference_pixels[ch], vals.ravel()])

    for ch in range(3):
        if all_reference_pixels[ch].size == 0:
            sample_vals = []
            for swath in grouped_dayside:
                for entry_idx in swath:
                    entry = hduls_dayside[entry_idx]
                    PRIMARY = entry["hdul"]["PRIMARY"].data
                    rgb_image = turn_detector_image_to_3_channels(PRIMARY)
                    sample_vals.append(rgb_image[..., ch].ravel()[:1000])
                    if sum(len(x) for x in sample_vals) >= 1000:
                        break
                if sum(len(x) for x in sample_vals) >= 1000:
                    break
            if sample_vals:
                all_reference_pixels[ch] = np.concatenate(sample_vals)
            else:
                all_reference_pixels[ch] = np.array([0.0])

    # DAYSIDE swath x positions
    swath_x_start = []
    x_start = 0
    for w in swath_widths:
        swath_x_start.append(x_start)
        x_start += w

    n_swaths = len(grouped_dayside)
    if REVERSE_SWATHS:
        display_order = list(reversed(range(n_swaths)))
    else:
        display_order = list(range(n_swaths))

    display_swath_meta = []
    dayside_swath_data = []
    
    for draw_idx, orig_idx in enumerate(display_order):
        swath = grouped_dayside[orig_idx]
        
        swath_rgb_images = []
        swath_sza_maps = []
        
        for entry_idx in swath:
            entry = hduls_dayside[entry_idx]
            hdul = entry["hdul"]
            PRIMARY = hdul["PRIMARY"].data
            PIXEL_GEOM = hdul["PixelGeometry"].data
            sza_data = PIXEL_GEOM["PIXEL_SOLAR_ZENITH_ANGLE"]

            rgb_image = turn_detector_image_to_3_channels(PRIMARY)

            for ch in range(3):
                rgb_image[..., ch] = histogram_equalize_grayscale_image_with_reference(
                    rgb_image[..., ch], all_reference_pixels[ch]
                )

            rgb_image = np.clip(rgb_image, 0, 255) / 255.0
            
            swath_rgb_images.append(rgb_image)
            swath_sza_maps.append(sza_data)
        
        swath_rgb_full = np.vstack(swath_rgb_images)
        swath_sza_full = np.vstack(swath_sza_maps)
        
        total_rows = swath_rgb_full.shape[0]
        total_cols = swath_rgb_full.shape[1]
        
        row_start = nightside_padding_above + (dayside_max_height - total_rows)
        
        dayside_swath_data.append({
            "rgb": swath_rgb_full,
            "sza": swath_sza_full,
            "orig_idx": orig_idx,
            "row_start": row_start,
            "total_rows": total_rows,
            "total_cols": total_cols
        })
        
        display_swath_meta.append({
            "orig_swath_idx": orig_idx,
            "x0": swath_x_start[draw_idx],
            "x1": swath_x_start[draw_idx] + total_cols,
            "y0": row_start,
            "y1": row_start + total_rows,
            "cum_heights": cum_file_heights_per_swath[orig_idx],
            "widths": [hduls_dayside[i]["hdul"]["PRIMARY"].data.shape[1] for i in swath],
            "is_dayside": True
        })

    # Plot DAYSIDE swaths
    for idx, (swath_data, meta) in enumerate(zip(dayside_swath_data, display_swath_meta)):
        col_start = meta["x0"]
        row_start = meta["y0"]
        h, w = swath_data["rgb"].shape[:2]
        rgb_all[row_start:row_start+h, col_start:col_start+w, :] = swath_data["rgb"]

    # Build NIGHTSIDE metadata and plot
    nightside_swath_meta = []
    cmapSS = colors.LinearSegmentedColormap.from_list("", ["#000000", "#555555", "#AAAAAA"])
    normSS = colors.Normalize(vmin=vmi_night_SS, vmax=vma_night_SS)
    
    for night_idx, day_idx in nightside_to_dayside_map.items():
        night_swath = grouped_nightside[night_idx]
        position    = nightside_position_map[night_idx]
        
        dayside_swath_idx = [i for i, m in enumerate(display_swath_meta) if m["orig_swath_idx"] == day_idx][0]
        dayside_meta = display_swath_meta[dayside_swath_idx]
        
        night_raw_images = []
        night_sza_maps   = []
        
        for entry_idx in night_swath:
            entry  = hduls_nightside[entry_idx]
            hdul   = entry["hdul"]
            PRIMARY = hdul["PRIMARY"].data
            PIXEL_GEOM = hdul["PixelGeometry"].data
            sza_data   = PIXEL_GEOM["PIXEL_SOLAR_ZENITH_ANGLE"]
            grayscale  = np.sum(PRIMARY, axis=2)
            night_raw_images.append(grayscale)
            night_sza_maps.append(sza_data)
        
        night_raw_full = np.vstack(night_raw_images)
        night_sza_full = np.vstack(night_sza_maps)
        
        original_rows, original_cols = night_raw_full.shape
        zoom_factor = nightside_zoom_factors[night_idx]
        
        enlarged_grayscale = zoom(night_raw_full, (zoom_factor, zoom_factor), order=0)
        enlarged_sza       = zoom(night_sza_full, (zoom_factor, zoom_factor), order=0)
        
        normalized     = normSS(enlarged_grayscale)
        night_rgb_full = cmapSS(normalized)[:, :, :3]
        
        total_rows = night_rgb_full.shape[0]
        total_cols = night_rgb_full.shape[1]

        if position == 'above':
            row_start = dayside_meta["y0"] - total_rows
        else:
            row_start = dayside_meta["y1"]

        col_start = dayside_meta["x0"]
        
        nightside_swath_meta.append({
            "night_swath_idx": night_idx,
            "x0": col_start,
            "x1": col_start + total_cols,
            "y0": row_start,
            "y1": row_start + total_rows,
            "cum_heights": cum_file_heights_nightside[night_idx],
            "widths": [hduls_nightside[i]["hdul"]["PRIMARY"].data.shape[1] for i in night_swath],
            "swath_indices": night_swath,
            "is_dayside": False,
            "zoom_factor": zoom_factor,
            "original_cols": original_cols,
            "original_rows": original_rows,
            "position": position
        })
        
        plot_row_start = max(0, row_start)
        plot_row_end   = min(total_height, row_start + total_rows)
        plot_col_start = max(0, col_start)
        plot_col_end   = min(total_width, col_start + total_cols)
        
        img_row_start = plot_row_start - row_start
        img_row_end   = img_row_start + (plot_row_end - plot_row_start)
        img_col_start = plot_col_start - col_start
        img_col_end   = img_col_start + (plot_col_end - plot_col_start)
        
        if plot_row_end > plot_row_start and plot_col_end > plot_col_start:
            rgb_all[plot_row_start:plot_row_end, plot_col_start:plot_col_end, :] = \
                night_rgb_full[img_row_start:img_row_end, img_col_start:img_col_end, :]

    # Combine metadata
    all_swath_meta = display_swath_meta + nightside_swath_meta

    # -----------------------------
    # Helper: map click to file and local
    # -----------------------------
    def map_click_to_file_and_local(row, col):
        for swath_meta in all_swath_meta:
            if swath_meta["x0"] <= col < swath_meta["x1"]:
                if not (swath_meta["y0"] <= row < swath_meta["y1"]):
                    continue
                    
                local_col = int(col - swath_meta["x0"])
                local_row_from_bottom = int(row - swath_meta["y0"])
                cum_heights = swath_meta["cum_heights"]
                
                if swath_meta.get("is_dayside", False):
                    orig_swath_idx = swath_meta["orig_swath_idx"]
                    swath = grouped_dayside[orig_swath_idx]
                    file_within_swath = np.searchsorted(cum_heights, local_row_from_bottom, side='right') - 1
                    file_within_swath = max(0, min(file_within_swath, len(swath)-1))
                    entry_idx = swath[file_within_swath]
                    entry = hduls_dayside[entry_idx]
                else:
                    zoom_factor = swath_meta["zoom_factor"]
                    original_local_col = local_col // zoom_factor
                    original_local_row = local_row_from_bottom // zoom_factor
                    
                    swath = swath_meta["swath_indices"]
                    file_within_swath = np.searchsorted(cum_heights, original_local_row, side='right') - 1
                    file_within_swath = max(0, min(file_within_swath, len(swath)-1))
                    entry_idx = swath[file_within_swath]
                    entry = hduls_nightside[entry_idx]
                    
                    local_col = original_local_col
                    local_row_from_bottom = original_local_row
                
                n_rows, n_cols = entry["hdul"]["PRIMARY"].data.shape[:2]
                if local_col >= n_cols:
                    return None
                local_row = local_row_from_bottom - cum_heights[file_within_swath]
                if local_row < 0 or local_row >= n_rows:
                    return None
                return entry, file_within_swath, int(local_row), int(local_col)
        
        return None

    # -----------------------------
    # Plot image
    # -----------------------------
    fig, ax = plt.subplots(figsize=(14,7))
    ax.imshow(rgb_all, origin="lower")
    ax.set_title(f"Orbit {orbit_number} — Dayside + Nightside RGB (MCP threshold 750V)")
    ax.set_xlabel("Spatial bin")
    ax.set_ylabel("Integration row")

    # -----------------------------
    # Click handler
    # -----------------------------
    def on_click(event):
        global current_cloud_id, all_selected_pixels_df, marker_artists, next_sel_type
        if event.inaxes != ax or not getattr(event, "dblclick", False):
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= col < rgb_all.shape[1] and 0 <= row < rgb_all.shape[0]):
            return

        mapped = map_click_to_file_and_local(row, col)
        if mapped is None:
            print("Click outside valid data region.")
            return

        entry, file_within_swath, local_row, local_col = mapped
        pg = entry["hdul"]["PixelGeometry"].data
        try:
            lat = float(pg["PIXEL_CORNER_LAT"][local_row, local_col, -1])
            lon = float(pg["PIXEL_CORNER_LON"][local_row, local_col, -1])
            sza = float(pg["PIXEL_SOLAR_ZENITH_ANGLE"][local_row, local_col])
            phase_angle = float(pg["PIXEL_PHASE_ANGLE"][local_row, local_col])
            emission_angle = float(pg["PIXEL_EMISSION_ANGLE"][local_row, local_col])
            local_time = float(pg["PIXEL_LOCAL_TIME"][local_row, local_col])
            utc_time = entry["hdul"]["Integration"].data["UTC"][local_row]
        except Exception as e:
            print(f"Geometry lookup failed: {e}")
            return

        solar_lon = float(entry["hdul"]["Observation"].data["SOLAR_LONGITUDE"][0])

        if next_sel_type is not None:
            sel_type = next_sel_type
            next_sel_type = None
        else:
            existing = all_selected_pixels_df[all_selected_pixels_df["cloud_id"] == current_cloud_id]
            if existing.empty or "day_edge" not in existing["sel_type"].values:
                sel_type = "day_edge"
            elif "middle" not in existing["sel_type"].values:
                sel_type = "middle"
            elif "night_edge" not in existing["sel_type"].values:
                sel_type = "night_edge"
            else:
                current_cloud_id += 1
                sel_type = "day_edge"

        all_selected_pixels_df.loc[len(all_selected_pixels_df)] = {
            "cloud_id": int(current_cloud_id),
            "sel_type": sel_type,
            "orbit": entry["file"].split("orbit")[-1][:5],
            "fits_file": entry["file"],
            "integration": int(local_row),
            "spatial": int(local_col),
            "pixel": [int(local_row), int(local_col)],
            "lat": lat,
            "lon": lon,
            "sza": sza,
            "phase_angle": phase_angle,
            "emission_angle": emission_angle,
            "solar_lon": solar_lon,
            "local_time": local_time,
            "utc_time": utc_time
        }

        if sel_type == "middle":
            artist, = ax.plot(col, row, marker="X", color="red", markersize=6)
        elif sel_type == "day_edge":
            artist, = ax.plot(col, row, marker="^", color="gold", markersize=6)
        else:
            artist, = ax.plot(col, row, marker="D", color="cyan", markersize=6)

        marker_artists.append((int(current_cloud_id), sel_type, artist))
        fig.canvas.draw_idle()

        print(f"[cloud {current_cloud_id}] Added {sel_type}: row {local_row}, col {local_col}, lat {lat:.3f}, lon {lon:.3f}, SZA {sza:.3f}")

    # -----------------------------
    # Key handler
    # -----------------------------
    def on_key(event):
        global current_cloud_id, all_selected_pixels_df, marker_artists, next_sel_type
        key = getattr(event, "key", None)

        if key == '1':
            next_sel_type = "middle"
            print("Next selection forced to: middle")
        elif key == '2':
            next_sel_type = "day_edge"
            print("Next selection forced to: day_edge")
        elif key == '3':
            next_sel_type = "night_edge"
            print("Next selection forced to: night_edge")
        elif key == 'n':
            current_cloud_id += 1
            print(f"Started new cloud {current_cloud_id}")
        elif key == 'u':
            if all_selected_pixels_df.empty:
                print("Nothing to undo.")
                return
            last_idx = all_selected_pixels_df.index[-1]
            last_row = all_selected_pixels_df.loc[last_idx]
            cid = int(last_row["cloud_id"])
            stype = last_row["sel_type"]
            all_selected_pixels_df.drop(last_idx, inplace=True)
            all_selected_pixels_df.reset_index(drop=True, inplace=True)
            for i in range(len(marker_artists)-1, -1, -1):
                aid, atype, art = marker_artists[i]
                if aid == cid and atype == stype:
                    try:
                        art.remove()
                    except Exception:
                        pass
                    marker_artists.pop(i)
                    break
            fig.canvas.draw_idle()
            print(f"Undid last selection: cloud {cid} {stype}")
        elif key == 's':
            save_path = os.path.join(OUTPUT_DIR, "all_selected_pixels_REVAMPED.csv")
            if os.path.exists(save_path):
                existing_df = pd.read_csv(save_path)
                existing_has_type = ("sel_type" in existing_df.columns)
                if existing_has_type:
                    existing_keys = set(zip(existing_df["fits_file"], existing_df["pixel"], existing_df["sel_type"]))
                else:
                    existing_keys = set(zip(existing_df["fits_file"], existing_df["pixel"]))
            else:
                existing_df = None
                existing_has_type = False
                existing_keys = set()

            to_write = []
            for _, row in all_selected_pixels_df.iterrows():
                if existing_has_type:
                    key = (row["fits_file"], str(row["pixel"]), row["sel_type"])
                else:
                    key = (row["fits_file"], str(row["pixel"]))
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                out = {
                    "fits_file": row["fits_file"],
                    "utc_time": row["utc_time"],
                    "orbit": row["orbit"],
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "solar_lon": row["solar_lon"],
                    "sza": row["sza"],
                    "phase_angle": row["phase_angle"],
                    "local_time": row["local_time"],
                    "emission_angle": row["emission_angle"],
                    "pixel": str(row["pixel"]),
                    "sel_type": row["sel_type"]
                }
                to_write.append(out)

            if to_write:
                df_to_write = pd.DataFrame(to_write)
                col_order = [
                    "fits_file","utc_time","orbit","lat","lon","solar_lon","sza","phase_angle","local_time","emission_angle","pixel","sel_type"
                ]
                df_to_write.to_csv(save_path, mode="a", index=False, header=not os.path.exists(save_path), columns=col_order)
                print(f"Appended {len(df_to_write)} new pixel(s) to {save_path}")
                all_selected_pixels_df = pd.DataFrame(columns=all_selected_pixels_df.columns)
                for _cid, _stype, art in marker_artists:
                    try:
                        art.remove()
                    except Exception:
                        pass
                marker_artists.clear()
                fig.canvas.draw_idle()
            else:
                print("No new pixels to append.")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

# -----------------------------
# RUN ORBIT IF FILES FOUND
# -----------------------------
if fits_files:
    process_and_plot_orbit(ORBIT_TO_OPEN, fits_files)