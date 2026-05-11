# CODE DEVELOPED BY SAMUEL SCHILLER
# EDITED 05/11/2026

# Investigating TOTAL brightness vs SZA, across day+night

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

file_1 = "emission_files/Brightness_SZA_Plots/total_brightness_day_1"
file_2 = "emission_files/Brightness_SZA_Plots/total_brightness_day_2"
file_3 = "emission_files/Brightness_SZA_Plots/total_brightness_night"

def load_kr(npy_path, file_type=None, override_fits=None):

    # Load in .npy file containing filename and pixels selected
    data = np.load(npy_path, allow_pickle=True)
    
    # Rewrite the file to be the v14 calibration (l2b already has new calibration)
    if override_fits is not None:
        fits_ref = override_fits
    else:
        fits_ref = data[0]
    
    pixel_list = data[1]  # Second column is list of selected pixels

    hdul = fits.open(fits_ref)  # Open file
    primary_data = hdul["PRIMARY"].data  # Open calibrated spectral radiance HDU (l1b units kR/nm, l2b units kR)

    obs = hdul["Observation"].data  # Open observation HDU
    # Open wavelength (of center of each bin) array
    # Note it does not vary over the observation axis (which has dimension 1) or spatial dimension, so just take 0th
    wl = np.array(obs["WAVELENGTH"][0, 0, :])
    
    pix_geom = hdul["PixelGeometry"].data  # Open pixel geometry HDU
    sza = pix_geom["PIXEL_SOLAR_ZENITH_ANGLE"]  # Store SZA for later plotting
    ea  = pix_geom["PIXEL_EMISSION_ANGLE"]

    bin_width = np.asarray(obs["WAVELENGTH_WIDTH"]).ravel()[0]  # Bin width constant, so just take first

    # --- Group pixels by integration index ---
    from collections import defaultdict
    groups = defaultdict(list)
    for int_idx, bin_idx in pixel_list:
        groups[int_idx].append(bin_idx)

    spectra      = []  # Create empty list to store spectra
    kr_totals    = []  # Create empty list to store kr values
    sza_selected = []  # Create empty list to store SZAs
    ea_selected  = []  # Create empty list to store emission angles

    # Loop over integrations, averaging over all selected spatial bins within each
    for int_idx, bin_indices in sorted(groups.items()):
        bin_spectra = []
        bin_kr      = []
        bin_sza     = []
        bin_ea      = []

        for bin_idx in bin_indices:
            if file_type == "l1b":
                # PRIMARY shape: (integration, spatial, spectral)
                spectrum = primary_data[int_idx, bin_idx, :]  # Spectrum of selected pixel

                # Integrate: Sum spectrum over all wavelengths, multiply by bin width
                kr_total = np.sum(spectrum) * bin_width  # units kR

            elif file_type == "l2b":
                # PRIMARY shape: (integration, spatial, spectrum component)
                # Take 1th spectrum component (solar_radiance_dup)
                spectrum = primary_data[int_idx, bin_idx, 1]
                kr_total = spectrum  # L2B already in kR, no need for bin width multiplication (or sum!)

            bin_spectra.append(spectrum)
            bin_kr.append(kr_total)
            bin_sza.append(sza[int_idx, bin_idx])
            bin_ea.append(ea[int_idx, bin_idx])

        # Store average spectrum across spatial bins for this integration
        spectra.append(np.mean(bin_spectra, axis=0))

        # Store average kr value across spatial bins for this integration
        kr_totals.append(np.mean(bin_kr))

        # Store average SZA across spatial bins for this integration
        sza_selected.append(np.mean(bin_sza))

        # Store average EA across spatial bins for this integration
        ea_selected.append(np.mean(bin_ea))

    # Bin edges (to be used in convolution step much later)
    wl_min_IUVS = wl - 0.5 * bin_width
    wl_max_IUVS = wl + 0.5 * bin_width

    # Return measured photon flux and wavelengths for each pixel
    return spectra, kr_totals, wl, sza_selected, ea_selected, bin_width, wl_min_IUVS, wl_max_IUVS





# --- NOT USING v14's, fOR NOW! BC JUST WANT ORDER-OF-MAG ESTIMATE ---


###

# Load photon flux and wavelengths for dayside l1b pixels
spectra_1, kr_1, wl_1, sza_selected_1, ea_selected_1, bin_width_1, wl_min_IUVS_1, wl_max_IUVS_1 = load_kr(file_1, 
                                                            file_type="l1b", override_fits=None) 
# Load photon flux and wavelengths for dayside l1b pixels
spectra_2, kr_2, wl_2, sza_selected_2, ea_selected_2, bin_width_2, wl_min_IUVS_2, wl_max_IUVS_2 = load_kr(file_2, 
                                                            file_type="l1b", override_fits=None) 




#nightside l1b file, so we can get spectral information
file_nightside = "orbit08800_data/orbit08889_data_NEW/mvn_iuv_l1b_apoapse-orbit08889-muv_20190411T133454_v14_r01.fits.gz"

# Load photon flux and wavelengths for nightside l2b pixels
spectra_3, kr_3, wl_3, sza_selected_3, ea_selected_3, bin_width_3, wl_min_IUVS_3, wl_max_IUVS_3 = load_kr(file_3, 
                                                            file_type="l1b", override_fits=file_nightside)

###






#  ---- Creating total plot ----

plt.rcParams.update({
    'font.size': 16,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

plt.figure(figsize=(10,6))
ax = plt.gca()

ax.scatter(sza_selected_1, kr_1, label="Dayside File 1", zorder=3)
ax.scatter(sza_selected_2, kr_2, label="Dayside File 2", zorder=3)
ax.scatter(sza_selected_3, kr_3, label="Nightside File", zorder=3)

ax.set_xlabel("SZA (deg)")
ax.set_ylabel("Brightness (kR)")
ax.set_yscale('log')
ax.invert_xaxis()

ax.legend(prop={'weight': 'bold', 'size': 14})

ax.set_axisbelow(True)  
ax.grid(True, alpha=0.5)

plt.show()




# %%

# ============================================================
# ========== SPHERE MODEL + LOG-Y PLOT (from file 1) =========
# ============================================================

from matplotlib.patches import Ellipse

# --- Combine dayside files into one array (equivalent to sza_l1b/kr_l1b) ---
sza_day = np.concatenate([sza_selected_1, sza_selected_2])
kr_day  = np.concatenate([kr_1, kr_2])

# --- Nightside arrays (equivalent to sza_l2b/kr_l2b) ---
sza_night = np.array(sza_selected_3)
kr_night  = np.array(kr_3)

# --- Combined for noise floor and scale factor ---
sza_comb = np.concatenate([sza_day, sza_night])
kr_comb  = np.concatenate([kr_day,  kr_night])

# Noise floor: average brightness where SZA > 130°
noise_mask = sza_comb > 130.0
avg_noise  = np.mean(kr_comb[noise_mask])

# Scale factor: mean brightness on the sunlit side (SZA < 100°)
scale_factor_mask = sza_comb < 100.0
scale_factor = np.mean(kr_comb[scale_factor_mask])

# --- Sphere brightness function ---
def sphere_brightness_limb(sza_points, center, radius, scale):
    brightness = np.zeros_like(sza_points)
    outside    = sza_points >= (center + radius)
    d          = sza_points[outside] - center
    brightness[outside] = (scale * radius**2) / (4 * d**2)
    return brightness

# --- Sphere parameters ---
visibility_limit = 25.0           # degrees: max visible angular distance
n_spheres = 10
radius    = (100 - 90) / (2 * n_spheres)
centers   = 90 + radius + np.arange(n_spheres) * (2 * radius)
sza_max   = 125
n_points  = 1500

# --- Compute each sphere on its own SZA grid ---
sza_each, brightness_each = [], []
for c in centers:
    sza_local = np.linspace(c + radius, sza_max, n_points)
    b_local   = sphere_brightness_limb(sza_local, c, radius, scale_factor)
    sza_each.append(sza_local)
    brightness_each.append(b_local)

# --- Build common grid and apply visibility limit ---
sza_common = np.unique(np.concatenate(sza_each))
brightness_interp = np.zeros((len(centers), len(sza_common)))

for i, c in enumerate(centers):
    interp = np.interp(sza_common, sza_each[i], brightness_each[i], left=0.0, right=0.0)
    visible_mask = np.abs(sza_common - c) <= visibility_limit
    interp[~visible_mask] = 0.0
    brightness_interp[i] = interp

brightness_sum = brightness_interp.sum(axis=0)

# Trim to the terminator shadow region (SZA 100–125°)
mask = (sza_common >= 100) & (sza_common <= 125)
sza_sum        = sza_common[mask]
brightness_sum = brightness_sum[mask]

# ===================== LOG-Y PLOT =====================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_yscale('log')

# Individual sphere curves
for i, c in enumerate(centers):
    ax.plot(sza_each[i], brightness_each[i], lw=1.0, alpha=0.6)

# Summed model curve
ax.plot(sza_sum, brightness_sum, lw=2.5, color='black',
        label=f'Sum (visible ±{visibility_limit}°)')

# Compute aspect ratio for visually round ellipses on log-y axis
fig.canvas.draw()
bbox       = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
xrange     = np.diff(ax.get_xlim())[0]
log_ymin, log_ymax = np.log10(ax.get_ylim())
yrange     = log_ymax - log_ymin
aspect_ratio = (yrange / bbox.height) / (xrange / bbox.width)

# Ellipses at each sphere limb peak
for i, c in enumerate(centers):
    if np.any(np.abs(sza_sum - c) <= visibility_limit):
        peak_idx = np.argmin(np.abs(sza_each[i] - (c + radius)))
        peak_y   = brightness_each[i][peak_idx]
        ell = Ellipse((c, peak_y),
                      width=2 * radius,
                      height=2 * radius * aspect_ratio,
                      edgecolor='gray', facecolor='none', lw=1.0, alpha=0.8)
        ax.add_patch(ell)

# Dayside scatter (files 1 + 2 combined)
ax.scatter(sza_day, kr_day, s=20, color='orange', edgecolor=None, alpha=0.7, zorder=1)
ax.text(np.median(sza_day) + 2, np.median(kr_day) * 5,
        "Dayside", color='orange', fontsize=14, fontweight='bold',
        ha='center', va='bottom', alpha=0.8)

# Nightside scatter
ax.scatter(sza_night, kr_night, s=20, color='blue', edgecolor=None,
           marker='s', alpha=0.7, zorder=1)
ax.text(np.median(sza_night) + 4, np.median(kr_night) * 5,
        "Nightside", color='blue', fontsize=14, fontweight='bold',
        ha='center', va='bottom', alpha=0.8)

# Noise floor
ax.axhline(avg_noise, color='purple', linestyle='--', linewidth=2,
           label=f'Noise floor ({avg_noise:.2e} kR)')

ax.set_xlabel('SZA (deg)', fontsize=16, fontweight='bold')
ax.set_ylabel('Brightness (kR)', fontsize=16, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=14)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight('bold')
ax.grid(True, alpha=0.5)
ax.legend(fontsize='small')
ax.set_xlim(85,)
ax.invert_xaxis()
plt.tight_layout()
plt.show()


