# CODE DEVELOPED BY SAMUEL SCHILLER
# CREATED 01/13/2026 
# EDITED 05/11/2026

# Importing useful packaages
import spiceypy as spice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.io import readsav
from scipy.signal import convolve
import matplotlib.cm as cm
import matplotlib.colors as colors
import astropy.units as u
from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler

# ------ File paths -------
file_l1b = "emission_files/Brightness_SZA_Plots/DAYSIDE_08889_rectangle.npy"
file_l2b = "emission_files/Brightness_SZA_Plots/NIGHTSIDE_08889_rectangle.npy"

# New l1b with v14 calibration
new_cal_l1b = "orbit08800_data/orbit08889_data_NEW/mvn_iuv_l1b_apoapse-orbit08889-muv_20190411T133726_v14_r01.fits.gz"

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

# Load photon flux and wavelengths for dayside l1b pixels
spectra_l1b, kr_l1b, wl_l1b, sza_selected_l1b, ea_selected_l1b, bin_width_l1b, wl_min_IUVS_l1b, wl_max_IUVS_l1b = load_kr(file_l1b, 
                                                            file_type="l1b", override_fits=new_cal_l1b) 

# # Load photon flux and wavelengths for nightside l2b pixels
# spectra_l2b, kr_l2b, wl_l2b, sza_selected_l2b, ea_selected_l2b, bin_width_l2b, wl_min_IUVS_l2b, wl_max_IUVS_l2b = load_kr(file_l2b, 
#                                                             file_type="l2b", override_fits=None)


#nightside l1b file, so we can get spectral information
file_nightside = "/Volumes/MARS/spectral_tool_2/orbit08800_data/orbit08889_data_NEW/mvn_iuv_l1b_apoapse-orbit08889-muv_20190411T133454_v14_r01.fits.gz"

# Load photon flux and wavelengths for nightside l2b pixels
spectra_l2b, kr_l2b, wl_l2b, sza_selected_l2b, ea_selected_l2b, bin_width_l2b, wl_min_IUVS_l2b, wl_max_IUVS_l2b = load_kr(file_l2b, 
                                                            file_type="l1b", override_fits=file_nightside)





# l1b pixels
measured_radiance_kr_l1b = np.array(kr_l1b) # Convert to array to do math, units kR
measured_radiance_r_l1b = measured_radiance_kr_l1b * 1000 # Convert to R
measured_radiance_ph_l1b = (measured_radiance_r_l1b) * (1/(4*np.pi)) * (10**10) # photons s^-1 m^-2 sr^-1

# l2b pixels
measured_radiance_kr_l2b = np.array(kr_l2b) # Convert to array to do math, units kR
measured_radiance_r_l2b = measured_radiance_kr_l2b * 1000 # Convert to R
measured_radiance_ph_l2b = (measured_radiance_r_l2b) * (1/(4*np.pi)) * (10**10) # photons s^-1 m^-2 sr^-1

# Calibrated brightness has been obtained for all selected pixels!

# ===== Converting SOLSTICE --> "IUVS-like" spectrum =====

# ------ DEFINING CONSTANTS -------
planck_constant = 6.62607015e-34 # J*s
speed_light = 299792458 # m/s

# ------ Obtaining Sun-Mars distance with SPICE kernels ------
spice.furnsh("SPICE_kernels/naif0012.tls") # leapseconds
spice.furnsh("SPICE_kernels/de430s.bsp") # planetary ephemeris
spice.furnsh("SPICE_kernels/pck00009.tpc")  # Planetary constants
spice.furnsh("SPICE_kernels/maven_orb_rec_190401_190701_v1.bsp") # MAVEN orbit

# Convert UTC to ephemeris time
et = spice.utc2et("2019-04-11T13:27:04") 

suns_state, _ = spice.spkezr("SUN", et, "J2000", "NONE", "4") #Sun state (position and velocity) wrt Mars
# We are using a Mars centered frame, so the distance is just the magnitude of the position vector
sun_mars_dist_km = spice.vnorm(suns_state[:3]) #[:3] represents the position, the rest is the velocity
sun_mars_dist_m = sun_mars_dist_km * 1000
sun_mars_dist_au = sun_mars_dist_m / 1.495978707e11 

# ----- Reading in solar spectrum -------
solar_spec_columns = [
    "date", "nominal_date_jdn",
    "min_wavelength", "max_wavelength",
    "instrument_mode_id", "data_version",
    "irradiance", "irradiance_uncertainty", "quality"
]

solar_spec_df = pd.read_csv(
    "sorce_muv_solarspec.txt",
    comment=";",        # ignore all lines starting with ';'
    sep='\s+',
    names=solar_spec_columns
)

# Take midpoint of wavelength bins to have a single "representative" wavelength
solar_spec_df["wavelength_mid"] = (solar_spec_df["min_wavelength"] + solar_spec_df["max_wavelength"]) / 2

# Keep rows whose date starts with 20190411
april11 = solar_spec_df[solar_spec_df["date"].astype(str).str.startswith("20190411")]

solar_irradiance_earth = april11["irradiance"] # W m^-2 nm^-1
solar_wl = april11["wavelength_mid"] # nm

# Convert wavelengths to be in meters
solar_wl_m = solar_wl * (1e-9) # m

# photon m^-2 s^-1 nm^-1
# Solar photon flux at Earth (Convert to photon flux using E=hc/lambda)
N_photon_earth = (solar_irradiance_earth * solar_wl_m) / (planck_constant*speed_light)

solar_flux_incident_atmosphere = (N_photon_earth) / (sun_mars_dist_au**2) # Scaled to Mars


# # Test plot 1: After scaling by 1/R^2
# plt.figure(figsize=(10,6))
# plt.plot(solar_wl, solar_flux_incident_atmosphere, color="red", label="At Mars (1.40 AU)")
# plt.plot(solar_wl, N_photon_earth, color="blue", label="At Earth (1.00 AU)")

# plt.title("Solar Spectrum at Earth and Mars April 11 2019")
# plt.xlabel('Wavelength (nm)')
# plt.ylabel("Irradiance (photon per m^2 per s per nm)")
# plt.legend()
# plt.show()




# ----- Intermission :) -----




#  ------ KYLE'S METHOD (REFURBISHED TO BE SAM'S METHOD) ------

# ------ PSF -------

PSF_filepath = "iuvspsf_muv_save.sav" # PSF for the MUV detector 

PSF_data = readsav(PSF_filepath) # Read in the file

PSF_waveall = PSF_data["waveall"] # Å
PSF_psfarr = PSF_data["psfarr"] # Relative amplitude, dimensionless; shape (1024,181) = (spatial bins, ???)
PSF_fwhm = PSF_data["fwhm"] # Å
PSF_waveall_nm = PSF_waveall * 0.1  # nm (1 Å = 0.1 nm)

# Masking the PSF wavelengths to be over the SOLSTICE range

# Step 1: Identify PSF indices that fall within SOLSTICE wavelength range
wl_min_solstice = solar_wl.min()
wl_max_solstice = solar_wl.max()

# Mask: True where PSF wavelength is within SOLSTICE range
psf_mask = (PSF_waveall_nm >= wl_min_solstice) & (PSF_waveall_nm <= wl_max_solstice)

# Mask PSF average
PSF_waveall_nm_masked = PSF_waveall_nm[psf_mask]






# Plot 1D PSF vs ??? axis for 8 representative spatial bins (just to visualize spread)
# Define ??? axis
x = np.arange(PSF_psfarr.shape[1])  # 0..180

# Pick 8 representative spatial bins, every 128 bins
indices = np.arange(0, 1024, 128)

plt.figure(figsize=(10,6))

for i in indices:
    psf_line = PSF_psfarr[i]  # shape (181,)
    wl = PSF_waveall[i]
    
    # Normalize for plotting
    psf_line_norm = psf_line / np.max(psf_line)
    
    plt.plot(x, psf_line_norm, label=f'Spatial Bin {i}')

plt.xlabel('???')
plt.ylabel('Normalized Amplitude')
plt.title('Representative 1D PSFs Across Spatial Bins')
plt.legend()
plt.show()










# ------ PSF CONVOLUTION STEP ------

# --- Flux-preserving interpolation using the new function ---

# Define the wavelength edges per pixel
wl_min = april11["min_wavelength"].to_numpy()
wl_max = april11["max_wavelength"].to_numpy()

# Top-of-atmosphere flux per nm
J_TOA = solar_flux_incident_atmosphere  # shape (# of spectral bins,)

# Make sure your flux is a proper Quantity
flux_density_SOLSTICE = u.Quantity(J_TOA, unit=(u.photon / (u.s * u.m**2 * u.nm)))

# Make sure wavelength is a Quantity
solar_wl_q = u.Quantity(solar_wl, unit=u.nm)

# Create the Spectrum object
input_spectrum_SOLSTICE = Spectrum(flux=flux_density_SOLSTICE,spectral_axis=solar_wl_q
)

# Target PSF grid
PSF_wls = u.Quantity(PSF_waveall_nm_masked, unit=u.nm)

# Flux-conserving resampler
resampler = FluxConservingResampler()
resampled_spectrum = resampler(input_spectrum_SOLSTICE, PSF_wls)

# Extract J_fine
J_fine = resampled_spectrum.flux.to(u.photon / (u.s * u.m**2 * u.nm)).value





# Test plot 2: Rebinning SOLSTICE --> PSF
plt.figure(figsize=(10,6))
plt.plot(solar_wl, solar_flux_incident_atmosphere, color="blue", label="Original, SOLSTICE bins")
plt.plot(PSF_waveall_nm_masked, J_fine, color="red", label="Rebinned, IUVS PSF Bins")

plt.title("Solar Spectrum at Mars, Original and Rebinned")
plt.xlabel('Wavelength (nm)')
plt.ylabel("Irradiance (photon per m^2 per s per nm)")
plt.legend()
plt.show()





# ----- Convolve ! -----

# because we are taking the convolution on the PSF wavelength  grid (of which the data is a subset of),
# this is what we want already because it accounts for edge effects where PSF is blurring into
# regions for which the data is not defined on

# ----- METHOD 1: 1D CONVOLUTION WITH AVERAGE PSF OVER SPATIAL AXIS -----

# Define PSF bin widths explicitly, it is roughly constant through checking
PSF_bin_width = 0.16445923  # nm

# ensures correct PSF bin width
wl_min_psf = PSF_waveall_nm_masked - 0.5 * PSF_bin_width
wl_max_psf = PSF_waveall_nm_masked + 0.5 * PSF_bin_width

psf_avg = np.mean(PSF_psfarr, axis=0)      # (spectral,)
psf_avg /= psf_avg.sum() * PSF_bin_width   # integral-normalized

# Convolve J_fine with this averaged PSF
# Multiply by bin width because convolve computes a sum, we need to turn it into an integral
J_conv_matrix = convolve(J_fine, psf_avg, mode='same') * PSF_bin_width  # full array, flux is conserved




# Test plot 3: Before & After Convolution
plt.figure(figsize=(10,6))
plt.plot(PSF_waveall_nm_masked, J_fine, color="blue", label="SOLSTICE: Rebinned, IUVS PSF Bins")
plt.plot(PSF_waveall_nm_masked, J_conv_matrix, color="red", label="SOLSTICE: Convolved")

plt.title("Solar Spectrum at Mars, Before and After Convolution")
plt.xlabel('Wavelength (nm)')
plt.ylabel("Irradiance (photon per m^2 per s per nm)")
plt.legend()
plt.show()




# --- Verify flux preservation before & after convolution ---

# Total flux before convolution
F_before = np.trapezoid(J_fine, PSF_waveall_nm_masked)

# Total flux after convolution
F_after = np.trapezoid(J_conv_matrix, PSF_waveall_nm_masked)

print(F_before, F_after, F_after / F_before)





# # --- Integrate convolved spectrum back to IUVS bins ---



# --- Step 1: Make Spectrum on PSF fine grid ---
flux_psf_fine = u.Quantity(J_conv_matrix, unit=(u.photon / (u.s * u.m**2 * u.nm)))
wl_psf_fine_q = u.Quantity(PSF_waveall_nm_masked, unit=u.nm)

spec_psf_fine = Spectrum(flux=flux_psf_fine, spectral_axis=wl_psf_fine_q)

# --- Step 2: Define IUVS bin centers ---
wl_IUVS_centers_q_l1b = u.Quantity(wl_l1b, unit=u.nm)
wl_IUVS_centers_q_l2b = u.Quantity(wl_l2b, unit=u.nm)

# --- Step 3: Resample using FluxConservingResampler ---
resampler = FluxConservingResampler()
resampled_spectrum_l1b = resampler(spec_psf_fine, wl_IUVS_centers_q_l1b)
resampled_spectrum_l2b = resampler(spec_psf_fine, wl_IUVS_centers_q_l2b)

J_solar_per_nm_l1b = resampled_spectrum_l1b.flux.value
J_solar_per_nm_l2b = resampled_spectrum_l2b.flux.value


# Test plot 4: Before & After Rebinning back to IUVS Bins
plt.figure(figsize=(10,6))
plt.plot(PSF_waveall_nm_masked, J_conv_matrix, color="blue", label="SOLSTICE: Convolved")
plt.plot(wl_l1b, J_solar_per_nm_l1b, color="red", label="SOLSTICE: Convolved & Rebinned back to IUVS l1b")
plt.plot(wl_l2b, J_solar_per_nm_l2b, color="green", label="SOLSTICE: Convolved & Rebinned back to IUVS l2b")

plt.title("Solar Spectrum at Mars, Before and After Rebinning Back to IUVS Bins (Post Convolution)")
plt.xlabel('Wavelength (nm)')
plt.ylabel("Irradiance (photon per m^2 per s per nm)")
plt.legend()
plt.show()


# -------- Plot I/F vs Wavelength --------

# Units of spectra_l1b is kR/nm

spectra_l1b = np.array(spectra_l1b) # Turn into array with shape (N_pixels, N_spectral bins)
spectra_l2b = np.array(spectra_l2b)

measured_radiance_ph_per_bin_l1b = (spectra_l1b * 1000) * (1/(4 * np.pi)) * 1e10  # photons/s/m^2/sr/nm
measured_radiance_ph_per_bin_l2b = (spectra_l2b * 1000) * (1/(4 * np.pi)) * 1e10  

# Compute radiance factor per wavelength
radiance_factor_per_wl_l1b = (4 * np.pi * measured_radiance_ph_per_bin_l1b) / J_solar_per_nm_l1b
radiance_factor_per_wl_l2b = (4 * np.pi * measured_radiance_ph_per_bin_l2b) / J_solar_per_nm_l2b

# radiance_factor_per_wl_l1b has shape (N_pixels, N_spectral bins)

# # Average every 3 pixels to smooth noise, reduce plot clutter
# rf_binned_l1b = radiance_factor_per_wl_l1b.reshape(
#     57 // 3,
#     3,
#     radiance_factor_per_wl_l1b.shape[1]
# ).mean(axis=1)

plt.figure(figsize=(10, 5))

# Dayside
for i in range(radiance_factor_per_wl_l1b.shape[0]):
    plt.plot(wl_l1b, radiance_factor_per_wl_l1b[i, :], alpha=0.4)

# Nightside
for i in range(radiance_factor_per_wl_l2b.shape[0]):
    plt.plot(wl_l2b, radiance_factor_per_wl_l2b[i, :], linestyle='--', alpha=0.4)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Radiance Factor I/F')
plt.title('I/F per Pixel (No Binning)')
plt.show()




# Test plot (5 and 6): Non-Convolved vs Convolved I/F

# --- Step 1: Make Spectrum from original SOLSTICE flux (no convolution) ---
flux_TOA_q = u.Quantity(J_TOA, unit=(u.photon / (u.s * u.m**2 * u.nm)))
wl_TOA_q   = u.Quantity(solar_wl, unit=u.nm)

spec_TOA = Spectrum(flux=flux_TOA_q, spectral_axis=wl_TOA_q)

# --- Step 2: Resample using FluxConservingResampler ---
resampler = FluxConservingResampler()
resampled_spec_noconvolve_l1b = resampler(spec_TOA, wl_IUVS_centers_q_l1b)
resampled_spec_noconvolve_l2b = resampler(spec_TOA, wl_IUVS_centers_q_l2b)

# --- Step 3: Get flux per nm directly ---
J_solar_per_nm_l1b_noconvolve = resampled_spec_noconvolve_l1b.flux.to(u.photon / (u.s * u.m**2 * u.nm)).value
J_solar_per_nm_l2b_noconvolve = resampled_spec_noconvolve_l2b.flux.to(u.photon / (u.s * u.m**2 * u.nm)).value


plt.figure(figsize=(10,6))
plt.plot(solar_wl, J_TOA, color="blue", label="Solar Spectrum at Mars, SOLSTICE Bins")
plt.plot(wl_l1b, J_solar_per_nm_l1b_noconvolve, color="red", label="Solar Spectrum at Mars, IUVS l1b Bins")
plt.plot(wl_l2b, J_solar_per_nm_l2b_noconvolve, color="green", label="Solar Spectrum at Mars, IUVS l2b Bins")


plt.title("Solar Spectrum with SOLSTICE and IUVS Binning")
plt.xlabel('Wavelength (nm)')
plt.ylabel("Irradiance (photon per m^2 per s per nm)")
plt.legend()
plt.show()

# Compute radiance factor per wavelength
radiance_factor_per_wl_l1b_noconvolve = (4 * np.pi * measured_radiance_ph_per_bin_l1b) / J_solar_per_nm_l1b_noconvolve
radiance_factor_per_wl_l2b_noconvolve = (4 * np.pi * measured_radiance_ph_per_bin_l2b) / J_solar_per_nm_l2b_noconvolve

# # Average every 3 pixels to smooth noise, reduce plot clutter
# rf_binned_noconvolve = radiance_factor_per_wl_l1b_noconvolve.reshape(
#     57 // 3,
#     3,
#     radiance_factor_per_wl_l1b.shape[1]
# ).mean(axis=1)

plt.figure(figsize=(10,6))

# Non-convolved curves
for i in range(radiance_factor_per_wl_l1b_noconvolve.shape[0]):
    plt.plot(
        wl_l1b,
        radiance_factor_per_wl_l1b_noconvolve[i, :],
        color='blue',
        alpha=0.4,
        label='Non-Convolved l1b' if i == 0 else ""
    )
    
for i in range(radiance_factor_per_wl_l2b_noconvolve.shape[0]):
    plt.plot(
        wl_l2b,
        radiance_factor_per_wl_l2b_noconvolve[i, :],
        color='blue',
        alpha=0.4,
        label='Non-Convolved l2b' if i == 0 else ""
    )

# Convolved curves
for i in range(radiance_factor_per_wl_l1b.shape[0]):
    plt.plot(
        wl_l1b,
        radiance_factor_per_wl_l1b[i, :],
        color='red',
        alpha=0.4,
        label='Convolved l1b' if i == 0 else ""
    )

for i in range(radiance_factor_per_wl_l2b.shape[0]):
    plt.plot(
        wl_l2b,
        radiance_factor_per_wl_l2b[i, :],
        color='red',
        alpha=0.4,
        label='Convolved l2b' if i == 0 else ""
    )

plt.title("Non-Convolved vs Convolved I/F (No Averaging)")
plt.xlabel('Wavelength (nm)')
plt.ylabel("Radiance Factor I/F")
plt.yscale("log")
plt.legend()
plt.show()




# ----- Color curves by average SZA of the three pixels -----

# --- Bin SZA ---
sza_array_l1b = np.asarray(sza_selected_l1b)   # shape (57,)
sza_array_l2b = np.asarray(sza_selected_l2b)   # shape (35,)

# sza_binned = sza_array.reshape(57 // 3, 3).mean(axis=1)  # shape (19,)

# --- Curve-level mask ---
sza_mask_l1b = sza_array_l1b <= 112
sza_mask_l2b = sza_array_l2b <= 125

# Apply mask consistently
sza_binned_l1b = sza_array_l1b[sza_mask_l1b]
radiance_factor_per_wl_l1b_masked = radiance_factor_per_wl_l1b[sza_mask_l1b, :]

sza_binned_l2b = sza_array_l2b[sza_mask_l2b]
radiance_factor_per_wl_l2b_masked = radiance_factor_per_wl_l2b[sza_mask_l2b, :]

# --- Colormap normalization using masked SZA ---
norm = colors.Normalize(vmin=np.min(sza_binned_l1b), vmax=np.max(sza_binned_l2b))
cmap = cm.viridis

fig, ax = plt.subplots(figsize=(10, 5))

for i in range(len(sza_binned_l1b)):
    ax.plot(
        wl_l1b,
        radiance_factor_per_wl_l1b_masked[i, :],  # masked per-pixel data
        color=cmap(norm(sza_binned_l1b[i])),
        alpha=0.7
    )

# Colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Solar Zenith Angle (deg)")

ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Radiance Factor I/F')

plt.yscale('log')
plt.show()




# ----- Plot Apparent Scattering, Scaled by cos(EA) (=Optical depth!) -----

# --- μ₀ ---

mu0_l1b = np.abs(np.cos(np.deg2rad(ea_selected_l1b)))
mu0_l2b = np.abs(np.cos(np.deg2rad(ea_selected_l2b)))

# mu0_binned_l1b = mu0.reshape(57 // 3, 3).mean(axis=1)
mu0_binned_l1b = mu0_l1b[sza_mask_l1b]
mu0_binned_l2b = mu0_l2b[sza_mask_l2b]

# --- Apparent scattering ---
# apparent_scattering = rf_binned_masked * mu0_binned[:, None]

apparent_scattering_l1b = radiance_factor_per_wl_l1b_masked * mu0_binned_l1b[:, None]
apparent_scattering_l2b = radiance_factor_per_wl_l2b_masked * mu0_binned_l2b[:, None]

# ----- Plot Apparent Reflectance, Colored by SZA -----
fig, ax = plt.subplots(figsize=(10, 5))

for i in range(apparent_scattering_l1b.shape[0]):
    ax.plot(
        wl_l1b,
        apparent_scattering_l1b[i, :],
        color=cmap(norm(sza_binned_l1b[i]))
    )
    
for i in range(apparent_scattering_l2b.shape[0]):
    ax.plot(
        wl_l2b,
        apparent_scattering_l2b[i, :],
        color=cmap(norm(sza_binned_l2b[i]))
    )

# Axis labels and title
ax.set_xlabel('Wavelength (nm)', fontsize=22, labelpad=15, fontweight='bold')
ax.set_ylabel('Radiance Factor I/F', fontsize=22, labelpad=15, fontweight='bold')

# Tick styling
ax.tick_params(axis='both', labelsize=22)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

bounds = np.linspace(90, 125, 256)  # match SZA mask range
cbar = plt.colorbar(sm, ax=ax, boundaries=bounds)
cbar.set_label("Solar Zenith Angle (deg)", fontsize=22, labelpad=15, fontweight='bold')

tick_vals = np.linspace(90, 125, 8)
cbar.set_ticks(tick_vals)
cbar.set_ticklabels([f"{t:.0f}" for t in tick_vals])
cbar.ax.tick_params(labelsize=22)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')

plt.yscale('log')
ax.set_xlim(np.min(wl_l1b), np.max(wl_l2b))

plt.tight_layout()
plt.show()









sza_l1b = sza_array_l1b[sza_mask_l1b]
sza_l2b = sza_array_l2b[sza_mask_l2b]

rf_l1b = radiance_factor_per_wl_l1b[sza_mask_l1b, :]
rf_l2b = radiance_factor_per_wl_l2b[sza_mask_l2b, :]

mu0_l1b = np.abs(np.cos(np.deg2rad(np.asarray(ea_selected_l1b))))[sza_mask_l1b]
mu0_l2b = np.abs(np.cos(np.deg2rad(np.asarray(ea_selected_l2b))))[sza_mask_l2b]

def custom_bin_by_sza(sza, rf, mu0):
    sza_bin, rf_bin, tau_bin = [], [], []

    # Define the bin ranges and group sizes
    bin_ranges = [(90, 100, 3), (100, 115, 7), (115, 125, 2)]

    for lo, hi, group_size in bin_ranges:
        mask = (sza >= lo) & (sza < hi)
        idx = np.where(mask)[0]

        for i in range(0, len(idx), group_size):
            group = idx[i:i+group_size]
            sza_bin.append(sza[group].mean())
            rf_bin.append(rf[group, :].mean(axis=0))
            tau_bin.append((rf[group, :] * mu0[group, None]).mean(axis=0))

    return np.array(sza_bin), np.array(rf_bin), np.array(tau_bin)

# Apply to L1B and L2B
sza_binned_l1b, rf_binned_l1b, apparent_scattering_l1b = custom_bin_by_sza(
    sza_l1b, rf_l1b, mu0_l1b
)

sza_binned_l2b, rf_binned_l2b, apparent_scattering_l2b = custom_bin_by_sza(
    sza_l2b, rf_l2b, mu0_l2b
)

fig, ax = plt.subplots(figsize=(10, 5))

norm = colors.Normalize(vmin=90, vmax=125)
cmap = cm.viridis

for i in range(apparent_scattering_l1b.shape[0]):
    ax.plot(wl_l1b, apparent_scattering_l1b[i, :], color=cmap(norm(sza_binned_l1b[i])))

for i in range(apparent_scattering_l2b.shape[0]):
    ax.plot(wl_l2b, apparent_scattering_l2b[i, :], color=cmap(norm(sza_binned_l2b[i])))

ax.set_xlabel('Wavelength (nm)', fontsize=22, labelpad=15, fontweight='bold')
ax.set_ylabel('Radiance Factor I/F', fontsize=22, labelpad=15, fontweight='bold')
ax.tick_params(axis='both', labelsize=22)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Solar Zenith Angle (deg)", fontsize=22, labelpad=15, fontweight='bold')
cbar.ax.tick_params(labelsize=22)
for label in cbar.ax.get_yticklabels():
    label.set_fontweight('bold')
    
# Flip only the numbers on the vertical colorbar
cbar.ax.invert_yaxis()  # now 90 at bottom, 125 at top

plt.yscale('log')
ax.set_xlim(np.min(wl_l1b)+5.6219000837728, np.max(wl_l2b)) # Clip first few wavelength bin which we dont think is real "blueing"
plt.tight_layout()
plt.show()