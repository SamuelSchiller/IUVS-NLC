# CODE DEVELOPED BY SAMUEL SCHILLER, EDITED 05/11/2026

# Importing libraries
import time
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
import spiceypy as spice

# Load kernels
spice.kclear()
spice.furnsh("naif0012.tls") # leapseconds
spice.furnsh("de430.bsp") # planetary ephemeris
spice.furnsh("mar097s.bsp")
spice.furnsh("maven_orb_rec_150101_150401_v1.bsp") # MAVEN orbit
spice.furnsh("pck00009.tpc")  # Planetary constants
spice.furnsh("maven_v11.tf") # Frames

def unit_vector(v):
    return v / np.linalg.norm(v)

# Constants
c = spice.clight()  # Speed of light in km/s
R_mars_ellipsoid = [3396.19, 3396.19, 3376.20]  # Mars ellipsoid radii in km (mso2000 values)

# Star coordinates manually input (RA, Dec in degrees)
ra_deg = 263.40220833333335
dec_deg = -37.10384722222222
ra = np.radians(ra_deg)
dec = np.radians(dec_deg)

# Unit vector toward star in J2000
starvect_j2000 = np.array([np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)])

# Ephemeris time (observation time midpoint for now, because cloud persists across all orbits)
et_obs = spice.utc2et("2015-3-26T11:52:02.83696")

# Get Mars, spacecraft and Sun position relative to Solar System Barycenter (SSB=0)
mars_pos, _ = spice.spkpos("4", et_obs, "J2000", "NONE", "0")  # Mars barycenter relative to SSB
sc_pos, _ = spice.spkpos("MAVEN", et_obs, "J2000", "NONE", "0")  # Spacecraft relative to SSB
sun_pos, _ = spice.spkpos("SUN", et_obs, "J2000", "NONE", "0")   # Sun relative to SSB

# Initial vector SC to Mars and light time guess
vect_scmars = np.array(mars_pos) - np.array(sc_pos)
lt_corr = np.linalg.norm(vect_scmars) / c

# --- Iterative light-time correction ---

for _ in range(3):

    et_emit = et_obs - lt_corr

    # Get positions at emission time relative to SSB
    mars_pos_emit, _ = spice.spkpos("4", et_emit, "J2000", "NONE", "0")
    sun_pos_emit, _ = spice.spkpos("SUN", et_emit, "J2000", "NONE", "0")

    # Rotate from J2000 to MAVEN_MSO for general calculations
    rot_j2m = spice.pxform("J2000", "MAVEN_MSO", et_emit)
    # Also need J2000 to IAU_MARS for npedln
    rot_j2_iau = spice.pxform("J2000", "IAU_MARS", et_emit)

    # Calculate spacecraft position relative to Mars center in J2000
    sc_pos_rel_mars_j2000 = np.array(sc_pos) - np.array(mars_pos_emit)

    # Transform spacecraft position to MAVEN_MSO (for sun_mso, P2 calculations later)
    sc_mso = spice.mxv(rot_j2m, sc_pos_rel_mars_j2000)

    # Transform LOS vector to MAVEN_MSO
    vlos_mso = spice.mxv(rot_j2m, starvect_j2000)

    # Transform Sun position relative to Mars center to MAVEN_MSO
    sun_to_mars_vec_j2000 = sun_pos_emit - mars_pos_emit
    sun_mso = spice.mxv(rot_j2m, sun_to_mars_vec_j2000)

    # --- Tangent Point Calculation in IAU_MARS ---
    # Transform spacecraft position and LOS vector to IAU_MARS frame
    sc_iau_mars = spice.mxv(rot_j2_iau, sc_pos_rel_mars_j2000)
    vlos_iau_mars = spice.mxv(rot_j2_iau, starvect_j2000)

    # Calculate nearest point on Mars ellipsoid in IAU_MARS frame
    # R_mars_ellipsoid is aligned with IAU_MARS axes
    pnear_iau, alt_iau = spice.npedln(R_mars_ellipsoid[0], R_mars_ellipsoid[1], R_mars_ellipsoid[2], sc_iau_mars, vlos_iau_mars)

    # Project nearest point onto LOS vector in IAU_MARS frame
    pnear_proj_iau, dist_iau = spice.nplnpt(sc_iau_mars, vlos_iau_mars, pnear_iau)

    # Update light time based on distance to LOS intercept in IAU_MARS
    lt_corr = np.linalg.norm(pnear_proj_iau - sc_iau_mars) / c

    # First get rotation from IAU_MARS to J2000
    rot_iau_j2 = spice.pxform("IAU_MARS", "J2000", et_emit)
    # Then rotate to MAVEN_MSO
    pnear_proj = spice.mxv(rot_j2m, spice.mxv(rot_iau_j2, pnear_proj_iau))

# Final outputs
print("SC position in MAVEN_MSO frame (km):", sc_mso)
print("LOS vector in MAVEN_MSO frame:", vlos_mso)
print("Sun position in MAVEN_MSO frame (km):", sun_mso)
print("Tangent altitude (km) from IAU_MARS calculation:", alt_iau) # Use alt_iau
print("Tangent coordinates in MAVEN_MSO frame (km):", pnear_proj)

R_mars = R_mars_ellipsoid[0] # Use the equatorial radius as a representative single radius for spherical calculations

def compute_solar_zenith_angle(point_mso, sun_mso):
    """
    Compute the solar zenith angle at `point_pos` given the Sun's Mars‐fixed position.
    """
    zenith = unit_vector(point_mso)                  # spherical‐approx local normal
    sun_dir = unit_vector(sun_mso - point_mso)
    cos_sza = np.dot(zenith, sun_dir)
    return np.degrees(np.arccos(np.clip(cos_sza, -1, 1)))

# ——— FIND CLOUD 2 VIA LOS PARAMETERIZATION ———

Hmax = 100.0  # maximum altitude to search, km
sza_obs = 125 # Observed SZA for given event (highest SZA extent of orbit 8889)

def los_roots(h2):
    """Return sorted positive roots s for |maven_pos + s*los_vector|^2 = (R_mars+h2)^2."""
    A = 1  # los_vector is unit
    B = 2 * np.dot(sc_mso, vlos_mso)
    C = np.dot(sc_mso, sc_mso) - (R_mars+h2)**2
    disc = B*B - 4*A*C
    if disc < 0:
        return []
    s1 = (-B - np.sqrt(disc)) / (2*A)
    s2 = (-B + np.sqrt(disc)) / (2*A)
    return sorted(s for s in (s1, s2) if s > 0)

# Helper function 
def check_occultation(ray_origin, ray_target, sphere_radius):
    """
    Checks if the line segment from ray_origin to ray_target intersects a sphere
    centered at the origin. Returns True if the segment intersects the sphere.
    """
    O = ray_origin  # Since sphere_center = [0, 0, 0]
    D = ray_target - ray_origin # Ray direction vector

    a = np.dot(D, D)
    b = 2 * np.dot(O, D)
    c = np.dot(O, O) - sphere_radius**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return False

    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)

    epsilon = 1e-9
    if (epsilon < t1 < 1 - epsilon) or (epsilon < t2 < 1 - epsilon):
        return True

    return False

# Calculate intersections with the Hmax shell (R_mars + Hmax)
# This will return sorted positive roots if the LOS intersects the sphere of R_mars + Hmax
s_enter, s_exit = los_roots(Hmax)

# Define the scan range (s0 to s1) for atmospheric analysis. We use Hmax because it defines the largest possible region. 
# s0: The beginning of the atmospheric column of interest along the LOS.
# s1: The end of the atmospheric column of interest along the LOS.

# Search region is intersection points of Hmax shell, any lower altitude cloud will be contained within region
s0 = s_enter
s1 = s_exit

N_scan     = 5001
s_values   = np.linspace(s0, s1, N_scan)
best_error = np.inf
best_s     = None
best_sza   = None

for s in s_values:
    P2_cand     = sc_mso + s * vlos_mso
    h_cand      = np.linalg.norm(P2_cand) - R_mars

    if h_cand < 0:
        continue   # skip any below‐surface

    sza_cand    = compute_solar_zenith_angle(P2_cand, sun_mso)
    err         = abs(sza_cand - sza_obs)
    if err < best_error:
        best_error = err
        best_s     = s
        best_sza   = sza_cand

    P2_best    = sc_mso + best_s * vlos_mso
    h2_best    = np.linalg.norm(P2_best) - R_mars

print("\n=== LOS‐Based Cloud 2 Fit ===")
print(f"→ Best s           = {best_s:.3f} km along LOS")
print(f"→ Altitude h₂      = {h2_best:.3f} km")
print(f"→ SZA at P₂        = {best_sza:.3f}° (obs {sza_obs:.3f}°)")
print(f"→ Error            = {best_error:.3f}°")

if best_error <= 0.1:
    print("✔ Within 0.1° tolerance.")
else:
    print("⚠︎ Exceeds 0.1° tolerance; consider refining or root‐finding.")

# 6) Use best-fit Cloud 2 for all remaining analysis
P2 = P2_best
h2 = h2_best

# Calculate Cloud 2 position
# For 2D: place Cloud 2 at angle corresponding to the SZA
# If SZA = 121.03°, then angle from +x axis = 180° - 121.03° = 58.97°
phi2 = np.deg2rad(180 - sza_obs)  # Convert to radians
r2 = R_mars + h2
P2 = np.array([r2 * np.cos(phi2), r2 * np.sin(phi2)])

# Sun direction (parallel rays from the left, -x direction)
sun_dir = np.array([-1.0, 0.0]) # This is important for is_point_in_sunlight logic

# --- Helper Functions ---

def is_point_in_sunlight(point, planet_center, planet_radius):
    """
    Checks if a point is illuminated by parallel sun rays coming from the left (-x direction).
    A point is in sunlight if the horizontal ray from (-infinity, point.y) to point
    does NOT intersect the planet.
    """
    # Define a point far to the left at the same y-coordinate as 'point'
    # This simulates the origin of the parallel sun ray.
    ray_start = np.array([-planet_radius * 5, point[1]]) # 5 times planet radius is sufficiently far

    # If the line segment from ray_start to 'point' intersects the planet,
    # then 'point' is in shadow. Therefore, if it does NOT intersect, it's sunlit.
    return not check_occultation(ray_start, point, planet_radius)

# --- Main Logic ---

# Search grid
phis   = np.linspace(phi2, np.pi, 5000)  # φ for Cloud 1
h_grid = np.linspace(0, 100, 5000)       # altitudes [0..100 km]

all_pts_list = []   # will hold one (n_h,2) array per φ
fil_pts_list = []   # will hold valid P1’s

for phi in phis:
    # build candidate points for this φ
    h_arr = R_mars + h_grid                   # shape (500,)
    x1    = h_arr * np.cos(phi)
    y1    = h_arr * np.sin(phi)
    P1s   = np.stack((x1, y1), axis=1)        # shape (500,2)
    all_pts_list.append(P1s)

    # 1) clear LOS to P2?
    d12    = P2 - P1s                         # (500,2)
    top    = -np.einsum('ij,ij->i', P1s, d12)
    bot    = np.einsum('ij,ij->i', d12, d12)
    t_line = np.clip(top/bot, 0, 1)           # (500,)
    proj12 = P1s + (d12.T * t_line).T         # (500,2)
    dmin12 = np.linalg.norm(proj12, axis=1)   # (500,)
    mask1  = dmin12 > R_mars                  # clear of planet

    # 2) sunlit?
    ray_start = np.column_stack((-5*R_mars*np.ones_like(y1), y1))
    d_sun     = P1s - ray_start
    top2      = -np.einsum('ij,ij->i', ray_start, d_sun)
    bot2      = np.einsum('ij,ij->i', d_sun, d_sun)
    t_sun     = np.clip(top2/bot2, 0, 1)
    proj_sun  = ray_start + (d_sun.T * t_sun).T
    dmin_sun  = np.linalg.norm(proj_sun, axis=1)
    mask2     = dmin_sun > R_mars             # stays above planet

    # 3) 5 km tangent‐ray clearance?
    mask3 = dmin12 > (R_mars + 5.0)           # reuses dmin12

   # --- 4) Vectorized Sun-ray minimum height (≥ 5 km) ---
    n_samples = 50
    t = np.linspace(0, 1, n_samples)       # samples along ray
    t_exp = t[None, None, :]               # shape (1,1,n_samples)
    
    positions = ray_start[:, :, None] + d_sun[:, :, None] * t_exp  # (n_h,2,n_samples)
    ys_along_rays = positions[:, 1, :]                             # y-coords
    min_heights = np.min(ys_along_rays - R_mars, axis=1)
    mask4 = min_heights >= 5.0

    # 5) collect filtered
    good = mask1 & mask2 & mask3 & mask4
    if np.any(good):
        fil_pts_list.append(P1s[good])
        
# stack into (N,2) arrays
all_pts = np.vstack(all_pts_list)
fil_pts = np.vstack(fil_pts_list) if fil_pts_list else np.empty((0,2))

# --- Plotting ---
fig, ax = plt.subplots(figsize=(12, 14))
fig.patch.set_facecolor("black")   # figure background
ax.set_facecolor("black")          # plot background

# Make axis labels and tick labels white
ax.tick_params(axis='both', colors='white')       # ticks
ax.xaxis.label.set_color('white')                 # x-axis label
ax.yaxis.label.set_color('white')                 # y-axis label

# Make the spines (axis lines) white
for spine in ax.spines.values():
    spine.set_color('white')

# Make the title white too
ax.title.set_color('white')

# Axis labels with specific size
ax.set_xlabel('X (km)', fontsize=14, fontweight='bold')
ax.set_ylabel('Y (km)', fontsize=14, fontweight='bold')

# Tick labels with specific size
ax.tick_params(axis='both', which='major', labelsize=14, colors='white')
ax.tick_params(axis='both', which='minor', labelsize=12, colors='white')

# Set limits
xlim_min, xlim_max = -0.12 * R_mars, 0.82 * R_mars
ylim_min, ylim_max = 0.80 * R_mars, 1.1 * R_mars
ax.set_xlim(xlim_min, xlim_max)
ax.set_ylim(ylim_min, ylim_max)

# Dayside and nightside hemispheres
theta_dark = np.linspace(-np.pi/2, np.pi/2, 500)
x_dark = R_mars * np.cos(theta_dark)
y_dark = R_mars * np.sin(theta_dark)

theta_light = np.linspace(np.pi/2, 3*np.pi/2, 500)
x_light = R_mars * np.cos(theta_light)
y_light = R_mars * np.sin(theta_light)

# Plot all together
peru_rgb = (0.8039, 0.5216, 0.2471) # Original 'peru' RGB in 0-1 scale
darker_peru = tuple(0.4 * c for c in peru_rgb)
ax.fill(x_dark, y_dark, color=darker_peru, zorder=1)
ax.fill(x_light, y_light, color='peru', zorder=1)

# Plot filtered Cloud 1 points

if fil_pts.size > 0:
    hull = ConvexHull(fil_pts)               # find outer boundary
    polygon_pts = fil_pts[hull.vertices]     # vertices of convex hull

    cloud_outline = Polygon(
        polygon_pts,
        closed=True,
        facecolor='none',   # no filling
        edgecolor='red',    # only the outline
        linewidth=2,
        zorder=10
    )
    ax.add_patch(cloud_outline)

# --- Squiggly photon rays with arrowheads ---
n_rays = 5
amplitude = 0.001 * R_mars   # wiggle amplitude
wavelength = 0.01 * R_mars   # wiggle wavelength
x_ray = np.linspace(xlim_min, 0.8*R_mars, 1000)  # stop at Mars limb

for i in range(1, n_rays):
    # base heights
    y_base = R_mars + (0 + 0.020*i) * R_mars
    
    # sinusoidal wiggle
    y_ray = y_base + amplitude * np.sin(2*np.pi * x_ray / wavelength)
    ax.plot(x_ray, y_ray, color='gold', lw=1.5, zorder=2, alpha=0.5)

    # Flat horizontal arrowhead for squiggly photon
    arrow_tip_x, arrow_tip_y = x_ray[-1], y_ray[-1]
    
    ax.annotate("",
        xy=(arrow_tip_x+25, arrow_tip_y),                  # arrow tip
        xytext=(arrow_tip_x - 0.001*R_mars, arrow_tip_y),  # tail
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.4,head_length=0.4",  # arrowhead size
            color="gold", lw=1.5, alpha=0.4
        ),
        zorder=3
    )

# --- Solar ray from -x to Cloud 1, then to Cloud 2 ---

def draw_ray(ax, A, B, *, color="gold", lw=2, dashed=False, z=25, scale=14):
    """
    Draw a straight arrow from A -> B in data coordinates with the tip exactly at B.
    """
    patch = FancyArrowPatch(
        A, B,
        arrowstyle='-|>',          # closed head, sits on the end
        shrinkA=0, shrinkB=0,      # DO NOT trim either end
        mutation_scale=scale,      # head size (in points, scaled)
        lw=lw,
        linestyle="--" if dashed else "-",
        color=color,
        transform=ax.transData,    # ensure positions are in data coords
        zorder=z,
    )
    ax.add_patch(patch)

# --- Solar ray from -x to Cloud 1 to Cloud 2 to spacecraft ---

if fil_pts.size > 0:
    # 1) select points along the horizontal bottom edge (y = R_mars)
    bottom_edge_pts = fil_pts[np.isclose(fil_pts[:,1], fil_pts[:,1].min(), atol=1e-3)]
    bottom_left_pt = bottom_edge_pts[np.argmin(bottom_edge_pts[:,0])]

    # --- Draw Rays ---
    # Ray 1: from far left to bottom-left point
    ray_start = np.array([xlim_min, bottom_left_pt[1]])
    draw_ray(ax, ray_start, bottom_left_pt, color="gold", lw=2, dashed=False, z=25, scale=14)

    # Ray 2: bottom-left point → Cloud 2
    draw_ray(ax, bottom_left_pt, P2, color="gold", lw=2, dashed=False, z=25, scale=14)

    # Ray 3: Cloud 2 → spacecraft
    spacecraft_pt = P2 + np.array([0.07*R_mars, 0.07*R_mars])
    draw_ray(ax, P2, spacecraft_pt, color="gold", lw=2, dashed=False, z=25, scale=14)

    # Label spacecraft
    ax.text(spacecraft_pt[0]+0.155*R_mars, spacecraft_pt[1]-0.3,
            r'$\mathbf{To\ Spacecraft}$',
            fontsize=14, color="white",
            ha="left", va="top")
    
# Ice particle cloud

n_particles = 1000
phi_start = np.deg2rad(180 - 90)        # terminator
phi_end   = np.deg2rad(180 - sza_obs)   # observed cloud edge

phi_particles = np.random.uniform(phi_end, phi_start, n_particles)
r_particles = np.random.uniform(R_mars+70, R_mars+100, n_particles)

x_particles = r_particles * np.cos(phi_particles)
y_particles = r_particles * np.sin(phi_particles)

# Determine bottom of Cloud 1
if fil_pts.size > 0:
    y_bottom_cloud1 = fil_pts[:,1].min()
else:
    y_bottom_cloud1 = R_mars  # fallback

# Initialize color array
colors = np.zeros((n_particles, 3))

# Boolean mask for particles above/below Cloud 1 bottom
mask_above = y_particles > y_bottom_cloud1

# Assign colors directly as strings
colors = np.where(mask_above, "white", "dimgray")

ax.scatter(x_particles, y_particles,
           s=1,
           color=colors,
           alpha=0.8,
           zorder=5)

def draw_cartoon_cloud_with_bumps(ax, x_particles, y_particles,
                                  n_bumps=12, bump_irregularity=0.3,
                                  color='white', alpha=0.9,
                                  edgecolor='darkgray', edgewidth=2,
                                  zorder=15):
    """
    Alternative method: Draw a cartoon cloud using a single polygon with
    sinusoidal bumps for the top edge.
    
    Parameters
    ----------
    ax : matplotlib Axes
        Axis to draw on.
    x_particles, y_particles : arrays
        Particle coordinates (used to determine cloud extent).
    n_bumps : int
        Number of bumps along the top.
    bump_irregularity : float
        How irregular the bumps are (0 to 1).
    color : str
        Fill color of cloud.
    alpha : float
        Transparency.
    edgecolor : str
        Edge color for cartoon effect.
    edgewidth : float
        Width of edge line.
    zorder : int
        Drawing order.
    """
    # Convert particles to polar coordinates
    r_particles = np.hypot(x_particles, y_particles)
    phi_particles = np.arctan2(y_particles, x_particles)
    
    phi_min, phi_max = phi_particles.min(), phi_particles.max()
    r_min, r_max = r_particles.min(), r_particles.max()
    cloud_thickness = r_max - r_min
    
    # Create flat arc bottom
    n_points = 300
    phi_bottom = np.linspace(phi_min, phi_max, n_points)
    x_bottom = r_min * np.cos(phi_bottom)
    y_bottom = r_min * np.sin(phi_bottom)
    
    # Create bumpy top
    phi_top = np.linspace(phi_min, phi_max, n_points)
    
    # Create multiple overlapping sine waves for irregular bumps
    bump_amplitude = cloud_thickness * 0.4
    r_top_base = r_max - bump_amplitude * 0.5
    
    # Combine multiple frequencies for natural look
    bumps = np.zeros(n_points)
    for i in range(3):
        frequency = n_bumps * (1 + i * 0.5)
        phase = np.random.random() * 2 * np.pi
        amplitude = bump_amplitude / (i + 1)
        
        # Add irregularity
        irregular_amp = amplitude * (1 + bump_irregularity * 
                                     (np.random.random(n_points) - 0.5))
        bumps += irregular_amp * np.sin(frequency * 
                                        np.linspace(0, 2*np.pi, n_points) + phase)
    
    # Ensure bumps are always positive (cloud puffs outward)
    bumps = np.abs(bumps) + bump_amplitude * 0.3
    r_top = r_top_base + bumps
    
    x_top = r_top * np.cos(phi_top[::-1])
    y_top = r_top * np.sin(phi_top[::-1])
    
    # Combine into polygon
    x_cloud = np.concatenate([x_bottom, x_top])
    y_cloud = np.concatenate([y_bottom, y_top])
    
    # Draw filled cloud
    cloud_poly = Polygon(np.column_stack([x_cloud, y_cloud]),
                        closed=True, facecolor=color,
                        edgecolor=edgecolor if edgecolor else 'none',
                        linewidth=edgewidth,
                        alpha=alpha, zorder=zorder)
    ax.add_patch(cloud_poly)
    
    # Add internal detail lines for cartoon effect
    if edgecolor and edgecolor != 'none':
        # Add a few internal curved lines for cloud texture
        n_internal = 3
        for i in range(n_internal):
            phi_internal = np.linspace(phi_min + (phi_max - phi_min) * 0.2,
                                      phi_max - (phi_max - phi_min) * 0.2, 50)
            r_internal = r_min + (i + 1) * cloud_thickness / (n_internal + 2)
            
            # Add slight wave
            r_internal += np.sin(np.linspace(0, np.pi, 50)) * cloud_thickness * 0.05
            
            x_internal = r_internal * np.cos(phi_internal)
            y_internal = r_internal * np.sin(phi_internal)
            
            ax.plot(x_internal, y_internal,
                   color=edgecolor, linewidth=edgewidth*0.3,
                   alpha=0.3, zorder=zorder + 1)
            
# For the smooth bumpy cloud:
draw_cartoon_cloud_with_bumps(ax, x_particles, y_particles,
                              n_bumps=13,
                              bump_irregularity=0,
                              color='white',
                              alpha=0.3,
                              edgecolor='none',
                              edgewidth=2.5,
                              zorder=15)

# Make axis aspect ratio equal
ax.set_aspect('equal')

# Bold title and labels
ax.set_xlabel('X (km)', fontweight='bold')
ax.set_ylabel('Y (km)', fontweight='bold')

# Text labels directly on Mars' surface
ax.text(-0.06*R_mars, 0.85*R_mars, "Dayside",
        ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')

ax.text(0.07*R_mars, 0.85*R_mars, "Nightside",
        ha='center', va='center',
        fontsize=16, fontweight='bold', color='white')

# Labels for the rays
ax.text(xlim_max*0.80, R_mars + 0.021*R_mars, 
        "Solar Rays", 
        ha='right', va='bottom', 
        fontsize=14, fontweight='bold', color='white')

# Add text at specific SZA
sza_list = [90, 100, 110, 120]  # degrees
labels = ["SZA=90°", "100°", "110°", "120°"]

for sza, label in zip(sza_list, labels):
    # Convert SZA to angle from +x axis
    phi = np.deg2rad(180 - sza)  

    # Place text just below surface (slightly smaller radius)
    r_text = R_mars * 0.985
    x_text = r_text * np.cos(phi)
    y_text = r_text * np.sin(phi)

    # Compute rotation in degrees so text is tangential to the circle
    rotation_deg = -np.rad2deg(phi - np.pi/2)  # -90° aligns along tangent

    # Add rotated text
    ax.text(x_text, y_text, label,
            ha='center', va='center',    # center text on point
            fontsize=14, fontweight='bold',
            color='white',
            rotation=rotation_deg,
            rotation_mode='anchor')      # rotate around text anchor

# Add label for Cloud 2
offset_x = 0.11 * R_mars  # horizontal offset from Cloud 2
offset_y = -0.01 * R_mars  # vertical offset from Cloud 2

ax.text(P2[0] + offset_x, P2[1] + offset_y,
f"Deep Nightside \n Scattering Region",   # descriptive label
fontsize=14, fontweight='bold', color='white', ha='center', va='center')

# Add label for Cloud 1 (illuminated portion)
if fil_pts.size > 0:
    # Compute approximate center of filtered Cloud 1 points
    center_x = np.mean(fil_pts[:, 0])
    center_y = np.mean(fil_pts[:, 1])

    # Offset slightly to avoid overlap
    offset_x = 0.2 * R_mars
    offset_y = 0.017 * R_mars

    ax.text(center_x + offset_x, center_y + offset_y,
            "Sunlit Scattering Region",   # descriptive label
            fontsize=14, fontweight='bold', color='white')

plt.gca().invert_xaxis()

plt.show()

