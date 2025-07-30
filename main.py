"""
Complete Final Contrast Curve Generator with Debugging for Minima
================================================================

Modified version with extensive debugging output to diagnose minima issues
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter1d
from scipy.signal import find_peaks
from astropy.io import fits
from pathlib import Path
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


def load_fits_image(filename):
    """Load a FITS image with robust header handling."""
    try:
        # First attempt - normal loading
        with fits.open(filename) as hdul:
            data = hdul[0].data
            header = dict(hdul[0].header)
        return data, header
    except Exception as e:
        print(f"FITS header error detected: {e}")
        print("Attempting to fix corrupted FITS header...")

        # Second attempt - fix header
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with fits.open(filename, mode='update') as hdul:
                    hdul.verify('fix')
                    data = hdul[0].data
                    # Convert to dict to avoid further issues
                    header = {}
                    for card in hdul[0].header.cards:
                        try:
                            header[card.keyword] = card.value
                        except:
                            # Skip corrupted cards
                            pass
                print("FITS header successfully repaired.")
            return data, header
        except Exception as e2:
            print(f"Could not fix header, loading with minimal header: {e2}")
            # Final attempt - ignore header errors
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    hdul = fits.open(filename, ignore_missing_end=True)
                    data = hdul[0].data
                    hdul.close()
                header = {'COMMENT': 'Header reconstructed due to corruption'}
                return data, header
            except Exception as e3:
                raise ValueError(f"Could not load FITS file. Error: {e3}")


def find_star_center(image):
    """Find the brightest pixel as star center."""
    y, x = np.unravel_index(np.argmax(image), image.shape)
    return x, y


def find_all_extrema_in_annulus(image, center, r_inner, r_outer, footprint=3):
    """
    Determine ALL local maxima and minima in the annulus.
    Following the paper method exactly.
    """
    # Create annulus mask
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    cx, cy = center
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    annulus_mask = (dist >= r_inner) & (dist < r_outer)

    # Find ALL local maxima in the annulus
    max_filtered = maximum_filter(image, size=footprint)
    is_local_max = (image == max_filtered) & annulus_mask

    # Find ALL local minima in the annulus
    min_filtered = minimum_filter(image, size=footprint)
    is_local_min = (image == min_filtered) & annulus_mask

    # Remove pixels that are both maxima and minima (flat regions)
    is_local_min = is_local_min & ~is_local_max

    # Get all maxima values and positions
    max_values = image[is_local_max]
    max_positions = np.where(is_local_max)
    max_distances = np.sqrt((max_positions[1] - cx) ** 2 + (max_positions[0] - cy) ** 2)

    # Get all minima values and positions
    min_values = image[is_local_min]
    min_positions = np.where(is_local_min)
    min_distances = np.sqrt((min_positions[1] - cx) ** 2 + (min_positions[0] - cy) ** 2)

    return max_values, max_distances, min_values, min_distances


def calculate_contrast_curve_paper_method(image, pixel_scale=0.0135, min_radius=2,
                                          max_radius=90, annulus_width=3,
                                          telescope_diameter=2.54, wavelength=617e-9,
                                          target_name="Unknown Target", telescope_name="Hooker"):
    """
    Calculate contrast curve following the paper method exactly:
    Detection limit = mean(maxima) + 5 * average(σ_maxima, σ_minima)
    """

    # Find star center and flux
    center = find_star_center(image)
    star_flux = np.max(image)

    # Calculate additional parameters
    lambda_over_d = (wavelength / telescope_diameter) * 206265  # arcsec

    # Measure PSF FWHM
    fwhm_pixels = measure_psf_fwhm(image, center)
    fwhm_arcsec = fwhm_pixels * pixel_scale

    # Get background and dynamic range
    corner_size = min(50, image.shape[0] // 4, image.shape[1] // 4)
    corners = []
    corners.append(image[:corner_size, :corner_size])
    corners.append(image[:corner_size, -corner_size:])
    corners.append(image[-corner_size:, :corner_size])
    corners.append(image[-corner_size:, -corner_size:])

    corner_medians = [np.median(c.flatten()) for c in corners]
    darkest_corner = corners[np.argmin(corner_medians)]
    background = np.median(darkest_corner)
    noise_std = np.std(darkest_corner)
    dynamic_range = star_flux / max(noise_std, 1e-10)

    print(f"\n=== Contrast Curve Analysis ===")
    print(f"Star center: ({center[0]:.1f}, {center[1]:.1f})")
    print(f"Star flux: {star_flux:.2e}")
    print(f"Dynamic range: {dynamic_range:.1f}")
    print(f"PSF FWHM: {fwhm_pixels:.2f} pixels ({fwhm_arcsec:.3f} arcsec)")

    # ===== NEW DEBUGGING INFO =====
    print(f"\n=== Image Statistics ===")
    print(f"Image min value: {np.min(image):.2e}")
    print(f"Image max value: {np.max(image):.2e}")
    print(f"Image mean: {np.mean(image):.2e}")
    print(f"Background (darkest corner median): {background:.2e}")
    print(f"Any negative values in image? {np.any(image < 0)}")

    # Storage for all extrema (for plotting)
    all_max_separations = []
    all_max_values = []
    all_min_separations = []
    all_min_values = []

    # Storage for detection curve
    radii = []
    detection_limits = []

    # Add origin point
    radii.append(0)
    detection_limits.append(star_flux)

    # Process each annulus
    for r in range(min_radius, max_radius, 1):
        r_inner = r - annulus_width / 2
        r_outer = r + annulus_width / 2

        # Find ALL local maxima and minima in this annulus
        max_vals, max_dists, min_vals, min_dists = find_all_extrema_in_annulus(
            image, center, r_inner, r_outer, footprint=3
        )

        # Store extrema for plotting
        if len(max_vals) > 0:
            all_max_separations.extend(max_dists)
            all_max_values.extend(max_vals)

        if len(min_vals) > 0:
            all_min_separations.extend(min_dists)
            all_min_values.extend(min_vals)

        # Calculate detection limit following paper method
        if len(max_vals) > 0:
            # Step 1: Mean value of the maxima
            mean_maxima = np.mean(max_vals)

            # Step 2: Standard deviation of maxima
            if len(max_vals) > 1:
                sigma_maxima = np.std(max_vals, ddof=1)  # Sample std
            else:
                sigma_maxima = 0.0  # Can't calculate std with 1 point

            # Step 3: Standard deviation of minima
            if len(min_vals) > 1:
                sigma_minima = np.std(min_vals, ddof=1)  # Sample std
            elif len(min_vals) == 1:
                sigma_minima = 0.0  # Can't calculate std with 1 point
            else:
                # No minima - use sigma of maxima
                sigma_minima = sigma_maxima

            # Step 4: Average sigma of maxima and minima
            avg_sigma = (sigma_maxima + sigma_minima) / 2.0

            # Step 5: Detection limit = mean(maxima) + 5 * avg_sigma
            detection_limit = mean_maxima + 5 * avg_sigma

            # Debug info for first few annuli
            if r < min_radius + 5:
                print(f"\nRadius {r} pixels:")
                print(f"  {len(max_vals)} maxima, {len(min_vals)} minima")
                print(f"  mean(maxima) = {mean_maxima:.2f}")
                print(f"  σ(maxima) = {sigma_maxima:.2f}")
                print(f"  σ(minima) = {sigma_minima:.2f}")
                print(f"  avg σ = {avg_sigma:.2f}")
                print(f"  detection limit = {detection_limit:.2f}")

        else:
            # No maxima in this annulus - skip
            continue

        radii.append(r)
        detection_limits.append(detection_limit)

    # Convert to arrays
    all_max_separations = np.array(all_max_separations)
    all_max_values = np.array(all_max_values)
    all_min_separations = np.array(all_min_separations)
    all_min_values = np.array(all_min_values)
    radii = np.array(radii)
    detection_limits = np.array(detection_limits)

    # Smooth detection limits curve
    if len(detection_limits) > 10:
        # Apply smoothing (skip origin)
        smoothed = gaussian_filter1d(detection_limits[1:], sigma=2.5)
        detection_limits[1:] = smoothed

    # ===== NEW DEBUGGING INFO FOR EXTREMA =====
    print(f"\n=== Extrema Value Statistics ===")
    print(f"Total maxima found: {len(all_max_values)}")
    if len(all_max_values) > 0:
        print(f"Maxima value range: {np.min(all_max_values):.2e} to {np.max(all_max_values):.2e}")
        print(f"Maxima mean: {np.mean(all_max_values):.2e}")
        print(f"First 10 maxima values: {all_max_values[:10]}")

    print(f"\nTotal minima found: {len(all_min_values)}")
    if len(all_min_values) > 0:
        print(f"Minima value range: {np.min(all_min_values):.2e} to {np.max(all_min_values):.2e}")
        print(f"Minima mean: {np.mean(all_min_values):.2e}")
        print(f"First 10 minima values: {all_min_values[:10]}")
        print(f"Any negative minima? {np.any(all_min_values < 0)}")
        print(f"Number of minima < 1e-10: {np.sum(all_min_values < 1e-10)}")

    # Convert to magnitudes
    # Ensure no zeros or negative values
    noise_floor = 1e-10

    # Maxima magnitudes
    all_max_values = np.maximum(all_max_values, noise_floor)
    max_mags = -2.5 * np.log10(all_max_values / star_flux)

    # Minima magnitudes
    if len(all_min_values) > 0:
        # Use absolute value of minima for magnitude calculation
        # This treats the "depth" of valleys as a positive signal
        min_values_abs = np.abs(all_min_values)
        min_values_abs = np.maximum(min_values_abs, noise_floor)
        min_mags = -2.5 * np.log10(min_values_abs / star_flux)

        # Debug info for minima correction
        print(f"\n=== Minima Correction (Absolute Value Method) ===")
        print(f"Original minima range: {np.min(all_min_values):.2e} to {np.max(all_min_values):.2e}")
        print(f"Absolute value range: {np.min(min_values_abs):.2e} to {np.max(min_values_abs):.2e}")
        print(f"Expected magnitude range: {np.min(min_mags):.2f} to {np.max(min_mags):.2f}")
    else:
        min_mags = np.array([])

    # Detection limit magnitudes
    detection_limits = np.maximum(detection_limits, noise_floor)
    limit_mags = -2.5 * np.log10(detection_limits / star_flux)

    # ===== NEW DEBUGGING INFO FOR MAGNITUDES =====
    print(f"\n=== Magnitude Statistics ===")
    if len(max_mags) > 0:
        print(f"Maxima magnitude range: {np.min(max_mags):.2f} to {np.max(max_mags):.2f}")
        print(f"First 10 maxima mags: {max_mags[:10]}")

    if len(min_mags) > 0:
        print(f"Minima magnitude range: {np.min(min_mags):.2f} to {np.max(min_mags):.2f}")
        print(f"First 10 minima mags: {min_mags[:10]}")
        print(f"Number of minima mags > 10: {np.sum(min_mags > 10)}")
        print(f"Number of minima mags > 15: {np.sum(min_mags > 15)}")
        print(f"Number of minima mags > 20: {np.sum(min_mags > 20)}")

    # Convert pixels to arcsec
    all_max_separations_arcsec = all_max_separations * pixel_scale
    all_min_separations_arcsec = all_min_separations * pixel_scale
    radii_arcsec = radii * pixel_scale

    print(f"\nTotal: {len(all_max_values)} maxima and {len(all_min_values)} minima found")

    # Return all needed info for plotting
    return {
        'max_seps': all_max_separations_arcsec,
        'max_mags': max_mags,
        'min_seps': all_min_separations_arcsec,
        'min_mags': min_mags,
        'radii': radii_arcsec,
        'limits': limit_mags,
        'fwhm_pixels': fwhm_pixels,
        'fwhm_arcsec': fwhm_arcsec,
        'lambda_over_d': lambda_over_d,
        'dynamic_range': dynamic_range,
        'pixel_scale': pixel_scale
    }


def measure_psf_fwhm(image, center):
    """Measure PSF FWHM from radial profile."""
    cx, cy = center
    max_radius = 20
    radii = np.arange(0, max_radius, 0.5)
    profile = []

    for r in radii:
        if r == 0:
            profile.append(image[int(cy), int(cx)])
        else:
            # Sample points in a circle
            angles = np.linspace(0, 2 * np.pi, max(8, int(2 * np.pi * r)))
            x_coords = cx + r * np.cos(angles)
            y_coords = cy + r * np.sin(angles)

            values = []
            for x, y in zip(x_coords, y_coords):
                if 0 <= x < image.shape[1] - 1 and 0 <= y < image.shape[0] - 1:
                    x0, y0 = int(x), int(y)
                    dx, dy = x - x0, y - y0

                    val = (1 - dx) * (1 - dy) * image[y0, x0] + \
                          dx * (1 - dy) * image[y0, x0 + 1] + \
                          (1 - dx) * dy * image[y0 + 1, x0] + \
                          dx * dy * image[y0 + 1, x0 + 1]
                    values.append(val)

            if values:
                profile.append(np.mean(values))

    profile = np.array(profile)

    # Find FWHM
    peak_val = profile[0]
    half_max = peak_val / 2.0

    # Find where profile drops below half max
    below_half = np.where(profile < half_max)[0]
    if len(below_half) > 0:
        fwhm_radius = radii[below_half[0]]
        fwhm_pixels = 2 * fwhm_radius
    else:
        fwhm_pixels = 4.0  # Default

    return fwhm_pixels


def plot_contrast_curve_full(results, title="Contrast Curve", telescope_name="Hooker",
                             telescope_diameter=2.54, wavelength=617e-9, confidence_level=5.0,
                             save_path=None, show_plot=True, date_obs=None):
    """Plot contrast curve with full annotations."""

    # Extract results
    max_seps = results['max_seps']
    max_mags = results['max_mags']
    min_seps = results['min_seps']
    min_mags = results['min_mags']
    radii = results['radii']
    limits = results['limits']

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Add title with target name
    fig.suptitle(f'{title} - {telescope_name} {telescope_diameter:.1f}m Telescope',
                 fontsize=14, fontweight='bold')

    # ===== NEW DEBUGGING INFO FOR PLOTTING =====
    print(f"\n=== Plotting Statistics ===")
    print(f"Y-axis will be set to: 0 to {max(10, np.max(limits) + 0.5):.1f}")
    if len(min_mags) > 0:
        print(f"Minima mags that would be visible (< {max(10, np.max(limits) + 0.5):.1f}): "
              f"{np.sum(min_mags < max(10, np.max(limits) + 0.5))}")
        print(f"Minima mags that are off the plot: {np.sum(min_mags >= max(10, np.max(limits) + 0.5))}")

    # Plot local maxima (empty squares)
    ax.scatter(max_seps, max_mags, marker='s', s=30,
               facecolors='none', edgecolors='black',
               linewidth=0.5, alpha=0.7, label='Local Maxima')

    # Plot local minima (small dots)
    if len(min_mags) > 0:
        ax.scatter(min_seps, min_mags, marker='.', s=10,
                   color='black', alpha=0.7, label='Local Minima')

    # Plot 5σ detection limit curve
    ax.plot(radii, limits, 'r-', linewidth=2.5,
            label=f'{confidence_level}σ Detection Limit')

    # Find and annotate the peak delta-magnitude closest to the nearest detection
    if len(limits) > 10:  # Need enough points
        # First, check if we have any detections to reference
        peak_idx = None
        peak_sep = None
        peak_limit = None

        # Find detections (maxima below the limit curve)
        if len(max_seps) > 0:
            limit_at_max = np.interp(max_seps, radii, limits)
            detections = max_mags < limit_at_max

            if np.any(detections):
                # Find the detection with smallest separation
                det_seps = max_seps[detections]
                closest_detection_sep = np.min(det_seps)

                # Find all local maxima in the detection limit curve (skip origin)
                peaks, _ = find_peaks(limits[1:])  # Skip origin point
                peaks = peaks + 1  # Adjust indices to account for skipped origin

                if len(peaks) > 0:
                    # Find the peak closest to the closest detection
                    peak_distances_to_detection = np.abs(radii[peaks] - closest_detection_sep)
                    closest_peak_idx = np.argmin(peak_distances_to_detection)
                    peak_idx = peaks[closest_peak_idx]
                else:
                    # No local maxima found, use global maximum (skip origin)
                    peak_idx = np.argmax(limits[1:]) + 1

        # If no detections found, just use the global maximum (skip origin)
        if peak_idx is None:
            peak_idx = np.argmax(limits[1:]) + 1

        peak_sep = radii[peak_idx]
        peak_limit = limits[peak_idx]

        # Convert to pixels for annotation
        peak_sep_pixels = peak_sep / results['pixel_scale']

        # Dynamic positioning for peak annotation
        x_range = np.max(radii) - np.min(radii)
        y_range = np.max(limits) - np.min(limits)

        # Position peak annotation in upper RIGHT area of plot
        peak_annotation_x = peak_sep + 0.2 * x_range
        peak_annotation_y = peak_limit + 0.3 * y_range

        # Make sure annotation stays within plot bounds
        peak_annotation_x = min(peak_annotation_x, np.max(radii) - 0.05 * x_range)
        peak_annotation_y = min(peak_annotation_y, np.max(limits) + 0.4 * y_range)

        # If peak is too close to the right edge, position annotation to the left
        if peak_sep > 0.7 * np.max(radii):
            peak_annotation_x = peak_sep - 0.2 * x_range
            peak_annotation_x = max(peak_annotation_x, np.min(radii) + 0.05 * x_range)

        # Annotate the peak delta-magnitude point
        annotation_text = (f'Peak Δm:\n' +
                           f'Δm = {peak_limit:.1f} at {peak_sep:.2f}" ({peak_sep_pixels:.0f} pix)')
        ax.annotate(annotation_text,
                    xy=(peak_sep, peak_limit),
                    xytext=(peak_annotation_x, peak_annotation_y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1),
                    fontsize=10, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))

    # Find and highlight detections (maxima below the limit)
    if len(max_seps) > 0:
        # Interpolate limit at maxima positions
        limit_at_max = np.interp(max_seps, radii, limits)
        detections = max_mags < limit_at_max

        if np.any(detections):
            det_seps = max_seps[detections]
            det_mags = max_mags[detections]
            ax.scatter(det_seps, det_mags, marker='o', s=100,
                       facecolors='none', edgecolors='red', linewidth=2,
                       label='Detections')
            print(f"\nFound {np.sum(detections)} detections below {confidence_level}σ limit")

            # Annotate the brightest detection
            if len(det_mags) > 0:
                brightest_idx = np.argmin(det_mags)
                sep = det_seps[brightest_idx]
                mag = det_mags[brightest_idx]

                # Get the limiting magnitude at this separation
                limiting_mag = np.interp(sep, radii, limits)

                # Dynamic positioning for detection annotation
                x_range = np.max(radii) - np.min(radii)
                y_range = np.max(limits) - np.min(limits)

                # Position detection annotation in upper LEFT area of plot
                det_annotation_x = sep - 0.3 * x_range
                det_annotation_y = mag + 0.4 * y_range

                # Make sure annotation stays within plot bounds
                det_annotation_x = max(det_annotation_x, np.min(radii) + 0.05 * x_range)
                det_annotation_y = min(det_annotation_y, np.max(limits) + 0.4 * y_range)

                # If detection is too close to the left edge, position annotation to the right
                if sep < 0.3 * np.max(radii):
                    det_annotation_x = sep + 0.2 * x_range
                    det_annotation_x = min(det_annotation_x, np.max(radii) - 0.05 * x_range)

                # Annotation text
                sep_pixels = sep / results['pixel_scale']
                annotation_text = (f'Δm = {mag:.2f}\n' +
                                   f'Limiting Δm = {limiting_mag:.2f}\n' +
                                   f'Separation = {sep:.3f}" ({sep_pixels:.1f} pix)')
                ax.annotate(annotation_text,
                            xy=(sep, mag),
                            xytext=(det_annotation_x, det_annotation_y),
                            arrowprops=dict(arrowstyle='->', color='black', lw=0.5),
                            fontsize=10, ha='left',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    # Labels and formatting
    ax.set_xlabel('Separation [arcsec]', fontsize=13)
    ax.set_ylabel('Magnitude Difference', fontsize=13)

    # Secondary x-axis for pixels
    ax2 = ax.twiny()
    ax2.set_xlabel('Separation [pixels]', fontsize=13)
    x_min, x_max = ax.get_xlim()
    ax2.set_xlim(x_min / results['pixel_scale'], x_max / results['pixel_scale'])

    # Make sure the pixel axis ticks are at nice round numbers
    pixel_ticks = np.arange(0, x_max / results['pixel_scale'] + 10, 10)
    ax2.set_xticks(pixel_ticks)

    # Legend in upper right
    ax.legend(fontsize=10, loc='upper right', frameon=True, fancybox=False)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Set y-axis limits
    ax.set_ylim(0, max(10, np.max(limits) + 0.5))

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax2.tick_params(axis='both', which='major', labelsize=11)

    # Add information box
    info_text = []
    info_text.append(f'Target: {title}')
    info_text.append(f'Telescope: {telescope_name} {telescope_diameter:.2f}m')
    info_text.append(f'Wavelength: {wavelength * 1e9:.0f} nm')
    info_text.append(f'Pixel scale: {results["pixel_scale"]:.4f}"/pix ({results["pixel_scale"] * 1000:.1f} mas/pix)')
    info_text.append(f'PSF FWHM: {results["fwhm_arcsec"]:.3f}" ({results["fwhm_pixels"]:.1f} pix)')
    info_text.append(
        f'λ/D: {results["lambda_over_d"]:.3f}" ({results["lambda_over_d"] / results["pixel_scale"]:.1f} pix)')
    info_text.append(f'Dynamic Range: {results["dynamic_range"]:.0f}:1')
    info_text.append(f'Confidence: {confidence_level}σ')

    if date_obs:
        try:
            if 'T' in str(date_obs):
                date_part = str(date_obs).split('T')[0]
            else:
                date_part = str(date_obs)
            info_text.append(f'Date: {date_part}')
        except:
            pass

    # Create the text box
    info_str = '\n'.join(info_text)
    ax.text(0.98, 0.02, info_str, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.9))

    # Use a white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to: {save_path}")

    if show_plot:
        plt.show()

    return fig


class FinalContrastCurveGUI:
    """GUI for final contrast curve generation."""

    def __init__(self, root):
        self.root = root
        self.root.title("Contrast Curve Generator - Final Version")
        self.root.geometry("800x800")

        self.image = None
        self.header = None
        self.results = None

        # Variables
        self.file_path = tk.StringVar()
        self.target_name = tk.StringVar(value="Unknown Target")
        self.confidence_level = tk.DoubleVar(value=5.0)
        self.pixel_scale = tk.DoubleVar(value=0.0135)
        self.wavelength = tk.DoubleVar(value=617)
        self.telescope_name = tk.StringVar(value="Hooker")
        self.telescope_diameter = tk.DoubleVar(value=100.0)  # inches
        self.max_radius = tk.IntVar(value=90)
        self.min_radius = tk.IntVar(value=2)
        self.save_path = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        """Create GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Input File", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Entry(file_frame, textvariable=self.file_path, width=60).grid(row=0, column=0, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=1)
        ttk.Button(file_frame, text="Load", command=self.load_file).grid(row=0, column=2)

        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="5")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Target Name
        ttk.Label(param_frame, text="Target Name:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.target_name, width=30).grid(row=0, column=1, columnspan=3, sticky=tk.W)

        # Telescope
        ttk.Label(param_frame, text="Telescope Name:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.telescope_name, width=15).grid(row=1, column=1)

        ttk.Label(param_frame, text="Diameter (inches):").grid(row=1, column=2, sticky=tk.W)
        ttk.Spinbox(param_frame, from_=1.0, to=500.0, increment=1.0,
                    textvariable=self.telescope_diameter, width=10).grid(row=1, column=3)

        # Confidence Level
        ttk.Label(param_frame, text="Confidence Level (σ):").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(param_frame, from_=1.0, to=10.0, increment=0.1,
                    textvariable=self.confidence_level, width=10).grid(row=2, column=1)

        # Radius range
        ttk.Label(param_frame, text="Min Radius (pixels):").grid(row=3, column=0, sticky=tk.W)
        ttk.Spinbox(param_frame, from_=1, to=50, increment=1,
                    textvariable=self.min_radius, width=10).grid(row=3, column=1)

        ttk.Label(param_frame, text="Max Radius (pixels):").grid(row=3, column=2, sticky=tk.W)
        ttk.Spinbox(param_frame, from_=10, to=200, increment=10,
                    textvariable=self.max_radius, width=10).grid(row=3, column=3)

        # Pixel scale and wavelength
        ttk.Label(param_frame, text="Pixel Scale (arcsec/pixel):").grid(row=4, column=0, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.pixel_scale, width=10).grid(row=4, column=1)

        ttk.Label(param_frame, text="Wavelength (nm):").grid(row=4, column=2, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.wavelength, width=10).grid(row=4, column=3)

        # Filter presets
        filter_frame = ttk.Frame(param_frame)
        filter_frame.grid(row=5, column=0, columnspan=4, pady=5)
        ttk.Label(filter_frame, text="Filter Presets:").pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="Sloan r'", command=lambda: self.set_filter(617)).pack(side=tk.LEFT, padx=2)
        ttk.Button(filter_frame, text="Sloan i'", command=lambda: self.set_filter(748)).pack(side=tk.LEFT, padx=2)

        # Processing
        process_frame = ttk.LabelFrame(main_frame, text="Processing", padding="5")
        process_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(process_frame, text="Generate Contrast Curve",
                   command=self.generate_curve).grid(row=0, column=0, padx=5)

        # Save
        save_frame = ttk.LabelFrame(main_frame, text="Save Results", padding="5")
        save_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Entry(save_frame, textvariable=self.save_path, width=50).grid(row=0, column=0, padx=5)
        ttk.Button(save_frame, text="Browse", command=self.browse_save).grid(row=0, column=1)
        ttk.Button(save_frame, text="Save Plot", command=self.save_plot).grid(row=0, column=2)

        # Status
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.status_text = tk.Text(status_frame, height=10, width=70)
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        self.log_status("Contrast Curve Generator Ready")
        self.log_status("Method: mean(maxima) + 5×avg(σ_max, σ_min)")

    def log_status(self, message):
        """Add message to status area."""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update()

    def set_filter(self, wavelength_nm):
        """Set filter wavelength."""
        self.wavelength.set(wavelength_nm)
        filter_name = "Sloan r'" if wavelength_nm == 617 else "Sloan i'"
        self.log_status(f"Set filter to {filter_name} ({wavelength_nm} nm)")

    def browse_file(self):
        """Browse for FITS file."""
        filename = filedialog.askopenfilename(
            title="Select FITS file",
            filetypes=[("FITS files", "*.fits *.fit"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)

    def browse_save(self):
        """Browse for save location."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filename:
            self.save_path.set(filename)

    def load_file(self):
        """Load the selected FITS file."""
        filename = self.file_path.get()
        if not filename:
            messagebox.showerror("Error", "Please select a file first")
            return

        try:
            self.log_status(f"\nLoading: {Path(filename).name}")
            self.image, self.header = load_fits_image(filename)
            self.log_status(f"Image shape: {self.image.shape}")

            # Try to get parameters from header
            header_dict = dict(self.header)

            for key in ['PIXSCALE', 'PLATESCL', 'SCALE']:
                if key in header_dict:
                    self.pixel_scale.set(float(header_dict[key]))
                    self.log_status(f"Pixel scale from header: {self.pixel_scale.get()}")
                    break

            # Get target name
            if 'OBJECT' in header_dict:
                self.target_name.set(str(header_dict['OBJECT']))

            self.log_status("File loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            self.log_status(f"Error: {str(e)}")

    def generate_curve(self):
        """Generate the contrast curve."""
        if self.image is None:
            messagebox.showerror("Error", "Please load a FITS file first")
            return

        try:
            self.log_status("\nGenerating contrast curve...")

            # Calculate curve
            telescope_diameter_m = self.telescope_diameter.get() * 0.0254  # inches to meters

            results = calculate_contrast_curve_paper_method(
                self.image,
                pixel_scale=self.pixel_scale.get(),
                min_radius=self.min_radius.get(),
                max_radius=self.max_radius.get(),
                telescope_diameter=telescope_diameter_m,
                wavelength=self.wavelength.get() * 1e-9,
                target_name=self.target_name.get(),
                telescope_name=self.telescope_name.get()
            )

            self.results = results

            # Get date from header if available
            date_obs = None
            if hasattr(self, 'header'):
                header_dict = dict(self.header)
                date_obs = header_dict.get('DATE-OBS', None)

            # Plot
            self.fig = plot_contrast_curve_full(
                results,
                title=self.target_name.get(),
                telescope_name=self.telescope_name.get(),
                telescope_diameter=telescope_diameter_m,
                wavelength=self.wavelength.get() * 1e-9,
                confidence_level=self.confidence_level.get(),
                date_obs=date_obs
            )

            self.log_status("Contrast curve generated successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate curve: {str(e)}")
            self.log_status(f"Error: {str(e)}")
            import traceback
            self.log_status(traceback.format_exc())

    def save_plot(self):
        """Save the current plot."""
        if not hasattr(self, 'fig'):
            messagebox.showerror("Error", "Generate a curve first")
            return

        filename = self.save_path.get()
        if not filename:
            messagebox.showerror("Error", "Please specify a save location")
            return

        try:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            self.log_status(f"Saved: {Path(filename).name}")
            messagebox.showinfo("Success", "Plot saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")


# Command line interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Command line mode
        image, header = load_fits_image(sys.argv[1])

        # Get parameters from header
        header_dict = dict(header)
        pixel_scale = 0.0135  # Default
        for key in ['PIXSCALE', 'PLATESCL', 'SCALE']:
            if key in header_dict:
                pixel_scale = float(header_dict[key])
                print(f"Using pixel scale from header: {pixel_scale}")
                break

        # Get target name
        target_name = header_dict.get('OBJECT', 'Unknown Target')
        date_obs = header_dict.get('DATE-OBS', None)

        # Calculate curve
        results = calculate_contrast_curve_paper_method(
            image,
            pixel_scale=pixel_scale,
            target_name=target_name
        )

        # Plot
        plot_contrast_curve_full(
            results,
            title=target_name,
            date_obs=date_obs
        )
    else:
        # GUI mode
        root = tk.Tk()
        app = FinalContrastCurveGUI(root)
        root.mainloop()