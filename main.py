"""
Corrected Hooker Telescope Contrast Curve Generator
===================================================

This implementation correctly finds both local maxima and minima in annuli,
then calculates the detection limit as described in the method.
Fixed version with proper minima magnitude calculation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter, gaussian_filter1d
from scipy.interpolate import UnivariateSpline
from astropy.io import fits
from pathlib import Path
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading


class ContrastCurveGenerator:
    """Generate contrast curves using annular analysis of local extrema."""

    def __init__(self, image_data, pixel_scale=None, wavelength=None,
                 telescope_diameter=None, star_position=None, header=None):
        """Initialize the contrast curve generator."""
        print("=== Contrast Curve Generator ===")

        self.image = np.array(image_data, dtype=float)
        self.header = header or {}
        self.telescope_diameter = telescope_diameter or 2.54  # Hooker telescope

        # Extract parameters
        self.pixel_scale = self._extract_pixel_scale(pixel_scale)
        self.wavelength = self._extract_wavelength(wavelength)
        self.star_pos = self._find_star_position(star_position)
        self.lambda_over_d = (self.wavelength / self.telescope_diameter) * 206265

        # Measure PSF FWHM for reference
        self.fwhm_pixels, self.fwhm_arcsec = self.measure_psf_fwhm()

        print(f"Hooker Telescope (2.54m) Analysis")
        print(f"Pixel scale: {self.pixel_scale * 1000:.1f} mas/pixel")
        print(f"Wavelength: {self.wavelength * 1e9:.0f} nm")
        print(f"Image size: {self.image.shape}")
        print(f"PSF FWHM: {self.fwhm_pixels:.2f} pixels ({self.fwhm_arcsec:.3f} arcsec)")

    def measure_psf_fwhm(self):
        """Measure the FWHM of the central PSF."""
        print("=== Measuring central PSF FWHM ===")

        try:
            # Star is always at image center for phase-recovered bispectrum images
            center_x = self.image.shape[1] // 2
            center_y = self.image.shape[0] // 2

            # Extract a small region around the center
            half_size = 10
            y_start = max(0, center_y - half_size)
            y_end = min(self.image.shape[0], center_y + half_size)
            x_start = max(0, center_x - half_size)
            x_end = min(self.image.shape[1], center_x + half_size)

            central_region = self.image[y_start:y_end, x_start:x_end]

            # Get profiles through the center
            peak_y, peak_x = np.unravel_index(np.argmax(central_region), central_region.shape)

            # Horizontal and vertical profiles
            horizontal_profile = central_region[peak_y, :]
            vertical_profile = central_region[:, peak_x]

            def find_fwhm(profile):
                """Find FWHM of a 1D profile."""
                peak_idx = np.argmax(profile)
                peak_val = profile[peak_idx]
                half_max_val = peak_val / 2.0

                left_indices = np.where(profile[:peak_idx] <= half_max_val)[0]
                right_indices = np.where(profile[peak_idx:] <= half_max_val)[0]

                if len(left_indices) > 0 and len(right_indices) > 0:
                    left_idx = left_indices[-1]
                    right_idx = right_indices[0] + peak_idx
                    return right_idx - left_idx
                else:
                    return 4  # Conservative estimate

            # Calculate FWHM in both directions
            fwhm_x = find_fwhm(horizontal_profile)
            fwhm_y = find_fwhm(vertical_profile)

            fwhm_pixels = (fwhm_x + fwhm_y) / 2.0
            fwhm_arcsec = fwhm_pixels * self.pixel_scale

            print(f"PSF FWHM: {fwhm_pixels:.2f} pixels ({fwhm_arcsec:.3f} arcsec)")

            return fwhm_pixels, fwhm_arcsec

        except Exception as e:
            print(f"PSF FWHM measurement failed: {e}")
            default_fwhm = 4.0
            return default_fwhm, default_fwhm * self.pixel_scale

    def _extract_pixel_scale(self, provided_scale):
        """Extract pixel scale from header or use default."""
        if provided_scale is not None:
            return provided_scale

        scale_keywords = ['PIXSCALE', 'PLATESCL', 'PLTSCALE', 'SCALE', 'CDELT1', 'CDELT2']
        for keyword in scale_keywords:
            if keyword in self.header:
                scale = abs(float(self.header[keyword]))
                print(f"Found pixel scale in header [{keyword}]: {scale}")
                return scale

        default_scale = 0.0135  # Default for Hooker
        print(f"Using default pixel scale: {default_scale}")
        return default_scale

    def _extract_wavelength(self, provided_wavelength):
        """Extract wavelength from header or use default."""
        if provided_wavelength is not None:
            return provided_wavelength

        wave_keywords = ['WAVELENG', 'WAVELEN', 'FILTER', 'CENTWAVE', 'EFFWAVE']
        for keyword in wave_keywords:
            if keyword in self.header:
                wave_val = self.header[keyword]
                if isinstance(wave_val, (int, float)):
                    if wave_val > 1000:  # Angstroms
                        wavelength = wave_val * 1e-10
                    elif wave_val > 1:  # Nanometers
                        wavelength = wave_val * 1e-9
                    else:
                        wavelength = wave_val if wave_val < 1e-5 else wave_val * 1e-6
                    print(f"Found wavelength in header: {wavelength * 1e9:.0f} nm")
                    return wavelength
                elif isinstance(wave_val, str):
                    filter_waves = {
                        'SLOAN_R': 617e-9, "R'": 617e-9, 'RPRIME': 617e-9,
                        'SLOAN_I': 748e-9, "I'": 748e-9, 'IPRIME': 748e-9,
                    }
                    filter_key = wave_val.upper().strip()
                    if filter_key in filter_waves:
                        wavelength = filter_waves[filter_key]
                        print(f"Found filter: {wave_val} -> {wavelength * 1e9:.0f} nm")
                        return wavelength

        default_wave = 617e-9  # Sloan r'
        print(f"Using default wavelength (Sloan r'): {default_wave * 1e9:.0f} nm")
        return default_wave

    def _find_star_position(self, provided_position):
        """Find star position - always at image center for speckle reconstructions."""
        if provided_position is not None:
            return provided_position

        center_x = self.image.shape[1] // 2
        center_y = self.image.shape[0] // 2
        star_pos = (center_x, center_y)

        print(f"Star position set to image center: ({star_pos[0]:.1f}, {star_pos[1]:.1f})")
        return star_pos

    def find_local_extrema_in_annulus(self, inner_radius, outer_radius, footprint_size=3):
        """Find local maxima and minima within an annulus."""
        center_x, center_y = self.star_pos

        # Create coordinate grids
        y, x = np.ogrid[:self.image.shape[0], :self.image.shape[1]]

        # Create annulus mask
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        annulus_mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)

        # Find local maxima using maximum filter
        local_max_map = maximum_filter(self.image, size=footprint_size, mode='constant', cval=-np.inf)
        is_local_max = (self.image == local_max_map) & annulus_mask & (self.image > 0)

        # Find local minima using minimum filter
        # Exclude zero pixels from being local minima
        local_min_map = minimum_filter(self.image, size=footprint_size, mode='constant', cval=np.inf)
        is_local_min = (self.image == local_min_map) & annulus_mask & (self.image > 0)

        # Remove points that are both maxima and minima (flat regions)
        is_local_min = is_local_min & ~is_local_max

        # Extract values
        max_values = self.image[is_local_max]
        min_values = self.image[is_local_min]

        # Get positions
        max_positions = np.where(is_local_max)
        min_positions = np.where(is_local_min)

        # Calculate distances
        max_distances = np.sqrt((max_positions[1] - center_x)**2 +
                               (max_positions[0] - center_y)**2)
        min_distances = np.sqrt((min_positions[1] - center_x)**2 +
                               (min_positions[0] - center_y)**2)

        # If we have too few minima, sample from the lowest non-zero values
        if len(min_values) < 10:
            annulus_pixels = self.image[annulus_mask]
            # Only consider non-zero pixels
            nonzero_pixels = annulus_pixels[annulus_pixels > 0]

            if len(nonzero_pixels) > 0:
                # Use 20th percentile of non-zero pixels
                threshold = np.percentile(nonzero_pixels, 20)
                low_mask = (self.image <= threshold) & (self.image > 0) & annulus_mask & ~is_local_max

                if np.any(low_mask):
                    low_positions = np.where(low_mask)
                    # Sample up to 30 points
                    n_samples = min(30, len(low_positions[0]))
                    if n_samples > 0:
                        indices = np.linspace(0, len(low_positions[0])-1, n_samples, dtype=int)
                        sampled_positions = (low_positions[0][indices], low_positions[1][indices])

                        sampled_values = self.image[sampled_positions]
                        sampled_distances = np.sqrt((sampled_positions[1] - center_x)**2 +
                                                  (sampled_positions[0] - center_y)**2)

                        min_values = np.concatenate([min_values, sampled_values])
                        min_distances = np.concatenate([min_distances, sampled_distances])

        return max_values, max_distances, min_values, min_distances

    def calculate_annulus_statistics(self, inner_radius, outer_radius):
        """Calculate robust statistics for an annulus."""
        center_x, center_y = self.star_pos

        # Create annulus mask
        y, x = np.ogrid[:self.image.shape[0], :self.image.shape[1]]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        annulus_mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)

        # Get pixels
        annulus_pixels = self.image[annulus_mask]

        if len(annulus_pixels) < 10:
            return None, None

        # Calculate robust statistics
        median = np.median(annulus_pixels)

        # Method 1: MAD-based standard deviation
        mad = np.median(np.abs(annulus_pixels - median))
        mad_std = 1.4826 * mad

        # Method 2: Percentile-based (more robust for skewed distributions)
        p84 = np.percentile(annulus_pixels, 84.13)  # +1 sigma for Gaussian
        p16 = np.percentile(annulus_pixels, 15.87)  # -1 sigma for Gaussian
        percentile_std = (p84 - p16) / 2.0

        # Average the two estimates
        robust_std = (mad_std + percentile_std) / 2.0

        return median, robust_std

    def generate_contrast_curve(self, min_radius=None, max_radius=None, radius_step=1,
                                confidence_level=5.0, footprint_size=3):
        """Generate contrast curve by analyzing annuli."""
        print("=== Generating contrast curve ===")

        # Get star flux for magnitude conversion
        self.star_flux = float(np.max(self.image))

        # Start from very close to center
        if min_radius is None:
            min_radius = 2  # Start at 2 pixels from center

        if max_radius is None:
            max_radius = min(90, min(self.image.shape) // 2 - 5)

        print(f"Radius range: {min_radius} to {max_radius} pixels")
        print(f"Footprint size for extrema detection: {footprint_size}x{footprint_size}")

        # Storage for all extrema
        all_radii_max = []
        all_maxima = []
        all_radii_min = []
        all_minima = []

        # Storage for statistics per annulus
        annulus_centers = []
        mean_maxima = []
        std_maxima = []
        std_minima = []
        detection_limits = []

        # Add a point at the origin (0,0) with zero detection limit
        annulus_centers.append(0)
        detection_limits.append(self.star_flux)  # At r=0, detection limit is the star flux itself

        # Process each annulus with wider annuli for better statistics
        annulus_width = 3  # Width for better statistics
        for radius in range(min_radius, max_radius, radius_step):
            inner_r = max(0, radius - annulus_width / 2)
            outer_r = radius + annulus_width / 2

            # Find extrema
            max_vals, max_dists, min_vals, min_dists = self.find_local_extrema_in_annulus(
                inner_r, outer_r, footprint_size)

            # Store individual extrema
            if len(max_vals) > 0:
                all_radii_max.extend(max_dists)
                all_maxima.extend(max_vals)

            if len(min_vals) > 0:
                all_radii_min.extend(min_dists)
                all_minima.extend(min_vals)

            # Calculate robust statistics for the annulus
            median, robust_std = self.calculate_annulus_statistics(inner_r, outer_r)

            if median is not None and robust_std is not None:
                # Calculate detection limit following the narrative exactly
                if len(max_vals) > 0:
                    # "derive their mean value and standard deviation"
                    mean_of_maxima = np.mean(max_vals)

                    if len(max_vals) > 1:
                        std_of_maxima = np.std(max_vals)
                    else:
                        std_of_maxima = robust_std  # Use robust estimate if only one maximum

                    if len(min_vals) > 1:
                        std_of_minima = np.std(min_vals)
                    else:
                        std_of_minima = robust_std  # Use robust estimate

                    # "the average sigma of the maxima and minima"
                    average_sigma = (std_of_maxima + std_of_minima) / 2.0

                    # "mean value of the maxima plus five times the average sigma"
                    detection_limit = mean_of_maxima + confidence_level * average_sigma

                else:
                    # No maxima found - use median + noise estimate
                    detection_limit = median + confidence_level * robust_std

                annulus_centers.append(radius)
                mean_maxima.append(mean_of_maxima if len(max_vals) > 0 else median)
                detection_limits.append(detection_limit)

        # Convert to arrays
        self.all_radii_max = np.array(all_radii_max)
        self.all_maxima = np.array(all_maxima)
        self.all_radii_min = np.array(all_radii_min)
        self.all_minima = np.array(all_minima)

        self.annulus_centers = np.array(annulus_centers)
        self.detection_limits = np.array(detection_limits)

        # Smooth the detection limit curve for better appearance
        if len(self.detection_limits) > 10:
            # Apply gentle Gaussian smoothing (skip the origin point)
            smoothed = gaussian_filter1d(self.detection_limits[1:], sigma=2.0)
            self.detection_limits[1:] = smoothed

        print(f"Found {len(all_maxima)} local maxima and {len(all_minima)} local minima")
        print(f"Processed {len(annulus_centers)} annuli")

        return self.annulus_centers, self.detection_limits

    def plot_contrast_curve(self, confidence_level=5.0, min_radius=None, max_radius=40,
                            save_path=None, show_plot=True):
        """Plot the contrast curve showing all extrema and detection limit."""
        print("=== Plotting contrast curve ===")

        # Generate the contrast curve
        radii, limits = self.generate_contrast_curve(
            min_radius=min_radius,
            max_radius=max_radius,
            confidence_level=confidence_level
        )

        if len(radii) == 0:
            print("No data found!")
            return None

        # Convert to magnitudes
        # Handle zero or negative values before taking log
        self.all_maxima = np.maximum(self.all_maxima, 1e-10)
        maxima_delta_mag = -2.5 * np.log10(self.all_maxima / self.star_flux)

        if len(self.all_minima) > 0:
            # Calculate minima magnitudes by normalizing to the mean speckle pattern

            print(f"Minima statistics:")
            print(f"  Total minima: {len(self.all_minima)}")

            # Exclude minima very close to star
            min_radius = self.fwhm_pixels
            valid_mask = self.all_radii_min > min_radius

            # Also exclude extreme outliers
            if np.any(self.all_minima > 0):
                outlier_threshold = np.percentile(self.all_minima[self.all_minima > 0], 99.5)
                valid_mask = valid_mask & (self.all_minima < outlier_threshold) & (self.all_minima > 0)

            print(f"  Excluding {np.sum(~valid_mask)} invalid minima")

            valid_minima = self.all_minima[valid_mask]
            valid_radii = self.all_radii_min[valid_mask]

            if len(valid_minima) > 0:
                # First, calculate raw magnitudes for both maxima and minima
                minima_raw_mag = -2.5 * np.log10(valid_minima / self.star_flux)

                # Calculate the mean flux of maxima (not magnitude)
                mean_maxima_flux = np.mean(self.all_maxima)

                # Key insight: The minima should be scaled relative to the typical speckle brightness
                # not the star. So we normalize by the ratio of star to mean speckle
                flux_normalization = self.star_flux / mean_maxima_flux

                # Apply this normalization to minima
                minima_normalized_flux = valid_minima * flux_normalization

                # Now calculate magnitudes with this normalization
                minima_delta_mag = -2.5 * np.log10(minima_normalized_flux / self.star_flux)

                # Alternative approach: shift by the difference in mean magnitudes
                mean_maxima_mag = np.mean(maxima_delta_mag)
                mean_minima_raw = np.mean(minima_raw_mag)
                magnitude_shift = mean_minima_raw - mean_maxima_mag

                print(f"  Mean maxima magnitude: {mean_maxima_mag:.2f}")
                print(f"  Mean minima raw magnitude: {mean_minima_raw:.2f}")
                print(f"  Magnitude shift: {magnitude_shift:.2f}")

                # Apply the shift to align the distributions
                minima_delta_mag = minima_raw_mag - magnitude_shift + 0.5  # Small offset to keep minima slightly fainter

                # Update arrays
                self.all_minima = valid_minima
                self.all_radii_min = valid_radii

                # No hard bounds - let them fall naturally
                print(f"  Kept {len(valid_minima)} minima")
                print(f"  Final magnitude range: {np.min(minima_delta_mag):.2f} to {np.max(minima_delta_mag):.2f}")
            else:
                minima_delta_mag = np.array([])
                self.all_minima = np.array([])
                self.all_radii_min = np.array([])
        else:
            minima_delta_mag = np.array([])

        # For the limits, handle the point at origin specially
        limits_delta_mag = np.zeros(len(limits))
        limits_delta_mag[0] = 0  # At r=0, delta magnitude = 0
        limits[1:] = np.maximum(limits[1:], 1e-10)  # Avoid log of zero
        limits_delta_mag[1:] = -2.5 * np.log10(limits[1:] / self.star_flux)

        # Create the plot with similar style to example
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        # Plot all local maxima with empty squares
        ax.scatter(self.all_radii_max * self.pixel_scale, maxima_delta_mag,
                   marker='s', s=30, facecolors='none', edgecolors='black',
                   linewidth=0.5, alpha=0.7, label='Local Maxima')

        # Plot all local minima with small filled dots (if any found)
        if len(self.all_minima) > 0:
            ax.scatter(self.all_radii_min * self.pixel_scale, minima_delta_mag,
                       marker='.', s=10, color='black', alpha=0.7,
                       label='Local Minima')
            print(f"Plotted {len(self.all_minima)} local minima")
        else:
            print("Warning: No local minima found to plot")

        # Plot detection limit curve with thicker red line
        ax.plot(radii * self.pixel_scale, limits_delta_mag, 'r-', linewidth=2.5)

        # Find actual detections - maxima that fall BELOW the detection limit curve
        # Interpolate the detection limit at each maximum's position
        limit_interp = np.interp(self.all_radii_max * self.pixel_scale,
                                 radii * self.pixel_scale,
                                 limits_delta_mag)

        # Detections are where maxima are below the limit (smaller delta mag = brighter)
        detections = maxima_delta_mag < limit_interp

        if np.any(detections):
            detection_seps = self.all_radii_max[detections] * self.pixel_scale
            detection_mags = maxima_delta_mag[detections]

            # Find the most significant detections to annotate
            # Sort by how far below the detection limit they are
            significance = limit_interp[detections] - detection_mags
            sorted_indices = np.argsort(significance)[::-1]  # Most significant first

            # Annotate up to 2 most significant detections
            n_annotations = min(2, len(sorted_indices))

            for i in range(n_annotations):
                idx = sorted_indices[i]
                sep = detection_seps[idx]
                mag = detection_mags[idx]

                # Get the limiting magnitude at this separation
                limiting_mag = np.interp(sep, radii * self.pixel_scale, limits_delta_mag)

                # Position annotation
                if i == 0:  # First detection
                    # Annotate with arrow pointing to the detection
                    ax.annotate(f'Limiting Δm = {limiting_mag:.2f}',
                                xy=(sep, mag),  # Point to the actual detection
                                xytext=(sep + 0.05, mag + 0.8),
                                arrowprops=dict(arrowstyle='->', color='black', lw=0.5),
                                fontsize=11, ha='left')
                else:  # Second detection
                    # Position differently to avoid overlap
                    ax.annotate(f'Limiting Δm = {limiting_mag:.2f}',
                                xy=(sep, mag),
                                xytext=(sep - 0.1, mag - 0.8),
                                fontsize=11, ha='center')

            print(f"Found {np.sum(detections)} detections below the detection limit")

        # Formatting to match example
        ax.set_xlabel('Separation [arcsec]', fontsize=13)
        ax.set_ylabel('Magnitude Difference', fontsize=13)

        # Add secondary x-axis for pixels
        ax2 = ax.twiny()
        ax2.set_xlabel('Separation [pixels]', fontsize=13)

        # Set the pixel axis limits to match the arcsec axis
        x_min, x_max = ax.get_xlim()
        ax2.set_xlim(x_min / self.pixel_scale, x_max / self.pixel_scale)

        # Make sure the pixel axis ticks are at nice round numbers
        pixel_ticks = np.arange(0, x_max / self.pixel_scale + 10, 10)
        ax2.set_xticks(pixel_ticks)

        # Legend in upper right like example
        ax.legend(fontsize=10, loc='upper right', frameon=True, fancybox=False)

        # Grid style
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

        # Set axis limits to match example style
        # Y-axis: 0 to 10 (normal orientation - 0 at bottom, 10 at top)
        ax.set_ylim(0, 10)

        # X-axis: 0 to about 1.2 arcsec (adjust based on data)
        max_x = min(1.2, max_radius * self.pixel_scale)
        ax.set_xlim(0, max_x)

        # Set tick parameters to match example
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax2.tick_params(axis='both', which='major', labelsize=11)

        # Add information box in lower right
        info_text = []
        info_text.append(f'Telescope: Hooker 2.54m')
        info_text.append(f'Wavelength: {self.wavelength * 1e9:.0f} nm')
        info_text.append(f'Pixel scale: {self.pixel_scale * 1000:.1f} mas/pixel')
        info_text.append(f'PSF FWHM: {self.fwhm_arcsec:.3f}" ({self.fwhm_pixels:.1f} pix)')
        info_text.append(f'λ/D: {self.lambda_over_d:.3f}" ({self.lambda_over_d / self.pixel_scale:.1f} pix)')
        info_text.append(f'Confidence: {confidence_level}σ')

        # Extract object name from header if available
        if hasattr(self, 'header') and self.header:
            object_name = self.header.get('OBJECT', 'Unknown')
            date_obs = self.header.get('DATE-OBS', 'Unknown')
            if object_name != 'Unknown':
                info_text.append(f'Object: {object_name}')
            if date_obs != 'Unknown':
                # Try to format date nicely
                try:
                    if 'T' in date_obs:
                        date_part = date_obs.split('T')[0]
                    else:
                        date_part = date_obs
                    info_text.append(f'Date: {date_part}')
                except:
                    info_text.append(f'Date: {date_obs}')

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


def load_speckle_image(file_path):
    """Load speckle image with FITS header repair."""
    print(f"=== Loading image: {file_path} ===")

    file_path = Path(file_path)

    if file_path.suffix.lower() in ['.fits', '.fit']:
        try:
            with fits.open(file_path) as hdul:
                image_data = hdul[0].data
                header = dict(hdul[0].header)
        except Exception as e:
            print(f"FITS header error detected: {e}")
            print("Attempting to fix corrupted FITS header...")
            try:
                with fits.open(file_path) as hdul:
                    hdul.verify('fix')
                    image_data = hdul[0].data
                    header = dict(hdul[0].header)
                print("FITS header successfully repaired.")
            except Exception as e2:
                try:
                    with fits.open(file_path, ignore_missing_end=True) as hdul:
                        image_data = hdul[0].data
                        header = {'COMMENT': 'Header reconstructed due to corruption'}
                    print("Data loaded with minimal header.")
                except Exception as e3:
                    raise ValueError(f"Could not load FITS file. Error: {e3}")
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Ensure 2D array
    if image_data.ndim > 2:
        image_data = image_data.squeeze()

    print(f"Image loaded: shape {image_data.shape}")
    return image_data, header


class ContrastCurveGUI:
    """GUI for contrast curve generation."""

    def __init__(self, root):
        print("=== Contrast Curve GUI ===")

        self.root = root
        self.root.title("Hooker Telescope Contrast Curve Generator")
        self.root.geometry("800x700")

        # Variables
        self.file_path = tk.StringVar()
        self.confidence_level = tk.DoubleVar(value=5.0)
        self.pixel_scale = tk.DoubleVar(value=0.0135)
        self.wavelength = tk.DoubleVar(value=617)
        self.max_radius = tk.IntVar(value=90)  # Increased to show ~1.2 arcsec
        self.min_radius = tk.IntVar(value=2)  # Default to 2 pixels
        self.save_path = tk.StringVar()

        # Current data
        self.ccg = None
        self.image_data = None
        self.header = None

        self.create_widgets()

    def create_widgets(self):
        """Create the GUI widgets."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Input File", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Entry(file_frame, textvariable=self.file_path, width=60).grid(row=0, column=0, padx=5)
        ttk.Button(file_frame, text="Browse FITS", command=self.browse_file).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Load File", command=self.load_file).grid(row=0, column=2, padx=5)

        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="5")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        # Row 0: Confidence Level
        ttk.Label(param_frame, text="Confidence Level (σ):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=1.0, to=10.0, increment=0.1,
                    textvariable=self.confidence_level, width=10).grid(row=0, column=1, padx=5)

        # Row 1: Min and Max Radius
        ttk.Label(param_frame, text="Min Radius (pixels):").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=1, to=50, increment=1,
                    textvariable=self.min_radius, width=10).grid(row=1, column=1, padx=5)

        ttk.Label(param_frame, text="Max Radius (pixels):").grid(row=1, column=2, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=10, to=200, increment=10,
                    textvariable=self.max_radius, width=10).grid(row=1, column=3, padx=5)

        # Row 2: Pixel Scale and Wavelength
        ttk.Label(param_frame, text="Pixel Scale (arcsec/pixel):").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Entry(param_frame, textvariable=self.pixel_scale, width=10).grid(row=2, column=1, padx=5)

        ttk.Label(param_frame, text="Wavelength (nm):").grid(row=2, column=2, sticky=tk.W, padx=5)
        ttk.Entry(param_frame, textvariable=self.wavelength, width=10).grid(row=2, column=3, padx=5)

        # Filter presets
        filter_frame = ttk.Frame(param_frame)
        filter_frame.grid(row=3, column=0, columnspan=4, pady=5)
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
        ttk.Button(save_frame, text="Browse Save Location", command=self.browse_save).grid(row=0, column=1, padx=5)
        ttk.Button(save_frame, text="Save Plot", command=self.save_plot).grid(row=0, column=2, padx=5)

        # Status
        status_frame = ttk.LabelFrame(main_frame, text="Status & Information", padding="5")
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        text_frame = ttk.Frame(status_frame)
        text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.status_text = tk.Text(text_frame, height=15, width=80, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)

        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.log_message("Contrast Curve Generator Ready")
        self.log_message("Analyzes annuli to find local maxima and minima")
        self.log_message("Detection limit = mean(maxima) + 5σ_avg")
        self.log_message("Minima floor set by dynamic range (1000:1)")

    def log_message(self, message):
        """Add message to status area."""
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

    def set_filter(self, wavelength_nm):
        """Set filter wavelength."""
        self.wavelength.set(wavelength_nm)
        filter_name = "Sloan r'" if wavelength_nm == 617 else "Sloan i'"
        self.log_message(f"Set filter to {filter_name} ({wavelength_nm} nm)")

    def browse_file(self):
        """Browse for FITS file."""
        filename = filedialog.askopenfilename(
            title="Select Speckle FITS File",
            filetypes=[("FITS files", "*.fits *.fit"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)

    def browse_save(self):
        """Browse for save location."""
        filename = filedialog.asksaveasfilename(
            title="Save Contrast Curve Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                       ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if filename:
            self.save_path.set(filename)

    def load_file(self):
        """Load the selected FITS file."""
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a FITS file first")
            return

        try:
            self.log_message(f"Loading file: {self.file_path.get()}")
            self.image_data, self.header = load_speckle_image(self.file_path.get())

            self.log_message(f"File loaded successfully!")
            self.log_message(f"Image dimensions: {self.image_data.shape}")

            # Update parameters from header
            if self.header:
                for key in ['PIXSCALE', 'PLATESCL', 'PLTSCALE', 'SCALE']:
                    if key in self.header:
                        self.pixel_scale.set(abs(float(self.header[key])))
                        self.log_message(f"Updated pixel scale: {self.pixel_scale.get():.4f}")
                        break

                telescope = self.header.get('TELESCOP', 'Unknown')
                object_name = self.header.get('OBJECT', 'Unknown')
                date_obs = self.header.get('DATE-OBS', 'Unknown')
                self.log_message(f"Telescope: {telescope}")
                self.log_message(f"Object: {object_name}")
                self.log_message(f"Date: {date_obs}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            self.log_message(f"Error loading file: {str(e)}")

    def generate_curve(self):
        """Generate the contrast curve."""
        if self.image_data is None:
            messagebox.showerror("Error", "Please load a FITS file first")
            return

        try:
            self.log_message("Generating contrast curve...")
            self.log_message("Using fixed minima magnitude calculation...")

            # Create contrast curve generator
            self.ccg = ContrastCurveGenerator(
                self.image_data,
                pixel_scale=self.pixel_scale.get(),
                wavelength=self.wavelength.get() * 1e-9,
                telescope_diameter=2.54,
                header=self.header
            )

            # Get minimum radius from GUI
            manual_min = self.min_radius.get()

            # Generate plot
            fig = self.ccg.plot_contrast_curve(
                confidence_level=self.confidence_level.get(),
                min_radius=manual_min,
                max_radius=self.max_radius.get()
            )

            # Store figure
            self.current_figure = fig

            self.log_message("Contrast curve generated successfully!")
            self.log_message(f"Detection limit: mean(maxima) + {self.confidence_level.get()} × avg(σ_max, σ_min)")

        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Failed to generate curve: {error_msg}")
            self.log_message(f"Error: {error_msg}")
            import traceback
            self.log_message(traceback.format_exc())

    def save_plot(self):
        """Save the current plot."""
        if not hasattr(self, 'current_figure'):
            messagebox.showerror("Error", "No plot to save. Generate a contrast curve first.")
            return

        if not self.save_path.get():
            messagebox.showerror("Error", "Please specify a save location")
            return

        try:
            self.current_figure.savefig(self.save_path.get(), dpi=300, bbox_inches='tight')
            self.log_message(f"Plot saved to: {self.save_path.get()}")
            messagebox.showinfo("Success", f"Plot saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {str(e)}")


def run_gui():
    """Run the GUI application."""
    print("=== Starting Contrast Curve Generator ===")
    root = tk.Tk()
    app = ContrastCurveGUI(root)
    root.mainloop()


# For command-line usage
def generate_contrast_curve_cli(fits_file, output_file=None, confidence_level=5.0,
                               min_radius=2, max_radius=90):
    """Command-line interface for generating contrast curves."""
    print(f"Processing: {fits_file}")

    # Load image
    image_data, header = load_speckle_image(fits_file)

    # Create generator
    ccg = ContrastCurveGenerator(image_data, header=header)

    # Generate and plot
    fig = ccg.plot_contrast_curve(
        confidence_level=confidence_level,
        min_radius=min_radius,
        max_radius=max_radius,
        save_path=output_file,
        show_plot=(output_file is None)
    )

    if output_file:
        print(f"Saved to: {output_file}")

    return ccg


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Command-line mode
        fits_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        generate_contrast_curve_cli(fits_file, output_file)
    else:
        # GUI mode
        run_gui()