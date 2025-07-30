"""
Hooker Telescope Contrast Curve Generator with Aperture Photometry
================================================================

A GUI application for generating contrast curves using small aperture photometry
to find local maxima at each radius from the image center.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import astropy.units as u
from astropy.io import fits
from pathlib import Path
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.figure import Figure
import threading


class ApertureContrastCurveGenerator:
    """Generate contrast curves using aperture photometry."""

    def __init__(self, image_data, pixel_scale=None, wavelength=None,
                 telescope_diameter=None, star_position=None, header=None):
        """Initialize the contrast curve generator."""
        print("=== Aperture-Based Contrast Curve Generator ===")

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

        print(f"Hooker Telescope (2.54m) Aperture Analysis")
        print(f"Pixel scale: {self.pixel_scale * 1000:.1f} mas/pixel")
        print(f"Wavelength: {self.wavelength * 1e9:.0f} nm")
        print(f"Image size: {self.image.shape}")
        print(f"PSF FWHM: {self.fwhm_pixels:.2f} pixels ({self.fwhm_arcsec:.3f} arcsec)")

    def measure_psf_fwhm(self):
        """Measure the FWHM of the central PSF with 1.25x scaling."""
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

            # Calculate FWHM in both directions and multiply by 1.25
            fwhm_x = find_fwhm(horizontal_profile)
            fwhm_y = find_fwhm(vertical_profile)

            raw_fwhm_pixels = (fwhm_x + fwhm_y) / 2.0
            fwhm_pixels = raw_fwhm_pixels * 1.25
            fwhm_arcsec = fwhm_pixels * self.pixel_scale

            print(f"Raw PSF FWHM: {raw_fwhm_pixels:.2f} pixels")
            print(f"Scaled PSF FWHM (1.25x): {fwhm_pixels:.2f} pixels ({fwhm_arcsec:.3f} arcsec)")

            return fwhm_pixels, fwhm_arcsec

        except Exception as e:
            print(f"PSF FWHM measurement failed: {e}")
            default_fwhm = 4.0 * 1.25  # Apply 1.25x scaling to default too
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

    def aperture_photometry_at_radius(self, radius_pixels, aperture_size=3, max_radius=None):
        """Perform aperture photometry at a given radius using overlapping apertures (1-pixel steps)."""
        center_x, center_y = self.star_pos

        if max_radius is None:
            max_radius = min(self.image.shape) // 2 - aperture_size

        if radius_pixels > max_radius:
            return [], []

        aperture_values = []
        aperture_distances = []
        half_aperture = aperture_size // 2

        # Create a grid around the image center, stepping by 1 pixel
        # This allows overlapping apertures for finer sampling
        y_positions = range(half_aperture, self.image.shape[0] - half_aperture, 1)
        x_positions = range(half_aperture, self.image.shape[1] - half_aperture, 1)

        for iy in y_positions:
            for ix in x_positions:
                # Calculate distance from aperture center to image center
                aperture_center_dist = np.sqrt((ix - center_x) ** 2 + (iy - center_y) ** 2)

                # Check if this aperture is at approximately the desired radius
                # Allow some tolerance (±0.5 pixels) to capture apertures near this radius
                tolerance = 0.5
                if abs(aperture_center_dist - radius_pixels) <= tolerance:
                    # Extract aperture region
                    aperture_region = self.image[iy - half_aperture:iy + half_aperture + 1,
                                      ix - half_aperture:ix + half_aperture + 1]

                    # Find maximum in this aperture and its position
                    max_value = np.max(aperture_region)
                    max_pos = np.unravel_index(np.argmax(aperture_region), aperture_region.shape)

                    # Convert local maximum position to global image coordinates
                    global_max_y = iy - half_aperture + max_pos[0]
                    global_max_x = ix - half_aperture + max_pos[1]

                    # Calculate precise distance from image center to the actual maximum pixel
                    precise_distance = np.sqrt((global_max_x - center_x) ** 2 + (global_max_y - center_y) ** 2)

                    aperture_values.append(max_value)
                    aperture_distances.append(precise_distance)

        return aperture_values, aperture_distances

    def generate_radial_profile(self, min_radius=None, max_radius=None, radius_step=1, aperture_size=3,
                                use_fwhm_limit=True, manual_min_radius=None):
        """Generate radial profile of local maxima."""
        print("=== Generating radial profile with aperture photometry ===")

        if min_radius is None:
            if manual_min_radius is not None and manual_min_radius > 0:
                min_radius = manual_min_radius
                print(f"Using manual minimum radius: {min_radius} pixels")
            elif use_fwhm_limit:
                min_radius = max(2, int(self.fwhm_pixels))  # Start at FWHM
                print(f"Using FWHM-based minimum radius: {min_radius} pixels")
            else:
                min_radius = 2  # Start at 2 pixels if FWHM limit disabled
                print("Using default minimum radius: 2 pixels")

        if max_radius is None:
            # Default to 40 pixels from center if not specified
            max_radius = min(40, min(self.image.shape) // 2 - aperture_size)

        print(f"Final radius range: {min_radius} to {max_radius} pixels")
        print(f"Aperture size: {aperture_size}x{aperture_size} pixels")

        radii = []
        all_maxima = []

        for radius in range(min_radius, max_radius + 1, radius_step):
            aperture_values, aperture_distances = self.aperture_photometry_at_radius(radius, aperture_size, max_radius)

            if aperture_values:
                # Store all values and their precise distances
                for value, distance in zip(aperture_values, aperture_distances):
                    radii.append(distance)  # Use precise floating-point distance
                    all_maxima.append(value)

        print(f"Found {len(all_maxima)} aperture measurements across {len(set(radii))} radii")

        self.radii = np.array(radii)
        self.maxima_values = np.array(all_maxima)
        self.star_flux = float(np.max(self.image))

        return self.radii, self.maxima_values

    def compute_sigma_curve(self, confidence_level=5.0, bin_size=5):
        """Compute sigma detection curve by binning the radial data."""
        if not hasattr(self, 'radii') or len(self.radii) == 0:
            return np.array([]), np.array([])

        # Create radial bins
        min_radius = np.min(self.radii)
        max_radius = np.max(self.radii)

        # Create bins every bin_size pixels
        bin_edges = np.arange(min_radius, max_radius + bin_size, bin_size)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        from scipy.stats import binned_statistic

        # Calculate statistics in each bin
        means, _, _ = binned_statistic(self.radii, self.maxima_values, statistic='mean', bins=bin_edges)
        stds, _, _ = binned_statistic(self.radii, self.maxima_values, statistic='std', bins=bin_edges)
        counts, _, _ = binned_statistic(self.radii, self.maxima_values, statistic='count', bins=bin_edges)

        # Only keep bins with sufficient data
        valid_bins = counts >= 3

        if np.sum(valid_bins) < 2:
            print("Warning: Insufficient data for sigma curve")
            return bin_centers, means + confidence_level * stds

        # Filter to valid bins
        valid_centers = bin_centers[valid_bins]
        valid_means = means[valid_bins]
        valid_stds = stds[valid_bins]

        # Calculate threshold
        sigma_threshold = valid_means + confidence_level * valid_stds

        # Interpolate back to all bin centers
        if len(valid_centers) >= 2:
            interp_func = interp1d(valid_centers, sigma_threshold,
                                   kind='linear', bounds_error=False,
                                   fill_value=(sigma_threshold[0], sigma_threshold[-1]))
            threshold = interp_func(bin_centers)
        else:
            threshold = means + confidence_level * stds

        print(f"Computed {confidence_level}σ curve with {len(bin_centers)} radial bins")
        return bin_centers, threshold

    def plot_contrast_curve(self, confidence_level=5.0, aperture_size=3, max_radius=40, use_fwhm_limit=True,
                            manual_min_radius=None, save_path=None, show_plot=True):
        """Plot the aperture-based contrast curve."""
        print("=== Generating aperture-based contrast curve ===")

        # Generate radial profile
        radii, maxima = self.generate_radial_profile(aperture_size=aperture_size, max_radius=max_radius,
                                                     use_fwhm_limit=use_fwhm_limit, manual_min_radius=manual_min_radius)

        if len(radii) == 0:
            print("No data found!")
            return None

        # Compute sigma curve
        bin_centers, sigma_threshold = self.compute_sigma_curve(confidence_level)

        # Convert to delta magnitudes
        maxima_delta_mag = -2.5 * np.log10(maxima / self.star_flux)
        sigma_delta_mag = -2.5 * np.log10(sigma_threshold / self.star_flux)

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot all individual measurements
        ax.scatter(radii, maxima_delta_mag, color='black', s=2, alpha=0.5,
                   label=f'Aperture Maxima ({len(radii)} measurements)')

        # Plot sigma curve
        ax.plot(bin_centers, sigma_delta_mag, 'r-', linewidth=3,
                label=f'{confidence_level}σ Detection Limit')

        # Find candidates above threshold
        sigma_interp = np.interp(radii, bin_centers, sigma_delta_mag)
        candidates = maxima_delta_mag < sigma_interp  # Lower delta mag = brighter

        if np.any(candidates):
            ax.scatter(radii[candidates], maxima_delta_mag[candidates],
                       edgecolor='red', facecolor='none', s=50, linewidth=2,
                       label=f'Candidate Detections ({np.sum(candidates)})')

        # Formatting
        ax.set_xlabel('Radius (pixels)', fontsize=12)
        ax.set_ylabel('Contrast (Δ magnitude)', fontsize=12)
        ax.set_title(f'Aperture-Based Contrast Curve - {confidence_level}σ Detection Limit', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Set magnitude limits (7 mag range max)
        min_mag = max(0, np.min(sigma_delta_mag) - 0.5)
        max_mag = min(min_mag + 7, np.percentile(maxima_delta_mag, 95) + 1)
        ax.set_ylim(max_mag, min_mag)  # Inverted for magnitude scale

        # Add secondary x-axis with arcsec
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xlabel('Radius (arcsec)', fontsize=12)
        pixel_ticks = ax.get_xticks()
        arcsec_ticks = pixel_ticks * self.pixel_scale
        ax_top.set_xticks(pixel_ticks)
        ax_top.set_xticklabels([f'{x:.3f}' for x in arcsec_ticks])

        # Add info box (positioned on right middle)
        info_text = f'Wavelength: {self.wavelength * 1e9:.0f} nm\n'
        info_text += f'Pixel scale: {self.pixel_scale * 1000:.1f} mas/pixel\n'
        info_text += f'PSF FWHM: {self.fwhm_pixels:.2f} pix ({self.fwhm_arcsec:.3f}")\n'
        info_text += f'Aperture: {aperture_size}×{aperture_size} pixels\n'
        info_text += f'Measurements: {len(radii)}\n'
        if np.any(candidates):
            info_text += f'Candidates: {np.sum(candidates)}'

        ax.text(0.98, 0.5, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
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


class ApertureContrastGUI:
    """GUI for aperture-based contrast curves."""

    def __init__(self, root):
        print("=== Aperture Contrast Curve GUI ===")

        self.root = root
        self.root.title("Hooker Telescope Aperture-Based Contrast Curve Generator")
        self.root.geometry("800x700")

        # Variables
        self.file_path = tk.StringVar()
        self.confidence_level = tk.DoubleVar(value=5.0)
        self.pixel_scale = tk.DoubleVar(value=0.0135)
        self.wavelength = tk.DoubleVar(value=617)
        self.aperture_size = tk.IntVar(value=3)
        self.max_radius = tk.IntVar(value=40)
        self.min_radius = tk.IntVar(value=0)  # 0 means use automatic
        self.use_fwhm_limit = tk.BooleanVar(value=True)
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

        # Row 0: Confidence Level and Aperture Size
        ttk.Label(param_frame, text="Confidence Level (σ):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=1.0, to=10.0, increment=0.1,
                    textvariable=self.confidence_level, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(param_frame, text="Aperture Size (pixels):").grid(row=0, column=2, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=1, to=7, increment=2,
                    textvariable=self.aperture_size, width=10).grid(row=0, column=3, padx=5)

        # Row 1: Min Radius and Max Radius
        ttk.Label(param_frame, text="Min Radius (pixels, 0=auto):").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=0, to=50, increment=1,
                    textvariable=self.min_radius, width=10).grid(row=1, column=1, padx=5)

        ttk.Label(param_frame, text="Max Radius (pixels):").grid(row=1, column=2, sticky=tk.W, padx=5)
        ttk.Spinbox(param_frame, from_=10, to=200, increment=10,
                    textvariable=self.max_radius, width=10).grid(row=1, column=3, padx=5)

        # Row 2: Pixel Scale and Wavelength
        ttk.Label(param_frame, text="Pixel Scale (arcsec/pixel):").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Entry(param_frame, textvariable=self.pixel_scale, width=10).grid(row=2, column=1, padx=5)

        ttk.Label(param_frame, text="Wavelength (nm):").grid(row=2, column=2, sticky=tk.W, padx=5)
        ttk.Entry(param_frame, textvariable=self.wavelength, width=10).grid(row=2, column=3, padx=5)

        # Row 3: FWHM limit checkbox
        ttk.Checkbutton(param_frame, text="Use FWHM minimum radius limit",
                        variable=self.use_fwhm_limit).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5)

        # Filter presets
        filter_frame = ttk.Frame(param_frame)
        filter_frame.grid(row=4, column=0, columnspan=4, pady=5)
        ttk.Label(filter_frame, text="Filter Presets:").pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="Sloan r'", command=lambda: self.set_filter(617)).pack(side=tk.LEFT, padx=2)
        ttk.Button(filter_frame, text="Sloan i'", command=lambda: self.set_filter(748)).pack(side=tk.LEFT, padx=2)

        # Processing
        process_frame = ttk.LabelFrame(main_frame, text="Processing", padding="5")
        process_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(process_frame, text="Generate Aperture-Based Contrast Curve",
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

        self.log_message("Aperture-Based Contrast Curve Generator Ready")
        self.log_message("Uses small aperture photometry to find local maxima at each radius")

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
        """Generate the aperture-based contrast curve."""
        if self.image_data is None:
            messagebox.showerror("Error", "Please load a FITS file first")
            return

        try:
            self.log_message("Generating aperture-based contrast curve...")
            self.log_message("This may take a moment...")

            # Create contrast curve generator
            self.ccg = ApertureContrastCurveGenerator(
                self.image_data,
                pixel_scale=self.pixel_scale.get(),
                wavelength=self.wavelength.get() * 1e-9,
                telescope_diameter=2.54,
                header=self.header
            )

            # Determine manual minimum radius (0 means automatic)
            manual_min = self.min_radius.get() if self.min_radius.get() > 0 else None

            # Generate plot
            fig = self.ccg.plot_contrast_curve(
                confidence_level=self.confidence_level.get(),
                aperture_size=self.aperture_size.get(),
                max_radius=self.max_radius.get(),
                use_fwhm_limit=self.use_fwhm_limit.get(),
                manual_min_radius=manual_min
            )

            # Store figure
            self.current_figure = fig

            self.log_message("Aperture-based contrast curve generated successfully!")
            self.log_message(f"Using {self.aperture_size.get()}×{self.aperture_size.get()} pixel apertures")
            self.log_message(f"Maximum radius: {self.max_radius.get()} pixels")
            if self.min_radius.get() > 0:
                self.log_message(f"Manual minimum radius: {self.min_radius.get()} pixels")
            elif self.use_fwhm_limit.get():
                self.log_message("FWHM minimum radius limit: ENABLED")
            else:
                self.log_message("FWHM minimum radius limit: DISABLED")

        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Error", f"Failed to generate curve: {error_msg}")
            self.log_message(f"Error: {error_msg}")

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
    print("=== Starting Aperture-Based Contrast Curve Generator ===")
    root = tk.Tk()
    app = ApertureContrastGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()