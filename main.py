"""
Contrast Curve Generator - Simplified Robust Version with Smart Annotation Placement
===================================================================================

Back to basics with minimal filtering, robust statistics, and intelligent annotation positioning
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
                                          max_radius=50, annulus_width=3,
                                          telescope_diameter=2.54, wavelength=617e-9,
                                          target_name="Unknown Target", telescope_name="Hooker",
                                          smoothing_sigma=3.0):
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
            # Use robust statistics throughout
            # Step 1: Median value of the maxima (robust against outliers)
            mean_maxima = np.median(max_vals)

            # Step 2: MAD-based standard deviation of maxima
            if len(max_vals) > 1:
                mad_maxima = np.median(np.abs(max_vals - mean_maxima))
                sigma_maxima = mad_maxima * 1.4826  # Convert MAD to std equivalent
            else:
                sigma_maxima = 0.0

            # Step 3: MAD-based standard deviation of minima
            if len(min_vals) > 1:
                median_minima = np.median(min_vals)
                mad_minima = np.median(np.abs(min_vals - median_minima))
                sigma_minima = mad_minima * 1.4826
            elif len(min_vals) == 1:
                sigma_minima = 0.0
            else:
                # No minima - use sigma of maxima
                sigma_minima = sigma_maxima

            # Step 4: Average sigma of maxima and minima
            avg_sigma = (sigma_maxima + sigma_minima) / 2.0

            # Step 5: Detection limit = median(maxima) + 5 * avg_sigma
            detection_limit = mean_maxima + 5 * avg_sigma

            # Sanity check - if detection limit is negative or huge, cap it
            if detection_limit < 0:
                detection_limit = mean_maxima + 5 * noise_std
            elif detection_limit > star_flux:
                detection_limit = star_flux

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
        # Apply smoothing with adjustable sigma
        smoothed = gaussian_filter1d(detection_limits[1:], sigma=smoothing_sigma)
        detection_limits[1:] = smoothed

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
    else:
        min_mags = np.array([])

    # Detection limit magnitudes
    detection_limits = np.maximum(detection_limits, noise_floor)
    limit_mags = -2.5 * np.log10(detection_limits / star_flux)

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


def analyze_data_distribution(x_data_sets, y_data_sets, x_lims, y_lims):
    """
    Analyze the distribution of data points to understand where they cluster.

    Returns:
    - density_map: 2D array showing data density in different regions
    - clear_regions: list of (region_name, score) sorted by clearness
    """

    # Combine all data points
    all_x = np.array([])
    all_y = np.array([])

    for x_data, y_data in zip(x_data_sets, y_data_sets):
        if len(x_data) > 0 and len(y_data) > 0:
            all_x = np.concatenate([all_x, x_data])
            all_y = np.concatenate([all_y, y_data])

    if len(all_x) == 0:
        # No data - return default
        return np.zeros((4, 4)), [('top-left', 1.0)]

    # Normalize data to 0-1 range
    x_min, x_max = x_lims
    y_min, y_max = y_lims
    norm_x = (all_x - x_min) / (x_max - x_min)
    norm_y = (all_y - y_min) / (y_max - y_min)

    # Create density map (4x4 grid)
    density_map = np.zeros((4, 4))

    for x, y in zip(norm_x, norm_y):
        if 0 <= x <= 1 and 0 <= y <= 1:
            grid_x = min(3, int(x * 4))
            grid_y = min(3, int((1-y) * 4))  # Flip y for top-bottom orientation
            density_map[grid_y, grid_x] += 1

    # Define regions with their grid positions
    regions = {
        'top-left': (0, 0),
        'top-center': (0, 1),
        'top-right': (0, 3),
        'mid-left': (1, 0),
        'mid-center': (1, 1),
        'mid-right': (1, 3),
        'bottom-left': (3, 0),
        'bottom-center': (3, 1),
        'bottom-right': (3, 3),
        'upper-left': (0, 0),
        'upper-right': (0, 2),
        'lower-left': (2, 0),
        'lower-right': (2, 2),
    }

    # Calculate clearness score for each region (lower density = higher score)
    max_density = np.max(density_map) if np.max(density_map) > 0 else 1
    clear_regions = []

    for region_name, (gy, gx) in regions.items():
        # Score is inverse of density, plus bonus for corner regions
        density = density_map[gy, gx]
        base_score = 1.0 - (density / max_density)

        # Add bonus for corner positions (usually clearer)
        if 'corner' in region_name or region_name in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
            base_score += 0.2

        clear_regions.append((region_name, base_score))

    # Sort by score (highest first)
    clear_regions.sort(key=lambda x: x[1], reverse=True)

    return density_map, clear_regions


def find_clear_annotation_area(ax, x_data_sets, y_data_sets, avoid_boxes=None, annotation_type="detection"):
    """
    Find a clear area for annotation placement using advanced spatial analysis.

    Parameters:
    - ax: matplotlib axis object
    - x_data_sets, y_data_sets: lists of arrays of data points to avoid
    - avoid_boxes: list of (x, y, width, height) regions to avoid in data coordinates
    - annotation_type: "detection" or "info" for different sizing considerations

    Returns:
    - (x, y): coordinates for annotation placement in data coordinates
    """

    # Get axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Analyze data distribution
    density_map, clear_regions = analyze_data_distribution(x_data_sets, y_data_sets, (x_min, x_max), (y_min, y_max))

    # Define annotation box size based on type
    if annotation_type == "detection":
        box_width_rel = 0.2   # Smaller box for detection info
        box_height_rel = 0.15
    else:
        box_width_rel = 0.25  # Larger box for general info
        box_height_rel = 0.25

    # Extended candidate positions with more granular options
    candidate_positions = [
        # Corners
        (0.15, 0.85, 'upper-left'),
        (0.85, 0.85, 'upper-right'),
        (0.15, 0.15, 'lower-left'),
        (0.85, 0.15, 'lower-right'),

        # Edges (more positions)
        (0.5, 0.9, 'top-center'),
        (0.3, 0.9, 'top-left-center'),
        (0.7, 0.9, 'top-right-center'),
        (0.5, 0.1, 'bottom-center'),
        (0.3, 0.1, 'bottom-left-center'),
        (0.7, 0.1, 'bottom-right-center'),

        # Side positions
        (0.05, 0.7, 'left-upper'),
        (0.05, 0.5, 'left-center'),
        (0.05, 0.3, 'left-lower'),
        (0.95, 0.7, 'right-upper'),
        (0.95, 0.5, 'right-center'),
        (0.95, 0.3, 'right-lower'),

        # Interior positions (last resort)
        (0.25, 0.75, 'inner-upper-left'),
        (0.75, 0.75, 'inner-upper-right'),
        (0.25, 0.25, 'inner-lower-left'),
        (0.75, 0.25, 'inner-lower-right'),
    ]

    best_position = None
    best_score = -1

    for rel_x, rel_y, pos_name in candidate_positions:
        # Convert to data coordinates
        test_x = x_min + rel_x * (x_max - x_min)
        test_y = y_min + rel_y * (y_max - y_min)

        # Calculate annotation box bounds in data coordinates
        box_width = box_width_rel * (x_max - x_min)
        box_height = box_height_rel * (y_max - y_min)

        # Adjust box position based on alignment
        if rel_x < 0.5:  # Left side - box extends right
            box_x_min = test_x
            box_x_max = test_x + box_width
        else:  # Right side - box extends left
            box_x_min = test_x - box_width
            box_x_max = test_x

        if rel_y > 0.5:  # Upper side - box extends down
            box_y_min = test_y - box_height
            box_y_max = test_y
        else:  # Lower side - box extends up
            box_y_min = test_y
            box_y_max = test_y + box_height

        # Score this position
        score = 0

        # 1. Distance to data points (most important factor)
        min_data_distance = 1.0
        for x_data, y_data in zip(x_data_sets, y_data_sets):
            if len(x_data) > 0 and len(y_data) > 0:
                # Check if any data points fall within the annotation box
                inside_box = ((x_data >= box_x_min) & (x_data <= box_x_max) &
                             (y_data >= box_y_min) & (y_data <= box_y_max))

                if np.any(inside_box):
                    min_data_distance = 0  # Data inside box - bad score
                    break

                # Calculate minimum distance to box edges
                # Distance to box center
                box_center_x = (box_x_min + box_x_max) / 2
                box_center_y = (box_y_min + box_y_max) / 2

                # Normalize for fair comparison
                norm_box_x = (box_center_x - x_min) / (x_max - x_min)
                norm_box_y = (box_center_y - y_min) / (y_max - y_min)
                norm_data_x = (x_data - x_min) / (x_max - x_min)
                norm_data_y = (y_data - y_min) / (y_max - y_min)

                distances = np.sqrt((norm_data_x - norm_box_x)**2 + (norm_data_y - norm_box_y)**2)
                min_data_distance = min(min_data_distance, np.min(distances))

        score += min_data_distance * 10  # Weight heavily

        # 2. Check against avoid_boxes (other annotations)
        box_conflict = False
        if avoid_boxes:
            for avoid_x, avoid_y, avoid_w, avoid_h in avoid_boxes:
                # Check for overlap
                if not (box_x_max < avoid_x or box_x_min > avoid_x + avoid_w or
                        box_y_max < avoid_y or box_y_min > avoid_y + avoid_h):
                    box_conflict = True
                    break

        if box_conflict:
            score -= 5  # Heavy penalty for overlapping other annotations

        # 3. Prefer edge positions over interior
        edge_distance = min(rel_x, 1-rel_x, rel_y, 1-rel_y)
        if edge_distance < 0.2:  # Close to edge
            score += 1

        # 4. Avoid extreme corners if data is sparse there
        if rel_x < 0.1 or rel_x > 0.9 or rel_y < 0.1 or rel_y > 0.9:
            score += 0.5  # Small bonus for corners

        # 5. Penalize positions that would place box outside plot area
        if (box_x_min < x_min or box_x_max > x_max or
            box_y_min < y_min or box_y_max > y_max):
            score -= 2

        # Update best position
        if score > best_score:
            best_score = score
            best_position = (test_x, test_y)

    # If no good position found, use fallback
    if best_position is None or best_score < 0:
        best_position = (x_min + 0.85 * (x_max - x_min),
                        y_min + 0.85 * (y_max - y_min))

    return best_position


def find_clear_info_box_position(ax, data_points, avoid_boxes=None):
    """
    Find the best position for the info box that avoids data and other annotations.

    Returns position in transform coordinates (0-1 range) and alignment.
    """

    # Get axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Analyze data distribution first
    x_data_sets = [dp[0] for dp in data_points]
    y_data_sets = [dp[1] for dp in data_points]
    density_map, clear_regions = analyze_data_distribution(x_data_sets, y_data_sets, (x_min, x_max), (y_min, y_max))

    # Candidate positions for info box (in transform coordinates)
    candidates = [
        (0.02, 0.98, 'top-left', 'top', 'left'),
        (0.02, 0.5, 'mid-left', 'center', 'left'),
        (0.02, 0.02, 'bottom-left', 'bottom', 'left'),
        (0.98, 0.98, 'top-right', 'top', 'right'),
        (0.98, 0.5, 'mid-right', 'center', 'right'),
        (0.98, 0.02, 'bottom-right', 'bottom', 'right'),
        (0.5, 0.98, 'top-center', 'top', 'center'),
        (0.5, 0.02, 'bottom-center', 'bottom', 'center'),
    ]

    best_score = -1
    best_pos = (0.02, 0.98)  # Default to top-left
    best_va = 'top'
    best_ha = 'left'

    # Info box dimensions in relative coordinates
    info_box_width = 0.3
    info_box_height = 0.35

    for rel_x, rel_y, position_name, va, ha in candidates:

        # Calculate info box bounds in data coordinates
        if ha == 'left':
            box_x_min = x_min + rel_x * (x_max - x_min)
            box_x_max = box_x_min + info_box_width * (x_max - x_min)
        elif ha == 'right':
            box_x_max = x_min + rel_x * (x_max - x_min)
            box_x_min = box_x_max - info_box_width * (x_max - x_min)
        else:  # center
            box_center_x = x_min + rel_x * (x_max - x_min)
            box_x_min = box_center_x - info_box_width * (x_max - x_min) / 2
            box_x_max = box_center_x + info_box_width * (x_max - x_min) / 2

        if va == 'top':
            box_y_max = y_min + rel_y * (y_max - y_min)
            box_y_min = box_y_max - info_box_height * (y_max - y_min)
        elif va == 'bottom':
            box_y_min = y_min + rel_y * (y_max - y_min)
            box_y_max = box_y_min + info_box_height * (y_max - y_min)
        else:  # center
            box_center_y = y_min + rel_y * (y_max - y_min)
            box_y_min = box_center_y - info_box_height * (y_max - y_min) / 2
            box_y_max = box_center_y + info_box_height * (y_max - y_min) / 2

        # Calculate score
        score = 0

        # 1. Count data points inside the box area (heavily penalized)
        points_inside = 0
        for x_data, y_data in zip(x_data_sets, y_data_sets):
            if len(x_data) > 0 and len(y_data) > 0:
                inside = ((x_data >= box_x_min) & (x_data <= box_x_max) &
                         (y_data >= box_y_min) & (y_data <= box_y_max))
                points_inside += np.sum(inside)

        score -= points_inside * 3  # Heavy penalty for each point inside

        # 2. Calculate minimum distance to nearest data point
        min_distance = 1.0
        for x_data, y_data in zip(x_data_sets, y_data_sets):
            if len(x_data) > 0 and len(y_data) > 0:
                box_center_x = (box_x_min + box_x_max) / 2
                box_center_y = (box_y_min + box_y_max) / 2

                norm_box_x = (box_center_x - x_min) / (x_max - x_min)
                norm_box_y = (box_center_y - y_min) / (y_max - y_min)
                norm_data_x = (x_data - x_min) / (x_max - x_min)
                norm_data_y = (y_data - y_min) / (y_max - y_min)

                distances = np.sqrt((norm_data_x - norm_box_x)**2 + (norm_data_y - norm_box_y)**2)
                min_distance = min(min_distance, np.min(distances))

        score += min_distance * 5  # Reward distance from data

        # 3. Check against avoid_boxes (other annotations)
        if avoid_boxes:
            for avoid_x, avoid_y, avoid_w, avoid_h in avoid_boxes:
                # Check for overlap
                if not (box_x_max < avoid_x or box_x_min > avoid_x + avoid_w or
                        box_y_max < avoid_y or box_y_min > avoid_y + avoid_h):
                    score -= 10  # Huge penalty for overlapping annotations

        # 4. Prefer traditional positions
        if position_name in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
            score += 1

        # 5. Ensure box stays within plot bounds
        if (box_x_min >= x_min and box_x_max <= x_max and
            box_y_min >= y_min and box_y_max <= y_max):
            score += 2
        else:
            score -= 3  # Penalty for going outside bounds

        # Update best position
        if score > best_score:
            best_score = score
            best_pos = (rel_x, rel_y)
            best_va = va
            best_ha = ha

    return best_pos, best_va, best_ha


def estimate_annotation_box_size(text_content, fontsize=9):
    """
    Estimate the size of a text box in relative plot coordinates.

    Returns:
    - (width, height) in relative coordinates (0-1 range)
    """

    lines = text_content.split('\n')
    num_lines = len(lines)
    max_line_length = max(len(line) for line in lines) if lines else 0

    # Rough estimates based on typical font metrics
    char_width_rel = 0.008  # Approximate character width in relative coords
    line_height_rel = 0.025  # Approximate line height in relative coords
    padding_rel = 0.02  # Padding around text

    width = max_line_length * char_width_rel + padding_rel
    height = num_lines * line_height_rel + padding_rel

    return width, height


def plot_contrast_curve_full(results, title="Contrast Curve", telescope_name="Hooker",
                             telescope_diameter=2.54, wavelength=617e-9, confidence_level=5.0,
                             save_path=None, show_plot=True, date_obs=None):
    """Plot contrast curve with smart annotation placement that avoids data."""

    # Close any existing figures to prevent data persistence
    plt.close('all')

    # Extract results
    max_seps = results['max_seps']
    max_mags = results['max_mags']
    min_seps = results['min_seps']
    min_mags = results['min_mags']
    radii = results['radii']
    limits = results['limits']

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Clear the figure explicitly
    fig.clear()
    ax = fig.add_subplot(111)

    # Add title with target name
    fig.suptitle(f'{title} - {telescope_name} {telescope_diameter:.1f}m Telescope',
                 fontsize=14, fontweight='bold')

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

    # Set initial axis limits
    if len(max_seps) > 0:
        x_min, x_max = 0, max(np.max(max_seps), np.max(radii)) * 1.1
    else:
        x_min, x_max = 0, np.max(radii) * 1.1

    y_min, y_max = 0, max(10, np.max(limits) + 0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Prepare data sets for smart annotation placement
    x_data_sets = [max_seps, radii]
    y_data_sets = [max_mags, limits]
    if len(min_seps) > 0:
        x_data_sets.append(min_seps)
        y_data_sets.append(min_mags)

    # Track annotation boxes to prevent overlaps
    annotation_boxes = []  # (x, y, width, height) in data coordinates

    # Find and highlight detections (maxima below the limit)
    detection_annotation_placed = False
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

            # Annotate the brightest detection with smart positioning
            if len(det_mags) > 0:
                brightest_idx = np.argmin(det_mags)
                sep = det_seps[brightest_idx]
                mag = det_mags[brightest_idx]

                # Get the limiting magnitude at this separation
                limiting_mag = np.interp(sep, radii, limits)

                # Find clear area for detection annotation (avoiding existing boxes)
                annotation_pos = find_clear_annotation_area(ax, x_data_sets, y_data_sets,
                                                          avoid_boxes=annotation_boxes,
                                                          annotation_type="detection")

                # Annotation text
                sep_pixels = sep / results['pixel_scale']
                annotation_text = (f'Δm = {mag:.2f}\n' +
                                   f'Limit = {limiting_mag:.2f}\n' +
                                   f'Sep = {sep:.3f}" ({sep_pixels:.1f} px)')

                # Estimate annotation box size and add to avoid list
                box_width, box_height = estimate_annotation_box_size(annotation_text, fontsize=9)
                x_range = x_max - x_min
                y_range = y_max - y_min
                box_data_width = box_width * x_range
                box_data_height = box_height * y_range

                # Add buffer around the annotation
                buffer = 0.02
                annotation_boxes.append((
                    annotation_pos[0] - box_data_width/2 - buffer*x_range,
                    annotation_pos[1] - box_data_height/2 - buffer*y_range,
                    box_data_width + 2*buffer*x_range,
                    box_data_height + 2*buffer*y_range
                ))

                ax.annotate(annotation_text,
                            xy=(sep, mag),
                            xytext=annotation_pos,
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.0,
                                          connectionstyle="arc3,rad=0.2"),
                            fontsize=9, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow',
                                    edgecolor='red', alpha=0.9, linewidth=1))

                detection_annotation_placed = True

    # Labels and formatting
    ax.set_xlabel('Separation [arcsec]', fontsize=13)
    ax.set_ylabel('Magnitude Difference', fontsize=13)

    # Secondary x-axis for pixels
    ax2 = ax.twiny()
    ax2.set_xlabel('Separation [pixels]', fontsize=13)
    ax2.set_xlim(x_min / results['pixel_scale'], x_max / results['pixel_scale'])

    # Make sure the pixel axis ticks are at nice round numbers
    pixel_ticks = np.arange(0, x_max / results['pixel_scale'] + 10, 10)
    ax2.set_xticks(pixel_ticks)

    # Legend in upper right
    ax.legend(fontsize=10, loc='upper right', frameon=True, fancybox=False)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax2.tick_params(axis='both', which='major', labelsize=11)

    # Create info box with smart positioning that avoids detections
    info_text = []
    info_text.append(f'Target: {title}')
    info_text.append(f'Telescope: {telescope_name} {telescope_diameter:.2f}m')
    info_text.append(f'Wavelength: {wavelength * 1e9:.0f} nm')
    info_text.append(f'Pixel scale: {results["pixel_scale"]:.4f}"/pix ({results["pixel_scale"] * 1000:.1f} mas/pix)')
    info_text.append(f'PSF FWHM: {results["fwhm_arcsec"]:.3f}" ({results["fwhm_pixels"]:.1f} pix)')
    info_text.append(f'λ/D: {results["lambda_over_d"]:.3f}" ({results["lambda_over_d"] / results["pixel_scale"]:.1f} pix)')
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

    # Find best position for info box (avoiding detection annotations)
    info_pos, va, ha = find_clear_info_box_position(ax, list(zip(x_data_sets, y_data_sets)),
                                                   avoid_boxes=annotation_boxes)

    # Create the info text box
    info_str = '\n'.join(info_text)
    info_box = ax.text(info_pos[0], info_pos[1], info_str, transform=ax.transAxes,
                      fontsize=9, verticalalignment=va, horizontalalignment=ha,
                      bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                edgecolor='gray', alpha=0.95, linewidth=1))

    # Add a subtle enhancement: if both annotations are present, make sure they don't visually compete
    if detection_annotation_placed:
        # Make the info box slightly more transparent to de-emphasize it
        info_box.get_bbox_patch().set_alpha(0.85)

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
        self.root.title("Contrast Curve Generator - Smart Annotations")
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
        self.max_radius = tk.IntVar(value=50)
        self.min_radius = tk.IntVar(value=2)
        self.smoothing_sigma = tk.DoubleVar(value=3.0)
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

        # Smoothing control
        ttk.Label(param_frame, text="Smoothing (sigma):").grid(row=5, column=0, sticky=tk.W)
        ttk.Spinbox(param_frame, from_=0.0, to=10.0, increment=0.5,
                    textvariable=self.smoothing_sigma, width=10).grid(row=5, column=1)
        ttk.Label(param_frame, text="(0=none, 5=aggressive)", font=('TkDefaultFont', 8)).grid(row=5, column=2,
                                                                                              columnspan=2, sticky=tk.W)

        # Filter presets
        filter_frame = ttk.Frame(param_frame)
        filter_frame.grid(row=6, column=0, columnspan=4, pady=5)
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

        self.log_status("Contrast Curve Generator Ready (Smart Annotations)")
        self.log_status("Method: median(maxima) + 5×avg(MAD_max, MAD_min)")
        self.log_status("Annotations will automatically avoid data points")

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
            self.log_status("\nGenerating contrast curve with smart annotations...")

            # Close any existing figures to prevent data persistence
            plt.close('all')

            # Clear any existing figure reference
            if hasattr(self, 'fig'):
                plt.close(self.fig)
                del self.fig

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
                telescope_name=self.telescope_name.get(),
                smoothing_sigma=self.smoothing_sigma.get()
            )

            self.results = results

            # Get date from header if available
            date_obs = None
            if hasattr(self, 'header'):
                header_dict = dict(self.header)
                date_obs = header_dict.get('DATE-OBS', None)

            # Plot and store the new figure
            self.fig = plot_contrast_curve_full(
                results,
                title=self.target_name.get(),
                telescope_name=self.telescope_name.get(),
                telescope_diameter=telescope_diameter_m,
                wavelength=self.wavelength.get() * 1e-9,
                confidence_level=self.confidence_level.get(),
                date_obs=date_obs
            )

            self.log_status("Contrast curve generated with optimized annotation placement")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate curve: {str(e)}")
            self.log_status(f"Error: {str(e)}")
            import traceback
            self.log_status(traceback.format_exc())

    def save_plot(self):
        """Save the current plot."""
        # Check if there are any figures open
        if len(plt.get_fignums()) == 0:
            messagebox.showerror("Error", "No plot to save. Generate a curve first.")
            return

        filename = self.save_path.get()
        if not filename:
            messagebox.showerror("Error", "Please specify a save location")
            return

        try:
            # Get the current figure from pyplot
            current_fig = plt.gcf()

            # Save it
            current_fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')

            self.log_status(f"Saved: {Path(filename).name}")
            messagebox.showinfo("Success", "Plot saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
            self.log_status(f"Error: {str(e)}")


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