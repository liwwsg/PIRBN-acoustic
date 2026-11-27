"""
COPYRIGHT NOTICE
© 2026 Weiwei Li, Chenchen Wang. All Rights Reserved.

Redistribution and use in source or binary forms, with or without
modification, are strictly prohibited unless expressly permitted
in writing by the copyright holder.

DISCLAIMER:
This code is provided "AS IS," without warranty of any kind, express or implied, including
but not limited to the warranties of merchantability, fitness for a particular purpose, and
non-infringement. In no event shall the copyright holder be liable for any claim, damages,
or other liability, whether in an action of contract, tort, or otherwise, arising from, out
of or in connection with the code or the use or other dealings in the code.
--------------------------------------------------------------------------
References：
Chenchen Yang, Weiwei Li. A physics-informed radial basis network for acoustic scattering and radiation problems, submitted.
--------------------------------------------------------------------------
School of Transportation and Vehicle Engineering, Shandong University of Technology University
E-mail: liww@sdut.edu.cn
"""

"""Gaussian PIRBN for two-dimensional sound-soft circular-cylinder scattering."""

import math
import time
import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib import cm
from matplotlib import ticker as mticker
import scipy.special as sp
from scipy.stats import qmc

# Problem and wavenumbers
CASE_NAME = "Sound-soft circular-cylinder scattering"
RUN_BATCH = True  # Use K_VALUES when True; otherwise use SINGLE_K.
SINGLE_K = 15
K_VALUES = [5, 10, 15, 20]

# Initial point-layout output
POINT_LAYOUT_ONLY = False  # Save the initial point layout without training.
DRAW_INITIAL_POINT_LAYOUT = False
SHOW_INITIAL_POINT_LAYOUT = False
INITIAL_LAYOUT_LEGEND_FONTSIZE = 12

# Training and learning-rate schedule
N_EPOCHS = 20000
LEARNING_RATE = 0.005
PRINT_EVERY = 500
TRAINING_PROGRESS_ITERATIONS = (5000, 10000, 15000, 20000)
BETA0_COEFF = 0.49  # Initial raw beta = BETA0_COEFF * k**2.

# Geometry and boundary conditions
OUTER_R = 3.0
INNER_R = 1.0
INCIDENT_DIR = np.array([1.0, 0.0], dtype=np.float32)
OUTER_BC_TYPE = "multi_dtn"

# Fourier-series DtN constraints
DTN_CIRCLE_COUNT = 3
OBSTACLE_RADIUS = 1
DTN_MIN_RADIUS = OBSTACLE_RADIUS + (OUTER_R - OBSTACLE_RADIUS) / DTN_CIRCLE_COUNT
DTN_DELTA_N = 10  # Fourier cutoff N_m = ceil(k*R_m + DTN_DELTA_N).
if (
    not isinstance(DTN_CIRCLE_COUNT, int)
    or isinstance(DTN_CIRCLE_COUNT, bool)
    or DTN_CIRCLE_COUNT < 1
):
    raise ValueError("DTN_CIRCLE_COUNT must be a positive integer.")
DTN_RADII = tuple(
    (round(float(r), 12) for r in np.linspace(DTN_MIN_RADIUS, OUTER_R, DTN_CIRCLE_COUNT))
)

# Loss weights and residual normalization
W_PDE = 1.0
W_BC1 = 10.0
W_BC2 = 3.0
NORMALIZE_TRAINING_RESIDUALS = False  # Divide PDE residuals by k**2 and DtN residuals by k.

# Wavelength-based point and center densities
lamda = 4.0
PPW_PDE = 8.0  # Interior collocation density per wavelength.
PPW_BC1 = PPW_PDE * lamda
PPW_BC2 = PPW_PDE * lamda
PPW_C_INTERIOR = PPW_PDE / 3
PPW_C_INNER = PPW_C_INTERIOR * lamda
PPW_C_OUTER = PPW_C_INTERIOR * lamda

# Trainable center offsets
USE_CENTER_TRAIN_SCALE = True
CENTER_TRAIN_SCALE = 0.2
INPUT_DIM = 2

# Mixed residual-based adaptive distribution (RAD)
USE_RAD = True
RAD_START_ITER = 2000  # Earliest resampling iteration, counted from one.
RAD_PERIOD = 1000
RAD_MIX_RATIO = 0.15  # Fraction of points selected by residual-weighted sampling.
RAD_POOL_FACTOR = 4
RAD_POOL_MIN = 4000
RAD_BATCH_SIZE = 2048  # Candidate-scoring batch size; training uses all collocation points.
RAD_POWER = 1.0
RAD_C = 0.0
RAD_EPS = 0.0001
RAD_NORMALIZATION_EPS = 1e-12
RAD_RANDOM_SEED = 1234
RAD_REFRESH_UNIFORM_PART = True  # Regenerate background points at each RAD update.
RAD_DIAGNOSTIC_IF_NO_EVENT = False


# Output and evaluation
OUTPUT_DIR = f"PIRBN_CircleScattering_MultiDtN{('_Normalized' if NORMALIZE_TRAINING_RESIDUALS else '')}_FixedLR_{LEARNING_RATE:g}_results"
EVAL_GRID_SIZE = 241
OBSERVATION_RADIUS = 2.0
OBSERVATION_N = 150
OBSERVATION_OFFSET_INDEX = 0.0
EXACT_TERMS = 120  # Highest order in the analytical reference series.
EVALUATION_BOUNDARY_POLICY = "inclusive"
ERROR_COLOR_PERCENTILE = 100.0  # Percentile used for error-plot color limits.
PREDICTION_BATCH_SIZE = 2048
INCIDENT_DIR = INCIDENT_DIR / np.linalg.norm(INCIDENT_DIR)

# Plot settings
MAKE_SIX_PANEL = True
SAVE_FIGURES = True
SHOW_FIGURES = False
SAVE_SINGLE_K_LOSS_FIGURES = True
PLOT_ORIGINAL_FIGURES = False
PAPER_FONT_SIZE = 11
PAPER_TICK_SIZE = 10
PAPER_DPI = 220
FIELD_CMAP = "coolwarm"
ERROR_CMAP = "coolwarm"
REFERENCE_ERROR_CMAP = "viridis"
FIELD_LEVELS = 101
ERROR_LEVELS = 101
FIELD_PERCENTILE = 100.0
ERROR_PERCENTILE = ERROR_COLOR_PERCENTILE
DRAW_CONTOUR_LINES = True
CONTOUR_LINE_LEVELS = 14
plt.rcParams.update(
    {
        "font.size": PAPER_FONT_SIZE,
        "axes.titlesize": PAPER_FONT_SIZE,
        "axes.labelsize": PAPER_FONT_SIZE,
        "xtick.labelsize": PAPER_TICK_SIZE,
        "ytick.labelsize": PAPER_TICK_SIZE,
        "legend.fontsize": PAPER_TICK_SIZE,
        "figure.titlesize": PAPER_FONT_SIZE,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.8,
    }
)


# File and resource utilities
def format_k_label(k_value):
    """Format a wavenumber for filenames."""
    k_float = float(k_value)
    if abs(k_float - round(k_float)) < 1e-12:
        return str(int(round(k_float)))
    return str(k_float).replace(".", "p")


TRAINING_RESOURCE_COLUMNS = (
    "training_time_seconds",
    "training_time_minutes",
    "training_time_hms",
    "training_iterations",
    "training_device",
    "training_device_name",
    "gpu_memory_status",
    "peak_cuda_allocated_mib",
    "peak_cuda_reserved_mib",
    "training_resources_path",
)


def format_elapsed_time(seconds):
    """Format elapsed seconds as HH:MM:SS.mmm."""
    total_milliseconds = int(round(float(seconds) * 1000.0))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def start_training_resource_monitor(device):
    """Start timing and reset CUDA peak-memory counters for the current wavenumber."""
    device = torch.device(device)
    is_cuda = device.type == "cuda"
    device_name = torch.cuda.get_device_name(device) if is_cuda else "CPU"
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return {"device": device, "device_name": device_name, "started_at": time.perf_counter()}


def finish_training_resource_monitor(state, completed_iterations):
    """Record training-loop time and peak CUDA allocation before final evaluation."""
    device = state["device"]
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - state["started_at"]
    allocated_bytes = int(torch.cuda.max_memory_allocated(device)) if is_cuda else None
    reserved_bytes = int(torch.cuda.max_memory_reserved(device)) if is_cuda else None
    return {
        "training_time_seconds": float(elapsed),
        "training_time_minutes": float(elapsed / 60.0),
        "training_time_hms": format_elapsed_time(elapsed),
        "training_iterations": int(completed_iterations),
        "training_device": str(device),
        "training_device_name": state["device_name"],
        "gpu_memory_status": "measured" if is_cuda else "not_applicable_cpu",
        "peak_cuda_allocated_bytes": allocated_bytes,
        "peak_cuda_reserved_bytes": reserved_bytes,
        "peak_cuda_allocated_mib": allocated_bytes / 1024.0**2 if is_cuda else None,
        "peak_cuda_reserved_mib": reserved_bytes / 1024.0**2 if is_cuda else None,
    }


def save_training_resource_summary(batch_results, output_dir):
    """Write per-wavenumber training time and memory statistics."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "training_resources.csv")
    fieldnames = ["k", "use_rad"] + list(TRAINING_RESOURCE_COLUMNS)
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in batch_results:
            if result.get("point_layout_only"):
                continue
            row = {name: result.get(name) for name in fieldnames}
            row["k"] = format_k_label(result["k"])
            writer.writerow(row)
    return path


# Evaluation utilities
def compute_complex_metrics_from_arrays(u_pred, u_ref):
    """Compute relative L2, absolute-error, and R2 metrics on finite complex-valued pairs."""
    u_pred = np.asarray(u_pred, dtype=np.complex128)
    u_ref = np.asarray(u_ref, dtype=np.complex128)
    valid = (
        np.isfinite(np.real(u_pred))
        & np.isfinite(np.imag(u_pred))
        & np.isfinite(np.real(u_ref))
        & np.isfinite(np.imag(u_ref))
    )
    u_pred = u_pred[valid]
    u_ref = u_ref[valid]
    if u_pred.size == 0:
        nan = float("nan")
        return {
            "rel_l2_complex": nan,
            "rel_l2_real": nan,
            "rel_l2_imag": nan,
            "max_abs_error": nan,
            "max_abs_error_real": nan,
            "max_abs_error_imag": nan,
            "mean_abs_error": nan,
            "mean_abs_error_real": nan,
            "mean_abs_error_imag": nan,
            "r2_real": nan,
            "r2_imag": nan,
            "r2_avg": nan,
        }
    diff = u_pred - u_ref
    eps = 1e-12
    rel_l2_complex = np.linalg.norm(diff) / (np.linalg.norm(u_ref) + eps)
    rel_l2_real = np.linalg.norm(np.real(diff)) / (np.linalg.norm(np.real(u_ref)) + eps)
    rel_l2_imag = np.linalg.norm(np.imag(diff)) / (np.linalg.norm(np.imag(u_ref)) + eps)
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    max_abs_real = np.max(np.abs(np.real(diff)))
    max_abs_imag = np.max(np.abs(np.imag(diff)))
    mean_abs_real = np.mean(np.abs(np.real(diff)))
    mean_abs_imag = np.mean(np.abs(np.imag(diff)))

    def r2_score(a, b):
        ss_res = np.sum((a - b) ** 2)
        ss_tot = np.sum((b - np.mean(b)) ** 2)
        return 1.0 - ss_res / (ss_tot + eps)

    r2_real = r2_score(np.real(u_pred), np.real(u_ref))
    r2_imag = r2_score(np.imag(u_pred), np.imag(u_ref))
    r2_avg = 0.5 * (r2_real + r2_imag)
    return {
        "rel_l2_complex": float(rel_l2_complex),
        "rel_l2_real": float(rel_l2_real),
        "rel_l2_imag": float(rel_l2_imag),
        "max_abs_error": float(max_abs),
        "max_abs_error_real": float(max_abs_real),
        "max_abs_error_imag": float(max_abs_imag),
        "mean_abs_error": float(mean_abs),
        "mean_abs_error_real": float(mean_abs_real),
        "mean_abs_error_imag": float(mean_abs_imag),
        "r2_real": float(r2_real),
        "r2_imag": float(r2_imag),
        "r2_avg": float(r2_avg),
    }


def observation_circle_points():
    """Return equiangular observation points without duplicating the endpoint."""
    theta = (np.arange(OBSERVATION_N, dtype=np.float64) + OBSERVATION_OFFSET_INDEX) * (
        2.0 * np.pi / OBSERVATION_N
    )
    xy = np.column_stack((OBSERVATION_RADIUS * np.cos(theta), OBSERVATION_RADIUS * np.sin(theta)))
    return xy.astype(np.float32)


def postprocess_analytical_reference(xy, k):
    """Evaluate the circular-cylinder scattering series in complex128."""
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    rr = np.hypot(points[:, 0], points[:, 1])
    theta = np.arctan2(points[:, 1], points[:, 0])
    ka = k * INNER_R
    value = -(sp.jv(0, ka) / sp.hankel1(0, ka)) * sp.hankel1(0, k * rr)
    for order in range(1, EXACT_TERMS + 1):
        value -= (
            2.0
            * 1j**order
            * (sp.jv(order, ka) / sp.hankel1(order, ka))
            * sp.hankel1(order, k * rr)
            * np.cos(order * theta)
        )
    return np.asarray(value, dtype=np.complex128)


def aligned_error_metrics(prediction, reference, prefix):
    """Compute complex and componentwise error metrics on finite reference pairs."""
    prediction = np.asarray(prediction, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    valid = np.isfinite(prediction.real) & np.isfinite(prediction.imag)
    valid &= np.isfinite(reference.real) & np.isfinite(reference.imag)
    prediction, reference = (prediction[valid], reference[valid])
    names = ("rel_l2_complex", "rel_l2_real", "rel_l2_imag", "max_abs_complex", "mean_abs_complex")
    if not len(reference):
        return {prefix + name: float("nan") for name in names}
    error = prediction - reference
    with np.errstate(divide="ignore", invalid="ignore"):
        values = (
            np.linalg.norm(error) / np.linalg.norm(reference),
            np.linalg.norm(error.real) / np.linalg.norm(reference.real),
            np.linalg.norm(error.imag) / np.linalg.norm(reference.imag),
            np.max(np.abs(error)),
            np.mean(np.abs(error)),
        )
    return {prefix + name: float(value) for name, value in zip(names, values)}


def save_observation_circle_csv(path, xy, prediction, reference, domain_valid):
    """Save observation coordinates, predicted and reference fields, and pointwise errors."""
    theta_deg = (np.arange(OBSERVATION_N, dtype=np.float64) + OBSERVATION_OFFSET_INDEX) * (
        360.0 / OBSERVATION_N
    )
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "x",
                "y",
                "prediction_real",
                "prediction_imag",
                "reference_real",
                "reference_imag",
                "abs_error_complex",
                "theta_deg",
                "abs_error_real",
                "abs_error_imag",
                "domain_valid",
                "reference_valid",
            ]
        )
        for index, (point, pred, ref) in enumerate(zip(xy, prediction, reference)):
            writer.writerow(
                [
                    point[0],
                    point[1],
                    pred.real,
                    pred.imag,
                    ref.real,
                    ref.imag,
                    abs(pred - ref),
                    theta_deg[index],
                    abs(pred.real - ref.real),
                    abs(pred.imag - ref.imag),
                    int(domain_valid[index]),
                    int(np.isfinite(ref.real) and np.isfinite(ref.imag)),
                ]
            )


# Scattering solver
def run_single_case(k_value, output_dir):
    """Train and evaluate the scattering solution for one wavenumber."""
    start_time = time.time()
    if OUTER_BC_TYPE not in ("multi_dtn", "exact_dirichlet"):
        raise ValueError('OUTER_BC_TYPE must be "multi_dtn" or "exact_dirichlet".')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    np.random.seed(1234)
    print("Using device:", device)
    print("CUDA available:", torch.cuda.is_available())
    _original_plt_show = plt.show
    case_output_dir = os.path.join(output_dir, f"k_{format_k_label(k_value)}")
    os.makedirs(case_output_dir, exist_ok=True)
    k = float(k_value)
    if NORMALIZE_TRAINING_RESIDUALS and (not math.isfinite(k) or k <= 0.0):
        raise ValueError("Residual normalization requires a finite, positive wavenumber.")
    pde_residual_scale = k**2 if NORMALIZE_TRAINING_RESIDUALS else 1.0
    derivative_residual_scale = k if NORMALIZE_TRAINING_RESIDUALS else 1.0
    n_epochs = N_EPOCHS
    lr = LEARNING_RATE
    print_every = PRINT_EVERY
    beta0 = BETA0_COEFF * k**2
    input_dim = INPUT_DIM

    def safe_divide(a, b, eps=1e-12):
        return a / (b + eps)

    def sample_annulus(n, r_in, r_out):
        """Sample points uniformly in area within an annulus."""
        r = np.sqrt(np.random.rand(n) * (r_out**2 - r_in**2) + r_in**2)
        t = np.random.rand(n) * 2.0 * np.pi
        return np.stack([r * np.cos(t), r * np.sin(t)], axis=1).astype(np.float32)

    def sample_circle(n, radius):
        """Return equiangular circle points and outward radial normals."""
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        xy = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1).astype(np.float32)
        normals = (xy / radius).astype(np.float32)
        return (xy, normals)

    def sample_pde_domain(n):
        """Sample points in the fluid annulus."""
        return sample_annulus(n, INNER_R, OUTER_R)

    def exact_solution_dirichlet_circle_scattering(rr, tt, n_terms=120):
        """Evaluate the outgoing scattered field for a sound-soft circular cylinder."""
        a = INNER_R
        ka = k * a
        H0_ka = sp.hankel1(0, ka)
        J0_ka = sp.jv(0, ka)
        H0_kr = sp.hankel1(0, k * rr)
        u_exact = -safe_divide(J0_ka, H0_ka) * H0_kr
        for n in range(1, n_terms + 1):
            Hn_ka = sp.hankel1(n, ka)
            Jn_ka = sp.jv(n, ka)
            Hn_kr = sp.hankel1(n, k * rr)
            term = 1j**n * safe_divide(Jn_ka, Hn_ka) * Hn_kr * np.cos(n * tt)
            u_exact -= 2.0 * term
        return u_exact

    def exact_solution_from_xy(xy_np):
        """Evaluate the circular scattering series at Cartesian coordinates."""
        rr = np.sqrt(xy_np[:, 0] ** 2 + xy_np[:, 1] ** 2)
        tt = np.arctan2(xy_np[:, 1], xy_np[:, 0])
        return exact_solution_dirichlet_circle_scattering(rr, tt)

    if OUTER_BC_TYPE == "multi_dtn":
        if (
            not DTN_RADII
            or any((not math.isfinite(r) or r <= 0 for r in DTN_RADII))
            or any((b <= a for a, b in zip(DTN_RADII, DTN_RADII[1:])))
            or (not math.isclose(DTN_RADII[-1], OUTER_R))
        ):
            raise ValueError("DTN_RADII must be strictly increasing and end at OUTER_R.")
        obstacle_radius_bound = INNER_R
        invalid_radii = [r for r in DTN_RADII if r <= obstacle_radius_bound or r > OUTER_R]
        if invalid_radii:
            raise ValueError(
                f"Obstacle radius: {obstacle_radius_bound:.6g}. Invalid DtN radii: {invalid_radii}. Each DtN radius must exceed the obstacle radius and not exceed OUTER_R={OUTER_R}."
            )
    if not PLOT_ORIGINAL_FIGURES:
        plt.show = lambda *args, **kwargs: plt.close("all")
    wavelength = 2.0 * math.pi / k
    area_domain = math.pi * (OUTER_R**2 - INNER_R**2)
    N_wave_inner = INNER_R * k
    outer_boundary_length = 2.0 * math.pi * OUTER_R
    N_wave_outer = outer_boundary_length / wavelength
    N_wave_area = area_domain / wavelength**2
    M_pde = math.ceil(PPW_PDE**2 * N_wave_area)
    M_bc1 = math.ceil(PPW_BC1 * N_wave_inner)
    M_bc2 = math.ceil(PPW_BC2 * N_wave_outer)
    Nc_interior = math.ceil(PPW_C_INTERIOR**2 * N_wave_area)
    Nc_inner = math.ceil(PPW_C_INNER * N_wave_inner)
    Nc_outer = math.ceil(PPW_C_OUTER * N_wave_outer)
    Nc = Nc_interior + Nc_inner + Nc_outer
    print("---------- Wavelength-based resolution ----------")
    print("Boundary: Dirichlet / sound-soft circle")
    print(f"k = {k:.4f}")
    print(f"wavelength = {wavelength:.6f}")
    print(f"area_domain = {area_domain:.6f}")
    print(f"N_wave_inner_boundary = {N_wave_inner:.3f}")
    print(f"N_wave_outer_boundary = {N_wave_outer:.3f}")
    print(f"N_wave_area = {N_wave_area:.3f}")
    print(f"Nc_interior = {Nc_interior}")
    print(f"Nc_inner = {Nc_inner}")
    print(f"Nc_outer = {Nc_outer}")
    print(f"Nc_total = {Nc}")
    print(f"M_pde = {M_pde}")
    print(f"M_bc1 = {M_bc1}")
    if OUTER_BC_TYPE == "multi_dtn":
        print(f"M_bc2 (outer sampling count; DtN uses Q_m=4*N_m+1) = {M_bc2}")
    else:
        print(f"M_bc2 = {M_bc2}")
    print(f"OUTER_BC_TYPE = {OUTER_BC_TYPE}")
    print("-------------------------------------------------")
    xy_pde_np = sample_annulus(M_pde, INNER_R, OUTER_R)
    xy_bc1_np, _ = sample_circle(M_bc1, INNER_R)
    xy_bc2_np, n_bc2_np = sample_circle(M_bc2, OUTER_R)
    center_interior_np = sample_annulus(Nc_interior, INNER_R, OUTER_R)
    center_inner_np, _ = sample_circle(Nc_inner, INNER_R)
    center_outer_np, _ = sample_circle(Nc_outer, OUTER_R)
    xy_pde_uniform_base_np = xy_pde_np[np.random.permutation(len(xy_pde_np))].copy()
    xy_pde = torch.tensor(xy_pde_np, dtype=torch.float32, device=device)
    xy_bc1 = torch.tensor(xy_bc1_np, dtype=torch.float32, device=device)
    xy_bc2 = torch.tensor(xy_bc2_np, dtype=torch.float32, device=device)
    n_bc2 = torch.tensor(n_bc2_np, dtype=torch.float32, device=device)
    center_phys = np.vstack([center_interior_np, center_inner_np, center_outer_np]).astype(
        np.float32
    )
    c_base = torch.tensor(center_phys, dtype=torch.float32, device=device).detach()

    def dtn_mode_cutoff(radius):
        """Return the Fourier cutoff ceil(k * radius + DTN_DELTA_N)."""
        return int(math.ceil(k * radius + DTN_DELTA_N))

    def dtn_multiplier(n_boundary, radius, n_dtn):
        """Construct Fourier-Hankel multipliers with zero values beyond the cutoff."""
        modes = np.rint(np.fft.fftfreq(n_boundary) * n_boundary).astype(np.int64)
        absolute = np.abs(modes)
        z_value = k * radius
        multiplier = np.zeros(n_boundary, dtype=np.complex64)
        for order in np.unique(absolute[absolute <= n_dtn]):
            multiplier[absolute == order] = (
                k * sp.h1vp(int(order), z_value, 1) / sp.hankel1(int(order), z_value)
            )
        return torch.tensor(multiplier, dtype=torch.complex64, device=device)

    dtn_contours = []
    dtn_truncation_orders = []
    dtn_point_counts = []
    dtn_circle_weights = []
    if OUTER_BC_TYPE == "multi_dtn":
        for circle_index, radius in enumerate(DTN_RADII, start=1):
            n_dtn = dtn_mode_cutoff(radius)
            n_point = 4 * n_dtn + 1
            xy_circle_np, normal_circle_np = sample_circle(n_point, radius)
            xy_circle = torch.tensor(xy_circle_np, dtype=torch.float32, device=device)
            normal_circle = torch.tensor(normal_circle_np, dtype=torch.float32, device=device)
            circle_weight = 1.0 / circle_index
            dtn_contours.append((xy_circle, normal_circle, dtn_multiplier(n_point, radius, n_dtn)))
            dtn_truncation_orders.append(n_dtn)
            dtn_point_counts.append(n_point)
            dtn_circle_weights.append(circle_weight)
        print(f"DtN radii={DTN_RADII}")
        print(f"DtN N_m={tuple(dtn_truncation_orders)}, Q_m={tuple(dtn_point_counts)}")
        print(
            f"DtN circle weights={tuple(dtn_circle_weights)}; BC2: residual / {derivative_residual_scale:g}, real/imag MSE"
        )
    d_inc = torch.tensor(INCIDENT_DIR, dtype=torch.float32, device=device)
    phase = k * (xy_bc1 @ d_inc)
    u_bc1_real = (-torch.cos(phase)).detach()
    u_bc1_imag = (-torch.sin(phase)).detach()
    use_exact_outer = OUTER_BC_TYPE == "exact_dirichlet"
    if use_exact_outer:
        u_bc2_exact_np = exact_solution_from_xy(xy_bc2_np)
        u_bc2_real = torch.tensor(
            np.real(u_bc2_exact_np), dtype=torch.float32, device=device
        ).detach()
        u_bc2_imag = torch.tensor(
            np.imag(u_bc2_exact_np), dtype=torch.float32, device=device
        ).detach()
    else:
        u_bc2_real = None
        u_bc2_imag = None

    c_delta_init = 0.01 * torch.randn(Nc, input_dim, dtype=torch.float32, device=device)
    raw_beta_init = beta0 + 0.02 * torch.randn(Nc, dtype=torch.float32, device=device)
    w_init = 0.01 * torch.randn(Nc, dtype=torch.float32, device=device)
    b_init = torch.zeros(1, dtype=torch.float32, device=device)
    theta_r = (
        torch.cat([c_delta_init.reshape(-1), raw_beta_init, w_init, b_init])
        .clone()
        .detach()
        .requires_grad_(True)
    )
    theta_i = (
        torch.cat(
            [
                c_delta_init.clone().reshape(-1),
                raw_beta_init.clone(),
                w_init.clone(),
                b_init.clone(),
            ]
        )
        .clone()
        .detach()
        .requires_grad_(True)
    )
    if DRAW_INITIAL_POINT_LAYOUT or POINT_LAYOUT_ONLY:
        initial_centers_np = center_phys.copy()
        if USE_CENTER_TRAIN_SCALE:
            initial_centers_np += CENTER_TRAIN_SCALE * c_delta_init.detach().cpu().numpy()
        center_interior_initial = initial_centers_np[:Nc_interior]
        inner_start = Nc_interior
        inner_end = inner_start + Nc_inner
        center_inner_initial = initial_centers_np[inner_start:inner_end]
        center_outer_initial = initial_centers_np[inner_end:]
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.scatter(
            xy_pde_np[:, 0],
            xy_pde_np[:, 1],
            s=5,
            color="lightgray",
            alpha=0.45,
            label=f"PDE sampling points ({len(xy_pde_np)})",
            zorder=1,
        )
        ax.scatter(
            xy_bc1_np[:, 0],
            xy_bc1_np[:, 1],
            s=13,
            color="red",
            alpha=0.75,
            label=f"Inner-boundary points ({len(xy_bc1_np)})",
            zorder=2,
        )
        if OUTER_BC_TYPE == "multi_dtn":
            dtn_colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(dtn_contours)))
            for m, ((xy_circle, _, _), color) in enumerate(zip(dtn_contours, dtn_colors), start=1):
                xy_circle_np = xy_circle.detach().cpu().numpy()
                ax.scatter(
                    xy_circle_np[:, 0],
                    xy_circle_np[:, 1],
                    s=9,
                    color=color,
                    alpha=0.65,
                    label=f"DtN circle {m}: R={DTN_RADII[m - 1]:.2f}, Q={len(xy_circle_np)}",
                    zorder=2,
                )
        else:
            ax.scatter(
                xy_bc2_np[:, 0],
                xy_bc2_np[:, 1],
                s=9,
                color="royalblue",
                alpha=0.65,
                label=f"Outer-boundary points ({len(xy_bc2_np)})",
                zorder=2,
            )
        ax.scatter(
            center_interior_initial[:, 0],
            center_interior_initial[:, 1],
            s=23,
            marker="x",
            linewidths=0.9,
            color="black",
            label=f"Interior centers ({len(center_interior_initial)})",
            zorder=4,
        )
        ax.scatter(
            center_inner_initial[:, 0],
            center_inner_initial[:, 1],
            s=30,
            marker="^",
            facecolors="none",
            edgecolors="magenta",
            linewidths=1.0,
            label=f"Inner-boundary centers ({len(center_inner_initial)})",
            zorder=4,
        )
        ax.scatter(
            center_outer_initial[:, 0],
            center_outer_initial[:, 1],
            s=30,
            marker="s",
            facecolors="none",
            edgecolors="deepskyblue",
            linewidths=1.0,
            label=f"Outer-boundary centers ({len(center_outer_initial)})",
            zorder=4,
        )
        plot_limit = 1.06 * OUTER_R
        ax.set_xlim(-plot_limit, plot_limit)
        ax.set_ylim(-plot_limit, plot_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(f"Initial sampling points and RBF centers\nk={k_value:g}")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=INITIAL_LAYOUT_LEGEND_FONTSIZE,
            markerscale=1.4,
            labelspacing=0.7,
            borderpad=0.8,
            frameon=True,
        )
        fig.tight_layout()
        initial_layout_path = os.path.join(
            case_output_dir, f"initial_point_layout_k_{format_k_label(k_value)}.png"
        )
        fig.savefig(initial_layout_path, dpi=300, bbox_inches="tight")
        print(f"Initial point layout saved to: {initial_layout_path}")
        if SHOW_INITIAL_POINT_LAYOUT:
            _original_plt_show()
        plt.close(fig)
    if POINT_LAYOUT_ONLY:
        print(
            f"[Point-layout-only] k={k_value:g}: initial point layout saved; training and evaluation skipped."
        )
        return {
            "k": float(k_value),
            "point_layout_only": True,
            "initial_layout_path": initial_layout_path,
            "case_output_dir": case_output_dir,
        }

    def unpack(theta):
        """Decode centers, positive Gaussian widths, output weights, and bias."""
        c_delta = theta[0 : input_dim * Nc].reshape(Nc, input_dim)
        if USE_CENTER_TRAIN_SCALE:
            c = c_base + CENTER_TRAIN_SCALE * c_delta
        else:
            c = c_base
        raw_beta = theta[input_dim * Nc : (input_dim + 1) * Nc]
        beta = F.softplus(raw_beta) + 1e-06
        w = theta[(input_dim + 1) * Nc : (input_dim + 2) * Nc]
        b = theta[-1]
        return (c, beta, w, b)

    def forward(theta, xy):
        """Evaluate the Gaussian RBF expansion using the mean squared coordinate distance."""
        c, beta, w, b = unpack(theta)
        xc = xy @ c.T
        x2 = (xy * xy).sum(1, keepdim=True)
        c2 = (c * c).sum(1).unsqueeze(0)
        d2 = 0.5 * (x2 - 2.0 * xc + c2)
        phi = torch.exp(-beta.unsqueeze(0) * d2)
        w_phi = w.unsqueeze(0) * phi
        u = w_phi.sum(1) + b
        return u

    def forward_with_derivatives(theta, xy):
        c, beta, w, b = unpack(theta)
        xc = xy @ c.T
        x2 = (xy * xy).sum(1, keepdim=True)
        c2 = (c * c).sum(1).unsqueeze(0)
        d2 = 0.5 * (x2 - 2.0 * xc + c2)
        phi = torch.exp(-beta.unsqueeze(0) * d2)
        w_phi = w.unsqueeze(0) * phi
        u = w_phi.sum(1) + b
        dx = xy[:, 0:1] - c[:, 0].unsqueeze(0)
        dy = xy[:, 1:2] - c[:, 1].unsqueeze(0)
        ax = -beta.unsqueeze(0) * dx
        ay = -beta.unsqueeze(0) * dy
        ux = (w_phi * ax).sum(1)
        uy = (w_phi * ay).sum(1)
        lap = (w_phi * (ax * ax + ay * ay - 2.0 * beta.unsqueeze(0))).sum(1)
        return (u, ux, uy, lap)

    def residuals(theta_r, theta_i):
        """Concatenate the real and imaginary PDE and physical-boundary residuals."""
        ur, _, _, lap_r = forward_with_derivatives(theta_r, xy_pde)
        ui, _, _, lap_i = forward_with_derivatives(theta_i, xy_pde)
        r_pde_r = (lap_r + k**2 * ur) / pde_residual_scale
        r_pde_i = (lap_i + k**2 * ui) / pde_residual_scale
        ur1 = forward(theta_r, xy_bc1)
        ui1 = forward(theta_i, xy_bc1)
        r_bc1_r = ur1 - u_bc1_real
        r_bc1_i = ui1 - u_bc1_imag
        return torch.cat([r_pde_r, r_pde_i, r_bc1_r, r_bc1_i])

    def outer_boundary_loss(theta_r, theta_i):
        """Compute the weighted circle average of real/imaginary DtN residual MSEs."""
        if use_exact_outer:
            r_bc2_r = forward(theta_r, xy_bc2) - u_bc2_real
            r_bc2_i = forward(theta_i, xy_bc2) - u_bc2_imag
            return torch.mean(torch.cat([r_bc2_r, r_bc2_i]) ** 2)
        # Each circle is transformed separately; real and imaginary MSEs share 1/(2Q_m).
        circle_losses = []
        for xy_circle, normal_circle, multiplier in dtn_contours:
            ur, ur_x, ur_y, _ = forward_with_derivatives(theta_r, xy_circle)
            ui, ui_x, ui_y, _ = forward_with_derivatives(theta_i, xy_circle)
            nx, ny = (normal_circle[:, 0], normal_circle[:, 1])
            pressure = torch.complex(ur, ui)
            radial_derivative = torch.complex(ur_x * nx + ur_y * ny, ui_x * nx + ui_y * ny)
            transformed = torch.fft.ifft(multiplier * torch.fft.fft(pressure))
            residual = (radial_derivative - transformed) / derivative_residual_scale
            circle_losses.append(torch.mean(torch.cat([residual.real, residual.imag]) ** 2))
        losses = torch.stack(circle_losses)
        weights = losses.new_tensor(dtn_circle_weights)
        return torch.sum(weights * losses) / torch.sum(weights)

    def pde_residual_magnitude(
        theta_r, theta_i, xy_eval_np, batch_size=RAD_BATCH_SIZE, normalized_for_rad=False
    ):
        """Evaluate PDE residual magnitudes; RAD uses RMS(laplacian/k**2 + u)."""
        if isinstance(xy_eval_np, torch.Tensor):
            xy_np = xy_eval_np.detach().cpu().numpy().astype(np.float32)
        else:
            xy_np = np.asarray(xy_eval_np, dtype=np.float32)
        res_chunks = []
        with torch.no_grad():
            for start in range(0, len(xy_np), batch_size):
                end = min(start + batch_size, len(xy_np))
                xy_batch = torch.tensor(xy_np[start:end], dtype=torch.float32, device=device)
                ur, _, _, lap_r = forward_with_derivatives(theta_r, xy_batch)
                ui, _, _, lap_i = forward_with_derivatives(theta_i, xy_batch)
                if normalized_for_rad:
                    residual = torch.stack((lap_r / k**2 + ur, lap_i / k**2 + ui), dim=1)
                    res = torch.sqrt(torch.mean(residual * residual, dim=1))
                else:
                    r_real = lap_r + k**2 * ur
                    r_imag = lap_i + k**2 * ui
                    res = torch.sqrt(r_real**2 + r_imag**2)
                res_chunks.append(res.detach().cpu().numpy())
        return np.concatenate(res_chunks, axis=0)

    def rad_sobol_domain(n_point, seed):
        """Generate scrambled Sobol candidates in the fluid annulus."""
        if n_point == 0:
            return np.empty((0, 2), dtype=np.float32)
        engine = qmc.Sobol(d=2, scramble=True, seed=seed)
        accepted = []
        count = 0
        while count < n_point:
            power = int(math.ceil(math.log2(max(64, 2 * (n_point - count)))))
            unit = engine.random_base2(power)
            xy = -OUTER_R + 2.0 * OUTER_R * unit
            radius = np.linalg.norm(xy, axis=1)
            valid = (radius < OUTER_R) & (radius > INNER_R)
            xy = xy[valid]
            accepted.append(xy)
            count += len(xy)
            if count < n_point:
                seed += 1009
                engine = qmc.Sobol(d=2, scramble=True, seed=seed)
        return np.vstack(accepted)[:n_point].astype(np.float32)

    def rad_resample_mixed(theta_r, theta_i, m_new=None, m_pool=None, event_id=0):
        """Combine background points with residual-weighted samples without replacement."""
        if m_new is None:
            m_new = M_pde
        mix = float(np.clip(RAD_MIX_RATIO, 0.0, 1.0))
        m_rad = int(round(m_new * mix))
        m_uniform = int(m_new - m_rad)
        if m_uniform > 0:
            if RAD_REFRESH_UNIFORM_PART:
                xy_uniform = rad_sobol_domain(m_uniform, RAD_RANDOM_SEED + 30000 + event_id)
            elif m_uniform <= len(xy_pde_uniform_base_np):
                xy_uniform = xy_pde_uniform_base_np[:m_uniform].astype(np.float32)
            else:
                extra = sample_pde_domain(m_uniform - len(xy_pde_uniform_base_np))
                xy_uniform = np.vstack([xy_pde_uniform_base_np, extra]).astype(np.float32)
        else:
            xy_uniform = np.empty((0, 2), dtype=np.float32)
        if m_rad > 0:
            if m_pool is None:
                m_pool = max(RAD_POOL_MIN, int(RAD_POOL_FACTOR * m_rad))
            m_pool = max(int(m_pool), int(m_rad))
            xy_pool = rad_sobol_domain(m_pool, RAD_RANDOM_SEED + 40000 + event_id)
            res_pool = pde_residual_magnitude(
                theta_r, theta_i, xy_pool, batch_size=RAD_BATCH_SIZE, normalized_for_rad=True
            )
            res_mean = float(np.mean(res_pool)) + RAD_NORMALIZATION_EPS
            res_norm = res_pool / res_mean
            prob_raw = (res_norm + RAD_EPS) ** RAD_POWER + RAD_C
            if not np.all(np.isfinite(prob_raw)) or np.sum(prob_raw) <= 0:
                prob = np.ones_like(prob_raw) / len(prob_raw)
            else:
                prob = prob_raw / np.sum(prob_raw)
            rng = np.random.default_rng(RAD_RANDOM_SEED + 50000 + event_id)
            idx = rng.choice(len(xy_pool), size=m_rad, replace=False, p=prob)
            xy_rad = xy_pool[idx].astype(np.float32)
        else:
            xy_pool = np.empty((0, 2), dtype=np.float32)
            res_pool = np.empty((0,), dtype=np.float32)
            res_norm = np.empty((0,), dtype=np.float32)
            prob = np.empty((0,), dtype=np.float32)
            xy_rad = np.empty((0, 2), dtype=np.float32)
            m_pool = 0
        xy_new = np.vstack([xy_uniform, xy_rad]).astype(np.float32)
        info = {
            "xy_pool": xy_pool,
            "xy_uniform": xy_uniform,
            "xy_rad": xy_rad,
            "xy_selected": xy_new,
            "res_pool": res_pool,
            "res_norm": res_norm,
            "prob_pool": prob,
            "res_mean": float(np.mean(res_pool)) if len(res_pool) else 0.0,
            "res_max": float(np.max(res_pool)) if len(res_pool) else 0.0,
            "res_min": float(np.min(res_pool)) if len(res_pool) else 0.0,
            "prob_max": float(np.max(prob)) if len(prob) else 0.0,
            "prob_min": float(np.min(prob)) if len(prob) else 0.0,
            "m_pool": int(m_pool),
            "m_new": int(m_new),
            "m_uniform": int(m_uniform),
            "m_rad": int(m_rad),
            "event_id": int(event_id),
        }
        return (xy_new, info)

    def predict_on_numpy_points(xy_np, batch_size=PREDICTION_BATCH_SIZE):
        """Evaluate both scalar networks in batches without parameter-gradient tracking."""
        real_list = []
        imag_list = []
        with torch.no_grad():
            for start in range(0, len(xy_np), batch_size):
                batch = xy_np[start : start + batch_size]
                xy_t = torch.tensor(batch, dtype=torch.float32, device=device)
                ur = forward(theta_r, xy_t)
                ui = forward(theta_i, xy_t)
                real_list.append(ur.detach().cpu().numpy())
                imag_list.append(ui.detach().cpu().numpy())
        real = np.concatenate(real_list)
        imag = np.concatenate(imag_list)
        return real + 1j * imag

    def postprocess_domain_mask(xy):
        """Select evaluation points in the closed fluid annulus."""
        points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        radius = np.linalg.norm(points, axis=1)
        return (radius >= INNER_R) & (radius <= OUTER_R)

    if EVALUATION_BOUNDARY_POLICY != "inclusive":
        raise ValueError('EVALUATION_BOUNDARY_POLICY must be "inclusive".')
    x_vis = np.linspace(-OUTER_R, OUTER_R, EVAL_GRID_SIZE)
    y_vis = np.linspace(-OUTER_R, OUTER_R, EVAL_GRID_SIZE)
    ms_x, ms_y = np.meshgrid(x_vis, y_vis, indexing="xy")
    rr_vis = np.hypot(ms_x, ms_y)
    tt_vis = np.arctan2(ms_y, ms_x)
    xy_vis_all_np = np.column_stack((ms_x.ravel(), ms_y.ravel()))
    valid_flat = postprocess_domain_mask(xy_vis_all_np)
    domain_mask = valid_flat.reshape(ms_x.shape)
    xy_vis_valid_np = xy_vis_all_np[valid_flat]
    progression_snapshots = {}
    progression_snapshot_paths = []
    progression_iterations = tuple(TRAINING_PROGRESS_ITERATIONS)
    if (
        len(progression_iterations) != 4
        or any((type(step) is not int or step <= 0 for step in progression_iterations))
        or list(progression_iterations) != sorted(set(progression_iterations))
    ):
        raise ValueError(
            "TRAINING_PROGRESS_ITERATIONS must contain four strictly increasing positive integers."
        )

    def capture_progression_snapshot(iteration):
        """Save the current predicted field after the specified Adam update."""
        prediction = predict_on_numpy_points(xy_vis_valid_np)
        real_grid = np.full(ms_x.shape, np.nan, dtype=np.float64)
        imag_grid = np.full(ms_x.shape, np.nan, dtype=np.float64)
        real_grid.ravel()[valid_flat] = np.real(prediction)
        imag_grid.ravel()[valid_flat] = np.imag(prediction)
        snapshot = {"iteration": int(iteration), "real": real_grid, "imag": imag_grid}
        progression_snapshots[int(iteration)] = snapshot
        snapshot_dir = os.path.join(case_output_dir, "training_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, f"epoch_{iteration:06d}.npz")
        np.savez_compressed(
            snapshot_path,
            iteration=np.int64(iteration),
            k=np.float64(k),
            x=x_vis,
            y=y_vis,
            domain_mask=domain_mask,
            real=real_grid,
            imag=imag_grid,
        )
        progression_snapshot_paths.append(snapshot_path)
        print(f"[Training progression] Captured model after Adam update {iteration:,}")

    print(f"Problem: {CASE_NAME}")
    print(
        f"Training residual normalization: {NORMALIZE_TRAINING_RESIDUALS} | PDE divisor={pde_residual_scale:g}, DtN divisor={derivative_residual_scale:g}; Dirichlet unchanged; real/imag MSE"
    )
    print(
        f"PDE Mixed RAD: {USE_RAD} | start={RAD_START_ITER}, period={RAD_PERIOD}, mix={RAD_MIX_RATIO}, pool_factor={RAD_POOL_FACTOR}, power={RAD_POWER}, c={RAD_C}"
    )
    optimizer = torch.optim.Adam([theta_r, theta_i], lr=lr)
    print(f"Learning rate (constant) | lr={lr:g}")
    loss_list = []
    loss_pde_list = []
    loss_bc1_list = []
    loss_bc2_list = []
    rad_history = []
    rad_last_info = None
    training_resource_state = start_training_resource_monitor(device)
    for it in range(n_epochs):
        rad_iteration = it + 1
        if USE_RAD and rad_iteration >= RAD_START_ITER and (rad_iteration % RAD_PERIOD == 0):
            event_id = (rad_iteration - RAD_START_ITER) // RAD_PERIOD
            xy_new_np, rad_info = rad_resample_mixed(
                theta_r, theta_i, m_new=M_pde, event_id=event_id
            )
            xy_pde.copy_(torch.from_numpy(xy_new_np))
            rad_info["iteration"] = rad_iteration
            rad_last_info = rad_info
            rad_history.append(
                {
                    "iteration": rad_iteration,
                    "event_id": event_id,
                    "res_mean": rad_info["res_mean"],
                    "res_max": rad_info["res_max"],
                    "res_min": rad_info["res_min"],
                    "prob_max": rad_info["prob_max"],
                    "prob_min": rad_info["prob_min"],
                    "m_pool": rad_info["m_pool"],
                    "m_new": rad_info["m_new"],
                    "m_uniform": rad_info["m_uniform"],
                    "m_rad": rad_info["m_rad"],
                }
            )
            print(
                f"[PDE Mixed RAD] Iter {rad_iteration}: res_mean={rad_info['res_mean']:.3e}, res_max={rad_info['res_max']:.3e}, pool={rad_info['m_pool']}, selected={rad_info['m_new']} (uniform={rad_info['m_uniform']}, RAD={rad_info['m_rad']})"
            )
        # One backward pass over the full PDE and boundary loss per Adam update.
        optimizer.zero_grad(set_to_none=True)
        R = residuals(theta_r, theta_i)
        N_pde_current = len(xy_pde)
        N_bc1_current = len(xy_bc1)
        idx_pde_end = 2 * N_pde_current
        idx_bc1_end = idx_pde_end + 2 * N_bc1_current
        loss_pde = torch.mean(R[:idx_pde_end] ** 2)
        loss_bc1 = torch.mean(R[idx_pde_end:idx_bc1_end] ** 2)
        loss_bc2 = outer_boundary_loss(theta_r, theta_i)
        loss = W_PDE * loss_pde + W_BC1 * loss_bc1 + W_BC2 * loss_bc2
        loss.backward()
        optimizer.step()
        if it + 1 in progression_iterations:
            capture_progression_snapshot(it + 1)
        loss_list.append(loss.item())
        loss_pde_list.append(loss_pde.item())
        loss_bc1_list.append(loss_bc1.item())
        loss_bc2_list.append(loss_bc2.item())
        if it == 0 or (it + 1) % print_every == 0:
            print(
                f"Iter {it + 1}, Loss={loss.item():.6e}    | weighted loss   PDE: {(W_PDE * loss_pde).item():.3e}  |  bc1: {(W_BC1 * loss_bc1).item():.3e}  |  bc2({OUTER_BC_TYPE}): {(W_BC2 * loss_bc2).item():.3e}  |  N_pde={N_pde_current}, N_bc1={N_bc1_current}  |  lr={lr:g}"
            )
    training_resources = finish_training_resource_monitor(training_resource_state, len(loss_list))
    resource_path = os.path.join(case_output_dir, "training_resources.json")
    training_resources["training_resources_path"] = resource_path
    with open(resource_path, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "k": k,
                "use_rad": USE_RAD,
                **training_resources,
                "timing_scope": "Training loop, including RAD, logging and snapshot prediction/write; excludes setup and final plotting.",
                "memory_scope": "PyTorch CUDA allocator for this process and device, reset before the training loop.",
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    print(
        f"[Training resources] k={k:g}, time={training_resources['training_time_hms']}, peak CUDA allocated (MiB)={training_resources['peak_cuda_allocated_mib']}"
    )
    if USE_RAD and rad_last_info is None and RAD_DIAGNOSTIC_IF_NO_EVENT:
        print(
            "[PDE Mixed RAD] Training ended before the first RAD event. Creating one diagnostic sample for visualization only."
        )
        _, rad_last_info = rad_resample_mixed(theta_r, theta_i, m_new=M_pde)
        rad_last_info["iteration"] = "diagnostic-after-training"
    rad_history_path = os.path.join(case_output_dir, "rad_history.csv")
    rad_history_keys = [
        "iteration",
        "event_id",
        "m_new",
        "m_uniform",
        "m_rad",
        "m_pool",
        "res_mean",
        "res_max",
        "res_min",
        "prob_max",
        "prob_min",
    ]
    with open(rad_history_path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=rad_history_keys)
        writer.writeheader()
        writer.writerows(rad_history)
    print(f"[PDE Mixed RAD] Actual updates: {len(rad_history)}; history: {rad_history_path}")
    single_total_loss_path = None
    single_components_loss_path = None
    fig_loss, ax_loss = plt.subplots(figsize=(6.6, 4.8))
    ax_loss.plot(loss_list, linewidth=1.6, label="Total weighted loss")
    ax_loss.set_yscale("log")
    ax_loss.set_title(
        f"Adam loss ({CASE_NAME}, k={format_k_label(k_value)})", fontsize=PAPER_FONT_SIZE
    )
    ax_loss.set_xlabel("Iteration", fontsize=PAPER_FONT_SIZE)
    ax_loss.set_ylabel("Loss", fontsize=PAPER_FONT_SIZE)
    ax_loss.tick_params(axis="both", labelsize=PAPER_TICK_SIZE, length=3, width=0.8)
    ax_loss.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax_loss.legend(frameon=False, fontsize=PAPER_TICK_SIZE)
    fig_loss.tight_layout()
    if SAVE_FIGURES and SAVE_SINGLE_K_LOSS_FIGURES:
        single_total_loss_path = os.path.join(
            case_output_dir, f"single_k_total_loss_k_{format_k_label(k_value)}.png"
        )
        fig_loss.savefig(single_total_loss_path, dpi=PAPER_DPI, bbox_inches="tight")
        print(f"Single-k total loss figure saved to: {single_total_loss_path}")
    if PLOT_ORIGINAL_FIGURES:
        _original_plt_show()
    else:
        plt.close(fig_loss)
    fig_comp, ax_comp = plt.subplots(figsize=(6.6, 4.8))
    ax_comp.plot(loss_pde_list, linewidth=1.3, label="PDE")
    ax_comp.plot(loss_bc1_list, linewidth=1.3, label="BC1 inner")
    ax_comp.plot(
        loss_bc2_list,
        linewidth=1.3,
        label="BC2 multi-circle DtN" if not use_exact_outer else "BC2 exact Dirichlet",
    )
    ax_comp.set_yscale("log")
    ax_comp.set_title(
        f"Loss components ({CASE_NAME}, k={format_k_label(k_value)})", fontsize=PAPER_FONT_SIZE
    )
    ax_comp.set_xlabel("Iteration", fontsize=PAPER_FONT_SIZE)
    ax_comp.set_ylabel("Unweighted MSE", fontsize=PAPER_FONT_SIZE)
    ax_comp.tick_params(axis="both", labelsize=PAPER_TICK_SIZE, length=3, width=0.8)
    ax_comp.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax_comp.legend(frameon=False, fontsize=PAPER_TICK_SIZE)
    fig_comp.tight_layout()
    if SAVE_FIGURES and SAVE_SINGLE_K_LOSS_FIGURES:
        single_components_loss_path = os.path.join(
            case_output_dir, f"single_k_loss_components_k_{format_k_label(k_value)}.png"
        )
        fig_comp.savefig(single_components_loss_path, dpi=PAPER_DPI, bbox_inches="tight")
        print(f"Single-k loss components figure saved to: {single_components_loss_path}")
    if PLOT_ORIGINAL_FIGURES:
        _original_plt_show()
    else:
        plt.close(fig_comp)
    u_net_valid = predict_on_numpy_points(xy_vis_valid_np)
    u_real_net = np.full(ms_x.shape, np.nan, dtype=np.float64)
    u_imag_net = np.full(ms_x.shape, np.nan, dtype=np.float64)
    u_real_net.ravel()[valid_flat] = np.real(u_net_valid)
    u_imag_net.ravel()[valid_flat] = np.imag(u_net_valid)

    def get_hparam_text():
        text = f"Problem: {CASE_NAME}\nK: {k}\nOuter BC: {(OUTER_BC_TYPE if not use_exact_outer else 'exact_dirichlet')}\nInner BC: {'Dirichlet / sound-soft'}\nNc:{Nc} PDE:{Nc_interior} BC1:{Nc_inner} BC2:{Nc_outer}\nsampling:\n PDE:{M_pde}, BC1={M_bc1}, {('DtN Q_m=' + str(tuple(dtn_point_counts)) if not use_exact_outer else 'BC2=' + str(M_bc2))}\nAdaptive:\n PDE Mixed RAD={USE_RAD}, mix={RAD_MIX_RATIO}\nbeta0: {beta0:.6g}\nLoss weights:\n PDE={W_PDE}, BC1={W_BC1}, BC2={W_BC2}\n"
        return text

    def add_hparam_box(fig):
        fig.text(
            0.75,
            0.86,
            get_hparam_text(),
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
        )

    def plot_surface(x, y, z, title):
        fig = plt.figure(figsize=(10, 6.5))
        fig.subplots_adjust(right=0.78)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, cmap=cm.coolwarm)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        add_hparam_box(fig)
        plt.show()

    def draw_boundaries(ax):
        """Draw the obstacle and outer computational boundary."""
        outer = plt.Circle(
            (0, 0), OUTER_R, color="black", fill=False, linewidth=1.0, linestyle="--"
        )
        inner = plt.Circle((0, 0), INNER_R, color="black", fill=False, linewidth=1.2)
        ax.add_patch(outer)
        ax.add_patch(inner)

    def plot_rad_sampling_distribution(rad_info):
        if rad_info is None:
            return
        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        if len(rad_info.get("xy_pool", [])) > 0:
            ax.scatter(
                rad_info["xy_pool"][:, 0],
                rad_info["xy_pool"][:, 1],
                c="gray",
                s=4,
                alpha=0.2,
                label="candidate pool",
            )
        if len(rad_info.get("xy_uniform", [])) > 0:
            ax.scatter(
                rad_info["xy_uniform"][:, 0],
                rad_info["xy_uniform"][:, 1],
                c="blue",
                s=7,
                alpha=0.35,
                label="uniform background",
            )
        if len(rad_info.get("xy_rad", [])) > 0:
            ax.scatter(
                rad_info["xy_rad"][:, 0],
                rad_info["xy_rad"][:, 1],
                c="red",
                s=16,
                alpha=0.85,
                label="RAD selected",
            )
        draw_boundaries(ax)
        ax.set_aspect("equal")
        ax.set_title(f"PDE Mixed RAD Sampling ({CASE_NAME}, iter={rad_info.get('iteration')})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    def plot_rad_diagnostic(rad_info):
        if rad_info is None:
            print("[Plot] No PDE Mixed RAD event information available. Diagnostic plot skipped.")
            return
        if len(rad_info.get("xy_pool", [])) == 0:
            print(
                "[Plot] PDE Mixed RAD has no candidate pool information. Diagnostic plot skipped."
            )
            return
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8))
        sc = axes[0].scatter(
            rad_info["xy_pool"][:, 0],
            rad_info["xy_pool"][:, 1],
            c=rad_info["res_norm"],
            cmap="jet",
            s=8,
            alpha=0.95,
        )
        draw_boundaries(axes[0])
        axes[0].set_aspect("equal")
        axes[0].set_title("Residual heatmap")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        cbar = fig.colorbar(sc, ax=axes[0], shrink=0.85)
        cbar.set_label("residual / mean(residual)")
        axes[1].scatter(
            rad_info["xy_pool"][:, 0],
            rad_info["xy_pool"][:, 1],
            c="gray",
            s=4,
            alpha=0.16,
            label="candidate pool",
        )
        if len(rad_info.get("xy_uniform", [])) > 0:
            axes[1].scatter(
                rad_info["xy_uniform"][:, 0],
                rad_info["xy_uniform"][:, 1],
                c="blue",
                s=7,
                alpha=0.28,
                label="uniform",
            )
        if len(rad_info.get("xy_rad", [])) > 0:
            axes[1].scatter(
                rad_info["xy_rad"][:, 0],
                rad_info["xy_rad"][:, 1],
                c="red",
                s=18,
                alpha=0.9,
                label="RAD",
            )
        draw_boundaries(axes[1])
        axes[1].set_aspect("equal")
        axes[1].set_title("Mixed RAD sampling")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].legend(loc="upper right")
        fig.suptitle(f"Mixed RAD Diagnostic ({CASE_NAME}, iter={rad_info.get('iteration')})")
        plt.tight_layout()
        plt.show()

    def plot_rad_residual_distribution(rad_info):
        if rad_info is None or len(rad_info.get("xy_pool", [])) == 0:
            return
        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        sc = ax.scatter(
            rad_info["xy_pool"][:, 0],
            rad_info["xy_pool"][:, 1],
            c=rad_info["res_norm"],
            cmap="jet",
            s=8,
            alpha=0.9,
        )
        draw_boundaries(ax)
        ax.set_aspect("equal")
        ax.set_title(
            f"PDE Mixed RAD Candidate Pool Residual ({CASE_NAME}, iter={rad_info.get('iteration')})"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
        cbar.set_label("residual / mean(residual)")
        plt.tight_layout()
        plt.show()

    def plot_rad_probability_distribution(rad_info):
        if rad_info is None or len(rad_info.get("xy_pool", [])) == 0:
            return
        prob_scaled = rad_info["prob_pool"] / (np.mean(rad_info["prob_pool"]) + 1e-12)
        fig, ax = plt.subplots(figsize=(6.8, 6.2))
        sc = ax.scatter(
            rad_info["xy_pool"][:, 0],
            rad_info["xy_pool"][:, 1],
            c=prob_scaled,
            cmap="viridis",
            s=8,
            alpha=0.9,
        )
        draw_boundaries(ax)
        ax.set_aspect("equal")
        ax.set_title(
            f"PDE Mixed RAD Sampling Probability ({CASE_NAME}, iter={rad_info.get('iteration')})"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        cbar = fig.colorbar(sc, ax=ax, shrink=0.9)
        cbar.set_label("probability / mean(probability)")
        plt.tight_layout()
        plt.show()

    def plot_rad_history(rad_history):
        if len(rad_history) == 0:
            return
        its = [h["iteration"] for h in rad_history]
        mean_vals = [h["res_mean"] for h in rad_history]
        max_vals = [h["res_max"] for h in rad_history]
        plt.figure(figsize=(7, 5))
        plt.plot(its, mean_vals, marker="o", label="mean PDE residual in pool")
        plt.plot(its, max_vals, marker="s", label="max PDE residual in pool")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("PDE residual magnitude")
        plt.title("PDE Mixed RAD Residual Statistics")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_contour_2d(x, y, z, title, cmap=FIELD_CMAP, levels=FIELD_LEVELS, show_boundary=True):
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        z_arr = np.asarray(z, dtype=float).copy()
        finite_mask = np.isfinite(z_arr)
        z_min = np.nanmin(z_arr) if np.any(finite_mask) else 0.0
        if z_min < 0:
            z_vmax = _nice_field_vmax(z_arr, percentile=FIELD_PERCENTILE)
            vmin, vmax = (-z_vmax, z_vmax)
        else:
            z_vmax = _nice_error_vmax(z_arr, percentile=ERROR_PERCENTILE)
            vmin, vmax = (0.0, z_vmax)
        z_arr[finite_mask] = np.clip(z_arr[finite_mask], vmin, vmax)
        z_plot = np.ma.masked_invalid(z_arr)
        level_values = np.linspace(vmin, vmax, int(levels))
        cf = ax.contourf(
            x, y, z_plot, levels=level_values, cmap=cmap, vmin=vmin, vmax=vmax, extend="neither"
        )
        if DRAW_CONTOUR_LINES:
            try:
                ax.contour(
                    x,
                    y,
                    z_plot,
                    levels=CONTOUR_LINE_LEVELS,
                    colors="k",
                    linewidths=0.22,
                    alpha=0.28,
                )
            except Exception:
                pass
        if show_boundary:
            draw_boundaries(ax)
        _style_paper_axis(ax)
        ax.set_title(title, fontsize=PAPER_FONT_SIZE)
        ax.set_xlim(np.nanmin(x), np.nanmax(x))
        ax.set_ylim(np.nanmin(y), np.nanmax(y))
        cbar = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.025)
        cbar.ax.tick_params(labelsize=PAPER_TICK_SIZE, length=2.5, width=0.7)
        cbar.set_label("Value", fontsize=PAPER_FONT_SIZE)
        cbar.outline.set_linewidth(0.7)
        plt.tight_layout()
        plt.show()

    full_aligned_metrics = aligned_error_metrics([], [], prefix="full_")
    u_exact = np.full(ms_x.shape, complex(np.nan, np.nan), dtype=np.complex128)
    u_exact.ravel()[valid_flat] = postprocess_analytical_reference(xy_vis_valid_np, k)
    full_aligned_metrics = aligned_error_metrics(
        u_net_valid, u_exact.ravel()[valid_flat], prefix="full_"
    )
    u_real_exa = np.real(u_exact)
    u_imag_exa = np.imag(u_exact)
    plot_surface(ms_x, ms_y, u_real_exa, f"Exact Solution ({CASE_NAME}, Real Part)")
    plot_surface(ms_x, ms_y, u_imag_exa, f"Exact Solution ({CASE_NAME}, Imag Part)")
    u_real_error = u_real_net - u_real_exa
    u_imag_error = u_imag_net - u_imag_exa
    plot_surface(ms_x, ms_y, u_real_error, f"Error ({CASE_NAME}, Real)")
    plot_surface(ms_x, ms_y, u_imag_error, f"Error ({CASE_NAME}, Imag)")
    valid_error = np.isfinite(u_real_error) & np.isfinite(u_imag_error) & domain_mask
    u_pred_valid = u_real_net[valid_error] + 1j * u_imag_net[valid_error]
    u_ref_valid = u_real_exa[valid_error] + 1j * u_imag_exa[valid_error]
    metrics = compute_complex_metrics_from_arrays(u_pred_valid, u_ref_valid)
    print("Max error real:", metrics["max_abs_error_real"])
    print("Max error imag:", metrics["max_abs_error_imag"])
    print("L2 error real:", metrics["rel_l2_real"])
    print("L2 error imag:", metrics["rel_l2_imag"])
    print("Mean abs error real:", metrics["mean_abs_error_real"])
    print("Mean abs error imag:", metrics["mean_abs_error_imag"])
    observation_xy = observation_circle_points()
    observation_domain_valid = postprocess_domain_mask(observation_xy)
    observation_prediction = np.full(OBSERVATION_N, complex(np.nan, np.nan), dtype=np.complex128)
    if np.any(observation_domain_valid):
        observation_prediction[observation_domain_valid] = predict_on_numpy_points(
            observation_xy[observation_domain_valid]
        )
    observation_reference = np.full(OBSERVATION_N, complex(np.nan, np.nan), dtype=np.complex128)
    observation_reference_kind = "Analytical"
    if np.any(observation_domain_valid):
        observation_reference[observation_domain_valid] = postprocess_analytical_reference(
            observation_xy[observation_domain_valid], k
        )
    observation_finite = observation_domain_valid & np.isfinite(observation_reference.real)
    observation_finite &= np.isfinite(observation_reference.imag)
    observation_finite &= np.isfinite(observation_prediction.real) & np.isfinite(
        observation_prediction.imag
    )
    observation_valid_count = int(np.count_nonzero(observation_finite))
    observation_metrics = aligned_error_metrics(
        observation_prediction, observation_reference, prefix="r2_150_"
    )
    if observation_valid_count == OBSERVATION_N:
        observation_status = "complete"
    elif observation_reference_kind == "none":
        observation_status = "prediction_only_no_reference"
    else:
        observation_status = (
            "partial_reference" if observation_valid_count else "no_valid_reference"
        )
    observation_tag = f"r{format_k_label(OBSERVATION_RADIUS)}_{OBSERVATION_N}"
    observation_circle_path = os.path.join(
        case_output_dir, f"observation_circle_{observation_tag}.csv"
    )
    save_observation_circle_csv(
        observation_circle_path,
        observation_xy,
        observation_prediction,
        observation_reference,
        observation_domain_valid,
    )
    observation_summary = {
        "k": k,
        "observation_radius": OBSERVATION_RADIUS,
        "observation_points": OBSERVATION_N,
        "observation_offset_index": OBSERVATION_OFFSET_INDEX,
        "r2_150_valid_points": observation_valid_count,
        "observation_domain_valid_points": int(np.count_nonzero(observation_domain_valid)),
        "observation_reference_kind": observation_reference_kind,
        "observation_status": observation_status,
        **observation_metrics,
    }
    observation_summary_path = os.path.join(case_output_dir, "observation_circle_metrics.csv")
    with open(observation_summary_path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(observation_summary))
        writer.writeheader()
        writer.writerow(observation_summary)
    print(
        f"[Observation circle] R={OBSERVATION_RADIUS:g}, N={OBSERVATION_N}, valid reference={observation_valid_count}/{OBSERVATION_N}, status={observation_status}"
    )
    for name, value in observation_metrics.items():
        print(f"  {name}: {value:.6e}")
    if observation_valid_count != OBSERVATION_N:
        print(
            "[Observation circle] Metrics use only valid reference pairs; check coverage before a 150-point comparison."
        )
    print(f"Observation predictions saved to: {observation_circle_path}")
    plot_contour_2d(ms_x, ms_y, u_real_net, f"2D Contour - PIRBN Solution ({CASE_NAME}, Real Part)")
    plot_contour_2d(ms_x, ms_y, u_imag_net, f"2D Contour - PIRBN Solution ({CASE_NAME}, Imag Part)")
    plot_contour_2d(ms_x, ms_y, u_real_exa, f"2D Contour - Exact Solution ({CASE_NAME}, Real Part)")
    plot_contour_2d(ms_x, ms_y, u_imag_exa, f"2D Contour - Exact Solution ({CASE_NAME}, Imag Part)")
    plot_contour_2d(ms_x, ms_y, u_real_error, f"2D Contour - Error ({CASE_NAME}, Real)")
    plot_contour_2d(ms_x, ms_y, u_imag_error, f"2D Contour - Error ({CASE_NAME}, Imag)")
    if USE_RAD:
        plot_rad_sampling_distribution(rad_last_info)
        plot_rad_diagnostic(rad_last_info)
        plot_rad_residual_distribution(rad_last_info)
        plot_rad_probability_distribution(rad_last_info)
        plot_rad_history(rad_history)
    final_res_valid = pde_residual_magnitude(
        theta_r, theta_i, xy_vis_valid_np, batch_size=RAD_BATCH_SIZE
    )
    final_res_mag = np.full(ms_x.shape, np.nan, dtype=np.float64)
    final_res_mag.ravel()[valid_flat] = final_res_valid
    plot_contour_2d(
        ms_x,
        ms_y,
        final_res_mag,
        f"Final Raw PDE Residual Magnitude ({CASE_NAME})",
        cmap="jet",
        levels=60,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int(elapsed_time % 3600 // 60)
    seconds = elapsed_time % 60
    print("\n" + "=" * 50)
    print(f"Total runtime: {elapsed_time:.2f} seconds")
    if hours > 0:
        print(f"Elapsed: {hours} h {minutes} min {seconds:.2f} s")
    elif minutes > 0:
        print(f"Elapsed: {minutes} min {seconds:.2f} s")
    else:
        print(f"Elapsed: {seconds:.2f} s")
    print("=" * 50)
    plt.show = _original_plt_show
    six_panel_path = None
    six_panel_paths = {}
    if MAKE_SIX_PANEL:
        try:
            reference_kind = "Analytical"
            ref_real_grid = u_real_exa
            ref_imag_grid = u_imag_exa
            fig_specs = [
                ("l2_coolwarm", "coolwarm", "six_panel_L2_coolwarm"),
                ("l2_reference", REFERENCE_ERROR_CMAP, "six_panel_L2_reference_cmap"),
            ]
            for key, cmap_name, filename_prefix in fig_specs:
                path = os.path.join(
                    case_output_dir, f"{filename_prefix}_k_{format_k_label(k_value)}.png"
                )
                plot_six_panel_comparison(
                    ms_x=ms_x,
                    ms_y=ms_y,
                    u_real_pred=u_real_net,
                    u_imag_pred=u_imag_net,
                    u_real_ref=ref_real_grid,
                    u_imag_ref=ref_imag_grid,
                    k_value=k_value,
                    reference_kind=reference_kind,
                    draw_boundary_fn=draw_boundaries,
                    save_path=path,
                    show_figure=SHOW_FIGURES,
                    error_cmap=cmap_name,
                )
                six_panel_paths[key] = path
                print(f"Six-panel comparison figure saved to: {path}")
            six_panel_path = six_panel_paths.get("l2_coolwarm")
        except Exception as e:
            print(f"[Six-panel] Failed to create six-panel figure for k={k_value}: {e}")
    progression_path = None
    captured_iterations = sorted(progression_snapshots)
    missing_iterations = [
        value for value in progression_iterations if value not in progression_snapshots
    ]
    progression_metadata = {
        "method": "pirbn",
        "use_rad": USE_RAD,
        "k": k,
        "requested_iterations": list(progression_iterations),
        "captured_iterations": captured_iterations,
        "missing_iterations": missing_iterations,
        "capture_timing": "after Adam update",
        "grid_size": EVAL_GRID_SIZE,
        "grid_type": "cartesian",
        "snapshot_paths": progression_snapshot_paths,
        "status": "complete" if not missing_iterations else "incomplete",
    }
    if not missing_iterations:
        progress_ref_real, progress_ref_imag = (u_real_exa, u_imag_exa)
        progress_reference_kind = "analytical_reference"
        if SAVE_FIGURES:
            progression_path = os.path.join(
                case_output_dir, f"training_progression_8panel_k_{format_k_label(k_value)}.png"
            )
        progress_figure, progress_plot_metadata = plot_training_progression(
            ms_x,
            ms_y,
            [progression_snapshots[value] for value in progression_iterations],
            draw_boundary_fn=draw_boundaries,
            reference_real=progress_ref_real,
            reference_imag=progress_ref_imag,
            reference_kind=progress_reference_kind,
            save_path=progression_path,
            show_figure=SHOW_FIGURES,
        )
        plt.close(progress_figure)
        progression_metadata.update(progress_plot_metadata)
        if progression_path:
            print(f"Training progression figure saved to: {progression_path}")
        print(
            f"[Training progression] Common scale: {progress_plot_metadata['color_scale_source']}; [{progress_plot_metadata['color_min']:.6g}, {progress_plot_metadata['color_max']:.6g}]"
        )
    else:
        print(
            f"[Training progression] Missing actual iterations {missing_iterations}; eight-panel figure skipped. Captured fields remain available."
        )
    progression_metadata_path = os.path.join(case_output_dir, "training_progression_metadata.json")
    with open(progression_metadata_path, "w", encoding="utf-8") as stream:
        json.dump(progression_metadata, stream, ensure_ascii=False, indent=2)
    result = {
        "k": float(k_value),
        **training_resources,
        "training_progression_path": progression_path,
        "training_progression_metadata_path": progression_metadata_path,
        "training_progression_captured_iterations": "|".join(map(str, captured_iterations)),
        "outer_bc_type": OUTER_BC_TYPE,
        "dtn_radii": "|".join((f"{r:g}" for r in DTN_RADII)) if not use_exact_outer else "",
        "dtn_truncation_orders": "|".join((str(n) for n in dtn_truncation_orders)),
        "dtn_point_counts": "|".join((str(n) for n in dtn_point_counts)),
        "dtn_circle_weights": "|".join((f"{w:g}" for w in dtn_circle_weights)),
        "bc2_loss_definition": (
            (
                "weighted circle mean of real/imag MSE of (du/dr - T_R u)/k"
                if NORMALIZE_TRAINING_RESIDUALS
                else "weighted circle mean of raw real/imag MSE"
            )
            if not use_exact_outer
            else "exact Dirichlet MSE"
        ),
        "normalize_training_residuals": NORMALIZE_TRAINING_RESIDUALS,
        "pde_residual_divisor": pde_residual_scale,
        "dtn_residual_divisor": derivative_residual_scale,
                "n_epochs": int(n_epochs),
        "initial_learning_rate": float(lr),
        "final_learning_rate": float(lr),
        "lr_schedule_name": "constant",
        "loss_list": list(loss_list),
        "final_loss": float(loss_list[-1]) if len(loss_list) else np.nan,
        "six_panel_path": six_panel_path,
        "six_panel_paths": six_panel_paths,
        "single_total_loss_path": single_total_loss_path,
        "single_components_loss_path": single_components_loss_path,
        "metrics": metrics if "metrics" in locals() else None,
        "case_output_dir": case_output_dir,
        "evaluation_grid_size": EVAL_GRID_SIZE,
        "evaluation_grid_type": "cartesian",
        "evaluation_boundary_policy": EVALUATION_BOUNDARY_POLICY,
        "full_comparison_point_set": "cartesian_analytical_grid",
        "full_comparison_point_count": int(len(xy_vis_valid_np)),
        "full_comparison_interpolation": "not_applicable",
        "six_panel_plot_type": "contour_on_cartesian_grid",
        "exact_terms": EXACT_TERMS,
        "observation_radius": OBSERVATION_RADIUS,
        "observation_points": OBSERVATION_N,
        "observation_offset_index": OBSERVATION_OFFSET_INDEX,
        "observation_reference_kind": observation_reference_kind,
        "observation_status": observation_status,
        "r2_150_valid_points": observation_valid_count,
        "observation_circle_path": observation_circle_path,
        "observation_summary_path": observation_summary_path,
        "full_aligned_metrics": full_aligned_metrics,
        "observation_metrics": observation_metrics,
        "use_rad": USE_RAD,
        "rad_start_iter": RAD_START_ITER,
        "rad_period": RAD_PERIOD,
        "rad_mix_ratio": RAD_MIX_RATIO,
        "rad_refresh_uniform_part": RAD_REFRESH_UNIFORM_PART,
        "rad_pool_rule": (
            "max(4000,4*N_rad)"
            if RAD_POOL_MIN == 4000 and RAD_POOL_FACTOR == 4
            else f"max({RAD_POOL_MIN},{RAD_POOL_FACTOR}*N_rad)"
        ),
        "rad_pool_min": RAD_POOL_MIN,
        "rad_pool_factor": RAD_POOL_FACTOR,
        "rad_power": RAD_POWER,
        "rad_c": RAD_C,
        "rad_probability_eps": RAD_EPS,
        "rad_normalization_eps": RAD_NORMALIZATION_EPS,
        "rad_indicator": "RMS of laplacian/k^2 + value",
        "rad_sampling": "scrambled Sobol; event-seeded residual sampling without replacement",
        "rad_random_seed": RAD_RANDOM_SEED,
        "rad_update_count": len(rad_history),
        "rad_update_iterations": "|".join((str(item["iteration"]) for item in rad_history)),
        "rad_history_path": rad_history_path,
    }
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _nice_error_vmax(err, percentile=99.5):
    """Return the requested percentile of finite absolute errors."""
    err = np.asarray(err)
    valid = err[np.isfinite(err)]
    if valid.size == 0:
        return 1.0
    vmax = np.percentile(valid, percentile)
    return float(vmax) if vmax > 0 else float(np.max(valid) if np.max(valid) > 0 else 1.0)


def _nice_field_vmax(*arrays, percentile=97.0, default=1.0):
    """Return a symmetric field color limit from finite values."""
    vals = []
    for arr in arrays:
        a = np.asarray(arr)
        vals.append(np.abs(a[np.isfinite(a)]))
    vals = [v for v in vals if v.size > 0]
    if not vals:
        return default
    all_vals = np.concatenate(vals)
    vmax = np.percentile(all_vals, percentile)
    if vmax <= 0:
        vmax = np.max(all_vals) if np.max(all_vals) > 0 else default
    return float(vmax)


def _style_paper_axis(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("y", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=PAPER_TICK_SIZE, length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _add_panel_label_below(ax, label):
    ax.text(
        0.5, -0.17, label, transform=ax.transAxes, ha="center", va="top", fontsize=PAPER_FONT_SIZE
    )


def _plot_paper_field(ax, x, y, z, levels, cmap, vmin, vmax, draw_boundary_fn, is_error=False):
    """Plot a field with display-only saturation at the specified color limits."""
    z_arr = np.asarray(z, dtype=float).copy()
    finite_mask = np.isfinite(z_arr)
    z_arr[finite_mask] = np.clip(z_arr[finite_mask], vmin, vmax)
    z_plot = np.ma.masked_invalid(z_arr)
    if np.isscalar(levels):
        level_values = np.linspace(vmin, vmax, int(levels))
    else:
        level_values = levels
    cf = ax.contourf(
        x, y, z_plot, levels=level_values, cmap=cmap, vmin=vmin, vmax=vmax, extend="neither"
    )
    if DRAW_CONTOUR_LINES:
        try:
            if is_error:
                line_levels = np.linspace(vmin, vmax, max(6, CONTOUR_LINE_LEVELS // 2))
            else:
                line_levels = np.linspace(vmin, vmax, CONTOUR_LINE_LEVELS)
            ax.contour(x, y, z_plot, levels=line_levels, colors="k", linewidths=0.22, alpha=0.28)
        except Exception:
            pass
    draw_boundary_fn(ax)
    _style_paper_axis(ax)
    return cf


def _set_reference_field_colorbar_ticks(cbar, vmax):
    ticks = np.linspace(-float(vmax), float(vmax), 9)
    ticks[len(ticks) // 2] = 0.0
    cbar.locator = mticker.FixedLocator(ticks)
    cbar.formatter = mticker.FuncFormatter(lambda value, _pos: f"{value:.6g}")
    cbar.update_ticks()


def plot_training_progression(
    ms_x,
    ms_y,
    snapshots,
    draw_boundary_fn,
    reference_real=None,
    reference_imag=None,
    reference_kind=None,
    save_path=None,
    show_figure=False,
):
    """Plot four stored fields in a 2-by-4 real/imaginary layout with one color scale."""
    if len(snapshots) != 4:
        raise ValueError("The training progression plot requires four captured snapshots.")
    epochs = [int(item["iteration"]) for item in snapshots]
    if epochs != sorted(set(epochs)):
        raise ValueError("Snapshot iterations must be strictly increasing.")
    for item in snapshots:
        if np.shape(item["real"]) != np.shape(ms_x) or np.shape(item["imag"]) != np.shape(ms_x):
            raise ValueError("Snapshot fields and evaluation grids must have matching shapes.")
    reference_values = []
    for value in (reference_real, reference_imag):
        if value is not None:
            array = np.asarray(value)
            reference_values.append(np.abs(array[np.isfinite(array)]))
    reference_values = [value for value in reference_values if value.size]
    if reference_values:
        vmax = max((float(value.max()) for value in reference_values))
        color_scale_source = True
        if vmax == 0:
            vmax = 1.0
            color_scale_source += ":zero_field_display_range"
    else:
        valid_values = []
        for item in snapshots:
            for component in ("real", "imag"):
                array = np.asarray(item[component])
                valid_values.append(np.abs(array[np.isfinite(array)]))
        valid_values = [value for value in valid_values if value.size]
        if not valid_values:
            raise ValueError("The training snapshots contain no finite values to plot.")
        vmax = max((float(value.max()) for value in valid_values))
        color_scale_source = "all_training_snapshots"
        if vmax == 0:
            vmax = 1.0
            color_scale_source += ":zero_field_display_range"
    fig, axes = plt.subplots(2, 4, figsize=(18.4, 8.6))
    fig.subplots_adjust(left=0.045, right=0.91, bottom=0.075, top=0.95, wspace=0.28, hspace=0.28)
    levels = np.linspace(-vmax, vmax, FIELD_LEVELS)
    clipping = []
    for column, snapshot in enumerate(snapshots):
        for row, component, label in ((0, "real", "real"), (1, "imag", "imaginary")):
            ax = axes[row, column]
            values = np.asarray(snapshot[component])
            finite_values = values[np.isfinite(values)]
            cf = _plot_paper_field(
                ax,
                ms_x,
                ms_y,
                values,
                levels=levels,
                cmap=FIELD_CMAP,
                vmin=-vmax,
                vmax=vmax,
                draw_boundary_fn=draw_boundary_fn,
            )
            ax.set_xlim(float(np.min(ms_x)), float(np.max(ms_x)))
            ax.set_ylim(float(np.min(ms_y)), float(np.max(ms_y)))
            ax.set_title(
                f"Epoch {snapshot['iteration']:,} — {label}", fontsize=PAPER_FONT_SIZE, pad=7
            )
            clipping.append(
                {
                    "iteration": int(snapshot["iteration"]),
                    "component": component,
                    "finite_points": int(finite_values.size),
                    "below_display_min": int(np.count_nonzero(finite_values < -vmax)),
                    "above_display_max": int(np.count_nonzero(finite_values > vmax)),
                }
            )
    color_axis = fig.add_axes([0.931, 0.15, 0.012, 0.72])
    colorbar = fig.colorbar(
        cf,
        cax=color_axis,
        orientation="vertical",
        ticks=np.linspace(-vmax, vmax, 9),
        extend="neither",
    )
    colorbar.ax.tick_params(labelsize=PAPER_TICK_SIZE, length=2.5, width=0.7)
    colorbar.outline.set_linewidth(0.7)
    colorbar.set_label("Field value", fontsize=PAPER_FONT_SIZE, labelpad=8)
    if save_path is not None and SAVE_FIGURES:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=PAPER_DPI, bbox_inches="tight")
    if show_figure:
        plt.show()
    metadata = {
        "iterations": epochs,
        "layout": "2x4; real first row, imaginary second row",
        "color_scale_source": color_scale_source,
        "color_min": -vmax,
        "color_max": vmax,
        "shared_colorbar": True,
        "colorbar_extend": "neither",
        "has_suptitle": False,
        "display_saturation_counts": clipping,
        "raw_snapshot_values_clipped": False,
    }
    return (fig, metadata)


def plot_six_panel_comparison(
    ms_x,
    ms_y,
    u_real_pred,
    u_imag_pred,
    u_real_ref,
    u_imag_ref,
    k_value,
    reference_kind,
    draw_boundary_fn,
    save_path=None,
    show_figure=False,
    error_cmap="coolwarm",
):
    """Plot predicted fields, reference fields, and componentwise absolute errors."""
    real_diff = u_real_pred - u_real_ref
    imag_diff = u_imag_pred - u_imag_ref
    real_err = np.abs(real_diff)
    imag_err = np.abs(imag_diff)
    real_error_label = "(c) Pointwise absolute error, real part"
    imag_error_label = "(f) Pointwise absolute error, imaginary part"
    field_vmax = _nice_field_vmax(u_real_ref, u_imag_ref, percentile=100.0)
    real_err_vmax = _nice_error_vmax(real_err, percentile=ERROR_PERCENTILE)
    imag_err_vmax = _nice_error_vmax(imag_err, percentile=ERROR_PERCENTILE)
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.2), constrained_layout=False)
    model_label = "PIRBN-R" if USE_RAD else "PIRBN"
    panels = [
        (
            axes[0, 0],
            u_real_pred,
            f"(a) {model_label} real part",
            FIELD_CMAP,
            -field_vmax,
            field_vmax,
            False,
        ),
        (
            axes[0, 1],
            u_real_ref,
            f"(b) {reference_kind} real part",
            FIELD_CMAP,
            -field_vmax,
            field_vmax,
            False,
        ),
        (axes[0, 2], real_err, real_error_label, error_cmap, 0.0, real_err_vmax, True),
        (
            axes[1, 0],
            u_imag_pred,
            f"(d) {model_label} imaginary part",
            FIELD_CMAP,
            -field_vmax,
            field_vmax,
            False,
        ),
        (
            axes[1, 1],
            u_imag_ref,
            f"(e) {reference_kind} imaginary part",
            FIELD_CMAP,
            -field_vmax,
            field_vmax,
            False,
        ),
        (axes[1, 2], imag_err, imag_error_label, error_cmap, 0.0, imag_err_vmax, True),
    ]
    for ax, z, label, cmap, vmin, vmax, is_error in panels:
        cf = _plot_paper_field(
            ax=ax,
            x=ms_x,
            y=ms_y,
            z=z,
            levels=ERROR_LEVELS if is_error else FIELD_LEVELS,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            draw_boundary_fn=draw_boundary_fn,
            is_error=is_error,
        )
        _add_panel_label_below(ax, label)
        cbar = fig.colorbar(cf, ax=ax, shrink=0.86, pad=0.025)
        if not is_error:
            _set_reference_field_colorbar_ticks(cbar, vmax)
        cbar.ax.tick_params(labelsize=PAPER_TICK_SIZE, length=2.5, width=0.7)
        cbar.outline.set_linewidth(0.7)
    fig.subplots_adjust(left=0.055, right=0.965, top=0.965, bottom=0.095, wspace=0.22, hspace=0.28)
    if save_path is not None and SAVE_FIGURES:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=PAPER_DPI, bbox_inches="tight")
    if show_figure:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def plot_batch_total_loss(batch_results, output_dir):
    """Plot training loss histories for the selected wavenumbers."""
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for result in batch_results:
        losses = result.get("loss_list", [])
        if len(losses) == 0:
            continue
        ax.plot(losses, linewidth=1.6, label=f"k={format_k_label(result['k'])}")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=PAPER_FONT_SIZE)
    ax.set_ylabel("Total weighted loss", fontsize=PAPER_FONT_SIZE)
    ax.set_title("Total loss comparison for different $k$", fontsize=PAPER_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=PAPER_TICK_SIZE, length=3, width=0.8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(frameon=False, fontsize=PAPER_TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "batch_total_loss_all_k.png")
    if SAVE_FIGURES:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(save_path, dpi=PAPER_DPI, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def save_batch_summary(batch_results, output_dir):
    """Save per-wavenumber losses, error metrics, configuration, and output paths."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "batch_summary.csv")
    metric_keys = [
        "rel_l2_complex",
        "rel_l2_real",
        "rel_l2_imag",
        "max_abs_error",
        "max_abs_error_real",
        "max_abs_error_imag",
        "mean_abs_error",
        "mean_abs_error_real",
        "mean_abs_error_imag",
        "r2_real",
        "r2_imag",
        "r2_avg",
    ]
    aligned_metric_keys = [
        prefix + name
        for prefix in ("full_", "r2_150_")
        for name in (
            "rel_l2_complex",
            "rel_l2_real",
            "rel_l2_imag",
            "max_abs_complex",
            "mean_abs_complex",
        )
    ]
    evaluation_keys = [
        *TRAINING_RESOURCE_COLUMNS,
        "training_progression_path",
        "training_progression_metadata_path",
        "training_progression_captured_iterations",
        "normalize_training_residuals",
        "pde_residual_divisor",
        "dtn_residual_divisor",
        "n_epochs",
        "initial_learning_rate",
        "lr_schedule_name",
        "final_learning_rate",
        "evaluation_grid_type",
        "evaluation_grid_size",
        "evaluation_boundary_policy",
        "exact_terms",
        "full_comparison_point_set",
        "full_comparison_point_count",
        "full_comparison_interpolation",
        "six_panel_plot_type",
        "observation_radius",
        "observation_points",
        "observation_offset_index",
        "observation_reference_kind",
        "observation_status",
        "r2_150_valid_points",
        "observation_circle_path",
        "observation_summary_path",
        "use_rad",
        "rad_start_iter",
        "rad_period",
        "rad_mix_ratio",
        "rad_refresh_uniform_part",
        "rad_pool_rule",
        "rad_pool_min",
        "rad_pool_factor",
        "rad_power",
        "rad_c",
        "rad_probability_eps",
        "rad_normalization_eps",
        "rad_indicator",
        "rad_sampling",
        "rad_random_seed",
        "rad_update_count",
        "rad_update_iterations",
        "rad_history_path",
    ]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "k",
                "final_loss",
                "six_panel_path",
                "six_panel_l2_coolwarm_path",
                "six_panel_l2_reference_path",
                "single_total_loss_path",
                "single_components_loss_path",
                "outer_bc_type",
                "dtn_radii",
                "dtn_truncation_orders",
                "dtn_point_counts",
                "dtn_circle_weights",
                "bc2_loss_definition",
            ]
            + metric_keys
            + aligned_metric_keys
            + evaluation_keys
        )
        for r in batch_results:
            metrics = r.get("metrics")
            six_paths = r.get("six_panel_paths")
            aligned_metrics = {**r.get("full_aligned_metrics"), **r.get("observation_metrics")}
            writer.writerow(
                [
                    format_k_label(r.get("k")),
                    r.get("final_loss"),
                    r.get("six_panel_path"),
                    six_paths.get("l2_coolwarm", ""),
                    six_paths.get("l2_reference", ""),
                    r.get("single_total_loss_path"),
                    r.get("single_components_loss_path"),
                    r.get("outer_bc_type", ""),
                    r.get("dtn_radii", ""),
                    r.get("dtn_truncation_orders", ""),
                    r.get("dtn_point_counts", ""),
                    r.get("dtn_circle_weights", ""),
                    r.get("bc2_loss_definition", ""),
                ]
                + [metrics.get(key, "") for key in metric_keys]
                + [aligned_metrics.get(key, "") for key in aligned_metric_keys]
                + [r.get(key, "") for key in evaluation_keys]
            )
    return save_path


def main():
    """Run the selected wavenumbers and write aggregate results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    k_to_run = K_VALUES if RUN_BATCH else [SINGLE_K]
    batch_results = []
    for k_value in k_to_run:
        print("\n" + "#" * 80)
        print(f"Sound-soft circular-cylinder scattering: k={format_k_label(k_value)}")
        print("#" * 80)
        result = run_single_case(k_value, OUTPUT_DIR)
        batch_results.append(result)
        if not POINT_LAYOUT_ONLY:
            save_training_resource_summary(batch_results, OUTPUT_DIR)
    if POINT_LAYOUT_ONLY:
        print(f"Initial point layouts saved to: {os.path.abspath(OUTPUT_DIR)}")
        return
    if len(batch_results) > 1:
        loss_fig_path = plot_batch_total_loss(batch_results, OUTPUT_DIR)
        print(f"Batch total loss figure saved to: {loss_fig_path}")
    summary_path = save_batch_summary(batch_results, OUTPUT_DIR)
    print(f"Batch summary CSV saved to: {summary_path}")


if __name__ == "__main__":
    main()
