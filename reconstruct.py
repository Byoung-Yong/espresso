from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any
import os
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import differential_evolution, minimize

try:
    from numba import njit
except ImportError:
    print("Numba not found. Using slow python mode.")
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator


BASE_POROSITY = 0.33
MIN_POROSITY = 0.14
MAX_COMPACTION_POROSITY_DROP = 0.08
COMPACTION_PRESSURE_DOSE_SCALE = 150.0
MAX_EROSION_RECOVERY_FRACTION = 0.90
SWELLING_SHAPE_EXPONENT = 1.5
RETENTION_RESIDUAL_FRACTION = 0.55
RETENTION_RELEASE_TIME = 8.0
EROSION_START_BEVERAGE_RATIO = 0.15
EROSION_RAMP_BEVERAGE_RATIO = 0.25
VISCOSITY_START_BEVERAGE_RATIO = 0.10
VISCOSITY_RAMP_BEVERAGE_RATIO = 0.20
MAX_SOLUBLES_FRACTION = 0.30
MIN_WET_FACTOR = 1e-4
MODEL_VERSION = "espresso_0319"
R0_BOUND_LOW = 0.30
R0_BOUND_HIGH = 100.0

OBJ_W_FLOW = 10.0
OBJ_W_WEIGHT = 15.0
OBJ_W_FINAL_WEIGHT = 50.0
OBJ_W_RESISTANCE = 40.0
OBJ_W_RESISTANCE_EARLY = 30.0
OBJ_W_RESISTANCE_DROP = 20.0
OBJ_W_PREINFUSION = 50.0
OBJ_W_ONSET = 300.0


# ---------------------------------------------------------------------------
# 1. Numba Simulation (Wetting + Swelling)
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _smoothstep_01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * (3.0 - 2.0 * x)


@njit(fastmath=True)
def _kozeny_carman_resistance_factor(base_porosity: float, eff_porosity: float) -> float:
    base_perm = (base_porosity ** 3) / ((1.0 - base_porosity) ** 2)
    eff_perm = (eff_porosity ** 3) / ((1.0 - eff_porosity) ** 2)
    if eff_perm < 1e-6:
        eff_perm = 1e-6
    return base_perm / eff_perm


@njit(fastmath=True)
def _erosion_activation_gate(current_weight: float, dose: float, wet_factor: float) -> float:
    if dose <= 1e-9:
        return 0.0
    beverage_ratio = current_weight / dose
    mass_gate = _smoothstep_01((beverage_ratio - EROSION_START_BEVERAGE_RATIO) / EROSION_RAMP_BEVERAGE_RATIO)
    return wet_factor * mass_gate


@njit(fastmath=True)
def simulate_numba_core(
    t: np.ndarray,
    pressure: np.ndarray,
    params: np.ndarray,
    dose: float,
    max_solubles: float,
    mu_water: float
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray
]:

    # [R0, alpha_comp, alpha_ero, a_visc, b_visc, k_extr,
    #  t_off, wetting_tau, hold_capacity_ratio, beta_swelling]
    R0          = params[0]
    alpha_comp  = params[1]
    alpha_ero   = params[2]
    a_visc      = params[3]
    b_visc      = params[4]
    k_extr      = params[5]
    t_off       = params[6]
    wetting_tau = params[7]
    hold_capacity_ratio = params[8]
    beta_swelling = params[9]

    n = len(t)
    flow_out = np.zeros(n)
    flow_in = np.zeros(n)
    weight = np.zeros(n)
    tds = np.zeros(n)
    ey = np.zeros(n)
    R_app = np.zeros(n)
    hydration_state = np.zeros(n)
    swelling_state = np.zeros(n)
    porosity = np.zeros(n)
    wet_factor_out = np.zeros(n)
    R_intrinsic = np.zeros(n)
    retained_water = np.zeros(n)
    compaction_state = np.zeros(n)
    erosion_gate_out = np.zeros(n)
    erosion_state_out = np.zeros(n)
    erosion_relief_out = np.zeros(n)

    cumP = 0.0
    extracted = 0.0
    water_retained = 0.0

    dt = np.empty(n)
    dt[0] = 0.0
    dt[1:] = t[1:] - t[:-1]

    for i in range(n):
        Pi = pressure[i]
        if Pi < 0:
            Pi = 0.0
        dti = dt[i]

        # 1. Compaction
        cumP += Pi * dti

        # 2. Wetting gate on outflow. Inflow can still occur before the puck
        # is fully percolated, which lets hydration and swelling build first.
        wet_factor = 0.0
        if t[i] >= t_off:
            if wetting_tau <= 0.1:
                wet_factor = 1.0
            else:
                wet_factor = _smoothstep_01((t[i] - t_off) / wetting_tau)

        # 3. Physics model
        frac_ext = 0.0
        if max_solubles > 1e-9:
            frac_ext = extracted / max_solubles
        if frac_ext > 1.0:
            frac_ext = 1.0

        current_weight = weight[i-1] if i > 0 else 0.0
        if current_weight > 1e-3:
            tds_proxy = extracted / current_weight
        else:
            tds_proxy = 0.0

        retention_capacity = hold_capacity_ratio * dose
        if retention_capacity < 1e-6:
            retention_capacity = 1e-6

        hydration_ratio = water_retained / retention_capacity
        if hydration_ratio > 1.0:
            hydration_ratio = 1.0
        elif hydration_ratio < 0.0:
            hydration_ratio = 0.0
        swell_state = hydration_ratio ** SWELLING_SHAPE_EXPONENT

        comp_state = 1.0 - np.exp(-alpha_comp * cumP / COMPACTION_PRESSURE_DOSE_SCALE)
        comp_drop = MAX_COMPACTION_POROSITY_DROP * comp_state

        gross_structural_drop = comp_drop + beta_swelling * swell_state
        erosion_gate = _erosion_activation_gate(current_weight, dose, wet_factor)
        erosion_state = 1.0 - np.exp(-alpha_ero * frac_ext * erosion_gate)
        erosion_relief = gross_structural_drop * MAX_EROSION_RECOVERY_FRACTION * erosion_state
        net_structural_drop = gross_structural_drop - erosion_relief

        eff_porosity = BASE_POROSITY - net_structural_drop
        if eff_porosity < MIN_POROSITY:
            eff_porosity = MIN_POROSITY

        bed_fac = _kozeny_carman_resistance_factor(BASE_POROSITY, eff_porosity)
        viscosity_gate = wet_factor * _smoothstep_01(
            (current_weight / max(dose, 1e-9) - VISCOSITY_START_BEVERAGE_RATIO) / VISCOSITY_RAMP_BEVERAGE_RATIO
        )
        visc_fac = 1.0 + viscosity_gate * a_visc * (tds_proxy ** b_visc)

        R_val = mu_water * R0 * bed_fac * visc_fac
        if R_val < 1e-4:
            R_val = 1e-4

        Q_in = Pi / R_val
        Q_out = wet_factor * Q_in
        flow_in[i] = Q_in
        wet_factor_out[i] = wet_factor
        R_intrinsic[i] = R_val
        erosion_gate_out[i] = erosion_gate

        # 4. Water retained in the puck drives swelling.
        # Early inflow loads the bed, but once percolation develops the excess
        # stored water should relax toward a smaller residual hold-up instead of
        # accumulating monotonically to saturation.
        target_retained = retention_capacity * (
            RETENTION_RESIDUAL_FRACTION + (1.0 - RETENTION_RESIDUAL_FRACTION) * (1.0 - wet_factor)
        )
        retained_loading = (Q_in - Q_out) * dti
        if retained_loading > 0.0:
            water_retained += retained_loading

        retained_excess = water_retained - target_retained
        if retained_excess < 0.0:
            retained_excess = 0.0
        release_rate = wet_factor * retained_excess / RETENTION_RELEASE_TIME
        retained_release = release_rate * dti
        if retained_release > water_retained:
            retained_release = water_retained
        water_retained -= retained_release
        if water_retained < 0.0:
            water_retained = 0.0

        hydration_ratio_post = water_retained / retention_capacity
        if hydration_ratio_post > 1.0:
            hydration_ratio_post = 1.0
        elif hydration_ratio_post < 0.0:
            hydration_ratio_post = 0.0
        swell_state_post = hydration_ratio_post ** SWELLING_SHAPE_EXPONENT
        hydration_state[i] = hydration_ratio_post
        swelling_state[i] = swell_state_post
        retained_water[i] = water_retained
        compaction_state[i] = comp_state

        # Extraction (Solutes Mass)
        S_remaining = max_solubles - extracted
        if S_remaining < 0:
            S_remaining = 0.0

        dM = k_extr * Q_out * (S_remaining / dose) * dti
        if dM > S_remaining:
            dM = S_remaining
        extracted += dM

        # Beverage mass on the scale includes both expelled liquid and dissolved solids.
        solute_out_rate = 0.0
        if dti > 1e-12:
            solute_out_rate = dM / dti
        beverage_flow = Q_out + solute_out_rate
        flow_out[i] = beverage_flow

        beverage_mass_delta = beverage_flow * dti
        if i > 0:
            weight[i] = weight[i-1] + beverage_mass_delta
        else:
            weight[i] = beverage_mass_delta

        frac_ext_post = 0.0
        if max_solubles > 1e-9:
            frac_ext_post = extracted / max_solubles
        if frac_ext_post > 1.0:
            frac_ext_post = 1.0

        erosion_gate_post = _erosion_activation_gate(weight[i], dose, wet_factor)
        gross_structural_drop_post = comp_drop + beta_swelling * swell_state_post
        erosion_state_post = 1.0 - np.exp(-alpha_ero * frac_ext_post * erosion_gate_post)
        erosion_relief_post = gross_structural_drop_post * MAX_EROSION_RECOVERY_FRACTION * erosion_state_post
        net_structural_drop_post = gross_structural_drop_post - erosion_relief_post

        eff_porosity_post = BASE_POROSITY - net_structural_drop_post
        if eff_porosity_post < MIN_POROSITY:
            eff_porosity_post = MIN_POROSITY
        porosity[i] = eff_porosity_post
        erosion_gate_out[i] = erosion_gate_post
        erosion_state_out[i] = erosion_state_post
        erosion_relief_out[i] = erosion_relief_post

        # TDS / EY calculation
        if weight[i] > 1e-6:
            tds[i] = (extracted / weight[i]) * 100.0
        ey[i] = (extracted / dose) * 100.0
        R_app[i] = Pi / max(beverage_flow, MIN_WET_FACTOR)

    return (
        flow_out, flow_in, weight, tds, ey,
        R_app, hydration_state, swelling_state, porosity, wet_factor_out, R_intrinsic, retained_water, compaction_state, erosion_gate_out,
        erosion_state_out, erosion_relief_out
    )


def simulate_numba(
    t: np.ndarray,
    pressure: np.ndarray,
    params: np.ndarray,
    dose: float,
    max_solubles: float,
    mu_water: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flow_out, _, weight, tds, ey, R_app, _, _, _, _, _, _, _, _, _, _ = simulate_numba_core(
        t, pressure, params, dose, max_solubles, mu_water
    )
    return flow_out, weight, tds, ey, R_app


# ---------------------------------------------------------------------------
# 2. Wrapper Update
# ---------------------------------------------------------------------------

@dataclass
class ImpedanceStateParams:
    R0: float = 8.0
    alpha_comp: float = 0.02
    alpha_ero: float = 2.0
    a_visc: float = 3.0
    b_visc: float = 1.0
    k_extr: float = 0.5
    t_off: float = 5.0
    wetting_tau: float = 4.0
    hold_capacity_ratio: float = 0.20
    beta_swelling: float = 0.08


class ImpedanceStateCircuit:
    def __init__(self, dose_g: float = 18.0, efficiency_factor: float = 0.66):
        """
        efficiency_factor (η):
            - Empirical Extraction Efficiency Factor
            - Typical literature: EY_real ≈ 18–22%, max_solubles ≈ 30% -> η ≈ 0.6–0.75
            - Default: η = 0.70
        """
        self.dose = float(dose_g)
        self.efficiency_factor = float(efficiency_factor)

        # Adjusted available solubles
        self.max_solubles = MAX_SOLUBLES_FRACTION * self.dose * self.efficiency_factor

        self.mu_water = 1.0
        self.params = ImpedanceStateParams()

    def set_array(self, p: np.ndarray) -> None:
        self.params = ImpedanceStateParams(*p)

    def to_array(self) -> np.ndarray:
        p = self.params
        return np.array([
            p.R0, p.alpha_comp, p.alpha_ero, p.a_visc, p.b_visc,
            p.k_extr, p.t_off, p.wetting_tau, p.hold_capacity_ratio, p.beta_swelling
        ], dtype=float)

    def simulate(self, t: np.ndarray, pressure: np.ndarray, return_diagnostics: bool = False):
        p_arr = self.to_array()
        if not return_diagnostics:
            return simulate_numba(
                t, pressure, p_arr,
                self.dose, self.max_solubles, self.mu_water
            )

        flow_out, flow_in, weight, tds, ey, R_app, hydration, swelling, porosity, wet_factor, R_intrinsic, retained_water, compaction_state, erosion_gate, erosion_state, erosion_relief = simulate_numba_core(
            t, pressure, p_arr, self.dose, self.max_solubles, self.mu_water
        )
        return {
            "flow_out": flow_out,
            "flow_in": flow_in,
            "weight": weight,
            "tds": tds,
            "ey": ey,
            "R_app": R_app,
            "hydration": hydration,
            "swelling": swelling,
            "porosity": porosity,
            "wet_factor": wet_factor,
            "R_intrinsic": R_intrinsic,
            "retained_water": retained_water,
            "compaction_state": compaction_state,
            "erosion_gate": erosion_gate,
            "erosion_state": erosion_state,
            "erosion_relief": erosion_relief,
        }


# ---------------------------------------------------------------------------
# 3. Data Loading
# ---------------------------------------------------------------------------

def _candidate_json_paths(csv_path: str) -> List[Path]:
    csv_file = Path(csv_path)
    candidates: List[Path] = [csv_file.with_suffix(".json")]
    if csv_file.parent.name.lower() == "csv":
        candidates.append(csv_file.parent.parent / "json" / f"{csv_file.stem}.json")

    unique: List[Path] = []
    seen = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _load_json_sidecar_metadata(csv_path: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "json_present": False,
        "json_path": None,
        "profile_title": None,
        "bean_weight": None,
        "drink_weight": None,
        "drink_tds_reported": None,
        "drink_ey_reported": None,
        "grinder_model": None,
        "grinder_setting": None,
        "roast_level": None,
        "barista": None,
        "profile_url": None,
    }

    for candidate in _candidate_json_paths(csv_path):
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8-sig") as fh:
                obj = json.load(fh)
        except Exception:
            continue

        meta["json_present"] = True
        meta["json_path"] = str(candidate)
        meta["profile_title"] = obj.get("profile_title")
        meta["bean_weight"] = obj.get("bean_weight")
        meta["drink_weight"] = obj.get("drink_weight")
        meta["drink_tds_reported"] = obj.get("drink_tds")
        meta["drink_ey_reported"] = obj.get("drink_ey")
        meta["grinder_model"] = obj.get("grinder_model")
        meta["grinder_setting"] = obj.get("grinder_setting")
        meta["roast_level"] = obj.get("roast_level")
        meta["barista"] = obj.get("barista")
        meta["profile_url"] = obj.get("profile_url")
        break

    return meta


def _normalize_meta_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _meta_lower(value: Any) -> str:
    value = _normalize_meta_scalar(value)
    if value is None:
        return ""
    return str(value).strip().lower()


def load_and_prep_data(csv_path: str, dt: float = 0.1):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Read error: {csv_path} ({e})")

    # --- Extract Metadata (Name, Weight) ---
    meta_props: Dict[str, Any] = {
        "meta_name": None,
        "meta_weight": None,
        "csv_meta_name": None,
        "csv_meta_weight": None,
    }
    meta_props.update(_load_json_sidecar_metadata(csv_path))

    meta_props = {key: _normalize_meta_scalar(value) for key, value in meta_props.items()}

    if meta_props.get("profile_title") is not None:
        meta_props["meta_name"] = meta_props["profile_title"]
    if meta_props.get("drink_weight") is not None:
        meta_props["meta_weight"] = meta_props["drink_weight"]

    if "information_type" in df.columns:
        # Filter for metadata rows
        meta_df = df[df["information_type"] == "meta"]

        # Look for exact "Name" and "Weight" keys (case-insensitive check for robustness)
        # Meta rows contain many NaN placeholders; normalize each cell before string matching.
        for _, row in meta_df.iterrows():
            row_vals = row.values
            for idx, val in enumerate(row_vals):
                val_lower = _meta_lower(val)
                if not val_lower:
                    continue

                # Check for "Name"
                if val_lower == "name":
                    if idx + 1 < len(row_vals):
                        next_val = _normalize_meta_scalar(row_vals[idx+1])
                        meta_props["csv_meta_name"] = next_val
                        if not meta_props.get("meta_name"):
                            meta_props["meta_name"] = next_val

                # Check for "Weight"
                if val_lower == "weight":
                    if idx + 1 < len(row_vals):
                        next_val = _normalize_meta_scalar(row_vals[idx+1])
                        meta_props["csv_meta_weight"] = next_val
                        if not meta_props.get("meta_weight"):
                            meta_props["meta_weight"] = next_val

        # Now remove meta rows for data processing
        df = df[df["information_type"] != "meta"].copy()

    req = ["elapsed", "pressure", "flow_out", "current_total_shot_weight"]
    for c in req:
        if c not in df.columns:
            if c == "current_total_shot_weight" and "weight" in df.columns:
                df["current_total_shot_weight"] = df["weight"]
            else:
                raise ValueError(f"Missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df.dropna(subset=req, inplace=True)
    if len(df) < 10:
        raise ValueError("Empty data")

    df["elapsed"] -= df["elapsed"].min()
    t_raw = df["elapsed"].values
    t_new = np.arange(0.0, t_raw.max() + 0.5 * dt, dt)

    p_new = np.interp(t_new, t_raw, df["pressure"].values)
    q_new = np.interp(t_new, t_raw, df["flow_out"].values)
    w_new = np.interp(t_new, t_raw, df["current_total_shot_weight"].values)
    w_new -= w_new[0]

    # Smoothing
    win = int(1.5 / dt)
    if win % 2 == 0:
        win += 1
    if win >= 3:
        q_smooth = savgol_filter(q_new, win, 2)
        p_smooth = savgol_filter(p_new, win, 2)
    else:
        q_smooth = q_new
        p_smooth = p_new

    q_smooth = np.maximum(q_smooth, 0)
    p_smooth = np.maximum(p_smooth, 0)

    # Return meta_props as well
    return t_new, p_smooth, q_smooth, w_new, meta_props


def estimate_onset_time(t: np.ndarray, q: np.ndarray, threshold: float = 0.1, fallback: float = 5.0) -> float:
    idx = np.flatnonzero(q > threshold)
    if idx.size == 0:
        return float(fallback)
    return float(t[idx[0]])


def _resolve_effective_dose_g(meta_props: Dict[str, Any], default_dose_g: float) -> Tuple[float, Any, str]:
    reported = meta_props.get("bean_weight")
    try:
        reported_val = float(reported)
    except (TypeError, ValueError):
        return float(default_dose_g), reported, "default"

    # Keep clearly implausible crowd-sourced inputs from distorting the hydraulic model.
    if 12.0 <= reported_val <= 25.0:
        return reported_val, reported_val, "json_bean_weight"
    return float(default_dose_g), reported_val, "default_out_of_range"


def is_shot_profile_csv(csv_path: str) -> bool:
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f), [])
    except Exception:
        return False

    required = {"elapsed", "pressure", "flow_out"}
    return required.issubset(set(header))


def _collect_shot_csv_files(data_dir: str) -> Tuple[List[str], List[str]]:
    data_root = Path(data_dir).resolve()
    files: List[str] = []
    skipped: List[str] = []

    for path in sorted(data_root.rglob("*.csv")):
        if path.name == "curated_manifest.csv":
            skipped.append(str(path))
            continue
        if any(part.startswith("simulation_results_") for part in path.parts):
            skipped.append(str(path))
            continue
        if is_shot_profile_csv(str(path)):
            files.append(str(path))
        else:
            skipped.append(str(path))

    return files, skipped


def _load_manifest_file_list(manifest_path: str, group_filter: str = "") -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    df = pd.read_csv(manifest_path)
    if df.empty:
        return [], {}

    if group_filter and "condition_group" in df.columns:
        group_lower = group_filter.lower()
        mask = df["condition_group"].fillna("").astype(str).str.lower() == group_lower
        if mask.any():
            df = df.loc[mask].copy()

    path_col = None
    for candidate in ["local_csv_path", "source_path", "source_relpath"]:
        if candidate in df.columns:
            path_col = candidate
            break
    if path_col is None:
        raise ValueError(f"Manifest missing usable path column: {manifest_path}")

    files: List[str] = []
    meta_by_path: Dict[str, Dict[str, Any]] = {}
    for _, row in df.iterrows():
        raw_path = str(row[path_col])
        if not raw_path or raw_path.lower() == "nan":
            continue
        abs_path = os.path.abspath(raw_path)
        if not os.path.exists(abs_path):
            continue
        files.append(abs_path)
        meta_by_path[abs_path] = row.to_dict()

    return files, meta_by_path


def _result_relpath(csv_path: str, data_dir: str) -> str:
    return os.path.relpath(os.path.abspath(csv_path), start=os.path.abspath(data_dir))


def _output_path_for_result(base_dir: str, res: Dict[str, Any], ext: str) -> str:
    relpath = res.get("source_relpath")
    if relpath:
        out_path = os.path.join(base_dir, os.path.splitext(relpath)[0] + ext)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        return out_path
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, res["filename"].replace(".csv", ext))


# ---------------------------------------------------------------------------
# 4. Objective Function (Updated for Resistance & Mass)
# ---------------------------------------------------------------------------

def _objective_terms(
    t: np.ndarray,
    p: np.ndarray,
    q_ref: np.ndarray,
    w_ref: np.ndarray,
    q_sim: np.ndarray,
    w_sim: np.ndarray,
    R_sim: np.ndarray,
    dose: float,
    t_onset_ref: float,
    t_off: float,
) -> Dict[str, float]:
    dose_safe = max(float(dose), 1e-9)

    mask_shot = q_ref > 0.1
    weights = np.ones_like(q_ref)
    weights[mask_shot] = 5.0

    err_q = (q_sim - q_ref)
    mse_q = np.mean(weights * (err_q ** 2))
    mse_w = np.mean((w_sim - w_ref) ** 2)
    err_final_w = (w_sim[-1] - w_ref[-1]) ** 2

    mask_r = (q_ref > 0.5) & (p > 1.0)
    mse_log_r = 0.0
    mse_log_r_early = 0.0
    err_log_r_drop = 0.0
    if np.sum(mask_r) > 10:
        r_ref_val = p[mask_r] / q_ref[mask_r]
        r_sim_val = R_sim[mask_r]

        r_sim_val = np.maximum(r_sim_val, 1e-5)
        r_ref_val = np.maximum(r_ref_val, 1e-5)

        mse_log_r = np.mean((np.log(r_sim_val) - np.log(r_ref_val)) ** 2)

        w_ratio = w_ref / dose_safe
        early_limit = EROSION_START_BEVERAGE_RATIO + 0.05
        late_start = EROSION_START_BEVERAGE_RATIO + EROSION_RAMP_BEVERAGE_RATIO + 0.05
        mask_r_early = mask_r & (w_ratio <= early_limit)
        mask_r_late = mask_r & (w_ratio >= late_start)

        if np.sum(mask_r_early) >= 5:
            r_ref_early = np.maximum(p[mask_r_early] / q_ref[mask_r_early], 1e-5)
            r_sim_early = np.maximum(R_sim[mask_r_early], 1e-5)
            mse_log_r_early = np.mean((np.log(r_sim_early) - np.log(r_ref_early)) ** 2)

        if np.sum(mask_r_early) >= 5 and np.sum(mask_r_late) >= 8:
            r_ref_early = np.maximum(p[mask_r_early] / q_ref[mask_r_early], 1e-5)
            r_sim_early = np.maximum(R_sim[mask_r_early], 1e-5)
            r_ref_late = np.maximum(p[mask_r_late] / q_ref[mask_r_late], 1e-5)
            r_sim_late = np.maximum(R_sim[mask_r_late], 1e-5)

            log_r_ref_drop = np.median(np.log(r_ref_late)) - np.median(np.log(r_ref_early))
            log_r_sim_drop = np.median(np.log(r_sim_late)) - np.median(np.log(r_sim_early))
            err_log_r_drop = (log_r_sim_drop - log_r_ref_drop) ** 2

    mask_pre = t < t_off
    pen_pre = 0.0
    if np.any(mask_pre):
        pen_pre = np.sum(q_sim[mask_pre] ** 2)

    thr = 0.1
    if np.any(q_sim > thr):
        idx_sim = np.argmax(q_sim > thr)
        t_onset_sim = float(t[idx_sim])
    else:
        t_onset_sim = float(t[-1])
    onset_err = (t_onset_sim - t_onset_ref) ** 2

    loss_flow = OBJ_W_FLOW * mse_q
    loss_weight = OBJ_W_WEIGHT * mse_w
    loss_final_weight = OBJ_W_FINAL_WEIGHT * err_final_w
    loss_resistance = OBJ_W_RESISTANCE * mse_log_r
    loss_resistance_early = OBJ_W_RESISTANCE_EARLY * mse_log_r_early
    loss_resistance_drop = OBJ_W_RESISTANCE_DROP * err_log_r_drop
    loss_preinfusion = OBJ_W_PREINFUSION * pen_pre
    loss_onset = OBJ_W_ONSET * onset_err
    loss_total = (
        loss_flow
        + loss_weight
        + loss_final_weight
        + loss_resistance
        + loss_resistance_early
        + loss_resistance_drop
        + loss_preinfusion
        + loss_onset
    )

    return {
        "obj_mse_flow": float(mse_q),
        "obj_mse_weight": float(mse_w),
        "obj_err_final_weight": float(err_final_w),
        "obj_mse_log_resistance": float(mse_log_r),
        "obj_mse_log_resistance_early": float(mse_log_r_early),
        "obj_err_log_resistance_drop": float(err_log_r_drop),
        "obj_pen_preinfusion": float(pen_pre),
        "obj_err_onset": float(onset_err),
        "obj_loss_flow": float(loss_flow),
        "obj_loss_weight": float(loss_weight),
        "obj_loss_final_weight": float(loss_final_weight),
        "obj_loss_resistance": float(loss_resistance),
        "obj_loss_resistance_early": float(loss_resistance_early),
        "obj_loss_resistance_drop": float(loss_resistance_drop),
        "obj_loss_preinfusion": float(loss_preinfusion),
        "obj_loss_onset": float(loss_onset),
        "obj_loss_total": float(loss_total),
        "obj_t_onset_ref": float(t_onset_ref),
        "obj_t_onset_sim": float(t_onset_sim),
    }


def objective_state(param_arr, t, p, q_ref, w_ref, dose, max_sol, mu_w, t_onset_ref):
    q_sim, w_sim, tds_sim, ey_sim, R_sim = simulate_numba(t, p, param_arr, dose, max_sol, mu_w)
    terms = _objective_terms(
        t=t,
        p=p,
        q_ref=q_ref,
        w_ref=w_ref,
        q_sim=q_sim,
        w_sim=w_sim,
        R_sim=R_sim,
        dose=dose,
        t_onset_ref=t_onset_ref,
        t_off=float(param_arr[6]),
    )
    return terms["obj_loss_total"]


# ---------------------------------------------------------------------------
# 5. Fitting Process (Calculates R^2)
# ---------------------------------------------------------------------------

def fit_single_shot_2stage(csv_path, dose_g: float = 18.0, efficiency_factor: float = 0.70):
    # Unpack meta_props
    t, p, q_ref, w_ref, meta_props = load_and_prep_data(csv_path)
    effective_dose_g, reported_dose_g, dose_source = _resolve_effective_dose_g(meta_props, dose_g)

    t_start_est = estimate_onset_time(t, q_ref, threshold=0.1, fallback=5.0)

    # [R0, alpha_comp, alpha_ero, a_visc, b_visc, k_extr,
    #  t_off, wetting_tau, hold_capacity_ratio, beta_swelling]
    bounds = [
        (R0_BOUND_LOW, R0_BOUND_HIGH),  # R0 baseline hydraulic resistance
        (0.0, 0.5),                 # alpha_comp
        (0.1, 10.0),                # alpha_ero
        (0.0, 10.0),                # a_visc
        (0.1, 3.0),                 # b_visc
        (0.1, 5.0),                 # k_extr
        (max(0, t_start_est - 1.5), t_start_est + 1.5),  # t_off
        (0.5, 8.0),                 # wetting_tau
        (0.05, 0.60),               # hold_capacity_ratio [g_water / g_dry_coffee]
        (0.0, 0.18)                 # beta_swelling
    ]

    eff_factor = float(efficiency_factor)
    effective_max_solubles = MAX_SOLUBLES_FRACTION * effective_dose_g * eff_factor

    args = (t, p, q_ref, w_ref, effective_dose_g, effective_max_solubles, 1.0, t_start_est)

    # Stage 1: Global
    res_de = differential_evolution(
        objective_state,
        bounds,
        args=args,
        strategy='best1bin',
        maxiter=60,
        popsize=15,
        tol=0.02,
        seed=42,
        polish=False
    )

    # Stage 2: Local
    res_opt = minimize(
        objective_state,
        res_de.x,
        args=args,
        method='L-BFGS-B',
        bounds=bounds,
        tol=1e-6
    )

    best_params = res_opt.x

    circuit = ImpedanceStateCircuit(effective_dose_g, efficiency_factor=eff_factor)
    circuit.set_array(best_params)
    sim = circuit.simulate(t, p, return_diagnostics=True)

    q_sim = sim["flow_out"]
    w_sim = sim["weight"]
    objective_terms = _objective_terms(
        t=t,
        p=p,
        q_ref=q_ref,
        w_ref=w_ref,
        q_sim=q_sim,
        w_sim=w_sim,
        R_sim=sim["R_app"],
        dose=effective_dose_g,
        t_onset_ref=t_start_est,
        t_off=float(best_params[6]),
    )

    # --- R-Squared Calculation ---
    ss_res = np.sum((q_ref - q_sim) ** 2)
    ss_tot = np.sum((q_ref - np.mean(q_ref)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Result dictionary
    result = {
        "model_version": MODEL_VERSION,
        "filename": os.path.basename(csv_path),
        "source_path": os.path.abspath(csv_path),
        "params": asdict(circuit.params),
        "loss": res_opt.fun,
        "r_squared": r2,
        "t": t, "p": p, "q_ref": q_ref, "w_ref": w_ref,
        "q_sim": q_sim, "w_sim": w_sim,
        "q_in": sim["flow_in"],
        "tds": sim["tds"],
        "ey": sim["ey"],
        "R_sim": sim["R_app"],
        "R_intrinsic": sim["R_intrinsic"],
        "hydration": sim["hydration"],
        "swelling": sim["swelling"],
        "porosity": sim["porosity"],
        "wet_factor": sim["wet_factor"],
        "retained_water": sim["retained_water"],
        "compaction_state": sim["compaction_state"],
        "erosion_gate": sim["erosion_gate"],
        "erosion_state": sim["erosion_state"],
        "erosion_relief": sim["erosion_relief"],
        "dose_g_used": effective_dose_g,
        "dose_g_reported": reported_dose_g,
        "dose_source": dose_source,
    }
    result.update(objective_terms)

    # Add extracted metadata to the result
    result.update(meta_props)

    return result


# ---------------------------------------------------------------------------
# 6. Plotting (Modified for Structured Output)
# ---------------------------------------------------------------------------

def plot_result(res, out_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --- [0,0] Flow & Pressure ---
    ax1 = axes[0, 0]
    ax1.plot(res['t'], res['q_ref'], 'k.', alpha=0.3, label='Flow Meas')
    ax1.plot(res['t'], res['q_sim'], 'r-', lw=2, label='Flow Sim')
    if 'q_in' in res:
        ax1.plot(res['t'], res['q_in'], color='tab:orange', ls='--', lw=1.4, alpha=0.9, label='Flow Into Puck')
    ax1.set_ylabel("Flow (ml/s)")
    ax1.set_xlabel("Time (s)")

    ax2 = ax1.twinx()
    ax2.plot(res['t'], res['p'], 'b-', alpha=0.4, lw=1.5, label='Pressure')
    ax2.set_ylabel("Pressure (bar)", color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    meta_title = ""
    if res.get('meta_name'):
        meta_title = f" - {res['meta_name']}"
    ax1.set_title(f"Flow & Pressure (R2={res['r_squared']:.3f}){meta_title}")

    # --- [0,1] Resistance Only (Log Scale) ---
    ax = axes[0, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        R_meas = res['p'] / np.where(res['q_ref'] > 0.1, res['q_ref'], np.nan)

    ax.semilogy(res['t'], R_meas, 'k.', alpha=0.1, label='Meas R (Ref)')
    ax.semilogy(res['t'], res['R_sim'], 'm-', lw=2, label='Model R')
    if 'R_intrinsic' in res:
        ax.semilogy(res['t'], res['R_intrinsic'], color='0.45', ls='--', lw=1.3, label='Intrinsic Bed R')

    ax.set_title("Hydraulic Resistance (Log Scale)")
    ax.set_ylabel("Resistance (bar s/ml)")
    ax.set_xlabel("Time (s)")
    ax.legend()

    # --- [1,0] Accumulated Weight ---
    ax = axes[1, 0]
    ax.plot(res['t'], res['w_ref'], 'k--', label='Meas')
    ax.plot(res['t'], res['w_sim'], 'g-', label='Sim')
    ax.set_title("Accumulated Mass")
    ax.set_ylabel("Weight (g)")
    ax.set_xlabel("Time (s)")
    ax.legend()

    # --- [1,1] TDS & EY ---
    ax = axes[1, 1]
    ax.plot(res['t'], res['tds'], color='tab:brown', label='TDS')
    ax.set_ylabel("TDS (%)", color='tab:brown')
    ax.tick_params(axis='y', labelcolor='tab:brown')

    ax2 = ax.twinx()
    ax2.plot(res['t'], res['ey'], 'r--', label='EY')
    ax2.set_ylabel("Extraction Yield (%)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    title = f"Final TDS: {res['tds'][-1]:.2f}% / EY: {res['ey'][-1]:.2f}%"
    if 'swelling' in res and 'porosity' in res:
        title += f" / Hydr: {res.get('hydration', res['swelling'])[-1]:.2f} / Swell: {res['swelling'][-1]:.2f} / eps: {res['porosity'][-1]:.2f}"
    if 'retained_water' in res:
        title += f" / Hold-up: {res['retained_water'][-1]:.2f} g"
    if 'compaction_state' in res:
        title += f" / Comp: {res['compaction_state'][-1]:.2f}"
    if 'erosion_gate' in res:
        title += f" / EroGate: {res['erosion_gate'][-1]:.2f}"
    if 'erosion_state' in res:
        title += f" / EroState: {res['erosion_state'][-1]:.2f}"
    ax.set_title(title)
    ax.set_xlabel("Time (s)")

    plt.tight_layout()
    out_path = _output_path_for_result(out_dir, res, ".png")
    plt.savefig(out_path)
    plt.close()


def save_csv(res, out_dir: str):
    df = pd.DataFrame({
        'time':      res['t'],
        'pressure':  res['p'],
        'flow_in':   res.get('q_in', np.full_like(res['t'], np.nan)),
        'flow_meas': res['q_ref'],
        'flow_sim':  res['q_sim'],
        'weight':    res['w_sim'],
        'R_sim':     res['R_sim'],
        'R_intrinsic': res.get('R_intrinsic', np.full_like(res['t'], np.nan)),
        'TDS':       res['tds'],
        'EY':        res['ey'],
        'wet_factor': res.get('wet_factor', np.full_like(res['t'], np.nan)),
        'hydration': res.get('hydration', np.full_like(res['t'], np.nan)),
        'swelling': res.get('swelling', np.full_like(res['t'], np.nan)),
        'porosity': res.get('porosity', np.full_like(res['t'], np.nan)),
        'retained_water': res.get('retained_water', np.full_like(res['t'], np.nan)),
        'compaction_state': res.get('compaction_state', np.full_like(res['t'], np.nan)),
        'erosion_gate': res.get('erosion_gate', np.full_like(res['t'], np.nan)),
        'erosion_state': res.get('erosion_state', np.full_like(res['t'], np.nan)),
        'erosion_relief': res.get('erosion_relief', np.full_like(res['t'], np.nan)),
    })
    out_path = _output_path_for_result(out_dir, res, ".csv")
    df.to_csv(out_path, index=False)


def plot_batch_summary(base_out_dir: str):
    batch_path = os.path.join(base_out_dir, "batch_results.csv")
    if not os.path.exists(batch_path):
        return

    df = pd.read_csv(batch_path)
    if df.empty:
        return

    excluded_path = os.path.join(base_out_dir, "excluded_results.csv")
    df_excluded = pd.read_csv(excluded_path) if os.path.exists(excluded_path) else None

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    ax = axes[0, 0]
    ax.hist(df["r_squared"], bins=18, alpha=0.85, color="tab:green", label="Passed")
    if df_excluded is not None and not df_excluded.empty and "r_squared" in df_excluded:
        ax.hist(df_excluded["r_squared"], bins=18, alpha=0.55, color="tab:red", label="Excluded")
    ax.set_title("R-squared Distribution")
    ax.set_xlabel("R-squared")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[0, 1]
    sc = ax.scatter(
        df["hold_capacity_ratio"],
        df["beta_swelling"],
        c=df["final_swelling"],
        cmap="viridis",
        s=38,
        alpha=0.85
    )
    ax.set_title("Swelling Parameters")
    ax.set_xlabel("Hold Capacity Ratio (g/g)")
    ax.set_ylabel("Beta Swelling")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Final Swelling")

    ax = axes[0, 2]
    if "final_hydration_state" in df.columns:
        ax.hist(df["final_hydration_state"], bins=16, alpha=0.55, color="tab:green", label="Final Hydration")
    ax.hist(df["final_swelling"], bins=16, alpha=0.7, color="tab:orange", label="Final Swelling")
    ax.hist(df["final_porosity"], bins=16, alpha=0.7, color="tab:blue", label="Final Porosity")
    if "final_compaction_state" in df.columns:
        ax.hist(df["final_compaction_state"], bins=16, alpha=0.55, color="tab:red", label="Final Compaction")
    if "final_erosion_gate" in df.columns:
        ax.hist(df["final_erosion_gate"], bins=16, alpha=0.45, color="tab:purple", label="Final Erosion Gate")
    if "final_erosion_state" in df.columns:
        ax.hist(df["final_erosion_state"], bins=16, alpha=0.35, color="tab:brown", label="Final Erosion State")
    ax.set_title("Final State Variables")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1, 0]
    sc = ax.scatter(
        df["final_retained_water"],
        df["final_swelling"],
        c=df["final_porosity"],
        cmap="plasma",
        s=38,
        alpha=0.85
    )
    ax.set_title("Retained Water vs Swelling")
    ax.set_xlabel("Final Retained Water (g)")
    ax.set_ylabel("Final Swelling")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Final Porosity")

    ax = axes[1, 1]
    sc = ax.scatter(
        df["final_porosity"],
        df["final_tds"],
        c=df["final_ey"],
        cmap="cividis",
        s=38,
        alpha=0.85
    )
    ax.set_title("Porosity vs TDS")
    ax.set_xlabel("Final Porosity")
    ax.set_ylabel("Final TDS (%)")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Final EY (%)")

    ax = axes[1, 2]
    if "meta_name" in df.columns and df["meta_name"].notna().any():
        top = (
            df.dropna(subset=["meta_name"])
              .groupby("meta_name")
              .agg(final_swelling_mean=("final_swelling", "mean"), count=("filename", "count"))
              .sort_values(["count", "final_swelling_mean"], ascending=[False, False])
              .head(8)
              .sort_values("final_swelling_mean")
        )
        labels = [f"{name} (n={count})" for name, count in zip(top.index, top["count"])]
        ax.barh(labels, top["final_swelling_mean"], color="tab:purple", alpha=0.8)
        ax.set_xlabel("Mean Final Swelling")
        ax.set_title("Top Profiles by Mean Swelling")
    else:
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(base_out_dir, "batch_summary.png")
    plt.savefig(out_path, dpi=160)
    plt.close()


def _pick_example_rows(df: pd.DataFrame) -> List[pd.Series]:
    qvals = [0.15, 0.50, 0.85]
    targets = df["final_swelling"].quantile(qvals).to_numpy()
    chosen: List[pd.Series] = []
    used = set()
    for target in targets:
        order = (df["final_swelling"] - target).abs().sort_values().index
        for idx in order:
            filename = df.loc[idx, "filename"]
            if filename not in used:
                chosen.append(df.loc[idx])
                used.add(filename)
                break
    if len(chosen) < 3:
        for _, row in df.sort_values("final_swelling").iterrows():
            if row["filename"] not in used:
                chosen.append(row)
                used.add(row["filename"])
            if len(chosen) == 3:
                break
    return chosen[:3]


def plot_swelling_examples(base_out_dir: str):
    batch_path = os.path.join(base_out_dir, "batch_results.csv")
    trace_dir = os.path.join(base_out_dir, "traces")
    if not os.path.exists(batch_path) or not os.path.isdir(trace_dir):
        return

    df = pd.read_csv(batch_path)
    if df.empty:
        return

    rows = _pick_example_rows(df)
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 2, figsize=(15, 4.5 * len(rows)))
    if len(rows) == 1:
        axes = np.array([axes])

    labels = ["Lower Swelling", "Median Swelling", "Higher Swelling"]
    for i, row in enumerate(rows):
        trace_rel = row.get("source_relpath") if isinstance(row, pd.Series) else None
        if pd.isna(trace_rel):
            trace_rel = None
        trace_path = os.path.join(trace_dir, trace_rel if trace_rel else row["filename"])
        if not os.path.exists(trace_path):
            continue
        trace = pd.read_csv(trace_path)

        ax = axes[i, 0]
        ax.plot(trace["time"], trace["flow_meas"], "k.", alpha=0.25, label="Flow Meas")
        ax.plot(trace["time"], trace["flow_sim"], color="tab:red", lw=2, label="Flow Sim")
        ax.plot(trace["time"], trace["flow_in"], color="tab:orange", lw=1.2, ls="--", label="Flow Into Puck")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Flow (ml/s)")
        ax2 = ax.twinx()
        ax2.plot(trace["time"], trace["pressure"], color="tab:blue", lw=1.2, alpha=0.35, label="Pressure")
        ax2.set_ylabel("Pressure (bar)", color="tab:blue")
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        name = row.get("meta_name")
        if pd.isna(name):
            name = row["filename"]
        ax.set_title(f"{labels[i]}: {name}")

        ax = axes[i, 1]
        if "hydration" in trace.columns:
            ax.plot(trace["time"], trace["hydration"], color="tab:green", lw=1.6, label="Hydration")
        ax.plot(trace["time"], trace["swelling"], color="tab:red", lw=2, label="Swelling")
        ax.plot(trace["time"], trace["retained_water"], color="tab:olive", lw=1.4, label="Retained Water")
        ax.plot(trace["time"], trace["wet_factor"], color="0.4", lw=1.2, ls="--", label="Wet Factor")
        if "compaction_state" in trace:
            ax.plot(trace["time"], trace["compaction_state"], color="tab:purple", lw=1.3, ls="-.", label="Compaction")
        if "erosion_gate" in trace:
            ax.plot(trace["time"], trace["erosion_gate"], color="tab:orange", lw=1.3, ls=":", label="Erosion Gate")
        if "erosion_state" in trace:
            ax.plot(trace["time"], trace["erosion_state"], color="tab:brown", lw=1.3, ls="--", label="Erosion State")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Swelling / Hold-up")
        ax2 = ax.twinx()
        ax2.plot(trace["time"], trace["porosity"], color="tab:blue", lw=1.8, alpha=0.9, label="Porosity")
        ax2.set_ylabel("Porosity", color="tab:blue")
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax.set_title(
            f"final hydr={row.get('final_hydration_state', np.nan):.2f}, swell={row['final_swelling']:.2f}, comp={row.get('final_compaction_state', np.nan):.2f}, porosity={row['final_porosity']:.2f}, "
            f"hold-up={row['final_retained_water']:.2f} g"
        )

    plt.tight_layout()
    out_path = os.path.join(base_out_dir, "swelling_examples.png")
    plt.savefig(out_path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------


 # Public API: one production routine, with an optional terminal-TDS residual.

# Public API: one production routine, with an optional terminal-TDS residual.
PARAMETER_NAMES = ["R0", "alpha_comp", "alpha_ero", "a_visc", "b_visc", "k_extr",
                   "t_off", "wetting_tau", "hold_capacity_ratio", "beta_swelling"]
TDS_WEIGHT = 25.0

def _public_bounds(t, q):
    onset = estimate_onset_time(t, q, threshold=0.1, fallback=5.0)
    return [(R0_BOUND_LOW, R0_BOUND_HIGH), (0.0, 0.5), (0.1, 10.0),
            (0.0, 10.0), (0.1, 3.0), (0.1, 5.0),
            (max(0.0, onset - 1.5), onset + 1.5), (0.5, 8.0),
            (0.05, 0.60), (0.0, 0.18)]

def _public_joint_objective(x, t, p, q_ref, w_ref, dose, max_solubles, onset, tds_target):
    sim = simulate_numba_core(t, p, np.asarray(x), dose, max_solubles, 1.0)
    terms = _objective_terms(t=t, p=p, q_ref=q_ref, w_ref=w_ref,
                             q_sim=sim[0], w_sim=sim[2], R_sim=sim[5],
                             dose=dose, t_onset_ref=onset, t_off=float(x[6]))
    return float(terms["obj_loss_total"] + TDS_WEIGHT * (float(sim[3][-1]) - tds_target) ** 2)

def fit(csv_path, dose_g=18.0, tds_percent=None, efficiency_factor=0.70, seed=42):
    t, p, q_ref, w_ref, metadata = load_and_prep_data(str(csv_path))
    dose, reported, source = _resolve_effective_dose_g(metadata, dose_g)
    onset = estimate_onset_time(t, q_ref, threshold=0.1, fallback=5.0)
    bounds = _public_bounds(t, q_ref)
    max_solubles = MAX_SOLUBLES_FRACTION * dose * efficiency_factor
    if tds_percent is None:
        result = fit_single_shot_2stage(str(csv_path), dose_g=dose_g, efficiency_factor=efficiency_factor)
        return {"loss": float(result["loss"]), "flow_r2": float(result["r_squared"]),
                "predicted_tds": float(result["tds"][-1]), "parameters": result["params"],
                "trajectory": result}
    args = (t, p, q_ref, w_ref, dose, max_solubles, onset, float(tds_percent))
    de = differential_evolution(_public_joint_objective, bounds, args=args,
                                strategy="best1bin", maxiter=60, popsize=15,
                                tol=0.02, seed=seed, polish=False)
    local = minimize(_public_joint_objective, de.x, args=args, method="L-BFGS-B",
                     bounds=bounds, tol=1e-6)
    sim = simulate_numba_core(t, p, local.x, dose, max_solubles, 1.0)
    q_sim, w_sim, tds_sim = sim[0], sim[2], sim[3]
    r2 = 1.0 - float(np.sum((q_ref-q_sim)**2)) / max(float(np.sum((q_ref-q_ref.mean())**2)), 1e-12)
    return {"loss": float(local.fun), "flow_r2": float(r2),
            "predicted_tds": float(tds_sim[-1]),
            "parameters": {n: float(v) for n, v in zip(PARAMETER_NAMES, local.x)},
            "trajectory": {"t": t, "p": p, "q_ref": q_ref, "w_ref": w_ref,
                           "q_sim": q_sim, "w_sim": sim[2], "tds": tds_sim,
                           "R_sim": sim[5], "hydration": sim[6], "swelling": sim[7],
                           "porosity": sim[8], "wet_factor": sim[9],
                           "R_intrinsic": sim[10], "retained_water": sim[11],
                           "compaction_state": sim[12], "erosion_gate": sim[13],
                           "erosion_state": sim[14], "erosion_relief": sim[15],
                           "dose_g_used": dose}}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--tds", type=float, default=None)
    parser.add_argument("--dose", type=float, default=18.0)
    parser.add_argument("--out", default="fit.json")
    args = parser.parse_args()
    result = fit(args.csv_path, dose_g=args.dose, tds_percent=args.tds)
    result.pop("trajectory", None)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=str)
    print(args.out)
