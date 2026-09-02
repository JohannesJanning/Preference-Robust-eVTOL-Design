#!/usr/bin/env python3
"""Sweep cost/GWP utility weights on the profit-free simplex edge.

This script keeps the original optimization file untouched and instead runs a
separate design-space exploration where:

- utility profit weight is fixed at 0
- cost and GWP weights span the edge w_cost + w_gwp = 1.0
- the step size defaults to 0.1

For each optimization run, the script records the same key values that are
printed in the terminal by optimize_fixed_baseline.py, writes them to CSV, and
builds a Pareto-front figure for TOC_flight vs GWP_annual_ops.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models_jax.aerodynamics.ROC import gamma_from_roc
from src.models_jax.aerodynamics.drag import drag_calculation
from src.models_jax.momentum.T_climb_total import total_thrust_required_climb
from src.models_jax.momentum.T_cruise_total import total_thrust_required_cruise
from src.models_jax.momentum.T_hover_total import total_thrust_required_hover
from src.models_jax.momentum.T_prop import thrust_per_propeller
from src.optimizer.eVTOL_group_fixed_baseline import eVTOLGroupFixedBaseline
from src.parameters import model_parameters as parameters


@dataclass(frozen=True)
class DesignPoint:
    b: float
    c: float
    r_cruise: float
    r_hover: float
    V_cruise: float
    V_climb: float
    c_charge: float


CSV_FIELD_ORDER = [
    "w_cost",
    "w_gwp",
    "w_profit",
    "run_success",
    "run_error",
    "elapsed_s",
    "Utility",
    "Utility_Norm",
    "Cost_Utility",
    "GWP_Utility",
    "Profit_Utility_Value",
    "Cost_Utility_Norm",
    "GWP_Utility_Norm",
    "Profit_Utility_Norm",
    "Cost_Rating_Norm",
    "GWP_Rating_Norm",
    "Profit_Rating_Norm",
    "Profit_Utility",
    "cons_comp.c1",
    "AR",
    "vertiport_span",
    "MTOM",
    "CL_cruise",
    "CL_climb",
    "gamma_comp.gamma_deg",
    "SPL_hover",
    "b",
    "c",
    "r_cruise",
    "r_hover",
    "V_cruise",
    "V_climb",
    "rho_bat",
    "c_charge",
    "gamma_climb_deg",
    "ROC_target_mps",
    "D_climb_N",
    "D_cruise_N",
    "T_req_total_climb_N",
    "T_req_total_cruise_N",
    "T_req_total_hover_N",
    "T_req_prop_climb_N",
    "T_req_prop_cruise_N",
    "T_req_prop_hover_N",
    "Lift_to_Drag_Cruise",
    "Lift_to_Drag_Climb",
    "Wing_Loading_kg_m2",
    "Wing_Loading_N_m2",
    "Disk_Loading_Hover_kg_m2",
    "Disk_Loading_Hover_N_m2",
    "Disk_Loading_Cruise_kg_m2",
    "Disk_Loading_Cruise_N_m2",
    "Power_to_Weight_Hover_W_kg",
    "Power_to_Weight_Hover_hp_lb",
    "Power_to_Weight_Climb_W_kg",
    "Power_to_Weight_Climb_hp_lb",
    "Power_to_Weight_Cruise_W_kg",
    "Power_to_Weight_Cruise_hp_lb",
    "Empty_Mass_Fraction",
    "Battery_Mass_Fraction",
    "P_req_total_hover",
    "P_req_total_climb",
    "P_req_total_cruise",
    "E_total_req",
    "E_trip",
    "E_hover",
    "E_climb",
    "E_reserve",
    "t_cruise",
    "t_trip",
    "C_rate_hover",
    "C_rate_climb",
    "C_rate_cruise",
    "C_rate_avg",
    "DOD",
    "n_battery_lifecycle",
    "MTOM_est",
    "m_battery",
    "m_empty",
    "RPM_hover",
    "FC_a",
    "FoM",
    "FoM_time_rating",
    "FoM_co2_rating",
    "FoM_energy_rating",
    "FoM_cost_rating",
    "TOC_flight",
    "GWP_flight",
    "GWP_annual_ops",
    "Annual_Profit",
    "pareto_front",
]


def set_utility_weights(cost_w: float, gwp_w: float, profit_w: float) -> None:
    parameters.utility_cost_weight = float(cost_w)
    parameters.utility_gwp_weight = float(gwp_w)
    parameters.utility_profit_weight = float(profit_w)


def simplex_edge(step: float) -> list[tuple[float, float, float]]:
    n = int(round(1.0 / step))
    if not np.isclose(step * n, 1.0):
        raise ValueError("--step must divide 1.0 exactly (e.g. 0.2, 0.1)")

    weights: list[tuple[float, float, float]] = []
    for i in range(n + 1):
        w_cost = i * step
        w_gwp = 1.0 - w_cost
        weights.append((float(w_cost), float(w_gwp), 0.0))
    return weights


def build_problem(with_driver: bool) -> om.Problem:
    prob = om.Problem(model=eVTOLGroupFixedBaseline(parameters=parameters), reports=False)

    iv = om.IndepVarComp()
    iv.add_output("b", val=15.0)
    iv.add_output("c", val=1.0)
    iv.add_output("r_cruise", val=1.0)
    iv.add_output("r_hover", val=1.0)
    iv.add_output("V_cruise", val=60.0)
    iv.add_output("V_climb", val=60.0)
    iv.add_output("rho_bat", val=parameters.rho_bat)
    iv.add_output("c_charge", val=2.0)
    prob.model.add_subsystem("iv", iv, promotes=["*"])

    if with_driver:
        prob.model.add_design_var("b", lower=6.0, upper=15.0, ref0=6.0, ref=15.0)
        prob.model.add_design_var("c", lower=1.0, upper=2.5, ref0=1.0, ref=2.5)
        prob.model.add_design_var("r_cruise", lower=0.7, upper=1.2, ref0=0.6, ref=1.2)
        prob.model.add_design_var("r_hover", lower=0.6, upper=1.9, ref0=0.6, ref=1.3)
        prob.model.add_design_var("V_cruise", lower=40.0, upper=129.0, ref0=40.0, ref=129.0)
        prob.model.add_design_var("V_climb", lower=40.0, upper=129.0, ref0=40.0, ref=129.0)
        prob.model.add_design_var("c_charge", lower=1.0, upper=4.0, ref0=1.0, ref=4.0)

    prob.model.add_constraint("AR", lower=6.0, upper=10.0, ref=8.0)

    cons_comp = om.ExecComp("c1 = b - rotor_spacing", b=15.0, rotor_spacing=1.0)
    prob.model.add_subsystem("cons_comp", cons_comp)
    prob.model.connect("rotor_spacing", "cons_comp.rotor_spacing")
    prob.model.connect("b", "cons_comp.b")

    gamma_comp = om.ExecComp(
        "gamma_deg = rad2deg * asin(roc / V_climb)",
        gamma_deg=5.0,
        V_climb=60.0,
        roc=float(parameters.roc_climb_target),
        rad2deg=57.29577951308232,
    )
    prob.model.add_subsystem("gamma_comp", gamma_comp)
    prob.model.connect("V_climb", "gamma_comp.V_climb")

    prob.model.add_constraint("cons_comp.c1", lower=0.0, ref=1.0)
    prob.model.add_constraint("vertiport_span", upper=15.0, ref=1.0)
    prob.model.add_constraint("MTOM", upper=3750.0, ref=2000.0)
    prob.model.add_constraint("CL_cruise", lower=0, upper=0.7, ref=1.0)
    prob.model.add_constraint("CL_climb", lower=0.0, upper=1.2, ref=1.0)
    prob.model.add_constraint("gamma_comp.gamma_deg", lower=5.0, upper=15.0, ref=10.0)
    prob.model.add_constraint("SPL_hover", upper=77.0, ref=100.0)

    prob.model.add_objective("Utility_Norm", ref=-1.0)

    if with_driver:
        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options["optimizer"] = "SLSQP"
        prob.driver.options["tol"] = 1e-6
        prob.driver.options["disp"] = False
        if hasattr(prob.driver, "declare_coloring"):
            prob.driver.declare_coloring()

    return prob


def sval(prob: om.Problem, name: str) -> float | None:
    try:
        return float(np.asarray(prob.get_val(name)).reshape(-1)[0])
    except Exception:
        return None


def _fmt(val: Any) -> str:
    if val is None:
        return "None"
    arr = np.asarray(val)
    if arr.size == 1:
        try:
            return f"{float(arr.item()):.6g}"
        except Exception:
            return str(arr.item())
    return np.array2string(arr, precision=6, suppress_small=True, threshold=40)


def _value_or_nan(prob: om.Problem, name: str) -> float:
    value = sval(prob, name)
    return float("nan") if value is None else float(value)


def _safe_compute_thrusts(prob: om.Problem) -> dict[str, float]:
    try:
        b = sval(prob, "b")
        c = sval(prob, "c")
        cd_cr = sval(prob, "CD_cruise")
        cd_cl = sval(prob, "CD_climb")
        v_cr = sval(prob, "V_cruise")
        v_cl = sval(prob, "V_climb")
        mtom = sval(prob, "MTOM")
        if any(v is None for v in [b, c, cd_cr, cd_cl, v_cr, v_cl, mtom]):
            return {}

        gamma_deg = float(np.asarray(gamma_from_roc(parameters.roc_climb_target, v_cl)).item())
        d_climb = float(np.asarray(drag_calculation(parameters.rho, v_cl, c, b, cd_cl)).item())
        d_cruise = float(np.asarray(drag_calculation(parameters.rho, v_cr, c, b, cd_cr)).item())

        t_total_climb = float(np.asarray(total_thrust_required_climb(d_climb, mtom, parameters.g, gamma_deg)).item())
        t_total_cruise = float(np.asarray(total_thrust_required_cruise(d_cruise)).item())
        t_total_hover = float(np.asarray(total_thrust_required_hover(mtom, parameters.g)).item())

        t_prop_climb = float(np.asarray(thrust_per_propeller(t_total_climb, parameters.n_prop_hor)).item())
        t_prop_cruise = float(np.asarray(thrust_per_propeller(t_total_cruise, parameters.n_prop_hor)).item())
        t_prop_hover = float(np.asarray(thrust_per_propeller(t_total_hover, parameters.n_prop_vert)).item())

        return {
            "gamma_climb_deg": gamma_deg,
            "ROC_target_mps": float(parameters.roc_climb_target),
            "D_climb_N": d_climb,
            "D_cruise_N": d_cruise,
            "T_req_total_climb_N": t_total_climb,
            "T_req_total_cruise_N": t_total_cruise,
            "T_req_total_hover_N": t_total_hover,
            "T_req_prop_climb_N": t_prop_climb,
            "T_req_prop_cruise_N": t_prop_cruise,
            "T_req_prop_hover_N": t_prop_hover,
        }
    except Exception:
        return {}


def _safe_compute_sizing_metrics(prob: om.Problem) -> dict[str, float]:
    try:
        b = sval(prob, "b")
        c = sval(prob, "c")
        mtom = sval(prob, "MTOM")
        r_hover = sval(prob, "r_hover")
        r_cruise = sval(prob, "r_cruise")

        p_hover = sval(prob, "P_req_total_hover")
        p_climb = sval(prob, "P_req_total_climb")
        p_cruise = sval(prob, "P_req_total_cruise")

        cl_cruise = sval(prob, "CL_cruise")
        cd_cruise = sval(prob, "CD_cruise")
        cl_climb = sval(prob, "CL_climb")
        cd_climb = sval(prob, "CD_climb")

        m_empty = sval(prob, "m_empty")
        m_battery = sval(prob, "m_battery")

        if any(v is None for v in [
            b, c, mtom, r_hover, r_cruise,
            p_hover, p_climb, p_cruise,
            cl_cruise, cd_cruise, cl_climb, cd_climb,
            m_empty, m_battery,
        ]):
            return {}

        ld_cruise = cl_cruise / max(cd_cruise, 1e-12)
        ld_climb = cl_climb / max(cd_climb, 1e-12)

        s_ref = max(b * c, 1e-12)
        wing_loading_kg_m2 = mtom / s_ref
        wing_loading_n_m2 = wing_loading_kg_m2 * parameters.g

        a_disk_hover_total = max(float(parameters.n_prop_vert) * np.pi * r_hover**2, 1e-12)
        a_disk_cruise_total = max(float(parameters.n_prop_hor) * np.pi * r_cruise**2, 1e-12)

        disk_loading_hover_kg_m2 = mtom / a_disk_hover_total
        disk_loading_cruise_kg_m2 = mtom / a_disk_cruise_total
        disk_loading_hover_n_m2 = mtom * parameters.g / a_disk_hover_total
        disk_loading_cruise_n_m2 = mtom * parameters.g / a_disk_cruise_total

        pw_hover_w_kg = p_hover / max(mtom, 1e-12)
        pw_climb_w_kg = p_climb / max(mtom, 1e-12)
        pw_cruise_w_kg = p_cruise / max(mtom, 1e-12)

        pw_hover_hp_lb = pw_hover_w_kg * 0.00134102
        pw_climb_hp_lb = pw_climb_w_kg * 0.00134102
        pw_cruise_hp_lb = pw_cruise_w_kg * 0.00134102

        empty_mass_fraction = m_empty / max(mtom, 1e-12)
        battery_mass_fraction = m_battery / max(mtom, 1e-12)

        return {
            "Lift_to_Drag_Cruise": ld_cruise,
            "Lift_to_Drag_Climb": ld_climb,
            "Wing_Loading_kg_m2": wing_loading_kg_m2,
            "Wing_Loading_N_m2": wing_loading_n_m2,
            "Disk_Loading_Hover_kg_m2": disk_loading_hover_kg_m2,
            "Disk_Loading_Hover_N_m2": disk_loading_hover_n_m2,
            "Disk_Loading_Cruise_kg_m2": disk_loading_cruise_kg_m2,
            "Disk_Loading_Cruise_N_m2": disk_loading_cruise_n_m2,
            "Power_to_Weight_Hover_W_kg": pw_hover_w_kg,
            "Power_to_Weight_Hover_hp_lb": pw_hover_hp_lb,
            "Power_to_Weight_Climb_W_kg": pw_climb_w_kg,
            "Power_to_Weight_Climb_hp_lb": pw_climb_hp_lb,
            "Power_to_Weight_Cruise_W_kg": pw_cruise_w_kg,
            "Power_to_Weight_Cruise_hp_lb": pw_cruise_hp_lb,
            "Empty_Mass_Fraction": empty_mass_fraction,
            "Battery_Mass_Fraction": battery_mass_fraction,
        }
    except Exception:
        return {}


def _collect_row(
    prob: om.Problem,
    weights: tuple[float, float, float],
    elapsed_s: float,
    run_success: bool,
    run_error: str | None,
) -> dict[str, Any]:
    thrusts = _safe_compute_thrusts(prob)
    sizing_metrics = _safe_compute_sizing_metrics(prob)

    row: dict[str, Any] = {
        "w_cost": float(weights[0]),
        "w_gwp": float(weights[1]),
        "w_profit": float(weights[2]),
        "run_success": bool(run_success),
        "run_error": "" if run_error is None else str(run_error),
        "elapsed_s": float(elapsed_s),
        "Utility": _value_or_nan(prob, "Utility"),
        "Utility_Norm": _value_or_nan(prob, "Utility_Norm"),
        "Cost_Utility": _value_or_nan(prob, "Cost_Utility"),
        "GWP_Utility": _value_or_nan(prob, "GWP_Utility"),
        "Profit_Utility_Value": _value_or_nan(prob, "Profit_Utility_Value"),
        "Cost_Utility_Norm": _value_or_nan(prob, "Cost_Utility_Norm"),
        "GWP_Utility_Norm": _value_or_nan(prob, "GWP_Utility_Norm"),
        "Profit_Utility_Norm": _value_or_nan(prob, "Profit_Utility_Norm"),
        "Cost_Rating_Norm": _value_or_nan(prob, "Cost_Rating_Norm"),
        "GWP_Rating_Norm": _value_or_nan(prob, "GWP_Rating_Norm"),
        "Profit_Rating_Norm": _value_or_nan(prob, "Profit_Rating_Norm"),
        "Profit_Utility": _value_or_nan(prob, "Profit_Utility"),
        "cons_comp.c1": _value_or_nan(prob, "cons_comp.c1"),
        "AR": _value_or_nan(prob, "AR"),
        "vertiport_span": _value_or_nan(prob, "vertiport_span"),
        "MTOM": _value_or_nan(prob, "MTOM"),
        "CL_cruise": _value_or_nan(prob, "CL_cruise"),
        "CL_climb": _value_or_nan(prob, "CL_climb"),
        "gamma_comp.gamma_deg": _value_or_nan(prob, "gamma_comp.gamma_deg"),
        "SPL_hover": _value_or_nan(prob, "SPL_hover"),
        "b": _value_or_nan(prob, "b"),
        "c": _value_or_nan(prob, "c"),
        "r_cruise": _value_or_nan(prob, "r_cruise"),
        "r_hover": _value_or_nan(prob, "r_hover"),
        "V_cruise": _value_or_nan(prob, "V_cruise"),
        "V_climb": _value_or_nan(prob, "V_climb"),
        "rho_bat": _value_or_nan(prob, "rho_bat"),
        "c_charge": _value_or_nan(prob, "c_charge"),
        "gamma_climb_deg": thrusts.get("gamma_climb_deg", float("nan")),
        "ROC_target_mps": thrusts.get("ROC_target_mps", float("nan")),
        "D_climb_N": thrusts.get("D_climb_N", float("nan")),
        "D_cruise_N": thrusts.get("D_cruise_N", float("nan")),
        "T_req_total_climb_N": thrusts.get("T_req_total_climb_N", float("nan")),
        "T_req_total_cruise_N": thrusts.get("T_req_total_cruise_N", float("nan")),
        "T_req_total_hover_N": thrusts.get("T_req_total_hover_N", float("nan")),
        "T_req_prop_climb_N": thrusts.get("T_req_prop_climb_N", float("nan")),
        "T_req_prop_cruise_N": thrusts.get("T_req_prop_cruise_N", float("nan")),
        "T_req_prop_hover_N": thrusts.get("T_req_prop_hover_N", float("nan")),
        "Lift_to_Drag_Cruise": sizing_metrics.get("Lift_to_Drag_Cruise", float("nan")),
        "Lift_to_Drag_Climb": sizing_metrics.get("Lift_to_Drag_Climb", float("nan")),
        "Wing_Loading_kg_m2": sizing_metrics.get("Wing_Loading_kg_m2", float("nan")),
        "Wing_Loading_N_m2": sizing_metrics.get("Wing_Loading_N_m2", float("nan")),
        "Disk_Loading_Hover_kg_m2": sizing_metrics.get("Disk_Loading_Hover_kg_m2", float("nan")),
        "Disk_Loading_Hover_N_m2": sizing_metrics.get("Disk_Loading_Hover_N_m2", float("nan")),
        "Disk_Loading_Cruise_kg_m2": sizing_metrics.get("Disk_Loading_Cruise_kg_m2", float("nan")),
        "Disk_Loading_Cruise_N_m2": sizing_metrics.get("Disk_Loading_Cruise_N_m2", float("nan")),
        "Power_to_Weight_Hover_W_kg": sizing_metrics.get("Power_to_Weight_Hover_W_kg", float("nan")),
        "Power_to_Weight_Hover_hp_lb": sizing_metrics.get("Power_to_Weight_Hover_hp_lb", float("nan")),
        "Power_to_Weight_Climb_W_kg": sizing_metrics.get("Power_to_Weight_Climb_W_kg", float("nan")),
        "Power_to_Weight_Climb_hp_lb": sizing_metrics.get("Power_to_Weight_Climb_hp_lb", float("nan")),
        "Power_to_Weight_Cruise_W_kg": sizing_metrics.get("Power_to_Weight_Cruise_W_kg", float("nan")),
        "Power_to_Weight_Cruise_hp_lb": sizing_metrics.get("Power_to_Weight_Cruise_hp_lb", float("nan")),
        "Empty_Mass_Fraction": sizing_metrics.get("Empty_Mass_Fraction", float("nan")),
        "Battery_Mass_Fraction": sizing_metrics.get("Battery_Mass_Fraction", float("nan")),
        "P_req_total_hover": _value_or_nan(prob, "P_req_total_hover"),
        "P_req_total_climb": _value_or_nan(prob, "P_req_total_climb"),
        "P_req_total_cruise": _value_or_nan(prob, "P_req_total_cruise"),
        "E_total_req": _value_or_nan(prob, "E_total_req"),
        "E_trip": _value_or_nan(prob, "E_trip"),
        "E_hover": _value_or_nan(prob, "E_hover"),
        "E_climb": _value_or_nan(prob, "E_climb"),
        "E_reserve": _value_or_nan(prob, "E_reserve"),
        "t_cruise": _value_or_nan(prob, "t_cruise"),
        "t_trip": _value_or_nan(prob, "t_trip"),
        "C_rate_hover": _value_or_nan(prob, "C_rate_hover"),
        "C_rate_climb": _value_or_nan(prob, "C_rate_climb"),
        "C_rate_cruise": _value_or_nan(prob, "C_rate_cruise"),
        "C_rate_avg": _value_or_nan(prob, "C_rate_avg"),
        "DOD": _value_or_nan(prob, "DOD"),
        "n_battery_lifecycle": _value_or_nan(prob, "n_battery_lifecycle"),
        "MTOM_est": _value_or_nan(prob, "MTOM_est"),
        "m_battery": _value_or_nan(prob, "m_battery"),
        "m_empty": _value_or_nan(prob, "m_empty"),
        "RPM_hover": _value_or_nan(prob, "RPM_hover"),
        "FC_a": _value_or_nan(prob, "FC_a"),
        "FoM": _value_or_nan(prob, "FoM"),
        "FoM_time_rating": _value_or_nan(prob, "FoM_time_rating"),
        "FoM_co2_rating": _value_or_nan(prob, "FoM_co2_rating"),
        "FoM_energy_rating": _value_or_nan(prob, "FoM_energy_rating"),
        "FoM_cost_rating": _value_or_nan(prob, "FoM_cost_rating"),
        "TOC_flight": _value_or_nan(prob, "TOC_flight"),
        "GWP_flight": _value_or_nan(prob, "GWP_flight"),
        "GWP_annual_ops": _value_or_nan(prob, "GWP_annual_ops"),
        "Annual_Profit": _value_or_nan(prob, "Annual_Profit"),
        "pareto_front": False,
    }
    return row


def _print_section(title: str, row: dict[str, Any], names: list[str]) -> None:
    print("-------------------------------")
    print(title)
    for name in names:
        print(f"{name}:", _fmt(row.get(name)))


def _print_run_summary(row: dict[str, Any]) -> None:
    print("-------------------------------")
    print("RUN SUMMARY")
    print("Objective Utility_Norm:", _fmt(row.get("Utility_Norm")))
    print(
        "Utility weights:",
        {
            "cost": float(row["w_cost"]),
            "gwp": float(row["w_gwp"]),
            "profit": float(row["w_profit"]),
        },
    )


def _print_derived_sections(row: dict[str, Any]) -> None:
    _print_section(
        "CONSTRAINT SNAPSHOT",
        row,
        ["cons_comp.c1", "AR", "vertiport_span", "MTOM", "CL_cruise", "CL_climb", "gamma_comp.gamma_deg", "SPL_hover"],
    )

    _print_section(
        "DESIGN VARIABLES",
        row,
        ["b", "c", "r_cruise", "r_hover", "V_cruise", "V_climb", "rho_bat", "c_charge"],
    )

    if all(np.isfinite(row.get(k, float("nan"))) for k in ["gamma_climb_deg", "D_climb_N", "D_cruise_N"]):
        print("-------------------------------")
        print("THRUST + DRAG (derived)")
        for key in [
            "gamma_climb_deg",
            "ROC_target_mps",
            "D_climb_N",
            "D_cruise_N",
            "T_req_total_climb_N",
            "T_req_total_cruise_N",
            "T_req_total_hover_N",
            "T_req_prop_climb_N",
            "T_req_prop_cruise_N",
            "T_req_prop_hover_N",
        ]:
            print(f"{key}: {_fmt(row.get(key))}")

    if all(np.isfinite(row.get(k, float("nan"))) for k in ["Lift_to_Drag_Cruise", "Wing_Loading_kg_m2"]):
        print("-------------------------------")
        print("SIZING METRICS (derived)")
        for key in [
            "Lift_to_Drag_Cruise",
            "Lift_to_Drag_Climb",
            "Wing_Loading_kg_m2",
            "Wing_Loading_N_m2",
            "Disk_Loading_Hover_kg_m2",
            "Disk_Loading_Hover_N_m2",
            "Disk_Loading_Cruise_kg_m2",
            "Disk_Loading_Cruise_N_m2",
            "Power_to_Weight_Hover_W_kg",
            "Power_to_Weight_Hover_hp_lb",
            "Power_to_Weight_Climb_W_kg",
            "Power_to_Weight_Climb_hp_lb",
            "Power_to_Weight_Cruise_W_kg",
            "Power_to_Weight_Cruise_hp_lb",
            "Empty_Mass_Fraction",
            "Battery_Mass_Fraction",
        ]:
            print(f"{key}: {_fmt(row.get(key))}")

    _print_section(
        "AERODYNAMICS",
        row,
        ["CL_cruise", "CD_cruise", "CL_climb", "CD_climb", "V_climb_hor"],
    )

    _print_section(
        "POWER",
        row,
        ["P_req_total_hover", "P_req_total_climb", "P_req_total_cruise"],
    )

    _print_section(
        "ENERGY + TIMES",
        row,
        ["E_total_req", "E_trip", "E_hover", "E_climb", "E_reserve"],
    )

    _print_section(
        "BATTERY C-RATES",
        row,
        ["C_rate_hover", "C_rate_climb", "C_rate_cruise", "C_rate_avg", "DOD", "n_battery_lifecycle"],
    )

    _print_section(
        "MASS",
        row,
        ["MTOM", "MTOM_est", "m_battery", "m_empty"],
    )

    _print_section(
        "NOISE",
        row,
        ["SPL_hover", "RPM_hover"],
    )

    _print_section(
        "OPS + TRANSPORT",
        row,
        ["FC_a", "FoM", "FoM_time_rating", "FoM_co2_rating", "FoM_energy_rating", "FoM_cost_rating"],
    )

    _print_section(
        "ECONOMICS + GWP",
        row,
        ["TOC_flight", "GWP_flight", "GWP_annual_ops"],
    )

    if np.isfinite(row.get("Annual_Profit", float("nan"))):
        print(f"Annual_Profit: {row['Annual_Profit']:,.2f}")

    _print_section(
        "UTILITY",
        row,
        [
            "Utility",
            "Utility_Norm",
            "Cost_Utility",
            "GWP_Utility",
            "Profit_Utility_Value",
            "Cost_Utility_Norm",
            "GWP_Utility_Norm",
            "Profit_Utility_Norm",
            "Cost_Rating_Norm",
            "GWP_Rating_Norm",
            "Profit_Rating_Norm",
            "Profit_Utility",
        ],
    )


def optimize_case(weights: tuple[float, float, float], x0: DesignPoint) -> dict[str, Any]:
    set_utility_weights(*weights)
    prob = build_problem(with_driver=True)
    prob.setup()

    prob.set_val("b", x0.b)
    prob.set_val("c", x0.c)
    prob.set_val("r_cruise", x0.r_cruise)
    prob.set_val("r_hover", x0.r_hover)
    prob.set_val("V_cruise", x0.V_cruise)
    prob.set_val("V_climb", x0.V_climb)
    prob.set_val("rho_bat", parameters.rho_bat)
    prob.set_val("c_charge", x0.c_charge)

    run_error: str | None = None
    run_success = True
    t0 = time.time()
    try:
        prob.run_driver()
    except Exception as exc:
        run_success = False
        run_error = str(exc)
    elapsed_s = time.time() - t0

    row = _collect_row(prob, weights, elapsed_s, run_success, run_error)
    return row


def _is_pareto_efficient(rows: list[dict[str, Any]], x_key: str, y_key: str) -> np.ndarray:
    values = np.array([[float(r[x_key]), float(r[y_key])] for r in rows], dtype=float)
    efficient = np.ones(len(values), dtype=bool)
    for i, point in enumerate(values):
        if not efficient[i]:
            continue
        dominates = np.all(values <= point, axis=1) & np.any(values < point, axis=1)
        dominates[i] = False
        efficient[dominates] = False
    return efficient


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELD_ORDER)
        writer.writeheader()
        for row in rows:
            serializable = {key: row.get(key, "") for key in CSV_FIELD_ORDER}
            writer.writerow(serializable)


def plot_pareto_front(rows: list[dict[str, Any]], out_path: Path) -> None:
    finite_rows = [r for r in rows if np.isfinite(r.get("TOC_flight", float("nan"))) and np.isfinite(r.get("GWP_annual_ops", float("nan")))]
    if not finite_rows:
        raise ValueError("No finite rows available for Pareto plot")

    efficient = _is_pareto_efficient(finite_rows, "TOC_flight", "GWP_annual_ops")
    for row, is_front in zip(finite_rows, efficient):
        row["pareto_front"] = bool(is_front)

    front_rows = [row for row in finite_rows if row["pareto_front"]]
    front_rows.sort(key=lambda r: float(r["TOC_flight"]))

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cost = np.array([float(r["TOC_flight"]) for r in finite_rows], dtype=float)
    gwp = np.array([float(r["GWP_annual_ops"]) for r in finite_rows], dtype=float)
    wc = np.array([float(r["w_cost"]) for r in finite_rows], dtype=float)

    sc = ax.scatter(cost, gwp, c=wc, cmap="viridis", s=72, edgecolors="black", linewidths=0.35, alpha=0.9, label="Sweep runs")
    if len(front_rows) >= 2:
        fx = np.array([float(r["TOC_flight"]) for r in front_rows], dtype=float)
        fy = np.array([float(r["GWP_annual_ops"]) for r in front_rows], dtype=float)
        ax.plot(fx, fy, color="#d62728", linewidth=2.2, marker="o", markersize=5, label="Pareto front")

    for row in front_rows:
        ax.annotate(
            f"{row['w_cost']:.1f}/{row['w_gwp']:.1f}",
            (float(row["TOC_flight"]), float(row["GWP_annual_ops"])),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Pareto Front: Cost vs GWP Weight Sweep")
    ax.set_xlabel("TOC_flight")
    ax.set_ylabel("GWP_annual_ops")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("w_cost")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _default_output_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "results" / f"pareto_front_simplex_{ts}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep the fixed-baseline utility weights along the cost/GWP simplex edge.")
    parser.add_argument("--step", type=float, default=0.1, help="Step size along the weight edge (must divide 1 exactly).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to pareto front simplex/results/pareto_front_simplex_<timestamp>",
    )
    args = parser.parse_args()

    out_dir = _default_output_dir() if args.out_dir is None else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = simplex_edge(args.step)
    x0 = DesignPoint(b=9.0, c=1.0, r_cruise=1.0, r_hover=1.0, V_cruise=60.0, V_climb=60.0, c_charge=1.0)

    rows: list[dict[str, Any]] = []
    print(f"Running {len(weights)} cost/GWP weight optimizations with profit weight fixed at 0.0...")
    for idx, weight_triplet in enumerate(weights, start=1):
        row = optimize_case(weight_triplet, x0)
        rows.append(row)
        print(
            f"[{idx:02d}/{len(weights):02d}] w=({row['w_cost']:.1f},{row['w_gwp']:.1f},{row['w_profit']:.1f}) "
            f"success={row['run_success']} Utility_Norm={_fmt(row['Utility_Norm'])} TOC={_fmt(row['TOC_flight'])} GWP={_fmt(row['GWP_annual_ops'])}"
        )

    finite_rows = [r for r in rows if np.isfinite(r.get("TOC_flight", float("nan"))) and np.isfinite(r.get("GWP_annual_ops", float("nan")))]
    if finite_rows:
        efficient = _is_pareto_efficient(finite_rows, "TOC_flight", "GWP_annual_ops")
        for row, is_front in zip(finite_rows, efficient):
            row["pareto_front"] = bool(is_front)

    out_csv = out_dir / "pareto_front_simplex_results.csv"
    out_fig = out_dir / "pareto_front_simplex_figure.png"
    write_csv(rows, out_csv)
    plot_pareto_front(rows, out_fig)

    print("-------------------------------")
    print(f"Elapsed sweep output directory: {out_dir}")
    print(f"CSV written to: {out_csv}")
    print(f"Pareto figure written to: {out_fig}")


if __name__ == "__main__":
    main()