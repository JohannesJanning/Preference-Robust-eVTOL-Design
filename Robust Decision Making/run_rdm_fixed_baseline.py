import argparse
import csv
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import openmdao.api as om

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.parameters import model_parameters as parameters
from src.optimizer.eVTOL_group_fixed_baseline import eVTOLGroupFixedBaseline

# ---------------------------------------------------------------------------
# User-managed configuration
# ---------------------------------------------------------------------------
# Change this value to control the simplex grid resolution.
# Examples: 0.1, 0.2, 0.05, 0.25
STEP_SIZE = 0.5

# Bootstrap constants used only until measured timings become available.
INITIAL_OPT_TIME = 3.1
INITIAL_EVAL_TIME = 0.3


@dataclass
class RuntimeTracker:
    optimization_times: list[float] = field(default_factory=list)
    evaluation_times: list[float] = field(default_factory=list)

    @property
    def avg_optimization_time(self) -> float:
        if not self.optimization_times:
            return INITIAL_OPT_TIME
        return float(np.mean(self.optimization_times))

    @property
    def avg_evaluation_time(self) -> float:
        if not self.evaluation_times:
            return INITIAL_EVAL_TIME
        return float(np.mean(self.evaluation_times))

    def estimate_total_runtime(self, n_optimizations: int, n_evaluations: int) -> float:
        return (
            n_optimizations * self.avg_optimization_time
            + n_evaluations * self.avg_evaluation_time
        )


@dataclass
class DesignPoint:
    b: float
    c: float
    r_cruise: float
    r_hover: float
    V_cruise: float
    V_climb: float
    c_charge: float


def set_utility_weights(cost_w: float, gwp_w: float, profit_w: float) -> None:
    parameters.utility_cost_weight = float(cost_w)
    parameters.utility_gwp_weight = float(gwp_w)
    parameters.utility_profit_weight = float(profit_w)


def simplex_grid(step: float) -> list[tuple[float, float, float]]:
    n = int(round(1.0 / step))
    if not np.isclose(step * n, 1.0):
        raise ValueError('--step must divide 1.0 exactly (e.g. 0.25, 0.2, 0.1, 0.05)')

    weights: list[tuple[float, float, float]] = []
    for i in range(n + 1):
        w_cost = i * step
        for j in range(n + 1 - i):
            w_gwp = j * step
            w_profit = 1.0 - w_cost - w_gwp
            weights.append((float(w_cost), float(w_gwp), float(w_profit)))
    return weights


def build_problem(with_driver: bool) -> om.Problem:
    prob = om.Problem(model=eVTOLGroupFixedBaseline(parameters=parameters), reports=False)

    iv = om.IndepVarComp()
    iv.add_output('b', val=15.0)
    iv.add_output('c', val=1.0)
    iv.add_output('r_cruise', val=1.0)
    iv.add_output('r_hover', val=1.0)
    iv.add_output('V_cruise', val=60.0)
    iv.add_output('V_climb', val=60.0)
    iv.add_output('rho_bat', val=parameters.rho_bat)
    iv.add_output('c_charge', val=2.0)
    prob.model.add_subsystem('iv', iv, promotes=['*'])

    if with_driver:
        prob.model.add_design_var('b', lower=6.0, upper=15.0, ref0=6.0, ref=15.0)
        prob.model.add_design_var('c', lower=1.0, upper=2.5, ref0=1.0, ref=2.5)
        prob.model.add_design_var('r_cruise', lower=0.7, upper=1.2, ref0=0.6, ref=1.2)
        prob.model.add_design_var('r_hover', lower=0.6, upper=1.9, ref0=0.6, ref=1.3)
        prob.model.add_design_var('V_cruise', lower=40.0, upper=129.0, ref0=40.0, ref=129.0)
        prob.model.add_design_var('V_climb', lower=40.0, upper=129.0, ref0=40.0, ref=129.0)
        prob.model.add_design_var('c_charge', lower=1.0, upper=4.0, ref0=1.0, ref=4.0)

    try:
        prob.model.set_solver_print(level=0)
    except Exception:
        pass
    try:
        prob.model.nonlinear_solver.options['iprint'] = 0
    except Exception:
        pass
    try:
        prob.model.linear_solver.options['iprint'] = 0
    except Exception:
        pass

    cons_comp = om.ExecComp('c1 = b - rotor_spacing', b=15.0, rotor_spacing=1.0)
    prob.model.add_subsystem('cons_comp', cons_comp)
    prob.model.connect('rotor_spacing', 'cons_comp.rotor_spacing')
    prob.model.connect('b', 'cons_comp.b')

    gamma_comp = om.ExecComp(
        'gamma_deg = rad2deg * asin(roc / V_climb)',
        gamma_deg=5.0,
        V_climb=60.0,
        roc=float(parameters.roc_climb_target),
        rad2deg=57.29577951308232,
    )
    prob.model.add_subsystem('gamma_comp', gamma_comp)
    prob.model.connect('V_climb', 'gamma_comp.V_climb')

    if with_driver:
        prob.model.add_constraint('AR', lower=6.0, upper=10.0, ref=8.0)
        prob.model.add_constraint('cons_comp.c1', lower=0.0, ref=1.0)
        prob.model.add_constraint('vertiport_span', upper=15.0, ref=1.0)
        prob.model.add_constraint('MTOM', upper=3750.0, ref=2000.0)
        prob.model.add_constraint('CL_cruise', lower=0.0, upper=0.7, ref=1.0)
        prob.model.add_constraint('CL_climb', lower=0.0, upper=1.2, ref=1.0)
        prob.model.add_constraint('gamma_comp.gamma_deg', lower=5.0, upper=15.0, ref=10.0)
        prob.model.add_constraint('SPL_hover', upper=77.0, ref=100.0)
        prob.model.add_objective('Utility_Norm', ref=-1.0)

        prob.driver = om.ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-6
        prob.driver.options['disp'] = False

    return prob


def sval(prob: om.Problem, name: str) -> float:
    return float(np.asarray(prob.get_val(name)).reshape(-1)[0])


def safe_float(prob: om.Problem, name: str, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(np.asarray(prob.get_val(name)).reshape(-1)[0])
    except Exception:
        return default if default is not None else float('nan')


def run_model_quiet(prob: om.Problem) -> None:
    with redirect_stdout(io.StringIO()):
        prob.run_model()


def run_driver_quiet(prob: om.Problem) -> None:
    with redirect_stdout(io.StringIO()):
        prob.run_driver()


def optimize_for_weights(cost_w: float, gwp_w: float, profit_w: float, x0: DesignPoint) -> dict:
    set_utility_weights(cost_w, gwp_w, profit_w)
    prob = build_problem(with_driver=True)
    prob.setup()

    prob.set_val('b', x0.b)
    prob.set_val('c', x0.c)
    prob.set_val('r_cruise', x0.r_cruise)
    prob.set_val('r_hover', x0.r_hover)
    prob.set_val('V_cruise', x0.V_cruise)
    prob.set_val('V_climb', x0.V_climb)
    prob.set_val('rho_bat', parameters.rho_bat)
    prob.set_val('c_charge', x0.c_charge)

    run_driver_quiet(prob)

    return {
        'b': sval(prob, 'b'),
        'c': sval(prob, 'c'),
        'r_cruise': sval(prob, 'r_cruise'),
        'r_hover': sval(prob, 'r_hover'),
        'V_cruise': sval(prob, 'V_cruise'),
        'V_climb': sval(prob, 'V_climb'),
        'c_charge': sval(prob, 'c_charge'),
        'Utility': sval(prob, 'Utility'),
        'Utility_Norm': sval(prob, 'Utility_Norm'),
        'Profit_Utility': sval(prob, 'Profit_Utility'),
        'Annual_Profit': sval(prob, 'Annual_Profit'),
        'TOC_flight': sval(prob, 'TOC_flight'),
        'GWP_flight': sval(prob, 'GWP_flight'),
        'GWP_annual_ops': sval(prob, 'GWP_annual_ops'),
        'FoM': sval(prob, 'FoM'),
        'FoM_energy_rating': safe_float(prob, 'FoM_energy_rating', float('nan')),
        'SPL_hover': sval(prob, 'SPL_hover'),
        'MTOM': sval(prob, 'MTOM'),
        'm_battery': safe_float(prob, 'm_battery', float('nan')),
        'm_empty': safe_float(prob, 'm_empty', float('nan')),
        'P_req_total_hover': safe_float(prob, 'P_req_total_hover', float('nan')),
        'P_req_total_climb': safe_float(prob, 'P_req_total_climb', float('nan')),
        'P_req_total_cruise': safe_float(prob, 'P_req_total_cruise', float('nan')),
        'E_total_req': safe_float(prob, 'E_total_req', float('nan')),
        'E_trip': safe_float(prob, 'E_trip', float('nan')),
        'E_hover': safe_float(prob, 'E_hover', float('nan')),
        'E_climb': safe_float(prob, 'E_climb', float('nan')),
        'E_reserve': safe_float(prob, 'E_reserve', float('nan')),
        'C_rate_hover': safe_float(prob, 'C_rate_hover', float('nan')),
        'C_rate_climb': safe_float(prob, 'C_rate_climb', float('nan')),
        'C_rate_cruise': safe_float(prob, 'C_rate_cruise', float('nan')),
        'C_rate_avg': safe_float(prob, 'C_rate_avg', float('nan')),
        'DOD': safe_float(prob, 'DOD', float('nan')),
        'n_battery_lifecycle': safe_float(prob, 'n_battery_lifecycle', float('nan')),
        'RPM_hover': safe_float(prob, 'RPM_hover', float('nan')),
        'FC_a': safe_float(prob, 'FC_a', float('nan')),
        'FoM_time_rating': sval(prob, 'FoM_time_rating'),
        'FoM_co2_rating': sval(prob, 'FoM_co2_rating'),
        'FoM_cost_rating': sval(prob, 'FoM_cost_rating'),
        'Cost_Rating_Norm': sval(prob, 'Cost_Rating_Norm'),
        'GWP_Rating_Norm': sval(prob, 'GWP_Rating_Norm'),
        'Profit_Rating_Norm': sval(prob, 'Profit_Rating_Norm'),
        'AR': sval(prob, 'AR'),
        'CL_cruise': sval(prob, 'CL_cruise'),
        'CL_climb': sval(prob, 'CL_climb'),
        'gamma_comp.gamma_deg': sval(prob, 'gamma_comp.gamma_deg'),
    }


def evaluate_design_for_weights(cost_w: float, gwp_w: float, profit_w: float, x: DesignPoint) -> dict:
    set_utility_weights(cost_w, gwp_w, profit_w)
    prob = build_problem(with_driver=False)
    prob.setup()

    prob.set_val('b', x.b)
    prob.set_val('c', x.c)
    prob.set_val('r_cruise', x.r_cruise)
    prob.set_val('r_hover', x.r_hover)
    prob.set_val('V_cruise', x.V_cruise)
    prob.set_val('V_climb', x.V_climb)
    prob.set_val('rho_bat', parameters.rho_bat)
    prob.set_val('c_charge', x.c_charge)

    run_model_quiet(prob)

    return {
        'Utility': sval(prob, 'Utility'),
        'Utility_Norm': sval(prob, 'Utility_Norm'),
        'Profit_Utility': sval(prob, 'Profit_Utility'),
        'Annual_Profit': sval(prob, 'Annual_Profit'),
        'TOC_flight': sval(prob, 'TOC_flight'),
        'GWP_flight': sval(prob, 'GWP_flight'),
        'GWP_annual_ops': sval(prob, 'GWP_annual_ops'),
        'FoM': sval(prob, 'FoM'),
        'FoM_energy_rating': safe_float(prob, 'FoM_energy_rating', float('nan')),
        'SPL_hover': sval(prob, 'SPL_hover'),
        'MTOM': sval(prob, 'MTOM'),
        'm_battery': safe_float(prob, 'm_battery', float('nan')),
        'm_empty': safe_float(prob, 'm_empty', float('nan')),
        'P_req_total_hover': safe_float(prob, 'P_req_total_hover', float('nan')),
        'P_req_total_climb': safe_float(prob, 'P_req_total_climb', float('nan')),
        'P_req_total_cruise': safe_float(prob, 'P_req_total_cruise', float('nan')),
        'E_total_req': safe_float(prob, 'E_total_req', float('nan')),
        'E_trip': safe_float(prob, 'E_trip', float('nan')),
        'E_hover': safe_float(prob, 'E_hover', float('nan')),
        'E_climb': safe_float(prob, 'E_climb', float('nan')),
        'E_reserve': safe_float(prob, 'E_reserve', float('nan')),
        'C_rate_hover': safe_float(prob, 'C_rate_hover', float('nan')),
        'C_rate_climb': safe_float(prob, 'C_rate_climb', float('nan')),
        'C_rate_cruise': safe_float(prob, 'C_rate_cruise', float('nan')),
        'C_rate_avg': safe_float(prob, 'C_rate_avg', float('nan')),
        'DOD': safe_float(prob, 'DOD', float('nan')),
        'n_battery_lifecycle': safe_float(prob, 'n_battery_lifecycle', float('nan')),
        'RPM_hover': safe_float(prob, 'RPM_hover', float('nan')),
        'FC_a': safe_float(prob, 'FC_a', float('nan')),
        'FoM_time_rating': sval(prob, 'FoM_time_rating'),
        'FoM_co2_rating': sval(prob, 'FoM_co2_rating'),
        'FoM_cost_rating': sval(prob, 'FoM_cost_rating'),
        'Cost_Rating_Norm': sval(prob, 'Cost_Rating_Norm'),
        'GWP_Rating_Norm': sval(prob, 'GWP_Rating_Norm'),
        'Profit_Rating_Norm': sval(prob, 'Profit_Rating_Norm'),
        'AR': sval(prob, 'AR'),
        'CL_cruise': sval(prob, 'CL_cruise'),
        'CL_climb': sval(prob, 'CL_climb'),
        'gamma_comp.gamma_deg': sval(prob, 'gamma_comp.gamma_deg'),
    }


def evaluate_design_components(x: DesignPoint) -> dict:
    prob = build_problem(with_driver=False)
    prob.setup()

    prob.set_val('b', x.b)
    prob.set_val('c', x.c)
    prob.set_val('r_cruise', x.r_cruise)
    prob.set_val('r_hover', x.r_hover)
    prob.set_val('V_cruise', x.V_cruise)
    prob.set_val('V_climb', x.V_climb)
    prob.set_val('rho_bat', parameters.rho_bat)
    prob.set_val('c_charge', x.c_charge)

    run_model_quiet(prob)

    return {
        'Utility': sval(prob, 'Utility'),
        'Utility_Norm': sval(prob, 'Utility_Norm'),
        'Profit_Utility': sval(prob, 'Profit_Utility'),
        'Annual_Profit': sval(prob, 'Annual_Profit'),
        'TOC_flight': sval(prob, 'TOC_flight'),
        'GWP_flight': sval(prob, 'GWP_flight'),
        'GWP_annual_ops': sval(prob, 'GWP_annual_ops'),
        'FoM': sval(prob, 'FoM'),
        'FoM_energy_rating': safe_float(prob, 'FoM_energy_rating', float('nan')),
        'SPL_hover': sval(prob, 'SPL_hover'),
        'MTOM': sval(prob, 'MTOM'),
        'm_battery': safe_float(prob, 'm_battery', float('nan')),
        'm_empty': safe_float(prob, 'm_empty', float('nan')),
        'P_req_total_hover': safe_float(prob, 'P_req_total_hover', float('nan')),
        'P_req_total_climb': safe_float(prob, 'P_req_total_climb', float('nan')),
        'P_req_total_cruise': safe_float(prob, 'P_req_total_cruise', float('nan')),
        'E_total_req': safe_float(prob, 'E_total_req', float('nan')),
        'E_trip': safe_float(prob, 'E_trip', float('nan')),
        'E_hover': safe_float(prob, 'E_hover', float('nan')),
        'E_climb': safe_float(prob, 'E_climb', float('nan')),
        'E_reserve': safe_float(prob, 'E_reserve', float('nan')),
        'C_rate_hover': safe_float(prob, 'C_rate_hover', float('nan')),
        'C_rate_climb': safe_float(prob, 'C_rate_climb', float('nan')),
        'C_rate_cruise': safe_float(prob, 'C_rate_cruise', float('nan')),
        'C_rate_avg': safe_float(prob, 'C_rate_avg', float('nan')),
        'DOD': safe_float(prob, 'DOD', float('nan')),
        'n_battery_lifecycle': safe_float(prob, 'n_battery_lifecycle', float('nan')),
        'RPM_hover': safe_float(prob, 'RPM_hover', float('nan')),
        'FC_a': safe_float(prob, 'FC_a', float('nan')),
        'FoM_time_rating': sval(prob, 'FoM_time_rating'),
        'FoM_co2_rating': sval(prob, 'FoM_co2_rating'),
        'FoM_cost_rating': sval(prob, 'FoM_cost_rating'),
        'Cost_Rating_Norm': sval(prob, 'Cost_Rating_Norm'),
        'GWP_Rating_Norm': sval(prob, 'GWP_Rating_Norm'),
        'Profit_Rating_Norm': sval(prob, 'Profit_Rating_Norm'),
        'AR': sval(prob, 'AR'),
        'CL_cruise': sval(prob, 'CL_cruise'),
        'CL_climb': sval(prob, 'CL_climb'),
        'gamma_comp.gamma_deg': sval(prob, 'gamma_comp.gamma_deg'),
    }


def reconstruct_utility_from_components(components: dict[str, Any], cost_w: float, gwp_w: float, profit_w: float) -> float:
    return (
        float(cost_w) * float(components['Cost_Rating_Norm'])
        + float(gwp_w) * float(components['GWP_Rating_Norm'])
        + float(profit_w) * float(components['Profit_Rating_Norm'])
    )


def evaluate_design_static(x: DesignPoint) -> dict:
    return evaluate_design_for_weights(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, x)


def to_design(row: dict) -> DesignPoint:
    return DesignPoint(
        b=float(row['b']),
        c=float(row['c']),
        r_cruise=float(row['r_cruise']),
        r_hover=float(row['r_hover']),
        V_cruise=float(row['V_cruise']),
        V_climb=float(row['V_climb']),
        c_charge=float(row['c_charge']),
    )


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def cand_name(index: int) -> str:
    return f'cand_{index:02d}'


def scenario_name(index: int) -> str:
    return f'scenario_{index}'


def _matrix_row(base: dict[str, Any], values: dict[str, float], labels: list[str]) -> dict[str, Any]:
    row = dict(base)
    for label in labels:
        row[label] = float(values[label])
    return row


def next_run_number(results_root: str) -> int:
    if not os.path.isdir(results_root):
        return 1

    max_run = 0
    for name in os.listdir(results_root):
        match = re.match(r'^run\s+(\d+)\s*-\s*', name, flags=re.IGNORECASE)
        if match:
            max_run = max(max_run, int(match.group(1)))
    return max_run + 1


def create_run_dir(base_results_root: str) -> str:
    run_num = next_run_number(base_results_root)
    now = datetime.now()
    run_label = f'run {run_num:02d} - {now.strftime("%H:%M")} - {now.strftime("%d.%m.%Y")}'
    out_dir = os.path.join(base_results_root, run_label)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.1f}s'
    if seconds < 3600:
        return f'{seconds / 60.0:.1f} min'
    hours = seconds / 3600.0
    return f'{hours:.2f} h'


def print_progress(
    label: str,
    completed: int,
    total: int,
    elapsed: float,
    avg_time: float,
    width: int = 24,
) -> None:
    if total <= 0:
        return

    fraction = min(1.0, completed / total)
    filled = int(width * fraction)
    bar = '█' * filled + '░' * (width - filled)

    remaining = max(0, total - completed)
    eta = remaining * avg_time

    line = (
        f'{label} [{bar}] '
        f'{fraction * 100:5.1f}% | '
        f'{completed}/{total} | '
        f'avg {format_duration(avg_time)} | '
        f'ETA {format_duration(eta)} | '
        f'total ~{format_duration(elapsed + eta)}'
    )

    print(f'\033[2K\r{line}', end='', flush=True)


def estimate_total_runtime(n_optimizations: int, n_evaluations: int, timer: Optional[RuntimeTracker] = None) -> float:
    if timer is None:
        timer = RuntimeTracker()
    return timer.estimate_total_runtime(n_optimizations, n_evaluations)


def main() -> None:
    parser = argparse.ArgumentParser(description='Robust Decision Making for the fixed-baseline eVTOL model from optimize_fixed_baseline.py')
    parser.add_argument('--step', type=float, default=STEP_SIZE, help='Simplex grid step size (must divide 1.0 exactly).')
    parser.add_argument('--out-dir', type=str, default=None, help='Optional explicit output directory override.')
    args = parser.parse_args()

    step = float(args.step)
    if step <= 0.0:
        raise ValueError('Step size must be positive.')

    results_root = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_root, exist_ok=True)

    start_total = time.perf_counter()
    scenarios = simplex_grid(step)
    centroid_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

    x0 = DesignPoint(
        b=9.0,
        c=1.0,
        r_cruise=1.0,
        r_hover=1.0,
        V_cruise=60.0,
        V_climb=60.0,
        c_charge=1.0,
    )

    candidate_ids = [cand_name(i) for i in range(len(scenarios))] + ['centroid']
    scenario_ids = [scenario_name(i) for i in range(len(scenarios))]
    n_optimizations = len(scenarios) + 1
    n_evaluations = len(candidate_ids) * (len(scenarios) + 1)

    if args.out_dir is not None:
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = create_run_dir(results_root)

    timer = RuntimeTracker()
    print('==================================================')
    print('Robust Decision Making run for fixed-baseline model')
    print(f'Step size: {step}')
    print(f'Scenarios: {len(scenarios)}')
    print(f'Output directory: {out_dir}')
    print(
        f'Estimated total runtime: ~{format_duration(timer.estimate_total_runtime(n_optimizations, n_evaluations))}'
    )
    print('==================================================')
    print(f'Started at {datetime.now().strftime("%H:%M:%S")}')
    print()

    candidate_rows = []
    candidate_designs: dict[str, DesignPoint] = {}

    print('Running optimization sweep...')
    scenario_start = time.perf_counter()
    for i, (wcost, wgwp, wprofit) in enumerate(scenarios):
        candidate_id = cand_name(i)
        t_iter_start = time.perf_counter()
        out = optimize_for_weights(wcost, wgwp, wprofit, x0)
        elapsed_iter = time.perf_counter() - t_iter_start
        timer.optimization_times.append(elapsed_iter)

        candidate_designs[candidate_id] = DesignPoint(
            b=float(out['b']),
            c=float(out['c']),
            r_cruise=float(out['r_cruise']),
            r_hover=float(out['r_hover']),
            V_cruise=float(out['V_cruise']),
            V_climb=float(out['V_climb']),
            c_charge=float(out['c_charge']),
        )

        candidate_rows.append(
            {
                'design_id': i,
                'design_label': scenario_name(i),
                'w_cost': float(wcost),
                'w_gwp': float(wgwp),
                'w_profit': float(wprofit),
                'b': float(out['b']),
                'c': float(out['c']),
                'r_cruise': float(out['r_cruise']),
                'r_hover': float(out['r_hover']),
                'V_cruise': float(out['V_cruise']),
                'V_climb': float(out['V_climb']),
                'c_charge': float(out['c_charge']),
                'TOC_flight': float(out['TOC_flight']),
                'GWP_flight': float(out['GWP_flight']),
                'GWP_annual_ops': float(out['GWP_annual_ops']),
                'Annual_Profit': float(out['Annual_Profit']),
                'FoM': float(out['FoM']),
                'FoM_energy_rating': float(out['FoM_energy_rating']),
                'FoM_cost_rating': float(out['FoM_cost_rating']),
                'FoM_co2_rating': float(out['FoM_co2_rating']),
                'FoM_time_rating': float(out['FoM_time_rating']),
                'Cost_Rating_Norm': float(out['Cost_Rating_Norm']),
                'GWP_Rating_Norm': float(out['GWP_Rating_Norm']),
                'Profit_Rating_Norm': float(out['Profit_Rating_Norm']),
                'Profit_Utility': float(out['Profit_Utility']),
                'Utility': float(out['Utility']),
                'Utility_Norm_star_at_design_scenario': float(out['Utility_Norm']),
                'AR': float(out['AR']),
                'CL_cruise': float(out['CL_cruise']),
                'CL_climb': float(out['CL_climb']),
                'Gamma_deg': float(out['gamma_comp.gamma_deg']),
                'SPL_hover': float(out['SPL_hover']),
                'MTOM': float(out['MTOM']),
                'm_battery': float(out['m_battery']),
                'm_empty': float(out['m_empty']),
                'P_req_total_hover': float(out['P_req_total_hover']),
                'P_req_total_climb': float(out['P_req_total_climb']),
                'P_req_total_cruise': float(out['P_req_total_cruise']),
                'E_total_req': float(out['E_total_req']),
                'E_trip': float(out['E_trip']),
                'E_hover': float(out['E_hover']),
                'E_climb': float(out['E_climb']),
                'E_reserve': float(out['E_reserve']),
                'C_rate_hover': float(out['C_rate_hover']),
                'C_rate_climb': float(out['C_rate_climb']),
                'C_rate_cruise': float(out['C_rate_cruise']),
                'C_rate_avg': float(out['C_rate_avg']),
                'DOD': float(out['DOD']),
                'n_battery_lifecycle': float(out['n_battery_lifecycle']),
                'RPM_hover': float(out['RPM_hover']),
                'FC_a': float(out['FC_a']),
            }
        )
        completed = i + 1
        elapsed_total = time.perf_counter() - start_total
        avg_opt_time = timer.avg_optimization_time
        remaining_optimizations = max(0, len(scenarios) - completed)
        eta_optimization = remaining_optimizations * avg_opt_time
        print_progress(
            label='Optimization',
            completed=completed,
            total=len(scenarios),
            elapsed=elapsed_total,
            avg_time=avg_opt_time,
        )
        if completed == len(scenarios):
            print()

    print('Running centroid optimization...')
    c_start = time.perf_counter()
    centroid_t_start = time.perf_counter()
    centroid_out = optimize_for_weights(*centroid_weights, x0)
    centroid_elapsed = time.perf_counter() - centroid_t_start
    timer.optimization_times.append(centroid_elapsed)
    candidate_designs['centroid'] = DesignPoint(
        b=float(centroid_out['b']),
        c=float(centroid_out['c']),
        r_cruise=float(centroid_out['r_cruise']),
        r_hover=float(centroid_out['r_hover']),
        V_cruise=float(centroid_out['V_cruise']),
        V_climb=float(centroid_out['V_climb']),
        c_charge=float(centroid_out['c_charge']),
    )
    candidate_rows.append(
        {
            'design_id': len(scenarios),
            'design_label': 'centroid',
            'w_cost': centroid_weights[0],
            'w_gwp': centroid_weights[1],
            'w_profit': centroid_weights[2],
            'b': float(centroid_out['b']),
            'c': float(centroid_out['c']),
            'r_cruise': float(centroid_out['r_cruise']),
            'r_hover': float(centroid_out['r_hover']),
            'V_cruise': float(centroid_out['V_cruise']),
            'V_climb': float(centroid_out['V_climb']),
            'c_charge': float(centroid_out['c_charge']),
            'TOC_flight': float(centroid_out['TOC_flight']),
            'GWP_flight': float(centroid_out['GWP_flight']),
            'GWP_annual_ops': float(centroid_out['GWP_annual_ops']),
            'Annual_Profit': float(centroid_out['Annual_Profit']),
            'FoM': float(centroid_out['FoM']),
            'FoM_energy_rating': float(centroid_out['FoM_energy_rating']),
            'FoM_cost_rating': float(centroid_out['FoM_cost_rating']),
            'FoM_co2_rating': float(centroid_out['FoM_co2_rating']),
            'FoM_time_rating': float(centroid_out['FoM_time_rating']),
            'Cost_Rating_Norm': float(centroid_out['Cost_Rating_Norm']),
            'GWP_Rating_Norm': float(centroid_out['GWP_Rating_Norm']),
            'Profit_Rating_Norm': float(centroid_out['Profit_Rating_Norm']),
            'Profit_Utility': float(centroid_out['Profit_Utility']),
            'Utility': float(centroid_out['Utility']),
            'Utility_Norm_star_at_design_scenario': float(centroid_out['Utility_Norm']),
            'AR': float(centroid_out['AR']),
            'CL_cruise': float(centroid_out['CL_cruise']),
            'CL_climb': float(centroid_out['CL_climb']),
            'Gamma_deg': float(centroid_out['gamma_comp.gamma_deg']),
            'SPL_hover': float(centroid_out['SPL_hover']),
            'MTOM': float(centroid_out['MTOM']),
            'm_battery': float(centroid_out['m_battery']),
            'm_empty': float(centroid_out['m_empty']),
            'P_req_total_hover': float(centroid_out['P_req_total_hover']),
            'P_req_total_climb': float(centroid_out['P_req_total_climb']),
            'P_req_total_cruise': float(centroid_out['P_req_total_cruise']),
            'E_total_req': float(centroid_out['E_total_req']),
            'E_trip': float(centroid_out['E_trip']),
            'E_hover': float(centroid_out['E_hover']),
            'E_climb': float(centroid_out['E_climb']),
            'E_reserve': float(centroid_out['E_reserve']),
            'C_rate_hover': float(centroid_out['C_rate_hover']),
            'C_rate_climb': float(centroid_out['C_rate_climb']),
            'C_rate_cruise': float(centroid_out['C_rate_cruise']),
            'C_rate_avg': float(centroid_out['C_rate_avg']),
            'DOD': float(centroid_out['DOD']),
            'n_battery_lifecycle': float(centroid_out['n_battery_lifecycle']),
            'RPM_hover': float(centroid_out['RPM_hover']),
            'FC_a': float(centroid_out['FC_a']),
        }
    )
    print(f'Centroid optimization complete in {centroid_elapsed:.1f}s')

    candidate_count = len(candidate_designs)
    print(f'Candidate pool: {candidate_count} candidates ({len(scenarios)} scenario optima + centroid)')

    eval_scenario_labels = scenario_ids + ['centroid']

    print('Cross-evaluating candidates across scenario grid...')
    cross_eval_start = time.perf_counter()
    total_grid_evaluations = len(candidate_ids) * len(scenarios)
    utility_rows: list[dict[str, float]] = []
    utility_rows_with_centroid: list[dict[str, Any]] = []
    regret_rows: list[dict[str, float]] = []
    regret_rows_with_centroid: list[dict[str, Any]] = []
    utility_cache: dict[str, dict[str, float]] = {}
    regret_cache: dict[str, dict[str, float]] = {}

    candidate_components = {cid: evaluate_design_components(candidate_designs[cid]) for cid in candidate_ids}

    validation_sample = min(3, len(scenarios))
    for idx in range(validation_sample):
        wcost, wgwp, wprofit = scenarios[idx]
        for cid in candidate_ids[: min(2, len(candidate_ids))]:
            expected = evaluate_design_for_weights(wcost, wgwp, wprofit, candidate_designs[cid])['Utility_Norm']
            reconstructed = reconstruct_utility_from_components(candidate_components[cid], wcost, wgwp, wprofit)
            if not np.isclose(expected, reconstructed, atol=1e-8, rtol=1e-6):
                raise ValueError(
                    f'Validation failed for candidate {cid} at scenario {idx}: '
                    f'expected {expected}, reconstructed {reconstructed}'
                )

    scenario_weights = np.asarray(scenarios, dtype=float)
    cost_values = np.asarray([float(candidate_components[cid]['Cost_Rating_Norm']) for cid in candidate_ids], dtype=float)
    gwp_values = np.asarray([float(candidate_components[cid]['GWP_Rating_Norm']) for cid in candidate_ids], dtype=float)
    profit_values = np.asarray([float(candidate_components[cid]['Profit_Rating_Norm']) for cid in candidate_ids], dtype=float)
    rating_matrix = np.vstack([cost_values, gwp_values, profit_values])

    if not np.all(np.isfinite(scenario_weights)):
        bad = np.argwhere(~np.isfinite(scenario_weights))[:10]
        raise RuntimeError(f'Non-finite scenario weights detected before matrix multiplication. Sample: {bad.tolist()}')
    if not np.all(np.isfinite(rating_matrix)):
        bad = np.argwhere(~np.isfinite(rating_matrix))[:10]
        raise RuntimeError(f'Non-finite rating_matrix entries detected before utility matrix calculation. Sample: {bad.tolist()}')

    utility_matrix = scenario_weights @ rating_matrix
    if not np.all(np.isfinite(utility_matrix)):
        bad = np.argwhere(~np.isfinite(utility_matrix))[:10]
        raise RuntimeError(f'Non-finite utility_matrix after matrix multiplication. Sample: {bad.tolist()}')

    regret_matrix = np.max(utility_matrix, axis=1, keepdims=True) - utility_matrix
    if not np.all(np.isfinite(regret_matrix)):
        bad = np.argwhere(~np.isfinite(regret_matrix))[:10]
        raise RuntimeError(f'Non-finite regret_matrix after matrix subtraction. Sample: {bad.tolist()}')

    rng = np.random.default_rng(0)
    sample_sids = rng.choice(len(scenarios), size=min(5, len(scenarios)), replace=False)
    sample_cids = rng.choice(len(candidate_ids), size=min(5, len(candidate_ids)), replace=False)
    for sid in sample_sids:
        wcost, wgwp, wprofit = scenarios[sid]
        for cid_idx in sample_cids:
            cid = candidate_ids[cid_idx]
            expected = evaluate_design_for_weights(wcost, wgwp, wprofit, candidate_designs[cid])['Utility_Norm']
            actual = float(utility_matrix[sid, cid_idx])
            if not np.isclose(expected, actual, atol=1e-8, rtol=1e-6):
                raise RuntimeError(
                    f'Vectorized utility mismatch at scenario {sid}, candidate {cid}: '
                    f'expected={expected}, actual={actual}, diff={actual - expected}'
                )

    cross_elapsed = time.perf_counter() - cross_eval_start
    print(f'Vectorized cross-evaluation complete in {cross_elapsed:.1f}s for {len(scenarios)} scenarios x {len(candidate_ids)} candidates')

    for sid in range(len(scenarios)):
        scenario_row = utility_matrix[sid]
        scenario_base = {
            'scenario_id': sid,
            'w_cost': float(scenarios[sid][0]),
            'w_gwp': float(scenarios[sid][1]),
            'w_profit': float(scenarios[sid][2]),
        }
        utility_cache[scenario_name(sid)] = {cid: float(value) for cid, value in zip(candidate_ids, scenario_row)}
        utility_rows.append(_matrix_row(scenario_base, utility_cache[scenario_name(sid)], candidate_ids))

        regret_row = dict(scenario_base)
        for cid, value in zip(candidate_ids, regret_matrix[sid]):
            regret_row[cid] = float(value)
        regret_cache[scenario_name(sid)] = {cid: float(value) for cid, value in zip(candidate_ids, regret_matrix[sid])}
        regret_rows.append(regret_row)

    centroid_utility = (cost_values + gwp_values + profit_values) / 3.0
    centroid_regret = np.max(centroid_utility) - centroid_utility

    print('Centroid evaluation...')
    print_progress(
        label='Centroid evaluation',
        completed=len(candidate_ids),
        total=len(candidate_ids),
        elapsed=0.0,
        avg_time=timer.avg_evaluation_time,
    )
    print()

    centroid_base = {
        'scenario_id': 'centroid',
        'w_cost': centroid_weights[0],
        'w_gwp': centroid_weights[1],
        'w_profit': centroid_weights[2],
    }
    utility_rows_with_centroid = list(utility_rows)
    utility_cache['centroid'] = {cid: float(value) for cid, value in zip(candidate_ids, centroid_utility)}
    utility_rows_with_centroid.append(_matrix_row(centroid_base, utility_cache['centroid'], candidate_ids))

    regret_rows_with_centroid = list(regret_rows)
    centroid_regret_row = dict(centroid_base)
    regret_cache['centroid'] = {cid: float(value) for cid, value in zip(candidate_ids, centroid_regret)}
    for cid, value in zip(candidate_ids, centroid_regret):
        centroid_regret_row[cid] = float(value)
    regret_rows_with_centroid.append(centroid_regret_row)

    print('Computing regret summaries...')
    summary_rows = []
    for idx, cid in enumerate(candidate_ids):
        rr = np.asarray(regret_matrix[:, idx], dtype=float)
        summary_rows.append(
            {
                'candidate': cid,
                'max_regret': float(np.max(rr)),
                'mean_regret': float(np.mean(rr)),
                'p90_regret': float(np.quantile(rr, 0.9)),
                'satisficing_frac_regret_le_0p2': float(np.mean(rr <= 0.2)),
            }
        )

    summary_rows.sort(key=lambda r: r['max_regret'])
    robust = summary_rows[0]
    robust_candidate = robust['candidate']

    utility_matrix_fields = ['scenario_id', 'w_cost', 'w_gwp', 'w_profit'] + candidate_ids
    utility_matrix_transposed_fields = ['candidate'] + eval_scenario_labels
    regret_matrix_fields = ['scenario_id', 'w_cost', 'w_gwp', 'w_profit'] + candidate_ids
    regret_matrix_transposed_fields = ['candidate'] + eval_scenario_labels
    candidate_csv_fields = [
        'design_id', 'design_label', 'w_cost', 'w_gwp', 'w_profit',
        'b', 'c', 'r_cruise', 'r_hover', 'V_cruise', 'V_climb', 'c_charge',
        'TOC_flight', 'GWP_flight', 'GWP_annual_ops', 'Annual_Profit',
        'FoM', 'FoM_energy_rating', 'FoM_cost_rating', 'FoM_co2_rating', 'FoM_time_rating',
        'Cost_Rating_Norm', 'GWP_Rating_Norm', 'Profit_Rating_Norm',
        'Profit_Utility', 'Utility', 'Utility_Norm_star_at_design_scenario',
        'AR', 'CL_cruise', 'CL_climb', 'Gamma_deg',
        'SPL_hover', 'MTOM', 'm_battery', 'm_empty',
        'P_req_total_hover', 'P_req_total_climb', 'P_req_total_cruise',
        'E_total_req', 'E_trip', 'E_hover', 'E_climb', 'E_reserve',
        'C_rate_hover', 'C_rate_climb', 'C_rate_cruise', 'C_rate_avg', 'DOD', 'n_battery_lifecycle',
        'RPM_hover', 'FC_a',
    ]

    write_csv(
        os.path.join(out_dir, 'candidate_summary.csv'),
        candidate_rows,
        candidate_csv_fields,
    )
    write_csv(
        os.path.join(out_dir, 'utility_norm_cross_eval_matrix.csv'),
        utility_rows,
        utility_matrix_fields,
    )
    write_csv(
        os.path.join(out_dir, 'utility_norm_cross_eval_matrix_with_centroid.csv'),
        utility_rows_with_centroid,
        utility_matrix_fields,
    )
    write_csv(
        os.path.join(out_dir, 'utility_norm_cross_eval_matrix_candidates_as_rows.csv'),
        [
            {
                'candidate': cid,
                **{label: float(utility_cache[label][cid]) for label in eval_scenario_labels},
            }
            for cid in candidate_ids
        ],
        utility_matrix_transposed_fields,
    )
    write_csv(
        os.path.join(out_dir, 'regret_matrix.csv'),
        regret_rows_with_centroid,
        regret_matrix_fields,
    )
    write_csv(
        os.path.join(out_dir, 'regret_matrix_candidates_as_rows.csv'),
        [
            {
                'candidate': cid,
                **{label: float(regret_cache[label][cid]) for label in eval_scenario_labels},
            }
            for cid in candidate_ids
        ],
        regret_matrix_transposed_fields,
    )
    write_csv(
        os.path.join(out_dir, 'regret_summary.csv'),
        summary_rows,
        ['candidate', 'max_regret', 'mean_regret', 'p90_regret', 'satisficing_frac_regret_le_0p2'],
    )

    payload = {
        'step': step,
        'n_scenarios': len(scenarios),
        'n_candidates': len(candidate_ids),
        'weights_note': 'Grid sampling used for simplex space-filling, not probabilistic belief.',
        'objective_note': 'Optimization and regret are based on Utility_Norm for the fixed-baseline model in optimize_fixed_baseline.py.',
        'run_label': os.path.basename(out_dir),
        'robust_choice_minimax_regret': robust,
        'robust_candidate': robust_candidate,
        'output_dir': out_dir,
    }
    with open(os.path.join(out_dir, 'run_summary.json'), 'w') as f:
        json.dump(payload, f, indent=2)

    elapsed_total = time.perf_counter() - start_total
    print('\nRDM run complete.')
    print(f'Output folder: {out_dir}')
    print(f'Total elapsed time: {format_duration(elapsed_total)}')
    print(f'Robust choice (minimax regret): {robust}')


if __name__ == '__main__':
    main()
