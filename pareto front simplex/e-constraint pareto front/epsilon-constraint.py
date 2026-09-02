import time
import copy
import csv
from types import SimpleNamespace
import numpy as np
import openmdao.api as om

from src.parameters import model_parameters as parameters
from src.optimizer.eVTOL_group_fixed_baseline import eVTOLGroupFixedBaseline
from src.models_jax.aerodynamics.drag import drag_calculation
from src.models_jax.aerodynamics.ROC import gamma_from_roc
from src.models_jax.momentum.T_climb_total import total_thrust_required_climb
from src.models_jax.momentum.T_cruise_total import total_thrust_required_cruise
from src.models_jax.momentum.T_hover_total import total_thrust_required_hover
from src.models_jax.momentum.T_prop import thrust_per_propeller


# =========================================================
# USER SETTINGS
# =========================================================
OUTPUT_CSV = 'pareto_epsilon_results.csv'

# Baseline initial guess
X0 = {
    'b': 9.0,
    'c': 1.0,
    'r_cruise': 1.0,
    'r_hover': 1.0,
    'V_cruise': 60.0,
    'V_climb': 60.0,
    'rho_bat': parameters.rho_bat,
    'c_charge': 1.0,
}

# Driver settings
OPTIMIZER = 'SLSQP'
TOL = 1e-6
DISP = False
USE_COLORING = True

# Pareto sweep settings
N_GWP = 12
N_PROFIT = 12
WARM_START = True

# Main objective for epsilon-constraint runs:
# choose one of: 'TOC_flight', 'GWP_flight', 'neg_Annual_Profit'
PRIMARY_OBJECTIVE = 'TOC_flight'


# =========================================================
# HELPERS
# =========================================================
def make_params():
    p = SimpleNamespace(
        **{
            key: value
            for key, value in vars(parameters).items()
            if not key.startswith('__') and not callable(value)
        }
    )
    p.utility_cost_weight = 0.0
    p.utility_gwp_weight = 0.0
    p.utility_profit_weight = 0.0
    return p


def add_common_model(prob, p):
    # Main model
    prob.model = eVTOLGroupFixedBaseline(parameters=p)

    iv = om.IndepVarComp()
    iv.add_output('b', val=X0['b'])
    iv.add_output('c', val=X0['c'])
    iv.add_output('r_cruise', val=X0['r_cruise'])
    iv.add_output('r_hover', val=X0['r_hover'])
    iv.add_output('V_cruise', val=X0['V_cruise'])
    iv.add_output('V_climb', val=X0['V_climb'])
    iv.add_output('rho_bat', val=X0['rho_bat'])
    iv.add_output('c_charge', val=X0['c_charge'])
    prob.model.add_subsystem('iv', iv, promotes=['*'])

    prob.model.add_design_var('b', lower=6.0, upper=15.0, ref0=6.0, ref=15.0)
    prob.model.add_design_var('c', lower=1.0, upper=2.5, ref0=1.0, ref=2.5)
    prob.model.add_design_var('r_cruise', lower=0.7, upper=1.2, ref0=0.6, ref=1.2)
    prob.model.add_design_var('r_hover', lower=0.6, upper=1.9, ref0=0.6, ref=1.3)
    prob.model.add_design_var('V_cruise', lower=40.0, upper=129.0, ref0=40.0, ref=129.0)
    prob.model.add_design_var('V_climb', lower=40.0, upper=129.0, ref0=40.0, ref=129.0)
    prob.model.add_design_var('c_charge', lower=1.0, upper=4.0, ref0=1.0, ref=4.0)
    prob.model.add_constraint('AR', lower=6.0, upper=10.0, ref=8.0)

    cons_comp = om.ExecComp('c1 = b - rotor_spacing', b=15.0, rotor_spacing=1.0)
    prob.model.add_subsystem('cons_comp', cons_comp)
    prob.model.connect('rotor_spacing', 'cons_comp.rotor_spacing')
    prob.model.connect('b', 'cons_comp.b')

    gamma_comp = om.ExecComp(
        'gamma_deg = rad2deg * asin(roc / V_climb)',
        gamma_deg=5.0,
        V_climb=60.0,
        roc=float(p.roc_climb_target),
        rad2deg=57.29577951308232,
    )
    prob.model.add_subsystem('gamma_comp', gamma_comp)
    prob.model.connect('V_climb', 'gamma_comp.V_climb')

    prob.model.add_constraint('cons_comp.c1', lower=0.0, ref=1.0)
    prob.model.add_constraint('vertiport_span', upper=15.0, ref=1.0)
    prob.model.add_constraint('MTOM', upper=3750.0, ref=2000.0)
    prob.model.add_constraint('CL_cruise', lower=0.0, upper=0.7, ref=1.0)
    prob.model.add_constraint('CL_climb', lower=0.0, upper=1.2, ref=1.0)
    prob.model.add_constraint('gamma_comp.gamma_deg', lower=5.0, upper=15.0, ref=10.0)
    prob.model.add_constraint('SPL_hover', upper=77.0, ref=100.0)

    # Helper objective for profit maximization as minimization
    prob.model.add_subsystem(
        'profit_helper',
        om.ExecComp('neg_Annual_Profit = -Annual_Profit',
                    Annual_Profit=1.0,
                    neg_Annual_Profit=0.0),
        promotes=[]
    )
    prob.model.connect('Annual_Profit', 'profit_helper.Annual_Profit')


def set_initial_guess(prob, xdict):
    for k, v in xdict.items():
        prob.set_val(k, v)


def sval(prob, name):
    try:
        return float(np.asarray(prob.get_val(name)).item())
    except Exception:
        return np.nan


def collect_row(prob, tag, success, elapsed, gwp_eps=np.nan, profit_eps=np.nan):
    return {
        'tag': tag,
        'success': int(bool(success)),
        'elapsed_s': elapsed,
        'gwp_eps': gwp_eps,
        'profit_eps': profit_eps,
        'b': sval(prob, 'b'),
        'c': sval(prob, 'c'),
        'r_cruise': sval(prob, 'r_cruise'),
        'r_hover': sval(prob, 'r_hover'),
        'V_cruise': sval(prob, 'V_cruise'),
        'V_climb': sval(prob, 'V_climb'),
        'c_charge': sval(prob, 'c_charge'),
        'TOC_flight': sval(prob, 'TOC_flight'),
        'GWP_flight': sval(prob, 'GWP_flight'),
        'Annual_Profit': sval(prob, 'Annual_Profit'),
        'Utility_Norm': sval(prob, 'Utility_Norm'),
        'MTOM': sval(prob, 'MTOM'),
        'AR': sval(prob, 'AR'),
        'SPL_hover': sval(prob, 'SPL_hover'),
    }


def solve_anchor(primary_objective):
    p = make_params()
    prob = om.Problem()
    add_common_model(prob, p)

    if primary_objective == 'neg_Annual_Profit':
        prob.model.add_objective('profit_helper.neg_Annual_Profit', ref=1.0)
    else:
        prob.model.add_objective(primary_objective, ref=100.0)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = OPTIMIZER
    prob.driver.options['tol'] = TOL
    prob.driver.options['disp'] = DISP
    if USE_COLORING and hasattr(prob.driver, 'declare_coloring'):
        prob.driver.declare_coloring()

    prob.setup()
    set_initial_guess(prob, X0)

    t0 = time.time()
    ok = True
    try:
        prob.run_driver()
    except Exception as exc:
        ok = False
        print(f'Anchor run failed for {primary_objective}: {exc}')
    t1 = time.time()

    row = collect_row(prob, f'anchor_{primary_objective}', ok, t1 - t0)
    return row


def solve_epsilon_run(primary_objective, gwp_eps, profit_eps, xstart):
    p = make_params()
    prob = om.Problem()
    add_common_model(prob, p)

    prob.model.add_constraint('GWP_flight', upper=float(gwp_eps), ref=max(float(gwp_eps), 1.0))
    prob.model.add_constraint('Annual_Profit', lower=float(profit_eps), ref=max(abs(float(profit_eps)), 1.0))

    if primary_objective == 'neg_Annual_Profit':
        prob.model.add_objective('profit_helper.neg_Annual_Profit', ref=1.0)
    else:
        prob.model.add_objective(primary_objective, ref=100.0)

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = OPTIMIZER
    prob.driver.options['tol'] = TOL
    prob.driver.options['disp'] = DISP
    if USE_COLORING and hasattr(prob.driver, 'declare_coloring'):
        prob.driver.declare_coloring()

    prob.setup()
    set_initial_guess(prob, xstart)

    t0 = time.time()
    ok = True
    try:
        prob.run_driver()
    except Exception as exc:
        ok = False
        print(f'Epsilon run failed for gwp<= {gwp_eps:.4f}, profit>= {profit_eps:.4f}: {exc}')
    t1 = time.time()

    row = collect_row(prob, 'epsilon_run', ok, t1 - t0, gwp_eps=gwp_eps, profit_eps=profit_eps)

    xnext = {
        'b': sval(prob, 'b'),
        'c': sval(prob, 'c'),
        'r_cruise': sval(prob, 'r_cruise'),
        'r_hover': sval(prob, 'r_hover'),
        'V_cruise': sval(prob, 'V_cruise'),
        'V_climb': sval(prob, 'V_climb'),
        'rho_bat': X0['rho_bat'],
        'c_charge': sval(prob, 'c_charge'),
    }
    return row, xnext


def is_dominated(i, rows):
    gi = rows[i]['GWP_flight']
    ci = rows[i]['TOC_flight']
    pi = rows[i]['Annual_Profit']

    if not np.isfinite(gi) or not np.isfinite(ci) or not np.isfinite(pi):
        return True

    for j, rj in enumerate(rows):
        if i == j:
            continue

        gj = rj['GWP_flight']
        cj = rj['TOC_flight']
        pj = rj['Annual_Profit']

        if not np.isfinite(gj) or not np.isfinite(cj) or not np.isfinite(pj):
            continue

        weakly_better = (gj <= gi) and (cj <= ci) and (pj >= pi)
        strictly_better = (gj < gi) or (cj < ci) or (pj > pi)

        if weakly_better and strictly_better:
            return True

    return False


def flag_pareto(rows):
    out = []
    for i, r in enumerate(rows):
        rr = dict(r)
        rr['pareto_flag'] = int(not is_dominated(i, rows))
        out.append(rr)
    return out


def write_csv(rows, filename):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    print('=== STEP 1: Anchor runs ===')
    anchor_toc = solve_anchor('TOC_flight')
    anchor_gwp = solve_anchor('GWP_flight')
    anchor_profit = solve_anchor('neg_Annual_Profit')

    anchors = [anchor_toc, anchor_gwp, anchor_profit]
    for a in anchors:
        print(a)

    gwp_vals = [a['GWP_flight'] for a in anchors if np.isfinite(a['GWP_flight'])]
    profit_vals = [a['Annual_Profit'] for a in anchors if np.isfinite(a['Annual_Profit'])]

    gwp_min = min(gwp_vals)
    gwp_max = max(gwp_vals)
    profit_min = min(profit_vals)
    profit_max = max(profit_vals)

    gwp_grid = np.linspace(gwp_min, gwp_max, N_GWP)
    profit_grid = np.linspace(profit_min, profit_max, N_PROFIT)

    print('\n=== STEP 2: Epsilon-constraint sweep ===')
    print(f'Primary objective: {PRIMARY_OBJECTIVE}')
    print(f'GWP range: {gwp_min:.4f} -> {gwp_max:.4f}')
    print(f'Profit range: {profit_min:.4f} -> {profit_max:.4f}')
    print(f'Grid size: {N_GWP} x {N_PROFIT} = {N_GWP * N_PROFIT}')

    rows = []
    xstart = dict(X0)

    for gwp_eps in gwp_grid:
        for profit_eps in profit_grid[::-1]:
            print(f'Running gwp <= {gwp_eps:.4f}, profit >= {profit_eps:.4f}')
            row, xnext = solve_epsilon_run(PRIMARY_OBJECTIVE, gwp_eps, profit_eps, xstart)
            rows.append(row)

            if WARM_START and row['success']:
                xstart = xnext

    print('\n=== STEP 3: Pareto filtering ===')
    successful_rows = [r for r in rows if r['success'] == 1]
    pareto_rows = flag_pareto(successful_rows)
    anchor_rows = flag_pareto([r for r in anchors if r['success'] == 1])

    all_rows = []
    all_rows.extend(anchor_rows)
    all_rows.extend(pareto_rows)

    write_csv(all_rows, OUTPUT_CSV)

    n_ok = sum(r['success'] for r in rows)
    n_pf = sum(r['pareto_flag'] for r in pareto_rows)

    print(f'Successful epsilon runs: {n_ok}/{len(rows)}')
    print(f'Pareto points among successful epsilon runs: {n_pf}')
    print(f'Wrote results to: {OUTPUT_CSV}')
    print('\nDone.')
