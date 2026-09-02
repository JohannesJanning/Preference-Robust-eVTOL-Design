import time
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

# Utility objective setup requested for this run.
parameters.utility_cost_weight = 1
parameters.utility_gwp_weight = 0
parameters.utility_profit_weight = 0


prob = om.Problem(model=eVTOLGroupFixedBaseline(parameters=parameters))

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
    roc=float(parameters.roc_climb_target),
    rad2deg=57.29577951308232,
)
prob.model.add_subsystem('gamma_comp', gamma_comp)
prob.model.connect('V_climb', 'gamma_comp.V_climb')

prob.model.add_constraint('cons_comp.c1', lower=0.0, ref=1.0)
prob.model.add_constraint('vertiport_span', upper=15.0, ref=1.0)
prob.model.add_constraint('MTOM', upper=3750.0, ref=2000.0)
prob.model.add_constraint('CL_cruise', lower=0, upper=0.7, ref=1.0)
prob.model.add_constraint('CL_climb', lower=0.0, upper=1.2, ref=1.0)
prob.model.add_constraint('gamma_comp.gamma_deg', lower=5.0, upper=15.0, ref=10.0)
prob.model.add_constraint('SPL_hover', upper=77.0, ref=100.0)

#prob.model.add_objective('Utility_Norm', ref=-1.0)
prob.model.add_objective('FC_a', ref=-5500.0)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-6
prob.driver.options['disp'] = True
if hasattr(prob.driver, 'declare_coloring'):
    prob.driver.declare_coloring()

x0 = np.array([9.0, 1.0, 1.0, 1.0, 60.0, 60.0, 1.0], dtype=float)

prob.setup()
prob.set_val('b', x0[0])
prob.set_val('c', x0[1])
prob.set_val('r_cruise', x0[2])
prob.set_val('r_hover', x0[3])
prob.set_val('V_cruise', x0[4])
prob.set_val('V_climb', x0[5])
prob.set_val('rho_bat', parameters.rho_bat)
prob.set_val('c_charge', x0[6])

t0 = time.time()
try:
    prob.run_driver()
except Exception as exc:
    print('Optimization failed:', exc)
    raise
t1 = time.time()


def sval(name):
    try:
        return float(np.asarray(prob.get_val(name)).item())
    except Exception:
        return None


def _fmt(val):
    if val is None:
        return 'None'
    arr = np.asarray(val)
    if arr.size == 1:
        try:
            return f'{float(arr.item()):.6g}'
        except Exception:
            return str(arr.item())
    return np.array2string(arr, precision=6, suppress_small=True, threshold=40)


def _print_section(title, names):
    print('-------------------------------')
    print(title)
    for name in names:
        print(f'{name}:', _fmt(sval(name)))


def _safe_compute_thrusts():
    try:
        b = sval('b')
        c = sval('c')
        cd_cr = sval('CD_cruise')
        cd_cl = sval('CD_climb')
        v_cr = sval('V_cruise')
        v_cl = sval('V_climb')
        mtom = sval('MTOM')
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
            'gamma_climb_deg': gamma_deg,
            'ROC_target_mps': float(parameters.roc_climb_target),
            'D_climb_N': d_climb,
            'D_cruise_N': d_cruise,
            'T_req_total_climb_N': t_total_climb,
            'T_req_total_cruise_N': t_total_cruise,
            'T_req_total_hover_N': t_total_hover,
            'T_req_prop_climb_N': t_prop_climb,
            'T_req_prop_cruise_N': t_prop_cruise,
            'T_req_prop_hover_N': t_prop_hover,
        }
    except Exception:
        return {}


def _safe_compute_sizing_metrics(thrust_data):
    try:
        b = sval('b')
        c = sval('c')
        mtom = sval('MTOM')
        r_hover = sval('r_hover')
        r_cruise = sval('r_cruise')

        p_hover = sval('P_req_total_hover')
        p_climb = sval('P_req_total_climb')
        p_cruise = sval('P_req_total_cruise')

        cl_cruise = sval('CL_cruise')
        cd_cruise = sval('CD_cruise')
        cl_climb = sval('CL_climb')
        cd_climb = sval('CD_climb')

        m_empty = sval('m_empty')
        m_battery = sval('m_battery')

        if any(v is None for v in [
            b, c, mtom, r_hover, r_cruise,
            p_hover, p_climb, p_cruise,
            cl_cruise, cd_cruise, cl_climb, cd_climb,
            m_empty, m_battery
        ]):
            return {}

        # ---------------------------------------------------------
        # Aerodynamic efficiency
        # ---------------------------------------------------------
        ld_cruise = cl_cruise / max(cd_cruise, 1e-12)
        ld_climb = cl_climb / max(cd_climb, 1e-12)

        # ---------------------------------------------------------
        # Wing geometry and loading
        # ---------------------------------------------------------
        s_ref = max(b * c, 1e-12)

        # Wing loading as mass loading
        wing_loading_kg_m2 = mtom / s_ref

        # Convert to true weight loading
        wing_loading_n_m2 = wing_loading_kg_m2 * parameters.g

        # ---------------------------------------------------------
        # Rotor disk areas
        # ---------------------------------------------------------
        a_disk_hover_total = max(
            float(parameters.n_prop_vert) * np.pi * r_hover**2,
            1e-12
        )

        a_disk_cruise_total = max(
            float(parameters.n_prop_hor) * np.pi * r_cruise**2,
            1e-12
        )

        # ---------------------------------------------------------
        # Disk loading
        # ---------------------------------------------------------
        # Conventional mass-based representation
        disk_loading_hover_kg_m2 = mtom / a_disk_hover_total
        disk_loading_cruise_kg_m2 = mtom / a_disk_cruise_total

        # True force-based disk loading
        disk_loading_hover_n_m2 = (
            mtom * parameters.g / a_disk_hover_total
        )

        disk_loading_cruise_n_m2 = (
            mtom * parameters.g / a_disk_cruise_total
        )

        # ---------------------------------------------------------
        # Power-to-weight
        # ---------------------------------------------------------
        pw_hover_w_kg = p_hover / max(mtom, 1e-12)
        pw_climb_w_kg = p_climb / max(mtom, 1e-12)
        pw_cruise_w_kg = p_cruise / max(mtom, 1e-12)

        # Equivalent hp/lb
        pw_hover_hp_lb = pw_hover_w_kg * 0.00134102
        pw_climb_hp_lb = pw_climb_w_kg * 0.00134102
        pw_cruise_hp_lb = pw_cruise_w_kg * 0.00134102

        # ---------------------------------------------------------
        # Mass fractions
        # ---------------------------------------------------------
        empty_mass_fraction = m_empty / max(mtom, 1e-12)
        battery_mass_fraction = m_battery / max(mtom, 1e-12)

        return {
            'Lift_to_Drag_Cruise': ld_cruise,
            'Lift_to_Drag_Climb': ld_climb,

            'Wing_Loading_kg_m2': wing_loading_kg_m2,
            'Wing_Loading_N_m2': wing_loading_n_m2,

            'Disk_Loading_Hover_kg_m2': disk_loading_hover_kg_m2,
            'Disk_Loading_Hover_N_m2': disk_loading_hover_n_m2,

            'Disk_Loading_Cruise_kg_m2': disk_loading_cruise_kg_m2,
            'Disk_Loading_Cruise_N_m2': disk_loading_cruise_n_m2,

            'Power_to_Weight_Hover_W_kg': pw_hover_w_kg,
            'Power_to_Weight_Hover_hp_lb': pw_hover_hp_lb,

            'Power_to_Weight_Climb_W_kg': pw_climb_w_kg,
            'Power_to_Weight_Climb_hp_lb': pw_climb_hp_lb,

            'Power_to_Weight_Cruise_W_kg': pw_cruise_w_kg,
            'Power_to_Weight_Cruise_hp_lb': pw_cruise_hp_lb,

            'Empty_Mass_Fraction': empty_mass_fraction,
            'Battery_Mass_Fraction': battery_mass_fraction,
        }

    except Exception:
        return {}


def _print_full_dump():
    print('-------------------------------')
    print('FULL OUTPUT DUMP (sorted)')
    try:
        outs = prob.model.list_outputs(out_stream=None, prom_name=True, units=True, val=True)
        outs = sorted(outs, key=lambda x: x[0])
        for abs_name, meta in outs:
            prom = meta.get('prom_name', abs_name)
            units = meta.get('units')
            unit_txt = f' [{units}]' if units else ''
            print(f'{prom}{unit_txt}: {_fmt(meta.get("val"))}')
    except Exception as exc:
        print('Could not print full output dump:', exc)


print(f'Elapsed (s): {t1 - t0:.1f}')

print('-------------------------------')
print('RUN SUMMARY')
print('Objective Utility_Norm:', _fmt(sval('Utility_Norm')))
print('Utility weights:', {
    'cost': float(parameters.utility_cost_weight),
    'gwp': float(parameters.utility_gwp_weight),
    'profit': float(parameters.utility_profit_weight),
})

_print_section('CONSTRAINT SNAPSHOT', [
    'cons_comp.c1', 'AR', 'vertiport_span', 'MTOM', 'CL_cruise', 'CL_climb', 'gamma_comp.gamma_deg', 'SPL_hover'
])

_print_section('DESIGN VARIABLES', [
    'b', 'c', 'r_cruise', 'r_hover', 'V_cruise', 'V_climb', 'rho_bat', 'c_charge'
])

derived_thrusts = _safe_compute_thrusts()
if derived_thrusts:
    print('-------------------------------')
    print('THRUST + DRAG (derived)')
    for k in sorted(derived_thrusts):
        print(f'{k}: {_fmt(derived_thrusts[k])}')

sizing_metrics = _safe_compute_sizing_metrics(derived_thrusts)
if sizing_metrics:
    print('-------------------------------')
    print('SIZING METRICS (derived)')
    for k in sorted(sizing_metrics):
        print(f'{k}: {_fmt(sizing_metrics[k])}')


_print_section('AERODYNAMICS', [
    'CL_cruise',
    'CD_cruise',
    'CL_climb',
    'CD_climb',
    'V_climb_hor'
])

print(f'Lift-to-drag ratio cruise: '
      f'{sizing_metrics.get("Lift_to_Drag_Cruise", None):.6f}')

print(f'Lift-to-drag ratio climb: '
      f'{sizing_metrics.get("Lift_to_Drag_Climb", None):.6f}')

print(f'AR: {_fmt(sval("AR"))}')

print(f'Wing Loading: '
      f'{sizing_metrics.get("Wing_Loading_kg_m2", None):.6f} kg/m^2 '
      f'({sizing_metrics.get("Wing_Loading_N_m2", None):.2f} N/m^2)')

print(f'Disk Loading hover (T/A): '
      f'{sizing_metrics.get("Disk_Loading_Hover_kg_m2", None) * 1.25:.6f} kg/m^2 '
      f'({sizing_metrics.get("Disk_Loading_Hover_N_m2", None) * 1.25:.2f} N/m^2)')

_print_section('POWER', [
    'P_req_total_hover',
    'P_req_total_climb',
    'P_req_total_cruise'
])

print(f'Power-to-weight hover: '
      f'{sizing_metrics["Power_to_Weight_Hover_W_kg"]:.6f} W/kg '
      f'({sizing_metrics["Power_to_Weight_Hover_hp_lb"]:.6f} hp/lb)')

print(f'Power-to-weight climb: '
      f'{sizing_metrics["Power_to_Weight_Climb_W_kg"]:.6f} W/kg '
      f'({sizing_metrics["Power_to_Weight_Climb_hp_lb"]:.6f} hp/lb)')

print(f'Power-to-weight cruise: '
      f'{sizing_metrics["Power_to_Weight_Cruise_W_kg"]:.6f} W/kg '
      f'({sizing_metrics["Power_to_Weight_Cruise_hp_lb"]:.6f} hp/lb)')



_print_section('ENERGY + TIMES', [
    'E_total_req',
    'E_trip',
    'E_hover',
    'E_climb',
    'E_reserve',
])

t_cruise = sval('t_cruise')
t_trip = sval('t_trip')

if t_cruise is not None:
    print(f't_cruise: {t_cruise:.6f} s '
          f'({t_cruise / 60.0:.6f} min)')

if t_trip is not None:
    print(f't_trip: {t_trip:.6f} s '
          f'({t_trip / 60.0:.6f} min)')

_print_section('BATTERY C-RATES', [
    'C_rate_hover', 'C_rate_climb', 'C_rate_cruise', 'C_rate_avg', 'DOD', 'n_battery_lifecycle'
])

_print_section('MASS', [
    'MTOM',
    'MTOM_est',
    'm_battery',
    'm_empty'
])

print(f'Empty mass / MTOM: '
      f'{sizing_metrics["Empty_Mass_Fraction"]:.6f}')

print(f'Battery mass / MTOM: '
      f'{sizing_metrics["Battery_Mass_Fraction"]:.6f}')

_print_section('NOISE', [
    'SPL_hover',
    'RPM_hover'
])

_print_section('OPS + TRANSPORT', [
    'FC_a',
    'FoM',
    'FoM_time_rating',
    'FoM_co2_rating',
    'FoM_energy_rating',
    'FoM_cost_rating'
])

_print_section('ECONOMICS + GWP', [
    'TOC_flight',
    'GWP_flight',
    'GWP_annual_ops'
])

annual_profit = sval('Annual_Profit')

if annual_profit is not None:
    print(f'Annual_Profit: {annual_profit:,.2f}')

_print_section('UTILITY', [
    'Utility', 'Utility_Norm',
    'Cost_Utility', 'GWP_Utility', 'Profit_Utility_Value',
    'Cost_Utility_Norm', 'GWP_Utility_Norm', 'Profit_Utility_Norm',
    'Cost_Rating_Norm', 'GWP_Rating_Norm', 'Profit_Rating_Norm',
    'Profit_Utility'
])




#_print_full_dump()
