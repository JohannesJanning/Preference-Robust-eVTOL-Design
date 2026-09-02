import openmdao.api as om
import numpy as np
import jax
import jax.numpy as jnp

from src.models_jax.economic.costs import (
    energy_cost_model,
    navigation_cost_model,
    crew_cost_model,
    wrap_maintenance_cost,
    battery_maintenance_cost,
    maintenance_cost_model,
    cash_operating_cost,
    ownership_cost_model,
    direct_operating_cost,
    indirect_operating_cost,
    total_operating_cost,
)
from src.models_jax.economic.revenue import (
    revenue_per_flight,
    ticket_price_per_passenger,
    profit_per_flight,
    annual_profit,
)

from src.models_jax.battery.E_battery_design import battery_energy_capacity
from src.models_jax.operations.ops_model import turnaround_time, time_efficiency_ratio, daily_flight_cycles


class EconomicComp(om.ExplicitComponent):
    """Compute operating cost metrics and expose `TOC_flight` for optimization.

    Inputs expected to be promoted at the group level: E_trip, MTOM, t_trip,
    m_battery, rho_bat, FC_a, DOD, n_batt_annual, m_empty, c_charge.
    """

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        # runtime inputs from other components
        self.add_input('E_trip', val=0.0)
        self.add_input('MTOM', val=1500.0)
        self.add_input('t_trip', val=3600.0)
        self.add_input('m_battery', val=0.0)
        self.add_input('rho_bat', val=300.0)
        self.add_input('FC_a', val=260.0)
        self.add_input('DOD', val=0.3)
        self.add_input('n_batt_annual', val=0.0)
        self.add_input('m_empty', val=0.0)
        self.add_input('c_charge', val=1.0)

        # primary outputs used for objective
        self.add_output('COC_flight', val=0.0)
        self.add_output('COO_value_flight', val=0.0)
        self.add_output('DOC_flight', val=0.0)
        self.add_output('IOC_value_flight', val=0.0)
        self.add_output('TOC_flight', val=0.0)
        self.add_output('Revenue_Flight', val=0.0)
        self.add_output('Ticket_Price', val=0.0)
        self.add_output('Profit_Flight', val=0.0)
        self.add_output('Annual_Profit', val=0.0)

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        p = self.options['parameters']

        E_trip = inputs['E_trip'][0]
        MTOM = inputs['MTOM'][0]
        t_trip = inputs['t_trip'][0]
        m_batt = inputs['m_battery'][0]
        rho_bat = inputs['rho_bat'][0]
        FC_a = inputs['FC_a'][0]
        DOD = inputs['DOD'][0]
        n_batt_annual = inputs['n_batt_annual'][0]
        m_empty = inputs['m_empty'][0]
        c_charge = inputs['c_charge'][0]

        # compute DH (time efficiency ratio) used by some cost terms
        time_turn = float(np.asarray(turnaround_time(c_charge, DOD)))
        DH = float(np.asarray(time_efficiency_ratio(time_turn, max(t_trip, 1e-6))))

        # energy cost (EUR per flight)
        C_energy = float(np.asarray(energy_cost_model(E_trip, p.P_e)))

        # navigation/airport related costs
        C_navigation = float(np.asarray(navigation_cost_model(MTOM, p.unitrate, p.distance_trip_km)))

        # crew cost per flight
        C_crew = float(np.asarray(crew_cost_model(p.S_P, p.N_wd, p.T_D, p.U_pilot, p.N_AC, t_trip, DH)))

        # maintenance costs
        wrap = float(np.asarray(wrap_maintenance_cost(t_trip)))

        # compute battery design energy (Wh)
        E_batt_design_Wh = float(np.asarray(battery_energy_capacity(rho_bat, m_batt)))
        batt_maint = float(np.asarray(battery_maintenance_cost(n_batt_annual, p.P_bat_s, E_batt_design_Wh, t_trip, DH, p.T_D, p.N_wd)))

        C_maintenance = float(np.asarray(maintenance_cost_model(batt_maint, wrap)))

        # cash operating cost per flight (COC)
        COC_flight = float(np.asarray(cash_operating_cost(C_energy, C_navigation, C_crew, C_maintenance)))

        # ownership and other values
        omega_empty = (m_empty / MTOM) if MTOM != 0.0 else 0.0
        COO_value = float(np.asarray(ownership_cost_model(COC_flight, omega_empty, MTOM, p.P_s_empty, p.N_wd, p.T_D, t_trip, DH)))

        # direct & indirect operating costs
        DOC_flight = float(np.asarray(direct_operating_cost(COC_flight, COO_value)))
        FC_d = float(np.asarray(daily_flight_cycles(p.T_D, t_trip, DH)))
        IOC_value = float(np.asarray(indirect_operating_cost(COC_flight, omega_empty, MTOM, p.P_s_empty, p.N_wd, FC_d)))

        TOC_flight = float(np.asarray(total_operating_cost(DOC_flight, IOC_value)))

        # revenue/profit metrics
        revenue_flight = float(np.asarray(revenue_per_flight(p.fare_km, p.distance_trip_km, p.N_s, p.LF)))
        profit_flight = float(np.asarray(profit_per_flight(revenue_flight, TOC_flight)))
        ticket_price = float(np.asarray(ticket_price_per_passenger(revenue_flight, p.N_s, p.LF)))
        annual_profit_value = float(np.asarray(annual_profit(revenue_flight, TOC_flight, FC_a)))

        outputs['COC_flight'] = COC_flight
        outputs['COO_value_flight'] = COO_value
        outputs['DOC_flight'] = DOC_flight
        outputs['IOC_value_flight'] = IOC_value
        outputs['TOC_flight'] = TOC_flight
        outputs['Revenue_Flight'] = revenue_flight
        outputs['Ticket_Price'] = ticket_price
        outputs['Profit_Flight'] = profit_flight
        outputs['Annual_Profit'] = annual_profit_value

    def compute_partials(self, inputs, partials):
        in_names = ['E_trip', 'MTOM', 't_trip', 'm_battery', 'rho_bat', 'FC_a', 'DOD', 'n_batt_annual', 'm_empty', 'c_charge']
        out_names = [
            'COC_flight',
            'COO_value_flight',
            'DOC_flight',
            'IOC_value_flight',
            'TOC_flight',
            'Revenue_Flight',
            'Ticket_Price',
            'Profit_Flight',
            'Annual_Profit',
        ]

        x = jnp.array([inputs[n][0] for n in in_names])

        def fun(x):
            E_trip, MTOM, t_trip, m_batt, rho_bat, FC_a, DOD, n_batt_annual, m_empty, c_charge = x
            p = self.options['parameters']
            time_turn = turnaround_time(c_charge, DOD)
            DH = time_efficiency_ratio(time_turn, jnp.maximum(t_trip, 1e-6))
            C_energy = energy_cost_model(E_trip, p.P_e)
            C_navigation = navigation_cost_model(MTOM, p.unitrate, p.distance_trip_km)
            C_crew = crew_cost_model(p.S_P, p.N_wd, p.T_D, p.U_pilot, p.N_AC, t_trip, DH)
            wrap = wrap_maintenance_cost(t_trip)
            E_batt_design_Wh = battery_energy_capacity(rho_bat, m_batt)
            batt_maint = battery_maintenance_cost(n_batt_annual, p.P_bat_s, E_batt_design_Wh, t_trip, DH, p.T_D, p.N_wd)
            C_maintenance = maintenance_cost_model(batt_maint, wrap)
            COC_flight = cash_operating_cost(C_energy, C_navigation, C_crew, C_maintenance)
            omega_empty = jnp.where(MTOM != 0.0, m_empty / MTOM, 0.0)
            COO_value = ownership_cost_model(COC_flight, omega_empty, MTOM, p.P_s_empty, p.N_wd, p.T_D, t_trip, DH)
            DOC_flight = direct_operating_cost(COC_flight, COO_value)
            FC_d = daily_flight_cycles(p.T_D, t_trip, DH)
            IOC_value = indirect_operating_cost(COC_flight, omega_empty, MTOM, p.P_s_empty, p.N_wd, FC_d)
            TOC_flight = total_operating_cost(DOC_flight, IOC_value)
            revenue_flight = revenue_per_flight(p.fare_km, p.distance_trip_km, p.N_s, p.LF)
            profit_flight = profit_per_flight(revenue_flight, TOC_flight)
            ticket_price = ticket_price_per_passenger(revenue_flight, p.N_s, p.LF)
            annual_profit_value = annual_profit(revenue_flight, TOC_flight, FC_a)
            return jnp.array([
                COC_flight,
                COO_value,
                DOC_flight,
                IOC_value,
                TOC_flight,
                revenue_flight,
                ticket_price,
                profit_flight,
                annual_profit_value,
            ])

        J = jax.jacfwd(fun)(x)
        J = np.asarray(J)
        J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)

        for i, out in enumerate(out_names):
            for j, inp in enumerate(in_names):
                partials[(out, inp)] = J[i, j]
