import openmdao.api as om
import numpy as np

from src.models_jax.transportation.transportation_modes_fixed_baseline import (
    transportation_mode_comparison_fixed_baseline,
)


class TransportationCompFixedBaseline(om.ExplicitComponent):
    """Compute eVTOL FoM against fixed baseline min/max bounds."""

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        self.add_input('t_trip', val=3600.0)
        self.add_input('E_trip', val=0.0)
        self.add_input('TOC_flight', val=0.0)
        self.add_input('GWP_flight', val=0.0)
        self.add_input('GWP_annual_ops', val=0.0)
        self.add_input('FC_a', val=0.0)

        self.add_output('FoM', val=0.0)
        self.add_output('FoM_time_rating', val=0.0)
        self.add_output('FoM_co2_rating', val=0.0)
        self.add_output('FoM_energy_rating', val=0.0)
        self.add_output('FoM_cost_rating', val=0.0)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        p = self.options['parameters']

        t_tot = float(inputs['t_trip'][0])
        e_trip = float(inputs['E_trip'][0])
        toc_flight = float(inputs['TOC_flight'][0])
        gwp_flight = float(inputs['GWP_flight'][0])

        # Align FoM diagnostic weights with the utility framing:
        # time and energy are excluded, and the co2/cost split follows utility weights.
        utility_cost_weight = float(getattr(p, 'utility_cost_weight', 1.0 / 3.0))
        utility_gwp_weight = float(getattr(p, 'utility_gwp_weight', 1.0 / 3.0))
        ce_sum = utility_cost_weight + utility_gwp_weight
        if ce_sum <= 1e-12:
            co2_weight = 0.5
            costs_weight = 0.5
        else:
            co2_weight = utility_gwp_weight / ce_sum
            costs_weight = utility_cost_weight / ce_sum

        results = transportation_mode_comparison_fixed_baseline(
            t_tot=t_tot,
            e_trip=e_trip,
            D_trip=float(p.distance_trip_km),
            toc_flight=toc_flight,
            time_weight=0.0,
            co2_weight=co2_weight,
            energy_weight=0.0,
            costs_weight=costs_weight,
            gwp_flight=gwp_flight,
            LF=float(p.LF),
            N_s=int(p.N_s),
            FC_a=float(inputs['FC_a'][0]),
            GWP_annual_ops=float(inputs['GWP_annual_ops'][0]),
        )

        evtol = results[-1]
        outputs['FoM_time_rating'] = float(np.asarray(evtol['Time Rating (0-1)']))
        outputs['FoM_co2_rating'] = float(np.asarray(evtol['CO2 Rating (0-1)']))
        outputs['FoM_energy_rating'] = float(np.asarray(evtol['Energy Rating (0-1)']))
        outputs['FoM_cost_rating'] = float(np.asarray(evtol['Cost Rating (0-1)']))
        outputs['FoM'] = float(np.asarray(evtol['FoM']))
