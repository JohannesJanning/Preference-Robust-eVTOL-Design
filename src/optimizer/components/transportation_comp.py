import openmdao.api as om
import numpy as np

from src.models_jax.transportation.transportation_modes import transportation_mode_comparison


class TransportationComp(om.ExplicitComponent):
    """Compute eVTOL Figure of Merit (FoM) relative to transport modes."""

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        self.add_input('t_trip', val=3600.0)
        self.add_input('E_trip', val=0.0)
        self.add_input('TOC_flight', val=0.0)
        self.add_input('GWP_flight', val=0.0)

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

        time_weight = float(getattr(p, 'time_weight', 0.25))
        co2_weight = float(getattr(p, 'co2_weight', 0.25))
        energy_weight = float(getattr(p, 'energy_weight', 0.25))
        costs_weight = float(getattr(p, 'costs_weight', 0.25))
        results = transportation_mode_comparison(
            t_tot=t_tot,
            e_trip=e_trip,
            D_trip=float(p.distance_trip_km),
            toc_flight=toc_flight,
            time_weight=time_weight,
            co2_weight=co2_weight,
            energy_weight=energy_weight,
            costs_weight=costs_weight,
            gwp_flight=gwp_flight,
            LF=float(p.LF),
            N_s=int(p.N_s),
        )

        evtol = results[-1]
        outputs['FoM_time_rating'] = float(np.asarray(evtol['Time Rating (0-1)']))
        outputs['FoM_co2_rating'] = float(np.asarray(evtol['CO2 Rating (0-1)']))
        outputs['FoM_energy_rating'] = float(np.asarray(evtol['Energy Rating (0-1)']))
        outputs['FoM_cost_rating'] = float(np.asarray(evtol['Cost Rating (0-1)']))
        outputs['FoM'] = float(np.asarray(evtol['FoM']))
