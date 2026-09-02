import openmdao.api as om

from src.optimizer.components.aerodynamics_comp import AerodynamicsComp
from src.optimizer.components.energy_comp import EnergyComp
from src.optimizer.components.geometry_comp import GeometryComp
from src.optimizer.components.mass_comp import MassComp
from src.optimizer.components.mtom_implicit import JaxMTOMImplicit
from src.optimizer.components.noise_comp import NoiseComp
from src.optimizer.components.performance_comp import PerformanceComp
from src.optimizer.components.transportation_comp_fixed_baseline import (
    TransportationCompFixedBaseline,
)
from src.optimizer.components.utility_comp import UtilityComp


class eVTOLGroupFixedBaseline(om.Group):
    """Parallel eVTOL group variant with fixed-baseline transportation FoM."""

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        params = self.options['parameters']

        self.add_subsystem('aero', AerodynamicsComp(parameters=params), promotes=['*'])
        self.add_subsystem('mtom', JaxMTOMImplicit(parameters=params))
        self.add_subsystem('geom', GeometryComp(parameters=params), promotes=['*'])
        self.add_subsystem('perf', PerformanceComp(parameters=params), promotes=['*'])
        self.add_subsystem('energy', EnergyComp(parameters=params), promotes=['*'])
        self.add_subsystem('mass', MassComp(parameters=params), promotes=['*'])

        from src.optimizer.components.ops_comp import OpsComp

        self.add_subsystem('ops', OpsComp(parameters=params), promotes=['*'])

        from src.optimizer.components.gwp_comp import GWPComp

        self.add_subsystem('gwp', GWPComp(parameters=params), promotes=['*'])

        from src.optimizer.components.economic_comp import EconomicComp

        self.add_subsystem('economic', EconomicComp(parameters=params), promotes=['*'])
        self.add_subsystem('noise', NoiseComp(parameters=params), promotes=['*'])
        self.add_subsystem('transport', TransportationCompFixedBaseline(parameters=params), promotes=['*'])
        self.add_subsystem('utility', UtilityComp(parameters=params), promotes=['*'])

        self.connect('MTOM_est', 'mtom.MTOM_est')
        self.connect('mtom.MTOM', 'MTOM')

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options['maxiter'] = 50
        self.nonlinear_solver.options['rtol'] = 1e-6
        self.linear_solver = om.DirectSolver(assemble_jac=True)
