import openmdao.api as om
import numpy as np

from src.models_jax.noise.SPL_hover import tonal_noise_hover, hover_rotor_rpm


class NoiseComp(om.ExplicitComponent):
    """Compute hover tonal SPL for use as an optimization constraint."""

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        self.add_input('MTOM', val=1500.0)
        self.add_input('P_req_total_hover', val=0.0)
        self.add_input('r_hover', val=1.0)

        self.add_output('SPL_hover', val=0.0)
        self.add_output('RPM_hover', val=0.0)

        # Keep this robust initially; can be replaced by analytic partials later.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        p = self.options['parameters']

        MTOM = float(inputs['MTOM'][0])
        P_hover_total = float(inputs['P_req_total_hover'][0])
        r_hover = float(inputs['r_hover'][0])

        T_req_total_hover = MTOM * p.g
        T_req_prop_hover = T_req_total_hover / max(float(p.n_prop_vert), 1.0)
        P_req_prop_hover = P_hover_total / max(float(p.n_prop_vert), 1.0)

        spl = tonal_noise_hover(
            T_req_prop_hover,
            P_req_prop_hover,
            r_hover,
            p.rho,
            p.n_prop_vert,
            p.n_blade_vert,
            p,
        )

        rpm_hover = hover_rotor_rpm(T_req_prop_hover, r_hover, p.rho, p.C_T_hover)

        outputs['SPL_hover'] = float(np.asarray(spl))
        outputs['RPM_hover'] = float(np.asarray(rpm_hover))
