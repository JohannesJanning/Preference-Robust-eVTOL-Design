import openmdao.api as om


class UtilityComp(om.ExplicitComponent):
    """Aggregate cost/GWP ratings and profit utility into a single objective."""

    def initialize(self):
        self.options.declare('parameters')

    def setup(self):
        self.add_input('FoM_cost_rating', val=0.0)
        self.add_input('FoM_co2_rating', val=0.0)
        self.add_input('Annual_Profit', val=0.0)
        self.add_input('TOC_flight', val=0.0)
        self.add_input('GWP_flight', val=0.0)


        self.add_output('Profit_Utility', val=0.0)
        self.add_output('Utility', val=0.0)
        self.add_output('Cost_Rating_Norm', val=0.0)
        self.add_output('GWP_Rating_Norm', val=0.0)
        self.add_output('Profit_Rating_Norm', val=0.0)
        self.add_output('Cost_Utility', val=0.0)
        self.add_output('GWP_Utility', val=0.0)
        self.add_output('Utility_Norm', val=0.0)
        self.add_output('Profit_Utility_Value', val=0.0)
        self.add_output('Cost_Utility_Norm', val=0.0)
        self.add_output('GWP_Utility_Norm', val=0.0)
        self.add_output('Profit_Utility_Norm', val=0.0)


        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        p = self.options['parameters']

        cost_rating = float(inputs['FoM_cost_rating'][0])
        gwp_rating = float(inputs['FoM_co2_rating'][0])
        annual_profit = float(inputs['Annual_Profit'][0])
        cost_flight = float(inputs['TOC_flight'][0])
        gwp_flight = float(inputs['GWP_flight'][0])

        w_cost = float(getattr(p, 'utility_cost_weight', 1.0 / 3.0))
        w_gwp = float(getattr(p, 'utility_gwp_weight', 1.0 / 3.0))
        w_profit = float(getattr(p, 'utility_profit_weight', 1.0 / 3.0))

        p_min = float(getattr(p, 'profit_utility_min', 488427.828))
        p_max = float(getattr(p, 'profit_utility_max', 1523038.89))
        c_factor = float(getattr(p, 'profit_utility_c', 1.0))

        denom = c_factor * p_max - p_min
        if abs(denom) < 1e-12:
            profit_utility = 0.0
        else:
            # Requested formulation without clipping.
            profit_utility = (annual_profit - p_min) / denom

        utility = (
            w_cost * cost_rating
            + w_gwp * gwp_rating
            + w_profit * profit_utility
        )

        cost_rating_min = float(getattr(p, 'cost_rating_min', 0.0))
        cost_rating_max = float(getattr(p, 'cost_rating_max', 1.0))
        cost_rating_norm = (cost_rating_max - cost_flight) / (cost_rating_max - cost_rating_min)

        gwp_rating_min = float(getattr(p, 'gwp_rating_min', 0.0))
        gwp_rating_max = float(getattr(p, 'gwp_rating_max', 1.0))
        gwp_rating_norm = (gwp_rating_max - gwp_flight) / (gwp_rating_max - gwp_rating_min)

        profit_rating_min = float(getattr(p, 'profit_rating_min', 0.0))
        profit_rating_max = float(getattr(p, 'profit_rating_max', 1.0))
        profit_rating_norm = (annual_profit - profit_rating_min) / (profit_rating_max - profit_rating_min)

        utility_norm = (
            w_cost * cost_rating_norm
            + w_gwp * gwp_rating_norm
            + w_profit * profit_rating_norm
        )


        cost_utility = w_cost * cost_rating 
        gwp_utility = w_gwp * gwp_rating
        profit_utility_value = w_profit * profit_utility

        cost_utility_norm = w_cost * cost_rating_norm
        gwp_utility_norm = w_gwp * gwp_rating_norm
        profit_utility_norm = w_profit * profit_rating_norm

        outputs['Profit_Utility_Value'] = profit_utility_value
        outputs['Cost_Utility_Norm'] = cost_utility_norm
        outputs['GWP_Utility_Norm'] = gwp_utility_norm
        outputs['Profit_Utility_Norm'] = profit_utility_norm

        outputs['Profit_Utility'] = profit_utility
        outputs['Utility'] = utility
        outputs['Cost_Rating_Norm'] = cost_rating_norm
        outputs['GWP_Rating_Norm'] = gwp_rating_norm
        outputs['Profit_Rating_Norm'] = profit_rating_norm
        outputs['Cost_Utility'] = cost_utility
        outputs['GWP_Utility'] = gwp_utility
        outputs['Utility_Norm'] = utility_norm