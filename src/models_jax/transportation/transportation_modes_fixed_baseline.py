import numpy as np


def transportation_mode_comparison_fixed_baseline(
    t_tot,
    e_trip,
    D_trip,
    toc_flight,
    time_weight,
    co2_weight,
    energy_weight,
    costs_weight,
    gwp_flight,
    LF,
    N_s,
    FC_a, 
    GWP_annual_ops,
):
    """Compare eVTOL against fixed ground/air baseline bounds.

    Unlike the default comparison model, min/max bounds for each criterion
    are computed from baseline modes only (excluding eVTOL), then reused for
    eVTOL rating. This removes self-referential normalization.
    """

    time_requirement = t_tot / 60.0
    denom = max(LF * N_s * D_trip, 1e-12)
    evtol_co2_kg_skm = GWP_annual_ops / (denom * FC_a)
    # evtol_co2_kg_skm = gwp_flight / denom
    evtol_energy_wh_skm = e_trip / denom
    evtol_costs_eur_skm = toc_flight / denom

    modes = [
        "Gasonline Vehicle (20%)",
        "Diesel Vehicle (20%)",
        "Electric Vehicle (20%)",
        "Gasonline Vehicle (100%)",
        "Diesel Vehicle (100%)",
        "Public Bus (100%)",
        "Electric Vehicle (100%)",
        "Train (100%)",
        "Diesel Vehicle (26%)",
        "Electric Vehicle (26%)",
        "Gasonline Vehicle (26%)",
        "Public Bus (60%)",
        "Train (50%)",
        "eVTOL",
    ]

    co2_kg_skm = np.array(
        [
            0.157,
            0.128,
            0.065,
            0.031,
            0.026,
            0.013,
            0.013,
            0.007,
            0.099,
            0.050,
            0.120,
            0.022,
            0.012,
            evtol_co2_kg_skm,
        ],
        dtype=float,
    )
    energy_consumption = np.array(
        [
            632.40,
            480.00,
            172.70,
            126.48,
            96.00,
            49.42,
            34.54,
            57.02,
            369.23,
            132.85,
            486.46,
            82.84,
            114.04,
            evtol_energy_wh_skm,
        ],
        dtype=float,
    )
    costs_eur_skm = np.array(
        [
            0.117,
            0.083,
            0.105,
            0.023,
            0.017,
            0.060,
            0.021,
            0.200,
            0.064,
            0.081,
            0.090,
            0.104,
            0.402,
            evtol_costs_eur_skm,
        ],
        dtype=float,
    )

    velocities = {
        "Cars_up_to_60km": 60.0,
        "Cars_above_60km": 85.0,
        "Airplane_below_400km": 74.0,
        "Airplane_above_400km": 151.0,
        "Bicycle": 18.8,
        "Public_Bus_up_to_60km": 39.7,
        "Public_Bus_above_60km": 64.0,
        "Train_up_to_60km": 49.1,
        "Train_above_60km": 99.0,
    }

    circuity = {
        "Cars_up_to_180km": 1.30,
        "Cars_above_180km": 1.20,
        "Train_all": 1.20,
        "Airplane_all": 1.05,
        "Bicycle_all": 1.28,
        "Bus_up_to_100km": 1.60,
        "Bus_above_100km": 1.25,
    }

    adjusted_dist = np.zeros(len(modes), dtype=float)
    for i, m in enumerate(modes):
        if "Vehicle" in m:
            adjusted_dist[i] = D_trip * (
                circuity["Cars_up_to_180km"] if D_trip <= 180 else circuity["Cars_above_180km"]
            )
        elif "Public Bus" in m:
            adjusted_dist[i] = D_trip * (
                circuity["Bus_up_to_100km"] if D_trip <= 100 else circuity["Bus_above_100km"]
            )
        elif "Train" in m:
            adjusted_dist[i] = D_trip * circuity["Train_all"]
        else:
            adjusted_dist[i] = D_trip

    co2_total = co2_kg_skm * adjusted_dist
    energy_total = energy_consumption * adjusted_dist
    costs_total = costs_eur_skm * adjusted_dist

    time_demand = np.zeros(len(modes), dtype=float)
    for i, m in enumerate(modes):
        dist = adjusted_dist[i]
        if "Vehicle" in m:
            v = velocities["Cars_up_to_60km"] if D_trip <= 60 else velocities["Cars_above_60km"]
            time_demand[i] = dist / v * 60.0
        elif "Public Bus" in m:
            v = velocities["Public_Bus_up_to_60km"] if D_trip <= 60 else velocities["Public_Bus_above_60km"]
            time_demand[i] = dist / v * 60.0
        elif "Train" in m:
            v = velocities["Train_up_to_60km"] if D_trip <= 60 else velocities["Train_above_60km"]
            time_demand[i] = dist / v * 60.0
        else:
            time_demand[i] = time_requirement

    baseline_slice = slice(0, len(modes) - 1)

    def calc_rating_from_baseline(x):
        base = x[baseline_slice]
        xmin = float(np.min(base))
        xmax = float(np.max(base))
        span = max(xmax - xmin, 1e-12)
        return (0 * (x - xmin) - 1 * (x - xmax)) / span

    time_rating = calc_rating_from_baseline(time_demand)
    co2_rating = calc_rating_from_baseline(co2_total)
    energy_rating = calc_rating_from_baseline(energy_total)
    cost_rating = calc_rating_from_baseline(costs_total)

    FoM = (
        time_weight * time_rating
        + co2_weight * co2_rating
        + energy_weight * energy_rating
        + costs_weight * cost_rating
    )

    results = []
    for i, mode in enumerate(modes):
        results.append(
            {
                "Mode (LF)": mode,
                "FoM": float(FoM[i]),
                "Time (min)": float(time_demand[i]),
                "Time Rating (0-1)": float(time_rating[i]),
                "CO2 (kg)": float(co2_total[i]),
                "CO2 Rating (0-1)": float(co2_rating[i]),
                "Energy (Wh)": float(energy_total[i]),
                "Energy Rating (0-1)": float(energy_rating[i]),
                "Cost (EUR)": float(costs_total[i]),
                "Cost Rating (0-1)": float(cost_rating[i]),
            }
        )

    return results
