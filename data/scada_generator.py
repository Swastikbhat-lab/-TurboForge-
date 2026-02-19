"""
TurboForge - Synthetic SCADA Data Generator
Simulates 50-turbine wind farm sensor data for training and testing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_scada_data(
    n_turbines: int = 50,
    n_hours: int = 8760,  # 1 year
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic SCADA wind farm data.

    Features per turbine (per 10-min interval):
        - wind_speed_ms       : Wind speed (m/s)
        - rotor_rpm           : Rotor rotational speed
        - power_output_kw     : Active power output
        - blade_pitch_deg     : Blade pitch angle
        - nacelle_temp_c      : Nacelle temperature
        - gearbox_temp_c      : Gearbox temperature
        - generator_temp_c    : Generator temperature
        - vibration_x, _y     : Vibration sensors
        - failure_label       : Binary failure flag (next 6h window)
    """
    np.random.seed(seed)
    timestamps = [
        datetime(2023, 1, 1) + timedelta(minutes=10 * i)
        for i in range(n_hours * 6)  # 10-min intervals
    ]
    records = []

    for turbine_id in range(1, n_turbines + 1):
        # Per-turbine baseline variability
        turbine_offset = np.random.uniform(-2, 2)
        wear_factor = np.random.uniform(0.9, 1.1)

        wind_speed = (
            8 + turbine_offset
            + 4 * np.sin(np.linspace(0, 4 * np.pi, len(timestamps)))
            + np.random.normal(0, 1.5, len(timestamps))
        ).clip(0, 25)

        rotor_rpm = (wind_speed * 1.8 + np.random.normal(0, 0.3, len(timestamps))).clip(0, 20)
        power_kw = (
            (wind_speed ** 3) * 0.08 * wear_factor
            + np.random.normal(0, 20, len(timestamps))
        ).clip(0, 2000)

        blade_pitch = (
            10 - wind_speed * 0.3 + np.random.normal(0, 0.5, len(timestamps))
        ).clip(0, 90)

        nacelle_temp = (
            25 + power_kw * 0.005 + np.random.normal(0, 1, len(timestamps))
        )
        gearbox_temp = (
            40 + power_kw * 0.008 * wear_factor + np.random.normal(0, 2, len(timestamps))
        )
        generator_temp = (
            55 + power_kw * 0.01 + np.random.normal(0, 2, len(timestamps))
        )

        vibration_x = np.random.normal(0.02, 0.005, len(timestamps))
        vibration_y = np.random.normal(0.02, 0.005, len(timestamps))

        # Inject failure events (~5% of turbines have higher failure rate)
        failure_prob = 0.002 if turbine_id % 20 != 0 else 0.008
        failures = np.zeros(len(timestamps))
        failure_indices = np.where(np.random.rand(len(timestamps)) < failure_prob)[0]
        for idx in failure_indices:
            window = slice(max(0, idx - 36), idx)  # 6h = 36 intervals
            failures[window] = 1
            # Pre-failure signature
            gearbox_temp[window] += np.linspace(0, 15, len(range(*window.indices(len(timestamps)))))
            vibration_x[window] *= 3
            vibration_y[window] *= 2.5

        for i, ts in enumerate(timestamps):
            records.append({
                "timestamp": ts,
                "turbine_id": turbine_id,
                "wind_speed_ms": round(wind_speed[i], 3),
                "rotor_rpm": round(rotor_rpm[i], 3),
                "power_output_kw": round(power_kw[i], 3),
                "blade_pitch_deg": round(blade_pitch[i], 3),
                "nacelle_temp_c": round(nacelle_temp[i], 3),
                "gearbox_temp_c": round(gearbox_temp[i], 3),
                "generator_temp_c": round(generator_temp[i], 3),
                "vibration_x": round(vibration_x[i], 5),
                "vibration_y": round(vibration_y[i], 5),
                "failure_label": int(failures[i]),
            })

    df = pd.DataFrame(records)
    print(f"[DataGen] Generated {len(df):,} records | {n_turbines} turbines | "
          f"Failure rate: {df['failure_label'].mean():.2%}")
    return df


if __name__ == "__main__":
    df = generate_scada_data(n_turbines=50, n_hours=720)  # 30 days for quick test
    df.to_csv("scada_data.csv", index=False)
    print(df.head())
    print(df.describe())
