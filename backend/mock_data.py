"""Generate mock CLSI EP05-A3 style COVID-19 RdRp precision validation Excel data."""

import numpy as np
import pandas as pd
from pathlib import Path

DAYS = ["Day_1", "Day_2", "Day_3", "Day_4", "Day_5"]
MACHINES = ["Machine_A", "Machine_B"]
LOTS = ["Lot_2024A", "Lot_2024B"]
REPLICATES_PER_COMBO = 5

BASE_CT = 28.5
DAY_EFFECT = {"Day_1": 0.0, "Day_2": 0.3, "Day_3": -0.2, "Day_4": 0.5, "Day_5": -0.1}
MACHINE_EFFECT = {"Machine_A": 0.0, "Machine_B": 0.4}
REPEATABILITY_SD = 0.35


def generate_mock_precision_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for day in DAYS:
        for machine in MACHINES:
            for lot in LOTS:
                for rep in range(1, REPLICATES_PER_COMBO + 1):
                    ct = (
                        BASE_CT
                        + DAY_EFFECT[day]
                        + MACHINE_EFFECT[machine]
                        + rng.normal(0, REPEATABILITY_SD)
                    )
                    rows.append(
                        {
                            "Day_Info": day,
                            "Machine_A_B": machine,
                            "Lot_Num": lot,
                            "Replicate_No": rep,
                            "Target_Gen_Ct": round(ct, 2),
                            "Sample_ID": f"SARS2-RdRp-{day[-1]}{machine[-1]}{rep}",
                            "Operator": f"Tech_{(rep % 2) + 1}",
                            "Run_Time": f"2024-03-{int(day[-1]):02d} 09:{rep * 10:02d}",
                        }
                    )

    return pd.DataFrame(rows)


def save_mock_excel(output_path: str | Path | None = None) -> Path:
    if output_path is None:
        output_path = Path(__file__).parent / "mock_covid_precision.xlsx"
    else:
        output_path = Path(output_path)

    df = generate_mock_precision_data()
    df.to_excel(output_path, index=False, sheet_name="Precision_Study")
    print(f"Generated {len(df)} rows -> {output_path}")
    return output_path


if __name__ == "__main__":
    save_mock_excel()
