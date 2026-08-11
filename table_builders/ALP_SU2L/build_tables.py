from .config import (
    MASSES_GEV,
    COUPLING_NORMALIZATION_GEV_INV,
)

from .production import build_B_to_Xa_tables
from .lifetime import (
    write_decay_json,
    write_lifetime_table,
)


def main():
    print("Building ALP SU(2)_L EventCalc tables")
    print(f"Number of masses: {len(MASSES_GEV)}")
    print(f"Reference coupling: c_W/f_a = {COUPLING_NORMALIZATION_GEV_INV:.6e} GeV^-1")

    print("\n1. Building production and kinematic tables")
    production_folder = build_B_to_Xa_tables()

    print("\n2. Building lifetime table")
    lifetime_path = write_lifetime_table()

    print("\n3. Building EventCalc decay JSON")
    decay_json_path = write_decay_json()

    print("\nAll ALP SU(2)_L tables completed.")
    print(f"Production output: {production_folder}")
    print(f"Lifetime output:   {lifetime_path}")
    print(f"Decay JSON:        {decay_json_path}")


if __name__ == "__main__":
    main()
