## SU(2)L ALP

The model assumes a non-zero ALP coupling to the SU(2)L gauge fields,
with c_B = c_aPhi = 0 at the chosen matching scale.

### Generate the tables

Place the B-meson momentum file at:
`table_builders/ALP_SU2L/exp1.txt`

Run from the repository root:
`python -m table_builders.ALP_SU2L.build_tables`

### Generated EventCalc inputs

- DoubleDistr-ALP-SU2L.txt
- Emax-ALP-SU2L.txt
- Total-yield-ALP-SU2L.txt
- ctau-ALP-SU2L.txt
- ALP-SU2L-decay.json

### Current limitations

Hadronic ALP decays are not yet included.