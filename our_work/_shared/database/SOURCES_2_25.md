# 2-25 um optical-constant sources

The Excel files listed below are headerless `wavelength_um, n, k` tables imported from the
refractiveindex.info database at commit `ff11b5897ef0754b15d939d921eb6c745693cbd1`.
Each selected source is a single tabulated-nk dataset covering the complete 2-25 um interval;
data from different publications are not spliced together.

| File | refractiveindex.info dataset | Sample/data form |
|---|---|---|
| Ag.xlsx | `main/Ag/nk/Hagemann.yml` | Far-IR to X-ray reference data |
| Al.xlsx | `main/Al/nk/Rakic-BB.yml` | Brendel-Bormann fit of metal-film data |
| Al2O3.xlsx | `main/Al2O3/nk/Franta.yml` | 121.5 nm e-beam film |
| Au.xlsx | `main/Au/nk/Ordal.yml` | Bulk/far-IR metal data |
| Cr.xlsx | `main/Cr/nk/Rakic-BB.yml` | Brendel-Bormann fit of metal-film data |
| Cu.xlsx | `main/Cu/nk/Ordal.yml` | Bulk/far-IR metal data |
| HfO2.xlsx | `main/HfO2/nk/Franta.yml` | 112.7 nm e-beam film |
| MgF2.xlsx | `main/MgF2/nk/Franta.yml` | 134.2 nm e-beam film |
| MgO.xlsx | `main/MgO/nk/Synowicki.yml` | Bulk c-MgO oscillator model |
| Si.xlsx | `main/Si/nk/Franta-300K.yml` | Float-zone crystalline Si at 300 K |
| SiO2.xlsx | `main/SiO2/nk/Franta-300C.yml` | 801.9 nm e-beam film |
| Ta2O5.xlsx | `main/Ta2O5/nk/Franta-2025.yml` | E-beam film, multi-sample model |
| Ti.xlsx | `main/Ti/nk/Rakic-BB.yml` | Brendel-Bormann fit of metal-film data |
| TiO2.xlsx | `main/TiO2/nk/Franta.yml` | 99.6 nm e-beam film |
| VO2.xlsx | `main/VO2/nk/Beaini-25C.yml` | 70 nm film at 25 C |
| ZnO.xlsx | `main/ZnO/nk/Querry.yml` | ZnO pellet |
| ZnS.xlsx | `main/ZnS/nk/Querry.yml` | Mineral/material reference data |

`ZnO/Querry.yml` contains two entries at 7.4074 um. They are merged into one row by averaging
their n and k values so that the wavelength column remains strictly increasing for SciPy interpolation.

The other Excel files are intentionally not part of the 2-25 um PSO material set because no
single refractiveindex.info nk dataset was found that covers the complete interval. The PSO
configuration must keep its explicit `materials` allowlist to prevent interpolation outside a
material's measured/modelled range.
