# DHW (Domestic Hot Water) Data Files

This directory contains data files used by the probabilistic DHW methods to generate domestic hot water demand time series.

## Directory Structure

The data files are organized by source:

- `jordan_vajen/`: Data files from Jordan & Vajen (2001)
  - `dhw_demand_by_dwelling.csv`: Annual water consumption per square meter of dwelling area
  - `dhw_activity.csv`: Weekday activity profiles for different DHW events

- `hendron_burch/`: Data files from Hendron & Burch (2007)
  - `dhw_demand_by_occupants.csv`: Daily water consumption based on number of occupants

- `iea_annex42/`: Data files from IEA Annex 42
  - `dhw_demand_by_household_type.csv`: Daily water consumption based on household type

- `ashrae/`: Data files from ASHRAE Handbook
  - `dhw_activity_weekend.csv`: Weekend activity profiles for different DHW events

- `vdi4655/`: Data files from VDI 4655
  - `cold_water_temperature.csv`: Monthly cold water temperature variations

- `user/`: Directory for user-provided data files
  - See `user/README.md` for details on how to provide your own data files

## Data Sources

### Jordan & Vajen (2001)
Jordan, U., & Vajen, K. (2001). Realistic Domestic Hot-Water Profiles in Different Time Scales. Universität Marburg.

### Hendron & Burch (2007)
Hendron, R., & Burch, J. (2007). Development of Standardized Domestic Hot Water Event Schedules for Residential Buildings. National Renewable Energy Laboratory (NREL).

### IEA Annex 42
IEA Annex 42: The Simulation of Building-Integrated Fuel Cell and Other Cogeneration Systems.

### ASHRAE Handbook
ASHRAE Handbook: HVAC Applications chapter on Service Water Heating.

### VDI 4655
VDI 4655: Reference load profiles of single-family and multi-family houses for the use of CHP systems.

## Usage

These data files are used by the probabilistic DHW methods in the `entise.methods.dhw` package. The methods are organized by source to match this data organization.

To use a specific source's method, specify the source in your objects.csv file:

```csv
id,dhw,weather,source,dwelling_size
1,ProbabilisticDHW,weather,jordan_vajen,100
```

Available sources:
- `jordan_vajen`: Based on dwelling size
- `hendron_burch`: Based on number of occupants
- `iea_annex42`: Based on household type
- `vdi4655`: Includes seasonal cold water temperature variations
- `user`: Uses user-provided data files with fallbacks

If no source is specified, the method will automatically select the appropriate source based on the parameters provided (dwelling_size, occupants, or household_type).