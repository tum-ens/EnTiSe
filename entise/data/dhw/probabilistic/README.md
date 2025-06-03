# Probabilistic DHW Data Files

This directory contains data files used by the probabilistic DHW method to generate domestic hot water demand time series.

## Data Sources

### DHW Demand Files

#### `dhw_demand_by_dwelling.csv`
- **Source**: Jordan & Vajen (2001): Realistic Domestic Hot-Water Profiles in Different Time Scales
- **Description**: Annual water consumption per square meter of dwelling area, with standard deviation for stochastic modeling.
- **Units**: m³ per m² per year

#### `dhw_demand_by_occupants.csv`
- **Source**: Hendron & Burch (2007): Development of Standardized Domestic Hot Water Event Schedules for Residential Buildings (NREL)
- **Description**: Daily water consumption based on number of occupants, with standard deviation for stochastic modeling.
- **Units**: liters per day

#### `dhw_demand_by_household_type.csv`
- **Source**: IEA Annex 42: The Simulation of Building-Integrated Fuel Cell and Other Cogeneration Systems
- **Description**: Daily water consumption based on household type, with standard deviation for stochastic modeling.
- **Units**: liters per day

### DHW Activity Files

#### `dhw_activity.csv`
- **Source**: Jordan & Vajen (2001): Realistic Domestic Hot-Water Profiles in Different Time Scales
- **Description**: Weekday activity profiles for different DHW events (shower, sink, bath) with probabilities, durations, and flow rates.
- **Units**: 
  - Probability: dimensionless [0-1]
  - Duration: seconds
  - Flow rate: liters per second

#### `dhw_activity_weekend.csv`
- **Source**: Adapted from Jordan & Vajen (2001) with modifications based on ASHRAE Handbook: HVAC Applications chapter on Service Water Heating
- **Description**: Weekend activity profiles for different DHW events (shower, sink, bath) with probabilities, durations, and flow rates.
- **Units**: 
  - Probability: dimensionless [0-1]
  - Duration: seconds
  - Flow rate: liters per second

### Other Data Files

#### `cold_water_temperature.csv`
- **Source**: VDI 4655: Reference load profiles of single-family and multi-family houses for the use of CHP systems
- **Description**: Monthly cold water temperature variations for more accurate energy demand calculations.
- **Units**: °C

## References

1. Jordan, U., & Vajen, K. (2001). Realistic Domestic Hot-Water Profiles in Different Time Scales. Universität Marburg.
2. Hendron, R., & Burch, J. (2007). Development of Standardized Domestic Hot Water Event Schedules for Residential Buildings. National Renewable Energy Laboratory (NREL).
3. IEA Annex 42: The Simulation of Building-Integrated Fuel Cell and Other Cogeneration Systems.
4. ASHRAE Handbook: HVAC Applications chapter on Service Water Heating.
5. VDI 4655: Reference load profiles of single-family and multi-family houses for the use of CHP systems.