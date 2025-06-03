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
URL: [https://www.researchgate.net/publication/257356121_Realistic_Domestic_Hot-Water_Profiles_in_Different_Time_Scales](https://www.researchgate.net/publication/257356121_Realistic_Domestic_Hot-Water_Profiles_in_Different_Time_Scales)

### Hendron & Burch (2007)
Hendron, R., & Burch, J. (2007). Development of Standardized Domestic Hot Water Event Schedules for Residential Buildings. National Renewable Energy Laboratory (NREL).  
URL: [https://www.nrel.gov/docs/fy08osti/40874.pdf](https://www.nrel.gov/docs/fy08osti/40874.pdf)

### IEA Annex 42
IEA/ECBCS Annex 42. (2007). The Simulation of Building-Integrated Fuel Cell and Other Cogeneration Systems. International Energy Agency.  
URL: [https://www.iea-ebc.org/projects/project?AnnexID=42](https://www.iea-ebc.org/projects/project?AnnexID=42)

### ASHRAE Handbook
ASHRAE. (2019). ASHRAE Handbook - HVAC Applications. Chapter 51: Service Water Heating. American Society of Heating, Refrigerating and Air-Conditioning Engineers.  
URL: [https://www.ashrae.org/technical-resources/ashrae-handbook](https://www.ashrae.org/technical-resources/ashrae-handbook)

### VDI 4655
VDI 4655. (2008). Reference load profiles of single-family and multi-family houses for the use of CHP systems. Verein Deutscher Ingenieure (Association of German Engineers).  
URL: [https://www.vdi.de/richtlinien/details/vdi-4655-reference-load-profiles-of-single-family-and-multi-family-houses-for-the-use-of-chp-systems](https://www.vdi.de/richtlinien/details/vdi-4655-reference-load-profiles-of-single-family-and-multi-family-houses-for-the-use-of-chp-systems)

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
