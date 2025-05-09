# User-Defined DHW Data Files

This directory is for user-provided data files for the probabilistic DHW method.

## Supported File Types

You can place the following types of files in this directory:

### DHW Demand Files

1. `dhw_demand_by_dwelling.csv`: Demand data based on dwelling size
   - Required columns: `dwelling_size`, `m3_per_m2_a`, `sigma`

2. `dhw_demand_by_occupants.csv`: Demand data based on number of occupants
   - Required columns: `occupants`, `liters_per_day`, `sigma`

3. `dhw_demand_by_household_type.csv`: Demand data based on household type
   - Required columns: `household_type`, `liters_per_day`, `sigma`

### DHW Activity Files

1. `dhw_activity.csv`: Weekday activity profiles
   - Required columns: `day`, `time`, `event`, `probability`, `duration`, `flow_rate`, `sigma_duration`, `sigma_flow_rate`

2. `dhw_activity_weekend.csv`: Weekend activity profiles
   - Required columns: `day`, `time`, `event`, `probability`, `duration`, `flow_rate`, `sigma_duration`, `sigma_flow_rate`

### Other Files

1. `cold_water_temperature.csv`: Monthly cold water temperature data
   - Required columns: `month`, `temperature_c`

## Usage

To use your custom data files, place them in this directory with the exact filenames listed above. The probabilistic DHW method will automatically use your files instead of the default ones.

Alternatively, you can specify the path to your custom files using the following parameters in your objects.csv file:

```csv
id,dhw,weather,dhw_demand_file,dhw_activity_file
1,ProbabilisticDHW,weather,path/to/your/demand_file.csv,path/to/your/activity_file.csv
```

## File Format Examples

### dhw_demand_by_dwelling.csv
```csv
dwelling_size,m3_per_m2_a,sigma
0,0.28,0.14
40,0.25,0.23
55,0.26,0.10
...
```

### dhw_activity.csv
```csv
day,time,event,probability,duration,flow_rate,sigma_duration,sigma_flow_rate
0,00:00:00,shower,0.007,300,0.133,120,0.033
0,00:00:00,sink,0.015,60,0.067,20,0.015
...
```

### cold_water_temperature.csv
```csv
month,temperature_c
1,8.0
2,7.5
...
```