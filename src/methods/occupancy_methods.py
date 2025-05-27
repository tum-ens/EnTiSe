# __author__ = "TUM-Doepfert"
# __license__ = ""
# __maintainer__ = "TUM-Doepfert"
# __email__ = "markus.doepfert@tum.de"
#
# import os.path
#
# from ruamel.yaml import YAML
# from pprint import pprint
# # from input import *
# import pandas as pd
# from pyfmi import load_fmu
# import math
# from statistics import mean
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
#
# pd.options.mode.chained_assignment = None  # default='warn'
#
#
# class Hts:
#
#     def __init__(self, config: dict = None, weather: pd.DataFrame = None, power: pd.DataFrame = None,
#                  path_config: str = "./input.yml", path_weather: str = "./input/weather.csv",
#                  path_power: str = "./input/power.csv", calc_occupancy: bool = True, resolution: int = 900,
#                  occup_resolution: int = None):
#
#         # Load config
#         self.config = config if config is not None else self.load_config(path_config)
#
#         # Set resolution of simulation (in seconds)
#         self.resolution = resolution
#
#         # Create weather file
#         self.weather = weather if weather is not None else self.get_weather(path=path_weather)
#         self.format_weather()
#         self.save_weather()
#
#         # Create electricity file
#         self.power = power if power is not None else self.get_power(path=path_power)
#         self.format_power()
#         self.save_power()
#
#         # Create occupancy file
#         if calc_occupancy:
#             occup_resolution = 1800
#             self.occup = self.prepare_occupancy(occup_resolution)  # TODO: replace once it is done for main branch (1800 s in accordance with the paper it is based on)
#             self.calc_occupancy()
#             self.save_occupancy()
#
#         # Set simulation parameters
#         self.start = 0
#         self.end = self.weather.iat[-1, 0] - self.weather.iat[0, 0]
#         self.step = self.weather.iat[1, 0] - self.weather.iat[0, 0]
#         self.params = self.load_params()
#         self.fmu_path = self.get_fmu_path()
#         self.model = None
#         self.res = None
#         self.results = None
#
#     def format_weather(self) -> pd.DataFrame:
#         """
#         Formats the weather data from the lemlab format to the required SimX format
#
#         :return: weather time series
#         """
#
#         # Change temperature format from Kelvin to Celsius
#         self.weather["temp"] = round(self.weather["temp"] - 273.15, 2)
#
#         # Make index datetime index
#         self.weather.index = pd.to_datetime(self.weather["timestamp"], unit="s")
#         self.weather.index.name = 'datetime'
#
#         # Insert column with 7-day average temperature
#         self.weather["avg_7"] = self.weather["temp"].rolling(window=f"7D").mean().round(1)
#
#         # Change order of columns to SimX format
#         self.weather = self.weather[["timestamp", "temp", "avg_7", "ghi", "dhi", "wind_speed", "wind_dir", "humidity"]]
#
#         return self.weather
#
#     def save_weather(self, path: str = "./simx/weather.txt") -> None:
#         """
#         Saves the weather file in the required SimulationX format
#
#         :param path: path where to save the weather file
#
#         :return: None
#         """
#
#         # Get time values for weather file (start and end)
#         data_time = {
#             "0": 0,
#             "start": self.weather['timestamp'].iat[0],
#             "end": self.weather['timestamp'].iat[-1],
#         }
#         df_time = pd.DataFrame([data_time])
#
#         # Save the weather data in the SimX format
#         with open(path, "w") as file:
#             file.write("#1\n")
#             # Time
#             file.write(f"double\tT_StartEnd ({df_time.shape[0]}, {df_time.shape[1]})\n")
#             df_time.to_csv(file, header=False, index=False, sep='\t', lineterminator='\n')
#             # Weather
#             file.write("\n")
#             file.write(f"double\tWeather({self.weather.shape[0]}, {self.weather.shape[1]})\n")
#             self.weather.to_csv(file, header=False, index=False, sep='\t', lineterminator='\n')
#
#     def format_power(self) -> pd.DataFrame:
#         """
#         Format the power profile from the lemlab format to the required GeoMA format
#
#         :return: edited input data frame that contains the appended columns required for the geoma calculation
#         """
#
#         # Limit range to length of weather file
#         self.power = self.power[(self.power["timestamp"] >= self.weather.iloc[0, 0])
#                                 & (self.power["timestamp"] <= self.weather.iloc[-1, 0])].reset_index(drop=True)
#
#         # Set power to zero if no power profile is to be used
#         if self.config["operation"]["electricity"] is False:
#             self.power["power"] = 0
#
#         return self.power
#
#     def save_power(self, path: str = "./simx/power.txt") -> None:
#         """
#         Saves the power demand time series in the SimX format.
#
#         :param path:   path where to save the power demand
#
#         :return: None
#         """
#
#         # Write independent header and save electricity consumption with the required SimX format
#         self.save_simx_file(df=self.power, header="Pel", path=path)
#
#     def prepare_occupancy(self, resolution: int = None) -> pd.DataFrame:
#         """
#         Formats the occupancy time series to the required format
#
#         :return: Formatted occupancy time series (does not contain occupancy yet)
#         """
#
#         # Append time relevant columns to dataframe
#         self.occup = self.power.copy()
#         self.occup["date"] = pd.to_datetime(self.occup["timestamp"], unit="s")
#         self.occup["time"] = self.occup["date"].dt.strftime("%X")
#         self.occup["hours"] = self.occup["date"].dt.strftime("%H").astype(int)
#
#         # Resample to desired resolution
#         if resolution:
#             self.occup = self.occup.resample(f'{resolution}S', on="date").mean(numeric_only=True).reset_index()
#             self.occup = self.occup.fillna(method="ffill")
#             self.occup = round(self.occup)
#
#         # Adjust data format
#         self.occup["timestamp"] = self.occup["date"].astype("int64") // 10 ** 9
#         self.occup['power'] = self.occup['power'].astype(int)
#         self.occup['hours'] = self.occup['hours'].astype(int)
#
#         return self.occup
#
#     def calc_occupancy(self, method: str = "geoma") -> pd.DataFrame:
#         """
#         Calculate the occupancy based on the power data using the provided method
#
#         :param method: method to be used to calculate the occupancy
#
#         :return: occupancy time series
#         """
#
#         if self.config["operation"]["occupancy"]:
#             if method == "geoma":
#                 self.occup = self._geoma()
#             elif method == "pht":
#                 self.occup = self._pht()
#             else:
#                 raise Warning(f"Method {method} for occupancy calculation unknown.")
#         else:
#             self.occup["occup"] = 1
#
#         # Change resolution to that of simulation
#         if self.occup['timestamp'].iat[1] - self.occup['timestamp'].iat[1] != self.resolution:
#             self.occup = self.occup.resample(f'{self.resolution}S', on="date").mean(numeric_only=True).reset_index()
#             self.occup['timestamp'] = self.occup['timestamp'].interpolate(method='linear')
#             self.occup = self.occup.fillna(method="ffill")
#
#             # Check if length of occupancy is equal to length of power file
#             if len(self.occup) < len(self.power):
#                 # Calculate the difference in length
#                 diff = len(self.power) - len(self.occup)
#
#                 # Append the last occupancy value to the occupancy file with the correct timestamp
#                 for i in range(diff):
#                     self.occup = self.occup.append(self.occup.iloc[-1], ignore_index=True)
#                     self.occup["timestamp"].iat[-1] = self.occup["timestamp"].iat[-2] + self.resolution
#
#
#             # Adjust data format
#             self.occup["timestamp"] = self.occup["timestamp"].astype(int)
#             self.occup['power'] = self.occup['power'].astype(int)
#             self.occup['hours'] = self.occup['hours'].astype(int)
#             self.occup['occup'] = self.occup['occup'].astype(int)
#
#         return self.occup
#
#     def _geoma(self, weight: float = 0.05, threshold_evening: int = 1, start_evening: int = 21,
#                end_night: int = 9, apply_night_rule: bool = True) -> pd.DataFrame:
#         """
#         Calculate the occupancy based on the power data using the GeoMA algorithm
#
#         :param weight: lambda value that weights the impact of the current power demand on the average
#         :param threshold_evening: hours that occupancy needs to be detected in the evening to activate night occupancy
#         :param start_evening: start of evening to start counting the occupancy events required for night occupancy
#         :param end_night: end of the night to switch back to GeoMA detection
#         :param apply_night_rule: whether or not to apply the night rule or stick to continuous GeoMA detection
#
#         :return: occupancy time series
#         """
#
#         # Auxiliary variables to be used in the for-loop
#         step_size = self.occup.at[1, "timestamp"] - self.occup.at[0, "timestamp"]  # step size in seconds
#         occ_evening = 0  # occupancy in the evening: when there is occupancy of at least n hours in the evening
#         #   (defined by threshold_evening), occupancy at night is assumed
#
#         # Check for occupancy at each time step using GeoMA algorithm
#         # Calculate values for first time step outside of the loop: current power demand is moving average
#         mov_avg = self.occup.at[0, "power"]
#         self.occup.at[0, "geoma"] = mov_avg
#         self.occup.at[0, "occup"] = 1
#
#         # Start for-loop
#         for idx, val in self.occup.iloc[1:].iterrows():
#
#             # Update moving average based on the weight for next time step
#             # Note: moving average is the weighted average of the current power demand and the previous moving average
#             #       (standard ratio: 5%/95%)
#             mov_avg = weight * val["power"] + (1 - weight) * mov_avg
#             self.occup.at[idx, "geoma"] = mov_avg
#
#             # Check for occupancy with moving average as threshold: if current value is above it, occupancy is assumed
#             if val["power"] >= mov_avg:
#                 self.occup.at[idx, "occup"] = 1
#             else:
#                 self.occup.at[idx, "occup"] = 0
#
#             # Apply night rule if parameter set to True
#             if apply_night_rule:
#                 self.occup, occ_evening = self.__nightrule(val=val, idx=idx, avg=mov_avg, occ_evening=occ_evening,
#                                                            threshold_evening=threshold_evening,
#                                                            start_evening=start_evening,
#                                                            end_night=end_night, step_size=step_size)
#
#         # Adjust data format
#         self.occup["occup"] = self.occup["occup"].astype("int8")
#
#         return self.occup
#
#     def _pht(self, mag_threshold: float = 0.05, detect_threshold: float = 0.3, threshold_evening: int = 1,
#              start_evening: int = 21, end_night: int = 9, apply_night_rule: bool = True) -> pd.DataFrame:
#         """
#         Calculate the occupancy based on the power data using the PHT algorithm
#
#         :param mag_threshold:
#         :param detect_threshold:
#         :param threshold_evening:
#         :param start_evening:
#         :param end_night:
#         :param apply_night_rule:
#
#         :return: occupancy time series
#         """
#
#         # Auxiliary variables to be used in the for-loop
#         current_state, mt, increasing_mt, decreasing_mt = 0, 0, 0, 0
#         step_size = self.occup.at[1, "timestamp"] - self.occup.at[0, "timestamp"]  # step size in seconds
#         occ_evening = 0  # occupancy in the evening: when there is occupancy of at least n hours in the evening
#         #   (defined by threshold_evening), occupancy at night is assumed
#         avg = mean(self.occup["power"])  # average power demand
#
#         # Start for-loop
#         for idx, val in self.occup.iterrows():
#
#             # Calculate all parameters
#             deviation = val["power"] - avg - mag_threshold
#             mt += deviation
#             increasing_mt = min(increasing_mt, mt)
#             decreasing_mt = max(decreasing_mt, mt)
#             increasing_pht = mt - increasing_mt
#             decreasing_pht = decreasing_mt - mt
#
#             # Detect change and set the occupancy according to whether there was a change or not
#             if increasing_pht > detect_threshold:  # somebody is home
#                 current_state = 1
#                 mt = 0
#             elif decreasing_pht > detect_threshold:  # nobody is home
#                 current_state = 0
#                 mt = 0
#             else:  # state is the same as previous state (no changes)
#                 pass
#
#             # Assign current state to data frame
#             self.occup.at[idx, "occup"] = current_state
#
#             # Apply night rule if parameter set to True
#             if apply_night_rule:
#                 self.occup, occ_evening = self.__nightrule(val=val, idx=idx, avg=avg, occ_evening=occ_evening,
#                                                            threshold_evening=threshold_evening,
#                                                            start_evening=start_evening,
#                                                            end_night=end_night, step_size=step_size)
#
#             # Adjust data format
#             self.occup["occup"] = self.occup["occup"].astype("int8")
#
#             return self.occup
#
#     def __nightrule(self, val: pd.Series, idx: int, avg: float, occ_evening: float, threshold_evening: int = 1,
#                     start_evening: int = 21, end_night: int = 9, step_size: int = 900) -> tuple:
#         """
#         Implements the night-rule that if occupancy is detected in the evening it is assumed that there is occupancy
#         during the night.
#
#         :param val:
#         :param idx:
#         :param avg:
#         :param occ_evening:
#         :param threshold_evening:
#         :param start_evening:
#         :param end_night:
#         :param step_size:
#
#         :return:
#         """
#
#         # Night rule: detect occupancy in the evening and count the number of time steps occupancy is detected
#         if val["hours"] >= start_evening and val["power"] >= avg:
#             occ_evening += step_size / 3600  # evening occupancy counter in hours
#
#         # If evening occupancy counter exceeds threshold, set night occupancy to true
#         if occ_evening >= threshold_evening:
#             occ_night = True
#         else:
#             occ_night = False
#
#         # Night rule: check for occupancy during the night from midnight to morning
#         if 0 <= val["hours"] < end_night:
#             occ_evening = 0  # reset evening occupancy variable
#
#             # If occupancy at night is set to True set occupancy to True regardless of power demand
#             if occ_night:
#                 self.occup.at[idx, "occup"] = 1
#             else:
#                 self.occup.at[idx, "occup"] = 0
#
#         # Night rule: set night occupancy to false during the day
#         if end_night <= val["hours"] < start_evening:
#             occ_night = False
#
#         # Save occupancy evening and night parameters to data frame
#         self.occup.at[idx, "occ_evening"] = occ_evening
#         self.occup.at[idx, "occ_night"] = occ_night
#
#         return self.occup, occ_evening
#
#     def save_occupancy(self, path: str = "./simx/occupancy.txt") -> None:
#         """
#         Saves the occupancy profile time series in the SimX format.
#
#         :param path:   path where to save the power demand
#         :return:            None
#         """
#
#         # Write independent header and save occupancy schedule with the required SimX format
#         self.save_simx_file(df=self.occup[["timestamp", "occup"]], header="presence", path=path)
#
#     def load_params(self) -> dict:
#         """
#         Loads and calculates parameters for the simulation
#
#         :return: parameters for the simulation
#         """
#
#         # Calculate heat flows from config
#         q_heat, q_flow = self.get_heating_params()
#
#         # Get system parameters: heating system exponent & and system temperatures
#         n, temp_flow, temp_return = self.get_system_params()
#
#         # Environment & building parameters
#         params = {  # keys correspond to FMU variable names
#             # Building
#             "building.nFloors": self.config["building"]["floors"],      # no. floors
#             "building.nAp": self.config["building"]["apartments"],      # no. apartments
#             "building.nPeople": self.config["building"]["occupants"],   # no. people in building
#             "building.ALH": self.config["building"]["area"],            # heated living area [m²]
#             "building.cRH": self.config["building"]["height"],          # ceiling height [m]
#             "building.flanking": 0,                                     # flanking buildings [0, 1, 2]
#             "building.outline": True,                                   # outline: True: long-stretched; False: compact
#             "building.livingTZoneInit": self.config["heating"]["setpoint"] + 273.15,  # initial room temperature [K]
#
#             # Heating
#             "building.QHeatNormLivingArea": q_heat,                         # area specific heating power [W/m²]
#             "building.n": n,                                                # heating system exponent
#             "building.TFlowHeatNorm": temp_flow + 273.15,                   # flow temperature [K]
#             "building.TReturnHeatNorm": temp_return + 273.15,               # return temperature [K]
#             "building.TRefHeating": self.config["heating"]["setpoint"] + 273.15,   # reference heating temperature [K]
#             "building.qvMaxLivingZone": q_flow / 1000 / 60,                 # max. flow rate [m³/s]
#
#             # Cooling
#             "building.ActivateCooling": True,                                     # activate cooling
#             "building.TRefCooling": self.config["cooling"]["setpoint"] + 273.15,  # reference cooling temperature [K]
#             "building.TFlowCooling": 15 + 273.15,                                 # flow temperature [K]
#
#             # Operation
#             "building.UseIndividualPresence": True,                                 # use occupancy file
#             "building.PresenceFile": "./simx/occupancy.txt",                        # occupancy file path
#             "building.UseIndividualElecConsumption": True,                          # use electricity file
#             "building.ElConsumptionFile": "./simx/power.txt",                       # electricity file path
#             "building.QPerson": 110,                                                # heat yield per person [W]
#             "building.ActivateNightTimeReduction": self.config["operation"]["nighttime_reduction"],  # use night time
#             "building.Tnight": self.config["operation"]["night_temp"] + 273.15,                      # night temp. [K]
#             "building.NightTimeReductionStart": self.config["operation"]["night_start"] * 3600,      # start night [s]
#             "building.NightTimeReductionEnd": self.config["operation"]["night_end"] * 3600,          # end night [s]
#             "building.VariableTemperatureProfile": self.config["operation"]["occupancy_reduction"],  # use occupancy
#             "building.TMin": self.config["operation"]["occupancy_temp"] + 273.15,    # non-occupancy temp. [K]
#
#             # Insulation
#             "building.livingZoneAirLeak": self.config["insulation"]["leakage"] / 3600,  # air exchange rate [1/s]
#             "building.VentilationLossesMinimum": 1 / 3600,                          # min. ventilation losses [m³/s]
#             "building.VentilationLossesMaximum": 5 / 3600,                          # max. ventilation losses [m³/s]
#             "building.roofInsul": self.config["insulation"]["roof"] / 100,          # additional roof insulation [m]
#             "building.ceilingInsul": self.config["insulation"]["ceiling"] / 100,    # additional ceiling insulation [m]
#             "building.wallInsul": self.config["insulation"]["wall"] / 100,          # additional wall insulation [m]
#             "building.floorInsul": self.config["insulation"]["floor"] / 100,        # additional floor insulation [m]
#             "building.newWindows": self.config["insulation"]["window_new"],         # new windows after construction
#
#             # Environment
#             "environment.Filename": "./simx/weather.txt",                                  # weather file path
#             "environment.Altitude": self.config["environment"]["elevation"],                # elevation [m]
#             "environment.alpha": self.config["environment"]["longitude"] * math.pi / 180,   # longitude [rad]
#             "environment.beta": self.config["environment"]["latitude"] * math.pi / 180,     # latitude [rad]
#             "environment.UnixTimeInit": self.weather.iloc[0, 0],        # initial unix timestamp [s]
#             "environment.cGround": 1.339,                               # spec. ground heat capacity [kJ/(kgK)]
#             "environment.lambdaGround": 1.45,                           # ground heat conductivity [W/(mK)]
#             "environment.rhoGround": 1800,                              # ground density [kg/m³]
#             "environment.GeoGradient": 0.025,                           # geothermal gradient [K]
#             "environment.alphaAirGround": 1.8,                          # transmission coeff. ground to air [W/(m²K)]
#             "environment.cpAir": 1.005,                                 # spec. air heat capacity [kJ/(kgK)]
#             "environment.rhoAir": 1.1839,                               # air density [kg/m³]
#         }
#
#         return params
#
#     def get_heating_params(self) -> tuple:
#         """
#         Calculates the area-specific heating power and the maximum flow rate
#
#         :return: area-specific heating power, maximum flow rate
#         """
#
#         # Get variables from config
#         power = self.config["heating"]["power"]
#         area = self.config["building"]["area"]
#         year = self.config["building"]["year"]
#         heating = self.config["heating"]["system"]
#
#         if year <= 1918:  # model: 1918
#             if heating == "low-temp":
#                 area_f = -5e-6 * area ** 2 + 6.9e-3 * area + 0.1051  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 27  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 8.8e-3 * power_norm + 0.2829
#                 else:
#                     power_f = 0.163
#             elif heating == "mid-temp":
#                 area_f = -5e-6 * area ** 2 + 6.7e-3 * area + 0.1302  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 23  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 1e-5 * power_norm ** 3 + 7e-4 * power_norm ** 2 - 1.66e-2 * power_norm + 0.3035
#                 else:
#                     power_f = 0.161
#             elif heating == "high-temp":
#                 area_f = -5e-6 * area ** 2 + 7e-3 * area + 0.1043  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 22  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 8.2e-3 * power_norm + 0.2558
#                 else:
#                     power_f = 0.168
#             else:
#                 raise Warning(f"{heating} is not a valid variable for 'system'.")
#         elif 1919 <= year < 1949:  # model: 1919
#             if heating == "low-temp":
#                 area_f = -5e-6 * area ** 2 + 7e-3 * area + 0.0975  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 28  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 1e-4 * power_norm ** 2 - 7.9e-3 * power_norm + 0.2822
#                 else:
#                     power_f = 0.167
#             elif heating == "mid-temp":
#                 area_f = -5e-6 * area ** 2 + 6.6e-3 * area + 0.139  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 44  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = -4e-6 * power_norm ** 3 + 4e-4 * power_norm ** 2 - 1.25e-2 * power_norm + 0.294
#                 else:
#                     power_f = 0.162
#             elif heating == "high-temp":
#                 area_f = -5e-6 * area ** 2 + 7e-3 * area + 0.1053  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 25  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 7.4e-3 * power_norm + 0.2553
#                 else:
#                     power_f = 0.17
#             else:
#                 raise Warning(f"{heating} is not a valid variable for 'system'.")
#         elif 1949 <= year < 1958:  # model: 1949
#             if heating == "low-temp":
#                 area_f = -4e-6 * area ** 2 + 6.7e-3 * area + 0.1265  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 25  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 1e-4 * power_norm ** 2 - 8.3e-3 * power_norm + 0.2831
#                 else:
#                     power_f = 0.167
#             elif heating == "mid-temp":
#                 area_f = -5e-6 * area ** 2 + 6.8e-3 * area + 0.1247  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 23  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 1.05e-2 * power_norm + 0.2866
#                 else:
#                     power_f = 0.163
#             elif heating == "high-temp":
#                 area_f = -5e-6 * area ** 2 + 6.9e-3 * area + 0.1033  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 30  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 1e-4 * power_norm ** 2 - 6.4e-3 * power_norm + 0.2491
#                 else:
#                     power_f = 0.168
#             else:
#                 raise Warning(f"{heating} is not a valid variable for 'system'.")
#         elif 1958 <= year < 1969:  # model: 1958
#             if heating == "low-temp":
#                 area_f = -5e-6 * area ** 2 + 6.9e-3 * area + 0.102  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 28  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 8.6e-3 * power_norm + 0.2831
#                 else:
#                     power_f = 0.165
#             elif heating == "mid-temp":
#                 area_f = -5e-6 * area ** 2 + 6.8e-3 * area + 0.1235  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 23  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 1.07e-2 * power_norm + 0.2855
#                 else:
#                     power_f = 0.162
#             elif heating == "high-temp":
#                 area_f = -5e-6 * area ** 2 + 7e-3 * area + 0.0986  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 22  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 8.2e-3 * power_norm + 0.2491
#                 else:
#                     power_f = 0.169
#             else:
#                 raise Warning(f"{heating} is not a valid variable for 'system'.")
#         elif 1969 <= year < 1979:  # model: 1969
#             if heating == "low-temp":
#                 area_f = -5e-6 * area ** 2 + 7.1e-3 * area + 0.1014  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 26  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = -1e-5 * power_norm ** 3 + 7e-4 * power_norm ** 2 - 1.59e-2 * power_norm + 0.2953
#                 else:
#                     power_f = 0.156
#             elif heating == "mid-temp":
#                 area_f = -4e-6 * area ** 2 + 6.5e-3 * area + 0.1404  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 23  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 3e-4 * power_norm ** 2 - 1.04e-2 * power_norm + 0.2636
#                 else:
#                     power_f = 0.157
#             elif heating == "high-temp":
#                 area_f = -5e-6 * area ** 2 + 7.1e-3 * area + 0.0865  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 25  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 7e-3 * power_norm + 0.237
#                 else:
#                     power_f = 0.165
#             else:
#                 raise Warning(f"{heating} is not a valid variable for 'system'.")
#         elif 1979 <= year < 1984:  # model: 1979
#             if heating == "low-temp":
#                 area_f = -5e-6 * area ** 2 + 7e-3 * area + 0.1069  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 25  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = -1e-5 * power_norm ** 3 + 8e-4 * power_norm ** 2 - 1.85e-2 * power_norm + 0.2981
#                 else:
#                     power_f = 0.153
#             elif heating == "mid-temp":
#                 area_f = -5e-6 * area ** 2 + 6.7e-3 * area + 0.1286  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 16  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 8.9e-3 * power_norm + 0.2445
#                 else:
#                     power_f = 0.156
#             elif heating == "high-temp":
#                 area_f = -5e-6 * area ** 2 + 7.1e-3 * area + 0.0855  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 21  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 8.2e-3 * power_norm + 0.2302
#                 else:
#                     power_f = 0.164
#             else:
#                 raise Warning(f"{heating} is not a valid variable for 'system'.")
#         elif 1984 <= year < 1995:  # model: 1984
#             if heating == "low-temp":
#                 area_f = -4e-6 * area ** 2 + 7e-3 * area + 0.1212  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 13  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 4e-4 * power_norm ** 2 - 1.32e-2 * power_norm + 0.2582
#                 else:
#                     power_f = 0.149
#             elif heating == "mid-temp":
#                 area_f = -5e-6 * area ** 2 + 6.8e-3 * area + 0.1194  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 12  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 7.6e-3 * power_norm + 0.22
#                 else:
#                     power_f = 0.153
#             elif heating == "high-temp":
#                 area_f = -6e-6 * area ** 2 + 7.2e-3 * area + 0.0751  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 16  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 5.5e-3 * power_norm + 0.208
#                 else:
#                     power_f = 0.163
#             else:
#                 raise Warning(f"{heating} is not a valid variable for 'system'.")
#         elif 1995 <= year < 2002:  # model: 1995
#             if heating == "low-temp":
#                 area_f = -5e-6 * area ** 2 + 7.2e-3 * area + 0.1  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 10  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f =  4e-4 * power_norm ** 2 - 1.42e-2 * power_norm + 0.2518
#                 else:
#                     power_f = 0.148
#             elif heating == "mid-temp":
#                 area_f = -5e-6 * area ** 2 + 6.8e-3 * area + 0.1157  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 10  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 3e-4 * power_norm ** 2 - 9.1e-3 * power_norm + 0.2153
#                 else:
#                     power_f = 0.152
#             elif heating == "high-temp":
#                 area_f = -5e-6 * area ** 2 + 7.2e-3 * area + 0.0761  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 12  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 3e-4 * power_norm ** 2 - 7.8e-3 * power_norm + 0.2099
#                 else:
#                     power_f = 0.162
#             else:
#                 raise Warning(f"{heating} is not a valid variable for 'system'.")
#         elif 2002 <= year < 2009:  # model: 2002
#             if heating == "low-temp":
#                 area_f = -6e-6 * area ** 2 + 7.2e-3 * area + 0.0761  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 9  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 3e-4 * power_norm ** 2 - 1.33e-2 * power_norm + 0.2406
#                 else:
#                     power_f = 0.147
#             elif heating == "mid-temp":
#                 area_f = -5e-6 * area ** 2 + 6.8e-3 * area + 0.1133  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 8  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = -5e-4 * power_norm ** 2 - 6e-4 * power_norm + 0.1891
#                 else:
#                     power_f = 0.152
#             elif heating == "high-temp":
#                 area_f = -5e-6 * area ** 2 + 7.2e-3 * area + 0.0761  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 13  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 3e-4 * power_norm ** 2 - 7.7e-3 * power_norm + 0.2063
#                 else:
#                     power_f = 0.162
#             else:
#                 raise Warning(f"{heating} is not a valid variable for 'system'.")
#         elif 2009 <= year < 2016:  # model: 2009
#             if heating == "low-temp":
#                 area_f = -5e-6 * area ** 2 + 7.3e-3 * area + 0.1002  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 6  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 2e-4 * power_norm ** 2 - 1.34e-2 * power_norm + 0.2229
#                 else:
#                     power_f = 0.146
#             elif heating == "mid-temp":
#                 area_f = -5e-6 * area ** 2 + 6.9e-3 * area + 0.1039  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 10  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 7e-4 * power_norm ** 2 - 1.18e-2 * power_norm + 0.1955
#                 else:
#                     power_f = 0.151
#             elif heating == "high-temp":
#                 area_f = -6e-6 * area ** 2 + 7.3e-3 * area + 0.0637  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 10  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 5e-4 * power_norm ** 2 - 8.9e-3 * power_norm + 0.1991
#                 else:
#                     power_f = 0.161
#             else:
#                 raise Warning(f"{heating} is not a valid variable for 'system'.")
#         elif 2016 <= year:  # model: 2016
#             if heating == "low-temp":
#                 area_f = -6e-6 * area ** 2 + 7.4e-3 * area + 0.0963  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 6  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 8e-4 * power_norm ** 2 - 1.9e-2 * power_norm + 0.2339
#                 else:
#                     power_f = 0.145
#             elif heating == "mid-temp":
#                 area_f = -5e-6 * area ** 2 + 6.9e-3 * area + 0.1039  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 10  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 7e-4 * power_norm ** 2 - 1.11e-2 * power_norm + 0.1901
#                 else:
#                     power_f = 0.151
#             elif heating == "high-temp":
#                 area_f = -6e-6 * area ** 2 + 7.3e-3 * area + 0.0637  # area factor to calculate normalized power
#                 power_norm = power / area_f  # "normalized" power (normalized at area = 140)
#
#                 power_thres = 15  # power threshold to use different equation
#                 if power_norm < power_thres:
#                     power_f = 1e-4 * power_norm ** 2 - 3.2e-3 * power_norm + 0.1795
#                 else:
#                     power_f = 0.161
#             else:
#                 raise Warning(f"{heating} is not a valid variable for 'system'.")
#         else:
#             raise Warning(f"No model for {year} available.")
#
#         power_f *= area_f
#
#         q_heat, q_flow = self.get_heat_flow(power=power, power_f=power_f)
#
#         return q_heat, q_flow
#
#     def get_heat_flow(self, power: float, power_f: float) -> tuple:
#         """
#         Calculates the area-independent heating power and the maximum flow rate
#
#         :param power: maximum heating power in kW
#         :param power_f: power factor
#
#         :return: area-independent heating power, maximum flow rate
#         """
#
#         # Get heating system from confing
#         heating = self.config["heating"]["system"]
#
#         if heating == "low-temp":
#             x = 2 / 15 * power / power_f - 2
#             q_flow = round(5 / 2 * x + 5, 2)
#             q_heat = round(5 * x + 10, 2)
#         elif heating == "mid-temp":
#             x = 2 / 25 * power / power_f - 30 / 25
#             q_flow = round(5 / 2 * x + 5, 2)
#             q_heat = round(10 * x + 10, 2)
#         elif heating == "high-temp":
#             x = 2 / 35 * power / power_f - 50 / 35
#             q_flow = round(5 / 2 * x + 5, 2)
#             q_heat = round(15 * x + 20, 2)
#         else:
#             raise Warning(f"{heating} is not a valid variable for 'system'.")
#
#         return q_heat, q_flow
#
#     def get_system_params(self) -> tuple:
#         """Returns heating system parameters, i.e. heating system exponent and flow temperatures.
#
#         :return: heating systemp exponent, flow temperature, return temperature
#         """
#
#         # Get heating system from confing
#         heating = self.config["heating"]["system"]
#
#         if heating == "low-temp":
#             n = 1.1
#             temp_flow = 40
#             temp_return = 30
#         elif heating == "mid-temp":
#             n = 1.3
#             temp_flow = 50
#             temp_return = 40
#         elif heating == "high-temp":
#             n = 1.3
#             temp_flow = 70
#             temp_return = 55
#         else:
#             raise Warning(f"{heating} is not a valid variable for 'system'.")
#
#         return n, temp_flow, temp_return
#
#     def get_fmu_path(self) -> str:
#         """Sets the path of the FMU file that is to be used."""
#
#         # Get year from config
#         year = self.config["building"]["year"]
#
#         # Available model years (needs to be sorted list)
#         models = [1918, 1919, 1949, 1958, 1969, 1979, 1984, 1995, 2002, 2009, 2016]
#
#         # Year to use for the house model
#         # Functionality: Looks for the years smaller than 'year' in 'models' and uses len() - 1 to find the index.
#         #   If 'year' is not in list, index = 0 is chosen.
#         built = models[max(0, len([x - year for x in models if x - year <= 0]) - 1)]
#
#         # print(f'Model year used for the simulation: {built}')
#
#         return f"./simx/{built}/model_{built}.fmu"
#
#     def run_simulation(self, log: int = 0, silent: bool = False) -> None:
#         """
#         Runs the simulation
#
#         :param log: If true, a log will be created
#
#         :return: results of the simulation
#         """
#
#         self.model = None
#
#         # Load model
#         self.model = load_fmu(self.fmu_path, kind="CS", log_level=log)
#
#         # Resets the FMU to its original state
#         self.model.reset()
#
#         # Set the parameters
#         self.model.set(list(self.params.keys()), list(self.params.values()))
#
#         # Returns the log information as a list
#         if log:
#             self.model.get_log()
#
#         # Set simulation options
#         opts = self.model.simulate_options()
#         opts["silent_mode"] = silent
#         opts["initialize"] = False
#         opts["ncp"] = math.ceil((self.end - self.start) / self.step)
#         opts["result_file_name"] = "./simx/results.mat"
#
#         # Initialize the model and compute initial values for all variables
#         self.model.initialize()
#
#         # pprint(dir(self.model))
#         # for var in self.model.get_model_variables():
#         #     try:
#         #         print(f'{var}: {self.model.get_variable_start(var)}')
#         #     except:
#         #         print(var)
#         # return
#
#         # Simulate
#         self.res = self.model.simulate(start_time=self.start, final_time=self.end, options=opts)
#
#         self.model.terminate()
#
#     def print_all_keys(self) -> None:
#         """Prints all available parameters to be retrieved from the simulation results."""
#
#         print("FMU-keys:")
#         pprint(self.res.keys())
#
#     def save_results(self, path: str = "./output/results.csv", keys: list = None, si_units: bool = False,
#                      power_from_energy: bool = True) -> pd.DataFrame:
#         """
#         Obtains the desired output results and saves them to a csv file.
#
#         :param path: path to which to save the results
#         :param keys: keys of the output parameters (all keys can be displayed with print_all_keys())
#         :param si_units: should results be returned as SI units or as the units of the input
#         :param power_from_energy: calculate heating and cooling power from energy time series or power time series
#                                   (power time series records momentous power and not average power over time)
#
#         :return: results of the simulation
#         """
#
#         # Use standard set of output if no keys are given
#         keys = keys if keys is not None else ['building.QHeat',
#                                               'building.QCold',
#                                               'building.TZone[2]']
#
#         # Dictionary containing the name and the factor of the respective key
#         # Note: The factor serves to convert back to non-SI units,
#         #           if K2C is given Kelvin is converted to Celsius (no factor but subtraction)
#         #           if None is given there is no conversion
#         names = {
#             # General
#             'time': {'factor': 1,                                                                   # [s]
#                      'name': 'time'},
#
#             # Results
#             'building.UnixTime': {'factor': 1,                                                      # [s]
#                                   'name': 'unixtime'},
#             'building.Heating': {'factor': None,                                                    # [1]
#                                  'name': 'heating'},
#             'building.Cooling': {'factor': None,                                                    # [1]
#                                  'name': 'cooling'},
#             'building.QHeat': {'factor': 1,                                                         # [W]
#                                'name': 'heat_power'},
#             'building.QCold': {'factor': 1,                                                         # [W]
#                                'name': 'cool_power'},
#             'building.EHeat': {'factor': 1 / 3600,                                                  # [Wh]
#                                'name': 'heat_energy'},
#             'building.ECold': {'factor': 1 / 3600,                                                  # [Wh]
#                                'name': 'cool_energy'},
#             'building.Pel': {'factor': 1,                                                           # [W]
#                              'name': 'electricity_power'},
#             'building.Eel': {'factor': 1 / 3600,                                                    # [Wh]
#                              'name': 'electricity_energy'},
#             'building.TZone[1]': {'factor': "K2C",                                            # [°C]
#                                         'name': 'temp_cellar'},
#             'building.TZone[2]': {'factor': "K2C",                                            # [°C]
#                                         'name': 'temp_indoor'},
#             'building.TZone[3]': {'factor': "K2C",                                            # [°C]
#                                         'name': 'temp_attic'},
#
#             # Building
#             'building.nFloors': {'factor': None,                                                    # [1]
#                                  'name': 'n_floors'},
#             'building.nAp': {'factor': None,                                                        # [1]
#                              'name': 'n_apartments'},
#             'building.nPeople': {'factor': None,                                                    # [1]
#                                  'name': 'n_people'},
#             'building.ALH': {'factor': 1,                                                           # [m]
#                              'name': 'area'},
#             'building.cRH': {'factor': 1,                                                           # [m]
#                              'name': 'height'},
#             'building.flanking': {'factor': None,                                                   # [1]
#                                   'name': 'flanking'},
#             'building.outline': {'factor': None,                                                    # [1]
#                                  'name': 'outline'},
#             'building.livingTZoneInit': {'factor': "K2C",                                           # [°C]
#                                          'name': 'temp_init'},
#
#             # Heating
#             'building.QHeatNormLivingArea': {'factor': 1,                                           # [W/m²]
#                                              'name': 'q_heat'},
#             'building.n': {'factor': 1,                                                             # [1]
#                            'name': 'n'},
#             'building.TFlowHeatNorm': {'factor': "K2C",                                             # [°C]
#                                        'name': 'temp_flow'},
#             'building.TReturnHeatNorm': {'factor': "K2C",                                           # [°C]
#                                          'name': 'temp_return'},
#             'building.TRefHeating': {'factor': "K2C",                                               # [°C]
#                                      'name': 'temp_heat_ref'},
#             'building.TRefSetHeating': {'factor': "K2C",                                            # [°C]
#                                         'name': 'temp_heat_ref_set'},
#             'building.qvMaxLivingZone': {'factor': 6e4,                                             # [l/min]
#                                          'name': 'q_flow'},
#
#             # Cooling
#             'building.TRefCooling': {'factor': "K2C",                                               # [°C]
#                                      'name': 'temp_cool_ref'},
#
#             # Operation
#             'building.UseIndividualPresence': {'factor': None,                                      # [1]
#                                                'name': 'use_presence'},
#             'building.UseIndividualElecConsumption': {'factor': None,                               # [1]
#                                                       'name': 'use_power'},
#             'building.QPerson': {'factor': 1,                                                       # [W]
#                                  'name': 'q_person'},
#             'building.ActivateNightTimeReduction': {'factor': None,                                 # [1]
#                                                     'name': 'use_nighttime'},
#             'building.Tnight': {'factor': "K2C",                                                    # [°C]
#                                 'name': 'temp_night'},
#             'building.NightTimeReductionStart': {'factor': 1 / 3600,                                # [h]
#                                                  'name': 'night_start'},
#             'building.NightTimeReductionEnd': {'factor': 1 / 3600,                                  # [h]
#                                                'name': 'night_end'},
#             'building.VariableTemperatureProfile': {'factor': None,                                 # [1]
#                                                     'name': 'use_occupancy'},
#             'building.TMin': {'factor': "K2C",                                                      # [°C]
#                               'name': 'temp_occup'},
#
#             # Insulation
#             'building.livingZoneAirLeak': {'factor': 3600,                                          # [1/h]
#                                            'name': 'leakage'},
#             'building.VentilationLossesMinimum': {'factor': 3600,                                   # [m³/h]
#                                                   'name': 'ventilation_min'},
#             'building.VentilationLossesMaximum': {'factor': 3600,                                   # [m³/h]
#                                                   'name': 'ventilation_max'},
#             'building.roofInsul': {'factor':  1 / 100,                                              # [cm]
#                                    'name': 'insulation_roof'},
#             'building.ceilingInsul': {'factor': 1 / 100,                                            # [cm]
#                                       'name': 'insulation_ceiling'},
#             'building.wallInsul': {'factor':  1 / 100,                                              # [cm]
#                                    'name': 'insulation_wall'},
#             'building.floorInsul': {'factor':  1 / 100,                                             # [cm]
#                                     'name': 'insulation_floor'},
#             'building.newWindows': {'factor': None,                                                 # [1]
#                                     'name': 'windows_new'},
#
#             # Environment
#             'environment.TAmbient': {'factor': "K2C",                                               # [°C]
#                                      'name': 'temp_ambient'},
#             'environment.Altitude': {'factor': 1,                                                   # [m]
#                                      'name': 'elevation'},
#             'environment.alpha': {'factor': 180 / math.pi,                                          # [°]
#                                   'name': 'longitude'},
#             'environment.beta': {'factor': 180 / math.pi,                                           # [°]
#                                  'name': 'latitude'},
#             'environment.UnixTimeInit': {'factor': 1,                                               # [s]
#                                          'name': 'unix_init'},
#             'environment.cGround': {'factor': 1,                                                    # [kJ/(kgK)]
#                                     'name': 'c_ground'},
#             'environment.lambdaGround': {'factor': 1,                                               # [W/(mK)]
#                                          'name': 'lambda_ground'},
#             'environment.rhoGround': {'factor': 1,                                                  # [kg/m³]
#                                       'name': 'rho_ground'},
#             'environment.GeoGradient': {'factor': 1,                                                # [K]
#                                         'name': 'geo_gradient'},
#             'environment.alphaAirGround': {'factor': 1,                                             # [W/(m²K)]
#                                            'name': 'alpha_air_ground'},
#             'environment.cpAir': {'factor': 1,                                                      # [kJ/(kgK)]
#                                   'name': 'cp_air'},
#             'environment.rhoAir': {'factor': 1,                                                     # [kg/m³]
#                                    'name': 'rho_air'},
#         }
#
#         # Get electricity and occupancy
#         self.results = self.power.copy()                # electrical power
#         self.results["occup"] = self.occup["occup"]     # occupancy
#
#         # Get all results defined by the keys
#         for key in keys:
#             self.results[names[key]["name"]] = self.res[key]
#
#             # Convert units if not SI
#             if not si_units:
#                 match names[key]["factor"]:
#                     case None:
#                         pass
#                     case "K2C":
#                         self.results[names[key]["name"]] -= 273.15
#                     case _:
#                         self.results[names[key]["name"]] *= names[key]["factor"]
#
#         # Calculate heating power from energy demand and not from power column, which contains momentous power
#         # Note: Use if accuracy of heating demand necessary, otherwise use power for peak heating power
#         if power_from_energy:
#             # Calculate heating power from energy
#             self.results["heat_power_en"] = self.res["building.EHeat"] * names["building.EHeat"]["factor"]
#             self.results.loc[:, "heat_power_en"] = self.results["heat_power_en"].diff() * 3600 / self.res["time"][1]
#             self.results["heat_power"].iloc[:-1] = self.results["heat_power_en"].iloc[1:]
#
#             # Calculate cooling power from energy
#             self.results["cool_power_en"] = self.res["building.ECold"] * names["building.ECold"]["factor"]
#             self.results.loc[:, "cool_power_en"] = self.results["cool_power_en"].diff() * 3600 / self.res["time"][1]
#             self.results["cool_power"].iloc[:-1] = self.results["cool_power_en"].iloc[1:]
#
#
#             # Column formatting
#             self.results["heat_power"] = round(self.results["heat_power"]).astype("int64")
#             self.results["cool_power"] = round(self.results["cool_power"]).astype("int64")
#
#             # Drop auxiliary columns
#             self.results.drop("heat_power_en", axis=1, inplace=True)
#             self.results.drop("cool_power_en", axis=1, inplace=True)
#
#         # Save results to csv file
#         self.results.to_csv(path, index=False)
#
#         return self.results
#
#     @staticmethod
#     def get_weather(path: str = "./input/weather/weather.csv") -> pd.DataFrame:
#
#         # Read weather file
#         cols = {  # columns to load from df and their dtypes
#             'timestamp': 'int64',
#             'temp': 'float64',
#             'humidity': 'float64',
#             'wind_speed': 'float64',
#             'wind_dir': 'int16',
#             'ghi': 'int16',
#             'dhi': 'int16',
#         }
#         df = pd.read_csv(path, usecols=cols.keys(), dtype=cols, float_precision='high')
#         # df = df[(df["timestamp"] >= 1609455600) & (df["timestamp"] <= 1640990700)]
#         # # df.reset_index(inplace=True, drop=True)
#         # print("\nchange back weather")
#
#         return df
#
#     @staticmethod
#     def load_config(path: str) -> dict:
#
#         with open(path) as input_file:
#             return YAML().load(input_file)
#
#     @staticmethod
#     def get_power(path):
#
#         return pd.read_csv(path)
#
#     @staticmethod
#     def save_simx_file(df: pd.DataFrame, header: str, path: str):
#         # Write independent header and save electricity consumption with the required SimX format
#         with open(path, "w") as file:
#             file.write("#1\n")
#             df.to_csv(file, index=False, sep='\t', lineterminator='\n',
#                       header=["double", f"{header}({df.shape[0]}, {df.shape[1]})"])
#
#     @staticmethod
#     def resample_timeseries(data: pd.DataFrame, delta: int) -> pd.DataFrame:
#         """
#         Resamples a timeseries with the index being the unix timestamp.
#         The resampling method (interpolate or mean) is decided based on the comparison between
#         the input delta and the time delta of the input timeseries.
#
#         Parameters:
#         df (pd.DataFrame): The input timeseries
#         delta (int): The desired time delta in seconds between two time steps
#
#         Returns:
#         pd.DataFrame: The resampled timeseries
#         """
#
#         # Calculate the delta of the input timeseries
#         original_delta = int(data.index[1] - data.index[0])
#
#         # Return the original timeseries if the delta is the same
#         if delta == original_delta:
#             return data
#
#         # Convert the index to datetime
#         data.index = pd.to_datetime(data.index, unit='s')
#
#         # Resample the timeseries (mean when delta > original_delta, interpolate otherwise)
#         if delta > original_delta:
#             # Copy the original timeseries
#             resampled = data.copy()
#
#             # Calculate the number of times the original delta can be divided by the desired delta
#             multiple = int(delta / original_delta)
#
#             # Add the shifted timeseries to the original timeseries (fillna(0) to ensure there is no NaN)
#             for i in range(1, multiple):
#                 resampled += data.shift(-i).fillna(0)
#
#             # Calculate the mean of the timeseries by dividing multiple and selecting every nth value
#             resampled = (resampled / multiple)[::multiple]
#         else:
#             # Interpolate the timeseries
#             resampled = data.resample(f'{delta}s').interpolate()
#
#         # Convert the index back to unix timestamp
#         resampled.index = resampled.index.astype(int) // 10 ** 9
#
#         # Convert the data types back to the original ones
#         for col, dtype in data.dtypes.items():
#             resampled[col] = resampled[col].astype(dtype)
#
#         return resampled
#
