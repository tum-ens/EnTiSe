"""
Probabilistic DHW (Domestic Hot Water) method for time series generation.

This module implements a probabilistic method for generating domestic hot water demand time series
based on either dwelling size, number of occupants, or household type.

This is the main facade class that selects the appropriate submethod based on the parameters provided.
"""

import os
import logging

from entise.core.base import Method
from entise.constants import Columns as C, Objects as O, Keys as K, Types

# Import all submethods
from entise.methods.dhw.jordan_vajen import JordanVajenDwellingSizeDHW, JordanVajenWeekdayActivityDHW
from entise.methods.dhw.hendron_burch import HendronBurchOccupantsDHW
from entise.methods.dhw.iea_annex42 import IEAAnnex42HouseholdTypeDHW
from entise.methods.dhw.ashrae import ASHRAEWeekendActivityDHW
from entise.methods.dhw.vdi4655 import VDI4655ColdWaterTemperatureDHW
from entise.methods.dhw.user import UserDefinedDHW

logger = logging.getLogger(__name__)

class ProbabilisticDHW(Method):
    """
    Probabilistic DHW demand method.

    This method generates domestic hot water demand time series using a probabilistic approach.
    It can calculate demand based on either dwelling size, number of occupants, or household type.
    
    This class serves as a facade that selects the appropriate submethod based on the parameters provided.
    """
    types = [Types.DHW]
    name = "ProbabilisticDHW"
    required_keys = [O.WEATHER]
    optional_keys = [
        O.DWELLING_SIZE, 
        O.OCCUPANTS, 
        O.HOUSEHOLD_TYPE,
        O.DHW_DEMAND_FILE, 
        O.DHW_ACTIVITY_FILE,
        O.TEMP_COLD,
        O.TEMP_HOT,
        O.SEASONAL_VARIATION,
        O.SEASONAL_PEAK_DAY,
        O.WEEKEND_ACTIVITY,
        O.SOURCE
    ]
    required_timeseries = [O.WEATHER]
    optional_timeseries = []
    output_summary = {
        f'{C.DEMAND}_{Types.DHW}_volume': 'total hot water demand in liters',
        f'{C.DEMAND}_{Types.DHW}_energy': 'total energy demand for hot water in kWh',
    }
    output_timeseries = {
        f'{C.LOAD}_{Types.DHW}_volume': 'hot water demand in liters',
        f'{C.LOAD}_{Types.DHW}_energy': 'energy demand for hot water in W',
    }

    def generate(self, obj, data, ts_type):
        """
        Generate DHW demand time series.

        Parameters:
        -----------
        obj : dict
            Object parameters
        data : dict
            Input data
        ts_type : str
            Time series type

        Returns:
        --------
        dict
            Dictionary with summary and time series data
        """
        # Determine which source to use
        source = obj.get(O.SOURCE, "jordan_vajen")  # Default to Jordan & Vajen
        
        # Select method based on source and parameters
        if source == "jordan_vajen":
            if O.DWELLING_SIZE in obj:
                method = JordanVajenDwellingSizeDHW()
            else:
                # If no specific parameters are provided, use dwelling size as default
                logger.warning("No specific parameters provided for jordan_vajen source. Using dwelling size method with default values.")
                obj_copy = obj.copy()
                obj_copy[O.DWELLING_SIZE] = 100  # Default dwelling size
                method = JordanVajenDwellingSizeDHW()
                return method.generate(obj_copy, data, ts_type)
                
        elif source == "hendron_burch":
            if O.OCCUPANTS in obj:
                method = HendronBurchOccupantsDHW()
            else:
                # If no specific parameters are provided, use occupants as default
                logger.warning("No specific parameters provided for hendron_burch source. Using occupants method with default values.")
                obj_copy = obj.copy()
                obj_copy[O.OCCUPANTS] = 3  # Default number of occupants
                method = HendronBurchOccupantsDHW()
                return method.generate(obj_copy, data, ts_type)
                
        elif source == "iea_annex42":
            if O.HOUSEHOLD_TYPE in obj:
                method = IEAAnnex42HouseholdTypeDHW()
            else:
                # If no specific parameters are provided, use household type as default
                logger.warning("No specific parameters provided for iea_annex42 source. Using household type method with default values.")
                obj_copy = obj.copy()
                obj_copy[O.HOUSEHOLD_TYPE] = "single_adult"  # Default household type
                method = IEAAnnex42HouseholdTypeDHW()
                return method.generate(obj_copy, data, ts_type)
                
        elif source == "vdi4655":
            # VDI4655 is a mixin that needs to be combined with a concrete demand calculation method
            if O.DWELLING_SIZE in obj:
                # Create a class that combines VDI4655 with JordanVajenDwellingSizeDHW
                class VDI4655DwellingSizeDHW(VDI4655ColdWaterTemperatureDHW, JordanVajenDwellingSizeDHW):
                    name = "VDI4655DwellingSizeDHW"
                method = VDI4655DwellingSizeDHW()
            elif O.OCCUPANTS in obj:
                # Create a class that combines VDI4655 with HendronBurchOccupantsDHW
                class VDI4655OccupantsDHW(VDI4655ColdWaterTemperatureDHW, HendronBurchOccupantsDHW):
                    name = "VDI4655OccupantsDHW"
                method = VDI4655OccupantsDHW()
            elif O.HOUSEHOLD_TYPE in obj:
                # Create a class that combines VDI4655 with IEAAnnex42HouseholdTypeDHW
                class VDI4655HouseholdTypeDHW(VDI4655ColdWaterTemperatureDHW, IEAAnnex42HouseholdTypeDHW):
                    name = "VDI4655HouseholdTypeDHW"
                method = VDI4655HouseholdTypeDHW()
            else:
                # If no specific parameters are provided, use dwelling size as default
                logger.warning("No specific parameters provided for vdi4655 source. Using dwelling size method with default values.")
                obj_copy = obj.copy()
                obj_copy[O.DWELLING_SIZE] = 100  # Default dwelling size
                class VDI4655DwellingSizeDHW(VDI4655ColdWaterTemperatureDHW, JordanVajenDwellingSizeDHW):
                    name = "VDI4655DwellingSizeDHW"
                method = VDI4655DwellingSizeDHW()
                return method.generate(obj_copy, data, ts_type)
                
        elif source == "user":
            # Use user-provided method with fallbacks
            method = UserDefinedDHW()
            
        else:
            # If source is not recognized, determine method based on parameters
            if O.HOUSEHOLD_TYPE in obj:
                method = IEAAnnex42HouseholdTypeDHW()
            elif O.DWELLING_SIZE in obj:
                method = JordanVajenDwellingSizeDHW()
            elif O.OCCUPANTS in obj:
                method = HendronBurchOccupantsDHW()
            else:
                raise ValueError("One of dwelling_size, occupants, or household_type must be specified")

        # Check if weekend activity profiles should be used
        if obj.get(O.WEEKEND_ACTIVITY, False):
            # Create a copy of the object with the weekend activity method
            obj_copy = obj.copy()
            obj_copy[O.DHW_ACTIVITY_FILE] = os.path.join('entise', 'data', 'dhw', 'ashrae', 'dhw_activity_weekend.csv')
            return method.generate(obj_copy, data, ts_type)

        # Generate using the selected method
        return method.generate(obj, data, ts_type)