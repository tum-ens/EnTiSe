solargainsiso13790
==================

Overview
--------

Calculates net solar gains: Solar Irradiance (Gains) - Sky Radiation (Losses).

This method extends SolarGainsPVLib to include the long-wave radiation heat loss
to the sky, as required by ISO 13790. It calculates solar gains from both glazed and opaque surfaces,
and subtracts the sky radiation losses to determine the net solar gains for a building.
Please note that this implementation uses simplified assumptions for certain parameters
as per ISO 13790 guidelines.


Key facts
---------

- Method key: ``SolarGainsISO13790``


Requirements
------------

Required keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``id``
- ``latitude[degree]``
- ``longitude[degree]``
- ``H_tr_em[W K-1]``



Optional keys (specify in objects)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``H_tr_op_sky[W K-1]``



Required data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- ``weather``
- ``windows``



Optional data (specify in data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


- None


Outputs
-------

Summary metrics
~~~~~~~~~~~~~~~


- None


Timeseries columns
~~~~~~~~~~~~~~~~~~


- None


Public methods
--------------


- get_input_data

  .. code-block:: python

         def get_input_data(self, obj, data):
           # Get standard inputs from parent
           inputs = super().get_input_data(obj, data)

           # Add thermal envelope properties required for sky loss
           # ISO 13790: phi_r = R_se * U * A * h_r * dT_er
           # We use H_tr (U*A) as the proxy for U*A.
           inputs["H_tr_em"] = float(obj.get(O.H_TR_EM))
           h_sky = obj.get(O.H_TR_OP_SKY)

           # If not provided, estimate using 70% factor
           if h_sky is None:
               h_sky = inputs["H_tr_em"] * 0.7  # Default heuristic: Excludes non-sky facing surfaces (e.g., ground)
           inputs["H_tr_op_sky"] = float(h_sky)
           return inputs


- run

  .. code-block:: python

         def run(self, weather, windows, latitude, longitude, H_tr_em, H_tr_op_sky):
           # All constants cast to float32 so downstream arithmetic stays in
           # float32 and avoids the copy that would otherwise land at the return.

           # 1. Calculate Glazed Surface Solar Gains (Windows)
           df_gains = super().run(weather, windows, latitude, longitude)
           FRAME_FACTOR = np.float32(0.2)  # p.70 - 11.4.5: Frame area fraction
           F_W = np.float32(0.9)  # p.73 - 11.4.2: Non-normal incidence correction
           df_gains = df_gains * F_W * (np.float32(1.0) - FRAME_FACTOR)  # p.67 - 11.3.3: Glazed gains
           gains_windows = df_gains[O.GAINS_SOLAR].to_numpy(dtype=np.float32, copy=False)

           # 2. Calculate Opaque Surface Solar Gains (Walls, Roofs)
           # We approximate I_sol with GHI and use H_tr_op_sky as U*A for sky-facing surfaces.
           R_SE = np.float32(0.04)  # External surface resistance [m2K/W] (simplification)
           F_R = np.float32(0.5)  # p. Form factor to sky (0.5 = vertical, 1.0 = horizontal) (simplification)
           ALPHA_OP = np.float32(0.6)  # Solar absorption coefficient (Standard default)
           F_SH_OP = np.float32(1.0)  # Shading factor
           H_tr_op_sky_f = np.float32(H_tr_op_sky)
           I_global = weather[C.SOLAR_GHI].to_numpy(dtype=np.float32, copy=False)
           gains_opaque = R_SE * H_tr_op_sky_f * ALPHA_OP * F_SH_OP * I_global * F_R  # p.68 - 11.3.4

           # 3. Determine Sky Radiation Losses
           # We approximate (U*A) with H_tr_op_sky
           H_R = np.float32(5.0)  # p.73 - 11.4.6: External radiative coefficient [W/m2K]
           DT_ER = np.float32(11.0)  # p.73 - 11.4.6: Average air-sky temperature difference [K]
           loss = F_R * R_SE * H_tr_op_sky_f * H_R * DT_ER  # p.69 - 11.3.5

           # 4. Net Gains
           net_gains = gains_windows + gains_opaque - loss

           return pd.DataFrame({O.GAINS_SOLAR: net_gains}, index=weather.index)
