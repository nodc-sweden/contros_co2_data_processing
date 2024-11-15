import pandas as pd
import numpy as np
from datetime import datetime


from file_reader import (list_files, read_files, combine_data_in_subfolders, merge_zero_cycle_with_measurements,
                         read_ferrybox_files, merge_ferrybox_with_pco2)
from calibration import check_calibration_data
from calculations import (calculation_of_s2beam, calculation_of_median_s2beam_z, interpolate_s2beam_z,
                          calculation_of_s_dc, calculation_of_sproc, calculation_of_drift_corrected_k1k2k3,
                          calculation_of_xco2wet, calculation_of_pco2wet,
                          calculation_of_fco2wet, calculation_of_pco2_fco2_at_sst)
from flag_state import (get_quality_of_processed_xco2wet, get_quality_of_processed_pco2wet, get_state_wash,
                        get_state_standby, get_state_extended_flush)
from file_exporter import export_processed_data, export_pcof_fco2_at_sst
from plot_data import plot_scatter


working_folder = r'C:\git\pCO2_processing\example_data\CO2FT_A'

# read zero cycle
zero_cycle_files = list_files(working_folder)
zero_cycle, zero_quality_flags, zero_info_flags = read_files(zero_cycle_files)

# read measurements
all_data, measurements, measurements_quality_flags, quality_flags, info_flags = (
    combine_data_in_subfolders(working_folder))

# read calibration data
serial_number = "CO2FT-0918-001"
calibration_dates = ["2019-10-16", "2021-04-28", "2023-05-26"]
calibration_data = check_calibration_data(serial_number, calibration_dates)

pre_calibration_date = {}
pre_k1 = np.nan
pre_k2 = np.nan
pre_k3 = np.nan
post_calibration_date = {}

for date in calibration_data:
    post_calibration_date = date
    post_k1 = calibration_data[date]["k1"]
    post_k2 = calibration_data[date]["k2"]
    post_k3 = calibration_data[date]["k3"]
    if measurements["time_series"][0] > calibration_data[date]["calibration_date"]:
        ft_sensor = calibration_data[date]["ft_sensor"]
        f = calibration_data[date]["f"]
        p0 = calibration_data[date]["p0"]
        t0 = calibration_data[date]["t0"]
        pre_calibration_date = date
        pre_k1 = calibration_data[date]["k1"]
        pre_k2 = calibration_data[date]["k2"]
        pre_k3 = calibration_data[date]["k3"]
    else:
        break
print('\nCalibration data are from pre calibration date:')
print(pre_calibration_date)
print('\nand post calibration date:')
print(post_calibration_date)

# calculate s'2beam
zero_cycle["s2beam"] = calculation_of_s2beam(zero_cycle["Signal_Raw"], zero_cycle["Signal_Ref"], ft_sensor)
measurements["s2beam"] = calculation_of_s2beam(measurements["Signal_Raw"], measurements["Signal_Ref"], ft_sensor)

# calculate s'2beam_z_median
measurements['s2beam_z_median'] = np.nan
zero_cycle, s2beam_z_median = calculation_of_median_s2beam_z(zero_cycle)

# check gaps between zero_cycles
gaps_between_zeroings = s2beam_z_median['mean_time'].loc[s2beam_z_median['mean_time'].diff().dt.total_seconds() > 60*60*5]

# check gaps between measurements
gaps_between_measurements = measurements['time_series'].loc[measurements['time_series'].diff().dt.total_seconds() > 60*60]

if not gaps_between_measurements.empty or not gaps_between_zeroings.empty:
    print('Data files contains gaps and need to be divided before data processing')
    print('Dates and times with gap larger than 5h between zero cycles')
    print(gaps_between_zeroings)
    print('Dates and times with gap larger than 1h between measurements')
    print(gaps_between_measurements)

else:
    # merge zero cycle into all measurements
    zero_cycle["Quality_Signal_Raw"] = zero_cycle["Quality"]
    zero_cycle["Quality_Signal_Ref"] = zero_cycle["Quality"]
    zero_cycle["Quality_T_Gas"] = zero_cycle["Quality"]
    zero_cycle["Quality_P_NDIR"] = zero_cycle["Quality"]
    zero_cycle["Quality_P_In"] = zero_cycle["Quality"]
    processed_data = merge_zero_cycle_with_measurements(zero_cycle, measurements)

    # interpolate s2beam_z
    processed_data = interpolate_s2beam_z(processed_data, s2beam_z_median)

    # calculate drift-corrected NDIR sensor signal, s_dc
    processed_data['s_dc'] = calculation_of_s_dc(processed_data['s2beam'], processed_data['s2beam_z_interpolated'])

    # calculate processed and final NDIR sensor signal
    processed_data['s_proc'] = calculation_of_sproc(processed_data['s_dc'], f)

    # calculate drift corrected polynomial coefficients, k1, k2, k3
    k1, k2, k3 = calculation_of_drift_corrected_k1k2k3(pre_calibration_date, pre_k1, pre_k2, pre_k3,
                                                       post_calibration_date, post_k1, post_k2, post_k3,
                                                       processed_data['time_series'])

    # calculate xco2wet
    processed_data['xco2wet'] = calculation_of_xco2wet(k1,  k2, k3, processed_data['s_proc'], p0, t0,
                                                       processed_data['T_Gas'], processed_data['P_NDIR'])
    # get quality flag of calculated xco2wet
    processed_data = get_quality_of_processed_xco2wet(processed_data)

    # calculate pco2wet
    processed_data['pco2wet'] = calculation_of_pco2wet(processed_data['xco2wet'], processed_data['P_In'], p0)

    # get quality flag of calculated pco2wet
    processed_data = get_quality_of_processed_pco2wet(processed_data)

    # get state for the extended wash cycle.
    processed_data = get_state_wash(processed_data)

    # get state for the extended standby.
    processed_data = get_state_standby(processed_data)

    # get state for the extended flush.
    processed_data = get_state_extended_flush(processed_data)

    # save processed data, note that the pCO2 corresponds to the water temperature at the waterside of the membrane.
    # this temperature is needed to calculate pCO2 in situ.
    export_path = r'C:\git\contros_co2_data_processing\export_processed_data'
    export_processed_data(processed_data, export_path)

    # prepare acceptable data to calculate pco2 and fco2 in situ
    operate_boolean = ((processed_data['State_Zero'] == 0) & (processed_data['State_Flush'] == 0) &
                       (processed_data['State_Wash'] == 0) & (processed_data['State_Standby'] == 0) &
                       (processed_data['State_Extended_Flush'] == 0) & (processed_data['Quality_pco2wet'] == 0))
    cols = ['time_series', 'P_In', 'xco2wet', 'pco2wet']
    pco2_data = processed_data[operate_boolean][cols]
    pco2_data = pco2_data.reset_index(drop=True)

    # get additional ferrybox measurements: either quality controlled from the archive of Obs_oceanografi or from
    # Exprapp (no qc)

    ferrybox_folder = r'C:\git\contros_co2_data_processing\example_data\ferrybox_all_sensors_no_qc'
    # ferrybox_folder = r'C:\git\contros_co2_data_processing\example_data\obs_oceanografi_ferrybox_qc'
    file_list = list_files(ferrybox_folder)

    # have the ferrybox data undergone Quality Control? Y or N
    qc = 'N'
    # qc = 'Y'
    ferrybox_data = read_ferrybox_files(file_list, qc)

    # merge ferrybox data with acceptable pCO2 data.
    pco2_data = merge_ferrybox_with_pco2(pco2_data, ferrybox_data)

    # calculate fco2
    # the water temperature at the waterside of the membrane is currently not measured. here we use the temperature
    # measured at the thermosalinograph as an approximation of the temperature at the CONTROS HydroC CO2 instrumnet.
    # this will result in an error, the size of this error is not properly assessed.
    pco2_data['fco2wet'] = calculation_of_fco2wet(pco2_data['SBE45_Temp'], pco2_data['P_In'], p0, pco2_data['pco2wet'],
                                               pco2_data['xco2wet'])

    # calculate pco2 and fco2 at in situ
    pco2_data = calculation_of_pco2_fco2_at_sst(pco2_data)

    # save calculated pco2 and fco2 data with additional ferrybox parameters
    export_pcof_fco2_at_sst(pco2_data, export_path)

    # plot fco2wet_sst
    plot_scatter(pco2_data, 'time_series', 'fco2wet_sst')









