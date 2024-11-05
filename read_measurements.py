import pandas as pd
import numpy as np
from datetime import datetime


from file_reader import list_files, read_files, combine_data_in_subfolders, merge_zero_cycle_with_measurements
from calibration import check_calibration_data
from calculations import (calculation_of_s2beam, calculation_of_median_s2beam_z, interpolate_s2beam_z,
                          calculation_of_s_dc, calculation_of_sproc, calculation_of_drift_corrected_k1k2k3,
                          calculation_of_xco2wet, get_quality_of_processed_xco2wet, calculation_of_pco2,
                          get_quality_of_processed_pco2wet, get_state_wash, get_state_standby, get_state_extended_flush)
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
    processed_data['pco2wet'] = calculation_of_pco2(processed_data['xco2wet'], processed_data['P_In'], p0)

    # get quality flag of calculated pco2wet
    processed_data = get_quality_of_processed_pco2wet(processed_data)

    # get state for the extended wash cycle.
    processed_data = get_state_wash(processed_data)

    # get state for the extended standby.
    processed_data = get_state_standby(processed_data)

    # get state for the extended flush.
    processed_data = get_state_extended_flush(processed_data)

    # save processed data
    export_path = r'C:\git\co2_data_processing_svea\export_processed_data'
    file_name = ('RV Svea processed pCO2 data_' + processed_data['time_series'][0].strftime('%Y%m%d') +
                 '_' + processed_data['time_series'][len(processed_data['time_series']) - 1].strftime('%Y%m%d') + '.txt')
    file_path = f"{export_path}\\{file_name}"
    processed_data.to_csv(file_path, sep='\t', index=False)

    # plot data
    plot_boolean = ((processed_data['State_Zero'] == 0) & (processed_data['State_Flush'] == 0) &
                    (processed_data['State_Wash'] == 0) & (processed_data['State_Standby'] == 0) &
                    (processed_data['State_Extended_Flush'] == 0) & (processed_data['Quality_pco2wet'] == 0))
    plot_scatter(processed_data[plot_boolean], 'time_series', 'pco2wet')



