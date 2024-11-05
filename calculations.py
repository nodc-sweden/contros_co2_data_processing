import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import scipy.stats as stats


def calculation_of_s2beam(signal_raw, signal_ref, ft_sensor):
    """
    :param signal_raw:  raw signal from dual-beam NDIR detector
    :param signal_ref: reference signal from dual-beam NDIR detector
    :param ft_sensor: NDIR unit specific temperature compensation factor
    :return: two-beam signal that includes the temperature compensation.
    """

    s2beam = (signal_raw / signal_ref) * ft_sensor
    return s2beam


def calculation_of_median_s2beam_z(zero_cycle: pd.DataFrame):
    """
    To avoid potential outliers that may impact the averaging process, the median value is taken from each zeroing.
    :param zero_cycle: a dataframe with all data from regular zeroings
    :return zero_cycle: with averaged two-beam zero-signal with temperature compensation,
    :return s2beam_z_median: dataframe with median two-beam zero-signals at the averaged times
    """

    start_and_stop = (zero_cycle["State_Zero"] == 1).astype(float).diff()
    if zero_cycle["State_Zero"][0] == 1:
        start_and_stop[0] = 1
    if zero_cycle["State_Zero"][len(zero_cycle["State_Zero"])-1] == 1:
        start_and_stop[len(start_and_stop)-1] = -1

    start_times = zero_cycle.loc[(start_and_stop == 1), "time_series"]
    stop_times = zero_cycle.loc[(start_and_stop == -1), "time_series"]
    zero_cycle["s2beam_z_median"] = np.nan
    s2beam_z_median = pd.DataFrame()
    s2beam_z_median['median'] = np.nan
    s2beam_z_median['mean_time'] = np.datetime64('nat')
    for i, start_time in enumerate(start_times):
        stop_time = stop_times.iloc[i]
        valid_values = {0, 128}
        zeroing_boolean = ((zero_cycle["time_series"] > (start_time + timedelta(seconds=45))) &
                           (zero_cycle["time_series"] < stop_time) & zero_cycle['Quality'].isin(valid_values))
        median_value = zero_cycle.loc[zeroing_boolean, "s2beam"].median()
        mean_time = zero_cycle.loc[zeroing_boolean, "time_series"].mean()
        closest_indices = ((zero_cycle["time_series"] - mean_time).abs().idxmin())
        if isinstance(closest_indices, pd.Index) and not closest_indices.empty:
            first_closest_index = closest_indices[0]  # Första värdet
        else:
            first_closest_index = closest_indices
        mean_time_closest = zero_cycle["time_series"].iloc[first_closest_index]
        zero_cycle.loc[first_closest_index, 's2beam_z_median'] = median_value
        if ~np.isnan(median_value):
            s2beam_z_median.at[i, 'median'] = median_value
            s2beam_z_median.at[i, 'mean_time'] = mean_time_closest
    return zero_cycle, s2beam_z_median


def interpolate_s2beam_z(processed_data: pd.DataFrame, s2beam_z_median: pd.DataFrame):
    """
    :param processed_data: dataframe with  measurements combined with data from zero cycles
    :param s2beam_z_median: dataframe with averaged two-beam zero-signal at averaged times
    :return: processed_data: dataframe with interpolated two-beam zero-signal with temperature compensation
    """

    processed_data['elapsed_time'] = (processed_data['time_series'] - processed_data['time_series'].iloc[0]).dt.total_seconds()
    processed_data.set_index('elapsed_time', inplace=True)
    processed_data['s2beam_z_interpolated'] = processed_data['s2beam_z_median'].interpolate(method='index')
    processed_data.reset_index(inplace=True)

    # extrapolate over start points
    x_start = (s2beam_z_median['mean_time'].iloc[0:2] - processed_data['time_series'].iloc[0]).dt.total_seconds()
    y_start = s2beam_z_median['median'].iloc[0:2]
    lin_reg = stats.linregress(x_start, y_start)
    slope = lin_reg.slope
    intercept = lin_reg.intercept
    start_boolean = ((processed_data['time_series'] < s2beam_z_median['mean_time'].iloc[0]) &
                     (processed_data['time_series'] > (s2beam_z_median['mean_time'].iloc[0]) - pd.Timedelta('5hr')))
    processed_data['s2beam_z_interpolated'].values[start_boolean] = slope * processed_data['elapsed_time'].values[
        start_boolean]+intercept

    # extrapolate over end points
    x_end = (s2beam_z_median['mean_time'].iloc[-2:] - processed_data['time_series'].iloc[0]).dt.total_seconds()
    y_end = s2beam_z_median['median'].iloc[-2:]
    lin_reg = stats.linregress(x_end, y_end)
    slope = lin_reg.slope
    intercept = lin_reg.intercept
    end_boolean = ((processed_data['time_series'] > s2beam_z_median['mean_time'].iloc[-1]) &
                   (processed_data['time_series'] < (s2beam_z_median['mean_time'].iloc[-1]) + pd.Timedelta('5hr')))
    processed_data['s2beam_z_interpolated'].values[end_boolean] = slope * processed_data['elapsed_time'].values[
        end_boolean]+intercept
    return processed_data


def calculation_of_s_dc(s2beam, s2beam_z_interpolated):
    """
    :param s2beam: two-beam signal that includes the temperature compensation.
    :param s2beam_z_interpolated: interpolated two-beam zero-signal with temperature compensation
    :return: s_dc: drift-corrected NDIR sensor signal
    """

    s_dc = s2beam / s2beam_z_interpolated

    return s_dc


def calculation_of_sproc(s_dc, f):
    """
    :param s_dc: drift-corrected NDIR sensor signal
    :param f: NDIR-unit specific scale factor
    :return: sproc: processed and final NDIR sensor signal
    """
    sproc = f * (1 - s_dc)

    return sproc


def calculation_of_drift_corrected_k1k2k3(pre_calibration_date, pre_k1, pre_k2, pre_k3, post_calibration_date, post_k1,
                                          post_k2, post_k3, time_series):
    """

    :param pre_calibration_date: pre-calibration date
    :param pre_k1: pre-calibration k1 coefficient
    :param pre_k2: pre-calibration k2 coefficient
    :param pre_k3: pre-calibration k3 coefficient
    :param post_calibration_date: post-calibration date
    :param post_k1: post-calibration k1 coefficient
    :param post_k2: post-calibration k2 coefficient
    :param post_k3: post-calibration k3 coefficient
    :param time_series: time series at measurements
    :return: interpolated k1, k2, k3 at measured time
    """
    pre_date = datetime.strptime(pre_calibration_date, '%Y-%m-%d')
    post_date = datetime.strptime(post_calibration_date, '%Y-%m-%d')
    # The interpolation should be done over runtime which the instrument provides.
    # The calibration sheets normally contains a runtime value as well except for 2019.
    # The installation on Svea however is not  set up to get runtime, this needs to be fixed.
    #
    # To handle historic data: ideally the coefficients should be interpolated using the timestamps from the first
    # measurement after calibration and the last measurement before calibration. There will also be gaps in the
    # measurements that ultimately should be accounted for in order to estimate the most accurate runtime.
    #
    # In the meantime, the timestamps at the calibrations dates will be used and an elapsed time will be derived.

    elapsed_time = [et.total_seconds() for et in [pre_date - pre_date, post_date - pre_date]]
    elapsed_time_data = (time_series - pre_date).dt.total_seconds()

   # interpolate k1
    slope_k1, intercept_k1, rk1, p_k1, std_err_k1 = stats.linregress(elapsed_time, [pre_k1, post_k1])
    k1 = slope_k1 * elapsed_time_data + intercept_k1

    # interpolate k2
    slope_k2, intercept_k2, rk2, p_k2, std_err_k2 = stats.linregress(elapsed_time, [pre_k2, post_k2])
    k2 = slope_k2 * elapsed_time_data + intercept_k2

    # interpolate k3
    slope_k3, intercept_k3, rk3, p_k3, std_err_k3 = stats.linregress(elapsed_time, [pre_k3, post_k3])
    k3 = slope_k3 * elapsed_time_data + intercept_k3

    return k1, k2, k3


def calculation_of_xco2wet(k1,  k2, k3, sproc, p0, t0, tgas, pndir):
    """

    :param k1: calibration coefficient k1
    :param k2: calibration coefficient k2
    :param k3: calibration coefficient k3
    :param sproc: processed and final NDIR sensor signal
    :param p0: normal pressure, 1013.25 mbar
    :param t0: normal temperature, 273.15 K
    :param tgas: gas temperature
    :param pndir: cell pressure
    :return: CO2 mole fraction in wet air
    """
    xco2wet = (k1*sproc + k2*sproc**2 + k3*sproc**3)*(p0*(tgas+t0))/(t0*pndir)
    return xco2wet


def get_quality_of_processed_xco2wet(processed_data: pd.DataFrame()):
    """
    xco2wet is calculated from the following measurements:
    "Signal_Raw"
    "Signal_Ref"
    "T_Gas"
    "P_NDIR"
    """
    quality_of_parameters_used_in_calculations = ["Quality_Signal_Raw", "Quality_Signal_Ref", "Quality_T_Gas",
                                                  "Quality_P_NDIR"]
    same_value = (processed_data['Quality_Signal_Raw'] ==
                  processed_data['Quality_Signal_Ref']) & (
                         processed_data['Quality_Signal_Raw'] ==
                         processed_data['Quality_T_Gas']) & (
                         processed_data['Quality_Signal_Raw'] ==
                         processed_data['Quality_P_NDIR'])

    processed_data['Quality_xco2wet'] = processed_data[
        'Quality_Signal_Raw'].where(same_value, processed_data[quality_of_parameters_used_in_calculations].astype(
                                                                                       str).agg('_'.join, axis=1))

    return processed_data


def calculation_of_pco2(xco2wet, p_in, p0):
    """

    :param xco2wet:
    :param p_in:
    :param p0:
    :return:
    """
    pco2wet = xco2wet * p_in / p0
    return pco2wet


def get_quality_of_processed_pco2wet(processed_data: pd.DataFrame()):
    """
    pco2wet is calculated from the following measurements:
    "Signal_Raw"
    "Signal_Ref"
    "T_Gas"
    "P_NDIR"
    "P_In"
    """
    quality_of_parameters_used_in_calculations = ["Quality_Signal_Raw", "Quality_Signal_Ref", "Quality_T_Gas",
                                                  "Quality_P_NDIR", "Quality_P_In"]
    same_value = (processed_data['Quality_Signal_Raw'] ==
                  processed_data['Quality_Signal_Ref']) & (
                         processed_data['Quality_Signal_Raw'] ==
                         processed_data['Quality_T_Gas']) & (
                         processed_data['Quality_Signal_Raw'] ==
                         processed_data['Quality_P_NDIR']) & (
                         processed_data['Quality_Signal_Raw'] ==
                         processed_data['Quality_P_In'])

    processed_data['Quality_pco2wet'] = processed_data[
        'Quality_Signal_Raw'].where(same_value, processed_data[quality_of_parameters_used_in_calculations].astype(str).
                                    agg('_'.join, axis=1))
    return processed_data


def get_state_wash(processed_data: pd.DataFrame()):
    """
    The diurnal wash cycle affects data for approximately 2 hours after its initiation
    :param processed_data:
    :return: processed_data: including a column with State_Wash that is 1 when data are affected by wash, else 0
    """

    # contains på 64 kan ge flaggor på platser där inte önskat
    processed_data["State_Wash"] = np.where(
        (processed_data["Quality"] == 64) |
        (processed_data["Quality"].astype(str).str.contains(r'(^|_)64(_|$)')), 1, 0
    )

    start_and_stop = (processed_data["State_Wash"] == 1).astype(float).diff()
    if processed_data["State_Wash"][0] == 1:
        start_and_stop[0] = 1
    if processed_data["State_Wash"][len(processed_data["State_Wash"]) - 1] == 1:
        start_and_stop[len(start_and_stop) - 1] = -1

    start_times = processed_data.loc[(start_and_stop == 1), "time_series"]

    for i, start_time in enumerate(start_times):
        wash_boolean = (processed_data["time_series"] > start_time) & (processed_data["time_series"] <
                                                                       (start_time + timedelta(minutes=120)))
        processed_data.loc[wash_boolean, "State_Wash"] = 1
    return processed_data


def get_state_standby(processed_data: pd.DataFrame()):
    """
    The sensor takes a while to adjust after a longer standby period. There is also often an increase in pCO2 before
    a standby period suggesting that the instrument is set to standby when something is off or when reaching a harbour.
    The following adds a standby flag of 1 whenever the standby flag is raised in "Quality", but also to the following
    90 minutes after a standby period. A flag of 1 is also added to the last 30 minutes before a standby period.
    :param processed_data:
    :return: processed_data: including a column with State_Standby that is 1 when data are affected by standby, else 0
    """
    # Double check with Lena that 10 actually is a standby flag!!!
    standby_list = [10, 128, 12416, 16512, 28800]
    standby_pattern = '|'.join(map(str, standby_list))

    processed_data["State_Standby"] = np.where(
        processed_data["Quality"].astype(str).str.match(r'(^|\D)(' + standby_pattern + r')($|\D)') |
        processed_data["Quality"].isin(standby_list),
        1,
        0
    )

    start_and_stop = (processed_data["State_Standby"] == 1).astype(float).diff()
    if processed_data["State_Standby"][0] == 1:
        start_and_stop[0] = 1
    if processed_data["State_Standby"][len(processed_data["State_Standby"]) - 1] == 1:
        start_and_stop[len(start_and_stop) - 1] = -1

    start_times = processed_data.loc[(start_and_stop == 1), "time_series"]
    stop_times = processed_data.loc[(start_and_stop == -1), "time_series"]

    for i, start_time in enumerate(start_times):
        stop_time = stop_times.iloc[i]
        standby_boolean = ((processed_data["time_series"] > (start_time - timedelta(minutes=30))) &
                           (processed_data["time_series"] < (stop_time + timedelta(minutes=90))))
        processed_data.loc[standby_boolean, "State_Standby"] = 1
    return processed_data


def get_state_extended_flush(processed_data: pd.DataFrame()):
    """
    The sensor takes a while to adjust after a flush, typically 30 min from the start of the flush.
    :param processed_data:
    :return: processed_data: including a column with State_Wash that is 1 when data are affected by wash O, else 0
    """
    processed_data["State_Extended_Flush"] = np.where(
        processed_data["State_Flush"] > 0,
        1,
        0
    )

    start_and_stop = (processed_data["State_Extended_Flush"] == 1).astype(float).diff()
    if processed_data["State_Extended_Flush"][0] == 1:
        start_and_stop[0] = 1
    if processed_data["State_Extended_Flush"][len(processed_data["State_Extended_Flush"]) - 1] == 1:
        start_and_stop[len(start_and_stop) - 1] = -1

    start_times = processed_data.loc[(start_and_stop == 1), "time_series"]

    for i, start_time in enumerate(start_times):
        flush_boolean = (processed_data["time_series"] > start_time) & (processed_data["time_series"]
                                                                        < (start_time + timedelta(minutes=30)))
        processed_data.loc[flush_boolean, "State_Extended_Flush"] = 1
    return processed_data


def calculation_of_fco2():
    """
    This requires additional ferrybox data """
    return




