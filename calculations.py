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




def calculation_of_fco2(t_c, p_in, p0, pco2wet, xco2wet):
    """
    :param t_c: temperature in degrees Celsius at waterside of membrane
    :param p_in: air pressure at gaseous side of membrane
    :param p0: 1013.25 for conversion to atm
    :param pco2wet: partial pressure of CO2 in wet air
    :param xco2wet: molar fraction of CO2 in wet air
    :return: fugacity of CO2
    """
    t_k = t_c + 273.15
    # virial coefficient, B
    b_virial_coef = -1636.75 + 12.0408 * t_k - 0.0327957 * pow(t_k, 2) + (3.16528 * 1e-5) * pow(t_k, 3)
    # virial coefficient, delta
    delta = 57.7 - 0.118 * t_k
    r_gas_constant = 82.0578  # atm cm3 K-1 mol-1, recommended by Pierrot et al. (2009)
    fco2wet = pco2wet * np.exp((p_in/p0 * (b_virial_coef + 2 * pow(1 - xco2wet * 1e-6, 2) * delta)) /
                               (r_gas_constant * t_k))
    return fco2wet


def calculation_of_pco2_fco2_at_sst(df: pd.DataFrame):
    df['pco2wet_sst'] = df['pco2wet'] * np.exp(0.0423 * (df['SBE38_Temp'] - df['SBE45_Temp']))
    df['fco2wet_sst'] = df['fco2wet'] * np.exp(0.0423 * (df['SBE38_Temp'] - df['SBE45_Temp']))
    return df




