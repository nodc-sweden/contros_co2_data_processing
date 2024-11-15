import pandas as pd
import numpy as np
from datetime import datetime, timedelta


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
