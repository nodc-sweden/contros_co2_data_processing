import pandas as pd
import json
from datetime import datetime

from calculations import calculation_of_xco2wet


def check_calibration_data(serial_number, calibration_dates):
    if serial_number == "CO2FT-0918-001":
        calibration_data = {}
        print("Checks drift between all calibrations")
        for calibration_date in calibration_dates:
            with open("./calibration_docs/calibration_sheets.json") as fid:
                calibration_data[calibration_date] = json.load(fid)[serial_number][calibration_date]
                calibration_data[calibration_date]["calibration_date"] = datetime.strptime(calibration_date, "%Y-%m-%d")
                if calibration_date == "2021-04-28":
                    bath_data = pd.DataFrame()
                    bath_data["s_proc_calibration"] = calibration_data[calibration_dates[1]]["s_proc_calibration"]
                    bath_data["t_gas_calibration"] = calibration_data[calibration_dates[1]]["t_gas_calibration"]
                    bath_data["p_ndir_calibration"] = calibration_data[calibration_dates[1]]["p_ndir_calibration"]
                    bath_data["xco2_ref_calibration"] = calibration_data[calibration_dates[1]]["xco2_ref_calibration"]
                    xco2_pre = calculation_of_xco2wet(
                        calibration_data[calibration_dates[0]]["k1"], calibration_data[calibration_dates[0]]["k2"],
                        calibration_data[calibration_dates[0]]["k3"],
                        bath_data["s_proc_calibration"], calibration_data[calibration_dates[0]]["p0"],
                        calibration_data[calibration_dates[0]]["t0"], bath_data["t_gas_calibration"],
                        bath_data["p_ndir_calibration"])
                    xco2_post = calculation_of_xco2wet(
                        calibration_data[calibration_dates[1]]["k1"], calibration_data[calibration_dates[1]]["k2"],
                        calibration_data[calibration_dates[1]]["k3"],
                        bath_data["s_proc_calibration"], calibration_data[calibration_dates[1]]["p0"],
                        calibration_data[calibration_dates[1]]["t0"], bath_data["t_gas_calibration"],
                        bath_data["p_ndir_calibration"])
                    print('\nPre calibration_date: ' + calibration_dates[0])
                    print('Min and max difference between xCO2 pre-calibration and xCO2 post-calibration reference')
                    print(min(xco2_pre - bath_data["xco2_ref_calibration"]))
                    print(max(xco2_pre - bath_data["xco2_ref_calibration"]))
                    print('\nPost calibration_date: ' + calibration_date)
                    print('Min and max difference between xCO2 post-calibration and xCO2 post-calibration reference')
                    print(min(xco2_post - bath_data["xco2_ref_calibration"]))
                    print(max(xco2_post - bath_data["xco2_ref_calibration"]))
                if calibration_date == "2023-05-26":
                    bath_data = pd.DataFrame()
                    bath_data["s_proc_calibration"] = calibration_data[calibration_dates[2]]["s_proc_calibration"]
                    bath_data["t_gas_calibration"] = calibration_data[calibration_dates[2]]["t_gas_calibration"]
                    bath_data["p_ndir_calibration"] = calibration_data[calibration_dates[2]]["p_ndir_calibration"]
                    bath_data["xco2_ref_calibration"] = calibration_data[calibration_dates[2]]["xco2_ref_calibration"]
                    xco2_pre = calculation_of_xco2wet(
                        calibration_data[calibration_dates[1]]["k1"], calibration_data[calibration_dates[1]]["k2"],
                        calibration_data[calibration_dates[1]]["k3"],
                        bath_data["s_proc_calibration"], calibration_data[calibration_dates[1]]["p0"],
                        calibration_data[calibration_dates[1]]["t0"], bath_data["t_gas_calibration"],
                        bath_data["p_ndir_calibration"])
                    xco2_post = calculation_of_xco2wet(
                        calibration_data[calibration_dates[2]]["k1"], calibration_data[calibration_dates[2]]["k2"],
                        calibration_data[calibration_dates[2]]["k3"],
                        bath_data["s_proc_calibration"], calibration_data[calibration_dates[2]]["p0"],
                        calibration_data[calibration_dates[2]]["t0"], bath_data["t_gas_calibration"],
                        bath_data["p_ndir_calibration"])
                    print('\nPre calibration_date: ' + calibration_dates[1])
                    print('Min and max difference between xCO2 pre-calibration and xCO2 post-calibration reference')
                    print(min(xco2_pre - bath_data["xco2_ref_calibration"]))
                    print(max(xco2_pre - bath_data["xco2_ref_calibration"]))
                    print('\nPost calibration_date: ' + calibration_date)
                    print('Min and max difference between xCO2 post-calibration and xCO2 post-calibration reference')
                    print(min(xco2_post - bath_data["xco2_ref_calibration"]))
                    print(max(xco2_post - bath_data["xco2_ref_calibration"]))
    return calibration_data


