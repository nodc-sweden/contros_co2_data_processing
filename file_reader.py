import pandas as pd
import os


#  method to list all files in a folder
def list_files(folder: str):
    file_list = [os.path.join(folder, f) for f in
                 os.listdir(folder) if os.path.join(folder, f).endswith('.txt')]
    return file_list


#  method to read all listed files into a pandas dataframe
def read_files(file_list: list):
    df = pd.DataFrame()
    quality_flags = {}
    info_flags = {}
    header_line = 0
    qf_line = 0
    info_line = 0
    row_qf = 999
    row_info = 999
    row_format = 999
    for f in file_list:
        with (open(f, 'r') as fid):
            for r, line in enumerate(fid):
                if "$DATASETS" in line:
                    header_line = [r + 1, r + 2]
                    fid.close()
                    break
                elif "QUALITYBITS" in line:
                    row_qf = r+1
                elif "$INFOBITS" in line:
                    row_info = r+1
                elif "$FORMATS" in line:
                    row_format = r
                elif (r > row_info) & (r < row_format):
                    line_is = 0
                    for key, value in info_flags.items():
                        if line in value:
                            line_is = 1
                            break
                    if line_is == 0:
                        info_flags[info_line] = line
                        info_line = info_line+1
                elif (r > row_qf) and (r < (row_info-1)):
                    line_is = 0
                    for key, value in quality_flags.items():
                        if line in value:
                            line_is = 1
                            break
                    if line_is == 0:
                        quality_flags[qf_line] = line
                        qf_line = qf_line+1
            df = pd.concat([df, pd.read_csv(f, sep='\t', encoding='cp1252', header=header_line,
                                            decimal='.')], ignore_index=True)
    df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'$Timestamp': 'Timestamp'})
    df['time_series'] = pd.to_datetime(df["Timestamp"], format='%Y.%m.%d %H:%M:%S')
    return df, quality_flags, info_flags


#  method to read files in subfolders in to a combined dictionary and a measurement dataframe
def combine_data_in_subfolders(working_folder: str):
    sub_folders = [f.path for f in os.scandir(working_folder) if f.is_dir()]
    all_data = {}
    measurements = pd.DataFrame()
    measurements_quality_flags = pd.DataFrame()
    i = 0
    for item in sub_folders:
        files = list_files(item)
        name = (item.rsplit(f'\\', 1)[1]).replace('CO2FT_A_', '')
        all_data[name], quality_flags, info_flags = read_files(files)
        all_data[name] = all_data[name].sort_values(by='Timestamp').reset_index()
        parameters_used_in_calculations = ["Signal_Raw", "Signal_Ref", "T_Gas", "P_NDIR", "P_In"]
        if i == 0:
            measurements["Timestamp"] = all_data[name]["Timestamp"]
            measurements["time_series"] = all_data[name]["time_series"]
            measurements['Quality'] = all_data[name]["Quality"].astype(str)
            i = 1
        elif measurements["Timestamp"].equals(all_data[name]["Timestamp"]):
            measurements["Quality"] = (measurements["Quality"] + "_" + all_data[name]["Quality"].astype(str))
        else:
            measurements[name + " Timestamp"] = all_data[name]["Timestamp"]
            measurements["Quality"] = (measurements["Quality"] + "_" + all_data[name]["Quality"].astype(str))
        measurements[name] = all_data[name][name]
        if name in parameters_used_in_calculations:
            measurements["Quality_" + name] = all_data[name]["Quality"]

        measurements_quality_flags["Quality_" + name] = all_data[name]["Quality"]
    return all_data, measurements,measurements_quality_flags, quality_flags, info_flags


def merge_zero_cycle_with_measurements(zero_cycle, measurements):
    cols = ['Timestamp', 'time_series', 'Quality', 'Conc_Estimate', 'H_Gas', 'PCO2_Comp', 'PCO2_Corr', 'PCO2_Corr_Flush',
            'PCO2_Corr_Zero', 'P_In', 'Quality_P_In', 'P_NDIR', 'Quality_P_NDIR', 'Signal_Proc', 'Signal_Raw',
            'Quality_Signal_Raw', 'Signal_Ref', 'Quality_Signal_Ref', 'State_Flush', 'State_Zero', 'T_Control', 'T_Gas',
            'Quality_T_Gas', 'T_Sensor', 'XCO2_Corr', 's2beam', 's2beam_z_median']
    processed_data = pd.concat([zero_cycle[cols], measurements[cols]],
                               ignore_index=True).sort_values(by='Timestamp').reset_index(drop=True)
    # processed_data['elapsed time'] = processed_data['time series'] - processed_data['time series'][0]
    return processed_data


def get_additional_parameters():
    return


