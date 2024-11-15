import pandas as pd


def export_processed_data(df: pd.DataFrame, export_path: str):
    file_name = ('RV Svea processed pCO2 data_' + df['time_series'][0].strftime('%Y%m%d') +
                 '_' + df['time_series'][len(df['time_series']) - 1].strftime('%Y%m%d') + '.txt')
    file_path = f"{export_path}\\{file_name}"
    return df.to_csv(file_path, sep='\t', index=False)


def export_pcof_fco2_at_sst(df: pd.DataFrame, export_path: str):
    file_name = ('RV Svea pCO2 and fCO2 at SST_' + df['time_series'][0].strftime('%Y%m%d') +
                 '_' + df['time_series'][len(df['time_series']) - 1].strftime('%Y%m%d') + '.txt')
    file_path = f"{export_path}\\{file_name}"
    data_to_export = ['time_series', 'time_series ferrybox', 'Latitude', 'Longitude', 'Course', 'Height', 'SBE38_Temp',
                      'SBE45_Temp', 'SBE45_Salinity', 'O2_Corr', 'Optode_AirSaturation', 'WetLabs_Chlorophyll-A',
                      'flow_pCO2', 'pco2wet_sst', 'fco2wet_sst']
    df_export = df[data_to_export]

    return df_export.to_csv(file_path, sep='\t', index=False)

