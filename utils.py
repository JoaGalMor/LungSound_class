import pandas as pd
import glob

patient_path='C:/Users/Joaquin/BIG_DATA_analysis_master/TFM/datos/ICBHI_final_database/patient_diagnosis.csv'
sound_path='C:/Users/Joaquin/BIG_DATA_analysis_master/TFM/datos/ICBHI_final_database/sound_txt'
directory_path='C:/Users/Joaquin/BIG_DATA_analysis_master/TFM/'

patient_data=pd.read_csv(patient_path,sep=';',names=['ID','disease'])

path_processed="C:/Users/Joaquin/BIG_DATA_analysis_master/TFM/processed_audio_files"
path_noised='C:/Users/Joaquin/BIG_DATA_analysis_master/TFM/noised_audio_files_2'
path_shifted='C:/Users/Joaquin/BIG_DATA_analysis_master/TFM/shifted_audio_files_2'
path_experiments='C:/Users/Joaquin/BIG_DATA_analysis_master/TFM/experiments'
all_sound_files=glob.glob(sound_path+'/**.txt')

dict_chronic_healthy = {'Healthy': "Healthy", 'COPD': 'Chronic', "URTI": "Non_chronic",
                                "Bronchiectasis": "Chronic", "Pneumonia": "Non_chronic", "Bronchiolitis": "Non_chronic"}
CNN_data=directory_path+'data_CNN/'

dict_diseases_numbers = {"COPD": 0, "Healthy": 1, "URTI": 2, "Bronchiectasis": 3, "Pneumonia": 4,
                         "Bronchiolitis": 5}

dict_diseases_numbers_chronic = {"Healthy": 0, "Chronic": 1, "Non_chronic": 2}

files_plots={'COPD':'106_2b1_Pr_mc_LittC2SE','URTI':'188_1b1_Tc_sc_Meditron',
       'Helathy':'183_1b1_Pl_sc_Meditron','Bronchiectasis':'196_1b1_Pr_sc_Meditron',
       'Pneumonia':'191_2b1_Pr_mc_LittC2SE','Bronchiolitis':'206_1b1_Lr_sc_Meditron'}

path_plots='C:/Users/Joaquin/BIG_DATA_analysis_master/TFM/signal_plots/'


