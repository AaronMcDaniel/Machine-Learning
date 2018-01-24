#with files in HW01 folder
parent_path = '.'

#Dota config presets
ORIGINAL = 0
FORMATTED = 1
NOSERVER = 2

#Pulsar config presets
ORIGINAL = 0

#choose which dota data to use
dota_config = NOSERVER

#chose which pulsar data to use
pulsar_config = ORIGINAL

if dota_config is ORIGINAL:
    dota_test_data_path = '%s/data/dota2Test(original).csv' % parent_path
    dota_train_data_path = '%s/data/dota2Train(original).csv' % parent_path
    dota_results = "Original_Results_Dota.csv"
    dota_analysis = "Original_Analysis_Dota.csv"


elif dota_config is FORMATTED:
    dota_test_data_path = '%s/data/dota2Test.csv' % parent_path
    dota_train_data_path = '%s/data/dota2Train.csv' % parent_path
    dota_results = "Formatted_Results_Dota.csv"
    dota_analysis = "Formatted_Analysis_Dota.csv"

elif dota_config is NOSERVER:
    dota_test_data_path = '%s/data/(no_server)dota2Test.csv' % parent_path
    dota_train_data_path = '%s/data/(no_server)dota2Train.csv' % parent_path
    dota_results = "No_Server_Results_Dota.csv"
    dota_analysis = "No_Server_Analysis_Dota.csv"


if pulsar_config is ORIGINAL:
  pulsar_test_data_path = '%s/data/Test_HTRU2.csv' % parent_path
  pulsar_train_data_path = '%s/data/Train_HTRU2.csv' % parent_path
  pulsar_results = "Results_HTRU2.csv"
  pulsar_analysis = "Analysis_HTRU2.csv"





"""
#Original file paths, kept for testing. Delete before submission
data_path = './data(original)/letter-recognition.data.txt'
test_data_path = './data(original)/letter-recognition.data.txt_0.2_test_part'
train_data_path = './data(original)/letter-recognition.data.txt_0.8_train_part'


abalone_predict_test_data_path = './data(original)/homework4abalone.data.txt_0.2_test_part'
abalone_predict_train_data_path = '%s/data(original)/homework4abalone.data.txt_0.8_train_part' % parent_path
"""


