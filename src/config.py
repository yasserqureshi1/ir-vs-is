import numpy as np

FILE = [
    'Banfora/UT/UT tracking files/Banfora_UT_rep1_savedresults_PPv72.mat',
    'Banfora/rep 2/Banfora_UT_rep2_savedresults_PPv72.mat',
    'Banfora/rep 3/Banfora_UT_rep3_savedresults_PPv72.mat',
    'Banfora/rep 4/Banfora_UT_rep4_savedresults_PPv72.mat',

    'Kisumu/UT/UT rep1/Kisumu_UT_rep1_savedresults_PPv72.mat',
    'Kisumu/UT/Kisumu_UT_rep2_savedresults_PPv72.mat',
    'Kisumu/rep 3/Kis_UT_rep3_savedresults_PPv72.mat',
    'Kisumu/Kisumu_UT_rep4_savedresults_PPv72.mat',
    'Kisumu/rep 5/Kis_UT_rep5_savedresults_PPv72.mat',

    'Ngoussu/rep 1/Ngoussu_UT_rep1_savedresults_PPv72.mat',
    'Ngoussu/rep 2/Ngoussu_UT_rep2_savedresults_PPv72.mat',
    'Ngoussu/rep 3/Ngoussu_UT_rep3_savedresults_PPv72.mat',
    'Ngoussu/rep 4/Ngoussu_UT_rep4_savedresults_PPv72.mat',

    'VK7/rep 1/VK7_UT_rep1_savedresults_PPv72.mat',
    'VK7/rep 2/VK7_UT_rep2_savedresults_PPv72.mat',
    'VK7/rep 3/VK7_UT_rep3_savedresults_PPv72.mat',
    'VK7/rep 4/VK7_UT_rep4_savedresults_PPv72.mat'
]

IS_RESISTANT = np.array([
    1,1,1,1,
    0,0,0,0,0,
    0,0,0,0,
    1,1,1,1
])
DATA_PATH = "E:/UT_LSTM_DATA/" 
PATH = "E:/IR_VS_IS/"