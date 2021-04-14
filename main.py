from discriminate_correct import Discriminate_accrural_creator
from tern_customer_industry_concat import TSI_concat
from Four_sheet_sorting import Sheet_sortor
from audit_failure import Audit_failure
from concat_all import Concat_all
from time_sequal import Time_sequal_concat
import DNN
import os
import dropbox

path = os.path.dirname(os.path.realpath(__file__)).replace('\\','/')
data_path = path + '/data'
model_path = path + '/model'
log_path = path + '/log'
if not os.path.isdir(data_path):
    os.makedirs(data_path)
if not os.path.isdir(model_path):
    os.makedirs(model_path)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
dbx = dropbox.Dropbox("S7iV9SzKNDoAAAAAAAAAAdsKGCOfq3BL_tb5VjJTTdJlNLUmbHTK0mTeMHp4wVuY")

for i in range(24,50):
    if not os.path.exists(model_path+'/model_{}_19.pkl'.format(i)):
        dbx.files_download_to_file(download_path=(model_path+'/model_{}_19.pkl').format(i),path="/Essay_model/model_{}_19.pkl".format(i))
        print('model_{}_19.pkl download complete'.format(i))
data_name = ['任期.xlsx', '客戶重要性.xlsx', '重編彙整總資料庫.xlsx', '產業專家.xlsx', '裁決性應計數的公式_正確.xlsx', 'BS_income_equity_cashflow.xlsx']
for i in data_name:
    if not os.path.exists((data_path+'/{}').format(i)):
        dbx.files_download_to_file(download_path=(data_path+'/{}').format(i),path="/Essay_data/{}".format(i))
        print('{} download complete'.format(i))
'''
dis_creator= Discriminate_accrural_creator(os.path.join(path, 'data/裁決性應計數的公式_正確.xlsx'.replace('\\','/')))
dis_creator.compute_accrural()
dis_creator.output_excel(os.path.join(path, 'data/裁決性應計數的公式_已產生各項應變數_correct.xlsx'.replace('\\','/')))

TSI_concator = TSI_concat(os.path.join(path, 'data/任期.xlsx').replace('\\','/'), os.path.join(path, 'data/客戶重要性.xlsx').replace('\\','/'), os.path.join(path, 'data/產業專家.xlsx').replace('\\', '/'))
TSI_concator.concat()
TSI_concator.output_excel(os.path.join(path, 'data/任期_客戶重要性_產業專家_合併.xlsx').replace('\\', '/'))
sortor = Sheet_sortor(os.path.join(path, 'data/BS_income_equity_cashflow.xlsx').replace('\\', '/'))
sortor.sort()
sortor.output(os.path.join(path, 'data/BS_income_equity_cashflow_改正順序.xlsx').replace('\\', '/'))
audit_failure_finder = Audit_failure(os.path.join(path,'data/重編彙整總資料庫.xlsx').replace('\\', '/'), TSI_concator.where_is_my_output_file())
audit_failure_finder.compute()
audit_failure_finder.output(os.path.join(path,'data/bs_income_equity_cashflow_增加審計失敗.xlsx').replace('\\', '/'))
#--------------------------------
'''
concator = Concat_all(audit_qulity_loc=os.path.join(path,'data/bs_income_equity_cashflow_增加審計失敗.xlsx').replace('\\', '/'), BIEC_loc = os.path.join(path, 'data/BS_income_equity_cashflow_改正順序.xlsx').replace('\\', '/'), dis_accr_loc=os.path.join(path, 'data/裁決性應計數的公式_已產生各項應變數_correct.xlsx').replace('\\','/'))
#concator = Concat_all(audit_qulity_loc=audit_failure_finder.where_is_my_output_file(), BIEC_loc = sortor.where_is_my_output_file(), dis_accr_loc=dis_creator.where_is_my_output_file())
concator.run()
concator.output(os.path.join(path, 'data/整合.xlsx').replace('\\','/'))
#--------------------------------

time_sequal = Time_sequal_concat(concator.where_is_my_output_file())
time_sequal.merge()
time_sequal.output(os.path.join(path, 'data/raw_data.xlsx').replace('\\','/'))
# --------------------------------

cofing = {
    'Seed': 777,
    'EPOCH': 20,
    'Semi_training_round': 50,
    'batch_size': 200,
    'lr': 0.0001,
    'weight_decay': 0,
    'Information_Entropy_loss_rate': 0.05,
    'model_num_for_testing': 19,
    'raw_data_loc': os.path.join(path, 'data/raw_data.xlsx')
    .replace('\\', '/'),
    'log_file_save_in': os.path.join(path, 'log')
    .replace('\\', '/'),
    'model_save_in': os.path.join(path, 'model')
    .replace('\\', '/'),
    'past_vote_thread': 24,
    'testing_year': 2019,
    'output_loc': os.path.join(path, "結果.xlsx")
    .replace('\\', '/'),
}
dnn = DNN.Dnn_run(cofing)
dnn.load_data()
# dnn.run()  # If you want to train models yourself, use this code
dnn.test()
