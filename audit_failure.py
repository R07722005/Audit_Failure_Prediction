import pandas as pd
import numpy as np


class Audit_failure():
    """
    這個py檔是利用TEJ中的"衡量變數-重編彙整資料庫"中的重編原因當作審計失敗
    若重編原因1至3中有'轉投資相關'、'會計估計及評價'、
    '損益期間歸屬', '虛增(漏列)交易'、'所得稅及租賃會計'、'重大重分類/CPA更新意見'等項目
    就算是審計失敗

    Attributes:
        file_location: "衡量變數-重編彙整資料庫"資料的Excel檔案位置
        TSI_concat_location: 經"tern_customer_industry_concat"轉出的Excel資料的檔案位置

    --------------------------------------------
    Input:
        資料種類: 上市+上櫃
        DB類別:TEJ審計品質分析資料庫
        資料名稱: 衡量變數-重編彙總資料庫
        欄位All
        公司分類: 非金電(不含TDR)+電子產業(不含TDR)
        日期:季
        時間:2003/0101~2020/0907
        記得取消時間的降逆
        輸出使用樣式 預設: 公司主key，日期副key
    """
    def __init__(self, file_location, TSI_concat_location):
        self.file_location = file_location
        self.TSI_concat_location = TSI_concat_location
        self.outcome = None
        self.output_loc = None

    def compute(self):
        """
        將衡量變數-重編彙總資料庫中重編原因有出現'
        轉投資相關'、'會計估計及評價'、'損益期間歸屬', '虛增(漏列)交易'、'所得稅及租賃會計'、'重大重分類/CPA更新意見'
        的資料當作是審計失敗
        輸出excel請使用output function

        Args:
            None

        Returns:
            None
        """
        print('開始選擇審計失敗')
        remake_raw = pd.read_excel(self.file_location)
        fraud_list = ['轉投資相關', '會計估計及評價',
                      '損益期間歸屬', '虛增(漏列)交易',
                      '所得稅及租賃會計', '重大重分類/CPA更新意見']
        remake_fraud = remake_raw[(remake_raw['重編原因1'].isin(fraud_list)) |
                                  (remake_raw['重編原因2'].isin(fraud_list)) |
                                  (remake_raw['重編原因3'].isin(fraud_list))
                                  ][['公司碼', '年月']].reset_index()

        # pandas會把時間直接讀成時間2003/12/31=>2003-12-31，所以不能用/來分，要用-，還要先轉成string，最後年月日要改成
        remake_fraud[['年', '月', '日']] = remake_fraud[
                '年月'].astype(str).str.split('-', expand=True).astype(int)
        remake_fraud[['公司', '簡稱']] = remake_fraud['公司碼'].str.split(
                ' ', expand=True)
        remake_fraud['審計失敗'] = pd.Series(np.ones(remake_fraud.shape[0]))
        remake_fraud.drop(['公司碼', '簡稱', '年月', '日', 'index'],
                          axis=1, inplace=True)
        tern_customer_industry_combine = pd.read_excel(
                self.TSI_concat_location)
        tern_customer_industry_combine['公司'] =\
            tern_customer_industry_combine['公司'].astype(str)
        tern_customer_industry_combine = tern_customer_industry_combine.merge(
                remake_fraud, on=['公司', '年', '月'], how='left')
        tern_customer_industry_combine.fillna(value={'審計失敗': 0}, inplace=True)
        self.outcome = tern_customer_industry_combine
        print('完成選擇審計失敗')

    def output(self, output_location):
        """
        輸出審計失敗的Excel檔

        Args:
            output_location: Excel輸出的檔案位置

        Returns:
            審計失敗的Excel檔
        """
        print('輸出審計失敗')
        self.output_loc = output_location
        self.outcome.to_excel(output_location, float_format='%g',
                              encoding="utf-8", index=False)
        print('審計失敗輸出完成')

    def where_is_my_output_file(self):
        '''
        回傳Output file的位置，在使用過output才能用，不然就是None

        Args:
            None

        Returns:
            Output file的位置
        '''
        return self.output_loc
