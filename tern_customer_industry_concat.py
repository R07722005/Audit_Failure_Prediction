import pandas as pd


class TSI_concat():
    """
    使用此python檔請先安裝XLRD 與 OpenPyXL套件，才能read_excel
    這個py檔是將TEJ審計品質分析資料庫當中的'影響變數-任期與經驗'、'影響變數-客戶重要性'、'影響變數-產業專家'的資料結合在一起
    依照公司名稱、年、月、日來結合資料
    並將'簽證意見類型','繼續經營假設是否有疑慮','是否為大型事務所'以虛無變數表示

    Attributes:
        cpa_file: "影響變數-任期與經驗"資料的Excel檔案位置
        customer_file: "影響變數-客戶重要性"Excel資料的檔案位置
        industry_file: "影響變數-產業專家"Excel資料的檔案位置

    ---------------------------------------------
    Input: 請使用TEJ提供的Excel套件，適用Excel特殊轉檔來找資料
        影響變數-任期與經驗:
            資料種類: 上市+上櫃
            DB類別:TEJ審計品質分析資料庫
            資料名稱: 影響變數-任期與經驗
            欄位All
            公司分類: 非金電(不含TDR)+電子產業(不含TDR)
            日期:季
            時間:2003/0101~2020/0907
            記得取消時間的降逆
            輸出使用樣式12: 日期主key，公司副key
        影響變數-客戶重要性:
            資料種類: 上市+上櫃
            DB類別:TEJ審計品質分析資料庫
            資料名稱: 影響變數-客戶重要性
            欄位All
            公司分類: 非金電(不含TDR)+電子產業(不含TDR)
            日期:季
            時間:2003/0101~2020/0907
            記得取消時間的降逆
            輸出使用樣式12: 日期主key，公司副key
        影響變數-產業專家:
            資料種類: 上市+上櫃
            DB類別:TEJ審計品質分析資料庫
            資料名稱: 影響變數-產業專家
            欄位All
            公司分類: 非金電(不含TDR)+電子產業(不含TDR)
            日期:季
            時間:2003/0101~2020/0907
            記得取消時間的降逆
            輸出使用樣式12: 日期主key，公司副key
"""

    def __init__(self, cpa_file, customer_file, industry_file):
        self.cpa_file = cpa_file
        self.customer_file = customer_file
        self.industry_file = industry_file
        self.outcome = None
        self.output_loc = None

    def concat(self):
        """
        依照公司名、年、月、日結合任期與經驗、客戶重要性、產業專家
        '簽證意見類型', '繼續經營假設是否有疑慮', '是否為大型事務所'以虛擬變數表示
        請使用output_excel來輸出結果

        Args:
            None

        Returns:
            None
        """
        print('審計品質資料銜接開始')
        cpa_tern_raw = pd.read_excel(self.cpa_file)
        customer_importance_raw = pd.read_excel(self.customer_file)
        industry_expert_raw = pd.read_excel(self.industry_file)

        # pandas會把時間直接讀成時間2003/12/31=>2003-12-31，所以不能用/來分，要用-，還要先轉成string，最後年月日要改成
        cpa_tern_raw[['年', '月', '日']] =\
            cpa_tern_raw['年度'].astype(str).str.split(
                    '-', expand=True).astype(int)
        customer_importance_raw[['年', '月', '日']] =\
            cpa_tern_raw['年度'].astype(str).str.split(
                    '-', expand=True).astype(int)
        industry_expert_raw[['年', '月', '日']] =\
            cpa_tern_raw['年度'].astype(str).str.split(
                    '-', expand=True).astype(int)

        cpa_tern_raw[['公司', '簡稱']] =\
            cpa_tern_raw['公司'].str.split(' ', expand=True)
        customer_importance_raw[['公司', '簡稱']] =\
            customer_importance_raw['公司'].str.split(' ', expand=True)
        industry_expert_raw[['公司', '簡稱']] =\
            industry_expert_raw['公司'].str.split(' ', expand=True)

        cpa_tern_dummy = pd.get_dummies(
            cpa_tern_raw[['簽證意見類型',
                          '繼續經營假設是否有疑慮',
                          '是否為大型事務所']],
            dummy_na=False,
            drop_first=False)
        cpa_tern_raw.drop(['簽證意見類型',
                           '繼續經營假設是否有疑慮',
                           '是否為大型事務所'],
                          axis=1, inplace=True)
        cpa_tern_raw = pd.concat([cpa_tern_raw, cpa_tern_dummy], axis=1)

        customer_nodul = customer_importance_raw.columns.difference(
                cpa_tern_raw.columns).tolist() + ['公司', '年', '月', '日']

        # 留下任期與產業專家中不同的column
        industry_modul = industry_expert_raw.columns.difference(
                cpa_tern_raw.columns).tolist() + ['公司', '年', '月', '日']

        merge_data = cpa_tern_raw.merge(
                customer_importance_raw[customer_nodul],
                on=['公司', '年', '月', '日'], how='left')
        merge_data.drop_duplicates(subset=['公司', '年', '月', '日'], inplace=True)

        merge_data = merge_data.merge(
                industry_expert_raw[industry_modul],
                on=['公司', '年', '月', '日'],
                how='left')
        merge_data.drop_duplicates(subset=['公司', '年', '月', '日'], inplace=True)
        self.outcome = merge_data
        print('審計品質資料銜接結束')

    def output_excel(self, output_location):
        """
        輸出任期與經驗、客戶重要性、產業專家三張表的結合的Excel檔

        Args:
            output_location: Excel輸出的檔案位置

        Returns:
            結合的Excel檔
        """
        print('審計品質資料開始輸出')
        self.output_loc = output_location
        self.outcome.to_excel(output_location,
                              float_format='%g',
                              encoding="utf-8",
                              index=False)
        print('審計品質資料完成')

    def where_is_my_output_file(self):
        '''
        回傳Output file的位置，在使用過output_excel才能用，不然就是None

        Args:
            None

        Returns:
            Output file的位置
        '''
        return self.output_loc
