# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


class Discriminate_accrural_creator ():
    """
    使用此python檔請先安裝XLRD 與 OpenPyXL套件，才能read_excel
    這個Py檔要計算出各上市、上櫃公司的裁決應計數
    使用的公式詳見同資料夾的'裁決性應計數的公式.txt'
    裁決性應計數採用當年的同產業的迴歸係數去推測，而不是用該公司10年期間
    依照各期別的不同產業分開，個別使用Sklearn的linear_model去跑回歸
    跑出的回歸模型再帶入各公司的應變數算出預測的正常應計數
    再把當年真正的應計數減去預測的應計數得到裁決性應計數

    Attributes:
        file_location: 從TEJ存下來的Excel資料的檔案位置
    -----------------------------------------------------------------------------
    Input: 從TEJ下載的資料
    這一個不能用Excel的轉檔工具要直接用TEJ
    資料種類: 上市+上櫃
    DB類別:TEJ IERS Finance-國際會計準則
    資料名稱: IFRS以合併為主簡表(單季)-全產業

    欄位設定:
        1. 合併總損益 -----------------------
                                           |----組合成Total_accurla
        2. 其他權益調整額-非常項目損益--------
        3. 資產總額
        4. 營業收入淨額
        5. 應收帳款及票據
        6. 不動產廠房與設備
        7. ROA(A)稅後息前
    基本資料欄位設定:
        1. TSE產業別
        2. TEJ產業指數

    公司分類: 非金電(不含TDR)+電子產業(不含TDR)

    日期:季
    時間:2003/0101~2020/0907(但是只有從2005 6月開始的資料)
    記得取消時間的降逆

    分隔設定:
    用逗號
    公司分隔
    日期分隔
    顯示空值字串 = >NA
    ---------------------------------------------------------------------
    Output: 一個Excel，內涵各產業的回歸使用的係數，與裁決性應計數
    """

    def __init__(self, file_location):
        self.file_location = file_location
        self.excel_for_output = None
        self.output_loc = None

    def compute_accrural(self):
        """
        計算裁決性應計數，計算完成請使用output_excel來進行輸出

        Args:
            None

        Output:
            None
        """
        print('開始計算裁決性應計數')
        dis_origin = pd.read_excel(self.file_location)
        # TEJcolumn標題會有空白，用此行去除)
        dis_origin.columns = map(lambda x: x.strip(), dis_origin.columns)
        dis_origin.replace('^\s+', '', regex=True, inplace=True)  # front
        dis_origin.replace('\s+$', '', regex=True, inplace=True)  # end
        # 上兩行把內容的字前後的空格都去掉，因為TEJ轉檔出來的結果都會在每一格前後都留空白(ex:'N.A.'會長成'    N.A.')

        # 把TEJ倒出來的資料裡的"N.A." 改成pandas和numpy用的NaN，輸出後會變空格
        dis_origin.replace(to_replace='N.A.', value=np.nan, inplace=True)

        df_last_year = dis_origin[['公司', '年', '月', '現金及約當現金',
                                   '流動資產', '流動負債', '一年內到期長期負債',
                                   '折舊－CFO', '攤提－CFO', '資產總額',
                                   '營業收入淨額', '應收帳款及票據',
                                   '不動產廠房及設備', 'ROA(A)稅後息前']]
        # 準備把每一筆資料concat去年同期的資料方便計算
        df_last_year.columns = map(lambda x: x +
                                   't-1' if x not in ['公司', '月'] else x,
                                   df_last_year.columns)
        # 把這些資料的columns名加上t-1，if一定要加 else
        # 不然lambda if else不給用，把去年的資料加上t-1的名字做區隔(公司、月不用)
        df_last_year['年'] = df_last_year['年t-1'] + 1  # 製造用來對齊的key

        dis_merge = dis_origin.merge(df_last_year,
                                     on=['公司', '年', '月'],
                                     how='left')
        # 把去年的資料黏在今年的上面，用公司、年、月當key

        print('計算各項參數')
        dis_merge['change_ca'] = dis_merge['流動資產'] - dis_merge['流動資產t-1']
        dis_merge['change_cash'] = (dis_merge['現金及約當現金'] -
                                    dis_merge['現金及約當現金t-1'])
        dis_merge['change_cl'] = dis_merge['流動負債'] - dis_merge['流動負債t-1']
        dis_merge['change_current_long_debt'] = (dis_merge['一年內到期長期負債'] -
                                                 dis_merge['一年內到期長期負債t-1'])
        dis_merge['Total_accural/去年Asset'] = ((dis_merge['change_ca'] -
                                               dis_merge['change_cash'] -
                                               dis_merge['change_cl'] +
                                               dis_merge['change_current_long_debt'] -
                                               dis_merge['折舊－CFO'] -
                                               dis_merge['攤提－CFO']) /
                                              dis_merge['資產總額t-1'])
        dis_merge['1/去年Asset'] = 1/dis_merge['資產總額t-1']
        dis_merge['銷售額變動'] = dis_merge['營業收入淨額'] - dis_merge['營業收入淨額t-1']
        dis_merge['應收帳款變動'] = dis_merge['應收帳款及票據'] - dis_merge['應收帳款及票據t-1']
        dis_merge['(銷售變動-應收變動)/去年Asset'] = (
            dis_merge['銷售額變動'] - dis_merge['應收帳款變動']) / dis_merge['資產總額t-1']
        dis_merge['PPE/去年Asset'] = dis_merge['不動產廠房及設備'] / dis_merge['資產總額t-1']
        dis_merge['ROA/去年Asset'] = (dis_merge['ROA(A)稅後息前'] /
                                    dis_merge['資產總額t-1'])
        # 創造出需要的欄項

        print('計算模型')
        # reset_index是因為原本輸出的會是長度不一的一長串series
        dis_group = dis_merge.groupby(
                ['年', '月', 'TSE 產業別']).apply(self.model).reset_index()

        # 小心副作用:這會讓數字只保留最後六個數字
        predict = pd.DataFrame(dis_group[0].tolist()).apply(
                pd.Series, 1).stack()
        predict.index = predict.index.droplevel(-1)
        predict.name = 'predict'
        # 上面三行是因為所有的預測值會被卡在各個group中呈現一個list，前三行幫助這個list可以展開
        # https://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-rows/17116976#17116976

        del dis_group[0]
        dis_group = dis_group.join(predict)
        dis_group[['公司', '預測的Total_accural']] = pd.DataFrame(
                dis_group.predict.tolist(), index=dis_group.index)
        del dis_group['predict']
        dis_merge = dis_merge.merge(dis_group,
                                    on=['年', '月', '公司', 'TSE 產業別'],
                                    how='left')
        dis_merge['裁決性應計數'] = (dis_merge['Total_accural/去年Asset'] -
                               dis_merge['預測的Total_accural'])

        dis_coef = dis_merge.groupby(['年', '月', 'TSE 產業別']).apply(
                self.model_for_coef).reset_index()
        coef = dis_coef[[0]][0].values.tolist()  # 參數
        coef = pd.DataFrame(coef,
                            columns=['1/asset參數', 'sale-ar參數',
                                     'ppe參數', 'roa參數',
                                     'intercept參數'])
        coef = pd.concat([dis_coef, coef], axis=1).drop(columns=[0])

        dis_merge = dis_merge.merge(coef, on=['年', '月', 'TSE 產業別'], how='left')
        self.excel_for_output = dis_merge
        print('裁決性應計數計算完成')

    def model(self, df):
        """
        用來計算group_by中每個group的回歸
        並回傳預測的裁決性應計數

        Args:
            df(pd.Datafram): 要計算的dataframe

        Returns:
            預測的裁決性應計數
        """
        # 用來放在每一個小group裡的方程式，每個group都會獨立跑一次
        df.fillna(value=df.mean(), inplace=True)  # 把每個group裡的na值都按各產業平均值去填
        df.fillna(value=0, inplace=True)  # 把剩下還是沒有值的用0去填
        y = df[['Total_accural/去年Asset']].values
        x = df[['1/去年Asset', '(銷售變動-應收變動)/去年Asset',
                'PPE/去年Asset', 'ROA/去年Asset']].values
        data_label = df[['公司']].values
        linreg = LinearRegression()
        linreg.fit(x, y)
        predict = linreg.predict(x)  # 出來的值是<class 'numpy.ndarray'>
        # 把預測的結果跟公司名稱concat在一起，之後才能在group_by中分離
        output = np.concatenate((data_label, predict), axis=1).tolist()
        return output

    def model_for_coef(self, df):
        """
        用來計算group_by中每個group的回歸
        並回傳回歸線的每項係數的參數

        Args:
            df(pd.Datafram): 要計算的dataframe

        Returns:
            預測的裁決性應計數
        """
        # 用來放在每一個小group裡的方程式，每個group都會獨立跑一次
        df.fillna(value=df.mean(), inplace=True)  # 把每個group裡的na值都按各產業平均值去填
        df.fillna(value=0, inplace=True)  # 把剩下還是沒有值的用0去填
        y = df[['Total_accural/去年Asset']].values
        x = df[['1/去年Asset', '(銷售變動-應收變動)/去年Asset',
                'PPE/去年Asset', 'ROA/去年Asset']].values
        linreg = LinearRegression()
        linreg.fit(x, y)
        # coef_會輸出np array的coeffition串，和intercept_合併後再轉乘list會比較好處理
        coef = np.append(linreg.coef_, linreg.intercept_).tolist()
        return coef

    def output_excel(self, output_location):
        """
        將計算好的裁決性應計數輸出到Excel

        Args:
            output_location: Excel輸出的位置

        Returns:
            裁決性應計數與各項係數的Excel表
        """
        print('裁決性應計數開始輸出至Excel')
        self.output_loc = output_location
        # floating_format讓數字不要出現科學符號，g是直接顯示原本的數字不會四捨五入，index = False就不會產生index欄
        self.excel_for_output.to_excel(output_location,
                                       float_format='%g',
                                       encoding="utf-8",
                                       index=False)
        print('裁決性應計數輸出完成')

    def where_is_my_output_file(self):
        '''
        回傳Output file的位置，在使用過output_excel才能用，不然就是None

        Args:
            None

        Returns:
            Output file的位置
        '''
        return self.output_loc
