import pandas as pd
import numpy as np
from sklearn import preprocessing


class Concat_all():

    def __init__(self, audit_qulity_loc, BIEC_loc, dis_accr_loc):
        self.audit_qulity_loc = audit_qulity_loc
        self.BIEC_loc = BIEC_loc
        self.dis_accr_loc = dis_accr_loc
        self.outcome = None
        self.output_loc = None

    def run(self):
        cpa_fraud = pd.read_excel(self.audit_qulity_loc)
        dummy = pd.get_dummies(cpa_fraud[['產業名稱']],
                               dummy_na=False,
                               drop_first=False)
        cpa_fraud = pd.concat([cpa_fraud, dummy], axis=1)

        BIEC = pd.read_excel(self.BIEC_loc)
        dis_accr = pd.read_excel(self.dis_accr_loc)

        BIEC.drop(['簡稱'], axis=1, inplace=True)
        BIEC['月'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
        BIEC['月'] = BIEC['月'].astype('int64')

        dis_accr = dis_accr[['公司', '年', '月',
                             'Total_accural/去年Asset',
                             '1/去年Asset',
                             '銷售額變動',
                             '應收帳款變動',
                             '(銷售變動-應收變動)/去年Asset',
                             'PPE/去年Asset',
                             'ROA/去年Asset',
                             '預測的Total_accural',
                             '裁決性應計數',
                             '1/asset參數',
                             'sale-ar參數',
                             'ppe參數', 'roa參數',
                             'intercept參數']]
        dis_accr['月'].replace(r'^\s*$', np.nan, regex=True, inplace=True)
        dis_accr['月'] = dis_accr['月'].astype('int64')

        BIEC = BIEC.merge(dis_accr, on=['公司', '年', '月'], how='left')
        BIEC = BIEC.merge(cpa_fraud, on=['公司', '年'], how='left')
        BIEC['月'] = BIEC['月_x']
        BIEC.drop(['月_x', '月_y'], axis=1, inplace=True)

# ------------------------------------------------------------------
        # cpa曾經失敗 不一定同公司
        vertical_cpa_fail = set()
        # 用來存放有審計失敗的cpa，下一年才會放入vertical_cpa_fail 中
        vertical_cpa_fail_buffer = set()
        year = BIEC.iloc[[0]]['年'].values[0]
        vertical_cpa_fail_list = []
        for i in range(BIEC.shape[0]):
            if BIEC.iloc[[i]]['年'].values[0] > year:
                vertical_cpa_fail = vertical_cpa_fail.union(
                        vertical_cpa_fail_buffer)
                vertical_cpa_fail_buffer.clear()
                year = BIEC.iloc[[i]]['年'].values[0]

            if BIEC.iloc[[i]]['會計師1'].values[0] in\
                    vertical_cpa_fail or BIEC.iloc[[i]]['會計師2'].values[0]\
                    in vertical_cpa_fail:
                vertical_cpa_fail_list.append(1)
            else:
                vertical_cpa_fail_list.append(0)
            if BIEC.iloc[[i]]['審計失敗'].values[0] == 1:
                vertical_cpa_fail_buffer.add(BIEC.iloc[[i]]['會計師1'].values[0])
                vertical_cpa_fail_buffer.add(BIEC.iloc[[i]]['會計師2'].values[0])
        BIEC['全年會計師垂直傳染不同公司'] = vertical_cpa_fail_list
# -----------------------------------------------------
        # cpa 5年內曾經失敗 同公司
        # cpa曾經失敗 同公司
        vertical_cpa_fail_dict = dict()
        vertical_cpa_fail_in5year = []
        vertical_cpa_fail_sc = []
        year = BIEC.iloc[[0]]['年'].values[0]
        for i in range(BIEC.shape[0]):
            if BIEC.iloc[[i]]['年'].values[0] > year:
                year = BIEC.iloc[[i]]['年'].values[0]

            if BIEC.iloc[[i]]['審計失敗'].values[0] == 1:
                if year in vertical_cpa_fail_dict:
                    vertical_cpa_fail_dict[year].add(
                            BIEC.iloc[[i]]['公司'].values[0],
                            BIEC.iloc[[i]]['會計師1'].values[0])
                    vertical_cpa_fail_dict[year].add(
                            BIEC.iloc[[i]]['公司'].values[0],
                            BIEC.iloc[[i]]['會計師2'].values[0])
                else:
                    vertical_cpa_fail_dict[year] = Fail_recorder()
                    vertical_cpa_fail_dict[year].add(
                            BIEC.iloc[[i]]['公司'].values[0],
                            BIEC.iloc[[i]]['會計師1'].values[0])
                    vertical_cpa_fail_dict[year].add(
                            BIEC.iloc[[i]]['公司'].values[0],
                            BIEC.iloc[[i]]['會計師2'].values[0])
            is_infected = 0
            for j in range(1, 6):
                past_year = year - j
                if past_year in vertical_cpa_fail_dict:
                    if vertical_cpa_fail_dict[past_year].isin(
                            BIEC.iloc[[i]]['公司'].values[0],
                            BIEC.iloc[[i]]['會計師1'].values[0]) or\
                                    vertical_cpa_fail_dict[past_year].isin(
                                            BIEC.iloc[[i]]['公司'].values[0],
                                            BIEC.iloc[[i]]['會計師2'].values[0]):
                        is_infected = 1
            vertical_cpa_fail_in5year.append(is_infected)
            infected = 0
            for j in range(BIEC.iloc[[0]]['年'].values[0], year):
                if j in vertical_cpa_fail_dict:
                    if vertical_cpa_fail_dict[j].isin(
                            BIEC.iloc[[i]]['公司'].values[0],
                            BIEC.iloc[[i]]['會計師1'].values[0]) or\
                                    vertical_cpa_fail_dict[j].isin(
                                            BIEC.iloc[[i]]['公司'].values[0],
                                            BIEC.iloc[[i]]['會計師2'].values[0]):
                        infected = 1
            vertical_cpa_fail_sc.append(infected)

        BIEC['會計師垂直傳染五年內同公司'] = vertical_cpa_fail_in5year
        BIEC['全年會計師垂直傳染同公司'] = vertical_cpa_fail_sc

        vertical_cpa_fail_dict = None

# ----------------------------------------------------------------------------------------------
        # cpa 5年內曾經失敗 不同公司
        vertical_cpa_fail_dict = dict()
        vertical_cpa_fail_in5year = []
        year = BIEC.iloc[[0]]['年'].values[0]

        horizon_cpa_fail = []
        for i in range(BIEC.shape[0]):
            if BIEC.iloc[[i]]['年'].values[0] > year:
                year = BIEC.iloc[[i]]['年'].values[0]

            if BIEC.iloc[[i]]['審計失敗'].values[0] == 1:
                if year in vertical_cpa_fail_dict:
                    vertical_cpa_fail_dict[year].add(
                            BIEC.iloc[[i]]['會計師1'].values[0])
                    vertical_cpa_fail_dict[year].add(
                            BIEC.iloc[[i]]['會計師2'].values[0])
                else:
                    vertical_cpa_fail_dict[year] = set()
                    vertical_cpa_fail_dict[year].add(
                            BIEC.iloc[[i]]['會計師1'].values[0])
                    vertical_cpa_fail_dict[year].add(
                            BIEC.iloc[[i]]['會計師2'].values[0])
            is_infected = 0
            for j in range(1, 6):
                past_year = year - j
                if past_year in vertical_cpa_fail_dict:
                    if BIEC.iloc[[i]]['會計師1'].values[0] in\
                            vertical_cpa_fail_dict[past_year] or\
                            BIEC.iloc[[i]]['會計師2'].values[0] in\
                            vertical_cpa_fail_dict[past_year]:
                        is_infected = 1
            vertical_cpa_fail_in5year.append(is_infected)
            # 下面是給水平的
            horizon_is_infected = 0
            if year - 1 in vertical_cpa_fail_dict:
                if BIEC.iloc[[i]]['會計師1'].values[0] in\
                        vertical_cpa_fail_dict[year - 1] or\
                        BIEC.iloc[[i]]['會計師2'].values[0] in\
                        vertical_cpa_fail_dict[year - 1]:
                    horizon_is_infected = 1
            if year in vertical_cpa_fail_dict:  # 今年前期也算
                if BIEC.iloc[[i]]['會計師1'].values[0] in\
                        vertical_cpa_fail_dict[year] or\
                        BIEC.iloc[[i]]['會計師2'].values[0] in\
                        vertical_cpa_fail_dict[year]:
                    horizon_is_infected = 1
            horizon_cpa_fail.append(horizon_is_infected)
        BIEC['會計師垂直傳染五年內不同公司'] = vertical_cpa_fail_in5year
        BIEC['會計師水平傳染'] = horizon_cpa_fail
# ------------------------------------------------------
        # 事務所曾經失敗 不一定同公司
        vertical_office_fail = set()
        vertical_office_fail_buffer = set()
        # 用來存放有審計失敗的cpa，下一年才會放入vertical_cpa_fail 中
        year = BIEC.iloc[[0]]['年'].values[0]
        vertical_office_fail_list = []
        for i in range(BIEC.shape[0]):
            if BIEC.iloc[[i]]['年'].values[0] > year:
                vertical_office_fail =\
                    vertical_office_fail.union(vertical_office_fail_buffer)
                vertical_office_fail_buffer.clear()  # 清空
                year = BIEC.iloc[[i]]['年'].values[0]

            if BIEC.iloc[[i]]['事務所碼'].values[0] in vertical_office_fail:
                vertical_office_fail_list.append(1)
            else:
                vertical_office_fail_list.append(0)
            if BIEC.iloc[[i]]['審計失敗'].values[0] == 1:
                vertical_office_fail_buffer.add(
                        BIEC.iloc[[i]]['事務所碼'].values[0])
        BIEC['全年事務所垂直傳染不同公司'] = vertical_office_fail_list
# -------------------------------------------------------------------------------------------------------------
        # 事務所 5年內曾經失敗 同公司
        # 事務所 曾經失敗 同公司
        vertical_office_fail_dict = dict()
        vertical_office_fail_in5year = []
        vertical_office_fail_sc = []
        year = BIEC.iloc[[0]]['年'].values[0]

        for i in range(BIEC.shape[0]):
            if BIEC.iloc[[i]]['年'].values[0] > year:
                year = BIEC.iloc[[i]]['年'].values[0]

            if BIEC.iloc[[i]]['審計失敗'].values[0] == 1:
                if year in vertical_office_fail_dict:
                    vertical_office_fail_dict[year].add(
                            BIEC.iloc[[i]]['公司'].values[0],
                            BIEC.iloc[[i]]['事務所碼'].values[0])
                else:
                    vertical_office_fail_dict[year] = Fail_recorder()
                    vertical_office_fail_dict[year].add(
                            BIEC.iloc[[i]]['公司'].values[0],
                            BIEC.iloc[[i]]['事務所碼'].values[0])
            is_infected = 0
            for j in range(1, 6):
                past_year = year - j
                if past_year in vertical_office_fail_dict:
                    if vertical_office_fail_dict[past_year].isin(
                            BIEC.iloc[[i]]['公司'].values[0],
                            BIEC.iloc[[i]]['事務所碼'].values[0]):
                        is_infected = 1
            vertical_office_fail_in5year.append(is_infected)
            for j in range(BIEC.iloc[[0]]['年'].values[0], year):
                if j in vertical_office_fail_dict:
                    if vertical_office_fail_dict[j].isin(
                            BIEC.iloc[[i]]['公司'].values[0],
                            BIEC.iloc[[i]]['事務所碼'].values[0]):
                        infected = 1
            vertical_office_fail_sc.append(infected)

        BIEC['事務所垂直傳染五年內同公司'] = vertical_office_fail_in5year
        BIEC['全年事務所垂直傳染同公司'] = vertical_office_fail_sc
        vertical_office_fail_dict = None

# ------------------------------------------------------
        # cpa事務所 5年內曾經失敗 不同公司
        vertical_office_fail_dict = dict()
        vertical_office_fail_in5year = []

        horizon_office_fail = []
        year = BIEC.iloc[[0]]['年'].values[0]
        for i in range(BIEC.shape[0]):
            if BIEC.iloc[[i]]['年'].values[0] > year:
                year = BIEC.iloc[[i]]['年'].values[0]

            if BIEC.iloc[[i]]['審計失敗'].values[0] == 1:
                if year in vertical_office_fail_dict:
                    vertical_office_fail_dict[year].add(
                            BIEC.iloc[[i]]['事務所碼'].values[0])
                else:
                    vertical_office_fail_dict[year] = set()
                    vertical_office_fail_dict[year].add(
                            BIEC.iloc[[i]]['事務所碼'].values[0])
            is_infected = 0
            for j in range(1, 6):
                past_year = year - j
                if past_year in vertical_office_fail_dict:
                    if BIEC.iloc[[i]]['事務所碼'].values[0] in\
                            vertical_office_fail_dict[past_year]:
                        is_infected = 1
            vertical_office_fail_in5year.append(is_infected)

            horizon_is_infected = 0
            if year - 1 in vertical_office_fail_dict:
                if BIEC.iloc[[i]]['事務所碼'].values[0] in\
                        vertical_office_fail_dict[year - 1]:
                    horizon_is_infected = 1
            if year in vertical_office_fail_dict:  # 今年前期也算
                if BIEC.iloc[[i]]['事務所碼'].values[0] in\
                        vertical_office_fail_dict[year]:
                    horizon_is_infected = 1
            horizon_office_fail.append(horizon_is_infected)
        BIEC['事務所垂直傳染五年內不同公司'] = vertical_office_fail_in5year
        BIEC['事務所水平傳染'] = horizon_office_fail

# -------------------------------------------------------------------------------------------
        BIEC_dummy1 = pd.get_dummies(BIEC[['會計師1', '會計師2', '事務所代碼']],
                                     dummy_na=False,
                                     drop_first=False)

        BIEC.drop(['年度', '上市狀況', '產業別', '產業名稱',
                   '事務所碼', '事務所代碼', '簽證事務所',
                   '會計師1', '會計師2', '財報重編次數', '簡稱', '日'], axis=1, inplace=True)

        dont_normolize = ['公司', '年', '月', '簽證意見類型_A',
                          '簽證意見類型_B', '簽證意見類型_C',
                          '簽證意見類型_E', '繼續經營假設是否有疑慮_N',
                          '繼續經營假設是否有疑慮_Y',
                          '是否為大型事務所_N',
                          '是否為大型事務所_Y',
                          '審計失敗', '全年會計師垂直傳染不同公司',
                          '會計師垂直傳染五年內同公司',
                          '全年會計師垂直傳染同公司',
                          '會計師垂直傳染五年內不同公司',
                          '會計師水平傳染', '全年事務所垂直傳染不同公司',
                          '事務所垂直傳染五年內同公司',
                          '全年事務所垂直傳染同公司',
                          '事務所垂直傳染五年內不同公司',
                          '事務所水平傳染', '產業名稱_PC系統',
                          '產業名稱_主機板', '產業名稱_光電/ IO',
                          '產業名稱_其他電子', '產業名稱_其它', '產業名稱_化學',
                          '產業名稱_半導體', '產業名稱_機電設備', '產業名稱_橡膠輪胎',
                          '產業名稱_水泥', '產業名稱_汽車', '產業名稱_消費性電子',
                          '產業名稱_營建', '產業名稱_玻璃陶瓷', '產業名稱_百貨',
                          '產業名稱_石化塑膠', '產業名稱_紡織人纖', '產業名稱_網路設備',
                          '產業名稱_觀光', '產業名稱_資訊通路', '產業名稱_軟體服務',
                          '產業名稱_通訊設備', '產業名稱_造紙', '產業名稱_運輸',
                          '產業名稱_銀行保險', '產業名稱_鋼鐵', '產業名稱_電子設備',
                          '產業名稱_電子零組件', '產業名稱_電線', '產業名稱_食品']

        BIEC.replace(r'^\s*$', np.nan, regex=True, inplace=True)  # 去掉TEJ給的一堆空白

        BIEC_dummy = BIEC[dont_normolize].copy()
        BIEC.drop(dont_normolize, axis=1, inplace=True)

        BIEC = BIEC.astype('float64')
        pd.options.display.max_rows = 999
        BICE_column = BIEC.columns

        BIEC.to_numpy(dtype='float', na_value=np.nan)

        scaler = preprocessing.StandardScaler().fit(BIEC)  # 標準化
        # 需要mean時使用scaler.mean_
        # 需要標準差時用scaler.scale_
        # https://scikit-learn.org/stable/modules/preprocessing.html
        BIEC = scaler.transform(BIEC)
        BIEC = pd.DataFrame(BIEC, columns=BICE_column)

        print('BIEC_dummy.shape:', BIEC_dummy.shape)
        print('BIEC.shape:', BIEC.shape)
        BIEC = pd.concat([BIEC_dummy, BIEC], axis=1)

        BIEC = pd.concat([BIEC, BIEC_dummy1], axis=1)  # 會計師和事務所在這
        BIEC.fillna(0, inplace=True)  # 要在normalize之後
        self.outcome = BIEC

    def output(self, output_location):
        self.output_loc = output_location
        self.outcome.to_csv(self.output_loc,
                            float_format='%g',
                            encoding="utf-8",
                            index=False)

    def where_is_my_output_file(self):
        return self.output_loc


class Fail_recorder():
    def __init__(self):
        self.fail = dict()

    def add(self, company, name):
        if company in self.fail:
            self.fail[company].add_cpa(name)
        else:
            self.fail[company] = Company_recorder()
            self.fail[company].add_cpa(name)

    def isin(self, company, name):
        answer = False
        if company in self.fail:
            if self.fail[company].isin(name):
                answer = True
        return answer


class Company_recorder():
    def __init__(self):
        self.company = set()

    def add_cpa(self, name):
        self.company.add(name)

    def isin(self, name):
        if name in self.company:
            return True
        else:
            return False
