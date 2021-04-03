import pandas as pd


class Time_sequal_concat():
    def __init__(self, concat_all_loc):
        self.concat_all_loc = concat_all_loc
        self.outcome = None
        self.output_loc = None

    def merge(self):
        before_time_merge = pd.read_csv(self.concat_all_loc, encoding='utf-8')
        copy = before_time_merge.copy()
        copy.drop(['審計失敗'], axis=1, inplace=True)
        for i in range(1, 3):
            past_year = copy.copy()
            title_name = 't-' + str(i)
            past_year.columns = map(lambda x: x +
                                    title_name if x not in ['公司',
                                                            '年', '月'] else x,
                                    past_year.columns)
            past_year['年'] = past_year['年'] + i
            before_time_merge = before_time_merge.merge(past_year,
                                                        on=['公司', '年', '月'],
                                                        how='left')
            before_time_merge.drop_duplicates(subset=['公司', '年', '月'],
                                              inplace=True)
            print('Complete merge:' + title_name)
        y = before_time_merge['審計失敗'].copy()
        before_time_merge.drop(['審計失敗'], axis=1, inplace=True)
        before_time_merge = pd.concat([y, before_time_merge], axis=1)
        before_time_merge = before_time_merge.loc[before_time_merge['年'] >
                                                  2006]
        before_time_merge = before_time_merge.loc[before_time_merge['年'] <
                                                  2020]
        before_time_merge.fillna(0, inplace=True)
        self.outcome = before_time_merge

    def output(self, output_loc):
        self.output_loc = output_loc
        self.outcome.to_csv(self.output_loc,
                            float_format='%g',
                            encoding="utf-8",
                            index=False)

    def where_is_my_file(self):
        return self.output_loc
