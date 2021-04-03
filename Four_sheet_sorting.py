import pandas as pd


class Sheet_sortor():
    '''
    這個py檔的功能是將TEJ特殊轉檔出來的四張財報依照年、月、公司的順序重新排列
    因為TEJ特殊轉檔一次只能輸出33家公司，會讓表單的順序十分混亂

    Attributes:
        sheet_address: 從TEJ存下來的資料的Excel檔案位置
    -----------------------------------------------------------
    Input:
        請直接用TEJ的特殊轉檔工具來找資料
        資料種類: 上市+上櫃
        DB類別:TEJ IERS Finance-國際會計準則
        資料名稱: IFRS以合併為主簡表(單季)-全產業
        欄位設定: 資產負債表、權益變動表、權益變動、現金流量 全部資料
        公司分類: 非金電(不含TDR)+電子產業(不含TDR)
        日期:季
        時間:2003/0101~2020/0907(但是只有從2005 6月開始的資料)
        記得取消時間的降逆(在日期選單裡)
        分隔設定:用逗號
        公司分隔
        日期分隔
        顯示空值字串 = >NA
    --------------------------------------------------------
    Output:
    整理好順序的四張表
    '''
    def __init__(self, sheet_address):
        self.sheet_address = sheet_address
        self.outcome = None
        self.output_loc = None

    def sort(self):
        print('四表排序開始')
        """
        將四張表依照年、月、公司的順序排序
        輸出excel請使用output function

        Args:
            None

        Returns:
            None
        """
        BIEC = pd.read_excel(self.sheet_address)
        BIEC.sort_values(by=['年', '月', '公司'], inplace=True)
        self.outcome = BIEC
        print('四表排序結束')

    def output(self, output_location):
        """
        輸出排序好的四張表的Excel檔

        Args:
            output_location: Excel輸出的檔案位置

        Returns:
            排序好的的Excel檔
        """
        print('已排序四表開始輸出')
        self.output_loc = output_location
        self.outcome.to_excel(output_location,
                              float_format='%g',
                              encoding="utf-8",
                              index=False)
        print('已排序四表輸出完成')

    def where_is_my_output_file(self):
        '''
        回傳Output file的位置，在使用過output才能用，不然就是None

        Args:
            None

        Returns:
            Output file的位置
        '''
        return self.output_loc
