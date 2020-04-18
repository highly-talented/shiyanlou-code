import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #用来正常显示中文标签
plt.rcParams["axes.unicode_minus"]=False #用来正常显示负号
print(pd.__version__)

"""
创建series
"""
# arr = [1,2,3,4,5]
# s1 = pd.Series(arr)
# print(s1)
# n = np.random.randn(5)
# index = [1,2,3,4,5]
# s2 = pd.Series(n,index = index)
# print(s2)
# d = {"0":1,'1':1,'2':2,'3':3,'4':3}
# s3 = pd.Series(d)
# print(s3)

"""
修改索引
"""
# s2.index = ["a","b","c","d",'e']
# print(s2)
"""
Series 纵向拼接
"""
# arr = [1,2,3,4,5]
# n = np.random.randn(5)
# index = [1,2,3,4,5]
# s1 = pd.Series(arr)
# s2 = pd.Series(n,index = index)
# s3 = s2.append(s1)
"""
Series 按指定索引删除元素：
"""
# s3 = s3.drop(3)
# print(s3)

"""
Series 修改指定索引元素
"""
# s3[3] = 100 #如果不存在索引，则增加
# print(s3)

"""
series间计算, 按照索引对应计算，如果索引不同则填充为 NaN（
# """
# s5 = s2.add(s1) # 相加
# s5 = s2.sub(s1) # 相减
# s5 = s2.mul(s1) # 乘法
# s5 = s2.div(s1) # 除法
"""
series计算
"""
# s10 = s1.sum() #求和
# s10 = s3.median() #求中位数
# s10 = s3.max() # 求最大值
# s10 = s3.min() # 求最小值
"""
创建 DataFrame 数据类型
"""
# dates = pd.date_range('today', periods=6)  # 定义时间序列作为 index
# columns = ['A', 'B', 'C', 'D']  # 将列表作为列名
# num_arr = np.random.randn(6, 4)  # 传入 numpy 随机数组
# df1 = pd.DataFrame(num_arr, index=dates, columns=columns)  # . 通过 NumPy 数组创建 DataFrame
# print(df1)

# data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
#         'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
#         'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
#         'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
# labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# df2 = pd.DataFrame(data, index=labels) # 通过字典数组创建 DataFrame：
# # df2.to_excel(r"C:\Users\yangxiu\Desktop\workhome\output18.xlsx")
# # print(df2.dtypes)

"""
DataFrame 基本操作
# """
# print(df2.head(3)) #显示前三条
# print(df2.tail(3)) #显示后三条
# print(df2.index)  #显示索引
# print(df2.columns) #显示列名
# print(df2.values) # 显示值（列表）
# print(df2.describe()) # 统计
# print(df2.T) # 转置
# print(df2.sort_values(by = 'age', ascending=False))
# print(df2[1:3]) #  对 DataFrame 数据切片：
# print(df2['age']) # 对 DataFrame 通过标签查询（单列）：
# print(df2.age) # 对 DataFrame 通过标签查询（单列）：
# print(df2[['age', 'animal']] )  对 DataFrame 通过标签查询（多列）
# print(df2.iloc[1:3]) # 对 DataFrame 通过位置查询：
# df30 = df2.copy()
# print(df30) #  DataFrame 副本拷贝：
# print(df30.isnull()) #判断 DataFrame 元素是否为空：
#
# num = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index = df30.index) #添加列数据：
# df30['No'] = num  # 添加以 'No.' 为列名的新数据列  #添加列数据：

#
# df30.iat[1, 1] = 100 #根据 DataFrame 的下标值进行更改。：
# df30.loc['f', 'age'] = 1.5  #根据 DataFrame 的标签对数据进行修改：
# df30['visits'].sum() #1. 对 DataFrame 中任意列做求和操作：
# print(df30.mean()) #平均值操作：
# print(df30['animal'].str.upper())
# df30.fillna(value=1)
# df30.dropna(how= 'any')

"""
DataFrame 按指定列对齐：
"""

# left = pd.DataFrame({'key': ['foo1', 'foo2'], 'one': [1, 2]})
# right = pd.DataFrame({'key': ['foo2', 'foo3'], 'two': [4, 5]})
#
# print(left)
# print(right)
#
# # 按照 key 列对齐连接，只存在 foo2 相同，所以最后变成一行
# print(pd.merge(left, right, on='key'))

"""
DataFrame 文件操作
"""
# df3.to_csv('animal.csv')
# df_animal = pd.read_csv('animal.csv')
# df3.to_excel('animal.xlsx', sheet_name='Sheet1')
# pd.read_excel('animal.xlsx', 'Sheet1', index_col=None, na_values=['NA'])

"""
高级操作
"""
# dti = pd.date_range(start='2018-01-01', end='2018-12-31', freq='D')
# columns = ['A','B']
# s = pd.Series(np.random.rand(len(dti)), index=dti) # 建立数组
# s_sum = s[s.index.weekday == 2].sum()# 统计s 中每一个周三对应值的和：
# s_mouth = s.resample('M').mean() #统计s中每个月值的平均值：
# print(s.describe())
# print(s)
# # s = pd.DataFrame(np.random.randn(len(dti), 2), index=dti, columns = columns)
#
# print(s_mouth)
# print(dti.dtype)

# 将 Series 中的时间进行转换（秒转分钟）
# s = pd.date_range('today', periods=200, freq='S')
# ts = pd.Series(np.random.randint(0, 500, len(s)), index=s)
# print(ts)
# print(ts.resample('Min').mean())

# UTC 世界时间标准：
# s = pd.date_range('today', periods=1, freq='D')  # 获取当前时间
# ts = pd.Series(np.random.randn(len(s)), s)  # 随机数值
# ts1 = pd.Series(np.random.randn(len(s)), index=s)  # 随机数值
# ts_utc = ts.tz_localize('UTC')  # 转换为 UTC 时间
# ts_utc1 = ts1.tz_localize('UTC')  # 转换为 UTC 时间
# ts_utc2 = ts_utc.tz_convert('Asia/Shanghai') # 转换为上海时间
# print(ts_utc)
# print(ts_utc1)

#不同时间表示方式的转换：
# rng = pd.date_range('1/1/2018', periods=5, freq='M')
# ts = pd.Series(np.random.randn(len(rng)), index=rng)
# print(ts)
# ps = ts.to_period()
# print(ps)
# ps1 = ps.to_timestamp()
# print(ps1)

"""
Series 多重索引
"""
#
# letters = ['A', 'B', 'C']
# numbers = list(range(10))
# mi = pd.MultiIndex.from_product([letters, numbers])  # 设置多重索引
# s = pd.Series(np.random.randint(1,20,len(mi)), index=mi)  # 随机数
# print(s.loc["A", [1, 3, 6]])
# # s.to_excel(r"C:\Users\yangxiu\Desktop\workhome\output15.xlsx")
# print(s)
#
# print(s.loc[pd.IndexSlice[:'B', 5:]])

"""
dateFrame 多重索引
"""
# frame = pd.DataFrame(np.arange(24).reshape(12, 2),
#                 index=[list('AAAAAABBBBBB'), list('123127123124')],
#                      columns=['hello', 'shiyanlou']) #根据多重索引创建 DataFrame：
# frame.index.names = ['first', 'second'] #多重索引设置列名称：
# print(frame.groupby('second').sum()) #DataFrame 多重索引分组求和：
# #
# # print(frame.stack() )#DataFrame 行列名称转换：
# print(frame.unstack())  #DataFrame 索引转换：

# DataFrame 条件查找：
# print(frame)
# 示例数据
# data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
#         'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
#         'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
#         'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
#
# labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# df = pd.DataFrame(data, index=labels)
# df[df['age'] > 3] #查找 age 大于 3 的全部信息
# df[(df['animal'] == 'cat') & (df['age'] < 3)] #查找 age<3 且为 cat 的全部数据。
# df[df['animal'].isin(['cat', 'dog'])] #DataFrame 按关键字查询：
# df.loc[df.index[[3, 4, 8]], ['animal', 'age']] # DataFrame 按标签及列名查询。：
# df.sort_values(by=['age', 'visits'], ascending=[False, True]) # DataFrame 多条件排序：
# df['priority'].map({'yes': True, 'no': False}) #DataFrame 多值替换：
# df.groupby('animal').sum() #DataFrame 分组求和：
# print(df2.T) # 转置
# print(df2.sort_values(by = 'age', ascending=False))
# print(df2[1:3]) #  对 DataFrame 数据切片：
# print(df2['age']) # 对 DataFrame 通过标签查询（单列）：
# print(df2.age) # 对 DataFrame 通过标签查询（单列）：
# print(df2[['age', 'animal']] )  对 DataFrame 通过标签查询（多列）
# print(df2.iloc[1:3]) # 对 DataFrame 通过位置查询：
# # 使用列表拼接多个 DataFrame：
# temp_df1 = pd.DataFrame(np.random.randn(5, 4))  # 生成由随机数组成的 DataFrame 1
# temp_df2 = pd.DataFrame(np.random.randn(5, 4))  # 生成由随机数组成的 DataFrame 2
# temp_df3 = pd.DataFrame(np.random.randn(5, 4))  # 生成由随机数组成的 DataFrame 3
#
# print(temp_df1)
# print(temp_df2)
# print(temp_df3)
#
# pieces = [temp_df1, temp_df2, temp_df3]
# print(pd.concat(pieces))
#
# print(df.iloc[2:4, 1:3]) #67. 根据行列索引切片：
# print(df)

# df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
# df = pd.DataFrame(np.random.randn(5,10), columns=list('abcdefghij'))
# print(df)
# print(df.sum().idxmin())  # idxmax(), idxmin() 为 Series 函数返回最大最小值的索引值
# df.sub(df.mean(axis=1), axis=0) # DataFrame 中每个元素减去每一行的平均值：
# df = pd.DataFrame({'A': list('aaabbcaabcccbbc'),
#                    'B': [12, 345, 3, 1, 45, 14, 4, 52, 54, 23, 235, 21, 57, 3, 87]})
# s_sum = df.groupby('A')['B'].nlargest(3).sum(level=0)  # DataFrame 分组，并得到每一组中最大三个数之和：
# print(s_sum)


"""
透视表
"""
#
# df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
#                    'B': ['A', 'B', 'C'] * 4,
#                    'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
#                    'D': np.random.randn(12),
#                    'E': np.random.randn(12)})
# print(df)
#
# pd.pivot_table(df, index=['A', 'B']) # 透视表的创建：
# pd.pivot_table(df, values=['D'], index=['A', 'B']) #透视表按指定行进行聚合：
# pd.pivot_table(df, values=['D'], index=['A', 'B'], aggfunc=[np.sum, len]) # 透视表聚合方式定义：
# pd.pivot_table(df, values=['D'], index=['A', 'B'],
#                columns=['C'], aggfunc=np.sum)  # D 列按照 A,B 列进行聚合时，若关心 C 列对 D 列的影响，可以加入 columns 值进行分析。
# pd.pivot_table(df, values=['D'], index=['A', 'B'],
#                columns=['C'], aggfunc=np.sum, fill_value=0) #在透视表中由于不同的聚合方式，相应缺少的组合将为缺省值，可以加入 fill_value 对缺省值处理。


"""
绝对类型
"""

# df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": [
#                   'a', 'b', 'b', 'a', 'a', 'e']})
# df["grade"] = df["raw_grade"].astype("category") #绝对型数据定义：
# df["grade"].cat.categories = ["very good", "good", "very bad"]  #对绝对型数据重命名：
# # df["grade"] = df["grade"].cat.set_categories(
# #     ["very bad", "bad", "medium", "good", "very good"])  #重新排列绝对型数据并补充相应的缺省值：
# df.sort_values(by="grade") # 对绝对型数据进行排序：
# df.groupby("grade").size() #对绝对型数据进行分组：
# print(df)

"""
数据清洗

"""
#
# df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm',
#                                'Budapest_PaRis', 'Brussels_londOn'],
#                    'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
#                    'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
#                    'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )',
#                                '12. Air France', '"Swiss Air"']})
# df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)
#
# temp = df.From_To.str.split('_', expand=True)
# temp.columns = ['From', 'To']  # 数据列拆分
#
# # 字符标准化：
# temp['From'] = temp['From'].str.capitalize()
# temp['To'] = temp['To'].str.capitalize()
#
#
# df = df.drop('From_To', axis=1)
# df = df.join(temp)  #删除坏数据加入整理好的数据：
#
#
# df['Airline'] = df['Airline'].str.extract(
#     '([a-zA-Z\s]+)', expand=False).str.strip() #  去除多余字符：
#
#
# delays = df['RecentDelays'].apply(pd.Series)
# delays.columns = ['delay_{}'.format(n)
#                   for n in range(1, len(delays.columns)+1)]
#
# df = df.drop('RecentDelays', axis=1).join(delays)  # 格式规范


"""
数据预处理
"""
#
# df = pd.DataFrame({'name': ['Alice', 'Bob', 'Candy', 'Dany', 'Ella',
#                             'Frank', 'Grace', 'Jenny'],
#                    'grades': [58, 83, 79, 65, 93, 45, 61, 88]})
#
#
# def choice(x):
#     if x > 60:
#         return 1
#     else:
#         return 0
#
#
# df.grades = pd.Series(map(lambda x: choice(x), df.grades))


# 数据去重
# df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
# print(df)

"""
Pandas 绘图操作
"""
# ts = pd.Series(np.random.randn(100), index=pd.date_range('today', periods=100))
# ts = ts.cumsum()
# ts.plot()

#Series 可视化：
# df = pd.DataFrame(np.random.randn(100, 4),  index=pd.date_range('today', periods=100),
#                   columns=['A', 'B', 'C', 'D'])
# df = df.cumsum()
# df.plot()  #  DataFrame 折线图：
#
# df = pd.DataFrame({"xs": [1, 5, 2, 8, 1], "ys": [4, 2, 1, 9, 6]})
# df = df.cumsum()
# df.plot.scatter("xs", "ys", color='red', marker="*") #DataFrame 散点图：
#
# df = pd.DataFrame({"revenue": [57, 68, 63, 71, 72, 90, 80, 62, 59, 51, 47, 52],
#                    "advertising": [2.1, 1.9, 2.7, 3.0, 3.6, 3.2, 2.7, 2.4, 1.8, 1.6, 1.3, 1.9],
#                    "month": range(12)
#                    })
#
# ax = df.plot.bar(x = "month", y = "revenue", color="yellow", title = "时间表")
# # plt.bar(df['month'],df['advertising'], color = "orange")
# # plt.tight_layout()
# plt.xticks(df['month'], rotation = '360')
# plt.xlabel('month')
# plt.ylabel('advertising')
# plt.title("喜欢",fontsize = 16,color = 'blue')
# df.plot(x ="month",  y ="advertising", secondary_y=True, ax=ax) #DataFrame 柱形图：
# plt.show()

for i in range(0,101):
    if i % 7 ==0 or i % 10 == 7 or i // 10 ==7:
        pass
    else:
        print(i)