#!/Users/zwl/miniconda3/bin/python3
import pandas as pd
import glob
import datetime
import os

# 流程
# 因为有两个比赛，因此要分别分析，如果一个队伍在两个比赛中提交了成绩，那么第二个比赛的分数会覆盖第一个比赛的分数
# 然后将所有分数进行排名
if os.path.exists('final_score.csv'):
    os.remove('final_score.csv')

# 1. 读取所有比赛排名数据
path = './'
file = []
for f in glob.glob(path + '*.csv'):
    file.append(f)

all_competition = []
for f in file:
    all_competition.append(pd.read_csv(f))

# 2. 找到两次比赛中均提交了成果的队伍
name = None
for df in all_competition:
    if name is None:
        name = set(df.iloc[:, 1])
    else:
        name &= set(df.iloc[:, 1])
common_name_set = name
name = {i:True for i in name}

# 3. 找到两个队伍中common的部分
# 进行提交时间对比, 找到最新时间队伍对应的分数
# 找到队伍名字
# 先将所有队伍df的时间从字符串转换成时间类型
# 进行时间对比
all_competition[0]['SubmissionDate'] = pd.to_datetime(all_competition[0]['SubmissionDate'])
all_competition[1]['SubmissionDate'] = pd.to_datetime(all_competition[1]['SubmissionDate'])

t0 = all_competition[0].iloc[:, 1]
t1 = all_competition[1].iloc[:, 1]
# 找到两个列表中共同队伍的df
df0 = all_competition[0][t0.map(name).fillna(False)]
df1 = all_competition[1][t1.map(name).fillna(False)]
# 因为名字顺序是乱的，所以这里进行拼接
df = pd.merge(df0, df1, on='TeamName')
#df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2])
#df.iloc[:, 5] = pd.to_datetime(df.iloc[:, 5])

# 因为x传入的是series，所以含有标签，直接按位置标签索引元素即可
def get_score(x):
    if x[2] >= x[5]:
        return x[3]
    else:
        return x[6]

# 这里要设置axis=1， 则是一行一行索引
df['Score'] = df.apply(get_score, axis=1)
info = ['TeamName', 'Score']
common_team_score = df[info]
# 上面找到共同的分数后，就要找最开始的比赛分数
# 先找到不同的名字，然后提取分数即可
# 找到旧比赛和新比赛，并进行区分
# 旧比赛的截止时间为
end_day = datetime.datetime(2021, 6, 25)
old_team_name = None
unknow_team_name = None
old_team_score = None
for d in all_competition:
    if d['SubmissionDate'].max() < end_day:
        old_team_name = list(set(d['TeamName']) - common_name_set)
        old_team_score = d.set_index('TeamName').loc[old_team_name]['Score'].reset_index()
    else:
        unknow_team_name = list(set(d['TeamName']) - common_name_set)

final_score = pd.concat([old_team_score, common_team_score], axis=0)
final_score.index = range(len(final_score))
final_score.drop(len(final_score)-1, inplace=True)
final_score.to_csv('final_score.csv', index=False)

with open('unknow_team.txt', 'w') as f:
    for line in unknow_team_name:
        f.write(line + '\n')
