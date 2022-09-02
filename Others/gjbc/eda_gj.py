#!/Users/zwl/miniconda3/bin/python3
import os
import datetime
import glob
import pandas as pd

class G:
    def __init__(self, path):
        '''
        使用方法：当前目录下应该包含下面三个文件，两个比赛文件和该脚本文件
        - competition1.csv
        - competition2.csv
        - eda.gj.py
        排名代码设计流程：
        - 比赛分为旧比赛comp1和新比赛comp2，因此会出现几个集合
            - 1. 只出现在comp1中的队伍set1
            - 2. 只出现在comp2中的队伍set2
            - 3. 既出现在comp1也出现在comp2的队伍set3
        - 排名计算方式按照提交时间越近的，作为最后的结果
        - 对于set1，那么可以直接拼接到最后的结果中，因为他们没有提交过了
        - 对于set2，也可以直接拼接
        - 对于set3，因为他们既在comp1中提交也在comp2中提交，因此set3中还会有以下情况,按照最新的提交结果进行计算
        - 最后将三个集合的人数排名进行合并

        特殊情况，如果这个人在comp1中提交了，但是也在comp2中提交了，但是两个组名不一样怎么办
        '''
        self.path = path
        # 1. 读取所有比赛排名数据
        file = []
        for f in glob.glob(path + '*.csv'):
            file.append(f)

        self.all_competition = []
        for f in file:
            self.all_competition.append(pd.read_csv(f))

        # 数据处理，转换成时间特征
        self.all_competition[0]['SubmissionDate'] = pd.to_datetime(self.all_competition[0]['SubmissionDate'])
        self.all_competition[1]['SubmissionDate'] = pd.to_datetime(self.all_competition[1]['SubmissionDate'])

        # 2. 找到两次比赛中均提交了成果的队伍，这里但是不知道comp1和comp2哪个是新的队伍
        self.common_name_set = set(self.all_competition[0].iloc[:, 1]) & set(self.all_competition[1].iloc[:, 1])
        self.comp1_name_set = set(self.all_competition[0].iloc[:, 1]) - set(self.all_competition[1].iloc[:, 1])
        self.comp2_name_set = set(self.all_competition[1].iloc[:, 1]) - set(self.all_competition[0].iloc[:, 1])
        self.common_name = {i:True for i in self.common_name_set}
        self.comp1_name = {i:True for i in self.comp1_name_set}
        self.comp2_name = {i:True for i in self.comp2_name_set}

        t0 = self.all_competition[0].iloc[:, 1]
        t1 = self.all_competition[1].iloc[:, 1]

        # 找到两个列表中共同队伍的df, 共同队伍df，就要比较提交的时间了
        df0 = self.all_competition[0][t0.map(self.common_name).fillna(False)]
        df1 = self.all_competition[1][t1.map(self.common_name).fillna(False)]

        # 因为名字顺序是乱的，所以这里进行拼接
        df = pd.merge(df0, df1, on='TeamName')
        def get_score(x):
            # 比较提交时间
            if x[2] >= x[5]:
                return x[3]
            else:
                return x[6]

        # 这里要设置axis=1， 则是一行一行索引, 比较共同队伍提交时间
        df['Score'] = df.apply(get_score, axis=1)
        info = ['TeamName', 'Score']
        self.common_team_score = df[info]

        df = self.all_competition[0][t0.map(self.comp1_name).fillna(False)]
        self.comp1_team_score = df[info]

        df = self.all_competition[1][t1.map(self.comp2_name).fillna(False)]
        self.comp2_team_score = df[info]

        # 设置比赛的终止时间
        #end_day = datetime.datetime(2021, 6, 25)
        #for d in self.all_competition:
        #    if d['SubmissionDate'].max() < end_day:

        final_score = pd.concat([self.comp1_team_score, self.comp2_team_score, self.common_team_score], axis=0)
        final_score = final_score.sort_values('Score', ascending=False)
        final_score.index = range(len(final_score))
        self.final_score = final_score.drop(len(final_score)-1)

    def get_rank(self, file_name='rank_score.txt'):
        print(self.final_score)
        self.final_score.to_csv(file_name, sep='\t', index=False)

    def get_obj_team(self, obj_score=0.6, file_name='obj_team.txt'):
        self.obj_team = self.final_score[self.final_score['Score'] > obj_score]
        print(self.obj_team)
        self.obj_team.to_csv(file_name)


data_path = './'
g = G(data_path)
# 获得最后的排名
g.get_rank()
# 获得大于多少分的队伍
g.get_obj_team(0.6)
