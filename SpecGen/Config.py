'''
Author: zx
Date: 2021-04-08 13:49:35
LastEditTime: 2022-05-17 09:36:01
LastEditors: xin zhangxinbjfu@gmail.com
Description: In User Settings Edit
FilePath: /undefined/Users/zhangxin/Work/SlitlessSim/sls_lit_demo/simDemo.py
'''



class Config(object):
    def __init__(self, dataDir = '/data/'):
        self.conFiles = {'GU': dataDir + 'conf/CSST_GU2.conf', 'GV': dataDir + 'conf/CSST_GV2.conf', 'GI': dataDir + 'conf/CSST_GI2.conf'}
        self.thrFisle = {'GU': dataDir + 'conf/GU.Throughput', 'GV': dataDir + 'conf/GV.Throughput', 'GI': dataDir + 'conf/GI.Throughput'}
        self.senFisle = {'GU': dataDir + 'conf/CSST_GU.sensitivity', 'GV': dataDir + 'conf/CSST_GV.sensitivity', 'GI': dataDir + 'conf/CSST_GI.sensitivity'}
        self.orderIDs = {'A': '1st', 'B': '0st', 'C': '2st', 'D': '-1st', 'E': '-2st'}
        self.bandRanges = {'GU':[2550, 4000], 'GV': [4000,6200], 'GI': [6200, 10000]}
