######################## import 선언부 #######################
import re
import os
import random
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

############## datafarme 보여주는 범위 설정 ###################

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 200
pd.options.display.float_format = '{:.5f}'.format


################ 경로 설정 및 파일 read #######################
load_path = './data/'  # 데이터 불러오는 경로
save_path = './preprocessing/'  # 전처리 이후 데이터 저장하는 경로

train1 = pd.read_csv(load_path + '행정안전부_도로구간_Raw_Data.csv', encoding="EUC-KR")
train2 = pd.read_csv(load_path + '자전거PM이륜차사고목록(2017~2020)_작업용.csv', encoding="EUC-KR")
trash = pd.read_csv(load_path + '버릴 csv.csv', encoding="EUC-KR")

train1 = train1.drop(columns=['ALWNC_RESN', 'ENG_RN', 'NTFC_DE',
                     'ALWNC_DE', 'MVM_RES_CD', 'MVMN_DE', 'OPERT_DE', 'MVMN_RESN'])   # 불필요 변수 삭제

#################### 기타 시각화 설정 ##############################
plt.rcParams['figure.figsize'] = [10, 6]




######################## 데이터 확인   ##################################
# print(train1.head(5))
#train1.info()
#print(train1.isnull().sum())
#print(train1[train1['RBP_CN'].isnull()])
#기점, 종점 사용가능성이 있는지 ?
train1.columns = ['기초간격', '기점', '도로구간종속구분', '도로구간일련번호', '종점',
                  '도로명', '도로명코드', '도로폭', '도로길이', '도로위계기능구분', '시군구코드', '광역도로구분코드']
#print(train1)
#train1.info()
# train1.columns=['BSI_INT','RBP_CN','RDS_DPN_SE','RDS_MAN_NO','REP_CN','RN','RN_CD','ROAD_BT','ROAD_LT','ROA_CLS_SE','SIG_CD','WDR_RD_CD']


#################### 속성 전처리 변환 함수 #############################
def make_year(x):  # year 데이터로 변환
    x = str(x)
    return int(x[:4])

def make_month(x):  # month 데이터로 변환
    x = str(x)
    return int(x[4:6])


def make_day(x):  # day 데이터로 변환
    x = str(x)
    return int(x[6:8])

def make_hour(x):  # hour 데이터로 변환
    x = str(x)
    return int(x[-3:-1])

def make_week(x):  # 요일 데이터로 변환
    week=['일','월','화','수','목','금','토']
    for i in range(0,7):
      if week[i]==x[0:1]:
        return i

def make_acci(x):  # 사고내용 데이터로 변환
    acci=['사','중','경','부'] #사망, 중상, 경상, 부상신고
    for i in range(0,4):
      if acci[i]==x[0:1]:
        return i

def make_acci_B(x):  # 사고내용 대분류 변환
    acci_B=['차대차','차대사람','차량단독'] #차대차, 차대사람, 차량단독
    for i in range(0,3):
      if acci_B[i]==x.split()[0]:
        return i

def make_acci_D(x):  # 사고내용 세부분류 변환
    #acci_B=['차대차','차대사람','차량단독'] #차대차, 차대사람, 차량단독
    #for i in range(0,3):
    return x.split()[2]

def make_road_stat_B(x):  # 노면상태 대분류 변환
    if '포장'==x.split()[0]:
      return 0
    else:
      return 1

def make_road_stat_D(x):  # 노면상태 세부분류 변환
    return x.split()[2]

def make_road_form(x):  # 도로형태 세부분류 변환
    return x.split()[2]

def make_at_car_type(x):  # 가해차종 변환
    at_car_type=['이륜','자전거','원동기','개인형이동수단(PM)'] 
    for i in range(0,4):
      if at_car_type[i]==x:
        return i

def make_gender(x):  # 성별 변환
    at_gender=['남','여','기타불명'] 
    for i in range(0,3):
      if at_gender[i]==x:
        return i

def make_number(x):  # 숫자만 추출
  if str(x) !="미분류":
    numbers = re.sub(r'[^0-9]', '', str(x))
    if numbers:
      return numbers
    else:
      return 0
  else:
    return -1


def make_h_acci(x):  # 운전자 상해정도
    h_acci=['사망','중상','경상','부상신고','상해없음','기타불명','미분류']
    check=1
    for i in range(0,7):
      if h_acci[i]==x:
        check=0
        return i
    if check==1:
      return 7


#####################################################################

train2.info()

copytrain2 = train2.copy()
pretrain2 = pd.DataFrame()

pretrain2['Year'] = copytrain2['사고번호'].apply(make_year)
pretrain2['Month'] = copytrain2['사고번호'].apply(make_month)
pretrain2['Day'] = copytrain2['사고번호'].apply(make_day)
pretrain2['Hour'] = copytrain2['사고일시'].apply(make_hour)
pretrain2['week'] = copytrain2['요일'].apply(make_week)
#시군구를 어떻게 활용할까?
pretrain2['accident'] = copytrain2['사고내용'].apply(make_acci)
pretrain2['D_acci'] = copytrain2['사망자수']       ############
pretrain2['S_acci'] = copytrain2['중상자수']
pretrain2['C_acci'] = copytrain2['경상자수']
pretrain2['I_acci'] = copytrain2['부상신고자수']   # 4가지를 EPDO같은 수치로 변환하여 사용할 것인지 ?
pretrain2['acci_case_B'] = copytrain2['사고유형'].apply(make_acci_B)  
pretrain2['acci_case_D'] = copytrain2['사고유형'].apply(make_acci_D)   # 추후 interaction을 통해 파생변수 생성
pretrain2['law_viol'] = copytrain2['법규위반']
pretrain2['road_stat_B'] = copytrain2['노면상태'].apply(make_road_stat_B)
pretrain2['road_stat_D'] = copytrain2['노면상태'].apply(make_road_stat_D)
pretrain2['road_form'] = copytrain2['도로형태'].apply(make_road_form)
pretrain2['at_car_type'] = copytrain2['가해운전자 차종'].apply(make_at_car_type)
pretrain2['at_gender'] = copytrain2['가해운전자 성별'].apply(make_gender)
pretrain2['at_age'] = copytrain2['가해운전자 연령'].apply(make_number)
pretrain2['at_acci'] = copytrain2['가해운전자 상해정도'].apply(make_h_acci)
pretrain2['vt_car_type'] = copytrain2['피해운전자 차종']
pretrain2['vt_gender'] = copytrain2['피해운전자 성별'].apply(make_gender)
pretrain2['vt_age'] = copytrain2['피해운전자 연령'].apply(make_number)
pretrain2['vt_acci'] = copytrain2['피해운전자 상해정도'].apply(make_h_acci)

print(pretrain2)
print(pretrain2['acci_case_D'].value_counts())     # 사고유형 세부사항 : 13가지 항목, 기타항목이 1/3을 차지함
print(copytrain2['법규위반'].value_counts())      # 법규위반 세부사항 : 11가지 항목, 사고가 일어나면 무조건 법규위반으로 처리 -> 안전운전불이행, 안전거리 미확보 등등으로 빠진 것 같음
print(pretrain2['road_stat_D'].value_counts())      # 노면상태 세부사항 : 7가지 항목, 침수&해빙은 데이터가 없다시피 한 수준
print(copytrain2['기상상태'].value_counts())      # 기상상태 세부사항 : 6가지 항목 
print(pretrain2['road_form'].value_counts())      # 도로형태 세부사항 : 10가지 항목, 기타가 절반이상을 차지
print(pretrain2['at_age'].value_counts())      # 가해운전자 연령 : 미분류 데이터 140개(-1로 임의부여), 필요 시 범위단위로 나눌 필요성 존재
print(pretrain2['acci_case_D'].value_counts())

a = pretrain2['Year'].value_counts().sort_index(ascending=True)
visu=['Year','Month','Day','Hour']

############################ 시각화 ############################################
#1------------------------- 각 컬럼별 빈도 막대그래프 ----------------------
fig, ax = plt.subplots(ncols=5)
num=0
for i in range(0,5):
  ax = pretrain2["Year"].value_counts().sort_index(ascending=True)

ab.plot(kind="bar")

plt.show()

#2------------------------- 각 변수간 상관관계 히트맵 ----------------------

# colormap = plt.cm.PuBu
# plt.figure(figsize=(30, 25))
# plt.title("Person Correlation of Features", y = 1.05, size = 15)
# sns.heatmap(trash.astype(float).corr(),
# linewidths = 0.1, vmax = 2.0, square = True,
# cmap = colormap, linecolor = "white", annot = True,
# annot_kws = {"size" : 9})

# plt.show()

###########################전처리 데이터 csv 저장#################################


train1.to_csv(save_path + "train_FeatureSelection.csv", encoding="EUC-KR", index=False)

sr = train1['도로명'].value_counts()
# print(sr)
# print(sr.index)
df_sr = pd.DataFrame({'RN': sr.index, 'Values': sr.values})
df_sr.to_csv(save_path + "RN_Series_test.csv", encoding="EUC-KR", index=False)
pretrain2.to_csv(save_path + "자전거,PM,이륜차사고목록_전처리(2017-2020).csv", encoding="EUC-KR", index=False)


########################################################################
