# coding: utf-8
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
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore')

############## datafarme 보여주는 범위 설정 ###################

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 200
pd.options.display.float_format = '{:.5f}'.format


################ 경로 설정 및 파일 read #######################
load_path = 'C:/Users/user/Koroad/data/'  # 데이터 불러오는 경로
save_path = './preprocessing/'  # 전처리 이후 데이터 저장하는 경로

train_pm = pd.read_csv(
    load_path + '2016_2020_가해자개인형이동수단(PM)사고정보.csv', encoding="EUC_KR")

train_byc= pd.read_csv(
    load_path + '2016_2020_가해자자전거사고정보.csv', encoding="EUC_KR")

train_moto= pd.read_csv(
    load_path + '2016_2020_가해자이륜차(사륜_원자포함)사고정보.csv', encoding="EUC_KR")


# train1 = pd.read_csv(load_path + '행정안전부_도로구간_Raw_Data.csv', encoding="EUC-KR")
# # train2 = pd.read_csv(
# #     load_path + '자전거PM이륜차사고목록(2017~2020)_작업용.csv', encoding="EUC-KR")
# # trash = pd.read_csv(load_path + '버릴 csv.csv', encoding="EUC-KR")

# train1 = train1.drop(columns=['ALWNC_RESN', 'ENG_RN', 'NTFC_DE',
#                      'ALWNC_DE', 'MVM_RES_CD', 'MVMN_DE', 'OPERT_DE', 'MVMN_RESN'])   # 불필요 변수 삭제

#################### 기타 시각화 설정 ##############################
plt.rcParams['figure.figsize'] = [10, 6]


######################## 데이터 확인   ##################################
# print(train1.head(5))
# train1.info()
# print(train1.isnull().sum())
# print(train1[train1['RBP_CN'].isnull()])
# # 기점, 종점 사용가능성이 있는지 ?
# train1.columns = ['기초간격', '기점', '도로구간종속구분', '도로구간일련번호', '종점',
#                   '도로명', '도로명코드', '도로폭', '도로길이', '도로위계기능구분', '시군구코드', '광역도로구분코드']
# print(train1)
# train1.info()
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
    return int(x[8:10])


def make_week(x):  # 요일 데이터로 변환
    week = ['일', '월', '화', '수', '목', '금', '토']
    for i in range(0, 7):
        if week[i] == x[0:1]:
            return i


def make_acci(x):  # 사고내용 데이터로 변환
    acci = ['사', '중', '경', '부']  # 사망, 중상, 경상, 부상신고
    for i in range(0, 4):
        if acci[i] == x[0:1]:
            return i


def make_acci_B(x):  # 사고내용 대분류 변환
    acci_B = ['차대차', '차대사람', '차량단독']  # 차대차, 차대사람, 차량단독
    for i in range(0, 3):
        if acci_B[i] == x.split()[0]:
            return i


def make_acci_D(x):  # 사고내용 세부분류 변환
    # acci_B=['차대차','차대사람','차량단독'] #차대차, 차대사람, 차량단독
    # for i in range(0,3):
    return x.split()[2]


def make_road_stat_B(x):  # 노면상태 대분류 변환
    if '포장' == x.split()[0]:
        return 0
    else:
        return 1


def make_road_stat_D(x):  # 노면상태 세부분류 변환
    return x.split()[2]


def make_road_form(x):  # 도로형태 세부분류 변환
    return x.split()[2]


def make_at_car_type(x):  # 가해차종 변환
    at_car_type = ['이륜', '자전거', '원동기', '개인형이동수단(PM)']
    for i in range(0, 4):
        if at_car_type[i] == x:
            return i


def make_gender(x):  # 성별 변환
    at_gender = ['남', '여', '기타불명']
    check=1
    for i in range(0, 3):
        if at_gender[i] == x:
            check=0
            return i
    if check ==1:
        return 2

def make_number(x):  # 숫자만 추출
    if str(x) != "미분류":
        numbers = re.sub(r'[^0-9]', '', str(x))
        if numbers:
            return numbers
        else:
            return 0
    else:
        return -1
    
    
def missing_age(age):
    if age==-1:
        return 4
    else:
        return age


def make_h_acci(x):  # 운전자 상해정도
    h_acci = ['사망', '중상', '경상', '부상신고', '상해없음', '기타불명', '미분류']
    check = 1
    for i in range(0, 7):
        if h_acci[i] == x:
            check = 0
            return i
    if check == 1:
        return 7

    
def make_byc_road(x):  # 자전거도로 전처리
    if x==str('아니오') or x==np.array(['아니오']):
        return 0
    elif x==str('예') or x==np.array(['예']):
        return 3
    else:
        return int(x)
    
def make_null(x):
    if x =='예':
        return 3
    elif x=='아니오':
        return 0
    else:
        return 4
    
    
def make_night(x):
    night=['주','야']
    for i in range(2):
        if night[i]==x[0:1]:
            return i

def alcohol(x):
    al1=['해당 없음','0.030~0.049%','0.05%~0.09%','0.10%~0.14%',
         '0.15%~0.19%','0.20%~0.24%']
    for i in range(8):
        if x in al1:
            if al1[i]==x:
                return i
        else:
            return -1

def alcohol_yes(x):
    al1=['해당 없음','음주운전']
    if x in al1:
        for i in range(2):
            if al1[i]==x:
                return i
    else:
        return -1
    
    
def make_linear(x):
    line=['직선','커브ㆍ곡각 ']
    if x!= '기타구역':
        for i in range(2):
            if line[i]==x:
                return i
    else:
        return -1

def make_straight(x):
    line=['직선','우','좌']
    if x!= '기타구역':
        for i in range(3):
            if line[i]==x:
                return i
    else:
        return -1   

def make_flat(x):
    flat=['평지','오르막','내리막']
    if x in flat:
        for i in range(3):
            if flat[i]==x:
                return i
    else:
        return -1   
    
def make_cross(x):
    check=1
    cross=['교차로아님','교차로']
    for i in range(2):
        if cross[i]==x:
            check =0
            return i
    if check==1:
        return 0
        
def make_cross_cnt(x):
    check=1
    cross=['교차로아님','교차로 - 삼지','교차로 - 사지','교차로 - 오지이상']
    for i in range(4):
        if cross[i]==x:
            check=0
            return i
    if check==1:
        return 0
    
def binning(age):
    bins=[i for i in range(0,100,10)]
    a=np.digitize(age, bins)
    return a-1

def Equip_pre(eqp):
    if eqp=='착용불명' or eqp=='기타불명':
        return 'unknown'
    else:
        return eqp
    
# def Slope_OHE(x):
#     if x=='기타':
#         return 0
#     elif x=='0-2%':
#         return 1
#     elif x=='2-7%':
#         return 2
#     elif x=='7-15%':
#         return 3
#     elif x=='15-30%':
#         return 4
#     elif x=='30-60%':
#         return 5
#     else:
#         return 6
    
def make_form_B(x):
    if x == '교차로':
        return 1
    else:
        return 0
def make_driving_B(x):
    if x == '회전관련':
        return 1
    else:
        return 0
copytrain2 = train_pm.copy()
copytrain3 = train_byc.copy()
copytrain4 = train_moto.copy()


copytrain2 = copytrain2[copytrain2['발생지_시도'] =='서울']
copytrain3 = copytrain3[copytrain3['발생지_시도'] =='서울']
copytrain4 = copytrain4[copytrain4['발생지_시도'] =='서울']




pretrain2 = pd.DataFrame()
pretrain3 = pd.DataFrame()
pretrain4 = pd.DataFrame()
categorical_feature = [
    col for col in copytrain2.columns if copytrain2[col].dtypes == "object"]

def basic_pre(pre_df,copy_df):
    pre_df['Year'] = copy_df['발생일시'].apply(make_year)
    pre_df['Month'] = copy_df['발생일시'].apply(make_month)
    pre_df['Day'] = copy_df['발생일시'].apply(make_day)
    pre_df['Hour'] = copy_df['발생일시'].apply(make_hour)
    pre_df['week'] = copy_df['요일'].apply(make_week)
    pre_df['night'] = copy_df['주야'].apply(make_night)

    pre_df['accident'] = copy_df['사고내용'].apply(make_acci)
    pre_df['D_acci'] = copy_df['사망자수']
    pre_df['S_acci'] = copy_df['중상자수']
    pre_df['C_acci'] = copy_df['경상자수']
    
    pre_df['I_acci'] = copy_df['부상신고자수']   # 4가지를 EPDO같은 수치로 변환하여 사용할 것인지 ?
    pre_df['acci_case_B'] = copy_df['사고유형_대분류']  
    pre_df['acci_case_D'] = copy_df['사고유형_중분류']   # 추후 interaction을 통해 파생변수 생성
    pre_df['law_viol'] = copy_df['법규위반가해자']
    pre_df['road_stat_B'] = copy_df['노면상태_대분류']
    pre_df['road_stat_D'] = copy_df['노면상태']
    pre_df['road_form_B'] = copy_df['도로형태_대분류'].apply(make_form_B) ###
    pre_df['road_form_D'] = copy_df['도로형태']
    
    pre_df['at_gender'] = copy_df['성별가해자'].apply(make_gender)
    pre_df['at_age'] = copy_df['연령가해자'].apply(make_number)
    pre_df['at_age'] = pre_df['at_age'].apply(binning)
    pre_df['at_age'] = pre_df['at_age'].apply(missing_age)

    pre_df['at_acci'] = copy_df['신체상해정도가해자'].apply(make_h_acci)
    pre_df['vt_car_class'] = copy_df['차량용도피해자_대분류']
    pre_df['vt_car_type_A'] = copy_df['차량용도피해자_중분류']
    pre_df['vt_gender'] = copy_df['성별가해자_1'].apply(make_gender)
    pre_df['vt_age'] = copy_df['연령피해자'].apply(make_number)
    pre_df['vt_age'] = pre_df['vt_age'].apply(binning)
    pre_df['vt_age'] = pre_df['vt_age'].apply(missing_age)

    pre_df['vt_acci'] = copy_df['신체상해정도피해자'].apply(make_h_acci)
    pre_df['gu'] = copy_df['발생지_시군구']
    pre_df['alch'] = copy_df['음주측정수치가해자_대분류'].apply(alcohol_yes)
    
    pre_df['alch_cont'] = copy_df['음주측정수치가해자'].apply(alcohol)
    pre_df['at_protect'] = copy_df['보호장구가해자']
    pre_df['at_protect']= pre_df['at_protect'].apply(Equip_pre)
    pre_df['bycA']= copy_df['자전거도로_여부']

    pre_df['vt_protect_type'] = copy_df['보호장구피해자_대분류']
    pre_df['vt_protect'] = copy_df['보호장구피해자']
    pre_df['at_driving_B'] = copy_df['행동유형가해자_중분류'].apply(make_driving_B)  ###
    pre_df['at_driving_D'] = copy_df['행동유형가해자']
    pre_df['at_acci_part'] = copy_df['가해자신체상해주부위']
    pre_df['vt_acci_part'] = copy_df['피해자신체상해주부위']
    pre_df['vt_car_type_B'] = copy_df['당사자종별피해자']
    pre_df['road_linear'] = copy_df['도로선형_대분류'].apply(make_linear)
    
    pre_df['road_straight'] = copy_df['도로선형_중분류'].apply(make_straight)
    pre_df['road_flat'] = copy_df['도로선형'].apply(make_flat)

    pre_df['cross'] = copy_df['교차로형태_대분류'].apply(make_cross)
    pre_df['cross_cnt'] = copy_df['교차로형태'].apply(make_cross_cnt)
    pre_df['weather'] = copy_df['기상상태']
    
    
    
    pre_df['EPDO'] =  pre_df.apply(lambda x: x['D_acci']*12 + x['S_acci']*5 + x['C_acci']*3 + x['I_acci'], axis=1)
    
    return pre_df
pretrain2 = basic_pre(pretrain2,copytrain2)
pretrain3 = basic_pre(pretrain3,copytrain3)
pretrain4 = basic_pre(pretrain4,copytrain4)
pretrain4
pretrain2['PM']=1
pretrain3['PM']=0
pretrain4['PM']=2
pmbyc_final=pd.concat([pretrain2,pretrain3],axis=0)
pmbycmoto_final=pd.concat([pmbyc_final,pretrain4],axis=0)
pmbycmoto_final
pmbycmoto_xy=pd.concat([copytrain2,copytrain3],axis=0)
pmbycmoto_xy=pd.concat([pmbycmoto_xy,copytrain4],axis=0)

coordtest=pmbycmoto_final
pmbycmoto_final
coordtest['X']=pmbycmoto_xy['X_POINT']
coordtest['Y']=pmbycmoto_xy['Y_POINT']
pmbycmoto_final.reset_index(drop=True, inplace=True)
# coordtest.to_csv("2017_2020_PM자전거이륜차좌표작업용.csv",encoding="EUC-KR")
coordtest_reidx=coordtest.reset_index(drop=True, inplace=False)
# coordtest_reidx.to_csv("2017_2020_PM자전거이륜차좌표작업용_index_reset.csv",encoding="EUC-KR")
pretrain3['at_age'].value_counts().sort_index(ascending=True).plot(kind='bar')
plt.title('at_age')
plt.show()
pmbyc_final['vt_age'].value_counts().sort_index(ascending=True).plot(kind='bar')
plt.title('vt_age')
plt.show()
# categorical_feature = [ col for col in pretrain_final.columns if pretrain_final[col].dtypes == "object"]

plt.rcParams['font.family'] = 'NanumGothic'

for col in pmbycmoto_final.columns:
    plt.figure(figsize=(8, 4))
    pmbycmoto_final[col].value_counts().sort_index(ascending=True).plot(kind='bar')
    plt.title(col)
    plt.show()
coord_slope=pd.read_csv("서울자전거_PM결합_경사도추가.csv",encoding="EUC-KR")
coord_slope_merge=pd.read_csv("서울자전거_PM_이륜차결합_경사도추가.csv",encoding="EUC-KR")
coord_slope_merge
# school_t= pd.read_csv('자전거_PM_어린이보호구역포함11.csv', encoding="EUC_KR")
school_t= pd.read_csv('자전거_PM_이륜차_어린이보호구역포함.csv', encoding="EUC_KR")
sch_index=school_t['field_1'].unique()

a= np.array(sch_index)
np.unique(a).shape
# for index, row in pmbyc_final.iterrows():
#     if index in sch_index.values:
#         pmbyc_final.loc[index,['schollA']]='1'
#     else:
#         pmbyc_final.loc[index,['schollA']]='0'
num=0
for index, row in pmbycmoto_final.iterrows():

    if index in sch_index:

        pmbycmoto_final.loc[index,['schollA']]='1'
    else:
        pmbycmoto_final.loc[index,['schollA']]='0'
pmbycmoto_final['schollA'].value_counts()
pmbycmoto_final.to_csv("검산용2.csv",encoding="EUC-KR")
silver_t = pd.read_csv('자전거_PM_이륜차_노인보호구역포함.csv', encoding="EUC_KR")
sil_index=silver_t['field_1'].unique()
print(sil_index)
# for index, row in pmbyc_final.iterrows():
#     if index in sil_index.values:
#         pmbyc_final.loc[index,['silverA']]='1'
#     else:
#         pmbyc_final.loc[index,['silverA']]='0'

for index, row in pmbycmoto_final.iterrows():
    if index in sil_index:
        pmbycmoto_final.loc[index,['silverA']]='1'
    else:
        pmbycmoto_final.loc[index,['silverA']]='0'
pmbycmoto_final['silverA'].value_counts().sort_index(ascending=True).plot(kind='bar')
plt.title('silverA')
plt.show()
pmbycmoto_final['silverA'].value_counts()
byc_road = pd.read_csv('2미터자전거도로Null값처리_자전거PM이륜차.csv', encoding="EUC_KR")
byc_road['field_1'].unique()
pmbycmoto_final['bycA']= pmbycmoto_final['bycA'].apply(make_null)
pmbycmoto_final['bycA'].value_counts()
bycr_index=byc_road['field_1'].unique()
print(bycr_index)
# for index, row in pmbyc_final.iterrows():     
#     if index in bycr_index.values:
#         if  pmbyc_final.loc[index,['bycA']].values[0]  == 4 :
#             pmbyc_final.loc[index,['bycA']]='2'
        
#     else:
#         if pmbyc_final.loc[index,['bycA']].values[0] == 4:
#             pmbyc_final.loc[index,['bycA']]='1'

for index, row in pmbycmoto_final.iterrows():     
    if index in bycr_index:
        if  pmbycmoto_final.loc[index,['bycA']].values[0]  == 4 :
            pmbycmoto_final.loc[index,['bycA']]='2'
        
    else:
        if pmbycmoto_final.loc[index,['bycA']].values[0] == 4:
            pmbycmoto_final.loc[index,['bycA']]='1'
pmbycmoto_final = pmbycmoto_final.astype({'bycA':'str'})
pmbycmoto_final['bycA']
pmbycmoto_final['bycA'].value_counts()
# pmbyc_final.drop(['X','Y'],axis=1,inplace=True)
testa=pmbycmoto_final.reset_index(drop=True, inplace=False)
testa['SOILSLOPE']=coord_slope_merge['SOILSLOPE']
testa.info()
testa = testa.astype({'schollA':'int64'})
testa = testa.astype({'silverA':'int64'})
testa = testa.astype({'bycA':'int64'})
# df_fin= pd.read_csv(load_path + 'df_fin.csv', encoding="EUC_KR")
from sklearn.preprocessing import OneHotEncoder

df=testa

def one_hot_encoding(var, col):
    global df
    enc = OneHotEncoder()
    enc_df = pd.DataFrame(enc.fit_transform(var).toarray())
    for i in range(len(enc_df.columns)):
        enc_df.iloc[:,i] = enc_df.iloc[:,i]
    enc_df.columns = col
    df = df.join(enc_df)
one_hot_encoding(df[['law_viol']],['law_overspeed','law_cross','law_etc','law_pedestrian','law_traffic','law_distance','law_safety','law_center'])
one_hot_encoding(df[['at_protect']],['unknown','at_prot_n','at_prot_y'])
one_hot_encoding(df[['at_age']],['at_0','at_10','at_20','at_30','at_40','at_50','at_60','at_70','at_80','at_90'])
one_hot_encoding(df[['vt_age']],['vt_0','vt_10','vt_20','vt_30','vt_40','vt_50','vt_60','vt_70','vt_80','vt_90'])
one_hot_encoding(df[['at_gender']],['at_gender_m','at_gender_f','at_gender_na'])
one_hot_encoding(df[['vt_gender']],['vt_gender_m','vt_gender_f','vt_gender_na'])
one_hot_encoding(df[['alch_cont']],['alch_na','alch_0','alch_0.030_0.049','alch_0.05_0.09','alch_0.10_0.14','alch_0.15_0.19','alch_0.20_0.24'])
one_hot_encoding(df[['road_straight']],['straight_na','straight_s','straight_r','straight_l'])
one_hot_encoding(df[['cross_cnt']],['cross_no','cross_3','cross_4','cross_5'])
one_hot_encoding(df[['SOILSLOPE']],['slope_0_2','slope_15_30','slope_2_7','slope_30_60','slope_60_100',
                                    'slope_7_15','slope_etc'])
one_hot_encoding(df[['road_flat']],['flat_etc','flat','flat_ascent','flat_downhill'])
one_hot_encoding(df[['road_stat_B']],['dirt_road','paved_road'])
df.tail()
temp_df = df.loc[:, [(df[col].dtype != 'object') for col in df.columns]]
temp_df['acci_case_B']=df['acci_case_B']
temp_df.info()
temp_df2=temp_df.copy()
temp_df= temp_df.iloc[:,77:]
# temp_df= temp_df.iloc[:,61:]
temp_df2= temp_df2.iloc[:,31:]
temp_df['bycA']=df['bycA']
temp_df['schollA']=df['schollA']
temp_df['silverA']=df['silverA']
# temp_df['road_num']=df_fin['road_num']
# temp_df['road_y']=df_fin['road_y']
# # temp_df['log_epdo']=df_fin['log_epdo']
# # temp_df['log_ari']=df_fin['log_ari']
# # temp_df['busstop_n']=df_fin['busstop_n']
# temp_df['tree_n']=df_fin['tree_n']
temp_df['EPDO']=df['EPDO']


temp_df['PM']=df['PM']
temp_df2['bycA']=df['bycA']
temp_df2['at_driving_B']=df['at_driving_B']
# temp_df2['road_num']=df_fin['road_num']
# temp_df2['road_y']=df_fin['road_y']
# temp_df2['tree_n']=df_fin['tree_n']
temp_df2['EPDO']=df['EPDO']

temp_df2['PM']=df['PM']
# temp_df['road_num']=df_fin['road_num']
# temp_df['road_y']=df_fin['road_y']
# # temp_df['log_epdo']=df_fin['log_epdo']
# # temp_df['log_ari']=df_fin['log_ari']
# # temp_df['busstop_n']=df_fin['busstop_n']
# temp_df['tree_n']=df_fin['tree_n']
temp_df2.info()
df_origin= temp_df.copy() ############원본

df_origin_fs = temp_df2.copy() ############원본
#####################################
temp_df2= df_origin_fs#[df_origin_fs['PM']!=2]
temp_df= df_origin#[df_origin['PM']!=2]
temp_df['PM'].value_counts()
dt_vv = temp_df[temp_df['acci_case_B']=='차대차']
dt_vp = temp_df[temp_df['acci_case_B']=='차대사람']
dt_va = temp_df[temp_df['acci_case_B']=='차량단독']
dt_vv2 = temp_df2[temp_df2['acci_case_B']=='차대차']
dt_vp2 = temp_df2[temp_df2['acci_case_B']=='차대사람']
dt_va2 = temp_df2[temp_df2['acci_case_B']=='차량단독']
dt_vv = dt_vv.loc[:,[(dt_vv[col].dtype != 'object') for col in dt_vv.columns]]
dt_vp = dt_vp.loc[:,[(dt_vp[col].dtype != 'object') for col in dt_vp.columns]]
dt_va = dt_va.loc[:,[(dt_va[col].dtype != 'object') for col in dt_va.columns]]
dt_vv2 = dt_vv2.loc[:,[(dt_vv2[col].dtype != 'object') for col in dt_vv2.columns]]
dt_vp2 = dt_vp2.loc[:,[(dt_vp2[col].dtype != 'object') for col in dt_vp2.columns]]
dt_va2 = dt_va2.loc[:,[(dt_va2[col].dtype != 'object') for col in dt_va2.columns]]
dt_vv['PM'].value_counts()
dt_vp
check_crosstap=pd.crosstab(temp_df['PM'], temp_df['bycA'], margins=True)
print(check_crosstap)
dt_vv['PM'].value_counts()
dt_vv.tail()
colormap = plt.cm.PuBu
plt.figure(figsize=(20, 15))
plt.title("Person Correlation of Features", y = 1.05, size = 40)
sns.heatmap(dt_vv.astype(float).corr(), linewidths = 0.1, vmax = 1.0,
            square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 10},fmt = '.2f')
from pycaret.classification import *
clf = setup(data =dt_vv, target = 'PM', fix_imbalance=True)
best_3 = compare_models()
lr = create_model('lr')
evaluate_model(lr)
lr = create_model('lr')
evaluate_model(lr)
clf = setup(data =dt_vp2, target = 'PM', fix_imbalance=True, numeric_features =['bycA'])
best_3 = compare_models()
lr = create_model('lr')
evaluate_model(lr)
get_ipython().run_line_magic('save', 'Pm_byc_moto_preprocessing.py 1-300')
