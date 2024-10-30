# Cell QC Data Analysis
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename
from collections import Counter
from datetime import datetime
import warnings
warnings.simplefilter("ignore")


''' Function Declaration Collection For Graph setting, Split Cycle by Current and Extraction Parameter '''
# Graph Font setting
def setGraphFont():
    import matplotlib.font_manager as fm

    # 설치된 폰트 출력
    font_list = [font.name for font in fm.fontManager.ttflist]

    # default font 설정
    plt.rcParams['font.family'] = font_list[np.min([i for i in range(len(font_list)) if 'Times New Roman' in font_list[i]])]  # -12 : Times New Roman, -14 : Nanum gothic
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rc("axes", unicode_minus=False)

    # Configure rcParams axes.prop_cycle to simultaneously cycle cases and colors.
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
setGraphFont()


# Split Cycle by Current
def Split_Cycle(Current):
    n = len(Current)  # assign n to Current index by
    Idx = {'SChg': [], 'SDis': [], 'EChg': [], 'EDis': []}  # set Idx as dictionary
    for i in np.arange(1, n):
        if ((Current[i - 1] == 0) | (Current[i - 1] > 0)) & (Current[i] < 0):
            Idx['SDis'].append(i)
        if ((Current[i - 1] == 0) | (Current[i - 1] < 0)) & (Current[i] > 0):
            Idx['SChg'].append(i)
        if (Current[i - 1] < 0) & ((Current[i] == 0) | (Current[i] > 0)):
            Idx['EDis'].append(i - 1)
        if (Current[i - 1] > 0) & ((Current[i] == 0) | (Current[i] < 0)):
            Idx['EChg'].append(i - 1)
    return Idx


# ICA Analysis
def ICA(Voltage, Current, dV_interval):
    dV = 0 # set dv initial value
    Idx_V = []
    for i in np.arange(1,len(Voltage)):
        dV =  dV + (Voltage[i] - Voltage[i-1])
        # if dV is upper than setting dV_interval(ex.10mV), append index i not value
        if abs(dV) > dV_interval:
            Idx_V.append(i)
            dV = 0 # dV initialization
    Idx_V = np.append(0,Idx_V) # Idx_V is setting voltage scale index

    RdQ, RdV = [], []
    for i in np.arange(1,len(Idx_V)):
        if i == 1: # set RdQ, RdV initial value as zero
            RdQ.append(abs(sum(Current[0:Idx_V[0]])) / 3600)
            RdV.append(Voltage[0] - Voltage[Idx_V[0] -1])
        else: # append delta value by Idx as dV_interval scale
            CT = (df[key_df[0]][Idx_i[key_df[0]]['SChg'][0] + Idx_V[i], 0])
            PT = (df[key_df[0]][Idx_i[key_df[0]]['SChg'][0] + Idx_V[i-1], 0])
            TD = (datetime.strptime(CT, '%m/%d %H:%M:%S.%f').replace(microsecond=0) - datetime.strptime(PT,'%m/%d %H:%M:%S.%f').replace(microsecond=0)).total_seconds()
            RdQ.append(abs(sum(Current[Idx_V[i - 1]:Idx_V[i]])) * TD/len(Current[Idx_V[i - 1]:Idx_V[i]]) / 3600) # delta Capacity by dV_interval scale
            RdV.append(Voltage[Idx_V[i - 1] - 1] - Voltage[Idx_V[i]])

    dQdV = np.array(RdQ) / np.array(RdV)
    ICA_V = Voltage[Idx_V[1:len(Idx_V)]]

    Capacity = []
    for i in np.arange(0, len(RdQ)):
        if i == 0:
            Capacity.append(RdQ[0])
        else:
            Capacity.append(Capacity[i - 1] + RdQ[i])

    return [dQdV, ICA_V, Capacity]


# DVA Analysis
def DVA(Voltage, Current, dAh_interval):
    dAh = 0
    Idx_Ah = []
    for i in np.arange(0, len(Current)):
        dAh = dAh + (Current[i] / 3600)
        if abs(dAh) > dAh_interval:
            Idx_Ah.append(i)
            dAh = 0
    Idx_Ah = np.append(0,Idx_Ah)

    RdQ, RdV = [], []
    for i in np.arange(1, len(Idx_Ah)):
        CT = (df[key_df[0]][Idx_i[key_df[0]]['SChg'][0] + Idx_Ah[i], 0])
        PT = (df[key_df[0]][Idx_i[key_df[0]]['SChg'][0] + Idx_Ah[i - 1], 0])
        TD = (datetime.strptime(CT, '%m/%d %H:%M:%S.%f').replace(microsecond=0) - datetime.strptime(PT,'%m/%d %H:%M:%S.%f').replace(microsecond=0)).total_seconds()
        RdQ.append(abs(sum(Current[Idx_Ah[i - 1]:Idx_Ah[i]]) * (TD / len(Current[Idx_Ah[i - 1]:Idx_Ah[i]]))) / 3600)
        RdV.append(Voltage[Idx_Ah[i - 1]] - Voltage[Idx_Ah[i]])

    dVdQ = np.array(RdV) / np.array(RdQ)
    DVA_Q = []
    for i in np.arange(0, len(RdQ)):
        DVA_Q.append(sum(RdQ[0:(i + 1)]))

    return [dVdQ, DVA_Q]


# set path
path = 'C:/Users/jeongbs1/오토실리콘/PJT-K2 - 문서/BDS/Working Directory/050 Design/050-070 Detailed Design/040 Algorithm Design/Plan/011 Data set/000 입고 실험/240924'

''' Save QC Raw Data from Excel File to Pickle file '''
# set Excel file list
# folder_list = os.listdir(us_path)
# folder_list = [forder for forder in folder_list if os.path.isdir(os.path.join(us_path, forder))]


# df = {}
# for folder in np.arange(len(folder_list)-2,len(folder_list)):
#    file_list = os.listdir(us_path + '/' + ''.join(folder_list[folder]))
#    file_list = [file for file in file_list if file.endswith('.xlsx')]
#    for file in np.arange(0,len(file_list)):
#        data = pd.read_excel(us_path + '/' + ''.join(folder_list[folder]) +'/'+ ''.join(file_list[file]), sheet_name=3, usecols=[5, 6, 7, 8])
#        data = data.to_numpy()
#        df[str(file_list[file]).split("-")[0]] = data
#        print(str(folder_list[folder])+" in "+str(file_list[file]).split("-")[0] + ' file load finish...' + str(len(file_list) - file - 1) + ' Remains...')
# len(df)


# Set dataframe from Excel file
file_list = os.listdir(path)
file_list = [file for file in file_list if file.endswith('.csv')]
# file_list = list(set(file_list))
len(file_list)


df = {}
for num in range(len(file_list)):
    data = pd.read_csv(path + './' + ''.join(file_list[num]))
    df[num] = data.to_numpy()
    # n = df[num].shape[num]
    # index = np.arange(n).reshape(n, 1)
    # df[num] = np.hstack((index, df[num]))
    print(str(file_list[num]) + ' file load finish...' + str(len(file_list) - num - 1) + ' Remains...')
len(df)




# df = {}
# for num in range(len(file_list)):
#     data = pd.read_excel(us_path + './' + ''.join(file_list[num]), sheet_name=3, usecols=[5,6,7,8])
#     data = data.to_numpy()
#     df[str(file_list[file]).split("-")[0]] = data
#     print(str(file_list[file]).split("-")[0] + ' file load finish...' + str(len(file_list) - num - 1) + ' Remains...')
# len(df)


# Set keys by US
key_df = list(df.keys())
for num in range(len(key_df)):
    if df[(key_df[num])][-1, 3] != 0:
        df[(key_df[num])][-1, 3] = 0
len(key_df)


# Find Index where is start of charge, start of discharge, end of charge, end of discharge...
Idx_i = {}
for num in range(len(key_df)):
    Current = df[(key_df[num])][:, 2]
    for i in range(len(Current)):
        if np.abs(Current[i]) < 1:
            Current[i] = 0
    dummy = Split_Cycle(Current)
    Idx_i[(key_df[num])] = dummy
    print(str(num + 1) + ' Finish..')


plt.figure()
plt.subplot(2,1,1)
plt.plot((df[key_df[0]][Idx_i[key_df[0]]["SChg"][0]-100:Idx_i[key_df[0]]["EDis"][0]+100, 2]), linewidth=1.5)
plt.xlabel('Index')
plt.ylabel('Current(A)')
plt.tight_layout()
plt.subplot(2,1,2)
plt.plot(np.mean(df[key_df[0]][Idx_i[key_df[0]]["SChg"][0]-100:Idx_i[key_df[0]]["EDis"][0]+100, 5:],axis=1), linewidth=1.5)
plt.xlabel('Index')
plt.ylabel('Cell Voltage(V)')
plt.tight_layout()


plt.figure()
for i in range(len(df)):
    plt.plot((df[key_df[i]][Idx_i[key_df[i]]["SChg"][0]-100:Idx_i[key_df[i]]["EDis"][0]+100, 2]), linewidth=1.5, label=str(i)+"th Test")
plt.xlabel('Index')
plt.ylabel('Current(A)')
plt.legend(loc='upper right')
plt.tight_layout()

plt.figure()
for i in range(len(df)):
    # plt.plot(np.mean(df[key_df[i]][Idx_i[key_df[i]]["SChg"][0]-100:Idx_i[key_df[i]]["EChg"][0]+100, 5:],axis=1), linewidth=1.5, label=str(i)+"th Test")
    plt.plot(np.mean(df[key_df[i]][Idx_i[key_df[i]]["SDis"][0]-100:Idx_i[key_df[i]]["EDis"][0]+100, 5:],axis=1), linewidth=1.5, label=str(i)+"th Test" )
plt.xlabel('Index')
plt.ylabel('Cell Voltage(A)')
plt.legend(loc='upper right')
plt.tight_layout()


plt.figure()
for i in range(len(df)):
    # plt.plot((df[key_df[i]][Idx_i[key_df[i]]["SChg"][0]-100:Idx_i[key_df[i]]["EChg"][0]+100, 5:]), linewidth=1.5, label=str(i)+"th Test")
    plt.plot((df[key_df[i]][Idx_i[key_df[i]]["SDis"][0]-100:Idx_i[key_df[i]]["EDis"][0]+100, 5:]), linewidth=1.5, label=str(i)+"th Test")
plt.xlabel('Index')
plt.ylabel('Cell Voltage(A)')
# plt.legend(loc='upper right')
plt.tight_layout()


TD = {}
for i in range(len(df)):
    dummy = []
    for num in np.arange(1, len(df[i])):
        CT = (df[key_df[i]][num, 0])
        PT = (df[key_df[i]][num - 1, 0])
        dummy.append((datetime.strptime(CT, '%m/%d %H:%M:%S.%f').replace(microsecond=0) - datetime.strptime(PT,'%m/%d %H:%M:%S.%f').replace(microsecond=0)).total_seconds())
    TD[i] = dummy

plt.figure()
for i in range(len(df)):
    plt.plot(TD[i][:3200], "o", label=str(i)+"th Test")
    plt.xlabel('Index')
    plt.ylabel('Time Scale(s)')
    plt.legend(loc='upper right')
    plt.tight_layout()


# Get Capacity
Capacity = {"Charge" : {}, "Discharge" : {}}
cmt_cap = {"Charge" : {}, "Discharge" : {}}
C_Q, D_Q = {}, {}
C_C, D_C = {}, {}
for num1 in range(len(df)):
    # Charge Capacity
    dummy, cmt = [], []
    for num2 in range(len(Idx_i[key_df[num1]]['SChg'])):
        current = (df[key_df[num1]][Idx_i[key_df[num1]]['SChg'][num2]:Idx_i[key_df[num1]]['EChg'][num2], 2])
        for num3 in np.arange(1,len(current)):
            CT = (df[key_df[num1]][Idx_i[key_df[num1]]['SChg'][num2] + num3, 0])
            PT = (df[key_df[num1]][Idx_i[key_df[num1]]['SChg'][num2] + num3 - 1, 0])
            TD = (datetime.strptime(CT, '%m/%d %H:%M:%S.%f').replace(microsecond=0) - datetime.strptime(PT,'%m/%d %H:%M:%S.%f').replace(microsecond=0)).total_seconds()
            if num3 == 1:
                cmt.append((current[num3] * TD)/3600)
            if TD >= 1:
                dummy.append((current[num3] * TD))
                cmt.append((current[num3] * TD)/3600 + cmt[num3-1])
            elif TD == 0:
                dummy.append((current[num3] * 1))
                cmt.append((current[num3] * 1)/3600 + cmt[num3-1])
    C_Q = np.sum(np.abs(dummy)) / 3600
    C_C = np.abs(cmt)
    # Discharge Capacity
    dummy, cmt = [], []
    for num2 in range(len(Idx_i[key_df[num1]]['SDis'])):
        current = (df[key_df[num1]][Idx_i[key_df[num1]]['SDis'][num2]:Idx_i[key_df[num1]]['EDis'][num2], 2])
        for num3 in np.arange(1,len(current)):
            CT = (df[key_df[num1]][Idx_i[key_df[num1]]['SDis'][num2] + num3, 0])
            PT = (df[key_df[num1]][Idx_i[key_df[num1]]['SDis'][num2] + num3 - 1, 0])
            TD = (datetime.strptime(CT, '%m/%d %H:%M:%S.%f').replace(microsecond=0) - datetime.strptime(PT,'%m/%d %H:%M:%S.%f').replace(microsecond=0)).total_seconds()
            if num3 == 1:
                cmt.append(((current[num3] * TD)/3600))
            if TD >= 1:
                dummy.append((current[num3] * TD))
                cmt.append(((current[num3] * TD) / 3600 + cmt[num3 - 1]))
            else:
                dummy.append((current[num3] * 1))
                cmt.append((current[num3] * 1) / 3600 + cmt[num3 - 1])
    D_Q = np.sum(np.abs(dummy)) / 3600
    D_C = np.abs(cmt)
    Capacity['Charge'][num1] = C_Q
    Capacity['Discharge'][num1] = D_Q
    cmt_cap['Charge'][num1] = C_C
    cmt_cap['Discharge'][num1] = D_C


plt.figure()
for i in range(len(df)):
    plt.plot(cmt_cap['Charge'][i], (df[key_df[i]][Idx_i[key_df[i]]['SChg'][0]:Idx_i[key_df[i]]['EChg'][0],5:]))
    # plt.plot(cmt_cap['Discharge'][i], (df[key_df[i]][Idx_i[key_df[i]]['SDis'][0]:Idx_i[key_df[i]]['EDis'][0], 5:]))
plt.xlabel("Charge Capacity(Ah)")
plt.ylabel("Cell Voltage(V)")
plt.tight_layout()

plt.figure()
for i in range(len(df)):
    plt.plot(cmt_cap['Discharge'][i], (df[key_df[i]][Idx_i[key_df[i]]['SDis'][0]:Idx_i[key_df[i]]['EDis'][0],5:]))
plt.xlabel("Discharge Capacity(Ah)")
plt.ylabel("Cell Voltage(V)")
plt.tight_layout()



Cell_num = np.arange(4,14,1)
ICA_C = {}
cyc = 0
for num in range(len(df)):
    ICA_P = {}
    for i in Cell_num:
        C_dQdV, C_ICA_V, C_ICA_CAP = {}, {}, {}
        D_dQdV, D_ICA_V, D_ICA_CAP = {}, {}, {}
        for cyc in np.arange(0,len(Idx_i[key_df[0]]['SChg'])):
            C_dQdV[cyc], C_ICA_V[cyc], C_ICA_CAP[cyc] = ICA(Voltage = np.array(np.array(df[key_df[num]][Idx_i[key_df[num]]['SChg'][cyc]:Idx_i[key_df[num]]['EChg'][cyc],i])) ,
                                                            Current = np.array(np.array(df[key_df[num]][Idx_i[key_df[num]]['SChg'][cyc]:Idx_i[key_df[num]]['EChg'][cyc],2])),
                                                            dV_interval = 0.005)

        for cyc in np.arange(0,len(Idx_i[key_df[0]]['SDis'])):
            D_dQdV[cyc], D_ICA_V[cyc], D_ICA_CAP[cyc] = ICA(Voltage = np.array(np.array(df[key_df[num]][Idx_i[key_df[num]]['SDis'][cyc]+1:Idx_i[key_df[num]]['EDis'][cyc],i])) ,
                                                            Current = np.array(np.array(-df[key_df[num]][Idx_i[key_df[num]]['SDis'][cyc]+1:Idx_i[key_df[num]]['EDis'][cyc],2])),
                                                            dV_interval = 0.005)

        ICA_P[i-4] = {"Chg_Vol" : C_ICA_V, "Chg_dQdV" : C_dQdV, "Chg_Cap" : C_ICA_CAP,
                    "Dis_Vol" : D_ICA_V, "Dis_dQdV" : D_dQdV, "Dis_Cap" : D_ICA_CAP}
    ICA_C[num] = ICA_P


Param_ICA = {}
Param_ICA_C_Height, Param_ICA_C_Position, Param_ICA_D_Height, Param_ICA_D_Position = {}, {}, {}, {}
for i in range(len(df)):
    Params_ICA = {}
    for num in Cell_num:
        C_Height, C_Position = {}, {}
        for cyc in np.arange(0, 1):
            C_Height[cyc] = np.max(np.abs(ICA_C[i][num-4]['Chg_dQdV'][cyc]))
            dummy = np.where(np.abs(ICA_C[i][num-4]['Chg_dQdV'][cyc]) == np.max(np.abs(ICA_C[i][num-4]['Chg_dQdV'][cyc])))
            C_Position[cyc] = float(ICA_C[i][num-4]['Chg_Vol'][cyc][dummy])

        D_Height, D_Position = {}, {}
        for cyc in np.arange(0, 1):
            # second_index = (np.abs(ICA_C[i][num-4]['Dis_dQdV'][cyc][ICA_C[i][num-4]['Dis_dQdV'][cyc]<65])).max()
            # dummy = np.where(np.abs(ICA_C[i][num-4]['Dis_dQdV'][cyc]) == second_index)
            D_Height[cyc] = np.max(np.abs(ICA_C[i][num-4]['Dis_dQdV'][cyc][:25]))
            # D_Height[cyc] = float(ICA_C[i][num - 4]['Dis_dQdV'][cyc][dummy])
            # dummy = np.where(np.abs(ICA_C[i][num-4]['Dis_dQdV'][cyc]) == np.max(np.abs(ICA_C[i][num-4]['Dis_dQdV'][cyc])))
            D_Position[cyc] = float(ICA_C[i][num-4]['Dis_Vol'][cyc][dummy])

        Params_ICA[num-4] = {'C_Height': C_Height, 'C_Position': C_Position, 'D_Height': D_Height, 'D_Position': D_Position}
    Param_ICA[i] = Params_ICA

    Param_ICA_C_Height[i] = [x[0] for x in Cell_num-np.array(4) for key, x in Param_ICA[i][x].items() if key == 'C_Height']
    Param_ICA_C_Position[i] = [x[0] for x in Cell_num-np.array(4) for key, x in Param_ICA[i][x].items() if key == 'C_Position']
    Param_ICA_D_Height[i] = [x[0] for x in Cell_num-np.array(4) for key, x in Param_ICA[i][x].items() if key == 'D_Height']
    Param_ICA_D_Position[i] = [x[0] for x in Cell_num-np.array(4) for key, x in Param_ICA[i][x].items() if key == 'D_Position']


cyc = 0
plt.figure()
for num in range(len(df)):
    for i in np.arange(0,10,1):
        for cyc in np.arange(0, 1,1):
            plt.plot(ICA_C[num][i]["Chg_Vol"][cyc],ICA_C[num][i]["Chg_dQdV"][cyc], "o-", color="blue", alpha=0.05 + 0.95 * (10 - i)/10, label=str(i)+" Cell")
            plt.plot(ICA_C[num][i]["Dis_Vol"][cyc],ICA_C[num][i]["Dis_dQdV"][cyc], "o-", color="red", alpha=0.05 + 0.95 * (10 - i)/10, label=str(i)+" Cell")
            # plt.plot(ICA_C[i]["Chg_Vol"][cyc],ICA_C[i]["Chg_dQdV"][cyc], color="blue",alpha=0.05 + 0.95 * (41 - i)/41, label=str(i)+" Cycle")
            # plt.plot(ICA_C[i]["Dis_Vol"][cyc],ICA_C[i]["Dis_dQdV"][cyc], color="red",alpha=0.05 + 0.95 * (41 - i)/41, label=str(i)+" Cycle")
        plt.xlabel('Voltage(V)', fontweight='bold')
        plt.ylabel('dQ/dV(Ah/V)', fontweight='bold')
        # plt.legend(loc='upper left', fontsize="small")
        plt.tight_layout()



plt.figure()
for i in range(len(df)):
    plt.subplot(3, 2, 1)
    plt.scatter(Param_ICA_C_Height[i], Param_ICA_C_Position[i])
    plt.subplot(3, 2, 2)
    plt.scatter(Param_ICA_D_Height[i], Param_ICA_D_Position[i])
    plt.subplot(3, 2, (3, 4))
    plt.plot(Param_ICA_C_Height[i], 'o-', color='blue', linewidth=2, label='Chr.')
    plt.plot(Param_ICA_D_Height[i], 'o-', color='red', linewidth=2, label='Dis.')
    plt.ylabel('Max. Height')
    # plt.legend()
    plt.subplot(3, 2, (5, 6))
    plt.plot(Param_ICA_C_Position[i], 'o-', color='blue', linewidth=2, label='Chr.')
    plt.plot(Param_ICA_D_Position[i], 'o-', color='red', linewidth=2, label='Dis.')
    plt.xlabel('Cell no.')
    plt.ylabel('Position @Max. Height')
#     plt.legend()
    plt.tight_layout()



# for i in Cell_num:
#     plt.figure()
#     for cyc in np.arange(0, len(Idx["SDis"])-1,10):
#         plt.plot(ICA_C[i]["Chg_Vol"][cyc],ICA_C[i]["Chg_dQdV"][cyc], label=str(cyc)+" Cycle")
#     plt.xlabel('Voltage(V)', fontweight='bold')
#     plt.ylabel('dQ/dV(Ah/V)', fontweight='bold')
#     plt.xlim([0.4, 1.53])
#     plt.legend(loc='upper left', fontsize="small")
#     plt.title(str(i)+" Cell", fontweight='bold')
#     plt.tight_layout()


DVA_C = {}
for num in range(len(df)):
    DVA_P = {}
    for i in Cell_num:
        C_dVdQ, C_DVA_Q = {}, {}
        D_dVdQ, D_DVA_Q = {}, {}
        for cyc in np.arange(0,1):
            C_dVdQ[cyc], C_DVA_Q[cyc] = DVA(Voltage = (np.array(df[key_df[num]][Idx_i[key_df[num]]['SChg'][0]:Idx_i[key_df[num]]['EChg'][0],i])) ,
                                            Current = (np.array(df[key_df[num]][Idx_i[key_df[num]]['SChg'][0]:Idx_i[key_df[num]]['EChg'][0],2])),
                                            dAh_interval = 0.5)

        for cyc in np.arange(0, 1):
            D_dVdQ[cyc], D_DVA_Q[cyc] = DVA(Voltage = (np.array(df[key_df[num]][Idx_i[key_df[num]]['SDis'][0]:Idx_i[key_df[num]]['EDis'][0],i])),
                                            Current = (np.array(-df[key_df[num]][Idx_i[key_df[num]]['SDis'][0]:Idx_i[key_df[num]]['EDis'][0],2])),
                                            dAh_interval = 0.5)

        DVA_P[i-4] = {"Chg_Cap" : C_DVA_Q, "Chg_dVdQ" : C_dVdQ,
                    "Dis_Cap" : D_DVA_Q, "Dis_dVdQ" : D_dVdQ,}
    DVA_C[num] = DVA_P


Param_DVA = {}
for i in range(len(df)):
    Params_DVA = {}
    for num in Cell_num:
        C_Start, C_End, C_Avg = {}, {}, {}
        for cyc in np.arange(0, 1):
            C_Start[cyc] = DVA_C[i][num-4]["Chg_dVdQ"][cyc][0]
            dummy = np.min(np.where(np.array(DVA_C[i][num-4]['Chg_Cap'][cyc]) > 5))  # 105 Ah 이상 방전된 지점에서 dVdQ 추출
            C_End[cyc] = DVA_C[i][num-4]['Chg_dVdQ'][cyc][dummy]
            dummy2 = np.min(np.where(np.array(DVA_C[i][num-4]['Chg_Cap'][cyc]) > 2))  # 20~80 Ah 방전 중 dVdQ를 추출하여
            dummy3 = np.min(np.where(np.array(DVA_C[i][num-4]['Chg_Cap'][cyc]) > 4))  # 평균 값을 계산하기 위해 인덱스 추출
            C_Avg[cyc] = np.mean(DVA_C[i][num-4]['Chg_dVdQ'][cyc][dummy2:dummy3])

        D_Start, D_End, D_Avg = {}, {}, {}
        for cyc in np.arange(0, 1):
            D_Start[cyc] = DVA_C[i][num-4]["Dis_dVdQ"][cyc][0]
            dummy_0 = np.min(np.where(np.array(DVA_C[i][num-4]['Dis_Cap'][cyc]) > 6))  # 105 Ah 이상 방전된 지점에서 dVdQ 추출
            dummy_1 = np.min(np.where(np.array(DVA_C[i][num-4]['Dis_Cap'][cyc]) < 7))  # 105 Ah 이상 방전된 지점에서 dVdQ 추출
            # C_End[cyc] = DVA_C[i][num-4]['Dis_dVdQ'][cyc][dummy]
            D_End[cyc] = np.mean(DVA_C[i][num-4]['Dis_dVdQ'][cyc][dummy_0:dummy_1])
            dummy2 = np.min(np.where(np.array(DVA_C[i][num-4]['Dis_Cap'][cyc]) > 3))  # 20~80 Ah 방전 중 dVdQ를 추출하여
            dummy3 = np.min(np.where(np.array(DVA_C[i][num-4]['Dis_Cap'][cyc]) > 5))  # 평균 값을 계산하기 위해 인덱스 추출
            D_Avg[cyc] = np.mean(DVA_C[i][num-4]['Dis_dVdQ'][cyc][dummy2:dummy3])

        Params_DVA[num-4] = {'C_Start': C_Start, 'C_End': C_End, 'C_Avg': C_Avg,
                          'D_Start': D_Start, 'D_End': D_End, 'D_Avg': D_Avg}
    Param_DVA[i] = Params_DVA

cyc = 0
plt.figure()
for num in range(len(df)):
    for i in np.arange(0,10,1):
        for cyc in np.arange(0, 1, 1):
            plt.plot(DVA_C[num][i]["Dis_Cap"][cyc],DVA_C[num][i]["Dis_dVdQ"][cyc], color="red",alpha=0.05 + 0.95 * (10 - i)/10, label='Cell' + str(i))
            plt.plot(DVA_C[num][i]["Chg_Cap"][cyc],DVA_C[num][i]["Chg_dVdQ"][cyc], color="blue",alpha=0.05 + 0.95 * (10 - i)/10, label='Cell' + str(i))
            # plt.plot(DVA_C[i]["Chg_Cap"][str(cyc)],DVA_C[i]["Chg_dVdQ"][str(cyc)], alpha=0.05 + 0.95 * (32 - i)/32, label='Cell' + str(i))
        plt.xlabel('Capacity(Ah)', fontweight='bold')
        plt.ylabel('dV/dQ(V/Ah)', fontweight='bold')
        # plt.legend(loc='upper left', fontsize="small")
        plt.tight_layout()







