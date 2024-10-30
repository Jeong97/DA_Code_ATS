# System Test
import os
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from pandas.core.interchange.dataframe_protocol import DataFrame

warnings.simplefilter("ignore")


# Graph Font setting
def setGraphFont():
    import matplotlib.font_manager as fm

    # import set font
    font_list = [font.name for font in fm.fontManager.ttflist]

    # set default font
    plt.rcParams['font.family'] = font_list[np.min([i for i in range(len(font_list)) if 'Times New Roman' in font_list[i]])]  # -12 : Times New Roman, -14 : Nanum gothic
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rc("axes", unicode_minus=False)

    # Configure rcParams axes.prop_cycle to simultaneously cycle cases and colors.
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)
setGraphFont()


# set Cycle by Current
def Split_Cycle(Current):
    n = len(Current)  # assign n to Current index
    Idx = {'SChg': [], 'SDis': [], 'EChg': [], 'EDis': []}  # set Idx to dictionary, key : list
    for i in np.arange(1, n): # i loop from 1 to current index
        if ((Current[i - 1] == 0) | (Current[i - 1] > 0)) & (Current[i] < 0):
            Idx['SDis'].append(i)
        if ((Current[i - 1] == 0) | (Current[i - 1] < 0)) & (Current[i] > 0):
            Idx['SChg'].append(i)
        if (Current[i - 1] < 0) & ((Current[i] == 0) | (Current[i] > 0)):
            Idx['EDis'].append(i - 1)
        if (Current[i - 1] > 0) & ((Current[i] == 0) | (Current[i] < 0)):
            Idx['EChg'].append(i - 1)
    return Idx


# set Module & Cell no
cell_num = []
for num1 in np.arange(100,230,1):
    cell_num.append((num1))

C_OCV = [3.574846,3.575994,3.57516,3.575699,3.57523,3.575704,3.574787,3.576123,3.576167,3.575749,
         3.576718,3.575321,3.574906,3.576467,3.576234,3.575583,3.574888,3.575889,3.575311,3.575109,
         3.573652,3.575333,3.57514,3.575561,3.575064,3.575912,3.575843,3.57618,3.576041,3.57541,
         3.5765,3.575095,3.575775,3.576146,3.575463,3.575105,3.575924,3.576723,3.575512,3.575282,
         3.575927,3.574873,3.575235,3.575808,3.575606,3.575006,3.57656,3.575558,3.576534,3.575734,
         3.575501,3.574994,3.574939,3.575203,3.5754,3.575944,3.575231,3.575696,3.575354,3.575306,
         3.575959,3.575662,3.575832,3.575774,3.574267,3.575263,3.575477,3.574955,3.575373,3.575471,
         3.575027,3.576117,3.575979,3.574619,3.575153,3.574568,3.576719,3.576156,3.57529,3.57635,
         3.576493,3.576655,3.575369,3.576437,3.576097,3.575905,3.576395,3.575655,3.575808,3.576182,
         3.576079,3.575437,3.575703,3.574974,3.575636,3.575108,3.575012,3.575801,3.57627,3.575969,
         3.575552,3.575903,3.576744,3.575871,3.575738,3.575874,3.575077,3.575996,3.57602,3.576306,
         3.576096,3.57685,3.57521,3.576806,3.576879,3.575154,3.575579,3.575843,3.575558,3.576312,
         3.575176,3.575146,3.575835,3.576539,3.576114,3.576682,3.575253,3.57559,3.575714,3.576007]


# set path and df
path = r'C:\Users\jeongbs1\오토실리콘\PJT-K2 - 문서\BDS\Working Directory\050 Design\050-070 Detailed Design\040 Algorithm Design\Plan\011 Data set\000 입고 실험\240416\SDICY4.9\Room temperature AFSC'

df_list = os.listdir(path)
df_list = [file for file in df_list if file.endswith(".csv")&file.startswith("AFSC")]
len(df_list)

df_columns = ['Frequency', 'MagRatio','PhaseRatio','Zre', 'Zim',]

df = {}
for num in range(len(df_list)):
    data = pd.read_csv(path + './' + ''.join(df_list[num]), header=1, usecols=[3,4,7,10,11])
    df[num] = data.to_numpy()
    print(str(df_list[num]) + ' file load finish...' + str(len(df_list) - num - 1) + ' Remains...')
len(df)

plt.figure()
plt.boxplot(C_OCV)
plt.xticks([1], ["OCV"])
plt.ylabel("Cell Voltage(V)")
plt.tight_layout()

plt.figure()
plt.plot(cell_num, C_OCV, "o-")
plt.xticks(np.arange(100,230,10))
plt.xlabel("Cell num")
plt.ylabel("Cell OCV(V)")
plt.tight_layout()
(np.max(C_OCV) - np.min(C_OCV))*1000
(np.max(C_OCV) - np.min(C_OCV))/np.min(C_OCV)*100



y_label = ["Mag_ratio", "Phase_ratio", "Zre", "-Zim"]
for i in range(len(y_label)):
    plt.figure()
    for num in np.arange(0,len(df)):
        data = df[num]
        sorted_indices = np.argsort(data[:, 0])[::-1]
        data_sorted = data[sorted_indices]
        plt.plot((df[num][:, 0]), (df[num][:, i+1]), "o-")
        plt.xlabel("Frequency")
        plt.ylabel(y_label[i])
        plt.tight_layout()
    plt.gca().invert_xaxis()


y_label = ["Mag_ratio", "Phase_ratio", "Zre", "-Zim"]
for i in range(len(y_label)):
    plt.figure()
    for num in np.arange(0,len(df)):
        if i == 3:
            plt.plot(-(df[num][:, i + 1]), "o-")
        else:
            plt.plot((df[num][:, i+1]), "o-")
        plt.xlabel("Index")
        plt.ylabel(y_label[i])
        plt.tight_layout()


plt.figure()
for num in np.arange(0,len(df)):
    # plt.plot((df[num][:, -1]) * np.array(-1), "o-")
    plt.plot((df[num][:, -2]) , "o-")
    plt.xlabel("Index")
    plt.ylabel("Zre")
    plt.tight_layout()


plt.figure()
for num in np.arange(0,len(df)):
    # plt.title(df_columns[num+3])
    plt.plot((df[num][:, 3]), (df[num][:, 4]) * (-1), "o-")
    plt.xlabel("Zre")
    plt.ylabel("-Zim")
    # plt.legend(loc = "right")
    plt.tight_layout()



df_zim, df_zre = [], []
for key, array in df.items():
    dummy_im = [col[-1] for col in array] * np.array(-1)
    dummy_re = [col[-2] for col in array]
    df_zim.append(dummy_im)
    df_zre.append(dummy_re)
df_zim = pd.DataFrame(df_zim).T
df_zre = pd.DataFrame(df_zre).T


plt.figure()
plt.plot((df_zim.max(axis = 1) - df_zim.min(axis = 1)),"o-")
plt.xlabel("Index")
plt.ylabel("-Zim")
plt.tight_layout()

plt.figure()
plt.plot((df_zre.max(axis = 1) - df_zre.min(axis = 1)),"o-")
plt.xlabel("Index")
plt.ylabel("Zre")
plt.tight_layout()



Mag, Phase, Zre, Zim = [], [], [], []
array = [Mag, Phase, Zre, Zim]
for i in np.arange(1,5,1):
    for num in range(len(df)):
        if i == 4:
            array[i-1].append(((-(df[num][10, i]))))
        elif i == 2:
            array[i - 1].append((((df[num][10, i]))))
        elif (i == 1)|(i == 3):
            array[i - 1].append((((df[num][20, i]))))


dff = {}
cn = ["Mag_ratio", "Phase_ratio", "Zre", "-Zim"]
df_parameter = pd.DataFrame(C_OCV, columns=["OCV"])
for ix in range(len(array)):
    dff[ix] = pd.DataFrame(array[ix], columns=[cn[ix]])
    df_parameter = pd.concat([df_parameter, dff[ix]], axis=1)


import seaborn as sns
plt.figure()
# sns.heatmap(df_parameter.corr(method='pearson'), mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidth = 2)
sns.heatmap(df_parameter.corr(method='pearson'), annot=True, cmap='coolwarm', fmt=".2f", linewidth = 2)
plt.title("Pearson Correlation")
plt.tight_layout()





np.max((np.max(df_zim,axis=0))-(np.min(df_zim,axis=0)))
np.where(((np.max(df_zim,axis=0))-(np.min(df_zim,axis=0)))==np.max((np.max(df_zim,axis=0))-(np.min(df_zim,axis=0))))







# BTS Data Analysis
data = pd.read_csv(path+'/'+bts_list[0])
data = data.to_numpy()
df_bts = data

# for i in range(len(bts_list)):
#     data = pd.read_csv(path+'/'+bts_list[i])
#     data = data.to_numpy()
#     if i == 0:
#         df_bts = data
#     else:
#         df_bts = np.vstack((df_bts, data))
# df_bts = np.delete(df_bts,np.s_[70788:70799],0)
df_bts = df_bts[:89000]


time = np.arange(0,len(df_bts))/3600
plt.figure()
plt.subplot(2,1,1)
plt.plot(df_bts[:, 4], linewidth=1.5)
# plt.plot(time, df_bts[:, 4], linewidth=1.5)
# plt.xlabel("Time(hour)")
plt.ylabel("Current(A)")
plt.tight_layout()

# plt.figure()
plt.subplot(2,1,2)
plt.plot(  time, df_bts[:, 3], linewidth=1.5)
plt.xlabel("Time(hour)")
plt.ylabel("Voltage(V)")
plt.tight_layout()


# Slite Cycle by bms current
Idx = {}
Current = np.array((df_bts[:, 4]))

for i in range(len(Current)):
    if np.abs(Current[i]) < 0.5:
        Current[i] = 0

Idx = Split_Cycle(Current)
len(Idx['SChg'])
len(Idx['EChg'])
len(Idx['SDis'])
len(Idx['EDis'])
np.array(Idx['SChg'])-np.array(Idx['EChg'])

dummy_1, dummy_2 = [],[]
for num in range(len(Idx['SDis'])):
    if np.abs(Idx['SDis'][num]-Idx['SDis'][num-1]) <= 4000:
        dummy_1.append(num)
        dummy_2.append(num-1)
Idx['SDis'] = np.delete(Idx['SDis'], dummy_1)
Idx['EDis'] = np.delete(Idx['EDis'], dummy_2)
np.diff(Idx['EDis'])


# Get Capacity
Capacity = {"Charge" : {}, "Discharge" : {}}
C_Q, D_Q = {}, {}

# Charge Capacity
dummy = []
for num2 in range(len(Idx['EChg'])):
    dummy.append(np.abs(np.sum(df_bts[Idx["SChg"][num2]:Idx["EChg"][num2],4]))/3600)
    # dummy.append(np.abs((df_bts[Idx["EChg"][num2],7])))
C_Q = dummy

# Discharge Capacity
dummy = []
for num2 in range(len(Idx['EDis'])):
    dummy.append(np.abs(np.sum(df_bts[Idx["SDis"][num2]:Idx["EDis"][num2],4]))/3600)
    # dummy.append(np.abs((df_bts[Idx["EDis"][num2],7])))
D_Q = dummy

Capacity['Charge'] = C_Q
Capacity['Discharge'] = D_Q

# Coulomb_efficiency
Coulomb_efficiency = np.array(Capacity['Discharge'][1:]) / np.array(Capacity['Charge']) * 100

plt.figure()
plt.subplot(2,1,1)
plt.plot(Capacity['Charge'][:], 'o-', linewidth = 2, color = 'blue', label = 'Charge')
plt.plot(Capacity['Discharge'][1:], 'o-', linewidth = 2, color = 'red', label = 'Discharge')
plt.xticks(np.arange(0,len(Capacity['Charge'][:]),1))
# plt.xlabel("Cycle")
plt.ylabel("Capacity(Ah)")
plt.legend(loc = 'right', fontsize = 10)
plt.tight_layout()

plt.figure()
# plt.subplot(2,1,2)
plt.plot(Coulomb_efficiency[:10], 'o-', linewidth = 2)
plt.xticks(np.arange(0,len(Capacity['Charge'][:10]),1))
plt.xlabel("Cycle")
plt.ylabel("Coulomb efficiency(%)")
plt.tight_layout()

fig, ax1 = plt.subplots()
ax1.plot(Capacity['Discharge'][1:], 'o-', linewidth = 2, color="red")
ax1.set_xlabel('Cycle')
ax1.set_xticks(np.arange(0,len(Capacity['Discharge'][1:]),1))
ax1.set_ylabel('Capacity(Ah)', color="red")
ax1.set_ylim([100,108])
# y1_interval = (107.5 - 104) / 5
# ax1.yaxis.set_major_locator(MultipleLocator(y1_interval))
ax1.grid(which='both', axis='y')
# ax1.grid(False)

ax2 = ax1.twinx()
ax2.plot(Coulomb_efficiency[:], 'o-', linewidth = 2, color="blue")
ax2.set_xlabel('Cycle')
ax2.set_xticks(np.arange(0,len(Capacity['Discharge'][1:]),1))
ax2.set_ylabel('Coulomb efficiency(%)', color="blue")
ax2.set_ylim([94,100])
# y2_interval = (100 - 94) / 5
# ax2.yaxis.set_major_locator(MultipleLocator(y2_interval))
ax2.grid(which='both', axis='y')
# ax2.grid(False)
fig.tight_layout()


# Get Energy
Energy = {"Charge" : {}, "Discharge" : {}}
C_Q, D_Q = {}, {}

# Charge Energy
dummy = []
for num2 in range(len(Idx['EChg'])):
    dummy.append(np.abs(np.sum((df_bts[Idx["SChg"][num2]:Idx["EChg"][num2],4])*(df_bts[Idx["SChg"][num2]:Idx["EChg"][num2],3])))/3600/1000)
    # dummy.append(Capacity['Charge'][num2]*(np.sum(df_bts[Idx["SChg"][num2]:Idx["EChg"][num2]:5, 3]))/(len(df_bts[Idx["SChg"][num2]:Idx["EChg"][num2]:5, 3]))/1000)
C_Q = dummy

# Discharge Energy
dummy = []
for num2 in range(len(Idx['EDis'])):
    dummy.append(np.abs(np.sum((df_bts[Idx["SDis"][num2]:Idx["EDis"][num2],4])*(df_bts[Idx["SDis"][num2]:Idx["EDis"][num2],3])))/3600/1000)
    # dummy.append(Capacity['Discharge'][num2]*(np.sum(df_bts[Idx["SDis"][num2]:Idx["EDis"][num2]:5, 3]))/(len(df_bts[Idx["SDis"][num2]:Idx["EDis"][num2]:5, 3]))/1000)
D_Q = dummy

Energy['Charge'] = C_Q
Energy['Discharge'] = D_Q

# Energy_efficiency
Energy_efficiency = np.array(Energy['Discharge'][1:]) / np.array(Energy['Charge']) * 100

plt.figure()
plt.subplot(2,1,1)
plt.plot(Energy['Charge'][:], 'o-', linewidth = 2, color = 'blue', label = 'Charge')
plt.plot(Energy['Discharge'][1:], 'o-', linewidth = 2, color = 'red', label = 'Discharge')
plt.xticks(np.arange(0,len(Energy['Charge'][:]),1))
# plt.xlabel("Cycle")
plt.ylabel("Energy(kWh)")
plt.legend(loc = 'right', fontsize = 10)
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(Energy_efficiency[:], 'o-', linewidth = 2)
plt.xticks(np.arange(0,len(Energy['Charge'][:]),1))
plt.xlabel("Cycle")
plt.ylabel("Energy efficiency(%)")
plt.tight_layout()

df_1 = pd.DataFrame({0: [Capacity['Charge'][29], Capacity['Discharge'][31], Capacity['Discharge'][31] / Capacity['Charge'][29] * 100]},index=[0,1,2])
df_2 = pd.DataFrame({0: [Energy['Charge'][29], Energy['Discharge'][31], Energy['Discharge'][31] / Energy['Charge'][29] * 100]},index=[3,4,5])
pd.concat([df_1,df_2])


fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax1.plot(Capacity['Discharge'][1:], 'o-', linewidth = 2, color="red")
# ax1.set_xlabel('Cycle')
ax1.set_xticks(np.arange(0,len(Capacity['Discharge'][1:]),1))
ax1.set_ylabel('Capacity(Ah)', color="red")
ax1.set_ylim([106,110])
# y1_interval = (107.5 - 104) / 5
# ax1.yaxis.set_major_locator(MultipleLocator(y1_interval))
ax1.grid(which='both', axis='y')
# ax1.grid(False)

ax2 = ax1.twinx()
ax2.plot(Coulomb_efficiency, 'o-', linewidth = 2, color="blue")
# ax2.set_xlabel('Cycle')
ax2.set_xticks(np.arange(0,len(Capacity['Discharge'][1:]),1))
ax2.set_ylabel('Coulomb efficiency(%)', color="blue")
ax2.set_yticks(np.arange(96,99,0.5))
# y2_interval = (100 - 94) / 5
# ax2.yaxis.set_major_locator(MultipleLocator(y2_interval))
ax2.grid(which='both', axis='y')
# ax2.grid(False)
fig.tight_layout()

ax3 = fig.add_subplot(2,1,2)
ax3.plot(Energy['Discharge'][1:], 'o-', linewidth = 2, color="red")
ax3.set_xlabel('Cycle')
ax3.set_xticks(np.arange(0,len(Energy['Discharge'][1:]),1))
ax3.set_ylabel('Energy(kWh)', color="red")
ax3.set_yticks(np.arange(18.4,19.2,0.2))
# y1_interval = (107.5 - 104) / 5
# ax3.yaxis.set_major_locator(MultipleLocator(y1_interval))
ax3.grid(which='both', axis='y')
# ax1.grid(False)

ax4 = ax3.twinx()
ax4.plot(Energy_efficiency, 'o-', linewidth = 2, color="blue")
ax4.set_xlabel('Cycle')
ax4.set_xticks(np.arange(0,len(Energy['Discharge'][1:]),1))
ax4.set_ylabel('Energy efficiency(%)', color="blue")
ax4.set_yticks(np.arange(91,93.5,0.5))
# y2_interval = (100 - 94) / 5
# ax4.yaxis.set_major_locator(MultipleLocator(y2_interval))
ax4.grid(which='both', axis='y')
# ax2.grid(False)
fig.tight_layout()


# Get IR
IR = {"Charge" : {}, "Discharge" : {}}
C_Q, D_Q = {}, {}

# Charge IR
dummy = []
for num2 in range(len(Idx['SDis'])):
    dummy.append(np.abs(((df_bts[Idx["SDis"][num2]-np.array(1),3])-(df_bts[Idx["SDis"][num2]+np.array(2),3]))/(df_bts[Idx["SDis"][num2]+np.array(2),4]))*1000)
D_Q = dummy

# Discharge IR
dummy = []
for num2 in range(len(Idx['SChg'])):
    dummy.append(np.abs(((df_bts[Idx["SChg"][num2]-np.array(2),3])-(df_bts[Idx["SChg"][num2]+np.array(2),3]))/(df_bts[Idx["SChg"][num2]+np.array(2),4]))*1000)
C_Q = dummy

IR['Charge'] = C_Q
IR['Discharge'] = D_Q

plt.figure()
plt.plot(IR['Charge'][:], 'o-', linewidth = 2, label="Charge IR")
plt.xticks(np.arange(0,len(IR['Charge'][:]),1))
plt.xlabel("Cycle")
plt.ylabel("IR(mOhm)")
plt.tight_layout()

ECI = df_bts[Idx["EChg"]-np.array(1),4]
plt.figure()
plt.plot(ECI[:], "o-")
plt.xticks(np.arange(0,len(ECI[:]),1))
# plt.ylim([52.4,52.6])
plt.xlabel("Cycle")
plt.ylabel("Charge End Current(A)")
plt.tight_layout()


V_avr = []
for i in range(len(Idx["SDis"])):
    V_avr.append((np.sum(df_bts[Idx["SDis"][i]:Idx["EDis"][i]:5, 3])) / (
        len(df_bts[Idx["SDis"][i]:Idx["EDis"][i]:5, 3])))
np.mean(V_avr[:])


# Particuler voltage point
int_val = (np.min(np.where(((df_bts[Idx['SDis'][1]:Idx['EDis'][1], 3])) <= 128 * 1.23 )))
# Capacity
int_val * np.mean(np.abs(df_bts[Idx['SDis'][1]:Idx['EDis'][1], 4])) / 3600
# Energy
np.sum(np.abs((df_bts[Idx['SDis'][0]:Idx['SDis'][0]+int_val, 4]) * (df_bts[Idx['SDis'][0]:Idx['SDis'][0]+int_val, 3])))/3600/1000



# BMS Data Analysis
pbms = 3
df_804 = {}
df_805 = {}
folderlist_Day = os.listdir(path)
folderlist_Day = [forder for forder in folderlist_Day if os.path.isdir(os.path.join(path, forder))]
date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')
folderlist_Day = [folder for folder in folderlist_Day if date_pattern.match(folder)]
folderlist_Time = {}

for day in np.arange(0, len(folderlist_Day)):
    folderlist_Time[day] = os.listdir(path + '/' + ''.join(folderlist_Day[day]))
    dummy_time_804 = []
    dummy_time_805 = []
    for time in np.arange(0, len(folderlist_Time[day])):
        try:
            dummy_804 = pd.read_csv(path + '/' + ''.join(folderlist_Day[day]) + '/' + ''.join(folderlist_Time[day][time]) + '/Pack BMS #' + str(pbms) + '/' + str('se804FixedBlock.csv'), usecols=[12,13], low_memory=False)
            dummy_805 = pd.read_csv(path + '/' + ''.join(folderlist_Day[day]) + '/' + ''.join(folderlist_Time[day][time]) + '/Pack BMS #' + str(pbms) + '/' + str('se805RepeatBlock.csv'), low_memory=False)
        except:
            continue
        if time == 0:
            dummy_time_804 = dummy_804
            dummy_time_805 = dummy_805
        else:
            dummy_time_804 = np.vstack((dummy_time_804, dummy_804))
            dummy_time_805 = np.vstack((dummy_time_805, dummy_805))
            print("date : " + str(folderlist_Day[day]) + ", " + "time " + str(folderlist_Time[day][time]) + "h load Finish")

    if day == 0:
        df_804 = dummy_time_804
        df_805 = dummy_time_805
    else:
        df_804 = np.vstack((df_804, dummy_time_804))
        df_805 = np.vstack((df_805, dummy_time_805))
        # print('Data loading finish')

    if isinstance(df_804, pd.DataFrame):
        df_804 = df_804.to_numpy()
        df_805 = df_805.to_numpy()
    elif isinstance(df_804, np.ndarray):
        df_804 = df_804
        df_805 = df_805
    else:
        print("Check the data form")
    print(str(folderlist_Day[day]) + ' Data loading finish')

df_bms = np.hstack((df_805,df_804))
# df_bms = np.delete(df_bms,np.s_[75361:75416],0)
df_bms = df_bms[1050:, :]


# filtering cell voltage data & temp data
module = 8
mod_Enum = (16 * module) + 1
# for i in np.arange(1,mod_Enum):
#     for n in np.arange(0,len(df_bms)):
#         if df_bms[n, i] < 16800:
#             df_bms[n, i] = df_bms[n, i]
#         else:
#             df_bms[n, i] = df_bms[n-1, i]
df_bms[:, 1:mod_Enum] = df_bms[:, 1:mod_Enum] / 10000

temp_Snum = mod_Enum
temp_Enum = temp_Snum+(16 * module)
# for i in np.arange(temp_Snum,temp_Enum):
#     for n in np.arange(0,len(df_bms)):
#         if (df_bms[n, i] < 5000)&(df_bms[n, i] > -10):
#             df_bms[n, i] = df_bms[n, i]
#         else:
#             df_bms[n, i] = df_bms[n-1, i]
df_bms[:, temp_Snum:temp_Enum] = df_bms[:, temp_Snum:temp_Enum] / 100

bal_Snum = temp_Enum
bal_Enum = bal_Snum+(16 * module)


# Visualization Test Profile
# -1 : System Voltage -2 : Current
time = np.arange(0,len(df_bms))/3600
plt.figure()
# plt.subplot(2,1,1)
plt.plot(-df_bms[:, -2], linewidth=1.5)
# plt.plot(time, -df_bms[:, -2]/100, linewidth=1.5)
# plt.plot(-df_bms[:, -2]/100, linewidth=1.5)
plt.xlabel("Time(hour)")
plt.ylabel("Current(A)")
plt.tight_layout()

plt.figure()
# plt.subplot(2,1,2)
plt.plot(  df_bms[:, -1]/10, linewidth=1.5)
# plt.plot(  time, df_bms[:, -1]/10, linewidth=1.5)
plt.xlabel("Time(hour)")
plt.ylabel("String Link Voltage(V)")
plt.tight_layout()


# Slite Cycle by bms current
Idx = {}
Current = np.array((-df_bms[:,-2]))

for i in range(len(Current)):
    if np.abs(Current[i]) < 300:
        Current[i] = 0
    elif (Current[i]<0)&(Current[i]>-1525):
        Current[i] = 0

Idx = Split_Cycle(Current)
len(Idx['SChg'])
len(Idx['EChg'])
len(Idx['SDis'])
len(Idx['EDis'])


plt.figure()
# plt.subplot(2,1,2)
plt.plot(  df_bms[:, -1]/10, linewidth=1.5)
# plt.axvline(105*3600/55+10,0,1,color="black")
plt.xlabel("Time(s)")
plt.ylabel("String Voltage(V)")
plt.tight_layout()


plt.figure()
# plt.plot((df_bms[Idx["SChg"][-3]:Idx["EChg"][-2], 1:mod_Enum]), linewidth=1.5)
plt.plot(  time, (df_bms[:, 1:mod_Enum]), linewidth=1.5)
plt.xlabel("Time(hour)")
plt.ylabel("Cell Voltage(V)")
# plt.legend(loc="upper right")
plt.tight_layout()


# End Charge Cell Voltage
np.sum(df_bms[Idx["SChg"]-np.array(1), 1:mod_Enum],axis=1)/128
np.sum(df_bms[Idx["EChg"]-np.array(1), 1:mod_Enum],axis=1)/128
np.sum(df_bms[Idx["SDis"]-np.array(1), 1:mod_Enum],axis=1)/128
np.sum(df_bms[Idx["EDis"]-np.array(1), 1:mod_Enum],axis=1)/128


# Particuler voltage point
int_val = (np.min(np.where((np.mean(df_bms[Idx['SDis'][1]:Idx['EDis'][1], 1:mod_Enum],axis=1)) <= 1.23 )))
# Capacity
int_val * np.mean(np.abs(df_bms[Idx['SDis'][1]:Idx['EDis'][1], -2])) / 3600 / 100
# Energy
np.sum(np.abs((df_bms[Idx['SDis'][1]:Idx['SDis'][1]+int_val, -2]) * (df_bms[Idx['SDis'][1]:Idx['SDis'][1]+int_val, -1])))/3600/1000/1000

# Visulization
plt.figure()
plt.plot((df_bms[Idx['SDis'][1]:Idx['EDis'][1], 1:mod_Enum]), linewidth=1.5)
plt.axvline(int_val,0, 1, color='red',linewidth=1.5)
plt.xlabel("Time(s)")
plt.ylabel("Voltage(V)")
# plt.xticks(np.arange(0,index,1))
plt.tight_layout()


# ECV Cell Voltage
ECV = df_bms[Idx["EChg"]-np.array(1),1:mod_Enum]
plt.figure()
plt.plot(  ECV[1:], "o-",linewidth=1.5)
plt.xlabel("Cycle")
plt.ylabel("Charge End Cell Voltage(V)")
plt.xticks(np.arange(0,len(ECV[1:]),1))
# plt.xticks(np.arange(0,3,1),labels=[2, 6, 10])
plt.tight_layout()


# OCV Cell Voltage
COCV = df_bms[Idx["SDis"]-np.array(2),1:mod_Enum]
plt.figure()
plt.plot(  COCV[1:], "o-",linewidth=1.5)
# plt.plot(  np.mean(COCV[1:],axis=1), "o-",linewidth=1.5)
plt.xlabel("Cycle")
plt.ylabel("Cell OCV(V)")
plt.xticks(np.arange(0,len(COCV[1:]),1))
plt.tight_layout()


import seaborn as sns
plt.figure()
for num in np.arange(1,len(COCV),1):
    sns.histplot(COCV[num],bins=[1.485, 1.490,1.495,1.500], alpha=0.75, kde=True, label=str(num)+" Cycle")
# plt.hist(COCV[0],bins=[1.510,1.515,1.520,1.525,1.530])
plt.xlim([1.485,1.500])
plt.xticks(np.arange(1.485,1.505,0.005))
plt.xlabel("Cell OCV(V)")
plt.ylabel("Count")
plt.legend(loc="upper right", fontsize="small")
plt.tight_layout()


# Cell IR
CIR = []
for num in range(len(Idx["SDis"])):
    CIR.append(np.abs(((df_bms[Idx["SDis"][num]-np.array(2),1:mod_Enum]) - (df_bms[Idx["SDis"][num]+np.array(2),1:mod_Enum])) / np.array(-df_bms[Idx["SDis"][num]+np.array(2),-2]/100)) *1000)
(np.max(CIR[-1]) - np.min(CIR[-1]))/np.min(CIR[-1])*100
np.where((CIR[-1] >= 0.75))+np.array(1)

plt.figure()
plt.plot(  CIR[0], "o-",linewidth=1.5, color='blue', label='1st Cycle')
# plt.plot(  CIR[-1], "o-",linewidth=1.5, color='red', label='last Cycle')
# plt.ylim([0.50,0.85])
plt.xlabel("Cell no")
plt.ylabel("Cell IR(mOhm)")
# plt.xticks(ticks=np.arange(0,129,16), labels=np.arange(1,130,16))
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()


plt.figure()
plt.plot(  np.abs(CIR[1]-CIR[22])/CIR[1]*100, "o-",linewidth=1.5)
plt.xlabel("Cell no")
plt.ylabel("Cell change rate(%)")
plt.tight_layout()


# Min, Max, Mean Cell Voltage Graph
plt.figure()
plt.plot( time, np.min(df_bms[:,1:mod_Enum],axis=1), color='blue', linewidth=1.5, label='Min CellV')
plt.plot( time, np.max(df_bms[:,1:mod_Enum],axis=1), color='red', linewidth=1.5, label='Max CellV')
plt.plot( time, np.mean(df_bms[:,1:mod_Enum],axis=1), color='green', linewidth=1.5, label='Avg CellV')
# plt.ylim(1.15,1.6)
plt.xlabel("Time(hour)")
plt.ylabel("Voltage(V)")
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()


# Min, Max, Mean Cell Voltage Graph
plt.figure()
for num in range(len(Idx["SDis"])-1):
    plt.plot( np.mean(df_bms[Idx["SDis"][num+1]:Idx["EDis"][num+1],1:mod_Enum],axis=1), linewidth=1.5, label=str(num+1)+' Cycle')
# plt.ylim(1.15,1.6)
plt.xlabel("Time(hour)")
plt.ylabel("Average Cell Voltage(V)")
plt.legend(loc="upper right", fontsize="small")
plt.tight_layout()


# Min, Max, Mean Temperature Graph
plt.figure()
# plt.plot(  time, (df_bms[:,temp_Snum:temp_Enum]), linewidth=1.5)
plt.plot(  time, np.min(df_bms[:,temp_Snum:temp_Enum],axis=1), color='blue', linewidth=1.5, label='Min Temp')
plt.plot(  time, np.max(df_bms[:,temp_Snum:temp_Enum],axis=1), color='red', linewidth=1.5, label='Max Temp')
# plt.ylim(0.2,1.6)
plt.xlabel("Time(hour)")
plt.ylabel("Temperature(℃)")
# plt.legend(loc='upper left')
plt.tight_layout()

plt.figure()
plt.plot( time, np.max(df_bms[:,temp_Snum:temp_Enum],axis=1)-np.min(df_bms[:,temp_Snum:temp_Enum],axis=1), linewidth=1.5)
# plt.ylim(3,5)
plt.xlabel("Time(hour)")
plt.ylabel("Temperature deviation(℃)")
plt.tight_layout()


# Min, Max, Mean Cell Voltage Graph
plt.figure()
for num in range(len(Idx["SDis"])):
    plt.plot( np.mean(df_bms[Idx["SDis"][num]:Idx["EDis"][num],1:mod_Enum],axis=1), linewidth=1.5, label=str(num)+' Cycle')
# plt.ylim(1.15,1.6)
plt.xlabel("Time(s)")
plt.ylabel("Average Voltage(V)")
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()


# Difference of Cell Voltage
diff = {"Charge": {}, "Discharge": {}}
Charge, Discharge = {}, {}
# Charge IR by cell
dummy = []
for num2 in range(len(Idx['SDis'])):
    dummy.append((np.max(df_bms[Idx['SDis'][num2] - np.array(2), 1:mod_Enum]) - np.min(df_bms[Idx['SDis'][num2] - np.array(2), 1:mod_Enum]))*1000)
    Charge = dummy
# DisCharge IR by cell
dummy = []
for num2 in range(len(Idx['EDis'])):
    # dummy.append((np.max(df_bms[Idx['SChg'][num2] - np.array(2), 1:mod_Enum]) - np.min(df_bms[Idx['SChg'][num2] - np.array(2), 1:mod_Enum]))*1000)
    dummy.append((np.max(df_bms[Idx['EDis'][num2], 1:mod_Enum]) - np.min(df_bms[Idx['EDis'][num2], 1:mod_Enum]))*1000)
    Discharge = dummy
# Update to IR dictionary
diff['Charge'] = Charge
diff['Discharge'] = Discharge
# diff['Charge'] = np.delete(diff['Charge'],[7,12])

# Charge Cell diff
plt.figure()
plt.plot( diff['Charge'][:],'o-' ,linewidth=1.5)
plt.xlabel('Cycle')
plt.ylabel('Charge Cell Voltage Deviation(mV)')
plt.xticks(np.arange(0,len(diff['Charge'][:]),1))
plt.tight_layout()

# Charge Cell diff
plt.figure()
plt.plot( diff['Discharge'][1:],'o-' ,linewidth=1.5)
plt.xlabel('Cycle')
plt.ylabel('Discharge Cell Voltage Deviation(mV)')
plt.xticks(np.arange(0,len(diff['Discharge'][1:]),1))
plt.tight_layout()


def SplitData(df, Mod_num = module, CellInMod = 16, TempInMod = 16, StatInMod = 16):
    TempStartIdx = Mod_num * CellInMod + 1
    StatStartIdx = Mod_num * CellInMod + Mod_num * TempInMod + 1
    CellV_Rack, CellV_Module, Temp_Module, Stat_Module = {}, {}, {}, {}
    for num in np.arange(0, Mod_num):
        CellV_Module[num + 1] = df[:, 1 + CellInMod * num: 1 + CellInMod + CellInMod * num]
        Temp_Module[num + 1] = df[:, TempStartIdx + TempInMod * num: TempStartIdx + TempInMod + TempInMod * num]
        Stat_Module[num + 1] = df[:, StatStartIdx + StatInMod * num: StatStartIdx + StatInMod + StatInMod * num]
    return CellV_Module, Temp_Module, Stat_Module

CellV_Module = SplitData(df_bms)[0]
Temp_Module = SplitData(df_bms)[1]
Stat_Module = SplitData(df_bms)[2]


plt.figure()
# plt.plot(CellV_Module[4], linewidth=1.5, label=cell_num)
plt.plot(time, CellV_Module[1][:,:], linewidth=1.5, label=cell_num)
plt.xlabel("Time(s)")
plt.ylabel("Voltage(V)")
plt.legend(loc="upper right", fontsize="small")
plt.tight_layout()


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
# 각 변수에 대한 데이터 생성
index = np.max(np.max(df_bms[:,temp_Snum:temp_Enum],axis=1)-np.min(df_bms[:,temp_Snum:temp_Enum],axis=1))
Point = np.where((np.max(df_bms[:,temp_Snum:temp_Enum],axis=1)-np.min(df_bms[:,temp_Snum:temp_Enum],axis=1))==index)[0][0]

M = {}
for num in np.arange(1,9,1):
    M[num] = np.array(Temp_Module[num][Point], dtype=np.float64)

# 층별 데이터
floors = [4, 3, 2, 1]  # 각 변수에 해당하는 층

# 3D 그래프 생성
fig = plt.figure()
gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 0.02,0.05], wspace=3)
axes = []
for num in np.arange(1, 3, 1):
    # Y축 데이터 (1부터 16까지의 역순)
    if num == 1:
        Y = np.arange(1, 17)[::]
        ax = fig.add_subplot(1, 2, 2, projection='3d')
    elif num == 2:
        Y = np.arange(1, 17)[::-1]
        ax = fig.add_subplot(1, 2, 1, projection='3d')
    axes.append(ax)
    # X축 데이터 배열
    N = num
    X = [M[N], M[N + 2], M[N + 4], M[N + 6]]
    # ax = fig.add_subplot(1, 2, num, projection='3d')

    # 컬러 맵 설정
    norm = Normalize(vmin=20, vmax=35)
    colormap = cm.ScalarMappable(norm=norm, cmap='jet')

    # 각 층별로 데이터 포인트 생성 및 연결
    for i, floor in enumerate(floors):
        x_layer = X[i]
        z_layer = floor * np.ones_like(Y)

        # 각 데이터 포인트의 색상을 설정
        colors = cm.jet(norm(x_layer))

        # 데이터 포인트를 "o" 형태로 표시
        ax.scatter(x_layer, Y, z_layer, color=colors, s=50, edgecolor='k')

        # 데이터 포인트를 선으로 연결
        for j in range(len(Y) - 1):
            ax.plot([x_layer[j], x_layer[j + 1]], [Y[j], Y[j + 1]], [z_layer[j], z_layer[j + 1]], color='k', linestyle='-')

    # 축 레이블 설정
    ax.set_xticks(np.arange(20, 35, 2))
    ax.set_yticks(np.arange(1, 17, 1))
    ax.set_zticks(np.arange(1, 5, 1))
    ax.set_xlabel('Temperature(°C)')
    ax.set_ylabel('Cell num')
    ax.set_zlabel('Layer')

# colormap.set_array([])
# cbar = fig.colorbar(colormap, cax=fig.add_subplot(gs[3]), shrink=5, aspect=10)
# cbar.set_label('Temperature (°C)')

    # num이 2일 때 Y축을 역순으로 설정
    # if num == 2:
    #     ax.invert_yaxis()
plt.tight_layout()


# Print under discharge cell lsit
index_5A = int(5*3600/(df_bms[Idx["EDis"][-1],-2]/100))
EDV = df_bms[Idx["EDis"][-1]-1,1:mod_Enum]
np.mean(df_bms[Idx["EDis"][-1]-index_5A,np.where(EDV>1.2)+np.array(1)] - df_bms[Idx["EDis"][-1]-1,np.where(EDV>1.2)+np.array(1)])*1000

ov_map, lc_map, dv_map, ir_map = {}, {}, {}, {}
for mod in np.arange(1,9):
    m, c = divmod(mod-1, 8)
    ov_index = (CellV_Module[mod][Idx["EChg"][-1]-1])
    lc_index = (CellV_Module[mod][Idx["EDis"][-1]-1])
    dv_index = (CellV_Module[mod][Idx["EDis"][-1]-index_5A] - CellV_Module[mod][Idx["EDis"][-1]-1])*1000
    ir_index = (CellV_Module[mod][Idx["SDis"][0]-1] - CellV_Module[mod][Idx["SDis"][0]+2])/(df_bms[Idx["SDis"][0]+2,-2]/100)*1000
    ov_map[("R"+str(m+1)+" - M"+str(c+1))] = list([np.where(ov_index > 1.536) + np.array(1)][0][0])
    lc_map[("R"+str(m+1)+" - M"+str(c+1))] = list([np.where(lc_index < 1.17) + np.array(1)][0][0])
    dv_map[("R"+str(m+1)+" - M"+str(c+1))] = list([np.where((dv_index > 30)&(lc_index >= 1.2))+np.array(1)][0][0])
    ir_map[("R"+str(m+1)+" - M"+str(c+1))] = list([np.where(ir_index > 0.70)+np.array(1)][0][0])
key_map = list(lc_map.keys())


abnormal_cell_lc,abnormal_cell_dv = [], []
for num in range(len(lc_map)):
    match_table = System_map.iloc[np.min(System_map[System_map.iloc[:,2] == key_map[num][0:2]].index+1) : np.min(System_map[System_map.iloc[:,2] == key_map[num][0:2]].index+9),:19]
    match_table = match_table[match_table.iloc[:,2] == key_map[num][-2:]].iloc[0,3:]
    match_table.index = np.arange(1,17)
    for i in range(len(lc_map[key_map[num]])):
        abnormal_cell_lc.append(match_table[match_table.index == lc_map[key_map[num]][i]].iloc[0])
    for i in range(len(dv_map[key_map[num]])):
        abnormal_cell_dv.append(match_table[match_table.index == dv_map[key_map[num]][i]].iloc[0])
len(abnormal_cell_lc)
len(abnormal_cell_dv)


# Normal Cell Discharge End Voltage
normal_cell = []
for mod in np.arange(1,41):
    lc_value = (CellV_Module[mod][Idx["EDis"][-1]-1])
    dv_value = (CellV_Module[mod][Idx["EDis"][-1]-index_5A] - CellV_Module[mod][Idx["EDis"][-1]-1])*1000
    for i in (list(np.where((dv_value < 35)&(lc_value >= 1.2)))[0]):
        normal_cell.append(CellV_Module[mod][Idx["EDis"][-1]-1,i])

plt.figure()
plt.plot(normal_cell,"o")
plt.xlabel("Cell no")
plt.ylabel("Discharge Cell Voltage(V)")
plt.tight_layout()

np.where(df_bms[Idx["EChg"][-1]+1000,1:mod_Enum]>1.530)+np.array(1)


# Visualization High Rack 10 Cell & Low Rack 10 Cell & Average Cell Voltage
c_voltage = df_bms[:,1:mod_Enum]
CellV_Rank_Chg = np.argsort(c_voltage[Idx['EChg'][-1], :])
CellV_Rank_Dis = np.argsort(c_voltage[Idx['EDis'][-1], :])

plt.figure()
plt.plot(  df_bms[:, 1:mod_Enum])
# plt.plot(  c_voltage[:, CellV_Rank_Chg[126:]], color = 'red', label="high rank 10th")
plt.plot(  c_voltage[:, CellV_Rank_Dis[:3]], color = 'blue', label="low rank 10")
# plt.plot(  np.mean(c_voltage[:, :],axis=1), color = 'green', label="mean")
# plt.plot(  np.mean(c_voltage[:, :],axis=1), color = 'green')
# plt.plot(  np.max(c_voltage[:, :],axis=1), color = 'red')
# plt.plot(  c_voltage[:, CellV_Rank_Dis[:10]], color = 'Orange')
plt.xlabel('Time(s)')
plt.ylabel('Voltage(V)')
# plt.legend(loc='upper right', fontsize='small')
plt.tight_layout()

plt.figure()
plt.plot(  np.abs((c_voltage[Idx["EDis"][0]-2, CellV_Rank_Dis[118:]])), "o-",color = 'red', label="high rank 10th")
plt.plot(  np.abs((c_voltage[Idx["EDis"][0]-2, CellV_Rank_Dis[:10]])), "o-",color = 'blue', label="low rank 10")
# plt.plot(  np.mean(c_voltage[:, :],axis=1), color = 'green', label="mean")
# plt.plot(  np.mean(c_voltage[:, :],axis=1), color = 'green')
# plt.plot(  np.max(c_voltage[:, :],axis=1), color = 'red')
# plt.plot(  c_voltage[:, CellV_Rank_Dis[:10]], color = 'Orange')
plt.xlabel('Time(s)')
plt.ylabel('Voltage(V)')
# plt.legend(loc='upper right', fontsize='small')
plt.tight_layout()


for n in np.arange(0, 8):
    m, c = divmod(n, 2)
    # print(m, c)
    print(m)

# Module Cell Voltage by Cycle
# plt.figure(figsize=(12, 10))
plt.figure()
for n in np.arange(0, module):
    m, c = divmod(n, 1)
    plt.subplot(4, 2, m+1)
    plt.suptitle("Cell Voltage (V)")
    plt.plot((CellV_Module[n+1][Idx["SDis"][-1]-10:Idx["EDis"][-1]+1000,:]), linewidth=1.5)
    # plt.ylim([1.15, 1.6])
    # plt.xticks(ticks=np.arange(0, len(Idx['SDis'])))
    # plt.xlabel('Time(s)')
    # plt.ylabel('Voltage(V)')
    # plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()


# Module Cell Deviation by cycle
plt.figure()
for n in np.arange(0, module):
    m, c = divmod(n, 1)
    plt.subplot(4, 2, m+1)
    plt.suptitle("Cell Voltage Deviation (mV)")
    plt.plot((np.max(CellV_Module[n+1], axis=1)[Idx['SDis']-np.array(2)] - np.min(CellV_Module[n+1], axis=1)[Idx['SDis']-np.array(2)])*1000, 'o-', label='Module #' + str(n+1))
    plt.ylim([0, 10])
    plt.xticks(ticks=np.arange(0, len(Idx['SDis']),1))
    plt.xlabel('Cycle')
    # plt.ylabel('Voltage(V)')
    plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()


# Module Cell End Voltage by Cycle
plt.figure()
for n in np.arange(0, module):
    m, c = divmod(n, 1)
    plt.subplot(4, 2, m+1)
    plt.suptitle("Charge End Cell Voltage (V)")
    # plt.plot((CellV_Module[n+1][Idx['SDis']-np.array(2)]), 'o-', label='Module #' + str(n+1))
    # plt.plot((CellV_Module[n+1][Idx['EDis']-np.array(1)]), 'o-')
    for i in [0,1,2]:
        plt.plot((CellV_Module[n+1][Idx['EChg'][i]-np.array(1)]), 'o-', label=str(i+1)+' Cycle Module #' + str(n+1))
    plt.ylim([1.520, 1.545])
    # plt.ylim([1.17, 1.24])
    # plt.xticks(ticks=np.arange(0, len(Idx['SDis'])))
    plt.xticks(np.arange(0, 16, 1), labels=cell_num, rotation=90)
    # plt.xlabel('Cycle')
    # plt.ylabel('Voltage(V)')
    plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()


# Module Cell IR by Cycle
# plt.figure(figsize=(12, 10))
plt.figure()
for n in np.arange(0, module):
    m, c = divmod(n, 2)
    plt.subplot(2, 2, m + 1)
    plt.plot(np.abs(np.abs((CellV_Module[n+1])[Idx['SDis'][-1]-np.array(2)] - (CellV_Module[n+1])[Idx['SDis'][-1]+np.array(2)])/(df_bms[Idx['SDis'][-1]+np.array(2),-2]/100)*1000), 'o-', label='Module #' + str(n+1))
    plt.ylim([0.4, 0.9])
    plt.xticks(np.arange(0, 16,1), labels=cell_num, rotation=90)
    # plt.xlabel('CEll no')
    # plt.ylabel('Cell IR(mOhm)')
    plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()


# Module OCV by Cycle
plt.figure(figsize=(10,6))
for n in np.arange(0, module):
    m, c = divmod(n, 1)
    plt.subplot(4, 2, m+1)
    plt.suptitle("Cell OCV (V)")
    plt.plot((CellV_Module[n+1][Idx['SDis'][1:]-np.array(2)]), 'o-', label='Module #' + str(n+1))
    # for i in [1,5,10]:
    #     plt.plot((CellV_Module[n+1][Idx['SDis'][i]-np.array(1)]), 'o-', label=str(i)+' Cycle Module #' + str(n+1))
    plt.ylim([1.480, 1.495])
    # plt.xlim([1, 9])
    # plt.xticks(np.arange(0, 16, 1), labels=cell_num, rotation=90)
    # plt.xlabel('Cycle')
    # plt.ylabel('Voltage(V)')
    # plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()


# Module Temperature by cycle
plt.figure()
for n in np.arange(0, module):
    m, c = divmod(n, 1)
    plt.subplot(4, 2, m+1)
    # plt.suptitle("Cell Voltage Deviation (mV)")
    plt.plot((Temp_Module[n+1]), label='Module #' + str(n+1))
    # plt.ylim([0, 7])
    # plt.xticks(ticks=np.arange(0, len(Idx['SDis'])))
    # plt.xlabel('Cycle')
    # plt.ylabel('Voltage(V)')
    # plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()


# Calculation balancing cell count
Bal_charge, Bal_rest = {}, {}
dummy_charge, dummy_rest = [], []
for i in np.arange(0,len(Idx['SChg'])):
    dummy_charge.append(np.sum(df_bms[Idx["SChg"][i]+1:Idx["EChg"][i]-1,bal_Snum:bal_Enum],axis=0))
    dummy_rest.append(np.sum(df_bms[Idx["EChg"][i]+1:Idx["SDis"][i+1]-1,bal_Snum:bal_Enum],axis=0))
Bal_charge = dummy_charge
Bal_rest = dummy_rest


bal_count_charge = [np.sum(Bal_charge[i]>100) for i in range(len(Bal_charge))]
bal_count_rest = [np.sum(Bal_rest[i]>100) for i in range(len(Bal_rest))]
# bal_Ah = [np.mean(Bal_calculate[i][Bal_calculate[i]>25])*2/3600*(bal_count[i]/(module*16)) for i in range(len(Bal_calculate))]

# np.corrcoef((bal_count_charge[1:]/np.array(128)*100), Coulomb_efficiency[1:])
# np.corrcoef(diff['Charge'][1:11], diff['Charge'][1:11])
#
# data = np.hstack((np.array(bal_Ah).reshape(10,1), np.array(diff['Charge'][1:11]).reshape(10,1), np.array(Coulomb_efficiency[1:11]).reshape(10,1)))
# (pd.DataFrame(data, columns=["Bal_Ah", "Dev", "CE"])).corr(method='pearson')

plt.figure()
plt.plot(Bal_rest[2], "o")
plt.ylabel("Balancing Cell Count")
# plt.xlabel("Cycle")
# plt.xticks(np.arange(0,len(bal_count),1))
plt.tight_layout()


plt.figure()
plt.plot(bal_count_charge/np.array(128)*100, "o-")
plt.ylabel("Balancing Ratio in Charge range(%)")
plt.xlabel("Cycle")
plt.xticks(np.arange(0,len(bal_count_charge),1))
plt.tight_layout()


fig, ax1 = plt.subplots()
ax1.plot(bal_count_charge/np.array(128)*100, "o-", color="blue")
ax1.set_xticks(np.arange(0,len(bal_count_charge),1))
ax1.set_ylabel("Balancing Ratio in Charge range(%)", color="blue")
ax1.set_ylim([0,70])
ax1.grid(which='both', axis='y')
# ax1.grid(False)

ax2 = ax1.twinx()
ax2.plot(Coulomb_efficiency, 'o-', linewidth = 2, color="red")
ax2.set_xlabel('Cycle')
ax2.set_xticks(np.arange(0,len(bal_count_charge),1))
ax2.set_ylabel('Coulomb efficiency(%)', color="red")
ax2.set_yticks(np.arange(97,101,0.5))
ax2.grid(which='both', axis='y')
# ax2.grid(False)
fig.tight_layout()


plt.figure()
plt.plot(bal_count_charge/np.array(128)*100, "o-", color="blue", label="Charge range")
plt.plot(bal_count_rest/np.array(128)*100, "o-", color="red", label="Rest range")
plt.ylabel("Balancing Ratio(%)")
plt.xlabel("Cycle")
plt.xticks(np.arange(0,len(bal_count_rest),1))
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()


fig, ax1 = plt.subplots()
ax1.plot((bal_count_charge[1:]/np.array(128)*100), "o-", color="blue")
# ax1.set_xlabel('Cycle')
ax1.set_xticks(np.arange(0,12,1), labels=np.arange(1,13,1))
ax1.set_ylabel('Balancing Ratio in Charge range(%)', color="blue")
ax1.set_ylim([40,100])
# y1_interval = (107.5 - 104) / 5
# ax1.yaxis.set_major_locator(MultipleLocator(y1_interval))
ax1.grid(which='both', axis='y')
# ax1.grid(False)

ax2 = ax1.twinx()
ax2.plot(Coulomb_efficiency[1:], 'o-', linewidth = 2, color="red")
ax2.set_xlabel('Cycle')
ax2.set_xticks(np.arange(0,12,1), labels=np.arange(1,13,1))
ax2.set_ylabel('Coulomb efficiency(%)', color="red")
ax2.set_ylim([95.6,96.8])
# ax2.set_yticks(np.arange(95.6,96.8,0.02))
# y2_interval = (100 - 94) / 5
# ax2.yaxis.set_major_locator(MultipleLocator(y2_interval))
ax2.grid(which='both', axis='y')
# ax2.grid(False)
fig.tight_layout()

plt.figure()
plt.plot(bal_count_charge/np.array(128)*100, "o-", color="blue", label="Charge range")
plt.plot(Coulomb_efficiency, "o-", color="red", label="Coulomb_efficiency")
plt.ylabel("Coulomb_efficiency(%)")
plt.xlabel("Cycle")
# plt.ylim([95.6,96.8])
plt.xticks(np.arange(0,len(Coulomb_efficiency),1))
plt.tight_layout()


charge_bal_Ah = [np.mean(Bal_charge[i][Bal_charge[i]>100]) for i in range(len(Bal_charge))]
charge_bal_Ah = np.nan_to_num(charge_bal_Ah, nan=0)
plt.figure()
plt.plot(charge_bal_Ah*np.array(2)/3600, "o-")
plt.ylabel("Balancing Capacity in Charge range(Ah)")
plt.xlabel("Cycle")
plt.xticks(np.arange(0,len(bal_count_charge),1))
plt.tight_layout()

plt.figure()
plt.plot(Bal_rest[4]*np.array(2)/3600, "o")
plt.ylabel("Balancing Capacity in rest range(Ah)")
plt.xlabel("4th Cycle")
plt.xticks(np.arange(0,129,8))
plt.tight_layout()

plt.figure()
plt.plot((charge_bal_Ah*np.array(2)/3600)*(bal_count_charge/np.array(128)), "o-")
plt.ylabel("System Balancing Capacity by ratio(Ah)")
plt.xlabel("Cycle")
plt.xticks(np.arange(0,len(bal_count_charge),1))
plt.tight_layout()



Bal_module = {}
for num in np.arange(1,9,1):
    dummy = []
    for i in np.arange(1,len(Idx['SChg'])):
        dummy.append(np.sum(Stat_Module[num][Idx["EChg"][i-1]:Idx["EChg"][i]],axis=0))
    Bal_module[num] = dummy
len(Bal_module[7][0])

md = 1
cell = 3
cell_bal = []
for num in range(len(Bal_module[md])):
    cell_bal.append(Bal_module[md][num][cell - 1])
plt.figure()
plt.plot(cell_bal, "o-")
plt.ylabel("Balancing Time(s)")
plt.xlabel("Cycle")
plt.ylim(0,3000)
# plt.legend(loc='upper right', fontsize=8)
plt.tight_layout()



Total_Bal_Time = {}
for n in np.arange(0, module):
    Total_Bal_Time[n+1] = np.sum(Stat_Module[n+1], axis=0)

plt.figure()
plt.plot(Total_Bal_Time[1], 'o-')
plt.xlabel('Cell No.')
plt.ylabel('Balancing Time(s)')
plt.xticks(np.arange(0,16,1),labels=np.arange(1,17,1))
plt.tight_layout()


# 완충 후 휴지 전압이 낮은 셀(자가방전이 높은 셀) -> 용량이 높은 셀이 껴있을 수 있으니 점검 필요
No_Bal_CellNo = {}
for n in np.arange(0, 40):
    No_Bal_CellNo[n+1] = np.where(Total_Bal_Time[n+1] < 10)[0] + 1

# 이상 모듈 및 셀 표시
for key, val in No_Bal_CellNo.items():
    if len(val) != 0:
        print(key, ":", val)

plt.figure()
for key, val in No_Bal_CellNo.items():
    if len(val) != 0:
        plt.plot(CellV_Module[key][:, val-1])
        plt.text(x = 0, y = CellV_Module[key][0, val-1], s = str(CellV_Module[key][0, val-1] - CellV_Module[key][-1, val-1]))
        print(CellV_Module[key][0, val-1] - CellV_Module[key][-1, val-1])


# 완충 후 묘듈 내 셀 별 전압 확인
Module_no = 6
plt.figure(figsize=(12, 5))
for cyc in np.arange(0, len(Idx['EChg'])):
    if cyc == 0:
        dummy = CellV_Module[Module_no][Idx['EChg'][cyc], :]
    else:
        dummy = np.vstack((dummy, CellV_Module[Module_no][Idx['EChg'][cyc], :]))

plt.subplot(1, 3, 1)
plt.plot(dummy, 'o-')
plt.xlabel('Cycle')
plt.ylabel('Voltage(V)')
plt.tight_layout()

plt.subplot(1, 3, 2)
plt.plot( CellV_Module[Module_no])
plt.ylim([1.1, 1.6])
plt.xlabel('Time(h)')
plt.ylabel('Voltage(V)')
# plt.legend(loc='upper left', labels=cell_no)
plt.subplot(1, 3, 3)
plt.plot(Total_Bal_Time[Module_no], 'o-')
plt.xticks(ticks = np.arange(0, 16), labels=np.arange(1, 17))
plt.xlabel('Cell No.')
plt.ylabel('Total Balancing Time(s)')
plt.tight_layout()



sbc, sac, ebc, eac = {}, {}, {}, {}
sbd, sad, ebd, ead = {}, {}, {}, {}
ebct, ebdt = {}, {}
for i in np.arange(0,5):
    sbc[i] = df_bms[Idx["SChg"][i]-np.array(2),1:mod_Enum]
    sac[i] = df_bms[Idx["SChg"][i]+np.array(2),1:mod_Enum]
    ebc[i] = df_bms[Idx["EChg"][i]-np.array(2),1:mod_Enum]
    eac[i] = df_bms[Idx["EChg"][i]+np.array(2),1:mod_Enum]
    sbd[i] = df_bms[Idx["SDis"][i+1]-np.array(2),1:mod_Enum]
    sad[i] = df_bms[Idx["SDis"][i+1]+np.array(2),1:mod_Enum]
    ebd[i] = df_bms[Idx["EDis"][i+1]-np.array(2),1:mod_Enum]
    ead[i] = df_bms[Idx["EDis"][i+1]+np.array(2),1:mod_Enum]
    ebct[i] = df_bms[Idx["EChg"][i]+np.array(2),temp_Snum:temp_Enum]
    ebdt[i] = df_bms[Idx["EDis"][i+1]+np.array(2),temp_Snum:temp_Enum]

cl = {}
cl["sbc"] = [str(i+1)+"Cycle 충전 시작 전 전압" for i in np.arange(0,5)]
cl["sac"] = [str(i+1)+"Cycle 충전 시작 후 전압" for i in np.arange(0,5)]
cl["ebc"] = [str(i+1)+"Cycle 충전 종료 전 전압" for i in np.arange(0,5)]
cl["eac"] = [str(i+1)+"Cycle 충전 종료 후 전압" for i in np.arange(0,5)]
cl["sbd"] = [str(i+1)+"Cycle 방전 시작 전 전압" for i in np.arange(0,5)]
cl["sad"] = [str(i+1)+"Cycle 방전 시작 후 전압" for i in np.arange(0,5)]
cl["ebd"] = [str(i+1)+"Cycle 방전 종료 전 전압" for i in np.arange(0,5)]
cl["ead"] = [str(i+1)+"Cycle 방전 종료 후 전압" for i in np.arange(0,5)]
cl["ebct"] = [str(i+1)+"Cycle 충전 종료 전 온도" for i in np.arange(0,5)]
cl["ebdt"] = [str(i+1)+"Cycle 방전 종료 전 온도" for i in np.arange(0,5)]
len(list(cl.keys()))

array = [sbc, sac, ebc, eac, sbd, sad, ebd, ead, ebct, ebdt]
len(array)

dff = {}
for i in np.arange(0,len(array)):
    dff[i] = pd.DataFrame(array[i])
    dff[i].columns = cl[list(cl.keys())[i]]
    if i == 0:
        df_table = dff[i]
    else:
        df_table = pd.concat([df_table,dff[i]],axis=1)
df_table.index.name = "US.no"

file_name = "240429_R-R0_BLC-off"
df_table.to_excel("C:/Users/ByeongseongJeong(정병성/Downloads/"+file_name+".xlsx", index=True)



# QC Data Analysis
# set pickle file list
pk_path = r'C:\Users\ByeongseongJeong(정병성\OneDrive - Standard Energy\문서 - 품질인증파트 (Quality_Certification Part)\400_품질보증(QA)\500_ Raw Data\00_VCAL1020_data\00_VCAL1020_Monobloc_data\00_QC_data\240528_USQ025\pk_data'
pk_list = os.listdir(pk_path)
pk_list = [file for file in pk_list if file.endswith('.pickle')]
len(pk_list)


# Set dataframe from pickle files
df = {}
for i in range(len(pk_list)):
    with open(pk_path+'/'+pk_list[i], 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
    df[int(pk_list[i][2:6])] = data
len(df)


# Set DataFrame by US
key_df = list(df.keys())
for num in range(len(key_df)):
    if df[(key_df[num])][-1, 0] != 0:
        df[(key_df[num])][-1, 0] = 0
len(key_df)


M50 = [8938,8937,8819,8722,8864,8927,8990,8868,8926,8587,9123,8901,8572,9386,8932,8954,
        9172,9219,8362,9184,9476,8933,9211,8425,8360,9472,9220,9405,8968,8871,9389,8981,
        9051,8907,8978,8417,8895,9117,9002,8961,8983,8906,8918,8884,9022,8872,9108,8894,
        9151,9097,9212,9049,9119,8888,9228,9169,9230,9165,9214,9128,8946,8910,8973,8716,
        8842,8903,8971,8900,8658,8676,9121,8855,8904,8911,9054,9075,9094,8317,9129,8965,
        8898,8608,9189,9137,9070,8994,8319,9218,8849,8682,9039,8909,8999,9255,8958,8683,
        9191,9415,8588,9045,8972,9009,9197,8870,8397,9154,9031,9055,8788,8419,8891,9021,
        8991,8624,9046,8997,9379,8960,9382,8440,8964,8828,9221,9112,8992,9459,9217,9100]


key_df = (list(set(key_df) & set(M50)))
len(key_df)

key_df = sorted(key_df, key=lambda x: {value:index for index, value in enumerate(M50)}[x])

# Find Index where is start of charge, start of discharge, end of charge, end of discharge...
# 0 : Current / 1 : Voltage / 3 : Capacity / 4: Energy
Idx_C = {}
for num in range(len(key_df)):
    Current = df[(key_df[num])][:, 0]
    for i in range(len(Current)):
        if np.abs(Current[i]) < 1:
            Current[i] = 0
    dummy = Split_Cycle(Current)
    Idx_C[(key_df[num])] = dummy
    print(str(num + 1) + ' Finish..')


avg_vol_time = []
for num in range(len(key_df)):
    avg_vol_time.append(np.min(np.where((df[key_df[num]][Idx_C[key_df[num]]["SDis"][-1]:Idx_C[key_df[num]]["EDis"][-1],1]) <= 1.322)[0]))
int(np.mean(avg_vol_time))

np.mean(df_bms[Idx["SDis"][0],1:mod_Enum])

plt.figure()
plt.plot(df_bms[Idx["SDis"][0]+1:Idx["EDis"][0],1:mod_Enum], color="red")
for num in range(len(key_df)):
    plt.plot(df[key_df[num]][Idx_C[key_df[num]]["SDis"][-1]+5670:Idx_C[key_df[num]]["EDis"][-1],1],color="blue")
plt.xlabel("Time(s)")
plt.ylabel("Cell Voltage(V)")
plt.tight_layout()


qc_ir = []
for num in range(len(key_df)):
    qc_ir.append(np.abs(np.abs(df[key_df[num]][Idx_C[key_df[num]]["SDis"][-1]-1,1]-df[key_df[num]][Idx_C[key_df[num]]["SDis"][-1]+1,1])/(df[key_df[num]][Idx_C[key_df[num]]["SDis"][-1]+1,0])*1000))

plt.figure()
plt.plot(CIR[1], "o-",color="blue", label="1st Cycle")
plt.plot(CIR[-1], "o-",color="red", label="4th Cycle")
plt.xlabel("Cell no")
plt.ylabel("IR(mhm)")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()

plt.figure()
plt.plot(CIR[1], "o-",color="red", label="1st Cycle")
plt.plot(qc_ir, "o-",color="blue", label="QC Cycle")
plt.xlabel("Cell no")
plt.ylabel("IR(mhm)")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()

ir_rate = (CIR[1]-qc_ir)/qc_ir*100
plt.figure()
plt.plot(ir_rate, "o")
plt.xlabel("US.no")
plt.ylabel("IR change rate(%)")
plt.tight_layout()


def func(x):
    return x >= np.abs(-(df_bms[Idx['SChg'][0]+1, -2])/100) - 0.4


len(df_bms[Idx['SChg'][0]:Idx['EChg'][0], -2])
len(list(filter(func, -df_bms[Idx['SChg'][0]:Idx['EChg'][0], -2]/100)))

plt.figure()
plt.plot(-df_bms[Idx['SChg'][0]:Idx['EChg'][0], -2]/100)

plt.figure()
plt.plot(df_bms[Idx["SChg"][0]-1:Idx["EChg"][0],1:mod_Enum])
plt.axvline(7020, linestyle="--", color="red", linewidth=1.5)
plt.xlabel("Time(s)")
plt.ylabel("Cell Voltage(V)")
plt.tight_layout()








US = 7373
index_CS = np.min(np.where((df[US][Idx_C[US]["SDis"][-1]:Idx_C[US]["EDis"][-1], 1])<=1.45))
index_CE = np.min(np.where((df[US][Idx_C[US]["SDis"][-1]:Idx_C[US]["EDis"][-1], 1])<=1.23))

module_name = 6
cell_name = 3
index_MS = np.min(np.where((CellV_Module[module_name][Idx["SDis"][-1]:Idx["EDis"][-1],cell_name-1])<=1.45))
index_ME = np.min(np.where((CellV_Module[module_name][Idx["SDis"][-1]:Idx["EDis"][-1],cell_name-1])<=1.23))

plt.figure()
plt.plot((df[US][Idx_C[US]["SDis"][-1]+index_CS:Idx_C[US]["SDis"][-1]+index_CE, 1]),linewidth=1.5, label="QC - US"+str(US), color="red")
plt.plot((CellV_Module[module_name][Idx["SDis"][-1]+index_MS:Idx["SDis"][-1]+index_ME,cell_name-1]),lw=1.5,label="Module US"+str(US), color="blue")
plt.xlabel('Time(s)')
plt.ylabel('Cell Voltage(V)')
plt.legend(loc='upper right')
plt.tight_layout()


index_CS = np.min(np.where((df[US][Idx_C[US]["SChg"][-1]:Idx_C[US]["EChg"][-1], 1])>=1.29))
index_CE = np.min(np.where((df[US][Idx_C[US]["SChg"][-1]:Idx_C[US]["EChg"][-1], 1])>=1.50))

index_MS = np.min(np.where((CellV_Module[module_name][Idx["SChg"][-1]:Idx["EChg"][-1],cell_name-1])>=1.29))
index_ME = np.min(np.where((CellV_Module[module_name][Idx["SChg"][-1]:Idx["EChg"][-1],cell_name-1])>=1.50))

plt.figure()
plt.plot((df[US][Idx_C[US]["SChg"][-1]+index_CS:Idx_C[US]["SChg"][-1]+index_CE, 1]),linewidth=1.5, label="QC - US"+str(US), color="red")
plt.plot((CellV_Module[module_name][Idx["SChg"][-1]+index_MS:Idx["SChg"][-1]+index_ME,cell_name-1]),lw=1.5,label="Module US"+str(US), color="blue")
plt.xlabel('Time(s)')
plt.ylabel('Cell Voltage(V)')
plt.legend(loc='upper right')
plt.tight_layout()
