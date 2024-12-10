# Cell P/F Data Analysis
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import scipy.stats as stats
import seaborn as sns
warnings.simplefilter("ignore")
import scipy.stats as st
from scipy.stats import norm, gaussian_kde
from statsmodels.graphics.gofplots import qqplot, ProbPlot
import scipy.integrate as integrate


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


# 1. 가우시안 분포 데이터셋 생성
# 샘플 크기, 평균, 표준편차 설정
def generate_normal_data(sample_size, mean=0, variance=1):
    """
    평균(mean)과 표준편차(std_dev)를 기반으로 정규분포 데이터를 생성합니다.
    """
    std_dev = np.sqrt(variance)
    data = np.random.normal(loc=mean, scale=std_dev, size=sample_size)
    return data


# 2. 왜도와 첨도를 설정하는 함수
def adjust_skewness_kurtosis(data, target_skew, target_kurt):
    """
    데이터의 왜도(skewness)와 첨도(kurtosis)를 설정한 값으로 조정합니다.
    target_skew가 음수면 왼쪽 꼬리, 양수면 오른쪽 꼬리를 생성합니다.
    """
    # 1. 데이터의 양수화 (Box-Cox 변환을 위해)
    if np.min(data) <= 0:
        data = data - np.min(data) + 1  # Box-Cox 변환에 적합하도록 양수화

    # 2. Box-Cox 변환으로 초기 왜도 조정
    transformed_data, _ = stats.boxcox(data)

    # 3. 목표 왜도에 맞추기 위해 부호 및 스케일 조정
    current_skew = stats.skew(transformed_data)
    transformed_data = (transformed_data - np.mean(transformed_data)) / np.std(transformed_data)
    transformed_data = transformed_data * (abs(target_skew) / abs(current_skew))

    # 목표 왜도가 음수인 경우 분포를 반전
    if target_skew < 0:
        transformed_data = -transformed_data

    # 4. 목표 첨도에 맞추기 (Pearson 비대칭 분포 사용)
    pearson_dist = stats.pearson3(skew=target_skew)
    adjusted_data = pearson_dist.rvs(size=len(transformed_data))

    # 첨도를 목표 첨도로 맞추기
    adjusted_data = (adjusted_data - np.mean(adjusted_data)) / np.std(adjusted_data)  # 표준화
    adjusted_data = adjusted_data * (target_kurt / stats.kurtosis(adjusted_data, fisher=False))

    return adjusted_data


# 3. 다양한 정규성 검정 수행 함수
def perform_normality_tests(data):
    """
    주어진 데이터에 대해 다양한 정규성 검정을 수행합니다.
    """
    results = {}

    # Shapiro-Wilk Test
    shapiro_pval = stats.shapiro(data).pvalue
    results['Shapiro-Wilk'] = shapiro_pval

    # Kolmogorov-Smirnov Test
    ks_stat, ks_pval = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    results['Kolmogorov-Smirnov'] = ks_pval

    # Anderson-Darling Test
    anderson_stat = stats.anderson(data, dist='norm')
    results['Anderson-Darling'] = anderson_stat.significance_level[
        np.argmax(anderson_stat.statistic < anderson_stat.critical_values)] if np.any(
        anderson_stat.statistic < anderson_stat.critical_values) else 0

    # D’Agostino and Pearson’s Test
    dagostino_pval = stats.normaltest(data).pvalue
    results['D\'Agostino-Pearson'] = dagostino_pval

    return results


def adjust_skewness_kurtosis_v7(sample_size, target_skew, target_kurt, iterations=50):
    """
    목표 왜도(target_skew)와 첨도(target_kurt)를 가지는 데이터를 생성합니다.
    샘플 크기에 따라 조정 강도를 제한하며 안정적으로 목표값에 수렴합니다.
    """
    # 1. 초기 데이터 생성 (표준 정규분포)
    data = np.random.normal(0, 1, sample_size)

    for i in range(iterations):
        # 현재 데이터의 왜도와 첨도 계산
        current_skew = stats.skew(data)
        current_kurt = stats.kurtosis(data, fisher=False)

        # 2. 왜도 조정 (선형 조정)
        if abs(target_skew - current_skew) > 0.01:  # 작은 차이는 무시
            skew_adjustment = (target_skew - current_skew) / (10 + np.sqrt(sample_size))
            data = data + skew_adjustment * data  # 선형적인 조정

        # 3. 첨도 조정 (선형 조정)
        if abs(target_kurt - current_kurt) > 0.01:  # 작은 차이는 무시
            kurt_adjustment = (target_kurt - current_kurt) / (10 + np.sqrt(sample_size))
            data = data + kurt_adjustment * (data - np.mean(data)) ** 2

        # 4. 데이터 정규화
        data = (data - np.mean(data)) / np.std(data)

    return data


size = [30, 31, 300, 1000]
target_skew = 2.5  # 목표 왜도
target_kurt = 6.0  # 목표 첨도
normal_data, adjusted_data, sk_data  = {}, {}, {}
for sample in size:
    normal_data[sample] = (generate_normal_data(sample))
    sk_data[sample] = (adjust_skewness_kurtosis_v7(sample, target_skew, target_kurt))
    adjusted_data[sample] = adjust_skewness_kurtosis(normal_data[sample], target_skew, target_kurt)

for sample in size:
    # print(f"{sample} sample : Gaussian")
    # print(f"Skewness: {stats.skew(normal_data[sample]) :.2f}, Kurtosis: {stats.kurtosis(normal_data[sample], fisher=False) :.2f}")
    # print(f"p-value : {shapiro(normal_data[sample]).pvalue:.4f}")
    # print(f"lopsided {sample} sample")
    # print(f"Skewness: {stats.skew(adjusted_data[sample]) :.2f}, Kurtosis: {stats.kurtosis(adjusted_data[sample], fisher=False) :.2f}")
    # 모든 정규성 검정 수행
#     normality_results = perform_normality_tests(adjusted_data[sample])
#     for test_name, p_val in normality_results.items():
#         print(f"{test_name} p-value: {p_val:.4f}")
#     print(" ")
    print(f"evenly skewed {sample} sample")
    print(f"min : {round(np.min(sk_data[sample]),3)}, max : {round(np.max(sk_data[sample]),3)} ")
    print(f"Skewness: {stats.skew(sk_data[sample]) :.2f}, Kurtosis: {stats.kurtosis(sk_data[sample], fisher=False) :.2f}")
    # 모든 정규성 검정 수행
    normality_results = perform_normality_tests(sk_data[sample])
    for test_name, p_val in normality_results.items():
        print(f"{test_name} p-value: {p_val:.4f}")
    print(" ")


# 플롯 생성
t_l = ["A", "B", "C", "D"]
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12, 6))
fig.suptitle("Example of Comparison for Sample's characteristic")
for i, sample in enumerate([30, 30, 300, 1000]):
    ax = ax.flatten()
    # data = adjusted_data[sample]
    data = sk_data[sample]
    min_val = min(data)
    max_val = max(data)
    margin = (max_val - min_val) * 0.3  # Add 10% margin on both sides

    # 표본 크기에 따른 Bandwidth 및 bins 계산
    n_samples = len(data)  # 표본 크기
    std_dev = np.std(data)  # 표준편차
    bandwidth = 1.06 * std_dev * n_samples ** (-1 / 5)  # 최적의 Bandwidth 계산
    bins = max(10, int((max_val - min_val) / bandwidth))  # bins 개수 계산

    # Extend the range for x_vals
    x_vals = np.linspace(min_val - margin, max_val + margin, 1000)
    kde = gaussian_kde(data, bw_method=0.5)
    kde_vals = kde(x_vals)
    kde_peak = x_vals[np.argmax(kde_vals)]
    pdf = norm.pdf(x_vals, loc=data.mean(), scale=data.std())

    # KDE, PDF 각각의 확률밀도함수 그리기
    # 첫 번째 서브플롯: dummy_draw_0
    ax[i].hist(data, bins=bins, color='red', edgecolor='black', density=True, alpha=0.3)
    ax[i].plot(x_vals, kde(x_vals), color='red', lw=2, label="KDE")
    ax[i].axvline(kde_peak, color='red', linestyle='--', lw=1.5)
    ax[i].plot(x_vals, pdf, color='blue', lw=2, label="PDF")
    ax[i].axvline(data.mean(), color='blue', linestyle='--', lw=1.5)
    ax[i].set_title(f"Sample {t_l[i]}")
    ax[i].legend(loc="upper right", fontsize=10)
    # ax[0].set_xlabel("Standard Deviation")
    # ax[0].set_ylabel("Density")
# 레이아웃 조정
fig.supxlabel("Standard Deviation")
fig.supylabel("Density")
plt.tight_layout()






















''' Raw Data EDA '''
# Set path
eis_path = "C:/Users/jeongbs1/오토실리콘/1. python_code/ATS_Code/PF_CF_Test/Raw_data"
drt_path = eis_path+"./DRT_data"

# Load EIS Raw data
eis_raw_df = pd.read_csv(eis_path+"./03-02 DS03 RDF - PassFail Discrimination V0.0 240910.csv")
eis_raw_df["P/F"] = eis_raw_df["P/F"].replace({"P":1, "F":0})
eis_raw_df.drop(columns=["ITER"], inplace=True)

# Load DRT Raw data
drt_flist = os.listdir(drt_path)
drt_flist = [i for i in drt_flist if not any(exclude in i for exclude in ["NC01", "NC02", "NC03"])]
drt_list = []
plt.figure()
for file in range(len(drt_flist)):
    drt_data = pd.read_csv(drt_path + "/" + "".join(drt_flist[file]), skiprows=2).reset_index(drop=True)
    file_name = drt_flist[file].split("_")[0]
    plt.plot(drt_data["tau"], drt_data["gamma"], "o-")
    drt_1 = drt_data["gamma"].max()
    # tau >= 0.05에서 gamma의 최대값
    drt_2 = drt_data[drt_data["tau"] >= 0.05]["gamma"].max()
    drt_list.append([file_name, drt_1, drt_2])
drt_df = pd.DataFrame(drt_list, columns=["Cell Name", "DRT_1", "DRT_2"])
len(drt_df)


# Remove outlier NC01,NC02,NC03,
eis_raw_df = eis_raw_df.drop(index=eis_raw_df[eis_raw_df["Cell Name"].isin(["NC01", "NC02", "NC03"])].index, inplace=False).reset_index(drop=True)
Cell_name = eis_raw_df["Cell Name"].unique()
# eis_raw_df = eis_raw_df[eis_raw_df["Freq"].isin([1000.0, 10.0, 100.0, 1.0])].reset_index(drop=True)
arg_cols = ["Cell Name", "Freq", "Zre (mohm)", "Zim (mohm)", "Mag (mohm)", "Phase", "P/F", "OCV-B"]
eis_raw_df = eis_raw_df[arg_cols]

# 열 이름 변경
re_cols = ["Cell Name", "Freq", "Re_Z (mohm)", "Im_Z (mohm)", "Mag_Z (mohm)", "Phase_Z", "P/F", "OCV"]
eis_raw_df = eis_raw_df.rename(columns=dict(zip(arg_cols, re_cols)))


# Set P/F Cell DataFrame
pf_raw_df = {}
Freq_list = list(sorted(set(eis_raw_df["Freq"])))
for fre in Freq_list:
    pf_raw_df[f'F_{fre}'] = eis_raw_df[eis_raw_df["Freq"]==fre].reset_index(drop=True)
    pf_raw_df[f'F_{fre}'] = pd.merge(pf_raw_df[f'F_{fre}'], drt_df, on='Cell Name', how='inner')


# Set OCV, DRT DataFrame : independent Frequency
f_df_list = list(pf_raw_df.keys())
in_f_df = pf_raw_df[f_df_list[0]][["Cell Name", "P/F", "OCV", "DRT_1", "DRT_2"]]


# Check the nyquist plot
# Split by Cell Name
# Make split dataframe
data = {}
for i in range(len(Cell_name)):
    data[(Cell_name[i])] = (eis_raw_df.groupby("Cell Name").get_group(Cell_name[i])).reset_index()
    data[(Cell_name[i])]["Im_Z (mohm)"] = -(data[(Cell_name[i])]["Im_Z (mohm)"])
len(data)
data[(Cell_name[0])].columns


# Check Frequency by Cell Name
# Nyquist plot by Cell Name
plt.figure()
labels_added = {"Pass cell": False, "Fail cell": False}
for i in range(len(Cell_name)):
    x = list(data[(Cell_name[i])]["Re_Z (mohm)"])
    y = list(data[(Cell_name[i])]["Im_Z (mohm)"])
    plt.plot(x, y, "o-")
plt.xlabel("Re_Z (mOhm)")
plt.ylabel("-Im_Z (mOhm)")
plt.tight_layout()


plt.figure()
labels_added = {"Pass cell": False, "Fail cell": False}
for i in range(len(Cell_name)):
    x = list(data[(Cell_name[i])]["Re_Z (mohm)"])
    y = list(data[(Cell_name[i])]["Im_Z (mohm)"])
    if Cell_name[i].startswith("N"):
        if not labels_added["Pass cell"]:  # Pass cell 레이블이 추가되지 않았다면
            plt.plot(x, y, "o-", color="blue", label="Pass cell")
            labels_added["Pass cell"] = True
        else:
            plt.plot(x, y, "o-", color="blue")
    else:
        if not labels_added["Fail cell"]:  # Fail cell 레이블이 추가되지 않았다면
            plt.plot(x, y, "o-", color="red", label="Fail cell")
            labels_added["Fail cell"] = True
        else:
            plt.plot(x, y, "o-", color="red")
plt.xlabel("Re_Z (mOhm)")
plt.ylabel("-Im_Z (mOhm)")
plt.legend(loc="right", fontsize=10)
plt.tight_layout()


# Board plot by Cell Name
plt.figure()
for num, ft in enumerate(['Re_Z (mohm)', 'Im_Z (mohm)', 'Mag_Z (mohm)', 'Phase_Z']):
    plt.subplot(2,2,num+1)
    for i in range(len(Cell_name)):
        # print(data[(Cell_name[i])]["Freq"][np.abs(data[(Cell_name[i])]["Zre (mohm)"]) == np.abs(data[(Cell_name[i])]["Zre (mohm)"]).min()])
        x = list(data[(Cell_name[i])]["Freq"])
        y = list((data[(Cell_name[i])][ft]))
        plt.plot(x, y, "o-")
    plt.xlabel("Freq")
    plt.ylabel(ft)
plt.tight_layout()


plt.figure()
for num, ft in enumerate(['Re_Z (mohm)', 'Im_Z (mohm)', 'Mag_Z (mohm)', 'Phase_Z']):
    plt.subplot(2,2,num+1)
    for i in range(len(Cell_name)):
        # print(data[(Cell_name[i])]["Freq"][np.abs(data[(Cell_name[i])]["Zre (mohm)"]) == np.abs(data[(Cell_name[i])]["Zre (mohm)"]).min()])
        x = list(data[(Cell_name[i])]["Freq"])
        y = list((data[(Cell_name[i])][ft]))
        if Cell_name[i].startswith("N"):
            if not labels_added["Pass cell"]:  # Pass cell 레이블이 추가되지 않았다면
                plt.plot(x, y, "o-", color="blue", label="Pass cell")
                labels_added["Pass cell"] = True
            else:
                plt.plot(x, y, "o-", color="blue")
        else:
            if not labels_added["Fail cell"]:  # Fail cell 레이블이 추가되지 않았다면
                plt.plot(x, y, "o-", color="red", label="Fail cell")
                labels_added["Fail cell"] = True
            else:
                plt.plot(x, y, "o-", color="red")
    plt.xlabel("Freq")
    plt.ylabel(ft)
plt.tight_layout()



print(data[(Cell_name[30])]["Freq"][np.abs(data[(Cell_name[30])]["Im_Z (mohm)"])==np.abs(data[(Cell_name[30])]["Im_Z (mohm)"]).min()])

plt.figure()
plt.plot((data[(Cell_name[30])]["Freq"][:-1]), np.abs(np.diff(data[(Cell_name[30])]["Re_Z (mohm)"])), "o", label="Z_re")
plt.plot((data[(Cell_name[30])]["Freq"][:-1]), np.abs(np.diff(data[(Cell_name[30])]["Im_Z (mohm)"])), "o", label="Z_im")
plt.xlabel("Frequency")
plt.ylabel("Z differential(mOhm)")
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()

np.abs(np.diff(data[(Cell_name[30])]["Zim (mohm)"]))





# Make Transfer to Z_score DataFrame
# OCV, DRT
inf_cols = ["OCV", "DRT_1", "DRT_2"]
dummy_snd_inf = in_f_df[inf_cols]
dummy_snd_inf = ((dummy_snd_inf - dummy_snd_inf.mean()) / dummy_snd_inf.std())
org_inf_df = in_f_df[["Cell Name", "P/F"]]
snd_inf_df = pd.concat([org_inf_df, dummy_snd_inf], axis=1)

# EIS Features
eis_ft_cols = ["Re_Z (mohm)", "Im_Z (mohm)", "Mag_Z (mohm)", "Phase_Z"]
snd_df = {}
for f in f_df_list:
    dummy = pf_raw_df[f][eis_ft_cols]
    dummy = ((dummy - dummy.mean()) / dummy.std())
    org_dummy_df = pf_raw_df[f][[pf_raw_df[f].columns[0], pf_raw_df[f].columns[1], pf_raw_df[f].columns[6]]]
    snd_df[f] = pd.concat([org_dummy_df, dummy], axis=1)


# 표본 수에 따른 feature 별 P-value 변화
p_val = []
pval_df = (pf_raw_df[f_df_list[1]][pf_raw_df[f_df_list[1]]["P/F"] == 1])
ex_sam_s = [35, 30, 25, 20, 15]
p_val_cols = [eis_ft_cols[3]]+[inf_cols[0]]
for ss in ex_sam_s:
    random_sample = pval_df.sample(n=ss).reset_index(drop=True)
    for column in p_val_cols:
        stat, p_value = stats.shapiro(random_sample[column])
        skewness = stats.skew(random_sample[column])
        Kurtosis = stats.kurtosis(random_sample[column], fisher=False)
        p_val.append({'sample': ss, 'Feature': column, 'p-value': p_value, 'skewness': skewness, 'Kurtosis': Kurtosis})
p_val_result_df = pd.DataFrame(p_val)
p_val_result_df[p_val_result_df["Feature"]=="Mag (mohm)"].reset_index(drop=True)


plt.figure()
data = p_val_result_df[p_val_result_df["Feature"]=="Phase"].reset_index(drop=True)
plt.plot(data["sample"], data["p-value"], "o")
plt.title("Phase")
plt.xlabel("Sample size")
plt.ylabel("P-value of shapiro-wilk test")
plt.xticks(np.arange(15,40,5))
plt.tight_layout()

plt.figure()
data = p_val_result_df[p_val_result_df["Feature"]=="OCV-B"].reset_index(drop=True)
plt.plot(data["sample"], data["p-value"], "o")
plt.title("OCV")
plt.xlabel("Sample size")
plt.ylabel("P-value of shapiro-wilk test")
plt.xticks(np.arange(15,40,5))
plt.tight_layout()



plt.figure()
for i, ft in enumerate(p_val_cols):
    plt.subplot(1,2,i+1)
    data = p_val_result_df[p_val_result_df["Feature"]==ft].reset_index(drop=True)
    plt.plot(data["sample"], data["p-value"], "o")
    plt.title(ft)
    plt.xlabel("Sample size")
    plt.ylabel("P-value of shapiro-wilk test")
    plt.xticks(np.arange(15,40,5))
plt.tight_layout()

fig, ax = plt.subplots(1,2)
fig.suptitle('P-value by sample size')
for i, ft in enumerate(p_val_cols):
    data = p_val_result_df[p_val_result_df["Feature"] == ft].reset_index(drop=True)
    ax[i].plot(data["sample"], data["p-value"], "o")
    ax[i].set_title(ft)
    ax[i].set_xticks(np.arange(15,40,5))
    # ax[i].set_ylim([0.0,0.1])
fig.supylabel("P-value of shapiro-wilk test")
fig.supxlabel("Sample size")
plt.tight_layout()


plt.figure()
data = p_val_result_df[p_val_result_df["Feature"]=="Phase"].reset_index(drop=True)
plt.plot(data["sample"], data["p-value"], "o")
plt.title("Phase")
plt.xlabel("Sample size")
plt.ylabel("P-value of shapiro-wilk test")
plt.xticks(np.arange(15,40,5))
plt.tight_layout()




# Normality Test of OCV, DRT
norm_ind_df_p, norm_ind_df_f = [], []
shapiro_ind_df_p = in_f_df[in_f_df["P/F"] == 1]
shapiro_ind_df_f = in_f_df[in_f_df["P/F"] == 0]
for column in inf_cols:
    stat, p_value = stats.shapiro(shapiro_ind_df_p[column])
    skewness = shapiro_ind_df_p[column].skew()
    Kurtosis = shapiro_ind_df_p[column].kurtosis()
    # print(f, column, stat, p_value)
    norm_ind_df_p.append({'Feature': column, 'Shapiro': stat, 'p-value': p_value, 'skewness': skewness, 'Kurtosis': Kurtosis})

for column in inf_cols:
    stat, p_value = stats.shapiro(shapiro_ind_df_f[column])
    skewness = shapiro_ind_df_f[column].skew()
    Kurtosis = shapiro_ind_df_f[column].kurtosis()
    # print(f, column, stat, p_value)
    norm_ind_df_f.append({'Feature': column, 'Shapiro': stat, 'p-value': p_value, 'skewness': skewness, 'Kurtosis': Kurtosis})
shapiro_norm_ind_df_p = pd.DataFrame(norm_ind_df_p)
shapiro_norm_ind_df_f = pd.DataFrame(norm_ind_df_f)


# Normality Test of EIS Features
norm_eis_df_p, norm_eis_df_f = [], []
for f in f_df_list:
    shapiro_eis_df_p = pf_raw_df[f][pf_raw_df[f]["P/F"] == 1]
    for column in eis_ft_cols:
        stat, p_value = stats.shapiro(shapiro_eis_df_p[column])
        skewness = shapiro_eis_df_p[column].skew()
        Kurtosis = shapiro_eis_df_p[column].kurtosis()
        # print(f, column, stat, p_value)
        norm_eis_df_p.append({'Frequency': f, 'Feature': column, 'Shapiro': stat, 'p-value': p_value, 'skewness': skewness, 'Kurtosis': Kurtosis})

    shapiro_eis_df_f = pf_raw_df[f][pf_raw_df[f]["P/F"] == 0]
    for column in eis_ft_cols:
        stat, p_value = stats.shapiro(shapiro_eis_df_f[column])
        skewness = shapiro_eis_df_f[column].skew()
        Kurtosis = shapiro_eis_df_f[column].kurtosis()
        # print(f, column, stat, p_value)
        norm_eis_df_f.append({'Frequency': f, 'Feature': column, 'Shapiro': stat, 'p-value': p_value, 'skewness': skewness, 'Kurtosis': Kurtosis})
shapiro_norm_eis_df_p = pd.DataFrame(norm_eis_df_p)
shapiro_norm_eis_df_f = pd.DataFrame(norm_eis_df_f)
shapiro_norm_eis_df_p[shapiro_norm_eis_df_p["p-value"]<0.05]
shapiro_norm_eis_df_f[shapiro_norm_eis_df_f["p-value"]<0.05]


# Z Distribution of EIS-Features
for f in f_df_list[0:1]:
    sdf_draw_eis_df_0 = snd_df[f][snd_df[f]["P/F"] == 0]
    sdf_draw_eis_df_1 = snd_df[f][snd_df[f]["P/F"] == 1]

    sdf_draw_eis_df_0_numeric = snd_df[f].select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)
    sdf_draw_eis_df_1_numeric = snd_df[f].select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)

    min_val = min(sdf_draw_eis_df_0_numeric.min().min(), sdf_draw_eis_df_1_numeric.min().min())
    max_val = max(sdf_draw_eis_df_0_numeric.max().max(), sdf_draw_eis_df_1_numeric.max().max())
    x_vals = np.linspace(min_val, max_val, 1000)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    for i, ft in enumerate(eis_ft_cols):
        # dummy_draw_0 = snd_df[f][ft]
        dummy_draw_0 = snd_df[f][ft]
        # dummy_draw_0 = sdf_draw_eis_df_0[ft]
        # dummy_draw_1 = sdf_draw_eis_df_1[ft]
        row = int(i / 2)
        col = i % 2

        # KDE plot for P/F == 0
        # kde_0 = gaussian_kde(dummy_draw_0)
        # KDE plot for P/F == 1
#         kde_1 = gaussian_kde(dummy_draw_1)

        # Plot histogram and KDE for P/F == 0 (blue color)
        axs[row][col].hist(dummy_draw_0, bins=30, edgecolor='black', density=True, alpha=0.3)
        # axs[row][col].hist(dummy_draw_0, bins=10, color='red', edgecolor='black', density=True, alpha=0.3,label="F")
        # axs[row][col].plot(x_vals, kde_0(x_vals), color='red', lw=2, label="F-KDE")
        # Plot histogram and KDE for P/F == 1 (red color)
        # axs[row][col].hist(dummy_draw_1, bins=10, color='blue', edgecolor='black', density=True, alpha=0.3,label="P")
#         axs[row][col].plot(x_vals, kde_1(x_vals), color='blue', lw=2, label="F-KDE")

        # 겹치는 영역 채우기
        # overlap = np.minimum(kde_0(x_vals), kde_1(x_vals))  # 두 KDE의 최소값 계산
        # axs[row][col].fill_between(x_vals, overlap, color='green', alpha=0.4, label="Overlap Area")

        # 레이블 설정
        axs[row][col].set_title(ft)
        axs[row][col].set_xlabel("Standard Deviation")
        axs[row][col].set_ylabel('Density')
        # axs[row][col].legend(loc="upper right", fontsize=8)
    plt.tight_layout()

































# KDE, PDF가 표본 개수 별로 어떻게 달라지는지 확인
n0, n1 = 10, 30
np.random.seed(42)
sdf_draw_inf_df_0 = snd_inf_df["OCV"].reset_index(drop=True).sample(n=n0)
sdf_draw_inf_df_1 = snd_inf_df["OCV"].reset_index(drop=True).sample(n=n1)

sdf_draw_inf_df_0_numeric = sdf_draw_inf_df_0
sdf_draw_inf_df_1_numeric = sdf_draw_inf_df_1

min_val = min(sdf_draw_inf_df_0_numeric.min().min(), sdf_draw_inf_df_1_numeric.min().min())
max_val = max(sdf_draw_inf_df_0_numeric.max().max(), sdf_draw_inf_df_1_numeric.max().max())

# 플롯 생성
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
fig.suptitle('Example of KDE and PDF Function Comparison for OCV Sample')
# 표본 크기에 따른 Bandwidth 및 bins 계산
n_samples = len(data)  # 표본 크기
std_dev = np.std(data)  # 표준편차
bandwidth = 1.06 * 1 * 15 ** (-1 / 5)  # 최적의 Bandwidth 계산
bins = max(10, int((max_val - min_val) / bandwidth))  # bins 개수 계산
# KDE 및 PDF 계산
x_vals_0 = np.linspace(min_val, max_val, 1000)
kde_0 = gaussian_kde(sdf_draw_inf_df_0, bw_method=0.5)
kde_vals_0 = kde_0(x_vals_0)
kde_peak_0 = x_vals_0[np.argmax(kde_vals_0)]
pdf_0 = norm.pdf(x_vals_0, loc=sdf_draw_inf_df_0.mean(), scale=sdf_draw_inf_df_0.std())
x_vals_1 = np.linspace(min_val, max_val, 1000)
kde_1 = gaussian_kde(sdf_draw_inf_df_1, bw_method=0.5)
kde_vals_1 = kde_1(x_vals_1)
kde_peak_1 = x_vals_0[np.argmax(kde_vals_1)]
pdf_1 = norm.pdf(x_vals_1, loc=sdf_draw_inf_df_1.mean(), scale=sdf_draw_inf_df_1.std())
# 첫 번째 서브플롯: dummy_draw_0
ax[0].hist(sdf_draw_inf_df_0, bins=10, color='red', edgecolor='black', density=True, alpha=0.3, label=f"Sample A(n={n0})")
ax[0].plot(x_vals_0, kde_0(x_vals_0), color='red', lw=2, label=f"Sample A(n={n0})")
ax[0].hist(sdf_draw_inf_df_1, bins=10, color='blue', edgecolor='black', density=True, alpha=0.3, label=f"Sample B(n={n1})")
ax[0].plot(x_vals_1, kde_1(x_vals_1), color='blue', lw=2, label=f"Sample B(n={n1})")
ax[0].axvline(kde_peak_0, color='red', linestyle='--', lw=1.5)
ax[0].axvline(kde_peak_1, color='blue', linestyle='--', lw=1.5)
ax[0].set_title("KDE(Kernel Density Estimate) of OCV Sample")
ax[0].legend(loc="upper left", fontsize=10)
# 두 번째 서브플롯: dummy_draw_1
ax[1].hist(sdf_draw_inf_df_0, bins=10, color='red', edgecolor='black', density=True, alpha=0.3, label=f"Sample A(n={n0})")
ax[1].plot(x_vals_0, pdf_0, color='red', lw=2, label=f"Sample A(n={n0})")
ax[1].hist(sdf_draw_inf_df_1, bins=10, color='blue', edgecolor='black', density=True, alpha=0.3, label=f"Sample B(n={n1})")
ax[1].plot(x_vals_1, pdf_1, color='blue', lw=2, label=f"Sample B(n={n1})")
ax[1].axvline(sdf_draw_inf_df_0.mean(), color='red', linestyle='--', lw=1.5)
ax[1].axvline(sdf_draw_inf_df_1.mean(), color='blue', linestyle='--', lw=1.5)
ax[1].set_title("PDF(Probability Density Function) of OCV Sample")
ax[1].legend(loc="upper left", fontsize=10)
# 레이아웃 조정
fig.supxlabel("Standard Deviation")
fig.supylabel("Density")
plt.tight_layout()


# QQ-Plot
# QQ-Plot of OCV, DRT
sdf_draw_inf_df_0 = snd_inf_df[snd_inf_df["P/F"] == 0]
sdf_draw_inf_df_1 = snd_inf_df[snd_inf_df["P/F"] == 1]

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 8))
fig.suptitle("QQ Plot")
for i, ft in enumerate(inf_cols):
    dummy_draw_0 = sdf_draw_inf_df_0[ft]  # P/F == 0 데이터
    dummy_draw_1 = sdf_draw_inf_df_1[ft]  # P/F == 1 데이터
    ax = ax.flatten()
    # P/F == 0인 데이터에 대한 QQ 플롯 (빨간색)
    probplot_0 = ProbPlot(dummy_draw_0)
    mean_0 = dummy_draw_0.mean()
    std_0 = dummy_draw_0.std()
    ax[i].scatter(probplot_0.theoretical_quantiles, probplot_0.sample_quantiles, color='red', alpha=0.6,label="F")
    ax[i].plot(probplot_0.theoretical_quantiles, mean_0 + std_0 * probplot_0.theoretical_quantiles / probplot_0.theoretical_quantiles.std(),'r--')
    # P/F == 1인 데이터에 대한 QQ 플롯 (파란색)
    probplot_1 = ProbPlot(dummy_draw_1)
    mean_1 = dummy_draw_1.mean()
    std_1 = dummy_draw_1.std()
    ax[i].scatter(probplot_1.theoretical_quantiles, probplot_1.sample_quantiles, color='blue', alpha=0.6,label="P")
    ax[i].plot(probplot_1.theoretical_quantiles, mean_1 + std_1 * probplot_1.theoretical_quantiles / probplot_1.theoretical_quantiles.std(),'b--')
    # 레이블 및 범례 설정
    ax[i].set_title(ft)
    ax[i].set_xlabel("Standard Deviation")
    ax[i].set_ylabel('Sample Quantiles')
    ax[i].legend()
plt.tight_layout()


# QQ-Plot of EIS Features
for f in f_df_list:
    sdf_draw_eis_df_0 = snd_df[f][snd_df[f]["P/F"] == 0]  # P/F == 0
    sdf_draw_eis_df_1 = snd_df[f][snd_df[f]["P/F"] == 1]  # P/F == 1
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    fig.suptitle('Frequency Point at ' + str(f[2:]) + "Hz")
    for i, ft in enumerate(eis_ft_cols):
        dummy_draw_0 = sdf_draw_eis_df_0[ft]  # P/F == 0 데이터
        dummy_draw_1 = sdf_draw_eis_df_1[ft]  # P/F == 1 데이터
        row = int(i / 2)
        col = i % 2
        # P/F == 0인 데이터에 대한 QQ 플롯 (빨간색)
        probplot_0 = ProbPlot(dummy_draw_0)
        mean_0 = dummy_draw_0.mean()
        std_0 = dummy_draw_0.std()
        axs[row][col].scatter(probplot_0.theoretical_quantiles, probplot_0.sample_quantiles, color='red', alpha=0.6,label="F")
        axs[row][col].plot(probplot_0.theoretical_quantiles, mean_0 + std_0 * probplot_0.theoretical_quantiles / probplot_0.theoretical_quantiles.std(),'r--')
        # P/F == 1인 데이터에 대한 QQ 플롯 (파란색)
        probplot_1 = ProbPlot(dummy_draw_1)
        mean_1 = dummy_draw_1.mean()
        std_1 = dummy_draw_1.std()
        axs[row][col].scatter(probplot_1.theoretical_quantiles, probplot_1.sample_quantiles, color='blue', alpha=0.6,label="P")
        axs[row][col].plot(probplot_1.theoretical_quantiles, mean_1 + std_1 * probplot_1.theoretical_quantiles / probplot_1.theoretical_quantiles.std(),'b--')
        # 레이블 및 범례 설정
        axs[row][col].set_title(ft)
        axs[row][col].set_xlabel("Standard Deviation")
        axs[row][col].set_ylabel('Sample Quantiles')
        axs[row][col].legend()
    plt.tight_layout()


# Z Distribution of OCV, DRT
sdf_draw_inf_df_0 = snd_inf_df[snd_inf_df["P/F"] == 0].reset_index(drop=True)
sdf_draw_inf_df_1 = snd_inf_df[snd_inf_df["P/F"] == 1].reset_index(drop=True)

sdf_draw_inf_df_0_numeric = sdf_draw_inf_df_0.select_dtypes(include=[np.number]).drop(["P/F"], axis=1)
sdf_draw_inf_df_1_numeric = sdf_draw_inf_df_1.select_dtypes(include=[np.number]).drop(["P/F"], axis=1)

min_val = min(sdf_draw_inf_df_0_numeric.min().min(), sdf_draw_inf_df_1_numeric.min().min())
max_val = max(sdf_draw_inf_df_0_numeric.max().max(), sdf_draw_inf_df_1_numeric.max().max())
x_vals = np.linspace(min_val, max_val, 1000)

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(12, 6))
fig.suptitle('Z distribution')
for i, ft in enumerate(inf_cols):
    dummy_draw_0 = sdf_draw_inf_df_0[ft]
    dummy_draw_1 = sdf_draw_inf_df_1[ft]
    ax = ax.flatten()
    row = int(i / 2)
    col = i % 2
    # KDE plot for P/F == 0
    kde_0 = gaussian_kde(dummy_draw_0)
    pdf_0 = norm.pdf(x_vals, loc=dummy_draw_0.mean(), scale=dummy_draw_0.std())
    # KDE plot for P/F == 1
    kde_1 = gaussian_kde(dummy_draw_1)
    pdf_1 = norm.pdf(x_vals, loc=dummy_draw_1.mean(), scale=dummy_draw_1.std())
    # Plot histogram and KDE for P/F == 0 (blue color)
    ax[i].hist(dummy_draw_0, bins=10, color='red', edgecolor='black', density=True, alpha=0.3,label="F")
    ax[i].plot(x_vals, kde_0(x_vals), color='red', lw=2, label="F-KDE")
    ax[i].plot(x_vals, pdf_0, color='red', lw=2, linestyle="--", label="F-PDF")
    # Plot histogram and KDE for P/F == 1 (red color)
    ax[i].hist(dummy_draw_1, bins=10, color='blue', edgecolor='black', density=True, alpha=0.3,label="P")
    ax[i].plot(x_vals, kde_1(x_vals), color='blue', lw=2, label="P-KDE")
    ax[i].plot(x_vals, pdf_1, color='blue', lw=2, linestyle="--", label="P-PDF")
    # 레이블 설정
    ax[i].set_title(ft)
    ax[i].set_xlabel("Standard Deviation")
    ax[i].set_ylabel('Density')
    ax[i].legend()
plt.tight_layout()


# Z Distribution of EIS-Features
for f in f_df_list[15:16]:
    sdf_draw_eis_df_0 = snd_df[f][snd_df[f]["P/F"] == 0]
    sdf_draw_eis_df_1 = snd_df[f][snd_df[f]["P/F"] == 1]

    sdf_draw_eis_df_0_numeric = sdf_draw_eis_df_0.select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)
    sdf_draw_eis_df_1_numeric = sdf_draw_eis_df_1.select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)

    min_val = min(sdf_draw_eis_df_0_numeric.min().min(), sdf_draw_eis_df_1_numeric.min().min())
    max_val = max(sdf_draw_eis_df_0_numeric.max().max(), sdf_draw_eis_df_1_numeric.max().max())
    x_vals = np.linspace(min_val, max_val, 1000)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    fig.suptitle('Z distribution Frequency Point at ' + str(f[2:]) + "Hz")
    for i, ft in enumerate(eis_ft_cols):
        dummy_draw_0 = sdf_draw_eis_df_0[ft]
        dummy_draw_1 = sdf_draw_eis_df_1[ft]
        row = int(i / 2)
        col = i % 2

        # KDE plot for P/F == 0
        kde_0 = gaussian_kde(dummy_draw_0)
        # KDE plot for P/F == 1
        kde_1 = gaussian_kde(dummy_draw_1)

        # Plot histogram and KDE for P/F == 0 (blue color)
        axs[row][col].hist(dummy_draw_0, bins=10, color='red', edgecolor='black', density=True, alpha=0.3,label="F")
        axs[row][col].plot(x_vals, kde_0(x_vals), color='red', lw=2, label="F-KDE")
        # Plot histogram and KDE for P/F == 1 (red color)
        axs[row][col].hist(dummy_draw_1, bins=10, color='blue', edgecolor='black', density=True, alpha=0.3,label="P")
        axs[row][col].plot(x_vals, kde_1(x_vals), color='blue', lw=2, label="F-KDE")

        # 겹치는 영역 채우기
        overlap = np.minimum(kde_0(x_vals), kde_1(x_vals))  # 두 KDE의 최소값 계산
        axs[row][col].fill_between(x_vals, overlap, color='green', alpha=0.4, label="Overlap Area")

        # 레이블 설정
        axs[row][col].set_title(ft)
        axs[row][col].set_xlabel("Standard Deviation")
        axs[row][col].set_ylabel('Density')
        axs[row][col].legend(loc="upper right", fontsize=8)
    plt.tight_layout()


# KDE, PDF별 밀도함수 비교
sdf_draw_inf_df_0 = snd_inf_df[snd_inf_df["P/F"] == 0]
sdf_draw_inf_df_1 = snd_inf_df[snd_inf_df["P/F"] == 1]
sdf_draw_eis_df_0 = snd_df["F_1.0"][snd_df["F_1.0"]["P/F"] == 0]
sdf_draw_eis_df_1 = snd_df["F_1.0"][snd_df["F_1.0"]["P/F"] == 1]

sdf_draw_inf_df_0_numeric = sdf_draw_inf_df_0.select_dtypes(include=[np.number]).drop(["P/F"], axis=1)
sdf_draw_inf_df_1_numeric = sdf_draw_inf_df_1.select_dtypes(include=[np.number]).drop(["P/F"], axis=1)
sdf_draw_eis_df_0_numeric = sdf_draw_eis_df_0.select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)
sdf_draw_eis_df_1_numeric = sdf_draw_eis_df_1.select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)

inf_min_val = min(sdf_draw_inf_df_0_numeric.min().min(), sdf_draw_inf_df_1_numeric.min().min())
inf_max_val = max(sdf_draw_inf_df_0_numeric.max().max(), sdf_draw_inf_df_1_numeric.max().max())
eis_min_val = min(sdf_draw_eis_df_0_numeric.min().min(), sdf_draw_eis_df_1_numeric.min().min())
eis_max_val = max(sdf_draw_eis_df_0_numeric.max().max(), sdf_draw_eis_df_1_numeric.max().max())

# 서브플롯 생성
fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(12, 8))
axs = axs.flatten()
total_cols = inf_cols + eis_ft_cols
for i, ft in enumerate(total_cols):
    row = int(i / 2)
    col = i % 2
    if ft in total_cols[:3]:  # inf_cols 처리
        dummy_draw_0 = sdf_draw_inf_df_0[ft]
        dummy_draw_1 = sdf_draw_inf_df_1[ft]
        x_vals_0 = np.linspace(inf_min_val, inf_max_val, 1000)
        x_vals_1 = np.linspace(inf_min_val, inf_max_val, 1000)
        axs[i].set_title(ft)
    else:  # eis_ft_cols 처리
        dummy_draw_0 = sdf_draw_eis_df_0[ft]
        dummy_draw_1 = sdf_draw_eis_df_1[ft]
        x_vals_0 = np.linspace(eis_min_val, eis_max_val, 1000)
        x_vals_1 = np.linspace(eis_min_val, eis_max_val, 1000)
        axs[i].set_title(ft+" at 1Hz")
    kde_0 = gaussian_kde(dummy_draw_0)
    kde_1 = gaussian_kde(dummy_draw_1)
    pdf_0 = norm.pdf(x_vals_0, loc=dummy_draw_0.mean(), scale=dummy_draw_0.std())
    pdf_1 = norm.pdf(x_vals_1, loc=dummy_draw_1.mean(), scale=dummy_draw_1.std())
    # 히스토그램 및 KDE/PDF 플롯
    axs[i].hist(dummy_draw_0, bins=10, color='red', edgecolor='black', density=True, alpha=0.2, label="F")
    axs[i].plot(x_vals_0, kde_0(x_vals_0), color='red', lw=2, label="F-KDE")
    axs[i].plot(x_vals_0, pdf_0, color='red', lw=1.5, linestyle="--", label="F-PDF")
    axs[i].hist(dummy_draw_1, bins=10, color='blue', edgecolor='black', density=True, alpha=0.2, label="P")
    axs[i].plot(x_vals_1, kde_1(x_vals_1), color='blue', lw=2, label="P-KDE")
    axs[i].plot(x_vals_1, pdf_1, color='blue', lw=1.5, linestyle="--", label="P-PDF")
    # 서브플롯 설정
    # axs[i].set_xlabel("Standard Deviation")
    # axs[i].set_ylabel('Density')
    axs[i].legend().remove()  # 각 서브플롯의 범례 제거
fig.suptitle("PDF, KDE, Histogram by Features for P/F dataset")
fig.supxlabel("Standard Deviation")
fig.supylabel("Density")
plt.tight_layout()
# 별도의 창에 범례 생성
fig_legend, ax_legend = plt.subplots(figsize=(2, 3))  # 세로로 표현할 크기 조정
ax_legend.axis("off")  # 축 숨김
handles, labels = axs[0].get_legend_handles_labels()  # 첫 번째 플롯에서 범례 정보 가져오기
ax_legend.legend(handles, labels, loc="center", fontsize=10, ncol=1, title="Legend")





# Calculate to PUR
# Calculate to PUR of OCV, DRT
intersection_dict = {ft: [] for ft in inf_cols}
left_dict = {ft: [] for ft in inf_cols}
right_dict = {ft: [] for ft in inf_cols}

# KDE 함수를 저장할 딕셔너리 생성
kde_functions_0 = {ft: [] for ft in inf_cols}
kde_functions_1 = {ft: [] for ft in inf_cols}

kde_dataframes_0 = {}
kde_dataframes_1 = {}

sdf_draw_df_0 = snd_inf_df[snd_inf_df["P/F"] == 0]
sdf_draw_df_1 = snd_inf_df[snd_inf_df["P/F"] == 1]

# 각 피처에 대해 KDE 생성 및 적분 계산
for i, ft in enumerate(inf_cols):
    dummy_draw_0 = sdf_draw_df_0[ft]
    dummy_draw_1 = sdf_draw_df_1[ft]

    # P/F 별 연속확률밀도함수 추정
    kde_0 = gaussian_kde(dummy_draw_0)
    kde_1 = gaussian_kde(dummy_draw_1)
    kde_functions_0[ft].append(kde_0)  # 피처별로 kde_0 함수 저장
    kde_functions_1[ft].append(kde_1)

    # 교체 영역에 대해 적분 구간 분할
    min_val = min(dummy_draw_0.min(), dummy_draw_1.min())
    max_val = max(dummy_draw_0.max(), dummy_draw_1.max())
    x_vals = np.linspace(min_val, max_val, 1000)

    # 연속확률밀도 값 추출
    kde_0_vals = kde_0(x_vals)
    kde_1_vals = kde_1(x_vals)

    # KDE 값을 DataFrame으로 저장
    kde_df_0 = pd.DataFrame({'X': x_vals, f'KDE_{ft}_0': kde_0_vals})
    kde_df_1 = pd.DataFrame({'X': x_vals, f'KDE_{ft}_1': kde_1_vals})
    kde_dataframes_0[ft] = kde_df_0
    kde_dataframes_1[ft] = kde_df_1

    # 교차 지점 찾기
    difference = kde_0_vals - kde_1_vals
    intersection_indices = np.where(np.diff(np.sign(difference)))[0]

    if len(intersection_indices) > 0:
        intersection_index = intersection_indices[0]

        # 교차 지점 이전 (왼쪽) 영역 적분
        x_vals_left = x_vals[:intersection_index + 1]
        min_kde_vals_left = np.minimum(kde_0_vals[:intersection_index + 1], kde_1_vals[:intersection_index + 1])
        left_area = integrate.simpson(min_kde_vals_left, x=x_vals_left)
        left_dict[ft].append(left_area)

        # 교차 지점 이후 (오른쪽) 영역 적분
        x_vals_right = x_vals[intersection_index:]
        min_kde_vals_right = np.minimum(kde_0_vals[intersection_index:], kde_1_vals[intersection_index:])
        right_area = integrate.simpson(min_kde_vals_right, x=x_vals_right)
        right_dict[ft].append(right_area)

        # 전체 교차 영역 적분
        intersection_area = left_area + right_area
        intersection_dict[ft].append(intersection_area)
    else:
        # 교차 지점이 없는 경우, 전체 영역을 적분
        intersection_area = integrate.simpson(np.minimum(kde_0_vals, kde_1_vals), x=x_vals)
        intersection_dict[ft].append(intersection_area)
        left_dict[ft].append(0)  # 교차 지점이 없으므로 왼쪽, 오른쪽이 없음
        right_dict[ft].append(0)

# 기존의 intersection_df를 생성하는 코드
intersection_ind_kde_df = pd.DataFrame.from_dict(intersection_dict, orient='index').transpose()

# left_dict와 right_dict를 데이터프레임으로 변환
left_df = pd.DataFrame.from_dict(left_dict, orient='index').transpose()
right_df = pd.DataFrame.from_dict(right_dict, orient='index').transpose()

# intersection_ind_kde_df에 왼쪽(left)과 오른쪽(right) 적분 값을 각각 추가
intersection_ind_kde_df = pd.concat([intersection_ind_kde_df, left_df.add_prefix("Left_"), right_df.add_prefix("Right_")], axis=1)
intersection_ind_kde_df*np.array(1.0e6)


# Calculate to PUR of EIS Features
# 결과 저장용 리스트 초기화
pur_M, pur_F, pur_Re, pur_Im = [], [], [], []
intersection_dict = {ft: [] for ft in eis_ft_cols}
left_dict = {ft: [] for ft in eis_ft_cols}
right_dict = {ft: [] for ft in eis_ft_cols}

# KDE 함수를 저장할 딕셔너리 생성
kde_functions_0 = {ft: [] for ft in eis_ft_cols}
kde_functions_1 = {ft: [] for ft in eis_ft_cols}
kde_dataframes_0 = {}
kde_dataframes_1 = {}

for f in f_df_list:
    sdf_draw_df_0 = snd_df[f][snd_df[f]["P/F"] == 0]  # P/F == 0
    sdf_draw_df_1 = snd_df[f][snd_df[f]["P/F"] == 1]  # P/F == 1

    # 각 피처에 대해 KDE 생성 및 적분 계산
    for i, ft in enumerate(eis_ft_cols):
        dummy_draw_0 = sdf_draw_df_0[ft]
        dummy_draw_1 = sdf_draw_df_1[ft]

        # P/F 별 연속확률밀도함수 추정
        kde_0 = gaussian_kde(dummy_draw_0)
        kde_1 = gaussian_kde(dummy_draw_1)
        kde_functions_0[ft].append(kde_0)  # 피처별로 kde_0 함수 저장
        kde_functions_1[ft].append(kde_1)

        # 교체 영역에 대해 적분 구간 분할
        min_val = min(dummy_draw_0.min(), dummy_draw_1.min())
        max_val = max(dummy_draw_0.max(), dummy_draw_1.max())
        x_vals = np.linspace(min_val, max_val, 1000)

        # 연속확률밀도 값 추출
        kde_0_vals = kde_0(x_vals)
        kde_1_vals = kde_1(x_vals)

        # KDE 값을 DataFrame으로 저장
        kde_df_0 = pd.DataFrame({'X': x_vals, f'KDE_{ft}_0': kde_0_vals})
        kde_df_1 = pd.DataFrame({'X': x_vals, f'KDE_{ft}_1': kde_1_vals})
        kde_dataframes_0[ft] = kde_df_0
        kde_dataframes_1[ft] = kde_df_1

        # 교차 지점 찾기
        difference = kde_0_vals - kde_1_vals
        intersection_indices = np.where(np.diff(np.sign(difference)))[0]

        if len(intersection_indices) > 0:
            intersection_index = intersection_indices[0]

            # 교차 지점 이전 (왼쪽) 영역 적분
            x_vals_left = x_vals[:intersection_index + 1]
            min_kde_vals_left = np.minimum(kde_0_vals[:intersection_index + 1], kde_1_vals[:intersection_index + 1])
            left_area = integrate.simpson(min_kde_vals_left, x=x_vals_left)
            left_dict[ft].append(left_area)

            # 교차 지점 이후 (오른쪽) 영역 적분
            x_vals_right = x_vals[intersection_index:]
            min_kde_vals_right = np.minimum(kde_0_vals[intersection_index:], kde_1_vals[intersection_index:])
            right_area = integrate.simpson(min_kde_vals_right, x=x_vals_right)
            right_dict[ft].append(right_area)

            # 전체 교차 영역 적분
            intersection_area = left_area + right_area
            intersection_dict[ft].append(intersection_area)
        else:
            # 교차 지점이 없는 경우, 전체 영역을 적분
            intersection_area = integrate.simpson(np.minimum(kde_0_vals, kde_1_vals), x=x_vals)
            intersection_dict[ft].append(intersection_area)
            left_dict[ft].append(0)  # 교차 지점이 없으므로 왼쪽, 오른쪽이 없음
            right_dict[ft].append(0)

# 기존의 intersection_df를 생성하는 코드
intersection_eis_kde_df = pd.DataFrame.from_dict(intersection_dict, orient='index').transpose()
intersection_eis_kde_df = pd.concat([intersection_eis_kde_df, pd.DataFrame(Freq_list, columns=["Frequency"])], axis=1)

# left_dict와 right_dict를 데이터프레임으로 변환
left_df = pd.DataFrame.from_dict(left_dict, orient='index').transpose()
right_df = pd.DataFrame.from_dict(right_dict, orient='index').transpose()

# intersection_eis_kde_df에 왼쪽(left)과 오른쪽(right) 적분 값을 각각 추가
intersection_eis_kde_df = pd.concat([intersection_eis_kde_df, left_df.add_prefix("Left_"), right_df.add_prefix("Right_")], axis=1)
# intersection_eis_kde_df.to_excel("C:/Users/jeongbs1/Downloads/PUR_Calculation.xlsx")

inter_df_col = ['Frequency',
                'Mag_Z (mohm)', 'Left_Mag_Z (mohm)', 'Right_Mag_Z (mohm)',
                'Phase_Z', 'Left_Phase_Z', 'Right_Phase_Z',
                'Re_Z (mohm)', 'Left_Re_Z (mohm)', 'Right_Re_Z (mohm)',
                'Im_Z (mohm)', 'Left_Im_Z (mohm)', 'Right_Im_Z (mohm)',]

intersection_eis_kde_df = intersection_eis_kde_df[inter_df_col]


fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
fig.suptitle("Frequency range : 1, 5, 10Hz")
label = ["PUR", "P(P|F)", "P(F|P)"]
features_per_plot = 3
num_plots = len(inter_df_col[1:]) // features_per_plot
for i in range(num_plots):
    row = int(i / 2)
    col = i % 2
    start_idx = i * features_per_plot
    end_idx = start_idx + features_per_plot

    for l, ft in enumerate(inter_df_col[start_idx + 1:end_idx + 1]):
        axs[row][col].plot(list(intersection_eis_kde_df.index[:]), intersection_eis_kde_df[ft][:] * np.array(1.0e6), "o-",label=label[l])

    axs[row][col].set_title(inter_df_col[end_idx-2])
    axs[row][col].set_xlabel("Frequency(Hz)")
    axs[row][col].set_ylabel("PUR(ppm)")
    axs[row][col].set_xticks(ticks=[list(intersection_eis_kde_df.index)[i] for i in [0, 10, 20, 30]], labels=[1, 10, 100, 1000])
    # axs[row][col].set_yticks(np.arange(0,100,10))
    axs[row][col].legend(loc="upper left", fontsize=10)  # 각 피처에 대한 범례 추가
plt.tight_layout()


fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
fig.suptitle("PUR based on EIS Features")
label = ["PUR", "P(P|F)", "P(F|P)"]
features_per_plot = 3
num_plots = len(inter_df_col[1:]) // features_per_plot
for i in range(num_plots):
    row = int(i / 2)
    col = i % 2
    start_idx = i * features_per_plot
    end_idx = start_idx + features_per_plot
    # for l, ft in enumerate(inter_df_col[start_idx + 1:end_idx + 1]):
    for l, ft in enumerate(inter_df_col[start_idx + 1:start_idx + 2]):
        axs[row][col].plot(
            list(intersection_eis_kde_df.index[:]),
            intersection_eis_kde_df[ft][:] * np.array(1.0e6),"o-",label=label[l],)
    axs[row][col].set_title(inter_df_col[end_idx - 2])
    axs[row][col].set_xlabel("Frequency(Hz)")
    axs[row][col].set_ylabel("PUR(ppm)")
    axs[row][col].set_xticks(ticks=[list(intersection_eis_kde_df.index)[i] for i in [0, 10, 20, 30]], labels=[1, 10, 100, 1000])
    # Y축에만 로그 스케일 적용
    axs[row][col].set_yscale("log")
    axs[row][col].set_ylim([1.0e3, 1.0e6])
    axs[row][col].legend(loc="upper left", fontsize=10)
plt.tight_layout()


# 하나의 plot으로 그리기
fig, ax = plt.subplots()
fig.suptitle("PUR for EIS Feature using KDE function")
label = ["PUR", "P(P|F)", "P(F|P)"]
features_per_plot = 3
num_plots = len(inter_df_col[1:]) // features_per_plot
for i in range(num_plots):
    row = int(i / 2)
    col = i % 2
    start_idx = i * features_per_plot
    end_idx = start_idx + features_per_plot
    # for l, ft in enumerate(inter_df_col[start_idx + 1:end_idx + 1]):
    for l, ft in enumerate(inter_df_col[start_idx + 1:start_idx + 2]):
        ax.plot(
            list(intersection_eis_kde_df.index[:]),
            intersection_eis_kde_df[ft][:] * np.array(1.0e6),"o-",label=ft)
    # ax.set_title(inter_df_col[end_idx - 2])
    ax.set_xlabel("Frequency(Hz)")
    ax.set_ylabel("PUR(ppm)")
    ax.set_xticks(ticks=[list(intersection_eis_kde_df.index)[i] for i in [0, 10, 20, 30]], labels=[1, 10, 100, 1000])
    # Y축에만 로그 스케일 적용
    ax.set_yscale("log")
    ax.set_ylim([1.0e3, 1.0e6])
    ax.legend(loc="upper left", fontsize=10)
plt.tight_layout()






##################################################################################################
'''PDF based on Z-score'''
# PDF 기반 Z-분포 시각화
sdf_draw_inf_df_0 = snd_inf_df[snd_inf_df["P/F"] == 0]
sdf_draw_inf_df_1 = snd_inf_df[snd_inf_df["P/F"] == 1]

sdf_draw_inf_df_0_numeric = sdf_draw_inf_df_0.select_dtypes(include=[np.number]).drop(["P/F"], axis=1)
sdf_draw_inf_df_1_numeric = sdf_draw_inf_df_1.select_dtypes(include=[np.number]).drop(["P/F"], axis=1)

min_val = min(sdf_draw_inf_df_0_numeric.min().min(), sdf_draw_inf_df_1_numeric.min().min())
max_val = max(sdf_draw_inf_df_0_numeric.max().max(), sdf_draw_inf_df_1_numeric.max().max())

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
fig.suptitle('Z distribution')
# 각 Feature에 대해 처리
for i, ft in enumerate(inf_cols):
    dummy_draw_0 = sdf_draw_inf_df_0[ft]
    dummy_draw_1 = sdf_draw_inf_df_1[ft]
    # 플롯 배열 평탄화
    ax = ax.flatten()
    # X축 범위 설정
    min_val = min(dummy_draw_0.min(), dummy_draw_1.min())
    max_val = max(dummy_draw_0.max(), dummy_draw_1.max())
    x_vals = np.linspace(min_val, max_val, 1000)
    # PDF 계산
    pdf_0 = norm.pdf(x_vals, loc=dummy_draw_0.mean(), scale=dummy_draw_0.std())
    pdf_1 = norm.pdf(x_vals, loc=dummy_draw_1.mean(), scale=dummy_draw_1.std())
    # 히스토그램과 PDF 플롯
    ax[i].hist(dummy_draw_0, bins=10, color='red', edgecolor='black', density=True, alpha=0.3, label="F")
    ax[i].plot(x_vals, pdf_0, color='red', lw=2, label="F-PDF")
    ax[i].hist(dummy_draw_1, bins=10, color='blue', edgecolor='black', density=True, alpha=0.3, label="P")
    ax[i].plot(x_vals, pdf_1, color='blue', lw=2, label="P-PDF")
    # 레이블 설정
    ax[i].set_title(ft)
    ax[i].set_xlabel("Standard Deviation")
    ax[i].set_ylabel('Density')
    ax[i].legend()
# 레이아웃 설정
plt.tight_layout()



# Z Distribution of EIS-Features
for f in f_df_list:
    # P/F == 0과 P/F == 1로 데이터 분리
    sdf_draw_eis_df_0 = snd_df[f][snd_df[f]["P/F"] == 0]
    sdf_draw_eis_df_1 = snd_df[f][snd_df[f]["P/F"] == 1]

    # 숫자형 데이터만 추출
    sdf_draw_eis_df_0_numeric = sdf_draw_eis_df_0.select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)
    sdf_draw_eis_df_1_numeric = sdf_draw_eis_df_1.select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)

    # 데이터 범위 설정
    min_val = min(sdf_draw_eis_df_0_numeric.min().min(), sdf_draw_eis_df_1_numeric.min().min())
    max_val = max(sdf_draw_eis_df_0_numeric.max().max(), sdf_draw_eis_df_1_numeric.max().max())

    # 플롯 생성
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    fig.suptitle('Z distribution Frequency Point at ' + str(f[2:]) + "Hz")
    for i, ft in enumerate(["Zre (mohm)", "Zim (mohm)", "Mag (mohm)", "Phase"]):
        dummy_draw_0 = sdf_draw_eis_df_0[ft]
        dummy_draw_1 = sdf_draw_eis_df_1[ft]
        row = int(i / 2)
        col = i % 2
        # X축 범위 생성
        x_vals = np.linspace(min_val, max_val, 1000)
        # PDF 계산 (Z-Score 기반 정규분포)
        pdf_0 = norm.pdf(x_vals, loc=dummy_draw_0.mean(), scale=dummy_draw_0.std())
        pdf_1 = norm.pdf(x_vals, loc=dummy_draw_1.mean(), scale=dummy_draw_1.std())
        # Plot histogram and PDF for P/F == 0 (red color)
        axs[row][col].hist(dummy_draw_0, bins=10, color='red', edgecolor='black', density=True, alpha=0.3, label="F")
        axs[row][col].plot(x_vals, pdf_0, color='red', lw=2, label="F-PDF")
        # Plot histogram and PDF for P/F == 1 (blue color)
        axs[row][col].hist(dummy_draw_1, bins=10, color='blue', edgecolor='black', density=True, alpha=0.3, label="P")
        axs[row][col].plot(x_vals, pdf_1, color='blue', lw=2, label="P-PDF")
        # 레이블 설정
        axs[row][col].set_title(ft)
        axs[row][col].set_xlabel("Standard Deviation")
        axs[row][col].set_ylabel('Density')
        axs[row][col].legend()
    # 레이아웃 조정
    plt.tight_layout()



# 결과를 저장할 딕셔너리 생성
intersection_dict = {ft: [] for ft in inf_cols}
left_dict = {ft: [] for ft in inf_cols}
right_dict = {ft: [] for ft in inf_cols}

sdf_draw_df_0 = snd_inf_df[snd_inf_df["P/F"] == 0]
sdf_draw_df_1 = snd_inf_df[snd_inf_df["P/F"] == 1]

# 각 피처에 대해 PDF 생성 및 적분 계산
for i, ft in enumerate(inf_cols):
    dummy_draw_0 = sdf_draw_df_0[ft]
    dummy_draw_1 = sdf_draw_df_1[ft]

    # PDF 기반 연속확률밀도 계산
    min_val = min(dummy_draw_0.min(), dummy_draw_1.min())
    max_val = max(dummy_draw_0.max(), dummy_draw_1.max())
    x_vals = np.linspace(min_val, max_val, 1000)

    # PDF 계산
    pdf_0_vals = norm.pdf(x_vals, loc=dummy_draw_0.mean(), scale=dummy_draw_0.std())
    pdf_1_vals = norm.pdf(x_vals, loc=dummy_draw_1.mean(), scale=dummy_draw_1.std())

    # 교차 지점 찾기
    difference = pdf_0_vals - pdf_1_vals
    intersection_indices = np.where(np.diff(np.sign(difference)))[0]

    if len(intersection_indices) > 0:
        intersection_index = intersection_indices[0]

        # 교차 지점 이전 (왼쪽) 영역 적분
        x_vals_left = x_vals[:intersection_index + 1]
        min_pdf_vals_left = np.minimum(pdf_0_vals[:intersection_index + 1], pdf_1_vals[:intersection_index + 1])
        left_area = integrate.simpson(min_pdf_vals_left, x=x_vals_left)
        left_dict[ft].append(left_area)

        # 교차 지점 이후 (오른쪽) 영역 적분
        x_vals_right = x_vals[intersection_index:]
        min_pdf_vals_right = np.minimum(pdf_0_vals[intersection_index:], pdf_1_vals[intersection_index:])
        right_area = integrate.simpson(min_pdf_vals_right, x=x_vals_right)
        right_dict[ft].append(right_area)

        # 전체 교차 영역 적분
        intersection_area = left_area + right_area
        intersection_dict[ft].append(intersection_area)
    else:
        # 교차 지점이 없는 경우, 전체 영역을 적분
        intersection_area = integrate.simpson(np.minimum(pdf_0_vals, pdf_1_vals), x=x_vals)
        intersection_dict[ft].append(intersection_area)
        left_dict[ft].append(0)  # 교차 지점이 없으므로 왼쪽, 오른쪽이 없음
        right_dict[ft].append(0)

# 데이터프레임 생성
intersection_ind_pdf_df = pd.DataFrame.from_dict(intersection_dict, orient='index').transpose()
left_df = pd.DataFrame.from_dict(left_dict, orient='index').transpose()
right_df = pd.DataFrame.from_dict(right_dict, orient='index').transpose()

# left와 right를 intersection_ind_pdf_df에 추가
intersection_ind_pdf_df = pd.concat([intersection_ind_pdf_df, left_df.add_prefix("Left_"), right_df.add_prefix("Right_")], axis=1)
intersection_ind_pdf_df*np.array(1.0e6)
intersection_ind_kde_df.iloc[:, :2]*np.array(1.0e6) - intersection_ind_pdf_df.iloc[:, :2]*np.array(1.0e6)


# 결과 저장용 딕셔너리 초기화
intersection_dict = {ft: [] for ft in eis_ft_cols}
left_dict = {ft: [] for ft in eis_ft_cols}
right_dict = {ft: [] for ft in eis_ft_cols}

# PDF 기반 적분 계산
for f in f_df_list:
    # 데이터 분리
    sdf_draw_df_0 = snd_df[f][snd_df[f]["P/F"] == 0]  # P/F == 0
    sdf_draw_df_1 = snd_df[f][snd_df[f]["P/F"] == 1]  # P/F == 1

    # 각 피처에 대해 PDF 생성 및 적분 계산
    for ft in eis_ft_cols:
        dummy_draw_0 = sdf_draw_df_0[ft]
        dummy_draw_1 = sdf_draw_df_1[ft]

        # PDF 기반 확률밀도 계산
        min_val = min(dummy_draw_0.min(), dummy_draw_1.min())
        max_val = max(dummy_draw_0.max(), dummy_draw_1.max())
        x_vals = np.linspace(min_val, max_val, 1000)

        pdf_0_vals = norm.pdf(x_vals, loc=dummy_draw_0.mean(), scale=dummy_draw_0.std())
        pdf_1_vals = norm.pdf(x_vals, loc=dummy_draw_1.mean(), scale=dummy_draw_1.std())

        # 교차 지점 찾기
        difference = pdf_0_vals - pdf_1_vals
        intersection_indices = np.where(np.diff(np.sign(difference)))[0]

        if len(intersection_indices) > 0:
            intersection_index = intersection_indices[0]

            # 교차 지점 이전 (왼쪽) 영역 적분
            x_vals_left = x_vals[:intersection_index + 1]
            min_pdf_vals_left = np.minimum(pdf_0_vals[:intersection_index + 1], pdf_1_vals[:intersection_index + 1])
            left_area = integrate.simpson(min_pdf_vals_left, x=x_vals_left)
            left_dict[ft].append(left_area)

            # 교차 지점 이후 (오른쪽) 영역 적분
            x_vals_right = x_vals[intersection_index:]
            min_pdf_vals_right = np.minimum(pdf_0_vals[intersection_index:], pdf_1_vals[intersection_index:])
            right_area = integrate.simpson(min_pdf_vals_right, x=x_vals_right)
            right_dict[ft].append(right_area)

            # 전체 교차 영역 적분
            intersection_area = left_area + right_area
            intersection_dict[ft].append(intersection_area)
        else:
            # 교차 지점이 없는 경우, 전체 영역을 적분
            intersection_area = integrate.simpson(np.minimum(pdf_0_vals, pdf_1_vals), x=x_vals)
            intersection_dict[ft].append(intersection_area)
            left_dict[ft].append(0)  # 교차 지점이 없으므로 왼쪽, 오른쪽 없음
            right_dict[ft].append(0)

# 교차 영역 결과를 데이터프레임으로 변환
intersection_eis_pdf_df = pd.DataFrame.from_dict(intersection_dict, orient='index').transpose()

# left_dict와 right_dict를 데이터프레임으로 변환
left_df = pd.DataFrame.from_dict(left_dict, orient='index').transpose()
right_df = pd.DataFrame.from_dict(right_dict, orient='index').transpose()

# intersection_eis_pdf_df에 왼쪽(left)과 오른쪽(right) 적분 값을 추가
intersection_eis_pdf_df = pd.concat([intersection_eis_pdf_df, left_df.add_prefix("Left_"), right_df.add_prefix("Right_")], axis=1)
intersection_eis_pdf_df = pd.concat([intersection_eis_pdf_df, pd.DataFrame(Freq_list, columns=["Frequency"])], axis=1)
intersection_eis_pdf_df = intersection_eis_pdf_df[inter_df_col]

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
fig.suptitle("Frequency range : 1, 5, 10Hz")
label = ["PUR", "P(P|F)", "P(F|P)"]
features_per_plot = 3
num_plots = len(inter_df_col[1:]) // features_per_plot
for i in range(num_plots):
    row = int(i / 2)
    col = i % 2
    start_idx = i * features_per_plot
    end_idx = start_idx + features_per_plot

    for l, ft in enumerate(inter_df_col[start_idx + 1:end_idx + 1]):
        axs[row][col].plot(list(intersection_eis_pdf_df.index[:]), intersection_eis_pdf_df[ft][:] * np.array(1.0e6), "o-",label=label[l])

    axs[row][col].set_title(inter_df_col[end_idx-2])
    axs[row][col].set_xlabel("Frequency(Hz)")
    axs[row][col].set_ylabel("PUR(ppm)")
    axs[row][col].set_xticks(ticks=[list(intersection_eis_pdf_df.index)[i] for i in [0, 10, 20, 30]], labels=[1, 10, 100, 1000])
    # axs[row][col].set_yticks(np.arange(0,100,10))
    axs[row][col].legend(loc="upper left", fontsize=10)  # 각 피처에 대한 범례 추가
plt.tight_layout()


fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
fig.suptitle("Frequency range : 1, 5, 10Hz")
label = ["PUR", "P(P|F)", "P(F|P)"]
features_per_plot = 3
num_plots = len(inter_df_col[1:]) // features_per_plot
for i in range(num_plots):
    row = int(i / 2)
    col = i % 2
    start_idx = i * features_per_plot
    end_idx = start_idx + features_per_plot
    for l, ft in enumerate(inter_df_col[start_idx + 1:end_idx + 1]):
        axs[row][col].plot(
            list(intersection_eis_pdf_df.index[:]),
            intersection_eis_pdf_df[ft][:] * np.array(1.0e6),
            "o-",
            label=label[l],)
    axs[row][col].set_title(inter_df_col[end_idx - 2])
    axs[row][col].set_xlabel("Frequency(Hz)")
    axs[row][col].set_ylabel("Log scale PUR(ppm)")
    axs[row][col].set_xticks(ticks=[list(intersection_eis_pdf_df.index)[i] for i in [0, 10, 20, 30]], labels=[1, 10, 100, 1000])
    # Y축에만 로그 스케일 적용
    axs[row][col].set_yscale("log")
    axs[row][col].set_ylim([1.0e3, 1.0e6])
    axs[row][col].legend(loc="upper left", fontsize=10)
plt.tight_layout()


# 하나의 plot으로 그리기
fig, ax = plt.subplots()
fig.suptitle("PUR for EIS Feature using pdf function")
label = ["PUR", "P(P|F)", "P(F|P)"]
features_per_plot = 3
num_plots = len(inter_df_col[1:]) // features_per_plot
for i in range(num_plots):
    row = int(i / 2)
    col = i % 2
    start_idx = i * features_per_plot
    end_idx = start_idx + features_per_plot
    # for l, ft in enumerate(inter_df_col[start_idx + 1:end_idx + 1]):
    for l, ft in enumerate(inter_df_col[start_idx + 1:start_idx + 2]):
        ax.plot(
            list(intersection_eis_pdf_df.index[:]),
            intersection_eis_pdf_df[ft][:] * np.array(1.0e6),"o-",label=ft)
    # ax.set_title(inter_df_col[end_idx - 2])
    ax.set_xlabel("Frequency(Hz)")
    ax.set_ylabel("PUR(ppm)")
    ax.set_xticks(ticks=[list(intersection_eis_pdf_df.index)[i] for i in [0, 10, 20, 30]], labels=[1, 10, 100, 1000])
    # Y축에만 로그 스케일 적용
    ax.set_yscale("log")
    ax.set_ylim([1.0e3, 1.0e6])
    ax.legend(loc="upper left", fontsize=10)
plt.tight_layout()


compare_eis_pur_kde = intersection_eis_kde_df[eis_ft_cols]
compare_eis_pur_pdf = intersection_eis_pdf_df[eis_ft_cols]
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
fig.suptitle("PUR based on EIS Features")
features_per_plot = 1
num_plots = len(inter_df_col[1:]) // features_per_plot
for i, ft in enumerate(eis_ft_cols):
    row = int(i / 2)
    col = i % 2
    axs[row][col].plot(list(intersection_eis_pdf_df.index[:]),compare_eis_pur_kde[ft][:] * np.array(1.0e6),"o-",label="KDE")
    axs[row][col].plot(list(intersection_eis_kde_df.index[:]),compare_eis_pur_pdf[ft][:] * np.array(1.0e6),"o-",label="PDF")
    axs[row][col].set_title(ft)
    axs[row][col].set_xlabel("Frequency(Hz)")
    axs[row][col].set_ylabel("PUR(ppm)")
    axs[row][col].set_xticks(ticks=[list(intersection_eis_pdf_df.index)[i] for i in [0, 10, 20, 30]], labels=[1, 10, 100, 1000])
    # Y축에만 로그 스케일 적용
    axs[row][col].set_yscale("log")
    axs[row][col].set_ylim([1.0e3, 1.0e6])
    axs[row][col].legend(loc="upper left", fontsize=10)
plt.tight_layout()




compare_eis_pur_kde = intersection_eis_kde_df[eis_ft_cols]*np.array(1.0e6)
compare_eis_pur_pdf = intersection_eis_pdf_df[eis_ft_cols]*np.array(1.0e6)
compare_ind_pur_kde = intersection_ind_kde_df[inf_cols]*np.array(1.0e6)
compare_ind_pur_pdf = intersection_ind_pdf_df[inf_cols]*np.array(1.0e6)


test_dff_df = ((np.abs((compare_eis_pur_kde) - (compare_eis_pur_pdf)))/(np.maximum((compare_eis_pur_kde), (compare_eis_pur_pdf))))*np.array(100)
test_dff_df = ((np.abs((compare_eis_pur_kde) - (compare_eis_pur_pdf)))/(np.maximum((compare_eis_pur_kde), (compare_eis_pur_pdf))))*np.array(100)

diff_eis_df = ((np.abs((compare_eis_pur_kde) - (compare_eis_pur_pdf)))/np.maximum((compare_eis_pur_kde), (compare_eis_pur_pdf)))*np.array(100)
diff_eis_df = pd.concat([pd.DataFrame(Freq_list, columns=["Frequency"]), diff_eis_df], axis=1)
diff_inf_df = ((np.abs((compare_ind_pur_kde) - (compare_ind_pur_pdf)))/np.maximum((compare_ind_pur_kde), (compare_ind_pur_pdf)))


fig, ax = plt.subplots(2,2)
for i, ft in enumerate(eis_ft_cols):
    ax = ax.flatten()
    p_val_eis = shapiro_norm_eis_df_p[shapiro_norm_eis_df_p["Feature"]==ft]["p-value"].reset_index(drop=True)
    ax[i].plot(diff_eis_df[ft], p_val_eis, "o")
    ax[i].set_title(ft)
    ax[i].set_xlabel("PUR differences between KDE and PDF(%)")
    ax[i].set_ylabel("P-value for pass group")
plt.tight_layout()


fig, ax = plt.subplots(2,2)
for i, ft in enumerate(eis_ft_cols):
    ax = ax.flatten()
    p_val_eis = shapiro_norm_eis_df_p[shapiro_norm_eis_df_p["Feature"]==ft]["skewness"].reset_index(drop=True)
    ax[i].plot(diff_eis_df[ft], p_val_eis, "o")
    ax[i].set_title(ft)
    ax[i].set_xlabel("PUR differences between KDE and PDF(%)")
    ax[i].set_ylabel("skewness for pass group")
plt.tight_layout()


fig, ax = plt.subplots(2,2)
for i, ft in enumerate(eis_ft_cols):
    ax = ax.flatten()
    p_val_eis = shapiro_norm_eis_df_p[shapiro_norm_eis_df_p["Feature"]==ft]["Kurtosis"].reset_index(drop=True)
    ax[i].plot(diff_eis_df[ft], p_val_eis, "o")
    ax[i].set_title(ft)
    ax[i].set_xlabel("PUR differences between KDE and PDF(%)")
    ax[i].set_ylabel("skewness for pass group")
plt.tight_layout()



# 조건에 따라 스코어 계산 함수 정의
def calculate_score(row):
    # 열 A의 조건
    score_A = 1 if row['p-value'] > 0.05 else row['p-value'] / 0.05
    # 열 B의 조건
    score_B = 1 if row['skewness'] < 3 else row['skewness'] / 3
    # 열 C의 조건
    score_C = 1 if row['Kurtosis'] < 10 else row['Kurtosis'] / 10
    # 최종 스코어 계산 (원한다면 합산, 곱셈 등 조합 가능)
    return (score_A+score_B+score_C)/3
    # return (score_B+score_C)/2


shapiro_norm_eis_df_p["Score"] = shapiro_norm_eis_df_p.apply(calculate_score, axis=1)
# shapiro_norm_eis_df_p['D'] = shapiro_norm_eis_df_p.apply(calculate_score, axis=1)


fig, ax = plt.subplots(2,2)
for i, ft in enumerate(eis_ft_cols):
    ax = ax.flatten()
    score = shapiro_norm_eis_df_p[shapiro_norm_eis_df_p["Feature"]==ft]["Score"].reset_index(drop=True)
    ax[i].plot(diff_eis_df[ft], score, "o")
    ax[i].set_title(ft)
    ax[i].set_xlabel("PUR differences ratio(%)")
    ax[i].set_ylabel("PSK Score")
plt.tight_layout()









# 분할 개수에 따라 ppm 변동성 확인
# Calculate to PUR of EIS Features
# 결과 저장용 리스트 초기화
pur_M, pur_F, pur_Re, pur_Im = [], [], [], []
intersection_dict = {ft: [] for ft in eis_ft_cols}
left_dict = {ft: [] for ft in eis_ft_cols}
right_dict = {ft: [] for ft in eis_ft_cols}

# KDE 함수를 저장할 딕셔너리 생성
kde_functions_0 = {ft: [] for ft in eis_ft_cols}
kde_functions_1 = {ft: [] for ft in eis_ft_cols}
kde_dataframes_0 = {}
kde_dataframes_1 = {}

for vals in [500, 1000, 1500, 2000, 2500, 3000, 5000]:
    sdf_draw_df_0 = snd_df[f_df_list[0]][snd_df[f_df_list[0]]["P/F"] == 0]  # P/F == 0
    sdf_draw_df_1 = snd_df[f_df_list[0]][snd_df[f_df_list[0]]["P/F"] == 1]  # P/F == 1

    # 각 피처에 대해 KDE 생성 및 적분 계산
    for i, ft in enumerate(eis_ft_cols):
        dummy_draw_0 = sdf_draw_df_0[ft]
        dummy_draw_1 = sdf_draw_df_1[ft]

        # P/F 별 연속확률밀도함수 추정
        kde_0 = gaussian_kde(dummy_draw_0)
        kde_1 = gaussian_kde(dummy_draw_1)
        kde_functions_0[ft].append(kde_0)  # 피처별로 kde_0 함수 저장
        kde_functions_1[ft].append(kde_1)

        # 교체 영역에 대해 적분 구간 분할
        min_val = min(dummy_draw_0.min(), dummy_draw_1.min())
        max_val = max(dummy_draw_0.max(), dummy_draw_1.max())
        x_vals = np.linspace(min_val, max_val, vals)

        # 연속확률밀도 값 추출
        kde_0_vals = kde_0(x_vals)
        kde_1_vals = kde_1(x_vals)

        # KDE 값을 DataFrame으로 저장
        kde_df_0 = pd.DataFrame({'X': x_vals, f'KDE_{ft}_0': kde_0_vals})
        kde_df_1 = pd.DataFrame({'X': x_vals, f'KDE_{ft}_1': kde_1_vals})
        kde_dataframes_0[ft] = kde_df_0
        kde_dataframes_1[ft] = kde_df_1

        # 교차 지점 찾기
        difference = kde_0_vals - kde_1_vals
        intersection_indices = np.where(np.diff(np.sign(difference)))[0]

        if len(intersection_indices) > 0:
            intersection_index = intersection_indices[0]

            # 교차 지점 이전 (왼쪽) 영역 적분
            x_vals_left = x_vals[:intersection_index + 1]
            min_kde_vals_left = np.minimum(kde_0_vals[:intersection_index + 1], kde_1_vals[:intersection_index + 1])
            left_area = integrate.simpson(min_kde_vals_left, x=x_vals_left)
            left_dict[ft].append(left_area)

            # 교차 지점 이후 (오른쪽) 영역 적분
            x_vals_right = x_vals[intersection_index:]
            min_kde_vals_right = np.minimum(kde_0_vals[intersection_index:], kde_1_vals[intersection_index:])
            right_area = integrate.simpson(min_kde_vals_right, x=x_vals_right)
            right_dict[ft].append(right_area)

            # 전체 교차 영역 적분
            intersection_area = left_area + right_area
            intersection_dict[ft].append(intersection_area)
        else:
            # 교차 지점이 없는 경우, 전체 영역을 적분
            intersection_area = integrate.simpson(np.minimum(kde_0_vals, kde_1_vals), x=x_vals)
            intersection_dict[ft].append(intersection_area)
            left_dict[ft].append(0)  # 교차 지점이 없으므로 왼쪽, 오른쪽이 없음
            right_dict[ft].append(0)

# 기존의 intersection_df를 생성하는 코드
intersection_eis_kde_df_vals = pd.DataFrame.from_dict(intersection_dict, orient='index').transpose()
intersection_eis_kde_df_vals*np.array(1.0e6)
# intersection_eis_kde_df_vals = pd.concat([intersection_eis_kde_df_vals, pd.DataFrame(Freq_list, columns=["Frequency"])], axis=1)