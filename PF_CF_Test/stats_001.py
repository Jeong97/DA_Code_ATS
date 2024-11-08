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
from scipy.stats import shapiro, norm, gaussian_kde
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


''' Raw Data EDA '''
# Set path
path = r"C:\Users\jeongbs1\오토실리콘\1. python_code\ATS_Code\PF_CF_Test"

# Load raw data
raw_data = pd.read_csv(path+"/03-02 DS03 RDF - PassFail Discrimination V0.0 240910.csv")
raw_data["P/F"] = raw_data["P/F"].replace({"P":1, "F":0})

# Split by Cell Name
Cell_name = raw_data["Cell Name"].unique()


# Make split dataframe
data = {}
for i in range(len(Cell_name)):
    data[(Cell_name[i])] = (raw_data.groupby("Cell Name").get_group(Cell_name[i])).reset_index()
    data[(Cell_name[i])]["Zim (mohm)"] = -data[(Cell_name[i])]["Zim (mohm)"]
len(data)
data[(Cell_name[0])].columns


# Nyquist plot by Cell Name
plt.figure()
for i in range(len(Cell_name)):
    x = list(data[(Cell_name[i])]["Zre (mohm)"])
    y = list(data[(Cell_name[i])]["Zim (mohm)"])
    if data[(Cell_name[i])]["P/F"][0] == 0:
        plt.plot(x, y, "o-", label=Cell_name[i], color="red")
    else:
        plt.plot(x, y, "o-", label=Cell_name[i], color="blue")
plt.xlabel("Z_Real(mOhm)")
plt.ylabel("Z_Imaginary(mOhm)")
plt.tight_layout()


# Board plot by Cell Name
plt.figure()
for i in range(len(Cell_name)):
    x = list(data[(Cell_name[i])]["Freq"])
    y = list((data[(Cell_name[i])]["Zre (mohm)"]))
    if data[(Cell_name[i])]["P/F"][0] == 0:
        plt.plot(x, y, "o-", label=Cell_name[i], color="red")
    else:
        plt.plot(x, y, "o-", label=Cell_name[i], color="blue")
plt.xlabel("Freq")
plt.ylabel("Z_Real(mOhm)")
plt.tight_layout()


# Board plot by Cell Name
plt.figure()
for i in range(len(Cell_name)):
    x = list(data[(Cell_name[i])]["Freq"])
    y = list((data[(Cell_name[i])]["Zim (mohm)"]))
    if data[(Cell_name[i])]["P/F"][0] == 0:
        plt.plot(x, y, "o-", label=Cell_name[i], color="red")
    else:
        plt.plot(x, y, "o-", label=Cell_name[i], color="blue")
plt.xlabel("Freq")
plt.ylabel("Z_Imaginary(mOhm)")
plt.tight_layout()



''' Data PreProcessing '''
# Make new dataframe by frequency
freq = list((data[(Cell_name[0])]["Freq"]))
del freq[3]


freq_df = {}
for f in freq:
    dummy = {}
    for i in range(len(Cell_name)):
        dummy[i] = data[(Cell_name[i])][data[(Cell_name[i])]["Freq"] == f][["Cell Name", "Freq", "Zre (mohm)", "Zim (mohm)", "Mag (mohm)", "Phase", "P/F"]].reset_index(drop=True)
        if i == 0:
            freq_df[f] = data[(Cell_name[i])][data[(Cell_name[i])]["Freq"] == f][["Cell Name", "Freq", "Zre (mohm)", "Zim (mohm)", "Mag (mohm)", "Phase", "P/F"]].reset_index(drop=True)
        else:
            freq_df[f] = pd.concat([freq_df[f], dummy[i]], axis=0, ignore_index=True)

        freq_df[f].drop(index=freq_df[f][freq_df[f]["Cell Name"].isin(["NC01", "NC02", "NC03"])].index, inplace=True)


# Make Transfer to Z_score DataFrame
snd_df = {}
for f in freq:
    dummy = freq_df[f][["Zre (mohm)", "Zim (mohm)", "Mag (mohm)", "Phase"]]
    dummy = ((dummy - dummy.mean()) / dummy.std())
    org_dummy_df = freq_df[f][[(freq_df[f].columns)[0], (freq_df[f].columns)[1], (freq_df[f].columns)[-1]]]
    snd_df[f] = pd.concat([org_dummy_df, dummy], axis=1)


# Normality Test
result_df = []
for f in freq:
    # shapiro_df = freq_df[f]
    shapiro_df = freq_df[f][freq_df[f]["P/F"] == 1]
    for column in ["Zre (mohm)", "Zim (mohm)", "Mag (mohm)", "Phase"]:
        stat, p_value = shapiro(shapiro_df[column])
        skewness = shapiro_df[column].skew()
        Kurtosis = shapiro_df[column].kurtosis()
        print(f, column, stat, p_value)
        result_df.append({'Frequency': f, 'Feature': column, 'Shapiro Stat': stat, 'p-value': p_value, 'skewness': skewness, 'Kurtosis': Kurtosis})
shapiro_result_df = pd.DataFrame(result_df)
# shapiro_result_df.to_excel("C:/Users/jeongbs1/Downloads/pass_pp_cell_normality.xlsx", index=False)
len(snd_df[f][snd_df[f]["P/F"] == 0])
len(snd_df[f][snd_df[f]["P/F"] == 1])




snd_col = ["Zre (mohm)", "Zim (mohm)", "Mag (mohm)", "Phase"]
# Normal Distribution
# for f in freq[(np.where(freq==np.array(10.0)))[0][0]:]:
for f in freq[19:20]:
    sdf_draw_df_0 = freq_df[f][freq_df[f]["P/F"] == 0]  # P/F == 0
    sdf_draw_df_1 = freq_df[f][freq_df[f]["P/F"] == 1]

    sdf_draw_df_0_numeric = sdf_draw_df_0.select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)
    sdf_draw_df_1_numeric = sdf_draw_df_1.select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)


    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    fig.suptitle('Normal distribution Frequency Point at ' + str(f) + "Hz")
    for i, ft in enumerate(snd_col):
        dummy_draw_0 = sdf_draw_df_0[ft]
        dummy_draw_1 = sdf_draw_df_1[ft]

        row = int(i / 2)
        col = i % 2

        min_val = min(dummy_draw_0.min(), dummy_draw_1.min())
        max_val = max(dummy_draw_0.max(), dummy_draw_1.max())

        # 동적으로 빈 수 설정: 히스토그램의 균형 유지
        num_bins_0 = max(10, int(np.sqrt(len(dummy_draw_0))))
        num_bins_1 = max(10, int(np.sqrt(len(dummy_draw_1))))
        bins = np.linspace(min_val, max_val, 30)
        # bins = np.linspace(min_val, max_val, max(num_bins_0, num_bins_1), 30)

        # KDE plot for P/F == 0
        kde_0 = gaussian_kde(dummy_draw_0)
        x_vals_0 = np.linspace(min_val, max_val, 100)

        # KDE plot for P/F == 1
        kde_1 = gaussian_kde(dummy_draw_1)
        x_vals_1 = np.linspace(min_val, max_val, 100)

        # Plot histogram and KDE for P/F == 0 (blue color)
        axs[row][col].hist(dummy_draw_0, bins=bins, color='red', edgecolor='black', density=True, alpha=0.6,label="F")
        axs[row][col].plot(x_vals_0, kde_0(x_vals_0), color='red', lw=2)

        # Plot histogram and KDE for P/F == 1 (red color)
        axs[row][col].hist(dummy_draw_1, bins=bins, color='blue', edgecolor='black', density=True, alpha=0.6,label="P")
        axs[row][col].plot(x_vals_1, kde_1(x_vals_1), color='blue', lw=2)

        # 레이블 설정
        axs[row][col].set_xlabel(ft)
        axs[row][col].set_ylabel('Density')
        axs[row][col].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # suptitle과 겹치지 않도록 설정


# Z Distribution
# for f in freq[(np.where(freq==np.array(10.0)))[0][0]:]:
for f in freq[19:20]:
    sdf_draw_df_0 = snd_df[f][snd_df[f]["P/F"] == 0]  # P/F == 0
    sdf_draw_df_1 = snd_df[f][snd_df[f]["P/F"] == 1]

    sdf_draw_df_0_numeric = sdf_draw_df_0.select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)
    sdf_draw_df_1_numeric = sdf_draw_df_1.select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)

    min_val = min(sdf_draw_df_0_numeric.min().min(), sdf_draw_df_1_numeric.min().min())
    max_val = max(sdf_draw_df_0_numeric.max().max(), sdf_draw_df_1_numeric.max().max())
    bins = np.linspace(min_val, max_val, 30)

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    fig.suptitle('Z distribution Frequency Point at ' + str(f) + "Hz")
    for i, ft in enumerate(snd_col):
        dummy_draw_0 = sdf_draw_df_0[ft]
        dummy_draw_1 = sdf_draw_df_1[ft]

        row = int(i / 2)
        col = i % 2

        # KDE plot for P/F == 0
        kde_0 = gaussian_kde(dummy_draw_0)
        x_vals_0 = np.linspace(min_val, max_val, 100)

        # KDE plot for P/F == 1
        kde_1 = gaussian_kde(dummy_draw_1)
        x_vals_1 = np.linspace(min_val, max_val, 100)

        # Plot histogram and KDE for P/F == 0 (blue color)
        axs[row][col].hist(dummy_draw_0, bins=bins, color='red', edgecolor='black', density=True, alpha=0.6,label="F")
        axs[row][col].plot(x_vals_0, kde_0(x_vals_0), color='red', lw=2)

        # Plot histogram and KDE for P/F == 1 (red color)
        axs[row][col].hist(dummy_draw_1, bins=bins, color='blue', edgecolor='black', density=True, alpha=0.6,label="P")
        axs[row][col].plot(x_vals_1, kde_1(x_vals_1), color='blue', lw=2)

        # 레이블 설정
        axs[row][col].set_xlabel(ft)
        axs[row][col].set_ylabel('Density')
        axs[row][col].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # suptitle과 겹치지 않도록 설정


# QQ-Plot
for f in freq[19:20]:
# for f in freq[(np.where(freq==np.array(10.0)))[0][0]:]:
    sdf_draw_df_0 = snd_df[f][snd_df[f]["P/F"] == 0]  # P/F == 0
    sdf_draw_df_1 = snd_df[f][snd_df[f]["P/F"] == 1]  # P/F == 1

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
    fig.suptitle('Frequency Point at ' + str(f) + "Hz")

    for i, ft in enumerate(snd_col):
        dummy_draw_0 = sdf_draw_df_0[ft]  # P/F == 0 데이터
        dummy_draw_1 = sdf_draw_df_1[ft]  # P/F == 1 데이터

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
        axs[row][col].set_xlabel(ft)
        axs[row][col].set_ylabel('Sample Quantiles')
        axs[row][col].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])


# Calculate to PUR
pur_M, pur_F, pur_Re, pur_Im = [], [], [], []
intersection_dict = {ft: [] for ft in snd_col}
for f in freq[::-1]:
    sdf_draw_df_0 = freq_df[f][freq_df[f]["P/F"] == 0]  # P/F == 0
    sdf_draw_df_1 = freq_df[f][freq_df[f]["P/F"] == 1]

    sdf_draw_df_0_numeric = sdf_draw_df_0.select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)
    sdf_draw_df_1_numeric = sdf_draw_df_1.select_dtypes(include=[np.number]).drop(["Freq", "P/F"], axis=1)

    for i, ft in enumerate(snd_col):
        dummy_draw_0 = sdf_draw_df_0[ft]
        dummy_draw_1 = sdf_draw_df_1[ft]

        # KDE plot 생성
        kde_0 = gaussian_kde(dummy_draw_0)
        kde_1 = gaussian_kde(dummy_draw_1)

        min_val = min(dummy_draw_0.min(), dummy_draw_1.min())
        max_val = max(dummy_draw_0.max(), dummy_draw_1.max())
        x_vals = np.linspace(min_val, max_val, 1000)

        kde_0_vals = kde_0(x_vals)
        kde_1_vals = kde_1(x_vals)

        # 교차 영역 계산
        intersection_vals = np.minimum(kde_0_vals, kde_1_vals)
        intersection_area = integrate.simpson(intersection_vals, x=x_vals)

        intersection_dict[ft].append(intersection_area)

        # 교차 면적 출력
        # print(f"{ft}의 교차 영역 면적: {intersection_area:.4f}")
intersection_df = pd.DataFrame.from_dict(intersection_dict, orient='index').transpose()
intersection_df = pd.concat([intersection_df, pd.DataFrame(freq[::-1], columns=["Frequency"])], axis=1)

fig, axs = plt.subplots(ncols=2, nrows=2)
fig.suptitle("PUR")
for i, ft in enumerate(intersection_df.columns):
    row = int(i / 2)
    col = i % 2
    axs[row][col].plot(intersection_df[ft], "o")
    axs[row][col].set_yticks(np.arange(0,1,0.1))
    # axs[row][col].set_xlabel("Frequency(Hz)")
    axs[row][col].set_ylabel(ft)
plt.tight_layout()




# T-Test
# homoscedasticity Test
def homoscedasticity(x1,x2):
    f_test = st.f_oneway(x1, x2)
    bartlett = st.bartlett(x1, x2)
    levene = st.levene(x1, x2)
    fligner = st.fligner(x1, x2)
    return {'F-Test p-value': f_test.pvalue, 'Bartlett p-value': bartlett.pvalue, 'Levene p-value': levene.pvalue, 'Fligner p-value': fligner.pvalue}


# Loop Through Features and Perform Tests
# Store Results
t_test_results_list = []

for f in freq[4:]:
    t_test_p_df = freq_df[f][freq_df[f]["P/F"] == 1]
    t_test_f_df = freq_df[f][freq_df[f]["P/F"] == 0]

    mag_p, mag_f = t_test_p_df["Mag (mohm)"], t_test_f_df["Mag (mohm)"]
    phase_p, phase_f = t_test_p_df["Phase"], t_test_f_df["Phase"]
    re_p, re_f = t_test_p_df["Zre (mohm)"], t_test_f_df["Zre (mohm)"]
    im_p, im_f = t_test_p_df["Zim (mohm)"], t_test_f_df["Zim (mohm)"]

    t_df = {"Mag":[mag_p, mag_f], "Phase":[phase_p, phase_f],"Zre":[re_p, re_f],"Zim":[im_p, im_f]}
    for feature in list(t_df.keys()):
        if (homoscedasticity(t_df[feature][0], t_df[feature][1]))["Levene p-value"] > 0.05:
            result_t = st.ttest_ind(t_df[feature][0], t_df[feature][1], equal_var=True)
            test_type = "Student's T-test"
        else:
            result_t = st.ttest_ind(t_df[feature][0], t_df[feature][1], equal_var=False)
            test_type = "Welch's T-test"
        # Append results to the list
        t_test_results_list.append({
            'Frequency': f,
            'Feature': feature,
            'Test Type': test_type,
            # 'T-statistic': result_t.statistic,
            'p-value': result_t.pvalue
        })
t_test_results_df = pd.DataFrame(t_test_results_list)
# t_test_results_df.to_excel("C:/Users/jeongbs1/Downloads/af_T-Test.xlsx", index=False)






