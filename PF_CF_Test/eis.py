# Cell P/F Data Analysis
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import scipy.stats as stats
import seaborn as sns
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


''' Raw Data EDA '''
# Set path
path = r"C:\Users\jeongbs1\오토실리콘\1. python_code\ATS_Code\PF_CF_Test"

# Load raw data
raw_data = pd.read_csv(path+"/03-02 DS03 RDF - PassFail Discrimination V0.0 240910.csv")

# Split by Cell Name
Cell_name = raw_data["Cell Name"].unique()


# Make split dataframe
data = {}
for i in range(len(Cell_name)):
    data[(Cell_name[i])] = (raw_data.groupby("Cell Name").get_group(Cell_name[i])).reset_index()
    data[(Cell_name[i])]["Zim (mohm)"] = -data[(Cell_name[i])]["Zim (mohm)"]
len(data)
data[(Cell_name[0])].columns

# Check Frequency by Cell Name
plt.figure()
for i in range(len(Cell_name)):
    x = list((data[(Cell_name[i])]["Freq"]).index)
    y = list(data[(Cell_name[i])]["Freq"])
    plt.plot(x, y, "o-", label=Cell_name[i])
plt.xlabel("Frequency Index")
plt.ylabel("Frequency(Hz)")
plt.tight_layout()

# Nyquist plot by Cell Name
plt.figure()
for i in range(len(Cell_name)):
    x = list(data[(Cell_name[i])]["Zre (mohm)"])
    y = list(data[(Cell_name[i])]["Zim (mohm)"])
    plt.plot(x, y, "o-", label=Cell_name[i])
plt.xlabel("Z_Real(mOhm)")
plt.ylabel("Z_Imaginary(mOhm)")
plt.tight_layout()

# Board plot by Cell Name
plt.figure()
for i in range(len(Cell_name)):
    x = list(data[(Cell_name[i])]["Freq"])
    y = list(data[(Cell_name[i])]["Zre (mohm)"])
    plt.plot(x, y, "o-", label=Cell_name[i])
plt.xlabel("Freq")
plt.ylabel("Z_Imaginary(mOhm)")
plt.tight_layout()

''' Data PreProcessing '''
# Make new dataframe by frequency
freq = list((data[(Cell_name[0])]["Freq"]))
freq_df = {}
for f in freq:
    dummy = {}
    for i in range(len(Cell_name)):
        dummy[i] = data[(Cell_name[i])][data[(Cell_name[i])]["Freq"] == f][["Cell Name", "Freq", "Zre (mohm)", "Zim (mohm)", "Mag (mohm)", "Phase", "P/F"]].reset_index(drop=True)
        if i == 0:
            freq_df[f] = data[(Cell_name[i])][data[(Cell_name[i])]["Freq"] == f][["Cell Name", "Freq", "Zre (mohm)", "Zim (mohm)", "Mag (mohm)", "Phase", "P/F"]].reset_index(drop=True)
        else:
            freq_df[f] = pd.concat([freq_df[f], dummy[i]], axis=0, ignore_index=True)


# Calculate to bin
n = len(data)
sturges_bins = int(np.ceil(np.log2(n) + 1)) # 1. Sturges' Formula
sqrt_bins = int(np.sqrt(n)) # Square-root Choice


# Visualization Feqtures's histogram
use_ft = ["Zre (mohm)", "Zim (mohm)", "Mag (mohm)", "Phase"]
slt_freq = freq[-11]
fig, axs = plt.subplots(ncols=2, nrows=2)
for i, ft in enumerate(use_ft):
    row = int(i / 2)
    col = i % 2
    kde_data = (freq_df[slt_freq][ft])
    axs[row][col].hist(kde_data, bins=7, color='blue', edgecolor='black', density=True)
    kde = stats.gaussian_kde(kde_data)
    x_vals = np.linspace(kde_data.min(), kde_data.max(), 100)
    axs[row][col].plot(x_vals, kde(x_vals), color='red', lw=2)
    axs[row][col].set_xlabel(ft)
    axs[row][col].set_ylabel('Density')
fig.suptitle('Frequency Point at '+str(slt_freq)+"Hz")
plt.tight_layout()



''' Machine Learning Model '''
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Set Feature & Target
Model_data = freq_df[freq[-11]]
Feature = Model_data.drop(["Cell Name", "Freq", "P/F"], axis=1, inplace=False)
Target = Model_data["P/F"]

# Split Train and Test set
X_train, X_test, y_train, y_test = train_test_split(Feature, Target, test_size=0.3 , random_state= 100)

# Use Random Forest Classifier Model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
rf_clf.fit(X_train , y_train)
pred = rf_clf.predict(X_test)
accuracy = cross_val_score(rf_clf, Feature, Target, scoring="accuracy", cv = 5)
print('예측 정확도: {0:.4f}'.format(np.mean(accuracy)))

# check the feature importance
ftr_importances_values = rf_clf.feature_importances_
ftr_importances = pd.Series(ftr_importances_values,index=X_train.columns).sort_values(ascending=False)

# Visualization the feature importance
plt.figure()
plt.title('Feature importance')
sns.barplot(x=ftr_importances , y = ftr_importances.index, palette="Spectral")
plt.tight_layout()


# Adjust the hyperparameter using GridserchCV
params = {'n_estimators': [1000], 'max_depth': [8, 16, 24], 'min_samples_leaf' : [1, 6, 12], 'min_samples_split' : [2, 8, 16]}
grid_cv = GridSearchCV(rf_clf , param_grid=params , cv=5, n_jobs=-1 )
grid_cv.fit(X_train , y_train)
print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))

# Apply best hyperparameter from GridserchCV
rf_clf_gscv = RandomForestClassifier(n_estimators=1000, max_depth=8, min_samples_leaf=1, min_samples_split=2, random_state=0, n_jobs=-1)
accuracy = cross_val_score(rf_clf_gscv, Feature, Target, scoring="accuracy", cv = 5).mean()
pred = rf_clf_gscv.fit(X_train, y_train).predict(X_test)
print('예측 정확도: {0:.4f}'.format(accuracy_score(y_test , pred)))

# check the feature importance
ftr_importances_values = rf_clf_gscv.feature_importances_
ftr_importances = pd.Series(ftr_importances_values,index=X_train.columns).sort_values(ascending=False)

# Visualization the feature importance
plt.figure(figsize=(8,6))
plt.title('Feature importance')
sns.barplot(x=ftr_importances , y = ftr_importances.index, palette="Spectral")
plt.tight_layout()