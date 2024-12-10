import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Generate artificial data with increased mean distance
np.random.seed(42)
data_size = 500

# Group O: Mean 50, Std 10
A_O = np.random.normal(50, 10, size=data_size // 2)
B_O = ['O'] * (data_size // 2)

# Group X: Mean 90, Std 20 (Larger mean distance, higher variance)
A_X = np.random.normal(90, 20, size=data_size // 2)
B_X = ['X'] * (data_size // 2)

# Combine data
A = np.concatenate([A_O, A_X])
B = np.concatenate([B_O, B_X])
df = pd.DataFrame({'A': A, 'B': B})

# 2. Standardize A
A_standardized = (df['A'] - df['A'].mean()) / df['A'].std()
df['A_standardized'] = A_standardized

# Separate standardized data
A_O_std = df[df['B'] == 'O']['A_standardized']
A_X_std = df[df['B'] == 'X']['A_standardized']

# 3. Calculate mean and std for both groups
mu_O, sigma_O = A_O_std.mean(), A_O_std.std()
mu_X, sigma_X = A_X_std.mean(), A_X_std.std()

# 4. Generate x values for plotting
x = np.linspace(-3, 3, 1000)

# 5. Calculate PDFs for both groups
pdf_O = norm.pdf(x, mu_O, sigma_O)
pdf_X = norm.pdf(x, mu_X, sigma_X)

# 6. Calculate intersection
intersection = np.minimum(pdf_O, pdf_X)

# 7. Plot
plt.figure(figsize=(10, 6))

# PDFs
plt.plot(x, pdf_O, color='blue')
plt.plot(x, pdf_X, color='red')

# Fill areas
plt.fill_between(x, pdf_O, intersection, where=(pdf_O > intersection), color='blue', alpha=0.3, label='Pass group region')
plt.fill_between(x, pdf_X, intersection, where=(pdf_X > intersection), color='red', alpha=0.3, label='Fail group region')
plt.fill_between(x, intersection, color='green', alpha=0.3, label='Intersection region')

# Graph styling
# plt.title('Histogram and Probability Density Function of A (Standardized)')
# plt.xlabel('Features', fontweight="bold")
# plt.ylabel('Probability Density', fontweight="bold")
plt.xticks([])
plt.yticks([])
# plt.legend(loc="upper left", fontsize=12)
plt.tight_layout()




# 8. Calculate the intersection point(s)
# 8. 정확한 교차점 탐색 (중앙 교차점 찾기)
diff = pdf_O - pdf_X
sign_change = np.diff(np.sign(diff))  # 두 함수의 부호 변화 확인
intersection_indices = np.where(sign_change)[0]  # 모든 교차점의 인덱스 가져오기

# 교차점이 여러 개인 경우 중앙 교차점 선택
if len(intersection_indices) > 1:
    middle_index = len(intersection_indices) // 2
    intersection_index = intersection_indices[middle_index]  # 중앙 교차점
else:
    intersection_index = intersection_indices[0]  # 단일 교차점

# Extract the x value of the intersection point
intersection_point = x[intersection_index]

# Plot updated visualization for both intersection and non-overlapping areas
plt.figure(figsize=(10, 6))

# PDFs
plt.plot(x, pdf_O, color='blue')
plt.plot(x, pdf_X, color='red')

# Calculate intersection (minimum values between two PDFs)
intersection = np.minimum(pdf_O, pdf_X)

# Fill non-overlapping areas
# Non-overlapping area for Group O (where pdf_O > pdf_X)
plt.fill_between(
    x, intersection, pdf_O, where=(pdf_O > pdf_X),
    color='blue', alpha=0.3, label='Pass group region'
)


# Non-overlapping area for Group X (where pdf_X > pdf_O)
plt.fill_between(
    x, intersection, pdf_X, where=(pdf_X > pdf_O),
    color='red', alpha=0.3, label='Fail group region'
)

# Highlight the intersection point
plt.scatter([intersection_point], [pdf_O[intersection_index]], color='black', label='Intersection Point', zorder=5)


# Fill intersection areas
# Left of the intersection point
plt.fill_between(
    x, 0, intersection, where=(x < intersection_point),
    color='green', alpha=0.3, label='Intersection region on Pass'
)

# Right of the intersection point
plt.fill_between(
    x, 0, intersection, where=(x > intersection_point),
    color='orange', alpha=0.3, label='Intersection region on Fail'
)

# Graph styling
# plt.xlabel('Features', fontweight="bold")
# plt.ylabel('Probability Density', fontweight="bold")
plt.xticks([])
plt.yticks([])
# plt.legend(loc="upper left", fontsize=10)
# plt.title("PDFs of Group O and X with Full Visualization", fontweight="bold")
plt.tight_layout()





import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon, gamma, beta, lognorm, weibull_min

# Generate x values for plots
x_norm = np.linspace(-4, 4, 1000)
x_uniform = np.linspace(0, 1, 1000)
x_expon = np.linspace(0, 5, 1000)
x_gamma = np.linspace(0, 10, 1000)
x_beta = np.linspace(0, 1, 1000)
x_lognorm = np.linspace(0.01, 5, 1000)
x_weibull = np.linspace(0, 3, 1000)

# PDF calculations for each distribution
pdf_norm = norm.pdf(x_norm, loc=0, scale=1)
pdf_uniform = uniform.pdf(x_uniform, loc=0, scale=1)
pdf_expon = expon.pdf(x_expon, scale=1)
pdf_gamma = gamma.pdf(x_gamma, a=2, scale=2)
pdf_beta = beta.pdf(x_beta, a=2, b=5)
pdf_lognorm = lognorm.pdf(x_lognorm, s=0.5, scale=np.exp(0))
pdf_weibull = weibull_min.pdf(x_weibull, c=1.5)

# Create subplots for each distribution
distributions = [
    ("Normal Distribution", x_norm, pdf_norm),
    ("Uniform Distribution", x_uniform, pdf_uniform),
    ("Exponential Distribution", x_expon, pdf_expon),
    ("Gamma Distribution", x_gamma, pdf_gamma),
    ("Beta Distribution", x_beta, pdf_beta),
    ("Log-Normal Distribution", x_lognorm, pdf_lognorm),
    ("Weibull Distribution", x_weibull, pdf_weibull)
]

for name, x, pdf in distributions:
    plt.figure(figsize=(6, 4))
    plt.plot(x, pdf, label=name)
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.legend()
