# %%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('seaborn')
matplotlib.style.use('seaborn-white')

time = pd.read_csv('./task1/time_domain_features.csv')
freq = pd.read_csv('./task1/freq_domain_features.csv')

# %%
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].scatter(
    range(500),
    time['Voiced Magnitude'],
    marker='o',
    label='Voiced',
    alpha=0.75
)
ax[0].scatter(
    range(500),
    time['Unvoiced Magnitude'],
    marker='x',
    label='Unvoiced',
    alpha=0.75
)
ax[0].set_title('Short-time Magnitude')
ax[0].legend()

ax[1].scatter(
    range(500),
    time['Voiced Energy'],
    marker='o',
    label='Voiced',
    alpha=0.75
)
ax[1].scatter(
    range(500),
    time['Unvoiced Energy'],
    marker='x',
    label='Unvoiced',
    alpha=0.75
)
ax[1].set_title('Short-time Energy')
ax[1].legend()

ax[2].scatter(
    range(500),
    time['Voiced ZCR'],
    marker='o',
    label='Voiced',
    alpha=0.75
)
ax[2].scatter(
    range(500),
    time['Unvoiced ZCR'],
    marker='x',
    label='Unvoiced',
    alpha=0.75
)
ax[2].set_title('Short-time ZCR')
ax[2].legend()
fig.savefig('./time_features.pdf')

# %%
fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))
ax2[0].scatter(
    range(500),
    freq['Voiced LowFreq'],
    marker='o',
    label='Voiced',
    alpha=0.75
)
ax2[0].scatter(
    range(500),
    freq['Unvoiced LowFreq'],
    marker='x',
    label='Unvoiced',
    alpha=0.75
)
ax2[0].set_title('Low Frequency Energy')
ax2[0].legend()

ax2[1].scatter(
    range(500),
    freq['Voiced MedFreq'],
    marker='o',
    label='Voiced',
    alpha=0.75
)
ax2[1].scatter(
    range(500),
    freq['Unvoiced MedFreq'],
    marker='x',
    label='Unvoiced',
    alpha=0.75
)
ax2[1].set_title('Medium Frequency Energy')
ax2[1].legend()

ax2[2].scatter(
    range(500),
    freq['Voiced HighFreq'],
    marker='o',
    label='Voiced',
    alpha=0.75
)
ax2[2].scatter(
    range(500),
    freq['Unvoiced HighFreq'],
    marker='x',
    label='Unvoiced',
    alpha=0.75
)
ax2[2].set_title('High Frequency Energy')
ax2[2].legend()
fig2.savefig('./freq_features.pdf')
