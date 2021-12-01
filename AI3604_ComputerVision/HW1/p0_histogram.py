# %%
import os
import cv2
import matplotlib.pyplot as plt

# %%
img_name = 'many_objects_2'
img_path = os.path.join('./data', f'{img_name}.png')
img = cv2.imread(img_path)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.hist(gray_image.flatten(), bins=25)
plt.title(f'Histogram for {img_name}')
plt.savefig(f'./output/{img_name}_histogram.pdf', bbox_anchor='tight')
plt.show()
