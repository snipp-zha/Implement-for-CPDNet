import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

lab1 = 'D:/HZX/Program files/cache/1761170395/FileRecv/磁瓦缺陷数据集/磁瓦缺陷数据集/MT_Blowhole/Imgs/exp1_num_3667.png'
img1 = 'D:/HZX/Program files/cache/1761170395/FileRecv/磁瓦缺陷数据集/磁瓦缺陷数据集/MT_Blowhole/Imgs/exp1_num_3667.jpg'
img1 = np.array(Image.open(img1), dtype='float32') 
img1 = cv2.resize(img1, (512,512), interpolation=cv2.INTER_AREA) 
lab1 = np.array(Image.open(lab1), dtype='uint8') 
lab1 = cv2.resize(lab1, (512,512), interpolation=cv2.INTER_AREA) 
lab1[lab1>=128]=255
lab1[lab1<128]=0

lab2 = 'D:/HZX/Program files/cache/1761170395/FileRecv/磁瓦缺陷数据集/磁瓦缺陷数据集/MT_Blowhole/Imgs/exp1_num_4727.png'
img2 = 'D:/HZX/Program files/cache/1761170395/FileRecv/磁瓦缺陷数据集/磁瓦缺陷数据集/MT_Blowhole/Imgs/exp1_num_4727.jpg'
img2 = np.array(Image.open(img2), dtype='float32') 
img2 = cv2.resize(img2, (512,512), interpolation=cv2.INTER_AREA) 
lab2 = np.array(Image.open(lab2), dtype='uint8') 
lab2 = cv2.resize(lab2, (512,512), interpolation=cv2.INTER_AREA) 
lab2[lab2>=128]=255
lab2[lab2<128]=0

plt.figure(dpi=200)

# 提取不规则区域并将其填充为255，其他区域填充为0
_, thresh = cv2.threshold(lab1, 128, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours, hierarchy= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(lab1)
# cv2.drawContours(mask, contours, -1, 255, -1)
if len(contours) > 0:
    mask = np.zeros_like(lab1)
    cv2.drawContours(mask, contours, -1, 255, -1)
else:
    print("No contours found")
# 随机选择图像中的一个区域
y, x = np.where(mask == 255)
x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
x1 = x1-10 if x1-10>0 else x1
y1 = y1-10 if y1-10>0 else y1
roi_lab = lab1[y1:y1+150, x1:x1+150]
roi_img = img1[y1:y1+150, x1:x1+150]
# 随机缩放和旋转区域
k = np.random.randint(0, 4)
roi_lab_transformed = np.rot90(roi_lab, k)
roi_img_transformed = np.rot90(roi_img, k)
axis = np.random.randint(0, 2)
roi_lab_transformed = np.flip(roi_lab_transformed, axis=axis).copy()
roi_img_transformed = np.flip(roi_img_transformed, axis=axis).copy()

# # 将变换后的区域复制到图像的其他位置
x_offset, y_offset = np.random.randint(0, lab1.shape[1]-roi_lab_transformed.shape[0]), np.random.randint(0, lab1.shape[0]-roi_lab_transformed.shape[1])
lab_mask = np.zeros_like(lab1)
img_mask = np.zeros_like(img1)
img_mask_new = np.zeros_like(img1)
lab_mask[x_offset : x_offset+roi_lab_transformed.shape[0], y_offset:y_offset+roi_lab_transformed.shape[1]] = roi_lab_transformed
img_mask[x_offset : x_offset+roi_img_transformed.shape[0], y_offset:y_offset+roi_img_transformed.shape[1]] = roi_img_transformed

img1 = img1/255
img2 = img2/255
lab1 = lab1/255
lab2 = lab2/255
img_mask = img_mask/255
img_mask_new[img_mask>0]=1

lab_mask = lab_mask/255
plt.figure(dpi=200)
plt.subplot(241)
plt.imshow(img1,cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.subplot(242)
plt.imshow(lab1,cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.subplot(243)
plt.imshow(img2,cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.subplot(244)
plt.imshow(lab2,cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.subplot(245)
plt.imshow(img_mask,cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.subplot(246)
plt.imshow(lab_mask,cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.subplot(247)
plt.imshow((1-img_mask_new)*lab2+img_mask_new*lab_mask,cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.subplot(248)
plt.imshow((1-img_mask_new)*img2+img_mask_new*img_mask,cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.show()

