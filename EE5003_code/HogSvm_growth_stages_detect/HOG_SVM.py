import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# PosNum = 8*11
# NegNum = 8*11
pos_list = []
neg_list = []
other_list = []
gradient_list = []
labels = []

gamma = 0.8
hog_win_size = (100, 100)

def gamma_trans(img, gamma):   # Gamma correction
    gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, gamma_table)

def computeHOGs(img_list, gradient_list, hog_win_size): # Compute HOG features
    hog = cv2.HOGDescriptor((100,100), (10,10), (5,5), (5,5), 9)
    for img in img_list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = gamma_trans(gray, gamma)
        gradient_list.append(hog.compute(gray))
    return gradient_list

CLASS1_TRAIN = "./Lettuce_Growth_Stages_Database/train_filter/0/"
CLASS2_TRAIN = "./Lettuce_Growth_Stages_Database/train_filter/1/"
CLASS3_TRAIN = "./Lettuce_Growth_Stages_Database/train_filter/2/"
# Read (seedling) images
# for filename in os.listdir("./train/0"):
for filename in os.listdir(CLASS1_TRAIN):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = cv2.imread(os.path.join(CLASS1_TRAIN, filename))
        img = cv2.resize(img, hog_win_size)
        pos_list.append(img) 
        labels.append(0)  # Seedling class

# Read other (growth) images
for filename in os.listdir(CLASS2_TRAIN):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = cv2.imread(os.path.join(CLASS2_TRAIN, filename))
        img = cv2.resize(img, hog_win_size)
        other_list.append(img) 
        labels.append(2)  # Other class

# Read (mature) images
for filename in os.listdir(CLASS3_TRAIN):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = cv2.imread(os.path.join(CLASS3_TRAIN, filename))
        img = cv2.resize(img, hog_win_size)
        neg_list.append(img) 
        labels.append(1)  # Mature class



# Compute HOG features
computeHOGs(pos_list + neg_list + other_list, gradient_list, hog_win_size)

# # Train SVM
# svm = cv2.ml.SVM_create()
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setGamma(0.001)
# svm.setC(30) #30
# svm.setKernel(cv2.ml.SVM_RBF)
# svm.train(np.array(gradient_list), cv2.ml.ROW_SAMPLE, np.array(labels))

# # Save SVM model
# svm.save("svm_model.xml")
# print('SVM model saved as svm_model.xml!')
# 创建 SVM 分类器

svm_classifier = SVC(kernel='linear', C=1.0)

# 在训练集上训练 SVM 分类器
svm_classifier.fit(np.array(gradient_list), np.array(labels))
joblib.dump(svm_classifier, 'SVM_Detect_growth_stages.pkl')
print('SVM model saved as SVM_Detect_growth_stages.pkl!')
# # 获取测试集上的决策函数输出值
# decision_values = svm_classifier.decision_function(X_test)
# print(decision_values)
# # 取每个样本中最大的输出值作为置信度分数
# confidence_scores = np.max(decision_values, axis=1)

# # 显示置信度分数
# for i, confidence_score in enumerate(confidence_scores):
#     print('Confidence Score for Sample {}: {:.2f}'.format(i, confidence_score))

# # 在测试集上进行预测
# y_pred = svm_classifier.predict(X_test)
# print('y_pred:', y_pred)
# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)
