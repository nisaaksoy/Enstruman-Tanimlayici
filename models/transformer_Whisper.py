# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:32:08 2025

@author: pc
"""

import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, Add, BatchNormalization, 
                                      Activation, Dropout, GlobalAveragePooling1D, Dense, LayerNormalization)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, roc_curve, auc)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Veriyi hazırlama
mfcc_data = []
labels = []

# Klasör isimleri ve etiketleri eşleştiren bir sözlük
enstruman_klasorleri = {
    "mfcc_cikti_gitar": "gitar",
    "mfcc_cikti_kanun": "kanun",
    "mfcc_cikti_keman": "keman",
    "mfcc_cikti_piyano": "piyano"
}

# Ana dizinin yolu
ana_dizin = r"C:\\Users\\pc\\Desktop\\makine\\Grup17_Yazlab1"

# Maksimum kare sayısını belirleme
TARGET_SHAPE = (100, 13)  # 100 zaman adımı, 13 MFCC katsayısı

# Verileri yükleme
for klasor, etiket in enstruman_klasorleri.items():
    klasor_yolu = os.path.join(ana_dizin, klasor)
    
    for dosya_adi in os.listdir(klasor_yolu):
        if dosya_adi.endswith(".npy"):
            dosya_yolu = os.path.join(klasor_yolu, dosya_adi)
            mfcc = np.load(dosya_yolu)
            
            # Veri boyutlarını ayarla
            mfcc = np.pad(mfcc, ((0, max(0, TARGET_SHAPE[0] - mfcc.shape[0])), 
                                  (0, max(0, TARGET_SHAPE[1] - mfcc.shape[1]))), 
                          mode='constant')[:TARGET_SHAPE[0], :TARGET_SHAPE[1]]
            
            mfcc_data.append(mfcc)
            labels.append(etiket)

mfcc_data = np.array(mfcc_data)
labels = np.array(labels)

# Etiketleri sayısal kodlama
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(mfcc_data, encoded_labels, test_size=0.2, random_state=42)

# SMOTE ile veri dengesizliğini giderme
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train.reshape((X_train.shape[0], -1)), y_train)
X_train_smote = X_train_smote.reshape((X_train_smote.shape[0], TARGET_SHAPE[0], TARGET_SHAPE[1]))
X_test = X_test.reshape((X_test.shape[0], TARGET_SHAPE[0], TARGET_SHAPE[1]))

# Hubert benzeri blok tanımlama
def hubert_block(inputs, output_dim):
    x = Conv1D(filters=output_dim, kernel_size=3, padding="same")(inputs)
    x = LayerNormalization()(x)
    x = Activation('relu')(x)
    shortcut = Dense(output_dim)(inputs)  # Boyut eşleştirme
    x = Add()([shortcut, x])
    x = Dropout(0.1)(x)
    return x

# Model oluşturma
input_layer = Input(shape=(TARGET_SHAPE[0], TARGET_SHAPE[1]))

x = hubert_block(input_layer, 64)
x = hubert_block(x, 128)
x = hubert_block(x, 256)

x = GlobalAveragePooling1D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

output_layer = Dense(4, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Eğitim zamanı ölçümü
start_time = time.time()
history = model.fit(X_train_smote, y_train_smote, epochs=50, batch_size=32, validation_data=(X_test, y_test))
training_time = time.time() - start_time

# Çıkarım zamanı ölçümü
start_time = time.time()
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
inference_time = time.time() - start_time

# Performans metriklerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
auc_score = roc_auc_score(tf.keras.utils.to_categorical(y_test, num_classes=4), y_pred_prob, multi_class='ovr')

# Karmaşıklık matrisi
cm = confusion_matrix(y_test, y_pred)

# Sensitivity ve Specificity hesaplama
sensitivity = np.diag(cm) / np.sum(cm, axis=1)
specificity = np.diag(cm) / np.sum(cm, axis=0)

# Sonuçları yazdırma
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC: {auc_score:.4f}")
print(f"Sensitivity: {np.mean(sensitivity):.4f}")
print(f"Specificity: {np.mean(specificity):.4f}")
print(f"Çıkarım Süresi: {inference_time:.4f} saniye")
print(f"Eğitim Süresi: {training_time:.4f} saniye")

# Grafikler
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Test Doğruluğu')
plt.legend()
plt.title('Doğruluk Eğrisi')
plt.show()

plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Test Kaybı')
plt.legend()
plt.title('Kayıp Eğrisi')
plt.show()

# ROC eğrisini çizme
fpr = {}
tpr = {}
roc_auc = {}

n_classes = 4  # Sınıf sayısı (gitar, kanun, keman, piyano)

# ROC eğrisini her sınıf için hesapla
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Mikro AUC (tüm sınıfların birleştirilmiş performansı)
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# ROC eğrisini çizme
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {label_encoder.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')

plt.plot(fpr["micro"], tpr["micro"], label=f'Micro Average (AUC = {roc_auc["micro"]:.2f})', linestyle='--')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Rastgele tahmin çizgisi
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi')
plt.legend(loc='lower right')
plt.show()

# Genel AUC değeri
print(f"Genel AUC: {roc_auc['micro']:.4f}")
