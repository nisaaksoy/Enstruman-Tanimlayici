# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 12:37:03 2025

@author: pc
"""

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, GlobalAveragePooling1D, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
ana_dizin = r"C:\Users\pc\Desktop\makine\Grup17_Yazlab1"

# Maksimum kare sayısını belirleme
TARGET_SHAPE = (100, 13)  # 100 zaman adımı, 13 MFCC katsayısı

# Her klasör için
for klasor, etiket in enstruman_klasorleri.items():
    klasor_yolu = os.path.join(ana_dizin, klasor)
    
    # Klasördeki her npy dosyasını yükleme
    for dosya_adi in os.listdir(klasor_yolu):
        if dosya_adi.endswith(".npy"):
            dosya_yolu = os.path.join(klasor_yolu, dosya_adi)
            mfcc = np.load(dosya_yolu)
            
            # MFCC verisini hedef boyuta getirme
            if mfcc.shape[1] > TARGET_SHAPE[1]:  # Çok fazla sütun varsa kes
                mfcc = mfcc[:, :TARGET_SHAPE[1]]
            elif mfcc.shape[1] < TARGET_SHAPE[1]:  # Eksik sütun varsa sıfır doldur
                padding_columns = np.zeros((mfcc.shape[0], TARGET_SHAPE[1] - mfcc.shape[1]))
                mfcc = np.hstack((mfcc, padding_columns))

            # Boyut eşitleme (satır bazında)
            if mfcc.shape[0] > TARGET_SHAPE[0]:  # Çok uzun ise kes
                mfcc = mfcc[:TARGET_SHAPE[0], :]
            elif mfcc.shape[0] < TARGET_SHAPE[0]:  # Çok kısa ise sıfır doldur
                padding_rows = np.zeros((TARGET_SHAPE[0] - mfcc.shape[0], TARGET_SHAPE[1]))
                mfcc = np.vstack((mfcc, padding_rows))
            
            # Veriyi ve etiketi listelere ekleyin
            mfcc_data.append(mfcc.flatten())  # Düzleştirilmiş hale getirildi
            labels.append(etiket)

# Eğitim için kullanabileceğiniz veri ve etiketler
mfcc_data = np.array(mfcc_data)
labels = np.array(labels)

# Etiketleri sayısal kodlama
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(mfcc_data, encoded_labels, test_size=0.2, random_state=42)

# SMOTE ile veri dengesizliğini giderme
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Transformer modeline uygun hale getirme (veri boyutlarını uyumlu hale getirme)
X_train_smote = X_train_smote.reshape((X_train_smote.shape[0], TARGET_SHAPE[0], TARGET_SHAPE[1]))
X_test = X_test.reshape((X_test.shape[0], TARGET_SHAPE[0], TARGET_SHAPE[1]))

# Functional API ile model
input_layer = Input(shape=(TARGET_SHAPE[0], TARGET_SHAPE[1]))

# Encoder layers (Transformer Encoder blokları)
attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(input_layer, input_layer)
x = GlobalAveragePooling1D()(attention_output)

# Fully connected layer
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer
output_layer = Dense(4, activation='softmax')(x)

# Modeli oluşturma
model = Model(inputs=input_layer, outputs=output_layer)

# Modeli derleme
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(X_train_smote, y_train_smote, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Modeli değerlendirme
y_pred = np.argmax(model.predict(X_test), axis=1)

# Performans metriklerini yazdırma
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Karmaşıklık matrisi
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Karmaşıklık Matrisi")
plt.colorbar()
tick_marks = np.arange(4)
plt.xticks(tick_marks, label_encoder.classes_)
plt.yticks(tick_marks, label_encoder.classes_)
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.show()

# Eğitim ve kayıp grafikleri
plt.figure()
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Test Doğruluğu')
plt.title('Doğruluk Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.figure()
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Test Kaybı')
plt.title('Kayıp Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# Eğitim ve çıkarım sürelerini yazdırma
import time
start_time = time.time()
model.predict(X_test)  # Çıkarım işlemi
inference_time = time.time() - start_time
print(f"Çıkarım süresi: {inference_time:.4f} saniye")

start_time = time.time()
model.fit(X_train_smote, y_train_smote, epochs=10, batch_size=32)  # Eğitim işlemi
training_time = time.time() - start_time
print(f"Eğitim süresi: {training_time:.4f} saniye")
