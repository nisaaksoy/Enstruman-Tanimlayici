# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:10:28 2025

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
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, auc)
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

# Her klasör için
for klasor, etiket in enstruman_klasorleri.items():
    klasor_yolu = os.path.join(ana_dizin, klasor)
    
    for dosya_adi in os.listdir(klasor_yolu):
        if dosya_adi.endswith(".npy"):
            dosya_yolu = os.path.join(klasor_yolu, dosya_adi)
            mfcc = np.load(dosya_yolu)
            
            if mfcc.shape[1] > TARGET_SHAPE[1]:
                mfcc = mfcc[:, :TARGET_SHAPE[1]]
            elif mfcc.shape[1] < TARGET_SHAPE[1]:
                padding_columns = np.zeros((mfcc.shape[0], TARGET_SHAPE[1] - mfcc.shape[1]))
                mfcc = np.hstack((mfcc, padding_columns))

            if mfcc.shape[0] > TARGET_SHAPE[0]:
                mfcc = mfcc[:TARGET_SHAPE[0], :]
            elif mfcc.shape[0] < TARGET_SHAPE[0]:
                padding_rows = np.zeros((TARGET_SHAPE[0] - mfcc.shape[0], TARGET_SHAPE[1]))
                mfcc = np.vstack((mfcc, padding_rows))
            
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

# Conformer Blok Tanımlama
def conformer_block(inputs, output_dim):
    x = Conv1D(filters=output_dim, kernel_size=3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Dilated Convolution layer
    x = Conv1D(filters=output_dim, kernel_size=3, dilation_rate=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Attention Block
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=output_dim)(x, x)
    x = Add()([x, attention])
    x = Dropout(0.1)(x)
    
    # Final Layer Normalization
    x = LayerNormalization()(x)
    return x

# Model oluşturma
input_layer = Input(shape=(TARGET_SHAPE[0], TARGET_SHAPE[1]))

x = conformer_block(input_layer, 64)
x = conformer_block(x, 128)
x = conformer_block(x, 256)

x = GlobalAveragePooling1D()(x)

x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)

output_layer = Dense(4, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modelin eğitim zamanı ölçümü
start_time = time.time()
history = model.fit(X_train_smote, y_train_smote, epochs=50, batch_size=32, validation_data=(X_test, y_test))
training_time = time.time() - start_time

# Modelin tahmin zamanı ölçümü
start_time = time.time()
y_pred_prob = model.predict(X_test)
inference_time = time.time() - start_time

# Tahminler
y_pred = np.argmax(y_pred_prob, axis=1)

# Performans metrikleri
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
print(f"AUC: {auc:.4f}")
print(f"Training Time: {training_time:.2f} seconds")
print(f"Inference Time: {inference_time:.2f} seconds")

# Karmaşıklık matrisi
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Karmaşıklık Matrisi")
plt.colorbar()
tick_marks = np.arange(len(label_encoder.classes_))
plt.xticks(tick_marks, label_encoder.classes_)
plt.yticks(tick_marks, label_encoder.classes_)
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.show()

# ROC Eğrileri
plt.figure()
for i, class_name in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_prob[:, i])
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc:.2f})")

plt.title("ROC Eğrisi")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Eğitim/Kayıp Grafikleri
plt.figure()
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Kayıp Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Doğruluk Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# Sınıf Bazlı Metrikler
for i, class_name in enumerate(label_encoder.classes_):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - (tp + fp + fn)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    print(f"\nSınıf: {class_name}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
