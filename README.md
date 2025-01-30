# Enstruman Tanimlayici
 Şarkılardaki Enstrümanların Tespiti

Müzik Enstrümanlarının Tespiti Projesi

Proje Hakkında

Bu proje, çeşitli müzik enstrümanlarının (piyano, keman, gitar, kanun) ses verilerinin toplanması, işlenmesi ve sınıflandırılması amacıyla geliştirilmiştir. Proje, ses dosyalarının toplanmasından önceki ve sonraki işlemleri, yani sessizlik temizleme, gürültü azaltma, MFCC (Mel-Frekans Kepstrum Katsayıları) çıkarımı ve etiketleme gibi adımları kapsamaktadır. Elde edilen verilerle çeşitli yapay zeka modelleri eğitilmiştir.
Şarkılardaki Enstrümanların Tespiti

Veri Toplama
Veri toplama aşamasında, her enstrümana ait ses verileri YouTube çalma listelerinden indirilmiştir. Her enstrümana özel ses dosyaları MP3 formatında kaydedilmiş ve her enstrüman için ayrı klasörlerde saklanmıştır.
Araçlar:
yt_dlp kütüphanesi kullanılarak YouTube playlist'lerinden ses dosyaları indirilmiştir.
İşlem:
Çalma listelerindeki videolar, MP3 formatında indirilip ilgili klasörlere kaydedilmiştir.

Veri Ön İşleme
Veri ön işleme aşamasında ses dosyalarındaki gereksiz ses ve gürültüler temizlenmiş, ardından her ses dosyasından MFCC özellikleri çıkarılmıştır.
İşlemler:
Sessizlik Temizleme: pydub kütüphanesi kullanılarak sessiz bölümler temizlenmiştir.
Gürültü Temizleme: noisereduce kütüphanesi ile arka plan gürültüsü azaltılmıştır.
Normalizasyon: Ses sinyali, sabit bir ses seviyesi için normalize edilmiştir.
Özellik Çıkarımı Her ses dosyası, 3 saniyelik segmentlere bölünmüş ve her segmentten 13 MFCC özelliği çıkarılmıştır. MFCC, ses verisinin frekans özelliklerini temsil eden ve özellikle ses tanıma uygulamalarında yaygın olarak kullanılan bir tekniktir.
Veri Segmentasyonu: Ses dosyaları, her biri 3 saniyelik parçalara ayrılmıştır.
Etiketleme: Her segment, ait olduğu enstrümana göre etiketlenmiştir (piyano, keman, gitar, kanun).
Sonuçta, yüksek kaliteli ve etiketlenmiş bir veri seti elde edilmiştir ve bu veri seti, yapay zeka modelleri için kullanılmak üzere hazırlanmıştır.
Toplanan ve işlenen verilerle çeşitli yapay zeka modelleri ile eğitilmiştir. Bu modeller, müzik enstrümanlarının doğru bir şekilde sınıflandırılması için kullanılmıştır.

Kullanılan Modeller

Audio Spectrogram Transformer (AST)
AST modeli, ses verilerini analiz etmek için Transformer tabanlı bir yapıyı kullanır. Bu model, enstrümanları sınıflandırma konusunda oldukça başarılı olmuştur.
Accuracy: 80.34%
Precision: 80.46%
Recall: 80.34%
F1-Score: 80.14%
AUC: 0.8626
Eğitim Süresi: 1009.95 saniye
Çıkarım Süresi: 0.87 saniye

Data2Vec
Data2Vec, ses verisinden anlamlı özellikler çıkarmak için kullanılan bir başka güçlü transformatör modelidir. Bu model de yüksek doğruluk oranları ve başarılı performans gösterdi.
Accuracy: 80.61%
Precision: 80.81%
Recall: 80.61%
F1-Score: 80.43%
AUC: 0.8637
Eğitim Süresi: 1015.96 saniye
Çıkarım Süresi: 0.91 saniye

Wave2Vec
Wave2Vec, ses verisini doğrudan ham formatta işler ve özellikle doğal dil işleme alanında başarılı sonuçlar verir. Bu model, en yüksek doğruluk oranını elde etmiştir.
Accuracy: 82.32%
Precision: 82.55%
Recall: 82.32%
F1-Score: 81.87%
AUC: 0.9594
Eğitim Süresi: 776.12 saniye
Çıkarım Süresi: 0.95 saniye

Hubert Modeli
Hubert modeli, ses verisinin daha derin özelliklerini öğrenme yeteneği ile dikkat çeker. Bu model de ses sınıflandırma görevlerinde etkili bir performans göstermektedir.
Accuracy: 80.99%
Precision: 81.53%
Recall: 80.99%
F1-Score: 81.04%
AUC: 95.01%
Average Sensitivity: 52.78%
Average Specificity: 54.90%
Eğitim Süresi: 924.04 saniye
Çıkarım Süresi: 1.28 saniye

Conformer Modeli
Conformer (Convolutional Transformer), dil modellemesinde yaygın olarak kullanılan Transformer yapısının ses verisiyle daha etkili çalışabilmesi için iyileştirilmiş bir versiyonudur. Ses tanıma ve sıralı verilerle çalışmada yüksek başarı elde etmiştir. Modelin eğitimi, MFCC öznitelikleriyle temsil edilen ses dosyalarından oluşturulmuştur.
Accuracy: 0.8792
Precision: 0.8798
Recall: 0.8792
F1-Score: 0.8767
AUC: 0.9097
Average Sensitivity: 0.8628
Average Specificity: 0.9565
Eğitim Süresi: 225.01 saniye
Çıkarım Süresi: 0.3522 saniye
Conformer modelinin temel yapı taşı, evrişimsel (Convolutional) katmanlar ve Transformer yapısının birleşimidir. Bu mimari, ses verisindeki önemli özellikleri çıkarmak için evrişimsel katmanları kullanırken, Transformer yapısı sıralı verilerdeki ilişkileri öğrenmek için kullanılır. Modelin eğitimi, 50 epoch boyunca, 32 örnekten oluşan batch'ler ile gerçekleştirilmiştir.
Kullanılan Yöntemler
SMOTE (Synthetic Minority Over-sampling Technique): Veri setinde görülen dengesizliği gidermek için SMOTE tekniği uygulanmıştır. Bu sayede, her sınıf için daha dengeli bir veri seti oluşturulmuştur.
Transformer Tabanlı Modeller: Bu modeller, ses sinyalleri gibi sıralı veriler üzerinde dikkat mekanizmaları kullanarak öğrenme yapar. Bu sayede, ses verisindeki önemli özellikler çıkarılır ve doğru sınıflandırma yapılır.
MFCC Özellik Çıkarımı: Her bir ses dosyasından öznitelikleri çıkarılarak, enstrümanların özellikleri sayısal verilere dönüştürülmüştür. Bu özellikler, modelin ses verisini anlamasında önemli rol oynamaktadır.
