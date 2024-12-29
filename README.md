# Enstruman Tanimlayici
 Şarkılardaki Enstrümanların Tespiti

Müzik Enstrümanlarının Tespiti Projesi

Proje Hakkında

Bu proje, çeşitli müzik enstrümanlarının (piyano, keman, gitar, kanun) ses verilerinin toplanması, işlenmesi ve sınıflandırılması amacıyla geliştirilmiştir. Proje, ses dosyalarının toplanmasından önceki ve sonraki işlemleri, yani sessizlik temizleme, gürültü azaltma, MFCC (Mel-Frekans Kepstrum Katsayıları) çıkarımı ve etiketleme gibi adımları kapsamaktadır. Elde edilen verilerle çeşitli makine öğrenmesi modelleri eğitilmiş ve en iyi performansı sergileyen model, Random Forest Regressor seçilmiştir. Ayrıca, kullanıcıların ses dosyalarını yükleyip, bu dosyalar üzerinde enstrüman tahmini yapmalarını sağlayacak bir grafiksel kullanıcı arayüzü (GUI) geliştirilmiştir.

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

Özellik Çıkarımı

Her ses dosyası, 3 saniyelik segmentlere bölünmüş ve her segmentten 13 MFCC özelliği çıkarılmıştır. MFCC, ses verisinin frekans özelliklerini temsil eden ve özellikle ses tanıma uygulamalarında yaygın olarak kullanılan bir tekniktir.

Veri Segmentasyonu: Ses dosyaları, her biri 3 saniyelik parçalara ayrılmıştır.

Etiketleme: Her segment, ait olduğu enstrümana göre etiketlenmiştir (piyano, keman, gitar, kanun).

Sonuçta, yüksek kaliteli ve etiketlenmiş bir veri seti elde edilmiştir ve bu veri seti, makine öğrenmesi modelleri için kullanılmak üzere hazırlanmıştır.

Makine Öğrenmesi Modelleri

Toplanan ve işlenen verilerle çeşitli makine öğrenmesi modelleri eğitilmiştir. Bu modeller, müzik enstrümanlarının doğru bir şekilde sınıflandırılması için kullanılmıştır.

Model Listesi

Linear Regression (Doğrusal Regresyon): Eğitim süresi hızlı ancak doğrusal olmayan ilişkilerde performansı sınırlıdır.

Ridge Regression (Ridge Regresyonu): Multikollineerlik durumunda başarılıdır ve aşırı uyum yapmayı engeller.

Lasso Regression (Lasso Regresyonu): Değişken seçimi için güçlüdür, ancak doğrusal olmayan ilişkilerde sınırlıdır.

Support Vector Regressor (Destek Vektör Regresyonu): Karmaşık veri setlerinde yüksek doğruluk sağlar.

Random Forest Regressor (Rastgele Orman Regresyonu): Çoklu karar ağaçlarıyla yüksek doğruluk ve düşük aşırı uyum sağlar.

Sonuçlar

En İyi Performansı Gösteren Model: Random Forest Regressor.

MSE: 0.0726

R2 Skoru: 0.9491

Eğitim Zamanı: 13.9364 saniye

Çıkarım Zamanı: 0.0157 saniye

Random Forest Regressor, diğer modellere göre çok daha iyi sonuçlar elde etmiştir ve en doğru sınıflandırma sonuçlarını vermektedir.

Arayüz
Eğitilen modellerin ardından, kullanıcı etkileşimini kolaylaştırmak amacıyla bir grafiksel kullanıcı arayüzü (GUI) geliştirilmiştir. Tkinter tabanlı bu GUI, kullanıcıların ses dosyalarını yükleyip, bu dosyalar üzerinde enstrüman tahmini yapmalarını sağlar.

İşleyiş

1-Kullanıcı ses dosyasını yükler.

2-Sistem, dosyayı işler, gerekli öznitelikleri çıkarır ve kaydedilen Random Forest modelini kullanarak tahmin yapar.

3-Sonuç olarak, kullanıcıya hangi enstrümanın çaldığına dair bir bilgi sunulur.
