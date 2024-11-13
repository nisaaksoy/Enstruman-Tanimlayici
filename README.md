# Enstruman Tanimlayici
 Şarkılardaki Enstrümanların Tespiti

Enstrüman Sınıflandırma Projesi - Veri Toplama ve Ön İşleme Adımları

Bu README dosyası, müzik enstrümanlarını sınıflandırmak için kullanılan veri toplama ve ön işleme sürecini açıklamaktadır. Proje, ses dosyalarından Mel Frekansı Cepstral Katsayıları (MFCC) çıkararak her bir enstrümanı sınıflandırmak için kullanılabilir.

Veri Toplama
Veri toplama aşamasında, YouTube çalma listelerinden müzik parçaları indirilmektedir. Bu parçalar, farklı enstrümanları temsil etmektedir ( piyano, keman, gitar, kanun).

Adımlar:

YouTube Playlist İndirme: yt_dlp kütüphanesi kullanılarak belirtilen bir YouTube çalma listesindeki tüm videolar indirilir ve her video MP3 formatında kaydedilir.

Çalma listesi URL’si (playlist_url), ses dosyalarının indirileceği klasör (output_folder) ile belirtilir. yt_dlp araçlarıyla ses dosyaları indirilir ve MP3 formatında kaydedilir.

Veri Temizleme
Veri temizleme süreci, ses dosyalarındaki gereksiz sessiz bölümleri ve gürültüyü ortadan kaldırmayı amaçlamaktadır. Bu adımda pydub ve noisereduce gibi kütüphaneler kullanılmaktadır.

Adımlar:

Sessizlik Temizleme: pydub'un silence.detect_nonsilent fonksiyonu kullanılarak sessiz bölümler temizlenir. Bu sayede yalnızca anlamlı ses segmentleri kalır.

Gürültü Temizleme ve Normalizasyon: noisereduce kütüphanesi ile ses dosyalarındaki gürültü temizlenir ve librosa ile normalizasyon işlemi yapılır.,

MFCC Çıkarımı
Bu aşama, her ses dosyasından Mel Frekansı Cepstral Katsayıları (MFCC) çıkarır. MFCC, sesin temel özelliklerini temsil eder ve ses tanıma ve sınıflandırma için yaygın olarak kullanılır.

Adımlar:

MFCC Çıkarımı: librosa kütüphanesi kullanılarak her ses parçasından 13 MFCC katsayısı çıkarılır.

Segmente Etme: Ses dosyaları, 3 saniyelik segmentlere ayrılır. Her segment üzerinde gürültü temizleme, normalizasyon ve MFCC çıkarma işlemleri uygulanır.

Etiketleme ve Veri Hazırlığı
Bu adımda, her enstrüman için oluşturulmuş MFCC dosyaları etiketlenir ve model eğitimi için kullanılabilir hale getirilir.

Adımlar:

Veri ve Etiketleri Yükleme: Her bir enstrüman için MFCC dosyaları yüklenir ve her bir dosya uygun etiketle (gitar, kanun, keman, piyano) ilişkilendirilir.

Veri Hazırlığı: MFCC verileri ve etiketler, modelin eğitiminde kullanılacak formatta birleştirilir.

Proje İçeriği

Veri Toplama: YouTube çalma listesinden ses dosyalarının indirilmesi.

Veri Temizleme: Sessiz bölümlerin çıkarılması ve gürültü temizleme işlemleri.

MFCC Çıkarımı: Ses dosyalarından MFCC özelliklerinin çıkarılması.

Etiketleme: MFCC dosyaları, her enstrümana ait etiketlerle ilişkilendirilir.
