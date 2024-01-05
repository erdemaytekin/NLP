# PROJEDE EMEĞİ GEÇENLER;
__Erdem Aytekin 203405020 / Rıfat Efe Arslan 213405009__

# IMPORT
![Import](https://github.com/erdemaytekin/NLP/assets/116784884/c02b2c64-0242-4e25-a0e4-b5446a3adfa0)

__pandas as pd:__ Python programlama dilinde veri analizi ve manipülasyonu için kullanılan güçlü bir kütüphanedir.

__numpy as np:__ Bilimsel hesaplama ve sayısal işlemler için kullanılan bir kütüphanedir.

__string:__ Metin işleme işlemlerinde yardımcı olan karakter dizileri ve metin işlemleri için kullanılan öntanımlı sabitleri ve işlevleri içeren bir Python modülüdür.

__TfidfVectorizer:__ Metin verilerini TF-IDF (Term Frequency-Inverse Document Frequency) yöntemiyle vektörlere dönüştürerek metin madenciliği veya doğal dil işleme uygulamalarında kullanılan bir vektörleştirme aracıdır.

__text_lemmatizer:__ Metinlerdeki kelimelerin köklerini bulan ve kelimeleri temel anlamlarına indirgeyen bir işlemdir, bu sayede metin işleme süreçlerinde kelime çeşitliliğini azaltır ve analizde daha tutarlı sonuçlar elde etmeye yardımcı olur.

__sent_tokenize:__ Bir metni cümlelere ayıran bir doğal dil işleme aracıdır ve metinlerin cümle düzeyinde analiz edilmesini sağlar, genellikle NLP (Natural Language Processing) uygulamalarında kullanılır.

__stopwords:__ Genellikle dil işleme ve metin analizi süreçlerinde kullanılan, metinlerde sıkça geçen ve genellikle anlam taşımayan yaygın kelimeler listesidir; bu kelimeler genellikle analizlerden önce kaldırılır, çünkü anahtar bilgi taşımazlar.

__train_test_split:__ Veri setini eğitim ve test alt kümelerine ayırmak için kullanılan bir fonksiyondur ve makine öğrenimi modelinin performansını değerlendirmek için veri setini bölerek kullanılabilir.

__f1_score:__ Sınıflandırma problemlerinde bir modelin doğruluğunu ölçmek için kullanılan bir metriktir ve hassasiyet (precision) ile geri çağrı (recall) değerlerinin harmonik ortalamasını verir, dengeli bir performans değeri sağlar.

__MultinomialNB:__ Çok sınıflı sınıflandırma problemleri için kullanılan Naive Bayes sınıflandırıcısının bir türüdür ve özellikle metin sınıflandırma gibi kategorik verilerle çalışırken yaygın olarak tercih edilir.
# DATASET ve ETİKET OLUŞTURMAK
![DATASET](https://github.com/erdemaytekin/NLP/assets/116784884/02caca49-f2df-483d-9eb3-5ff6d73174d6)

Görseldeki kodumuzda requests ve BeautifulSoup kütüphaneleri kullanılarak URL'leri çekiyoruz.

Kodumuzun en altındaki for döngüsünde URL'ye split metodu uygulayarak URL'leri çekerken aynı zamanda etiketleri de oluşturmuş olduk.

Örnek  __https://www.ensonhaber.com/ekonomi/4__  URL'sinde / lere göre split ettiğimizde "ekonomi" kısmını da elde ediyoruz.
# 
![son](https://github.com/erdemaytekin/NLP/assets/116784884/8270d78a-8fa7-479e-ae16-f1ddd18037d8)

Bu kodumuzda da URL'lerde gezinip html kodunu parçalayarak tüm metinleri elde ediyoruz.

Sonrasındaysa elde ettiğimiz metinleri bigdata adlı listemize ekliyoruz ve listeyi pandas dataframe'ine dönüştürüyoruz.

Son olarakta aşağıdaki görselde dataframe'i csv dosyasını dönüştürerek veri seti oluşturma işlemimizi sonlandırıyoruz.

![bigdatason](https://github.com/erdemaytekin/NLP/assets/116784884/7e9b9c33-b998-404e-89cf-48030a955480)

# STOPWORDS
![stopwords](https://github.com/erdemaytekin/NLP/assets/116784884/7d33e0ad-454f-4a67-81d0-18999c478dc2)

*__Türkçe kelime olup genellikle anlam taşımayan yaygın kelimeleri gürültüyü azaltmak için bu kod satırıyla temizliyoruz.__*

# CLEAN TEXT
![cleantext](https://github.com/erdemaytekin/NLP/assets/116784884/17c84044-e041-4ffc-8334-e7b034d5a970)

__Bu fonksiyonumuzda, metin temizleme işlemlerini gerçekleştiriyoruz;__

__1)__ "Bu videoyu izlemek için lütfen JavaScript'i etkinleştirin" ve "[/inlinetweet]" gibi özel metinleri kaldırıyor.

__2)__ Metni küçük harfe dönüştürüyor (text.lower()).

__3)__ Rakamları kaldırıyor (re.sub("[0-9]+", "", text)).

__4)__ Özel karakterleri (çoğunlukla tırnak işaretleri ve tireler gibi) boşluklarla değiştiriyor (re.sub("’|“|”|‘|–|—", " ", text)).

__5)__ HTTP veya HTTPS ile başlayan URL'leri kaldırıyor (re.sub(r"https?:\/\/\S+", " ", text)).

__6)__ Türkçe metinleri lemmatize ediyor __(köklerini buluyor)__ (text_lemmatizer(text, lang="tr")).

__7)__ Metni kelime listesine ayırıyor ve stopWords adlı listemizin içindeki kelimeleri çıkarıyor ([word for word in text if word not in stopWords]).

__8)__ Türkçe noktalama işaretlerini kaldırıyor.

__9)__ Temizlenmiş metni döndürüyor.
#
![dfclean](https://github.com/erdemaytekin/NLP/assets/116784884/f1c20569-80b8-4e6d-a9ea-4e7b68e55342)

__Bu fonksiyonumuzda metnimizi temizliyoruz ve temizlenmiş metnimizi clean sütünuna ekliyoruz.__
#
![cleanmitext](https://github.com/erdemaytekin/NLP/assets/116784884/cf0d6b1a-a8b3-4be3-ac47-be944150af07)

__Metnimiz temizlenmiş mi diye kontrol ediyoruz ve evet metnimiz temizlenmiş.__
#
![cleanichaber](https://github.com/erdemaytekin/NLP/assets/116784884/16273f6e-a5c8-4eb6-b9b2-39a6acde3846)


__Etiketlerdeki noktalama işaretlerini silmek için kullandık.__
#
![x ve y](https://github.com/erdemaytekin/NLP/assets/116784884/266a7f85-5cec-4dfc-8d6a-2f0f91ef7bda)

__Bu kodumuzda, DataFrame'deki temizlenmiş metin verilerini X değişkenine ve temizlenmiş etiketleri y değişkenine dönüştürüyoruz.__

__df.clean.to_numpy():__ "clean" adlı sütundaki temizlenmiş metin verilerini bir NumPy dizisine dönüştürür ve X değişkenine atar. Burada "clean" sütunu, metinlerin temizlenmiş versiyonlarını içerir.

__df.clean_label.to_numpy():__ "clean_label" adlı sütundaki temizlenmiş etiketleri (labels) bir NumPy dizisine dönüştürür ve y değişkenine atar. "clean_label" sütunu, etiketlerin temizlenmiş versiyonlarını içerir.
# TRAIN , TEST
![test ve train](https://github.com/erdemaytekin/NLP/assets/116784884/b4e3dfe4-3eac-43ba-b37e-1b7b5bdb87a3)

__Bu fonksiyon, verilen girdi verilerini (X) ve hedef etiketlerini (y) eğitim ve test kümelerine böler.__

__test_size=0.2:__ Veri setinin ne kadarının test kümesinde olacağını belirtir. Burada veri setimizin %20'sini test kümesi olarak kullandık.
# MODELİN OLUŞTURULMASI
![vectorizor](https://github.com/erdemaytekin/NLP/assets/116784884/82c2cf12-d29c-43b6-8171-d15d6480cfa1)

__Sözcükleri vektörleştirmek için TFidfVectorizer kullandık.__
## NAIVE BAYES 
![büyük](https://github.com/erdemaytekin/NLP/assets/116784884/1483a531-d43d-4ac4-9f43-5e7441932c21)

* Eğittiğimiz Naive Bayes modelinin, eğitim ve test sonuçları görseldeki gibidir.Veri setimiz dengeli olduğundan dolayı başarım oranlarımız hemen hemen birbirine yakın gözükmektedir.

* Başarım oranımız tüm kategorilerde %85 ve üzeri olduğundan modelimiz başarılı kabul edilebilir.
  
* Başarım oranımız, veri setimiz daha büyük olsaydı artabilirdi.
# 
 ![sıon](https://github.com/erdemaytekin/NLP/assets/116784884/c3d9f3d3-d46d-40c6-ba07-19ae93f25728)

 __Üstteki görselde de başka bir haber metnini, eğittiğimiz modele tahmin ettirdik ve modelimiz doğru bildi.__










 










