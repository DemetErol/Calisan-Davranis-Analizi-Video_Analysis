# Video-Analysis

Haar-Cascade ile yüz tespiti yapabilmek için önce videoda olabilecek kişilerin fotoğraflarının elimizde olması gerekiyor. Proje gereği şirket çalışanlarının fotoğrafları 
isimleri ile kaydedilmeli. Input videosunun her bir frameini video devam ettiği sürece kontrol etmeliyiz.

Yüz tespitinin ardından, toplantıya katılan çalışanların cinsiyetleri, toplantıda kalma sürelerini, ne kadar zaman toplantıda söz sahibi oldukları bulundu.
Bunun ardından bu süreçteki duygu analizlerini yapıldı. Bu şekilde bir şirketteki çalışanların detaylı analizi çıkarılmış ve bu çalışan analizleri ile insan kaynakları
bölümüne, toplantıya katılan kadın erkek oranlarını, konuşma sürelerini ve çalışanlarının şirket içi mutlulukları gibi gelişim sağlayabilmeleri için önemli verileri
verilebilir. Cinsiyet tespiti için hazır eğitilmiş bir model olan gender_net.caffemodel kullanıldı. 

 ![image](https://user-images.githubusercontent.com/45537416/120336777-5abf5300-c2fb-11eb-80ba-150bb99eeb0e.png)
 
Duygu analizleri, insanların yüz ifadelerinden çıkartarak belirlenir. Bu ise angry, disgust, fear, happy, sad, surprise, neutral olarak 6 ayrı grupta toplanır. İnsanların
duygularını, bu duygunun kaçıncı frame içerisinde yer aldığını, videoda bulundukları süreleri ve konuşma süreleri gibi bilgileri txt dosyası ile saklandı. Bunun için
yeni bir klasör oluşturulup içine karışıklık olmaması için zaman ile isimlendirerek kaydedildi. Txt dosyasında sırasıyla isim, videonun saniye bilgisi, kaçıncı frame'de olduğu, duygu durumu, o ana kadarki konuşma süresi listelenmektedir.

![image](https://user-images.githubusercontent.com/45537416/120336873-76c2f480-c2fb-11eb-9ddc-401ceb43ea19.png)

![image](https://user-images.githubusercontent.com/45537416/120336943-87736a80-c2fb-11eb-9a94-612c4d14d8a3.png)

