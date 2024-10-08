# etharium
 etharium stock price predict

1. Objective
The primary objective of this project is to build a Long Short-Term Memory (LSTM) model that predicts the future price of Ethereum based on historical data. The model is trained on past price data and is used to forecast the Ethereum price for the next 60 days.

2. Dataset Overview
The dataset used in this project contains daily Ethereum price data from 2018 to 2024. The key features of the dataset include:

time: Date of the observation.
Close: The closing price of Ethereum on that particular day, which is the primary variable used for prediction.
3. Data Preprocessing
The dataset was loaded into a pandas DataFrame and the 'time' column was converted to datetime format.
The 'time' column was then set as the index to facilitate easy plotting and manipulation of time-series data.
The MinMaxScaler was used to normalize the 'Close' price between 0 and 1, making the data suitable for LSTM models.
A sequence length of 60 days was chosen, meaning that the model would use the previous 60 days to predict the next day's closing price.
4. Splitting the Data
The dataset was split into two parts:
Training Set: 80% of the data was used to train the model.
Testing Set: The remaining 20% was used to evaluate the model's performance on unseen data.
5. LSTM Model Architecture
The LSTM model was built using the following layers:

Input Layer: The input shape corresponds to the sequence length of 60 days.
First LSTM Layer: This layer has 50 units and uses return_sequences=True to pass the sequence to the next LSTM layer.
Second LSTM Layer: A second LSTM layer with 50 units was added without returning the sequences, followed by a Dropout layer to prevent overfitting.
Dense Layers: Two dense layers were added to output the final prediction, with one unit in the final layer as we're predicting a single value (price).
6. Training the Model
The model was trained using the Adam optimizer and mean_squared_error as the loss function for 70 epochs with a batch size of 84. The model was validated using the test data, ensuring the learning process was generalized.

7. Model Evaluation
The model was evaluated using the Mean Squared Error (MSE) on the test data:

Mean Squared Error: The MSE for the model was computed and represents the average of the squared differences between actual and predicted prices. This metric helps measure how well the model fits the data.
Visualization: The model's performance was visualized by plotting the actual Ethereum prices vs. the predicted prices. The plot clearly shows how well the model predicted both the historical and future prices.
8. Future Price Prediction
Using the trained LSTM model, predictions for the next 60 days were made. The prediction process involved:

Taking the last 60 days of the available data.
Scaling and feeding it into the LSTM model.
Iteratively predicting the next day’s price, adding the prediction to the sequence, and repeating this process for 60 days.
9. Results
The model’s results were summarized in a table for clarity:

Metric	Value
Total Data Points	Number of data points in the dataset
Training Data Points	Length of training data
Testing Data Points	Length of testing data
Mean Squared Error (MSE)	mse
First Actual Price	y_test_scaled[0][0]
First Predicted Price	predictions[0][0]
Last Actual Price	y_test_scaled[-1][0]
Last Predicted Price	predictions[-1][0]
Predicted Future Price (next day)	predicted_price[0][0]
10. Visualization
Three key visualizations were generated:

Historical Price Plot: This shows the Ethereum closing price over the available data period.
Actual vs Predicted Prices: A plot comparing the actual Ethereum prices with the model's predicted prices on the test set.
Future Predictions: This plot adds the predicted Ethereum prices for the next 60 days alongside the actual prices.
11. Conclusions
The LSTM model was able to reasonably capture the trend of Ethereum prices and provided predictions for future prices.
The future predictions suggest an upward/downward trend for the next 60 days, but further tuning and data could improve accuracy.
The MSE can be further reduced by adjusting hyperparameters or adding more features to the model (e.g., trading volume, other market indicators).
12. Further Improvements
Data Enrichment: Including additional features such as trading volume, market trends, or external factors like news sentiment could improve the model’s accuracy.
Model Tuning: Further hyperparameter tuning (such as changing the number of LSTM units, sequence length, or dropout rates) might yield better results.
Ensemble Models: Combining LSTM with other models (e.g., GRU or Transformer-based models) could enhance predictive performance.
This project showcases how LSTM models can be effectively applied to predict time series data like cryptocurrency prices. The results are promising, but additional improvements can make the model more robust for real-world applications.













TÜRKÇE 


1. Amaç
Bu projenin temel amacı, geçmiş verilere dayanarak Ethereum'un gelecekteki fiyatını tahmin edebilecek bir Long Short-Term Memory (LSTM) modeli inşa etmektir. Model, geçmiş fiyat verilerini kullanarak eğitilmiş ve önümüzdeki 60 gün için Ethereum fiyatını tahmin etmek için kullanılmıştır.

2. Veri Seti Genel Bakış
Bu projede kullanılan veri seti, 2018'den 2024'e kadar olan günlük Ethereum fiyat verilerini içermektedir. Veri setindeki ana özellikler:

time: İlgili günün tarihini belirtir.
Close: Ethereum’un ilgili günkü kapanış fiyatı. Tahmin işlemi için bu değişken kullanılmıştır.
3. Veri Ön İşleme
Veri seti pandas DataFrame formatında yüklendi ve time sütunu datetime formatına dönüştürüldü.
Zaman serisi verilerini daha kolay görselleştirmek ve işlemek için time sütunu indeks olarak ayarlandı.
Veriler LSTM modeli için uygun hale getirilebilmesi amacıyla MinMaxScaler ile 0 ve 1 arasında normalize edildi.
60 günlük bir sekans uzunluğu seçildi, yani model bir sonraki günün kapanış fiyatını tahmin etmek için önceki 60 günü kullanacak şekilde tasarlandı.
4. Veri Setinin Bölünmesi
Veri seti iki kısma ayrıldı:
Eğitim Seti: Verilerin %80’i modelin eğitimi için kullanıldı.
Test Seti: Kalan %20’lik kısım, modelin performansını değerlendirmek amacıyla test verisi olarak kullanıldı.
5. LSTM Model Mimarisi
LSTM modeli aşağıdaki katmanlardan oluşmaktadır:

Girdi Katmanı: Girdi şekli, 60 günlük bir sekansa karşılık gelecek şekilde ayarlandı.
İlk LSTM Katmanı: 50 birimli ve return_sequences=True parametresiyle tanımlanan bu katman, diziyi bir sonraki LSTM katmanına aktarır.
İkinci LSTM Katmanı: 50 birimli ikinci bir LSTM katmanı ve ardından aşırı öğrenmeyi engellemek için bir Dropout katmanı kullanıldı.
Yoğun Katmanlar: Nihai tahmini üretmek için iki yoğun katman eklendi; son katmanda bir birimlik çıktı (fiyat) bulunmaktadır.
6. Modelin Eğitilmesi
Model, 70 epoch boyunca ve 84 batch boyutu ile Adam optimizasyon algoritması ve mean_squared_error kayıp fonksiyonu kullanılarak eğitildi. Eğitim sırasında test verileriyle doğrulama yapılarak modelin genelleme kapasitesi artırıldı.

7. Model Değerlendirmesi
Model, test verisi üzerinde Mean Squared Error (MSE) metriği ile değerlendirildi:

Mean Squared Error: MSE, gerçek ve tahmin edilen fiyatlar arasındaki kare farkların ortalamasını ifade eder ve modelin veriyi ne kadar iyi öğrendiğini gösterir.
Görselleştirme: Modelin performansı, gerçek Ethereum fiyatları ile tahmin edilen fiyatların karşılaştırıldığı bir grafikle görselleştirildi. Bu grafik modelin hem geçmiş hem de gelecek fiyatları ne kadar iyi tahmin ettiğini göstermektedir.
8. Gelecek Fiyat Tahmini
Eğitilen LSTM modeli kullanılarak önümüzdeki 60 gün için fiyat tahminleri yapılmıştır. Tahmin süreci şu adımlarla gerçekleştirilmiştir:

Son 60 günün verisi alınarak, modelin tahmin yapabilmesi için ölçeklendirilmiştir.
Ardışık olarak her bir gün için fiyat tahmini yapılmış ve bu tahminler, bir sonraki günün tahminini yapmak üzere veri dizisine eklenmiştir. Bu süreç 60 gün boyunca tekrarlanmıştır.
9. Sonuçlar
Modelin sonuçları aşağıdaki tabloda özetlenmiştir:

Metrik	Değer
Toplam Veri Noktası	Veri setindeki toplam veri sayısı
Eğitim Veri Noktaları	Eğitim verisinin uzunluğu
Test Veri Noktaları	Test verisinin uzunluğu
Ortalama Kare Hata (MSE)	mse
İlk Gerçek Fiyat	y_test_scaled[0][0]
İlk Tahmin Edilen Fiyat	predictions[0][0]
Son Gerçek Fiyat	y_test_scaled[-1][0]
Son Tahmin Edilen Fiyat	predictions[-1][0]
Gelecekteki Tahmini Fiyat	predicted_price[0][0]
10. Görselleştirme
Üç ana grafik oluşturulmuştur:

Tarihsel Fiyat Grafiği: Ethereum’un kapanış fiyatını gösteren tarihsel fiyat grafiği.
Gerçek vs Tahmin Edilen Fiyatlar: Test setinde modelin tahmin ettiği fiyatlar ile gerçek fiyatların karşılaştırıldığı grafik.
Gelecek Tahminleri: Gelecek 60 gün için modelin tahmin ettiği Ethereum fiyatlarının gösterildiği grafik.
11. Sonuçlar
LSTM modeli, Ethereum fiyatlarının trendini makul bir şekilde yakalayabilmiş ve gelecekteki fiyatlar için tahminler sunmuştur.
Gelecek tahminleri, önümüzdeki 60 gün için yukarıya/aşağıya doğru bir trend gösterebilir, ancak daha fazla ince ayar ve veri ile doğruluğu artırılabilir.
MSE, hiperparametrelerin ayarlanması veya modele ek özellikler (örneğin işlem hacmi, piyasa göstergeleri) eklenerek daha da azaltılabilir.
12. Gelecek İyileştirmeler
Veri Zenginleştirme: İşlem hacmi, piyasa trendleri veya dış faktörler (haber duyarlılığı gibi) gibi ek özelliklerin dahil edilmesi modelin doğruluğunu artırabilir.
Modelin Ayarlanması: LSTM birim sayısının, sekans uzunluğunun veya dropout oranlarının değiştirilmesi gibi hiperparametre ayarlamaları ile daha iyi sonuçlar elde edilebilir.
Model Birleşimi: LSTM ile birlikte diğer modellerin (örneğin GRU veya Transformer tabanlı modeller) kullanılması, tahmin performansını artırabilir.
Bu proje, LSTM modellerinin kripto para fiyat tahmini gibi zaman serisi verilerine nasıl etkili bir şekilde uygulanabileceğini göstermektedir. Sonuçlar ümit vericidir ancak modelin gerçek dünya uygulamaları için daha da sağlam hale getirilmesi için ek iyileştirmeler yapılabilir.


