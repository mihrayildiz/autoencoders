from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import fashion_mnist #default olarak kerasta bulunmakta
import matplotlib.pyplot as plt
import json, codecs
import warnings
warnings.filterwarnings("ignore")

(x_train, _), (x_test, _) = fashion_mnist.load_data() 
#labellara ihtiyaç yok inputta, outputta aynı olacağı için  _ ile kullanmadığımız belirtildi.

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((len(x_train), x_train.shape[1:][0]*x_train.shape[1:][1])) #28*28 =784
x_test = x_test.reshape((len(x_test), x_test.shape[1:][0]*x_test.shape[1:][1]))

plt.imshow(x_train[4000].reshape(28,28))
plt.axis("off")
plt.show()

#%%

input_img = Input(shape = (784,)) 
#görüntüleri 28*28=784 ile (784,) boyutuna getirilmişti input layer olarak eklendi.

encoded = Dense(32, activation="relu")(input_img)
 #32 node ile ilk encode layer oluşturuldu. () ile girdi olarak input_img verildi.

encoded = Dense(16, activation="relu")(encoded)
#(encoded) bir önceki encoded layer ile bağlandı.

decoded = Dense(32, activation="relu")(encoded)
#decode layer

decoded = Dense(784, activation="sigmoid")(decoded) 
#output layer

autoencoder = Model(input_img,decoded)
 #modeli inputtan başlanarak decodeda kadar bağlanıldı ve  Model oluşturuldu.

autoencoder.compile(optimizer="rmsprop",loss="binary_crossentropy")
#autoencoder modeli gerekli parametreler ile compile edildi.

hist = autoencoder.fit(x_train,      
                       x_train,
                       epochs=200,
                       batch_size=256,
                       shuffle=True,
                       validation_data = (x_train,x_train))

#%% save model
autoencoder.save_weights("autoencoder_model.h5") 
#weightler kaydedildi.

#%% evaluation
print(hist.history.keys())

plt.plot(hist.history["loss"],label = "Train loss")
plt.plot(hist.history["val_loss"],label = "Val loss")

plt.legend()
plt.show()

# %% save hist  .
with open("autoencoders_hist.json","w") as f:
    json.dump(hist.history,f)
#evalution kaydedildi

# %% load history   
with codecs.open("autoencoders_hist.json","r", encoding="utf-8")  as f:
    n = json.loads(f.read())

#evalution yüklendi
#%% 
print(n.keys())
plt.plot(n["loss"],label = "Train loss")
plt.plot(n["val_loss"],label = "Val loss")

#%% 
encoder = Model(input_img,encoded) 
encoded_img = encoder.predict(x_test) 
#inputtan başalyıp encode bölümüne kadar giden bir model oluşturuldu.
#encoded img leri predict edildi.

plt.imshow(x_test[1500].reshape(28,28)) 
plt.axis("off")
plt.show()
#normalde olması gereken resim

plt.figure()
plt.imshow(encoded_img[1500].reshape(4,4)) 
plt.axis("off")
plt.show()
#encoded yaptığımızda yani ikinci layerdan geçtikten sonra detect edilen özellik 

decoded_imgs = autoencoder.predict(x_test)

n = 10  #decoded img leri görselleştirdik.
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28)) #orjinalleri 
    plt.axis("off")

    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28)) #decode edilmiş halleri
    plt.axis("off")
plt.show()

















