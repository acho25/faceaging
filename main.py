## NOTE: This code has been tested on Google Colab.

import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scipy.io import loadmat
from datetime import datetime
from keras import Input, Model
from keras.applications import inception_resnet_v2
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from keras.layers import Reshape, concatenate, LeakyReLU, Lambda
from keras.layers import Activation, UpSampling2D, Dropout
from keras import backend as K
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras_preprocessing import image


# Réseau d'encodeur

'''La fonction build_encoder permet de créer l'encodeur, en premier temps on prend notre image et la convertit 
en un vecteur de 100 dimensions puis on doit cree quatre couches de convolution utilisant la fonction Conv2D 
et deux couches entièrement connectées chaque couche de convolution a pour fonction d'activation une LeakyReLU . 
Autrement dit, il y a toujours une couche de correction LeakyReLU et une couche de normalisation par lots 
 (BatchNormalization) après une couche de convolution. '''

def build_encoder():
  
  input_layer = Input(shape = (64, 64, 3))

  
  ## 1er bloc convolutionnel
  enc = Conv2D(filters = 32, kernel_size = 5, strides = 2, padding = 'same')(input_layer)
  # enc = BatchNormalization()(enc)
  enc = LeakyReLU(alpha = 0.2)(enc)
  
  ## 2eme bloc convolutif
  enc = Conv2D(filters = 64, kernel_size = 5, strides = 2, padding = 'same')(enc)
  enc = BatchNormalization()(enc)
  enc = LeakyReLU(alpha = 0.2)(enc)
  
  ## 3eme bloc convolutif
  enc = Conv2D(filters = 128, kernel_size = 5, strides = 2, padding = 'same')(enc)
  enc = BatchNormalization()(enc)
  enc = LeakyReLU(alpha = 0.2)(enc)
  
  ## 4eme bloc convolutif
  enc = Conv2D(filters = 256, kernel_size = 5, strides = 2, padding = 'same')(enc)
  enc = BatchNormalization()(enc)
  enc = LeakyReLU(alpha = 0.2)(enc)
  
  enc = Flatten()(enc)
  
  ## 1ere couche entièrement connectée
  enc = Dense(4096)(enc)
  enc = BatchNormalization()(enc)
  enc = LeakyReLU(alpha = 0.2)(enc)
  
  ## 2eme couche entièrement connectée
  enc = Dense(100)(enc)
  
  
  ## Créer un modèle
  model = Model(inputs = [input_layer], outputs = [enc])
  return model
  
  
# Réseau de générateurs
''' La fonction build_generator() permet de créer notre génerateur qui est chargé de générer une image, il prend
un vecteur latent de 100 dimension depuis l'encodeur et num_classe comme entrée et tente de générer des images 
réalistes. Le générateur prend deux valeurs d'entrée un vecteur de bruit et une valeur conditionnelle, Ce 
reseau la est un CNN composé de couches suivant les elements :dense, batch, Conv et LeakyRelu. La derniere couche est
suivie par une couche de correction tanh.  
'''
def build_generator():
  
  latent_dims = 100
  num_classes = 6
  
  input_z_noise = Input(shape = (latent_dims, ))
  input_label = Input(shape = (num_classes, ))
  
  x = concatenate([input_z_noise, input_label])
  
  
  x = Dense(2048, input_dim = latent_dims + num_classes)(x)
  x = LeakyReLU(alpha = 0.2)(x)
  x = Dropout(0.2)(x)
  
  x = Dense(256 * 8 * 8)(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(alpha = 0.2)(x)
  x = Dropout(0.2)(x)
  
  x = Reshape((8, 8, 256))(x)
  
  x = UpSampling2D(size = (2, 2))(x)
  x = Conv2D(filters = 128, kernel_size = 5, padding = 'same')(x)
  x = BatchNormalization(momentum = 0.8)(x)
  x = LeakyReLU(alpha = 0.2)(x)
  
  x = UpSampling2D(size = (2, 2))(x)
  x = Conv2D(filters = 64, kernel_size = 5, padding = 'same')(x)
  x = BatchNormalization(momentum = 0.8)(x)
  x = LeakyReLU(alpha = 0.2)(x)
  
  x = UpSampling2D(size = (2, 2))(x)
  x = Conv2D(filters = 3, kernel_size = 5, padding = 'same')(x)
  x = Activation('tanh')(x)

  
  
  model = Model(inputs = [input_z_noise, input_label], outputs = [x])
  return model
  
#
def expand_label_input(x):
  x = K.expand_dims(x, axis = 1)
  x = K.expand_dims(x, axis = 1)
  x = K.tile(x, [1, 32, 32, 1])
  return x
  
  
# Réseau de discrimination

''' La fonction build_discriminator permet de créer notre discrimianteur qui est chargée d'identifier 
si l'image fournie est fausse ou réelle par le faire passer à travers une série de couches de 
sous-échantillonnage et certaines couches de classification. Le discrimiateur est un CNN aussi les deux
réseaux précedents qu'on a crée la seule difference est que la premier couche de convolution manque la couche
de normalization par lots (batch)
'''
def build_discriminator():
  
  input_shape = (64, 64, 3)
  label_shape = (6, )
  image_input = Input(shape = input_shape)
  label_input = Input(shape = label_shape)
  
  x = Conv2D(64, kernel_size = 3, strides = 2, padding = 'same')(image_input)
  x = LeakyReLU(alpha = 0.2)(x)
  
  label_input1 = Lambda(expand_label_input)(label_input)
  x = concatenate([x, label_input1], axis = 3)
  
  
  x = Conv2D(128, kernel_size = 3, strides = 2, padding = 'same')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(alpha = 0.2)(x)
  
  x = Conv2D(256, kernel_size = 3, strides = 2, padding = 'same')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(alpha = 0.2)(x)
  
  x = Conv2D(512, kernel_size = 3, strides = 2, padding = 'same')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU(alpha = 0.2)(x)
  
  x = Flatten()(x)
  
  x = Dense(1, activation = 'sigmoid')(x)
  
  
  model = Model(inputs = [image_input, label_input], outputs = [x])

  
  return model


# Fonctions utilitaires

def build_fr_combined_network(encoder, generator, fr_model):
  input_image = Input(shape = (64, 64, 3))
  input_label = Input(shape = (6, ))
  
  latent0 = encoder(input_image)
  
  gen_images = generator([latent0, input_label])
  
  fr_model.trainable = False
  
  resized_images = Lambda(lambda x: K.resize_images(gen_images, height_factor = 2,
                                                    width_factor = 2, 
                                                    data_format = 'channels_last'))(gen_images)
  
  embeddings = fr_model(resized_images)
  
  
  model = Model(inputs = [input_image, input_label],
                outputs = [embeddings])
  return model
  

'''Cette fonction permet de crée un Réseau de reconnaissance faciale pour construire de model on utilise le pré-entraîné
InceptionResNetV2 model sans les couches entierement connecté. Ce reseau une fois pourvu d'une image, renvoie 
l'intégration correspondante
'''
def build_fr_model(input_shape):
  
  resnet_model = InceptionResNetV2(include_top = False, weights = 'imagenet',
                                   input_shape = input_shape, pooling = 'avg')
  image_input = resnet_model.input
  x = resnet_model.layers[-1].output
  out = Dense(128)(x)
  embedder_model = Model(inputs = [image_input], outputs = [out])
  
  input_layer = Input(shape = input_shape)
  
  x = embedder_model(input_layer)
  output = Lambda(lambda x: K.l2_normalize(x, axis = -1))(x)
  
  
  model = Model(inputs = [input_layer], outputs = [output])
  return model
  
  

#Reduction des dimension d'entrée
def build_image_resizer():
  
  input_layer = Input(shape = (64, 64, 3))
  
  resized_images = Lambda(lambda x: K.resize_images(x, height_factor = 3,
                                                    width_factor = 3,
                                                    data_format = 'channels_last'))(input_layer)
  
  model = Model(inputs = [input_layer],
                outputs = [resized_images])
  return model
  
  
def age_to_category(age_list):
  
  age_list1 = []
  
  for age in age_list:
    if 0 < age <= 18:
      age_category = 0
    elif 18 < age <= 29:
      age_category = 1
    elif 29 < age <= 39:
      age_category = 2
    elif 39 < age <= 49:
      age_category = 3
    elif 49 < age <= 59:
      age_category = 4
    elif age >= 60:
      age_category = 5
      
    age_list1.append(age_category)
    
  return age_list1
  
  
#Chargement des images
def load_images(data_dir, image_paths, image_shape):
  
  images = None
  
  for i, image_path in enumerate(image_paths):
    print(i)
    #if i == 1000: pour tester avec moin de temps de traitement 
      #break
    try:
      ## Load image
      loaded_image = image.load_img(os.path.join(data_dir, image_path),
                                    target_size = image_shape)
      
      
      ## Convert PIL image to numpy ndarray
      loaded_image = image.img_to_array(loaded_image)
      
      ## Add another dimension (Add batch dimension)
      loaded_image = np.expand_dims(loaded_image, axis = 0)
      
      
      ## Concatenate all images into one tensor:
      if images is None:
        images = loaded_image
      else:
        images = np.concatenate([images, loaded_image], axis = 0)
        
    except Exception as e:
      print("Error: ", i, e)
      
  return images
  

def euclidean_distance_loss(y_true, y_pred):
  
  """
  Euclidean distance
  https://en.wikipedia.org/wiki/Euclidean_distance
  y_true = TF / Theano tensor
  y_pred = TF / Theano tensor of the same shape as y_true
  returns float
  """
  
  return K.sqrt(K.sum(K.square(y_pred - y_true), axis = -1))
  
  
def write_log(callback, name, value, batch_no):
  summary = tf.Summary()
  summary_value = summary.value.add()
  summary_value.simple_value = value
  summary_value.tag = name
  callback.writer.add_summary(summary, batch_no)
  callback.writer.flush()
  
  
def save_rgb_img(img, path):
  
  """
  Enregistrer une image RGB
  """
  
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.imshow(img)
  ax.axis("off")
  ax.set_title("Image")
  
  plt.savefig(path)
  plt.close()
  
import mat73
import scipy.io

def calculate_age(taken, dob):
  birth = datetime.fromordinal(max(int(dob) - 366, 1))
  
  if birth.month < 7:
    return taken - birth.year
  else:
    return taken - birth.year - 1

def load_data(wiki_dir, dataset = 'wiki'):
  ## Charger le fichier wiki.mat
  meta = scipy.io.loadmat(os.path.join(wiki_dir, "{}.mat".format(dataset)))
  
  ## Charger la liste de tous les fichiers
  full_path = meta[dataset][0, 0]["full_path"][0]
  
  ## Liste des numéros de date de série Matlab
  dob = meta[dataset][0, 0]["dob"][0]
  
  ## Liste des années où la photo a été prise
  photo_taken = meta[dataset][0, 0]["photo_taken"][0]  # year
  
  ## Calculer l'âge pour tous les dobs
  age = [calculate_age(photo_taken[i], dob[i]) for i in range(len(dob))]
  
  ## Créer une liste de tuples contenant une paire de chemin d'image et d'âge
  images = []
  age_list = []
  for index, image_path in enumerate(full_path):
    images.append(image_path[0])
    age_list.append(age[index])
    print("done")
  
  ## Renvoie une liste de toutes les images et leur âge respectif
  return images, age_list

if __name__ == '__main__':
  
  ## Définir les hyperparamètres
  data_dir = "data"
  wiki_dir = os.path.join(data_dir, "wiki_crop")
  ##wiki_dir = "wiki_crop"
  epochs = 100
  batch_size = 2
  image_shape = (64, 64, 3)
  z_shape = 100
  TRAIN_GAN = True
  TRAIN_ENCODER = False
  TRAIN_GAN_WITH_FR = False
  fr_image_shape = (192, 192, 3)
  
  
  ## Définir les optimiseurs
  dis_optimizer = Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999, epsilon = 10e-8)
  gen_optimizer = Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999, epsilon = 10e-8)
  adversarial_optimizer = Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999, epsilon = 10e-8)
  
  
  """
  Build and compile networks
  """
  
  ## Construire et compiler le réseau discriminateur
  discriminator = build_discriminator()
  discriminator.compile(loss = ['binary_crossentropy'],
                        optimizer = dis_optimizer)
  
  
  ## Construire et compiler le réseau de générateurs
  generator = build_generator()
  generator.compile(loss = ['binary_crossentropy'],
                    optimizer = gen_optimizer)
  
  
  ## Construire et compiler le modèle contradictoire
  discriminator.trainable = False
  input_z_noise = Input(shape = (100, ))
  input_label = Input(shape = (6, ))
  recons_images = generator([input_z_noise, input_label])
  valid = discriminator([recons_images, input_label])
  adversarial_model = Model(inputs = [input_z_noise, input_label],
                            outputs = [valid])
  adversarial_model.compile(loss = ['binary_crossentropy'],
                            optimizer = gen_optimizer)
  
  tensorboard = TensorBoard(log_dir = "logs/{}".format(time.time()))
  tensorboard.set_model(generator)
  tensorboard.set_model(discriminator)
  
  
  """
  Charger la dataset
  """
  
  images, age_list = load_data(wiki_dir = wiki_dir, dataset = "wiki")
  age_cat = age_to_category(age_list)
  final_age_cat = np.reshape(np.array(age_cat), [len(age_cat), 1])
  classes = len(set(age_cat))
  y = to_categorical(final_age_cat, num_classes = classes)
  
  
  loaded_images = load_images(wiki_dir, images, (image_shape[0], image_shape[1]))
  
  
  ## Implementation de label smoothing
  real_labels = np.ones((batch_size, 1), dtype = np.float32) * 0.9
  fake_labels = np.zeros((batch_size, 1), dtype = np.float32) * 0.1
  
  


  """Dans cette étape, nous entraînons à la fois les réseaux générateurs et discriminateur.
Une fois le réseau du générateur formé, il peut générer des images floues dun visage.
Cette étape est comparable à la formation d un GAN , où nous formons les deux réseaux
simultanément."""
  
  if TRAIN_GAN:
    for epoch in range(epochs):
      print("Epoch: {}".format(epoch))
      
      gen_losses = []
      dis_losses = []
      
      number_of_batches = int(len(loaded_images) / batch_size)
      print("Number of batches: ", number_of_batches)
      for index in range(number_of_batches):
        print("Batch: {}".format(index + 1))
        
        images_batch = loaded_images[index * batch_size:(index + 1) * batch_size]
        images_batch = images_batch / 127.5 - 1.0
        images_batch = images_batch.astype(np.float32)
        
        y_batch = y[index * batch_size: (index + 1) * batch_size]
        z_noise = np.random.normal(0, 1, size = (batch_size, z_shape))
        
        
        """
        Train discriminateur
        """
        
        ## Générer de fausses images
        initial_recons_images = generator.predict_on_batch([z_noise, y_batch])
        
        d_loss_real = discriminator.train_on_batch([images_batch, y_batch], real_labels)
        d_loss_fake = discriminator.train_on_batch([initial_recons_images, y_batch], fake_labels)
        
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        print("d_loss: {}".format(d_loss))
        
        
        """
        Train de générateur
        """
        
        z_noise2 = np.random.normal(0, 1, size = (batch_size, z_shape))
        random_labels = np.random.randint(0, 6, batch_size).reshape(-1, 1)
        random_labels = to_categorical(random_labels, 6)
      
      
      """
      Générer des images floues après chaque 10ème époque
      """
      
      if epoch % 10 == 0:
        images_batch = loaded_images[0:batch_size]
        images_batch = images_batch / 127.5 - 1.0
        images_batch = images_batch.astype(np.float32)
        
        y_batch = y[0:batch_size]
        z_noise = np.random.normal(0, 1, size = (batch_size, z_shape))
        
        gen_images = generator.predict_on_batch([z_noise, y_batch])
        
        for i, img in enumerate(gen_images[:5]):
          save_rgb_img(img, path = "results/img_{}_{}.png".format(epoch, i))
          
        
    ## Enregistrer les réseaux
    try:
      generator.save_weights("generator.h5")
      discriminator.save_weights("discriminator.h5")
    except Exception as e:
      print("Error: ", e)
      
  
  """ 
  Nous entraînons le réseau de codeur sur les images générées et les images réelles.
  Une fois formé, il commencera à générer des vecteurs latents à partir de la distribution
  apprise.

  """
  
  if TRAIN_ENCODER:
    
    ## Construire et compiler l'encodeur
    encoder = build_encoder()
    encoder.compile(loss = euclidean_distance_loss,
                    optimizer = 'adam')
    
    
    ## Charger les poids du réseau de générateur
    try:
      generator.load_weights("generator.h5")
    except Exception as e:
      print("Error: ", e)
      
    
    z_i = np.random.normal(0, 1, size = (5000, z_shape))
    
    y = np.random.randint(low = 0, high = 6, size = (5000, ),
                          dtype = np.int64)
    num_classes = len(set(y))
    y = np.reshape(np.array(y), [len(y), 1])
    y = to_categorical(y, num_classes = num_classes)
    
    
    for epoch in range(epochs):
      print("Epoch: ", epoch)
      
      encoder_losses = []
      
      number_of_batches = int(z_i.shape[0] / batch_size)
      print("Number of batches: ", number_of_batches)
      
      for index in range(number_of_batches):
        print("Batch: ", index + 1)
        
        z_batch = z_i[index * batch_size: (index + 1) * batch_size]
        y_batch = y[index * batch_size: (index + 1) * batch_size]
        
        generated_images = generator.predict_on_batch([z_batch, y_batch])
        
        
        ## Train le modèle d'encodeur
        encoder_loss = encoder.train_on_batch(generated_images, z_batch)
        print("Encoder loss: ", encoder_loss)
        
        encoder_losses.append(encoder_loss)
        
      
    ## Enregistrer le modèle d'encodeur
    encoder.save_weights("encoder.h5")
    
    
  """
  Optimiser le codeur et le réseau du générateur
  dans cette étape, nous essayons de minimiser la distance afin de maximiser la
  préservation de l’identité.
  """
  
  if TRAIN_GAN_WITH_FR:
    
    ## Charger le réseau codeur
    encoder = build_encoder()
    encoder.load_weights("encoder.h5")
    
    
    ## Charger le réseau du générateur
    generator.load_weights("generator.h5")
    
    image_resizer = build_image_resizer()
    image_resizer.compile(loss = ['binary_crossentropy'],
                          optimzer = 'adam')
    
    
    ## Modèle de reconnaissance faciale
    fr_model = build_fr_model(input_shape = fr_image_shape)
    fr_model.compile(loss = ['binary_crossentropy'],
                     optimizer = 'adam')
    
    ## Rendre le modèle de reconnaissance faciale non entraînable
    fr_model.trainable = False
    
    
    ## Couches d'entrée
    input_image = Input(shape = (64, 64, 3))
    input_label = Input(shape = (6, ))
    
    
    ## Utiliser l'encodeur et le réseau du générateur
    latent0 = encoder(input_image)
    gen_images = generator([latent0, input_label])
    
    
    ## Redimensionner les images à la forme souhaitée
    resized_images = Lambda(lambda x: K.resize_images(gen_images, height_factor = 3,
                                                      width_factor = 3,
                                                      data_format = 'channels_last'))(gen_images) 
    embeddings = fr_model(resized_images)
    
    
    ## Créer un modèle Keras et spécifier les entrées et sorties pour le réseau
    fr_adversarial_model = Model(inputs = [input_image, input_label],
                                 outputs = [embeddings])
    
    
    ## Compiler le modèle
    fr_adversarial_model.compile(loss = euclidean_distance_loss,
                                 optimizer = adversarial_optimizer)
    
    for epoch in range(epochs):
      print("Epoch: ", epoch)
      
      reconstruction_losses = []
      
      number_of_batches = int(len(loaded_images) / batch_size)
      print("Number of batches: ", number_of_batches)
      for index in range(number_of_batches):
        print("Batch: ", index + 1)
        
        images_batch = loaded_images[index * batch_size: (index + 1) * batch_size]
        images_batch = images_batch / 127.5 - 1.0
        images_batch = images_batch.astype(np.float32)
        
        y_batch = y[index * batch_size: (index + 1) * batch_size]
        
        images_batch_resized = image_resizer.predict_on_batch(images_batch)
        
        real_embeddings = fr_model.predict_on_batch(images_batch_resized)
        
        reconstruction_loss = fr_adversarial_model.train_on_batch([images_batch, y_batch], real_embeddings)
        
        print("Reconstruction loss: ", reconstruction_loss)
        
        reconstruction_losses.append(reconstruction_loss)
        
        
      ## Écrire la perte de reconstruction à Tensorboard
      write_log(tensorboard, "reconstruction_loss", np.mean(reconstruction_losses), epoch)
      
      
      """
      Générer des images
      """
      
      if epoch % 10 == 0:
        images_batch = loaded_images[0:batch_size]
        images_batch = images_batch / 127.5 - 1.0
        images_batch = images_batch.astype(np.float32)
        
        y_batch = y[0:batch_size]
        z_noise = np.random.normal(0, 1, size = (batch_size, z_shape))
        
        gen_images = generator.predict_on_batch([z_noise, y_batch])
        
        for i, img in enumerate(gen_images[:5]):
          save_rgb_image(img, path = "results/img_opt_{}_{}.png".format(epoch, i))
        
        
    ## Enregistrer des poids améliorés pour les deux réseaux
    generator.save_weights("generator_optimized.h5")
    encoder.save_weights("encoder_optimized.h5")
