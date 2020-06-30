import librosa
import numpy as np
import soundfile as sf
import pyaudio
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

# Metodo para seleccionar archivos de audio
def parseo_datos(direc):
    X = []
    y = []

    for root, dirs, files in os.walk(direc):
        for name in files:
            emotion = name.split('_')
            if name.endswith('.WAV'):
                if not root.endswith('Test1') and not root.endswith('Test2'):
                    if emotion[0] is not 'Alegria' and not 'Asco' and not 'Ira' and not 'Miedo' and not 'Neutro' and not 'Sorpresa' and not 'Tristeza':
                        continue
                    try:
                        features = extract_feature( root + '/' + name)
                        X.append(features)
                        y.append(emotion[0])
                    except:
                        print(root + '/' + name + ' - Falla')
                        continue

    return train_test_split(np.array(X), y)

def getTestAudios(direc):
    X = []
    y = []
    for root, dirs, files in os.walk(direc):
        for name in files:
            emotion = name.split('_')
            if name.endswith('.WAV'):
                if root.endswith('Test1') or root.endswith('Test2'):
                    try:
                        features = extract_feature( root + '/' + name)
                        X.append(features)
                        y.append(emotion[0])
                    except:
                        print(root + '/' + name + ' - Falla')
                        continue
    return X, y

# Metodo para extraer las caracteristicas del audio en cuestion. MFCCs
def extract_feature(file_name):
       
    with sf.SoundFile(file_name):
        X, sample_rate = librosa.load(file_name, mono=True)
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=100).T
        mfccs_mean = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=100).T, axis=0)
        mfcc_delta = np.mean(librosa.feature.delta(mfccs, width=3).T, axis=1)
        mfcc_delta_delta = np.mean(librosa.feature.delta(mfccs, order=2).T, axis=1)

        mfccs = np.hstack((mfccs_mean, mfcc_delta, mfcc_delta_delta))
    return mfccs

# Metodo de Redes neuronales con los argumentos:
def neuralNetwork(X_train, X_test, y_train, y_test, x_final_test, y_final_test):
    neuralNetwork = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, learning_rate='adaptive')
    neuralNetwork.fit(X_train, y_train)
    y_pred = neuralNetwork.predict(X_test)
    print("Neural Network")
    print("Test de aprendizaje")
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    y_final_pred = neuralNetwork.predict(x_final_test)
    print("Test con las carpetas Test")
    print(confusion_matrix(y_final_test, y_final_pred))
    print(accuracy_score(y_final_test, y_final_pred))


# Metodo de Bayes 
def naiveBayes(X_train, X_test, y_train, y_test, x_final_test, y_final_test):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("Gaussian Naive Bayes")
    print("Test de aprendizaje")
    print(confusion_matrix(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    y_final_pred = classifier.predict(x_final_test)
    print("Test con las carpetas Test")
    print(confusion_matrix(y_final_test, y_final_pred))
    print(accuracy_score(y_final_test, y_final_pred))
    print()

def ejecucionModelos(X_train, X_test, y_train, y_test, x_final_test, y_final_test):
    naiveBayes(X_train, X_test, y_train, y_test, x_final_test, y_final_test)
    neuralNetwork(X_train, X_test, y_train, y_test, x_final_test, y_final_test)

if __name__ == '__main__':
    direc = 'Data'
    x_final_test, y_final_test = getTestAudios(direc)
    X_entre, X_testeo, y_entre, y_testeo = parseo_datos(direc)
    ejecucionModelos(X_entre, X_testeo, y_entre, y_testeo, x_final_test, y_final_test)
