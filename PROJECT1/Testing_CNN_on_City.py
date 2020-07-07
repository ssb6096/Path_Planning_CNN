import pickle
import tensorflow
import numpy as np
import keras
from keras.models import model_from_json
json_file = open('cnn_model_135_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("cnn_model_135_1.h5")
print("Loaded cnn model from disk")

with open('mazes_city_map_t.pkl', 'rb') as f:
    my_new_list1 = pickle.load(f)
with open('pathshortest_city_map_t.pkl', 'rb') as f:
    my_new_list2 = pickle.load(f)
x =np.asarray(my_new_list1)
y = np.asarray(my_new_list2)
X_test_nn = x.reshape(1,135,135,1)
Y_test_nn = y.reshape(1,18225)

h=18225
b=[]
'''
for r in range(len(my_new_list2)):
        print(my_new_list2[r], end=',')
        if my_new_list2[r] != 0:  b.append(my_new_list2[r])
        print()
print(len(b))'''

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


prediction = loaded_model.predict(X_test_nn)
_, accuracy = loaded_model.evaluate(X_test_nn, Y_test_nn)
print('Accuracy: %.2f' % (accuracy*100))

#predict_text is the text in "predict" file in format required by the model
#prediction = loaded_model.predict(predict_text)
print(prediction)