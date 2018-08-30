from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import os

def xception_init(weights_file, gpu = "0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    input_tensor = Input((299, 299, 3))
	x = Lambda(xception.preprocess_input)(input_tensor)

	base_model = Xception(input_tensor=x, weights=None, include_top=False)
	m_out = base_model.output
	p_out = GlobalAveragePooling2D()(m_out)
	p_out = Dropout(1.0)(p_out)
	fc_out = Dense(1024, activation='relu')(p_out)
	fc_out = Dropout(1.0)(fc_out)
	predictions = Dense(18, activation='softmax')(fc_out)
	model = Model(inputs=base_model.input, outputs=predictions)
	model.load_weights(weights_file)

	return model

def xception_predict(img_data, batch_size, model):

	predictions = []
	batches = int(len(img_data) / batch_size) + 1

    for i in range(batches):
    	batch_data =  img_data[i * batch_size : (i + 1) * batch_size]
		predictions.extend(model.predict_on_batch(batch_data))

    return predictions