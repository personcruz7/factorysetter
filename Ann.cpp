from tensorflow import keras
import numpy as np


!pip3 install ann_visualizer


from ann_visualizer.visualize import ann_viz
from graphviz import Source


train = np.array([[0,0],[0,1],[1,0],[1,1]])
test = np.array([[0],[0],[0],[1]])
  
  
model = keras.models.Sequential()
model.add(keras.layers.Dense(4, input_dim=2, activation='sigmoid'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
  

ann_viz(model, title="Simple AND Classifier")
graph_source = Source.from_file('network.gv')
graph_source
model.get_weights()
