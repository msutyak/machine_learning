#rescaler for feature training
#simple

x_min = None
x_max = None
x = None

x_prime = (x - x_min)/(x_max - x_min)

#with sklearn

from sklearn.preprocessing import MinMaxScaler
import numpy

weights = numpy.array([[115.], [140.], [175.]])
scaler = MinMaxScaler()
rescaled_weight = scaler.fit_transform(weights)
