import numpy as np
import neuron
import samples
import data_classification_utils as dcu
from util import raiseNotDefined

"""feel free to play with these values and see what happens"""
bias = False
num_times_to_train = 10
alpha = 0.01
num_train_examples = 50 #can increase up to 5000 if you are patient

def get_neuron_training_data():
	training_data = samples.loadDataFile("digitdata/trainingimages", num_train_examples, 28, 28)
	training_labels = np.array(samples.loadLabelsFile("digitdata/traininglabels", num_train_examples))
	training_labels = training_labels == 3

	featurized_training_data = np.array(map(dcu.simple_image_featurization, training_data))
	return training_data, featurized_training_data, training_labels

def get_neuron_test_data():
	test_data = samples.loadDataFile("digitdata/testimages", 1000, 28,28)
	test_labels = np.array(samples.loadLabelsFile("digitdata/testlabels", 1000))
	test_labels = test_labels == 3	

	featurized_test_data = np.array(map(dcu.simple_image_featurization, test_data))
	return test_data, featurized_test_data, test_labels

"""
if you want a bias, then apply that bias to your data, then create a perceptron to identify digits

Next, train that perceptron on the entire set of training data num_times_to_train times on num_train_examples.

Finally, use the zero_one_loss defined in data_classification_utils to find the 
final accuracy on both the training set and the test set, assigning them to the 
variables training_accuracy and test_accuracy respectively"""

raw_training_data, featurized_training_data, training_labels = get_neuron_training_data()
raw_test_data, featurized_test_data, test_labels = get_neuron_test_data()

"""YOUR CODE HERE"""

neur = neuron.SigmoidNeuron(len(featurized_training_data[0]))
# print 'raw: ', str(raw_training_data[0])
# print 'feat: ', featurized_training_data
# print 'len: ', len(featurized_training_data[0])

# print 'training labels: ', training_labels
# print 'test labels: ', test_labels

# print(str(raw_training_data[20]))
# lst = [neur.classify(a) for a in featurized_training_data]
# print 'datts: ', lst

# print 'data: ', featurized_training_data[0]
# print 'classy: ', per.classify(featurized_training_data[0])

for _ in range(num_times_to_train):
	neur.train(featurized_training_data[0:num_train_examples], alpha, training_labels[0:num_train_examples])
# print 'class: ', neur.classify(featurized_training_data[22])



training_accuracy = dcu.zero_one_loss_ss(neur, featurized_training_data, training_labels)
test_accuracy = dcu.zero_one_loss_ss(neur, featurized_test_data, test_labels)
print('Final training accuracy: ' + str(training_accuracy) + '% correct')

print("Test accuracy: " + str(test_accuracy) + '% correct')
