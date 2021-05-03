import mnist_loader
import fcnetwork


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = fcnetwork.FullNetwork([784, 10, 3], activation_function='sigmoid')
net.gradient_descent(training_data, 30, 10, 3, bootstrap=True)