from data_loader import MNISTDataLoader
from network import MLP
import numpy as np

loader = MNISTDataLoader("data/mnist_2000.csv", normalize=True, seed=21)
x_train, y_train, x_test, y_test = loader.train_test_split(test_size=0.3)

layers = [784, 128, 32, 10]
activations = ["relu", "relu", "softmax"]
model = MLP(layers, activations, seed= 67)

epochs = 100
learning_rate = 0.01
batch_size = 64

loss_history = []
accuracy_history = []


for epoch in range(epochs):

    epoch_loss = 0
    num_batches = 0

    for batch_x, batch_y in loader.batches(
        x_train,
        y_train,
        batch_size
    ):
        probs = model.forward(batch_x, training=True)
        loss = model.cross_entropy_loss(probs, batch_y)

        grad_w, grad_b = model.backward(batch_y)
        model.update_parameters(grad_w, grad_b, learning_rate)


        epoch_loss += loss
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    acc = model.accuracy(x_train, y_train)

    loss_history.append(avg_loss)
    accuracy_history.append(acc)

model.classification_report(x_test, y_test)

best_epoch = np.argmax(accuracy_history)
best_accuracy = accuracy_history[best_epoch]

print()
print(f"Best Epoch        : {best_epoch + 1}")
print(f"Best Accuracy     : {best_accuracy:.2%}")
print(f"Total improvement : {accuracy_history[-1] - accuracy_history[0]}")


# Use the line below to save the model architecture, weights and biases 
# model.save("models/mnist_model.npz")