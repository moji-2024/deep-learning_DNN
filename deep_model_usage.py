import class_deepLearning
import pandas as pd
from PIL import Image
import numpy as np
import os

model = class_deepLearning.deepLearner()
# # # sample.initialize_parameters_deep([2,4,5,1])
df_taining = pd.DataFrame({
    'Standards_Concentration': [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000],
    'abs_mean': [1.1915, 1.008, 0.733, 0.557, 0.4245, 0.253, 0.176, 0.107, 0.088]
})
df_taining['Standards_Concentration'] = df_taining['Standards_Concentration'] / 2000
y_train = df_taining['Standards_Concentration'].values.reshape(1,-1) # shape (1, n_samples)
X_train = df_taining[['abs_mean']].T.values  # shape (n_x, m)

layers_dims = [1,3,4,1]
model.fit(X_train,
           y_train,
           layers_dims, print_cost=False,
           activation_output=False,
           num_iterations=9500,
           learning_rate=0.075)
print('--------------------------------------------')
print('parameters= ',model.parameters)
print('--------------------------------------------')
print('prediction of X=[[0.4245,0.7330]] is',model.predict([[0.4245,0.7330]],output_activation=False) * 2000)
print('--------------------------------------------')
df_test = pd.DataFrame({
    'abs_mean': [1.1915, 1.008, 0.733, 0.557, 0.4245, 0.253, 0.176, 0.107, 0.088]
})
x_test = np.array(df_test).T
print('x_test',x_test)
print('--------------------------------------------')
pred_all = model.predict(x_test,output_activation=False) * 2000
print("\nPredictions for x_test array:\n", pred_all)


print('------------------Second usage--------------------------')


def load_images_from_folder(folder, target_size=(64, 64)):
    images = []
    labels = []
    images_name = []
    for filename in os.listdir(folder):
        if filename.endswith((".jpg", ".png")):
            # img = Image.open(os.path.join(folder, filename)).convert("RGB")
            img = Image.open(os.path.join(folder, filename)).convert("L")
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0  # normalize
            img_flat = img_array.flatten()     # shape (n_x,)
            images.append(img_flat)
            # Example: use folder name or file naming convention for labels
            labels.append(1 if "cat" in filename else 0)
            images_name.append('Cat' if "cat" in filename else 'Other')
    return np.array(images).T, np.array(labels), images_name  # shape: (n_x, m), (1, m) , (1, m)

X_train, Y_train, _ = load_images_from_folder(r".\train_images")
print(X_train.shape, Y_train.shape)
print('--------------------------------------------')
X_test, true_label, true_names = load_images_from_folder(r".\test_images")
layers_dims = [X_train.shape[0], 50, 30, 10, 1]  # Example: input layer size matches n_x
print('layers_dims: ',layers_dims)
print('--------------------------------------------')
model2 = class_deepLearning.deepLearner()
model2.fit(X_train, Y_train.reshape(1, -1),
           layers_dims,
           # num_iterations=8100,
           num_iterations=6000,
           learning_rate=0.075,
           print_cost=True)
predictions = model2.predict(X_test, output_activation=True)
print('predictions: ',predictions)
print('true_label: ',true_label)
print('true_names: ',true_names)

list_predictions = predictions[0]
Tools_Samples = len(true_names)
False_Negatives = len([index for index in range(Tools_Samples) if (list_predictions[index] != true_label[index]) and (list_predictions[index] == 0)])
False_Positives = len([index for index in range(Tools_Samples) if (list_predictions[index] != true_label[index]) and (list_predictions[index] == 1)])
True_Negatives = len([index for index in range(Tools_Samples) if (list_predictions[index] == true_label[index]) and (list_predictions[index] == 0)])
True_Positives = len([index for index in range(Tools_Samples) if (list_predictions[index] == true_label[index]) and (list_predictions[index] == 1)])
Accuracy = (True_Negatives + True_Positives) / Tools_Samples # Out of all predictions, what percent were correct?
Precision = True_Positives / (True_Positives + False_Positives) # High precision means almost every “yes” is actually a yes.
Recall = True_Positives / (True_Positives + False_Negatives) # High recall means few positives are missed.
F1_score = (Precision * Recall / (Precision + Recall)) * 2 # How good is my balance between precision and recall?

print(f"Out of all predictions: Accuracy: {Accuracy} → {Accuracy * 100}% of all predictions were correct.")
print(f"Out of all predictions: Precision: {Precision} → Of the positives predicted, {Precision * 100}% were actually positive.")
print(f"Out of all predictions: Recall: {Recall} → The model caught {Recall * 100}% actual positives.")
print(f"Out of all predictions: F1-score: {F1_score} →  is my balance between precision and recall")


