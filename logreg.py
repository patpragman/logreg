#  Logistic Regression Model using sci-kit learn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import wandb
import numpy as np
from skimage import color, io
import os
import pickle
from cm_handler import display_and_save_cm
from pprint import pformat
from pathlib import Path
HOME_DIRECTORY = Path.home()

# start up wandb!

wandb.init(
    project="Elodea LogReg", notes="trying to build the classifier with logistic regression, testing all the images",
)

sizes = [512, 256, 224, 128, 64]
#sizes.reverse()  # start with the smaller problem

results_filename = 'log_reg_results.md'
if os.path.isfile(results_filename):
    os.remove(results_filename)  # delete the old one, make a new one!

with open(results_filename, "w") as results_file:
    results_file.write("# Results for Logistic Regression:\n")


folder_paths = [f"{HOME_DIRECTORY}/0.35_reduced_then_balanced/data_{size}" for size in sizes]

for size, dataset_path in zip(sizes, folder_paths):
    wandb.log({"msg": f"Logistic Regression Model for {size} x {size} images"})

    # variables to hold our data
    data = []
    Y = []

    classifier = LogisticRegression(class_weight='balanced', max_iter=3000)
    classes = os.listdir(dataset_path)

    excluded = ["pkl", ".DS_Store", "md", "png"]
    classes = [str(klass).split("/")[-1] for klass in Path(dataset_path).iterdir()
               if klass.is_dir()]

    # create a mapping from the classes to each number class and demapping
    mapping = {n: i for i, n in enumerate(classes)}
    demapping = {i: n for i, n in enumerate(classes)}

    # now create an encoder
    encoder = lambda s: mapping[s]
    decoder = lambda i: demapping[i]

    # now walk through and load the data in the containers we constructed above
    for root, dirs, files in os.walk(dataset_path):

        for file in files:
            if ".JPEG" in file.upper() or ".JPG" in file.upper() or ".PNG" in file.upper():
                key = root.split("/")[-1]

                if "cm" in file:
                    continue
                else:
                    img = io.imread(f"{root}/{file}", as_gray=True)
                    arr = np.asarray(img).reshape(size * size, )  # reshape into an array
                    data.append(arr)

                    Y.append(encoder(key))  # simple one hot encoding

    y = np.array(Y)
    X = np.array(data)

    # now we've loaded all the X values into a single array
    # and all the Y values into another one, let's do a train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=42)  # for consistency

    # now fit the classifier
    # fit the model with data
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    cr = classification_report(
        y_test, y_pred, target_names=[key for key in mapping.keys()], output_dict=True
    )

    wandb.log(cr)
    # save it
    with open(f"{dataset_path}/log_reg_{size}.pkl", "wb") as file:
        pickle.dump(classifier, file)

    display_and_save_cm(
        y_actual=y_test, y_pred=y_pred, labels=[key for key in mapping.keys()],
        name=f"Logistic Regression Image Size {size}x{size}"
    )

    wandb.log({"msg": f"{size}x{size} complete!"})

    for d in [cr[k] for k in mapping.keys()]:
        wandb.log(d)

    with open(results_filename, "a") as outfile:
        outfile.write(f"{size}x{size} images\n")
        outfile.write(classification_report(
            y_test, y_pred, target_names=[key for key in mapping.keys()]
        ))
        outfile.write("\n")
