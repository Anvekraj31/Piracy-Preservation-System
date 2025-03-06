from tkinter import *
from tkinter import filedialog
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Global variables
main = None
train = None
cluster_labels = None

def uploadDataset():
    global dataset
    filename = filedialog.askopenfilename(initialdir=".", title="Select Dataset",
                                          filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*")))
    if filename:
        dataset = pd.read_csv(filename, nrows=100)
        text.insert(END, f"Dataset loaded: {filename}\n")
        text.insert(END, f"First 5 rows:\n{dataset.head()}\n")

def processDataset():
    global train, cluster_labels
    dataset.fillna(0, inplace=True)
    train = dataset[['latitude', 'longitude']]
    text.insert(END, f"Dataset preprocessed: {train.shape[0]} rows\n")
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(train)
    cluster_labels = kmeans.labels_
    text.insert(END, "KMeans clustering applied.\n")

def showConfusionMatrix():
    global cluster_labels, train
    if train is None or cluster_labels is None:
        text.insert(END, "Dataset not processed. Please preprocess the dataset first.\n")
        return

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(train)
    predictions = kmeans.predict(train)
    cm = confusion_matrix(cluster_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cluster 0", "Cluster 1"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# Tkinter GUI
main = Tk()
main.title("Privacy Preservation System Using Machine Learning")
main.geometry("800x600")

# GUI Components
font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=50, y=50)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50, y=100)
processButton.config(font=font1)

confusionButton = Button(main, text="Show Confusion Matrix", command=showConfusionMatrix)
confusionButton.place(x=50, y=150)
confusionButton.config(font=font1)

text = Text(main, height=20, width=70)
text.place(x=250, y=50)
text.config(font=font1)

main.mainloop()
