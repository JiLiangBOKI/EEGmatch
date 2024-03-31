import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox, QComboBox, QSpinBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from scipy.io import loadmat
from tqdm import tqdm


# Define Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Neural Network Classifier")
        self.setGeometry(100, 100, 900, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        # Load Data Button
        self.button_load_data = QPushButton("Load Data", self)
        self.button_load_data.setToolTip("Click to load data")
        self.button_load_data.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button_load_data.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px;")

        # Train Model Button
        self.button_train = QPushButton("Train Model", self)
        self.button_train.setToolTip("Click to train model")
        self.button_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button_train.setStyleSheet("background-color: #008CBA; color: white; border-radius: 10px; padding: 10px;")

        # Load Model Button
        self.button_load_model = QPushButton("Load Model", self)
        self.button_load_model.setToolTip("Click to load model")
        self.button_load_model.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button_load_model.setStyleSheet("background-color: #f44336; color: white; border-radius: 10px; padding: 10px;")

        # Close Button
        self.button_close = QPushButton("Close", self)
        self.button_close.setToolTip("Click to close application")
        self.button_close.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button_close.setStyleSheet("background-color: #FF5733; color: white; border-radius: 10px; padding: 10px;")

        # Horizontal Layout for Buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button_load_data)
        button_layout.addWidget(self.button_train)
        button_layout.addWidget(self.button_load_model)
        button_layout.addWidget(self.button_close)

        self.layout.addLayout(button_layout)

        # Data Load Information Label
        self.label_info = QLabel("No data loaded yet.", self)
        self.label_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label_info)

        # Separator Label
        self.label_separator = QLabel("------------------------------", self)
        self.label_separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label_separator)

        # Model Selection
        self.model_selection_label = QLabel("Select Model:", self)
        self.layout.addWidget(self.model_selection_label)
        self.model_selection_combo = QComboBox(self)
        self.model_selection_combo.addItems(["Neural Network", "Support Vector Machine", "Decision Tree"])
        self.layout.addWidget(self.model_selection_combo)

        # Model Parameters
        self.model_params_label = QLabel("Model Parameters:", self)
        self.layout.addWidget(self.model_params_label)
        self.num_epochs_spinbox = QSpinBox(self)
        self.num_epochs_spinbox.setRange(1, 1000)
        self.num_epochs_spinbox.setValue(100)
        self.num_epochs_spinbox.setSingleStep(10)
        self.layout.addWidget(self.num_epochs_spinbox)

        # Chart 1
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111, projection='3d')

        # Chart 2
        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        self.layout.addWidget(self.canvas2)

        self.ax2 = self.figure2.add_subplot(111, projection='3d')

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_bar.setStyleSheet("QProgressBar {border: 2px solid grey; border-radius: 5px; background-color: #f0f0f0;}"
                                         "QProgressBar::chunk {background-color: #4CAF50;}")
        self.layout.addWidget(self.progress_bar)

        # Accuracy Label
        self.label_accuracy = QLabel("Accuracy: ", self)
        self.label_accuracy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label_accuracy)

        self.data = None
        self.final_model_filename = "final_model.pth"  # 保存最终模型的文件名

        # Connect Buttons to Functions
        self.button_load_data.clicked.connect(self.load_data)
        self.button_train.clicked.connect(self.train_model)
        self.button_load_model.clicked.connect(self.load_saved_model)
        self.button_close.clicked.connect(self.close_application)

    def load_data(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("MAT files (*.mat)")
        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            if filenames:
                filename = filenames[0]
                self.data = loadmat(filename)
                self.label_info.setText(f"Data loaded from: {filename}")
                self.plot_original_data()

    def plot_original_data(self):
        if self.data is not None:
            features = self.data['struct_fl_set'][:, :100]
            labels = self.data['struct_fl_set'][:, 100]

            self.ax.clear()
            self.ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=labels, cmap='viridis', s=10)
            self.ax.set_title('Distribution of Original Data')
            self.ax.set_xlabel('Feature 1')
            self.ax.set_ylabel('Feature 2')
            self.ax.set_zlabel('Feature 3')
            self.canvas.draw()

    def train_model(self):
        if self.data is not None:
            features = torch.tensor(self.data['struct_fl_set'][:, :100], dtype=torch.float32)
            labels = torch.tensor(self.data['struct_fl_set'][:, 100], dtype=torch.long)

            numFold = 10
            skf = StratifiedKFold(n_splits=numFold)

            input_size = features.shape[1]
            output_size = len(torch.unique(labels))
            model = NeuralNetwork(input_size, output_size)

            accuracies = np.zeros(numFold)

            for i, (trainIdx, testIdx) in enumerate(skf.split(features, labels)):
                trainFeatures, testFeatures = features[trainIdx], features[testIdx]
                trainLabels, testLabels = labels[trainIdx], labels[testIdx]

                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                num_epochs = self.num_epochs_spinbox.value()  # Get user-defined number of epochs
                for epoch in tqdm(range(num_epochs), desc=f'Fold {i + 1}/{numFold}'):
                    optimizer.zero_grad()
                    outputs = model(trainFeatures)
                    loss = criterion(outputs, trainLabels)
                    loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    outputs = model(testFeatures)
                    _, predictedLabels = torch.max(outputs, 1)

                correctPredictions = (predictedLabels == testLabels).sum().item()
                accuracy = correctPredictions / len(testLabels)
                accuracies[i] = accuracy

                self.progress_bar.setValue((i + 1) * 100 // numFold)
                self.plot_predicted_labels(model, features)

            # 保存最终模型
            self.save_model(model, self.final_model_filename)
            print(f"Final model saved to {self.final_model_filename}")

            averageAccuracy = np.mean(accuracies)
            accuracyVariance = np.var(accuracies)
            self.label_accuracy.setText(f"Average accuracy: {averageAccuracy * 100:.2f}% | Accuracy variance: {accuracyVariance:.6f}")

    def plot_predicted_labels(self, model, features):
        with torch.no_grad():
            predictedLabels = model(features)
            _, predictedLabels = torch.max(predictedLabels, 1)

        self.ax2.clear()
        self.ax2.scatter(features[:, 0], features[:, 1], features[:, 2], c=predictedLabels, cmap='viridis', s=10)
        self.ax2.set_title('Distribution of Predicted Labels')
        self.ax2.set_xlabel('Feature 1')
        self.ax2.set_ylabel('Feature 2')
        self.ax2.set_zlabel('Feature 3')
        self.canvas2.draw()

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, model, filename):
        model.load_state_dict(torch.load(filename))
        print(f"Model loaded from {filename}")

    def load_saved_model(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Model files (*.pth)")
        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            if filenames:
                filename = filenames[0]
                if self.data is not None:
                    input_size = self.data['struct_fl_set'].shape[1] - 1
                    output_size = len(torch.unique(torch.tensor(self.data['struct_fl_set'][:, 100], dtype=torch.long)))
                    model = NeuralNetwork(input_size, output_size)
                    self.load_model(model, filename)
                    self.plot_predicted_labels(model, torch.tensor(self.data['struct_fl_set'][:, :100], dtype=torch.float32))
                else:
                    print("No data loaded yet.")

    def close_application(self):
        choice = QMessageBox.question(self, 'Exit', "Are you sure you want to exit?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if choice == QMessageBox.StandardButton.Yes:
            sys.exit()


def main():
    app = QApplication(sys.argv)

    # Set dark theme
    app.setStyle("Fusion")
    dark_palette = app.palette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(142, 45, 197).lighter())
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
