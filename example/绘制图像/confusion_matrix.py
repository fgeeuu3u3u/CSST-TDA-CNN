##########  python confusion_matrix.py

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

cm = np.array([[261, 39], [9, 18408]])

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stars', 'Galaxies'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_sklearn.png', dpi=150)
plt.show()