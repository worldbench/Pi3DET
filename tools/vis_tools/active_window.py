import sys
from PyQt5 import QtWidgets
import sys
from windows.main_window import MainWindow

def main():
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':

    main()