# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from debug_gui import DetectionMonitor, run_gui  # Import DetectionMonitor and run_gui
import multiprocessing


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(281, 163)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Title = QtWidgets.QLabel(self.centralwidget)
        self.Title.setGeometry(QtCore.QRect(70, 10, 141, 21))
        self.Title.setObjectName("Title")
        self.tf_id = QtWidgets.QLineEdit(self.centralwidget)
        self.tf_id.setGeometry(QtCore.QRect(40, 50, 221, 21))
        self.tf_id.setText("")
        self.tf_id.setObjectName("tf_id")
        self.tf_pw = QtWidgets.QLineEdit(self.centralwidget)
        self.tf_pw.setGeometry(QtCore.QRect(40, 80, 221, 20))
        self.tf_pw.setText("")
        self.tf_pw.setObjectName("tf_pw")
        self.Bt_login = QtWidgets.QPushButton(self.centralwidget)
        self.Bt_login.setGeometry(QtCore.QRect(10, 110, 251, 31))
        self.Bt_login.setObjectName("Bt_login")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 50, 21, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 80, 31, 16))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # Initialize shared data
        self.shared_data = multiprocessing.Manager().dict()
        self.shared_data["template_matching_results"] = "대기 중"
        self.shared_data["yolo_detection_results"] = "대기 중"
        self.shared_data["minimap_matching_results"] = "대기 중"

        self.retranslateUi(MainWindow)
        self.Bt_login.clicked.connect(self.on_login_clicked)  # Connect to new method
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Title.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt;\">squad M Login</span></p></body></html>"))
        self.tf_id.setToolTip(_translate("MainWindow", "<html><head/><body><p>ID</p></body></html>"))
        self.Bt_login.setText(_translate("MainWindow", "Login"))
        self.label.setText(_translate("MainWindow", "ID"))
        self.label_2.setText(_translate("MainWindow", "PW"))

    def on_login_clicked(self):
        # Create and show the GUI window
        self.gui_window = DetectionMonitor(self.shared_data)  # Pass shared_data to DetectionMonitor
        self.gui_window.show()
        # Optionally hide the login window
        self.Bt_login.window().hide()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
