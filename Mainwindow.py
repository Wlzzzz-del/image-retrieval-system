from PyQt5.QtWidgets import QRadioButton,QPushButton,QFileDialog, QGridLayout,QApplication,QGroupBox,QVBoxLayout
from PyQt5.QtWidgets import QLineEdit,QDialog,QProgressBar,QScrollArea,QWidget,QLabel,QComboBox
from PyQt5.QtCore import Qt,pyqtSignal
from PyQt5.QtGui import QPixmap,QMouseEvent
from progressbar import progressbar
import qdarkstyle
import sys
import ft
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

import os
import scipy.io as io
import math
import numpy as np
import sys

class MyQLabel(QLabel):
    """
        自定义可点击的label
    """
    button_clicked_signal = pyqtSignal()

    def __init__(self,parent=None):
        super(MyQLabel, self).__init__(parent)

    def mouseReleaseEvent(self, QMouseEvent) -> None:
        self.button_clicked_signal.emit()
    
    def connect_customized_slot(self,func):
        self.button_clicked_signal.connect(func)


class WidegtGallery(QDialog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("图像检索系统")
        self.resize(1500,900)


        main_layout = QGridLayout()
        self.setLayout(main_layout)

        self.ini_group = QGroupBox("初始化")
        self.create_ini()
        main_layout.addWidget(self.ini_group,0,0)

        self.frame1 = QGroupBox("检索模块")
        self.create_search()
        main_layout.addWidget(self.frame1,1,0)

        self.picframe = QGroupBox("检索结果")
        self.create_pic()
        main_layout.addWidget(self.picframe,0,1,2,2)

    def create_search(self):
        """
            创建检索界面
        """
        layout = QGridLayout()
        self.frame1.setLayout(layout)
        up_button = QPushButton("选择检索图像")
        self.combobutton = QComboBox()
        sear_button = QPushButton("搜索")
        self.imgpath = QLineEdit("待检索图像")
        self.sear_img = MyQLabel()
        self.sear_img.setStyleSheet("border:1px solid gray")

        self.combobutton.addItems(["基于语义(VGG16)","基于内容(SIFT)"])
        self.combobutton.setToolTip("选择使用的算法")
        self.combobutton.setItemData(0,"一种基于语义的检索方法，\n以图像的含义为检测标准，\n速度较快且准确率较高",Qt.ToolTipRole)
        self.combobutton.setItemData(1,"一种基于内容的检索方法，\n以物体形状为检测标准，\n速度最快但牺牲部分准确率",Qt.ToolTipRole)
        # 固定相框大小
        self.sear_img.setMaximumSize(300,300)
        self.sear_img.setMinimumSize(300,300)
        layout.setContentsMargins(30,30,30,30)
        layout.setSpacing(10)
        layout.addWidget(up_button,0,0)
        layout.addWidget(self.combobutton,1,0)
        layout.addWidget(sear_button,2,0)
        layout.addWidget(self.imgpath,3,0)
        layout.addWidget(self.sear_img,4,0)

        up_button.clicked.connect(self.search_)
        sear_button.clicked.connect(self.sear_)

        pass
    def search_(self):
        """
            检索事件中的选择图像
        """
        path = QFileDialog.getOpenFileName(self,"选择图像..",filter="图像(*.jpg;*.png)")
        self.imgpath.setText(path[0])
        photo = QPixmap()
        photo.load(path[0])
        self.sear_img.setPixmap(photo)
        pass

    def sear_(self):
        """
            检索事件
        """
        if(self.combobutton.currentText()=="基于语义(VGG16)"):
            mat = io.loadmat("VGG16_LIB.mat")
            conv_base = VGG16(weights="imagenet",include_top=False, input_shape=(150,150,3)) 
        elif(self.combobutton.currentText()=="基于内容(SIFT)"):
            mat = io.loadmat("VGG19_LIB.mat")
            conv_base = VGG19(weights="imagenet",include_top=False, input_shape=(150,150,3)) 
        feature1 = ft.extract_feature(conv_base,self.imgpath.text())
        dis_dict = dict()

        for i in mat:
            try:
                dis = mat[i]-feature1
                dis_dict[i] = np.linalg.norm(dis)
            except:pass
            # 待检测图像
            # 排序
        dis_dict = sorted(dis_dict.items(),key=lambda x:x[1],reverse=False)[:20]

        for i in range(len(dis_dict)):
            path = dis_dict[i][0].replace("/","\\")
            photo = QPixmap()
            photo.load(dis_dict[i][0])
            self.label_list[i].setPixmap(photo)
            self.label_list[i].setToolTip(path)
            self.label_list[i].connect_customized_slot(lambda:os.system("explorer.exe /select,%s" % path))
        pass

    def create_pic(self):
        """
            用于展示检索结果的界面
        """
        layout = QGridLayout()
        la = QGridLayout()
        total = 20
        sc = QScrollArea()
        sc.setMinimumSize(100,100)
        sc.setFixedSize(1200,1000)
        layout.addWidget(sc,0,0)
        a = QWidget()

        self.label_list = []
        # 添加显示的图像
        for i in range(total):
            photo = QPixmap()
            path = i
            label = MyQLabel()
            label.setMinimumSize(300,300)
            self.label_list.append(label)
            label.setPixmap(photo)
            label.setToolTip(str(path))
            label.setStyleSheet("border:1px solid gray")
            col = math.floor(i/4)
            row = i-col*4
            la.addWidget(label,col,row)
        a.setLayout(la)
        sc.setWidget(a)
        photo.load("data/noise_test/noise_test\image_0002.jpg")

        self.picframe.setLayout(layout)
        pass

    def create_ini(self):
        """
            创建初始化布局
        """

        # 定义控件
        ini_button = QPushButton("初始化")
        upini_button = QPushButton("选择图像库")
        self.libpath = QLineEdit("图像库路径")
        Linelabel = QLabel("初始化进度")

        group = QGroupBox("进度")
        gr_l = QGridLayout()
        gr_l.setContentsMargins(0,0,0,0)
        self.ini_bar = QProgressBar()
        gr_l.addWidget(self.ini_bar)
        group.setLayout(gr_l)

        # 把长度定长一点
        ini_button.setMinimumSize(100,60)
        ini_button.setMaximumSize(100,60)

        upini_button.setMaximumSize(100,60)
        upini_button.setMinimumSize(100,60)

        #self.libpath.setMaximumHeight(50)
        #self.libpath.setMinimumWidth(100)

        # 添加控件
        input_layout = QGridLayout()
        input_layout.setContentsMargins(30,30,30,30)
        # input_layout.addWidget(Linelabel)
        input_layout.addWidget(self.ini_bar,0,0,1,2)
        # input_layout.addWidget(group,0,0,2,2)
        input_layout.addWidget(upini_button,1,1)
        input_layout.addWidget(ini_button,1,0)
        """
        input_layout.addWidget(Linelabel,0,0)
        input_layout.addWidget(self.ini_bar,0,1)
        input_layout.addWidget(upini_button,1,0)
        input_layout.addWidget(ini_button,2,0)
        input_layout.addWidget(self.libpath)
        self.libpath.setVisible(False)
        """

        upini_button.clicked.connect(self.loadlib)
        ini_button.clicked.connect(self.ini_lib)

        self.ini_group.setLayout(input_layout)

    def loadlib(self):
        """
            加载图像库文件，连接于初始化界面
        """
        path = QFileDialog.getExistingDirectory(self,"选择图像库路径..")
        self.libpath.setText(path)

    def ini_lib(self):
        """
            初始化图像数据库
        """

        dir = self.libpath.text()
        conv_base16 = VGG16(weights="imagenet",include_top=False, input_shape=(150,150,3)) 
        conv_base19 = VGG19(weights="imagenet",include_top=False, input_shape=(150,150,3)) 
        
        feature_dic = dict()
        feature_dic19 = dict()

        self.ini_bar.setMinimum(0)
        for dirname,_,filesname in os.walk(dir):
            all = len(filesname)
            num = 0
            self.ini_bar.setMaximum(all)
            for i in filesname:
                path = dir+'/'+i
                feature = ft.extract_feature(conv_base16,path)
                feature_dic[path]=feature

                feature = ft.extract_feature(conv_base19,path)
                feature_dic19[path]=feature
                num+=1
                self.ini_bar.setValue(num)
        io.savemat("VGG16_LIB",feature_dic)
        io.savemat("VGG19_LIB",feature_dic19)

        pass



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    dialog = WidegtGallery()
    if dialog.exec_() == QDialog.Accepted:
        sys.exit(app.exec_())