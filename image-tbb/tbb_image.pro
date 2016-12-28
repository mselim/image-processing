TEMPLATE = app
CONFIG += console c++14
CONFIG -= app_bundle
CONFIG -= qt

TBB = '/home/mselim/Development/Libraries/tbb2017_20160916oss'
OSG = '/usr/local'



INCLUDEPATH +=$$TBB/include
INCLUDEPATH +=$$OSG/include

LIBS += -L$$TBB/lib -ltbb
LIBS += -L$$OSG/lib64 -losg -losgGA -losgDB -losgUtil -losgViewer -lOpenThreads

SOURCES += main.cpp
