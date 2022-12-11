#!/usr/bin/env python3
import cv2

'''
Python 7 nodarbības mājasdarbs Nr.2

Uzdevums: aizpildīt vietas ar atzīmi TODO

Izveidot klasi, kura pārveido 5. nodarbības mājasdarbu Nr. 3 saturu par klasi
'''

class IntroOpenCV:
    '''
    Izveidot klasi, kurai ir 5 publiskas metodes:
    - setBilde -  definē bildes failu
    - bilde 
    - melnbalts
    - EdgeDetection
    - ZilsSarkans

    !Klasei nav neviena publiski pieejami parametri!
    '''
    def set_picture(self,BildeFails):
        self.__picture = cv2.imread(BildeFails)

    def get_picture(self):
        return self.__picture

    def get_black_white(self):
        img_grayscale = cv2.cvtColor(self.__picture,cv2.COLOR_BGR2GRAY)
        return img_grayscale

    def get_edge_detection(self):
        img_canny = cv2.Canny(image=self.__picture, threshold1=100, threshold2=200)
        return img_canny

    def get_blue_red(self):
        img_zils_sarkans = cv2.cvtColor(self.__picture, cv2.COLOR_BGR2RGB)
        return img_zils_sarkans


if __name__ == "__main__":
    obj = IntroOpenCV()
    obj.set_picture("python.jpg")
    obj.get_black_white()
