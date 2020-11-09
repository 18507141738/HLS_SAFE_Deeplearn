# -*- coding: utf-8 -*-
import numpy as np
import time
import cv2
import config as cfg
import logging
import os
from PIL import Image, ImageDraw, ImageFont


class CycleList(object):
    def __init__(self,length):
        self._len = length
        self._pointer = 0
        self._container = []
        self._full = False


    def insert(self,item):
        if not self._full:
            self._container.append(item)
            if len(self._container) >= self._len:
                self._full = True
        else:
            self._container[self._pointer] = item
        self._pointer = (self._pointer+1) %self._len

    def get_len(self):
        return self._len

    def delete_last_item(self):
        del(self._container[self._pointer])
        self._full = False
        if self._pointer<0:
            self._pointer = len(self._container)-1

    def get_numpy_points(self):
        '''
        Assert that a item is [] or [(x0,y0),(x1,y1),...]
        :return: numpy array as array([[x0,y0],[x1,y1],...,[xn,yn]])
        '''
        points_list = []
        for points in self._container:
            for point in points:
                points_list.append(point)
        return np.array(points_list)

    def get_points(self):
        '''
        Assert that a item is [] or [(x0,y0),(x1,y1),...]
        :return: numpy array as array([[x0,y0],[x1,y1],...,[xn,yn]])
        '''
        # points_list = []
        # for points in self._container:
        #     points_list.append(points)
        return self._container

class PointsRecorder(object):
    def __init__(self,record_len = 30, num_thr = 25,dist_thr = 100):
        self.CycleList = CycleList(record_len)
        self.num_thr = num_thr
        self.dist_thr = dist_thr

    def update(self,centerPoints):
        self.CycleList.insert(centerPoints)

    def warning_flag(self, centerPoint):
        # warning = False
        np_record = self.CycleList.get_numpy_points()
        if len(np_record) ==0:
            return False
        mid_pt = np.median(np_record,axis=0)
        centerPoint = np.array(centerPoint)
        if np.sqrt(np.sum((mid_pt - centerPoint) ** 2)) > self.dist_thr:
            return False
        count = 0
        for point in np_record:
            if np.sqrt(np.sum((mid_pt - point) ** 2)) < self.dist_thr:
                count += 1
        # logging.info('Current num of warnings: %d.' % count)
        return (count>self.num_thr)

    def warning_flag1(self):
        record = self.CycleList.get_points()
        # print 22, record
        count = 0
        if len(record) >= 1:
            # if record[-1] == 1:
            for i in record:
                count += i
        return (count > self.num_thr)


    def getrecord(self):
        return self.CycleList.get_points()



