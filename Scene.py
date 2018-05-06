import numpy as np
from sklearn.cluster import MiniBatchKMeans
import DisjointSet
from LK import *
from HS import *

class Scene:

    def __init__(self, orig, accumulate,optical_flow='LK'):
        self = self
        if optical_flow == 'LK':
            self.model = LK()
        else:
            self.model = HS()                                                           # Optical Flow method (Lucas Kanade/ Horn Schunck)
        self.orig = orig                                                                # Original frame captured from video
        self.prvs = np.zeros(self.orig.shape)                                            # Previous frame used last calculation
        self.curr = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float64)  # Current frame used need to be calculated
        self.op_flow = np.zeros_like(self.prvs)                                         # Optical Flow result
        self.objs = []                                                                  # Set of objects detected so far
        self.diff = np.zeros(self.orig.shape)
        self.threshold = np.max(self.diff) * 0.1
        self.centers = []
        self.output = orig
        self.accumulate = accumulate
        self.road = np.zeros(self.curr.shape)
        self.name_of_flow = optical_flow

    def update_scene(self, orig):
        self.orig = orig
        self.output = orig
        self.prvs = self.curr
        self.curr = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float64)
        self.diff = abs(self.curr - self.prvs)
        self.road += self.diff 
        self.diff[self.accumulate != 0] = 0
        self.threshold = np.max(self.diff) * 0.1
        self.op_flow = self.model.optical_flow(self.curr, self.prvs)
        self.centers = self.__k_means()
        self.__drawRectangle()
        return self.output

    def __k_means(self):
        newImage = np.zeros(self.diff.shape)
        trainSet = []
        row = self.diff.shape[0]
        col = self.diff.shape[1]
        for i in range(row):
            for j in range(col):
                if self.diff[i][j] > self.threshold:
                    trainSet.append([i, j])
        trainSet = np.array(trainSet)
        kmeans = MiniBatchKMeans(n_clusters = 100).fit(trainSet)
        return kmeans.cluster_centers_.astype(np.int32)

    def __getMinDistance(self):
        minDiff = 1000
        for ctr1 in self.centers:
            for ctr2 in self.centers:
                diff = ctr2 - ctr1
                distance = (diff[0]**2 + diff[1]**2)**0.5
                if distance > 0: minDiff = np.minimum(minDiff, distance)
        return minDiff

    def __drawRectangle(self):
        DJSet = DisjointSet.DisjointSet()
        vehicleThreshold = self.__getMinDistance() * 5
        for i in range(self.centers.size // 2):
            for j in range(self.centers.size // 2):
                if i == j:
                    continue
                diff = self.centers[j] - self.centers[i]
                if (diff[0]**2 + diff[1]**2)**0.5 < vehicleThreshold:
                    DJSet.add(tuple(self.centers[i]), tuple(self.centers[j]))
        for leader in DJSet.group:
            if len(DJSet.group[leader]) < 7:
                continue
            frame = [leader[0], leader[1], leader[0], leader[1]]
            for member in DJSet.group[leader]:
                if member[0] < frame[0]:
                    frame[0] = member[0]
                elif member[0] > frame[2]:
                    frame[2] = member[0]
                if member[1] < frame[1]:
                    frame[1] = member[1]
                elif member[1] > frame[3]:
                    frame[3] = member[1]
            if frame[3] - frame[1] == 0 or frame[2] - frame[0] == 0:
                continue
            if self.name_of_flow == 'HS':
                speed = self.__speed_test(frame)
            else:
                speed = self.__speed_test_lucas(frame)
            """
            if speed < 5 or frame[3] - frame[1] < self.orig.shape[1] * 50 or frame[3] - frame[1] > self.orig.shape[1] * 0.5 \
                or frame[2] - frame[0] < self.orig.shape[0] * 50 or frame[2] - frame[0] > self.orig.shape[0] * 0.5:
                continue
            """
            color = (255, 255, 255)
            if speed < 40:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(self.output, (frame[1], frame[0]), (frame[3], frame[2]), color, 3)
            cv2.putText(self.output, str(np.around(speed)), (frame[1] + 5, frame[2] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def __speed_test(self, frame):
        x_max = frame[3]
        x_min = frame[1]
        y_max = frame[2]
        y_min = frame[0]
        car = np.array(self.op_flow[y_min:y_max, x_min:x_max])
        speed = np.zeros(len(car))
        for i in range(len(car)):
            speed[i] = np.max(car[i, :])
        return np.median(speed) / (2.5 * (y_max * 1.0 / self.orig.shape[0]))

    def __speed_test_lucas(self, frame):
        x_max = frame[3]
        x_min = frame[1]
        y_max = frame[2]
        y_min = frame[0]
        car = np.array(self.op_flow[y_min:y_max, x_min:x_max])
        speed = np.zeros(len(car))
        for i in range(len(car)):
            speed[i] = np.max(car[i, :])
        return np.median(speed) / (0.15 * (y_max * 1.0 / self.orig.shape[0]))

