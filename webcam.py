from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import time


GRAPH = 'graph/graph'
IMAGE = 'images/cat.jpg'
CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
input_size = (300, 300)
np.random.seed(3)
colors = 255 * np.random.rand(len(CLASSES), 3)

# discover our device
devices = mvnc.EnumerateDevices()
device = mvnc.Device(devices[0])
device.OpenDevice()

# load graph onto the device
with open(GRAPH, 'rb') as f:
    graph_file = f.read()

graph = device.AllocateGraph(graph_file)


def preprocess(src):
    img = cv2.resize(src, input_size)
    img = img - 127.5
    img = img / 127.5
    return img.astype(np.float16)


# graph => load the image to it, return a prediction
capture = cv2.VideoCapture(0)
_, image = capture.read()
height, width = image.shape[:2]

while True:
    stime = time.time()
    _, image = capture.read()
    image_pro = preprocess(image)
    graph.LoadTensor(image_pro, None)
    output, _ = graph.GetResult()

    valid_boxes = int(output[0])

    for i in range(7, 7 * (1 + valid_boxes), 7):
        if not np.isfinite(sum(output[i + 1: i + 7])):
            continue
        clss = CLASSES[int(output[i + 1])]
        conf = output[i + 2]
        color = colors[int(output[i + 1])]

        x1 = max(0, int(output[i + 3] * width))
        y1 = max(0, int(output[i + 4] * height))
        x2 = min(width, int(output[i + 5] * width))
        y2 = min(height, int(output[i + 6] * height))

        label = '{}: {:.0f}%'.format(clss, conf * 100)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        y = y1 - 5 if y1 - 15 > 15 else y1 + 18
        image = cv2.putText(image, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            color, 2)
    cv2.imshow('frame', image)
    print('FPS = {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
device.CloseDevice()
