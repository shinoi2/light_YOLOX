import time
from concurrent import futures
import grpc
import light_pb2_grpc
import light_pb2
import numpy as np
import onnxruntime
import multiprocessing
import cv2
from yolox.data.data_augment import preproc as preprocess
from yolox.utils import multiclass_nms, demo_postprocess



_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class Service(light_pb2_grpc.LightServiceServicer):
    def __init__(self, model="./light.onnx"):
        providers = ['CUDAExecutionProvider']
        self.model = onnxruntime.InferenceSession(model, providers=providers)
        self.input_shape = (640, 640)

    def predict(self, request, context):
        count = len(request.images)
        images = []
        ratios = []
        for origin_image in request.images:
            origin_img = cv2.imdecode(np.frombuffer(origin_image, np.uint8), cv2.IMREAD_UNCHANGED)
            img, ratio = preprocess(origin_img, self.input_shape)
            images.append(img)
            ratios.append(ratio)
        images = np.array(images)
        ort_inputs = {self.model.get_inputs()[0].name: images}
        outputs = self.model.run(None, ort_inputs)
        predictions = demo_postprocess(outputs[0], self.input_shape)
        boxes = predictions[:, :, :4]
        scores = predictions[:, :, 4:5] * predictions[:, :, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2]/2.
        boxes_xyxy[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3]/2.
        boxes_xyxy[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2]/2.
        boxes_xyxy[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3]/2.
        boxes_xyxy /= np.array(ratios)[:, None, None]
        responses = []
        for i in range(count):
            dets = multiclass_nms(boxes_xyxy[i], scores[i], nms_thr=0.45, score_thr=0.1)
            lights = []
            if dets is not None:
                for left, top, right, bottom, score, _ in dets:
                    lights.append(
                        light_pb2.Light(
                            score=float(score),
                            rect=light_pb2.Rect(
                                left=float(left),
                                right=float(right),
                                top=float(top),
                                bottom=float(bottom)
                            )
                        )
                    )
            responses.append(light_pb2.Lights(lights=lights))

        return light_pb2.LightResponse(responses=responses)
        

def run():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()*5))
    print(multiprocessing.cpu_count())
    light_pb2_grpc.add_LightServiceServicer_to_server(Service(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    print("start service...")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    run()
