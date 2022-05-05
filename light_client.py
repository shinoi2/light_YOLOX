import light_pb2_grpc
import light_pb2
import grpc
import cv2
import numpy as np

if __name__ == "__main__":
    with open("./image/image/2.jpeg", "rb") as f:
        image = f.read()
    image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_UNCHANGED)
    print(len(image.shape))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.imencode('.jpg', image)[1].tobytes()
    conn = grpc.insecure_channel('192.168.66.239:50052')
    client = light_pb2_grpc.LightServiceStub(channel=conn)
    print(len(image))
    request = light_pb2.LightRequest(images=[image])
    response = client.predict(request)
    print(response)
