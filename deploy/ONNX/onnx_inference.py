import time
import cv2
import numpy as np
import onnxruntime

from utils import xywh2xyxy, nms, draw_detections
#from yolov6.data.data_augment import letterbox


class YOLOv6:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        output = self.inference(input_tensor)

        # Process output data
        self.boxes, self.scores, self.class_ids = self.process_output(output)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        #input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = image

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        #input_img = letterbox(input_img, (self.input_height, self.input_width))[0]

        # Scale input pixel values to 0 to 1
        #input_img = input_img / 255.0
        #input_img = (input_img - 127.0) / 128.0 
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output)

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        predictions = predictions[obj_conf > self.conf_threshold]
        scores = scores[scores > self.conf_threshold]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes /= np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    #model_path = "/world/data-gpu-94/liyang/Github_projects/YOLOv6/runs/train/exp/weights/yolov6n_head_det.v1/best_ckpt.onnx"
    model_path = "/world/data-gpu-94/liyang/Github_projects/YOLOv6/runs/train/exp/weights/yolov6t_head_det.v1/last_ckpt.onnx"

    # Initialize YOLOv6 object detector
    yolov6_detector = YOLOv6(model_path, conf_thres=0.35, iou_thres=0.5)

    img_path = "/world/data-gpu-94/liyang/pedDetection/head_detection/badcase/ped3_badcase_frames/1795.jpg"
    image = cv2.imread(img_path)
    # Detect Objects
    yolov6_detector(image)
    # Draw detections
    combined_img = yolov6_detector.draw_detections(image)
    cv2.imwrite("./1.jpg", combined_img)
    print("Finished")

    json_path = "/world/data-gpu-94/liyang/pedDetection/head_detection/badcase/ped_head.badcase.gt.json"
    save_file = open("./eval/onnx_res.txt", 'w') 
    with open(json_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("\t")
            k = line[0]
            img = cv2.imread(k)
            xyxy, confs, classes = yolov6_detector(img)
            for i in range(xyxy.shape[0]): 
                x0 = float(xyxy[i][0])
                y0 = float(xyxy[i][1])
                x1 = float(xyxy[i][2])
                y1 = float(xyxy[i][3])
                conf = float(confs[i])
                cls = int(classes[i]) 
                save_file.write("{:s} {:.4f} {:.1f} {:.1f} {:.1f} {:.1f} {}".format(k, conf, x0, y0, x1, y1, cls) + "\n")
    save_file.close()        
            
