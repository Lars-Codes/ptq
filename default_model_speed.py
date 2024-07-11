import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
from super_gradients.common.object_names import Models
from super_gradients.training import models
import os 
import supervision as sv
import cProfile
import pstats

# https://discuss.bluerobotics.com/t/opencv-python-with-gstreamer-backend/8842

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

HOME = os.getcwd()
CKPT_PATH=f"{HOME}/checkpoints/vehicle_images/RUN_20240626_101822_978143/ckpt_best.pth"

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using YOLO-NAS and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the inference on (default: cuda)",
        type=str,
    )

    return parser.parse_args()

if __name__ == "__main__":

    print("info: ", cv2.getBuildInformation()) # uses ffmpeg or gstreamer? 

    with cProfile.Profile() as profile:

        args = parse_arguments()

        HOME = os.getcwd()

        device = args.device 

        video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
        model = models.get("yolo_nas_s", checkpoint_path=CKPT_PATH, num_classes=1)
        # model = models.get("yolo_nas_s", pretrained_weights="coco") # Get YOLONAS model (s, m, or l). Replace with optimized, custom-trained PyTorch/OpenVINO model 
        model.to(device)

        byte_track = sv.ByteTrack(
            frame_rate=video_info.fps, track_thresh=0.3
        )

        thickness = sv.calculate_optimal_line_thickness(
            resolution_wh=video_info.resolution_wh
        )
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
        bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER,
        )

        frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)
        view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

        coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

        vid = cv2.VideoCapture(args.source_video_path, cv2.CAP_GSTREAMER)

        while(vid.isOpened()): 
            ret, frame = vid.read()
            result = model.predict(frame)
            detections = sv.Detections.from_yolo_nas(result)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)

            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            annotated_frame = cv2.resize(annotated_frame, (539, 620)) # resize frame to fit

            cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    if args.device == "gpu":
        filename = "profile_cpu.txt"
    else:
        filename = "profile_gpu.txt"

    with open(filename, "w") as f: 
        results = pstats.Stats(profile, stream=f)
        results.sort_stats(pstats.SortKey.TIME)
        results.print_stats()