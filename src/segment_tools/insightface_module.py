import cv2
import insightface
import torch
from .utils import check_image_type


class Insightface:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda":
            self.app.prepare(ctx_id=0)
        else:
            self.app.prepare(ctx_id=-1)

    def run(self, image, no_image=False):
        """
        Runs the face detection and analysis on the given image.

        Args:
            image: The input image to perform face detection on.
            no_image: A boolean flag indicating whether to draw bounding boxes on the image or not.

        Returns:
            A dictionary containing the processed image (draw_img) and a list of detected faces (faces_list).
            Each face in the faces_list is represented as a dictionary with the following keys:
                - bbox: The bounding box coordinates of the face.
                - kps: The facial keypoints of the face.
                - det_score: The detection score of the face.
                - landmark_3d_68: The 3D landmark coordinates of the face.
                - pose: The pose estimation of the face.
                - landmark_2d_106: The 2D landmark coordinates of the face.
                - gender: The predicted gender of the face.
                - age: The predicted age of the face.
                - embedding: The face embedding vector.
        """
        image = check_image_type(image)
        faces = self.app.get(image)

        if no_image:
            draw_img = None
        else:
            draw_img = self.app.draw_on(image, faces)

        faces_list = []
        for face in faces:
            faces_list.append(
                {
                    "bbox": face.bbox,
                    "kps": face.kps,
                    "det_score": face.det_score,
                    "landmark_3d_68": face.landmark_3d_68,
                    "pose": face.pose,
                    "landmark_2d_106": face.landmark_2d_106,
                    "gender": face.gender,
                    "age": face.age,
                    "embedding": face.embedding,
                }
            )

        return {"image": draw_img, "faces": faces_list}

    def run_video(self, video_path, output_path, no_image=False):
        """
        Runs face detection on a video file and saves the output video with detected faces.

        Args:
            video_path (str): The path to the input video file.
            output_path (str): The path to save the output video file.
            no_image (bool, optional): Flag to indicate whether to include the detected faces in the output video. 
                                       Defaults to False.

        Returns:
            dict: A dictionary containing the path to the output video file and a list of detected faces for each frame.

        """
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            int(cap.get(cv2.CAP_PROP_FPS)),
            (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )

        faces_lists = []
        while cap.isOpened():
            faces_list = []
            ret, frame = cap.read()
            if not ret:
                break
            result = self.run(frame, no_image)
            faces_list.append(result["faces"])
            out.write(result["image"])
            faces_lists.append(faces_list)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return {"video": output_path, "faces": faces_lists}
