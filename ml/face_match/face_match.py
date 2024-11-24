import os
import sys
sys.path.append(os.getcwd())

from typing import Tuple, List

import insightface

from app.utils.data_access import DataAccessImage
from ml import model_paths
from app.models.asset import FaceMatchResponse


FACE_DETECTION_THRESHOLD = 0.2
FACE_MATCH_THRESHOLD = 0.4

class Face_Dict:
    def __init__(self, kps, embedding):
        self.kps = kps
        self.embedding = embedding


class FaceMatch:
    def __init__(self, face_detection_model_path: str, face_embedding_model_path: str):
        self.face_detector = insightface.model_zoo.get_model(name=face_detection_model_path, providers=['CPUExecutionProvider'])
        self.face_detector.prepare(ctx_id=0, input_size=(640, 640))
        
        self.face_embedding =  insightface.model_zoo.get_model(name=face_embedding_model_path, providers=['CPUExecutionProvider'])
        self.face_embedding.prepare(ctx_id=0)


    def detect_faces(self, data_access_image: DataAccessImage) -> Tuple[bool, str, List[List]]:
        """
        returns correct num faces
        msg
        landmarks
        """
        bboxes, landmarks = self.face_detector.detect(img=data_access_image.get_bgr_image())
        # print(f"bboxes: {bboxes}, landmarks: {landmarks}")
        if len(bboxes) > 1:
            return False, "Multiple faces detected", [[]]
        if len(bboxes) == 0:
            return False, "No faces detected", [[]]
            
        x_left, y_bottom, x_right, y_top, probability = bboxes[0]
        if probability < FACE_DETECTION_THRESHOLD:
            return False, "No faces detected", [[]]

        return True, None, landmarks[0]

    def get_embeddings(self, data_access_image: DataAccessImage, lmks: List[List]):
        face = Face_Dict(kps=lmks, embedding=None)
        embedding = self.face_embedding.get(img=data_access_image.get_bgr_image(), face=face)
        return embedding
    
    def get_cosine_similarity(self, emb1, emb2):
        cosine_score = self.face_embedding.compute_sim(emb1, emb2)
        return round(cosine_score, 3)

    def match_faces(self, source_file_path: str, target_file_path: str) -> FaceMatchResponse:
        data_access_image_0 = DataAccessImage(file_path=source_file_path)
        data_access_image_1 = DataAccessImage(file_path=target_file_path)

        correct_num_faces_0, msg_0, lmks0 = self.detect_faces(data_access_image_0)
        if not correct_num_faces_0:
            return FaceMatchResponse(is_similar=False, msg=msg_0)

        correct_num_faces_1, msg_1, lmks1 = self.detect_faces(data_access_image_1)
        if not correct_num_faces_1:
            return FaceMatchResponse(is_similar=False, msg=msg_1)

        emb1 = self.get_embeddings(data_access_image=data_access_image_0, lmks=lmks0)
        emb2 = self.get_embeddings(data_access_image=data_access_image_1, lmks=lmks1)
        cosine_score = self.get_cosine_similarity(emb1=emb1, emb2=emb2)
        print(f"cosine_score: {cosine_score}")
        if cosine_score >= FACE_MATCH_THRESHOLD:
            return FaceMatchResponse(is_similar=True, msg="Face matched")
        return FaceMatchResponse(is_similar=False, msg="Face didn't match")



if __name__ == "__main__":
    file_path = '/Users/divyanshnew/Pictures/Photo on 10-01-23 at 5.52 PM.jpg'
    file_path = '/Users/divyanshnew/Pictures/TB4QEBX6H-U02SHJ7RL6B-a34702b57242-512.jpeg'

    file_path_2 = '/Users/divyanshnew/Pictures/Photo on 21-12-22 at 11.23 AM.jpg'
    face_matcher = FaceMatch(face_detection_model_path=model_paths.FACE_MATCH_SCRFD, face_embedding_model_path=model_paths.FACE_MATCH_EMBEDDING)
    # face_matcher.detect_faces(data_access_image=DataAccessImage(file_path=file_path))
    res = face_matcher.match_faces(source_file_path=file_path, target_file_path=file_path_2)
    print(res)
