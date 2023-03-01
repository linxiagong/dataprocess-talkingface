import torch
import face_alignment

class FaceAlignmentLandmark:
    def __init__(self, lms_type:str) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.lms_type = lms_type
        if lms_type == '2D':
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=self.device)
        elif lms_type == '3D':
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=self.device)
        else:
            raise NotImplementedError

    def generate_face_alignment_lms(self, img_list: list) -> list:
        lms = []
        for img in img_list:
            preds = self.fa.get_landmarks(img)
            if preds and len(preds) > 0:
                lands = preds[0].reshape(-1, 2)[:, :2]
                lms.append(lands)
            else:
                lms.append(None)
        return lms