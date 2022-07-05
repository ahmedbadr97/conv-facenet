from .face_recongition import faces_features, verify_faces, load_image
from .face_detection import detect_face
from .utils import findCosineDistance, euclidean_distance
from torch.cuda import is_available

if "cuda_available" not in globals():
    cuda_available = is_available()
