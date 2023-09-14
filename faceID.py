import cv2
import numpy as np
from tqdm import tqdm
from .models import Emotion, Age, Gender, Race 
from .tools import functions

from .basemodels import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, DlibWrapper, ArcFace, SFace

def build_model(model_name):
    """
    This function builds a deepface model.

    Parameters:
        model_name (str): Face recognition or facial attribute model.

    Returns:
        Built deepface model.
    """

    # Singleton design pattern
    global model_obj

    # Dictionary mapping model names to their corresponding loading functions
    model_loading_functions = {
        "VGG-Face": VGGFace.loadModel,
        "OpenFace": OpenFace.loadModel,
        "Facenet": Facenet.loadModel,
        "Facenet512": Facenet512.loadModel,
        "DeepFace": FbDeepFace.loadModel,
        "DeepID": DeepID.loadModel,
        "Dlib": DlibWrapper.loadModel,
        "ArcFace": ArcFace.loadModel,
        "SFace": SFace.load_model,
        "Emotion": Emotion.loadModel,
        "Age": Age.loadModel,
        "Gender": Gender.loadModel,
        "Race": Race.loadModel,
    }

    if "model_obj" not in globals():
        model_obj = {}

    if model_name not in model_obj:
        model_loading_function = model_loading_functions.get(model_name)
        if model_loading_function:
            model = model_loading_function()
            model_obj[model_name] = model
        else:
            raise ValueError(f"Invalid model_name passed - {model_name}")

    return model_obj[model_name]


def analyze(img_path, actions=("emotion", "age", "gender", "race"),
            enforce_detection=True, detector_backend="opencv", align=True, silent=False):
    """
    This function analyzes facial attributes including age, gender, emotion and race.

    Parameters:
        img_path (str): Exact image path, numpy array (BGR), or base64 encoded image.
        actions (tuple): The default is ('age', 'gender', 'emotion', 'race'). You can drop some attributes.
        enforce_detection (bool): Whether to throw an exception if no face is detected.
        detector_backend (str): Face detector backend to use.
        align (bool): Whether to align according to eye positions.
        silent (bool): Disable log messages.

    Returns:
        List of dictionaries for each detected face.
    """

    # Validate actions
    if isinstance(actions, str):
        actions = (actions,)
    if not actions or not hasattr(actions, "__getitem__"):
        raise ValueError("`actions` must be a non-empty list of strings.")
    valid_actions = {"emotion", "age", "gender", "race"}
    if not all(action in valid_actions for action in actions):
        raise ValueError(f"Invalid action(s) in `actions` parameter.")

    # Build models
    models = {}
    if "emotion" in actions:
        models["emotion"] = build_model("Emotion")
    if "age" in actions:
        models["age"] = build_model("Age")
    if "gender" in actions:
        models["gender"] = build_model("Gender")
    if "race" in actions:
        models["race"] = build_model("Race")

    resp_objects = []

    img_objs = functions.extract_faces(
        img=img_path,
        target_size=(224, 224),
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    for img_content, img_region, _ in img_objs:
        if img_content.shape[0] > 0 and img_content.shape[1] > 0:
            obj = {"region": img_region}

            # Facial attribute analysis
            pbar = tqdm(actions, desc="Finding actions", disable=silent)
            for action in pbar:
                pbar.set_description(f"Action: {action}")

                if action == "emotion":
                    img_gray = cv2.cvtColor(img_content[0], cv2.COLOR_BGR2GRAY)
                    img_gray = cv2.resize(img_gray, (48, 48))
                    img_gray = np.expand_dims(img_gray, axis=0)
                    emotion_predictions = models["emotion"].predict(img_gray, verbose=0)[0, :]
                    sum_of_predictions = emotion_predictions.sum()

                    obj["emotion"] = {label: 100 * val / sum_of_predictions for label, val in
                                      zip(Emotion.labels, emotion_predictions)}
                    obj["dominant_emotion"] = Emotion.labels[np.argmax(emotion_predictions)]

                elif action == "age":
                    age_predictions = models["age"].predict(img_content, verbose=0)[0, :]
                    apparent_age = Age.findApparentAge(age_predictions)
                    obj["age"] = int(apparent_age)

                elif action == "gender":
                    gender_predictions = models["gender"].predict(img_content, verbose=0)[0, :]
                    obj["gender"] = {label: 100 * val for label, val in zip(Gender.labels, gender_predictions)}
                    obj["dominant_gender"] = Gender.labels[np.argmax(gender_predictions)]

                elif action == "race":
                    race_predictions = models["race"].predict(img_content, verbose=0)[0, :]
                    sum_of_predictions = race_predictions.sum()

                    obj["race"] = {label: 100 * val / sum_of_predictions for label, val in
                                   zip(Race.labels, race_predictions)}
                    obj["dominant_race"] = Race.labels[np.argmax(race_predictions)]

            resp_objects.append(obj)

    return resp_objects
