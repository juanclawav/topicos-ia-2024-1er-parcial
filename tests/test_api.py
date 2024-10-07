import pytest
import numpy as np
from src.predictor import annotate_detection
from src.models import Detection, PredictionType, Segmentation
from src.predictor import GunDetector, annotate_segmentation, match_gun_bbox
from fastapi.testclient import TestClient
from src.main import app 
import os

client = TestClient(app)

TEST_IMAGE_PATH = fr"test_image.jpg"

@pytest.fixture(scope="module")
def test_image():
    with open(TEST_IMAGE_PATH, "rb") as f:
        return f.read()

def test_get_model_info():
    expected_data = {
        "model_name": "Gun detector",
        "gun_detector_model": "DetectionModel",
        "semantic_segmentation_model": "SegmentationModel",
        "input_type": "image",
    }
    response = client.get("/model_info")
    assert response.status_code == 200
    assert response.json() == expected_data

def test_detect_guns(test_image):
    response = client.post(
        "/detect_guns",
        files={"file": ("test_image.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    data = response.json()
    assert all(key in data for key in ["n_detections", "boxes", "labels", "confidences"])
    assert data["pred_type"] == "OD"

def test_annotate_guns(test_image):
    response = client.post(
        "/annotate_guns",
        files={"file": ("test_image.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    with open(fr"annotated_guns.jpg", "wb") as f:
        f.write(response.content)

def test_detect_people(test_image):
    response = client.post(
        "/detect_people",
        files={"file": ("test_image.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    data = response.json()
    assert all(key in data for key in ["n_detections", "polygons", "boxes", "labels"])
    assert data["pred_type"] == "SEG"

def test_annotate_people(test_image):
    response = client.post(
        "/annotate_people",
        files={"file": ("test_image.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5", "draw_boxes": "true"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    with open(fr"annotated_people.jpg", "wb") as f:
        f.write(response.content)

def test_detect(test_image):
    response = client.post(
        "/detect",
        files={"file": ("test_image.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5"}
    )
    assert response.status_code == 200
    data = response.json()
    
    detection = data["detection"]
    segmentation = data["segmentation"]
    
    assert all(key in detection for key in ["n_detections", "boxes", "labels", "confidences"])
    assert detection["pred_type"] == "OD"
    
    assert all(key in segmentation for key in ["n_detections", "polygons", "boxes", "labels"])
    assert segmentation["pred_type"] == "SEG"

def test_annotate(test_image):
    response = client.post(
        "/annotate",
        files={"file": ("test_image.jpg", test_image, "image/jpeg")},
        data={"threshold": "0.5", "draw_boxes": "true"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    with open(fr"annotated_combined.jpg", "wb") as f:
        f.write(response.content)