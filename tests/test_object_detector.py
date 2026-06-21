import os
import urllib.request
import cv2
import pytest
from src.detectors.object_detector import ObjectDetector

# Fixtures directory
FIXTURES_DIR = os.path.join("tests", "fixtures")
os.makedirs(FIXTURES_DIR, exist_ok=True)

# Test images maps
IMAGE_URLS = {
    "empty_desk.jpg": "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=640",
    "phone_on_table.jpg": "https://images.unsplash.com/photo-1546054454-aa26e2b734c7?w=640",
    "two_people.jpg": "https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=640"
}


def download_fixture_if_missing(filename: str) -> str:
    """Helper to download a test fixture if it does not exist."""
    filepath = os.path.join(FIXTURES_DIR, filename)
    if not os.path.exists(filepath):
        url = IMAGE_URLS[filename]
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
            out_file.write(response.read())
    return filepath


@pytest.fixture
def empty_desk_frame():
    filepath = download_fixture_if_missing("empty_desk.jpg")
    frame = cv2.imread(filepath)
    assert frame is not None, "Failed to load empty desk frame"
    return frame


@pytest.fixture
def phone_frame():
    filepath = download_fixture_if_missing("phone_on_table.jpg")
    frame = cv2.imread(filepath)
    assert frame is not None, "Failed to load phone frame"
    return frame


@pytest.fixture
def two_people_frame():
    filepath = download_fixture_if_missing("two_people.jpg")
    frame = cv2.imread(filepath)
    assert frame is not None, "Failed to load two people frame"
    return frame


def test_empty_desk(empty_desk_frame):
    """Verifies detector output when no person and no devices are present."""
    detector = ObjectDetector()
    results = detector.analyze(empty_desk_frame)
    
    # 3 results returned
    assert len(results) == 3
    
    # Map by signal name
    res_dict = {res.signal_name: res for res in results}
    
    # person_count: count = 0, should trigger (count != 1)
    p_count = res_dict["person_count"]
    assert p_count.triggered is True
    assert p_count.confidence == 0.0
    assert p_count.details["count"] == 0
    
    # phone_detected: should not trigger
    phone = res_dict["phone_detected"]
    assert phone.triggered is False
    assert phone.confidence == 0.0
    
    # secondary_device_detected: should not trigger
    sec = res_dict["secondary_device_detected"]
    assert sec.triggered is False
    assert sec.confidence == 0.0


def test_phone_present(phone_frame):
    """Verifies detector output when a phone is present but no person."""
    detector = ObjectDetector()
    results = detector.analyze(phone_frame)
    
    assert len(results) == 3
    res_dict = {res.signal_name: res for res in results}
    
    # person_count: count = 0, should trigger
    p_count = res_dict["person_count"]
    assert p_count.triggered is True
    assert p_count.details["count"] == 0
    
    # phone_detected: should trigger
    phone = res_dict["phone_detected"]
    assert phone.triggered is True
    assert phone.confidence > 0.5
    
    # secondary_device_detected: should trigger
    sec = res_dict["secondary_device_detected"]
    assert sec.triggered is True
    assert sec.confidence > 0.5


def test_two_people_present(two_people_frame):
    """Verifies detector output when multiple people and devices are present."""
    detector = ObjectDetector()
    results = detector.analyze(two_people_frame)
    
    assert len(results) == 3
    res_dict = {res.signal_name: res for res in results}
    
    # person_count: should trigger because count > 1
    p_count = res_dict["person_count"]
    assert p_count.triggered is True
    assert p_count.details["count"] > 1
    assert p_count.confidence > 0.5
    
    # phone_detected: should trigger
    phone = res_dict["phone_detected"]
    assert phone.triggered is True
    assert phone.confidence > 0.5
    
    # secondary_device_detected: should trigger
    sec = res_dict["secondary_device_detected"]
    assert sec.triggered is True
    assert sec.confidence > 0.5


def test_frame_skipping_and_caching(empty_desk_frame):
    """Verifies caching behavior and state interpolation across skipped frames."""
    detector = ObjectDetector()
    
    # First frame: runs inference
    res1 = detector.analyze(empty_desk_frame)
    for r in res1:
        assert r.details["stale"] is False
        
    # Second frame: skipped, returns cached (stale) results
    res2 = detector.analyze(empty_desk_frame)
    for r in res2:
        assert r.details["stale"] is True
        
    # Third frame: skipped, returns cached (stale) results
    res3 = detector.analyze(empty_desk_frame)
    for r in res3:
        assert r.details["stale"] is True
        
    # Fourth frame: runs inference again (stale should be False)
    res4 = detector.analyze(empty_desk_frame)
    for r in res4:
        assert r.details["stale"] is False
