from mission.adapters.localization_adapter import LocalizationAdapter
from mission.types import LocalizationResult
from localization.quality import STATUS_INVALID_HOMOGRAPHY, validate_homography_matrix, validate_match_quality


class _NullLocalizer:
    def localize(self, frame):
        return None


class _SuccessLocalizer:
    def localize(self, frame):
        return {
            "H": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "method": "map_matching",
            "confidence": 0.8,
            "num_keypoints": 50,
            "matches": 30,
            "inliers": 20,
            "inlier_ratio": 2 / 3,
            "reprojection_error": 1.25,
            "message": "SUCCESS",
        }


def test_successful_localization_result_conversion():
    adapter = LocalizationAdapter()
    adapter.localizer = _SuccessLocalizer()

    result = adapter.localize(object())

    assert isinstance(result, LocalizationResult)
    assert result.localized is True
    assert result.success is True
    assert result.num_keypoints == 50
    assert result.num_matches == 30
    assert result.num_inliers == 20
    assert result.reprojection_error == 1.25
    assert result.to_json()["homography_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_failed_localization_does_not_crash_adapter():
    adapter = LocalizationAdapter()
    adapter.localizer = _NullLocalizer()

    result = adapter.localize(object())

    assert result.localized is False
    assert result.success is False
    assert result.message == "localization_failed"


def test_invalid_homography_is_rejected():
    valid, status = validate_homography_matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]])

    assert valid is False
    assert status == STATUS_INVALID_HOMOGRAPHY


def test_quality_thresholds_reject_low_inlier_ratio():
    valid, status = validate_match_quality(
        homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        num_matches=20,
        num_inliers=5,
        inlier_ratio=0.25,
        min_matches=20,
        min_inliers=5,
        min_inlier_ratio=0.4,
    )

    assert valid is False
    assert status == "FAILED_LOW_INLIER_RATIO"
