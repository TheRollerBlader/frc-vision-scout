import numpy as np
import pytest

from evaluator import (
    DetectionBox,
    cluster_alliance_bumpers,
    compute_hsv_color_ratios,
    draw_annotated_frame,
    parse_video_stem_metadata,
)


def _make_box(
    cls_id: int,
    cx: float,
    cy: float,
    conf: float = 0.85,
    img_w: int = 1000,
    img_h: int = 500,
    in_field_bounds: bool = True,
    color_purity_ok: bool = True,
) -> DetectionBox:
    w = 40.0
    h = 30.0
    return DetectionBox(
        cls_id=cls_id,
        cls_name="bumper_red" if cls_id == 0 else "bumper_blue",
        conf=conf,
        x1=cx - w / 2,
        y1=cy - h / 2,
        x2=cx + w / 2,
        y2=cy + h / 2,
        cx=cx,
        cy=cy,
        w=w,
        h=h,
        norm_cx=cx / img_w,
        norm_cy=cy / img_h,
        norm_area=(w * h) / (img_w * img_h),
        color_purity_ok=color_purity_ok,
        in_field_bounds=in_field_bounds,
    )


class TestEvaluatorInvariants:
    def test_parse_video_stem_metadata_extracts_match_fields(self):
        meta = parse_video_stem_metadata("2026_2026txirv_2026txirv_sf2m1_abc_XYZ")
        assert meta["year"] == 2026
        assert meta["event_key"] == "2026txirv"
        assert meta["match_key"] == "2026txirv_sf2m1"
        assert meta["comp_level"] == "sf"
        assert meta["match_number"] == 2
        assert meta["youtube_id"] == "abc_XYZ"

    def test_parse_video_stem_metadata_handles_minimal_stem_fallback(self):
        meta = parse_video_stem_metadata("unknown_event")
        assert meta["year"] == 2026
        assert meta["event_key"] == "event"
        assert meta["comp_level"] == "qm"
        assert meta["match_number"] == 1
        assert meta["youtube_id"] == ""

    def test_compute_hsv_color_ratios_distinguishes_red_blue_yellow_and_empty(self):
        assert compute_hsv_color_ratios(np.empty((0, 0, 3), dtype=np.uint8)) == (0.0, 0.0, 0.0)

        # Low-saturation gray image should be filtered out as non-colored
        gray_crop = np.full((20, 20, 3), 120, dtype=np.uint8)
        assert compute_hsv_color_ratios(gray_crop) == (0.0, 0.0, 0.0)

        # Pure BGR red, blue, and gold/yellow crops
        red_crop = np.zeros((20, 20, 3), dtype=np.uint8)
        red_crop[:, :, 2] = 220
        r, b, y = compute_hsv_color_ratios(red_crop)
        assert r == pytest.approx(1.0)
        assert b == 0.0
        assert y == 0.0

        blue_crop = np.zeros((20, 20, 3), dtype=np.uint8)
        blue_crop[:, :, 0] = 220
        r, b, y = compute_hsv_color_ratios(blue_crop)
        assert b == pytest.approx(1.0)
        assert r == 0.0

        yellow_crop = np.zeros((20, 20, 3), dtype=np.uint8)
        yellow_crop[:, :, 1] = 220
        yellow_crop[:, :, 2] = 220
        r, b, y = compute_hsv_color_ratios(yellow_crop)
        assert y == pytest.approx(1.0)

    def test_cluster_alliance_bumpers_merges_split_segments_and_ignores_oob(self):
        detections = [
            # Two segments on the same red chassis (norm dx=0.02 < 0.085)
            _make_box(0, cx=200.0, cy=250.0, conf=0.72),
            _make_box(0, cx=220.0, cy=255.0, conf=0.91),
            # Distinct distant red robot
            _make_box(0, cx=700.0, cy=300.0, conf=0.80),
            # Out-of-field false positive in scoreboard margin must be excluded
            _make_box(0, cx=500.0, cy=20.0, conf=0.99, in_field_bounds=False),
            # Blue robot should not enter red cluster list
            _make_box(1, cx=400.0, cy=250.0, conf=0.88),
        ]

        red_clusters = cluster_alliance_bumpers(detections, alliance_cls_id=0, img_w=1000, img_h=500)
        assert len(red_clusters) == 2
        # Sorted by max_conf descending, cluster_id renumbered 1..N
        assert red_clusters[0].cluster_id == 1
        assert red_clusters[0].max_conf == pytest.approx(0.91)
        assert red_clusters[0].num_bumper_segments == 2
        assert red_clusters[1].cluster_id == 2
        assert red_clusters[1].max_conf == pytest.approx(0.80)
        assert red_clusters[1].num_bumper_segments == 1

    def test_draw_annotated_frame_preserves_input_and_handles_excess_clusters(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        original_copy = frame.copy()

        dets = [_make_box(0, cx=100 + i * 110, cy=200, conf=0.9 - i * 0.05, img_w=640, img_h=360) for i in range(4)]
        red_clusters = cluster_alliance_bumpers(dets, alliance_cls_id=0, img_w=640, img_h=360)
        assert len(red_clusters) == 4  # triggers >3 excess FP path

        annotated = draw_annotated_frame(
            img_bgr=frame,
            detections=dets,
            red_clusters=red_clusters,
            blue_clusters=[],
            meta={"match_key": "2026txirv_qm1", "year": 2026},
            frame_label="middle_1",
        )

        assert annotated.shape == frame.shape
        assert np.array_equal(frame, original_copy)
        assert not np.array_equal(annotated, frame)
