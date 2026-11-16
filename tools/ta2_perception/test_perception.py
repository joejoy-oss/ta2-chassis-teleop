import json
import tempfile
import time
import unittest
from pathlib import Path
import cv2
import numpy as np
from core import Board, Stereo, Detector, crop, expire, fuse, ray_plane, rigid, sandbag_geometry, bottle_geometry
from server import Worker


def calibration():
    K = [[240, 0, 160], [0, 240, 120], [0, 0, 1]]
    return {"pair": "head", "regions_version": "test", "size": [320, 240],
            "validation": {"passed": True, "metric_error_m": .001},
            "K1": K, "K2": K, "D1": [0]*5, "D2": [0]*5,
            "R": np.eye(3).tolist(), "T": [-.06, 0, 0]}


class GeometryTests(unittest.TestCase):
    def test_board_roundtrip(self):
        board = Board({"squares_x": 7, "squares_y": 5, "square_m": .03, "marker_m": .022})
        im = board.board.generateImage((1400, 1000))
        found = board.corners(im)
        self.assertEqual(len(found[1]), 24)

    def test_stereo_metric_plane_and_bad_geometry(self):
        stereo = Stereo(calibration(), "test", "head", [(320, 240)]*2)
        rng = np.random.default_rng(42)
        left = rng.integers(0, 255, (240, 320, 3), np.uint8)
        # fx*b / disparity = 240*.06/24 = .6 m
        right = np.zeros_like(left); right[:, :-24] = left[:, 24:]
        points, valid, disparity = stereo.depth(left, right)
        use = valid.copy(); use[:, :200] = False; use[:, -30:] = False
        self.assertGreater(use.sum(), 1000)
        self.assertAlmostEqual(float(np.median(points[:, :, 2][use])), .6, delta=.005)
        with self.assertRaises(ValueError): Stereo(calibration(), "changed", "head", [(320, 240)]*2)
        bad = calibration(); bad["T"][0] *= -1
        with self.assertRaises(ValueError): Stereo(bad, "test", "head", [(320, 240)]*2)
        with self.assertRaises(ValueError): Stereo(calibration(), "test", "head", [(160, 120)]*2)

    def test_expiry_revokes_every_coordinate(self):
        r = {"received_at": 1, "state": "robot_coordinates", "target": {"position": [1,2,3]},
             "grasp_candidates": [1], "objects": [{"position_table_m": [1,2,3], "position_robot_m": [4,5,6], "obstacle": [1]}]}
        stale = expire(r, now=2)
        self.assertIsNone(stale["target"])
        self.assertEqual(stale["grasp_candidates"], [])
        self.assertIsNone(stale["objects"][0]["position_robot_m"])
        self.assertIsNotNone(r["target"])

    def test_fusion_conflict_and_uncertainty_floor(self):
        a = dict(position=[0,0,.03], sigma=[.005]*3, source="head", surface_candidate=[0,0,.03], yaw_rad=None)
        b = {**a,"source":"left_wrist","position":[.1,0,.03]}
        self.assertEqual(fuse([a,b])[1], "view_conflict")
        b["position"] = [.002,0,.03]
        result, reason = fuse([a,b])
        self.assertIsNone(reason)
        self.assertAlmostEqual(result["position"][0],.001)
        self.assertGreaterEqual(result["sigma"][0],.005)

    def test_unknown_bottle_cannot_be_free_space(self):
        self.assertIsNone(bottle_geometry(np.ones((30,30)),np.eye(3),np.eye(4),None))
        dims={"width_m":.1,"depth_m":.1,"height_m":.2}
        self.assertIsNone(bottle_geometry(np.ones((30,30)),np.eye(3),np.eye(4),dims))

    def test_plane_and_rigid(self):
        T=np.diag([1,-1,-1,1.]);T[2,3]=1
        p,valid=ray_plane(np.array([[0.,0.]]),np.eye(3),T)
        self.assertTrue(valid[0]);np.testing.assert_allclose(p,[[0,0,0]])
        self.assertTrue(rigid(T));T[0,0]=2;self.assertFalse(rigid(T))

    def test_no_model_is_not_red_object_detector(self):
        d=Detector("")
        self.assertIsNotNone(d.error)
        self.assertEqual(d.predict([np.full((32,32,3),[0,0,255],np.uint8)]),[[]])

    def test_worker_missing_calibration_and_model(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            config=json.loads(Path(__file__).with_name("config.json").read_text())
            config["data_dir"]=tmp
            path=Path(tmp)/"config.json";path.write_text(json.dumps(config))
            w=Worker(path)
            regions={n:[0,0,1,1] for n in ['head_left_eye','head_right_eye','left_wrist_left_eye','left_wrist_right_eye','right_wrist_left_eye','right_wrist_right_eye']}
            packet=dict(session_id="test",seq=1,regions=regions,regions_version="v",width=320,height=240,received_at=time.time(),timestamp_kind="host_decode_receive")
            r=w.process(packet,np.zeros((240,320,3),np.uint8))
            self.assertEqual(r["state"],"model_missing")
            self.assertEqual(r["grasp_candidates"],[])
            self.assertEqual(len(r["views"]),6)
            self.assertIsNone(r["target"])


if __name__ == '__main__': unittest.main()
