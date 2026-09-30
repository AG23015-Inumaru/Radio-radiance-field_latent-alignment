import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from rrf_alignment.alignment import RFMapEncoder, pair_caches, split_views, training_statistics
from rrf_alignment.cameras import camera_rays, load_views
from rrf_alignment.newrf import FrozenNeWRF, _module
from rrf_alignment.vae import GrayVectorVAE, VAEConfig, load_frozen_vae

torch.set_num_threads(1)


def view(i, receiver=None, x=None):
    pose = np.eye(4)
    pose[0, 3] = i if x is None else x
    return {"view_id": str(i), "receiver_id": str(i if receiver is None else receiver),
            "image": f"/image/{i}.png", "camera_to_world": pose.tolist(),
            "intrinsics": {"width": 3, "height": 3, "fx": 1.5, "fy": 1.5, "cx": 1.5, "cy": 1.5},
            "rf_origin": [pose[0, 3], 0, 0], "camera_signature": f"camera{i}", "split": None}


class CameraAndDataTests(unittest.TestCase):
    def test_center_corner_and_similarity_transform(self):
        transform = np.eye(4)
        transform[:3, :3] *= 2
        transform[:3, 3] = [3, 4, 5]
        doc = {"camera_convention": "opencv", "rf_from_world": transform.tolist()}
        origins, rays = camera_rays(view(1), doc, 3, 3)
        np.testing.assert_allclose(origins[4], [5, 4, 5])
        np.testing.assert_allclose(rays[4], [0, 0, 1])
        np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1, rtol=1e-6)
        self.assertLess(rays[0, 0], 0)
        self.assertLess(rays[0, 1], 0)
        doc["camera_convention"] = "opengl"
        _, gl = camera_rays(view(1), doc, 3, 3)
        np.testing.assert_allclose(gl[4], [0, 0, -1])
        self.assertGreater(gl[0, 1], 0)

    def test_pose_rotation_changes_the_forward_ray(self):
        v = view(0)
        v["camera_to_world"] = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
        _, rays = camera_rays(v, {"camera_convention": "opencv", "rf_from_world": np.eye(4).tolist()}, 3, 3)
        np.testing.assert_allclose(rays[4], [1, 0, 0])

    def test_rejects_unregistered_or_invalid_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "views.json"
            doc = {"schema_version": 1, "camera_convention": "opencv", "views": [view(0)]}
            path.write_text(json.dumps(doc))
            with self.assertRaises(KeyError):
                load_views(path)
            doc["rf_from_world"] = np.eye(4).tolist()
            doc["rf_from_world"][0][0] = 2  # nonuniform scale would bend the camera geometry
            path.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):
                load_views(path)

    def test_pairing_uses_view_id_and_checks_camera(self):
        features = {"kind": "rf_features", "schema_version": 1,
                    "features": torch.ones(2, 3, 4, 6), "views": [view(0), view(1)]}
        targets = {"kind": "visual_means", "schema_version": 1,
                   "mu": torch.tensor([[20.0], [10.0]]), "views": [view(1), view(0)]}
        _, y, _ = pair_caches(features, targets)
        torch.testing.assert_close(y[:, 0], torch.tensor([10.0, 20.0]))
        targets["views"][0]["camera_signature"] = "another-camera"
        with self.assertRaisesRegex(ValueError, "camera_signature"):
            pair_caches(features, targets)

    def test_rejects_zero_mass_before_training(self):
        features = {"kind": "rf_features", "schema_version": 1,
                    "features": torch.zeros(2, 3, 4, 6), "views": [view(0), view(1)]}
        targets = {"kind": "visual_means", "schema_version": 1,
                   "mu": torch.zeros(2, 2), "views": [view(0), view(1)]}
        with self.assertRaisesRegex(ValueError, "mass is zero"):
            pair_caches(features, targets)

    def test_receiver_groups_and_aliases_never_cross_splits(self):
        views = [view(i) for i in range(8)] + [view(8, receiver="alias", x=0)]
        result = split_views(views)
        for indices in result.values():
            self.assertEqual(0 in indices, 8 in indices)
        views[0]["split"] = "train"
        views[8]["split"] = "val"
        for v in views[1:8]:
            v["split"] = "train"
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            split_views(views)

    def test_normalization_never_uses_held_out_rows(self):
        x = torch.tensor([1.0, 3.0, 10000.0])[:, None, None, None].expand(-1, 2, 4, 6)
        y = torch.tensor([[2.0], [4.0], [9000.0]])
        stats = training_statistics(x, y, [0, 1])
        torch.testing.assert_close(stats["x_mean"], torch.full((1, 2, 1, 1), 2.0))
        torch.testing.assert_close(stats["y_mean"], torch.tensor([[3.0]]))


class LearningTests(unittest.TestCase):
    def test_cnn_can_fit_small_latent_targets(self):
        torch.manual_seed(4)
        x, y = torch.randn(6, 5, 16, 24), torch.randn(6, 3)
        model = RFMapEncoder(5, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        initial = float((model(x) - y).square().mean().detach())
        for _ in range(70):
            optimizer.zero_grad()
            loss = (model(x) - y).square().mean()
            loss.backward()
            optimizer.step()
        self.assertLess(float((model(x) - y).square().mean().detach()), initial * 0.05)

    def test_previous_vae_checkpoint_keys_and_freezing(self):
        cfg = VAEConfig(image_height=16, image_width=24, latent_dim=3, channels=(8, 16))
        model = GrayVectorVAE(cfg).eval()
        x = torch.rand(1, 1, 16, 24)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vae.pt"
            config_path = Path(directory) / "vae.json"
            config_path.write_text(json.dumps(asdict(cfg)))
            for key in ("model_state", "model_state_dict"):
                torch.save({key: model.state_dict(), "config": asdict(cfg)}, path)
                loaded = load_frozen_vae(path, config_path)
                self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
                torch.testing.assert_close(loaded.encode(x)[0], model.encode(x)[0])
                latent = torch.randn(1, 3, requires_grad=True)
                loaded.decode(latent).mean().backward()
                self.assertGreater(float(latent.grad.abs().sum()), 0)
                self.assertTrue(all(p.grad is None for p in loaded.parameters()))


@unittest.skipUnless(os.environ.get("NEWRF_TEST_ROOT"), "Set NEWRF_TEST_ROOT to test the real external NeWRF code")
class ExternalNeWRFTests(unittest.TestCase):
    def test_checkpoint_hook_sampling_and_chunking(self):
        root = Path(os.environ["NEWRF_TEST_ROOT"])
        cfg = yaml.safe_load((root / "config/default.yaml").read_text())
        m, e = _module(root, "models"), _module(root, "encoders")
        ec, mc = cfg["encoder"], cfg["models"]
        enc = e.PositionalEncoder(ec["d_input"], ec["n_freqs"])
        directions = e.PositionalEncoder(ec["d_input"], ec["n_freqs_views"])
        torch.manual_seed(3)
        state = {}
        for kind in ("coarse", "fine"):
            suffix = "_fine" if kind == "fine" else ""
            net = m.NeWRF(enc.d_output, mc[f"n_layers{suffix}"], mc[f"d_filter{suffix}"],
                          tuple(mc["skip"]), directions.d_output)
            with torch.no_grad():
                net.alpha_out.weight.zero_()
                net.alpha_out.bias.fill_(1.0)
            state[f"{kind}_model_state_dict"] = net.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "synthetic.pt"
            torch.save(state, checkpoint)
            origins = np.array([[0.0, 0, 0], [1.0, 0, 0]], dtype=np.float32)
            rays = np.array([[0.0, 0, 1], [0.0, 1, 0]], dtype=np.float32)
            for kind in ("coarse", "fine"):
                field = FrozenNeWRF(root, root / "config/default.yaml", checkpoint, kind)
                a, stats = field.extract(origins, rays, near=0.01, far=3, samples=12,
                                         fine_samples=8, ray_chunk=1, point_chunk=5)
                b, _ = field.extract(origins, rays, near=0.01, far=3, samples=12,
                                      fine_samples=8, ray_chunk=2, point_chunk=100)
                self.assertEqual(a.shape, (2, mc[f'd_filter{"_fine" if kind == "fine" else ""}'] // 2 + 1))
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                self.assertGreater(stats["mass_mean"], 0)
                self.assertTrue(all(not p.requires_grad for p in field.selected.parameters()))
                self.assertFalse(field.selected.output._forward_pre_hooks)
            # Respect the loaded synthesizer's unusual first-bin rule exactly.
            raw = torch.tensor([[[0.0, 0.0, 1.0]] * 3])
            z = torch.tensor([[0.0, 1.0, 2.0]])
            weights = field._weights(raw, z, 2.4)
            torch.testing.assert_close(weights[0, 0], torch.tensor(0.0))
            torch.testing.assert_close(weights[0, 1], (1 - torch.exp(torch.tensor(-1.0))) * torch.exp(torch.tensor(-1.0)))


if __name__ == "__main__":
    unittest.main()
