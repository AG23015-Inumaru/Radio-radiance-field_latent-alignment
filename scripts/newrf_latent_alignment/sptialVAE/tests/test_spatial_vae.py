"""Numeric and end-to-end checks for the spatial latent training workflow."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from scripts.newrf_latent_alignment.common import load_tensor_file
from scripts.newrf_latent_alignment.sptialVAE.encode_vae_map import main as encode_main
from scripts.newrf_latent_alignment.sptialVAE.evaluate_vae_map import main as evaluate_main
from scripts.newrf_latent_alignment.sptialVAE.map_utils import load_map_checkpoint, load_split_manifest
from scripts.newrf_latent_alignment.sptialVAE.train_vae_map import main as train_main
from scripts.newrf_latent_alignment.sptialVAE.vae_map import GrayMapVAE, MapVAEConfig, loss_terms


torch.set_num_threads(1)


class SpatialModelTests(unittest.TestCase):
    def test_spatial_round_trip_and_gradient_path(self):
        for channels in ((8, 16), (8, 16, 16), (8, 16, 16, 16)):
            cfg = MapVAEConfig(image_height=32, image_width=48, channels=channels, latent_channels=3)
            model = GrayMapVAE(cfg)
            image = torch.rand(2, 1, 32, 48)
            reconstructed, mu, logvar = model(image)
            self.assertEqual(tuple(mu.shape), (2, *cfg.latent_shape))
            self.assertEqual(mu.shape, logvar.shape)
            self.assertEqual(reconstructed.shape, image.shape)
            rec, _, kl = loss_terms(reconstructed, image, mu, logvar)
            (rec + 0.01 * kl).backward()
            for part in (model.mu_head, model.logvar_head, model.from_latent):
                self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                    for parameter in part.parameters()))
            self.assertFalse(any(isinstance(layer, nn_type) for layer in model.modules()
                                 for nn_type in (torch.nn.Linear, torch.nn.Flatten)))

    def test_kl_matches_known_gaussians_across_every_spatial_axis(self):
        image = torch.zeros(2, 1, 16, 24)
        mu = torch.ones(2, 3, 4, 6)
        logvar = torch.zeros_like(mu)
        rec, kl_sum, kl_mean = loss_terms(image, image, mu, logvar)
        self.assertEqual(rec.item(), 0.0)
        self.assertAlmostEqual(kl_sum.item(), 36.0)
        self.assertAlmostEqual(kl_mean.item(), 0.5)
        _, kl_sum, kl_mean = loss_terms(image, image, mu * 0, logvar)
        self.assertEqual(kl_sum.item(), 0.0)
        self.assertEqual(kl_mean.item(), 0.0)

    def test_mean_is_deterministic_and_sampling_is_stochastic_in_eval(self):
        model = GrayMapVAE(MapVAEConfig(image_height=32, image_width=48, channels=(8, 16))).eval()
        image = torch.rand(1, 1, 32, 48)
        mean_a, _, _ = model(image, sample=False)
        mean_b, _, _ = model(image, sample=False)
        self.assertTrue(torch.equal(mean_a, mean_b))
        sample_a, _, _ = model(image, sample=True)
        sample_b, _, _ = model(image, sample=True)
        self.assertFalse(torch.equal(sample_a, sample_b))

    def test_unsupported_vector_config_and_uneven_size_are_rejected(self):
        with self.assertRaises(ValueError):
            MapVAEConfig(image_height=31, image_width=48)
        with self.assertRaises(TypeError):
            MapVAEConfig(latent_dim=128)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps({"image_height": 32, "image_width": 48,
                              "channels": [8, 16], "latent_channels": 2, "residual_blocks": 1}))
        views = []
        for index in range(8):
            receiver = index // 2
            split = "train" if receiver < 2 else "val" if receiver == 2 else "test"
            pixels = np.zeros((32, 48), dtype=np.uint8)
            pixels[:, 4 + index:12 + index] = 220
            pixels[12:20] = 100
            image = self.root / f"image_{index}.png"
            Image.fromarray(pixels).save(image)
            pose = np.eye(4)
            pose[0, 3] = receiver
            views.append({"view_id": f"view_{index}", "receiver_id": f"rx_{receiver}",
                          "split": split, "image": image.name, "camera_to_world": pose.tolist(),
                          "intrinsics": {"width": 48, "height": 32, "fx": 24, "fy": 24, "cx": 24, "cy": 16}})
        self.manifest = self.root / "views.json"
        self.document = {"schema_version": 1, "camera_convention": "opencv",
                         "rf_from_world": np.eye(4).tolist(), "views": views}
        self.manifest.write_text(json.dumps(self.document))

    def tearDown(self):
        self.temporary.cleanup()

    def train(self, output, epochs, extra=()):
        args = ["--views", str(self.manifest), "--vae-config", str(self.config),
                "--output-dir", str(output), "--epochs", str(epochs), "--batch-size", "2",
                "--device", "cpu", "--seed", "7", "--warmup-epochs", "2", *extra]
        with redirect_stdout(io.StringIO()):
            train_main(args)

    def test_training_never_opens_test_images(self):
        for view in self.document["views"]:
            if view["split"] == "test":
                (self.root / view["image"]).unlink()
        output = self.root / "run"
        self.train(output, 1)
        self.assertTrue((output / "best.pt").is_file())
        with Image.open(output / "best_val_examples" / "00_mu.png") as preview:
            self.assertEqual(preview.size, (48, 32))

    def test_resume_matches_uninterrupted_training(self):
        uninterrupted, resumed = self.root / "continuous", self.root / "resumed"
        self.train(uninterrupted, 2)
        self.train(resumed, 1)
        self.train(resumed, 2, ["--resume"])
        first, second = load_tensor_file(uninterrupted / "last.pt"), load_tensor_file(resumed / "last.pt")
        self.assertEqual(first["history"], second["history"])
        for name, value in first["model_state"].items():
            self.assertTrue(torch.equal(value, second["model_state"][name]), name)
        with self.assertRaises(ValueError):
            self.train(resumed, 3, ["--resume", "--beta", "0.1"])

    def test_ae_disables_sampling_and_kl(self):
        output = self.root / "ae"
        self.train(output, 1, ["--mode", "ae"])
        row = load_tensor_file(output / "last.pt")["history"][0]
        self.assertEqual(row["train_kl_sum"], 0)
        self.assertEqual(row["train_weighted_kl"], 0)
        self.assertEqual(row["val_mu_mse"], row["val_sample_mse"])

    def test_evaluation_and_spatial_export(self):
        output = self.root / "run"
        self.train(output, 1)
        common = ["--views", str(self.manifest), "--vae-checkpoint", str(output / "best.pt"), "--device", "cpu"]
        with redirect_stdout(io.StringIO()):
            evaluate_main([*common, "--output-dir", str(self.root / "evaluation"), "--split", "test"])
            encode_main([*common, "--output-dir", str(self.root / "latents"), "--split", "all"])
        metrics = json.loads((self.root / "evaluation" / "metrics.json").read_text())
        self.assertEqual(metrics["count"], 2)
        self.assertIn("test_mu_mse", metrics)
        index = json.loads((self.root / "latents" / "index.json").read_text())
        self.assertEqual(len(index["views"]), 8)
        self.assertEqual(index["latent_shape"], [2, 8, 12])
        mu = load_tensor_file(self.root / "latents" / "000000.pt")["mu"]
        self.assertEqual(tuple(mu.shape), (2, 8, 12))
        model, _ = load_map_checkpoint(output / "best.pt")
        with Image.open(self.root / "image_0.png") as original:
            image = torch.from_numpy(np.array(original, dtype=np.float32)).unsqueeze(0).unsqueeze(0) / 255
        with torch.no_grad():
            expected, _ = model.encode(image)
        self.assertTrue(torch.equal(mu, expected[0]))

    def test_cross_split_receiver_leakage_is_rejected(self):
        self.document["views"][0]["split"] = "val"
        self.manifest.write_text(json.dumps(self.document))
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            load_split_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
