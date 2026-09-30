"""Use the external training implementation without editing its density rule."""
import importlib.util
from pathlib import Path
import sys

import torch
import yaml

from .common import finite_tensor, load_tensor_file, sha256_file


def _module(root, name):
    path = root / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing NeWRF source file: {path}")
    module_name = f"_rrf_external_{name}_{sha256_file(path)[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FrozenNeWRF:
    def __init__(self, root, config, checkpoint, model="coarse", device="cpu"):
        root = Path(root).expanduser().resolve()
        config, checkpoint = Path(config).expanduser(), Path(checkpoint).expanduser()
        cfg = yaml.safe_load(config.read_text())
        models = _module(root, "models")
        encoders = _module(root, "encoders")
        self.samplers = _module(root, "samplers")
        self.synthesizer = _module(root, "synthesizers").Synthesizer()
        self.synthesizer.return_bdc = False
        self.device = torch.device(device)
        self.model_name = model
        self.sampling_config = cfg.get("sampling", {})
        encoder_cfg, model_cfg = cfg["encoder"], cfg["models"]
        if not encoder_cfg["use_viewdirs"]:
            raise ValueError("This prototype requires NeWRF's direction-conditioned branch")
        if encoder_cfg.get("use_fc_encoder", False):
            raise ValueError("Frequency-conditioned NeWRF encoders are not supported by this adapter")
        self.encode = encoders.PositionalEncoder(encoder_cfg["d_input"], encoder_cfg["n_freqs"],
                                                 log_space=encoder_cfg["log_space"])
        self.encode_viewdirs = encoders.PositionalEncoder(encoder_cfg["d_input"], encoder_cfg["n_freqs_views"],
                                                          log_space=encoder_cfg["log_space"])
        state = load_tensor_file(checkpoint)

        def build(kind):
            suffix = "_fine" if kind == "fine" else ""
            net = models.NeWRF(d_input=self.encode.d_output,
                               n_layers=model_cfg[f"n_layers{suffix}"],
                               d_filter=model_cfg[f"d_filter{suffix}"],
                               skip=tuple(model_cfg["skip"]),
                               d_viewdirs=self.encode_viewdirs.d_output)
            net.load_state_dict(state[f"{kind}_model_state_dict"], strict=True)
            net.to(self.device).eval().requires_grad_(False)
            return net

        if model not in ("coarse", "fine"):
            raise ValueError("model must be coarse or fine")
        self.coarse = build("coarse")
        if model == "fine" and not model_cfg["use_fine_model"]:
            raise ValueError("The supplied configuration has no fine model")
        self.selected = build("fine") if model == "fine" else self.coarse
        self.feature_dim = self.selected.output.in_features
        self.provenance = {
            "checkpoint_sha256": sha256_file(checkpoint),
            "config_sha256": sha256_file(config), "model": model,
            "source_sha256": {n: sha256_file(root / f"{n}.py")
                              for n in ("models", "encoders", "samplers", "synthesizers")},
            "feature_layer": "output:input", "feature_dim": self.feature_dim,
        }

    def _query(self, model, points, directions, point_chunk, capture_features):
        flat = points.reshape(-1, 3)
        viewdirs = directions[:, None, :].expand_as(points).reshape(-1, 3)
        raw_parts, feature_parts, captured = [], [], []
        handle = None
        if capture_features:
            handle = model.output.register_forward_pre_hook(lambda _, args: captured.append(args[0]))
        try:
            for start in range(0, len(flat), point_chunk):
                captured.clear()
                raw = model(self.encode(flat[start:start + point_chunk]),
                            self.encode_viewdirs(viewdirs[start:start + point_chunk]))
                raw_parts.append(finite_tensor(raw, "NeWRF output"))
                if capture_features:
                    if len(captured) != 1:
                        raise RuntimeError("Expected exactly one call to NeWRF.output per query")
                    feature_parts.append(finite_tensor(captured[0], "hidden features"))
        finally:
            if handle is not None:
                handle.remove()
        raw = torch.cat(raw_parts).reshape(*points.shape[:-1], -1)
        features = (torch.cat(feature_parts).reshape(*points.shape[:-1], -1)
                    if capture_features else None)
        return raw, features

    def _weights(self, raw, z_vals, carrier_ghz):
        # Use precisely the loaded synthesizer, including its first/last-bin rules.
        result = self.synthesizer.synthesize(raw, z_vals, carrier_ghz, ray_batches=[len(raw)])
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise RuntimeError("Expected NeWRF synthesizer to return (cfr, depth, amplitude, weights)")
        weights = finite_tensor(result[3], "rendering weights")
        if weights.shape != z_vals.shape or (weights < 0).any():
            raise ValueError("Unexpected NeWRF rendering-weight shape or sign")
        return weights

    @torch.no_grad()
    def extract(self, origins, directions, *, near, far, samples, fine_samples,
                carrier_ghz=2.4, ray_chunk=32, point_chunk=8192, inverse_depth=False):
        if not (0 < near < far) or samples < 3 or fine_samples < 1:
            raise ValueError("Require 0 < near < far, samples >= 3, fine_samples >= 1")
        if ray_chunk < 1 or point_chunk < 1 or carrier_ghz <= 0:
            raise ValueError("Chunk sizes and carrier frequency must be positive")
        pooled_parts, masses = [], []
        positive, total, raw_min, raw_max = 0, 0, float("inf"), -float("inf")
        for start in range(0, len(origins), ray_chunk):
            o = torch.as_tensor(origins[start:start + ray_chunk], device=self.device)
            d = torch.as_tensor(directions[start:start + ray_chunk], device=self.device)
            points, z = self.samplers.sample_stratified(o, d, near, far, n_samples=samples,
                                                       perturb=False, inverse_depth=inverse_depth)
            raw, features = self._query(self.coarse, points, d, point_chunk, self.model_name == "coarse")
            weights = self._weights(raw, z, carrier_ghz)
            if self.model_name == "fine":
                points, z, _, _ = self.samplers.sample_hierarchical(o, d, z, weights,
                                                                  n_new_samples=fine_samples, perturb=False)
                raw, features = self._query(self.selected, points, d, point_chunk, True)
                weights = self._weights(raw, z, carrier_ghz)
            pooled_parts.append((weights[..., None] * features).sum(1).cpu())
            masses.append(weights.sum(1).cpu())
            sigma_raw = raw[..., -1]
            positive += int((sigma_raw > 0).sum())
            total += sigma_raw.numel()
            raw_min = min(raw_min, float(sigma_raw.min()))
            raw_max = max(raw_max, float(sigma_raw.max()))
        pooled, mass = torch.cat(pooled_parts), torch.cat(masses)
        feature_map = finite_tensor(torch.cat((pooled, mass[:, None]), -1), "pooled feature map")
        stats = {"raw_density_min": raw_min, "raw_density_max": raw_max,
                 "raw_density_positive_fraction": positive / total,
                 "mass_min": float(mass.min()), "mass_max": float(mass.max()),
                 "mass_mean": float(mass.mean()), "mass_nonzero_fraction": float((mass > 1e-8).float().mean()),
                 "pooled_feature_norm_mean": float(pooled.norm(dim=-1).mean())}
        return feature_map, stats
