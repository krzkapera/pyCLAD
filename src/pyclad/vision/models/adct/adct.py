import numpy as np
import open_clip
import torch

from pyclad.vision.models.adct.backbone import build_clip_backbone
from pyclad.vision.models.adct.config import AdctConfig
from pyclad.vision.models.adct.prompt_learner import AdctPromptLearner
from pyclad.vision.models.adct.prompts import ADCT_ANOMALY_PROMPTS, ADCT_NORMAL_PROMPTS
from pyclad.vision.models.adct.scoring import anomaly_scores_and_maps
from pyclad.vision.models.adct.text_encoder import ClipTextEncoder, encode_prompt_groups
from pyclad.vision.models.adct.visual_encoder import AdaptedVisualEncoder
from pyclad.vision.prediction_results import VisionPredictionResults

ADCT_PROMPT_GROUPS = {"normal": ADCT_NORMAL_PROMPTS, "abnormal": ADCT_ANOMALY_PROMPTS}
IMAGE_VALUE_RANGE = 255.0


class Adct:
    def __init__(self, config: AdctConfig):
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        clip_model = build_clip_backbone(config.clip_model_name, config.weights_path)
        self.visual_encoder = AdaptedVisualEncoder(
            visual=clip_model.visual,
            feature_layers=config.feature_layers,
            width=config.width,
            bottleneck=config.bottleneck,
            adapter_weight=config.adapter_weight,
            noise_sigma=config.noise_sigma,
        ).to(self.device)
        self.prompt_learner = AdctPromptLearner(
            token_embedding=clip_model.token_embedding,
            context_dim=clip_model.ln_final.weight.shape[0],
            prompt_groups=ADCT_PROMPT_GROUPS,
            n_ctx=config.n_ctx,
            tokenize=open_clip.tokenize,
        ).to(self.device)
        self.text_encoder = ClipTextEncoder(clip_model).to(self.device)

    def text_features(self) -> torch.Tensor:
        return encode_prompt_groups(self.text_encoder, self.prompt_learner())

    def predict(self, data: np.ndarray) -> VisionPredictionResults:
        scores, maps = [], []
        with torch.no_grad():
            text_features = self.text_features()
            for start in range(0, len(data), self.config.batch_size):
                batch = self._to_tensor(data[start : start + self.config.batch_size])
                tokens, _ = self.visual_encoder(batch)
                batch_scores, batch_maps = anomaly_scores_and_maps(
                    patch_tokens=tokens,
                    text_features=text_features,
                    logit_scale=self.config.logit_scale,
                    output_size=self.config.image_size,
                )
                scores.append(batch_scores.cpu().numpy())
                maps.append(batch_maps.cpu().numpy())

        anomaly_scores = np.concatenate(scores).astype(np.float32)
        return VisionPredictionResults(
            y_pred=(anomaly_scores > 0.5).astype(np.int64),
            anomaly_scores=anomaly_scores,
            score_maps=np.concatenate(maps).astype(np.float32),
        )

    def _to_tensor(self, images: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        return tensor.permute(0, 3, 1, 2).float().div_(IMAGE_VALUE_RANGE)

    def name(self) -> str:
        return "ADCT"
