import torch

from .builder import PREFERENCE_MODEL_FUNC_BUILDERS
from .models.emotion_aware_preference_model import EAS_PreferenceModel

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name='emotion_aware_preference_model_func')
def emotion_aware_preference_model_func_builder(cfg):
    emotion_aware_preference_model = EAS_PreferenceModel(
        model_pretrained_model_name_or_path=cfg.model_pretrained_model_name_or_path,
        processor_pretrained_model_name_or_path=cfg.processor_pretrained_model_name_or_path,
        ckpt_path=cfg.ckpt_path,
    ).eval().to(cfg.device)
    emotion_aware_preference_model.requires_grad_(False)
    
    @torch.no_grad()
    def preference_fn(img, extra_info):
        # b
        scores = emotion_aware_preference_model.get_preference_score(
            img, 
            extra_info['input_ids'],
            extra_info['timeemotions'],
        )
        return scores
    
    return preference_fn
