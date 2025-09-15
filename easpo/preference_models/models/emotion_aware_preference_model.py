import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import AutoProcessor

from  .time_conditioned_clep import HFTimeConditionedCLEPModel
from easpo.utils import huggingface_cache_dir

from accelerate.logging import get_logger

logger = get_logger(__name__)

class EAS_PreferenceModel(nn.Module):
    def __init__(
        self, 
        model_pretrained_model_name_or_path,
        processor_pretrained_model_name_or_path,
        ckpt_path=None,
    ):
        super().__init__()
        self.model = HFTimeConditionedCLEPModel.from_pretrained(
            model_pretrained_model_name_or_path,
            cache_dir=huggingface_cache_dir,
        )
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location='cpu')
            missing_keys, unexpected_keys = self.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
            if len(unexpected_keys) != 0:
                for key in unexpected_keys:
                    ckpt_value = state_dict[key]
                    current_value = self
                    for sub_key in key.split('.'):
                        current_value = getattr(current_value, sub_key)
                    assert torch.all(ckpt_value == current_value), f"unexpected key {key} have different values"
            assert len(missing_keys) == 0, f"missing keys: {missing_keys}"
            try:
                logger.info(f"Loaded step-aware preference model ckpt from {ckpt_path}")
            except:
                print(f"Loaded emotional aware step-aware preference model ckpt from {ckpt_path}")
        processor = AutoProcessor.from_pretrained(
            processor_pretrained_model_name_or_path,
            cache_dir=huggingface_cache_dir,
        )

    def get_text_features(self, *args, **kwargs):
        return self.model.get_text_features(*args, **kwargs)

    def get_audio_features(self, *args, **kwargs):
        return self.model.get_audio_features(*args, **kwargs)

    def forward(self, text_inputs=None, audio_inputs=None, time_cond=None):
        outputs = ()
        if text_inputs is not None:
            outputs += self.model.get_text_features(text_inputs),
        if audio_inputs is not None:
            outputs += self.model.get_audio_features(audio_inputs, time_cond),
        return outputs
    
    @property
    def logit_scale(self):
        return self.model.logit_scale
    
    def get_preference_score(self, audios, input_ids, timesteps):
        
        shortest_size = min(audios.size(-2), audios.size(-1))
        scale = self.img_size / shortest_size
        new_size = (
            int(audios.size(-2) * scale + 0.5), 
            int(audios.size(-1) * scale + 0.5), 
        )
        audios = F.interpolate(
            audios, 
            size=new_size,
            mode='bicubic', 
            align_corners=False,
        )
        
        if new_size[0] != new_size[1]:
            audios = self.center_crop(audios)
        
        audios = (audios / 2 + 0.5).clamp(0, 1).float()
        audios = self.normalization(audios)
        
        audios_embeds = self.model.get_audio_features(
            pixel_values=audios,
            time_cond=timesteps,
        )
        
        audios_embeds = audios_embeds / torch.norm(audios_embeds, dim=-1, keepdim=True)        
        
        text_embeds = self.model.get_text_features(
            input_ids=input_ids
        )
        
        text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)
        
        
        scores = self.model.logit_scale.exp() * (audios_embeds * text_embeds).sum(dim=-1)
        
        return scores
