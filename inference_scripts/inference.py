import argparse
import os
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

# HiFi-GAN vocoder
import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='path to text file (one caption per line)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Grad-TTS checkpoint path')
    parser.add_argument('--timesteps', type=int, default=10, help='reverse diffusion timesteps')
    parser.add_argument('--speaker_id', type=int, default=None, help='speaker id if multi-speaker')
    parser.add_argument('--output_dir', type=str, default='./out/', help='output directory for .wav files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.speaker_id is not None:
        assert params.n_spks > 1, "Multi-speaker model must be configured in params.py"
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None

    print('Initializing Grad-TTS...')
    generator = GradTTS(
        len(symbols)+1, params.n_spks, params.spk_emb_dim,
        params.n_enc_channels, params.filter_channels,
        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
        params.enc_kernel, params.enc_dropout, params.window_size,
        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale
    )
    generator.load_state_dict(torch.load(args.ckpt_path, map_location=lambda loc, storage: loc))
    generator.cuda().eval()

    print(f'Loaded Grad-TTS with {generator.nparams} parameters.')

    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    # Load text prompts
    with open(args.file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')

    print(f'Generating {len(texts)} samples...')
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'[{i}] Synthesizing: "{text}"')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

            start_time = dt.datetime.now()
            y_enc, y_dec, _ = generator.forward(
                x, x_lengths,
                n_timesteps=args.timesteps,
                temperature=1.5,
                stoc=False,
                spk=spk,
                length_scale=0.91
            )
            duration = (dt.datetime.now() - start_time).total_seconds()
            rtf = duration * 22050 / (y_dec.shape[-1] * 256)
            print(f'--> RTF: {rtf:.4f}')

            audio = vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy()
            audio = (audio * 32768).astype(np.int16)

            output_path = os.path.join(args.output_dir, f'sample_{i}.wav')
            write(output_path, 22050, audio)

    print(f'Done. Audio files are saved in {args.output_dir}')

if __name__ == '__main__':
    main()