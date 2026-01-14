import logging
from random import random
from pathlib import Path

import scipy
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import torchode
from huggingface_hub import hf_hub_download

from .models import FLowHigh, MelVoco
from .cfm_superresolution import ConditionalFlowMatcherWrapper
from .postprocessing import PostProcessing


REPO_ID = "ResembleAI/FlowHigh"


class FlowHighSR(ConditionalFlowMatcherWrapper):
    def __init__(
        self,
        flowhigh: FLowHigh,
        sigma = 0.,
        ode_atol = 1e-5,
        ode_rtol = 1e-5,
        use_torchode = False,
        cfm_method = 'basic_cfm',
        torchdiffeq_ode_method = 'midpoint',   # [euler, midpoint]
        torchode_method_klass = torchode.Tsit5,
        cond_drop_prob = 0.,
        #
        upsampling_method='scipy',
    ):
        super().__init__(
            flowhigh=flowhigh,
            sigma=sigma,
            ode_atol=ode_atol,
            ode_rtol=ode_rtol,
            use_torchode=use_torchode,
            cfm_method=cfm_method,
            torchdiffeq_ode_method=torchdiffeq_ode_method,
            torchode_method_klass=torchode_method_klass,
            cond_drop_prob=cond_drop_prob,
        )
        self.upsampling_method = upsampling_method
        self.postproc = PostProcessing(0)
        # self.device = device

    @torch.no_grad()
    def generate(
        self,
        audio: np.ndarray,
        sr: int,
        target_sampling_rate=48000,
        timestep=1,
    ):
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)

        if audio.max() > 1:
            audio = audio / 32768.0

        # Up sampling the input audio (in Numpy)
        if self.upsampling_method =='scipy':
            # audio, sr = librosa.load(wav_file, sr=None, mono=True)
            cond = scipy.signal.resample_poly(audio, target_sampling_rate, sr)
            cond /= np.max(np.abs(cond))
            if isinstance(cond, np.ndarray):
                cond = torch.tensor(cond).unsqueeze(0)
            cond = cond.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # [1, T]

        elif self.upsampling_method == 'librosa':
            # audio, sr = librosa.load(wav_file, sr=None, mono=True)
            cond = librosa.resample(audio, sr, target_sampling_rate, res_type='soxr_hq')
            cond /= np.max(np.abs(cond))
            if isinstance(cond, np.ndarray):
                cond = torch.tensor(cond).unsqueeze(0)
            cond = cond.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # [1, T]

        # Audio must be in torch.Tensor from now on
        if isinstance(cond, np.ndarray):
            cond = torch.from_numpy(cond)

        cond = cond.float().to(self.device)

        # reconstruct high resolution sample
        if self.cfm_method == 'basic_cfm':
            HR_audio = self.sample(cond = cond, time_steps = timestep, cfm_method = self.cfm_method)
        elif self.cfm_method == 'independent_cfm_adaptive':
            HR_audio = self.sample(cond = cond, time_steps = timestep, cfm_method = self.cfm_method, std_2 = 1.)
        elif self.cfm_method == 'independent_cfm_constant':
            HR_audio = self.sample(cond = cond, time_steps = timestep, cfm_method = self.cfm_method)
        elif self.cfm_method == 'independent_cfm_mix':
            HR_audio = self.sample(cond = cond, time_steps = timestep, cfm_method = self.cfm_method)

        HR_audio = HR_audio.squeeze(1) # [1, T]

        # post-proceesing w.r.s.t audio-level
        HR_audio_pp = self.postproc.post_processing(HR_audio, cond, cond.size(-1)) # [1, T]
        return HR_audio_pp

    @torch.inference_mode()  # Faster than no_grad()
    def generate_long(
        self,
        audio: np.ndarray,
        sr: int,
        target_sampling_rate=48000,
        timestep=1,
        chunk_duration_seconds=60.0,
        overlap_seconds=1.0,
    ):
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)

        if audio.max() > 1:
            audio = audio / 32768.0

        # NOTE: Ideally, upsampling should also be chunked for truly infinite length,
        # but that requires complex context/padding management to avoid clicking.
        # Keeping full upsampling here, but be aware this is the RAM bottleneck.
        if self.upsampling_method == 'scipy':
            cond_full = scipy.signal.resample_poly(audio, target_sampling_rate, sr)
        elif self.upsampling_method == 'librosa':
            cond_full = librosa.resample(audio, sr, target_sampling_rate, res_type='soxr_hq')
        else:
            raise ValueError(f"Unknown upsampling method: {self.upsampling_method}")

        cond_full /= np.max(np.abs(cond_full))
        
        # Pre-calculate dimensions
        total_samples = len(cond_full)
        chunk_samples = int(chunk_duration_seconds * target_sampling_rate)
        overlap_samples = int(overlap_seconds * target_sampling_rate)
        stride_samples = chunk_samples - overlap_samples

        if chunk_samples <= overlap_samples:
            raise ValueError("chunk_duration_seconds must be greater than overlap_seconds")

        # Pre-calculate cross-fade window (create on Device to avoid transfers later if doing GPU mixing)
        # However, since 'result' is on CPU, we keep weights on CPU to avoid mismatch errors during assignment
        fade_weights = torch.linspace(0, 1, overlap_samples + 2, device='cpu')[1:-1]
        
        # Allocate Final Result Buffer (CPU)
        result = torch.zeros(1, total_samples, device='cpu')
        
        start = 0
        total_chunks = (total_samples // stride_samples) + 1
        
        # Process loop
        current_chunk_idx = 0
        
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            
            # 1. Prepare Input Chunk
            chunk_np = cond_full[start:end]
            if len(chunk_np) == 0: 
                break
                
            chunk_tensor = torch.from_numpy(chunk_np).float().unsqueeze(0).to(self.device)
            
            # 2. Inference
            # Helper to consolidate method calls
            if self.cfm_method == 'independent_cfm_adaptive':
                kwargs = {'std_2': 1.}
            else:
                kwargs = {}
                
            HR_chunk = self.sample(
                cond=chunk_tensor, 
                time_steps=timestep, 
                cfm_method=self.cfm_method, 
                **kwargs
            )
            
            HR_chunk = HR_chunk.squeeze(1) # [1, T_chunk]
            
            # 3. Post Processing
            HR_chunk_pp = self.postproc.post_processing(
                HR_chunk, chunk_tensor, chunk_tensor.size(-1)
            )
            
            # 4. Move to CPU for stitching
            # We process audio on CPU here to save VRAM, assuming the target array is huge
            chunk_data = HR_chunk_pp.cpu()
            chunk_length = chunk_data.size(-1)
            
            # 5. Apply Fading & Stitch immediately
            # -----------------------------------------------
            
            # Fade In (Start) - apply to this chunk
            # Only if not the very first chunk
            if current_chunk_idx > 0:
                fade_end_idx = min(overlap_samples, chunk_length)
                chunk_data[:, :fade_end_idx] *= fade_weights[:fade_end_idx]

            # Fade Out (End) - apply to this chunk
            # Only if not the very last part of the file
            if end < total_samples:
                fade_start_idx = max(0, chunk_length - overlap_samples)
                # Calculate how much overlap we actually have
                actual_overlap = chunk_length - fade_start_idx
                if actual_overlap > 0:
                    chunk_data[:, fade_start_idx:] *= (1 - fade_weights[-actual_overlap:])

            # Add to result buffer
            # We add (+=) because the overlap region contains:
            #   Previous Chunk (Faded Out) + Current Chunk (Faded In)
            result[:, start:end] += chunk_data

            # -----------------------------------------------
            
            # 6. Cleanup to prevent OOM
            del chunk_tensor, HR_chunk, HR_chunk_pp, chunk_data
            # Optional: empty cache if VRAM is extremely tight, but slows down loop
            # torch.cuda.empty_cache() 
            
            start += stride_samples
            current_chunk_idx += 1

        return result.to(self.device)

    def set_cfm_method(self, cfm_method):
        self.cfm_method = cfm_method
        # torchdiffeq_ode_method
        # sigma

    @classmethod
    def from_local(cls, ckpt_dir: Path, device) -> 'FlowHighSR':
        ckpt_dir = Path(ckpt_dir)
        voc = MelVoco(
            vocoder_config=ckpt_dir / "bigvgan_48khz_256band.json",
            vocoder_path=ckpt_dir / "bigvgan_48khz_256band.pt",
        )

        SR_generator = FLowHigh(
            dim_in = voc.n_mels,
            audio_enc_dec = voc,
            depth =2, # args.n_layers,
        )
        SR_generator = SR_generator.cuda().eval()

        cfm_wrapper=cls(
            flowhigh=SR_generator,
            # cfm_method = args.cfm_method,
            # torchdiffeq_ode_method=args.ode_method,
            # sigma = args.sigma,
        )
        # checkpoint load
        model_checkpoint = torch.load(
            ckpt_dir / "FLowHigh_basic_400k.pt",
            map_location=device
        )
        cfm_wrapper.load_state_dict(model_checkpoint['model']) # dict_keys(['model', 'optim', 'scheduler'])
        cfm_wrapper = cfm_wrapper.cuda().eval()
        return cfm_wrapper

    @classmethod
    def from_pretrained(cls, device) -> 'FlowHighSR':
        for fpath in [
            "FLowHigh_basic_400k.json",
            "bigvgan_48khz_256band.json",
            "FLowHigh_basic_400k.pt",
            "bigvgan_48khz_256band.pt",
        ]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)
