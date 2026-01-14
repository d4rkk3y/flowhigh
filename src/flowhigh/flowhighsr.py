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

    @torch.no_grad()
    def generate_long(
        self,
        audio: np.ndarray,
        sr: int,
        target_sampling_rate=48000,
        timestep=1,
        chunk_duration_seconds=60.0,
        overlap_seconds=1.0,
    ):
        """
        Generate super-resolution audio for very long duration files without OOM error.
        Processes audio in overlapping chunks and blends them using linear cross-fade.
        
        Args:
            audio: Input audio as numpy array shape (samples,) or (1, samples)
            sr: Input sampling rate
            target_sampling_rate: Target sampling rate (default 48000)
            timestep: Number of ODE steps (default 1)
            chunk_duration_seconds: Duration of each chunk in seconds (default 10.0)
            overlap_seconds: Overlap between chunks in seconds (default 1.0)
            
        Returns:
            Super-resolved audio as torch.Tensor shape [1, T]
        """
        if len(audio.shape) == 2:
            audio = audio.squeeze(0)

        if audio.max() > 1:
            audio = audio / 32768.0

        # Upsample entire audio to target sampling rate (keep in CPU memory)
        if self.upsampling_method == 'scipy':
            cond_full = scipy.signal.resample_poly(audio, target_sampling_rate, sr)
        elif self.upsampling_method == 'librosa':
            cond_full = librosa.resample(audio, sr, target_sampling_rate, res_type='soxr_hq')
        else:
            raise ValueError(f"Unknown upsampling method: {self.upsampling_method}")
        
        # Normalize
        cond_full /= np.max(np.abs(cond_full))
        
        # Split into overlapping chunks
        chunk_samples = int(chunk_duration_seconds * target_sampling_rate)
        overlap_samples = int(overlap_seconds * target_sampling_rate)
        stride_samples = chunk_samples - overlap_samples
        
        if chunk_samples <= overlap_samples:
            raise ValueError("chunk_duration_seconds must be greater than overlap_seconds")
        
        total_samples = len(cond_full)
        chunks = []
        start = 0
        
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = cond_full[start:end]
            # Ensure chunk has at least some samples
            if len(chunk) == 0:
                break
            chunks.append((start, end, chunk))
            start += stride_samples
            # If we're at the last chunk and it's smaller than stride, break
            if end == total_samples:
                break
        
        # Process each chunk
        processed_chunks = []
        for i, (start, end, chunk) in enumerate(chunks):
            # Convert chunk to tensor
            chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)
            
            # Generate super-resolution for this chunk
            if self.cfm_method == 'basic_cfm':
                HR_chunk = self.sample(cond=chunk_tensor, time_steps=timestep, cfm_method=self.cfm_method)
            elif self.cfm_method == 'independent_cfm_adaptive':
                HR_chunk = self.sample(cond=chunk_tensor, time_steps=timestep, cfm_method=self.cfm_method, std_2=1.)
            elif self.cfm_method == 'independent_cfm_constant':
                HR_chunk = self.sample(cond=chunk_tensor, time_steps=timestep, cfm_method=self.cfm_method)
            elif self.cfm_method == 'independent_cfm_mix':
                HR_chunk = self.sample(cond=chunk_tensor, time_steps=timestep, cfm_method=self.cfm_method)
            
            HR_chunk = HR_chunk.squeeze(1)  # [1, T]
            
            # Post-processing
            HR_chunk_pp = self.postproc.post_processing(HR_chunk, chunk_tensor, chunk_tensor.size(-1))
            processed_chunks.append((start, end, HR_chunk_pp.cpu()))
        
        # Stitch chunks with linear cross-fade in overlap regions
        result = torch.zeros(1, total_samples, device='cpu')
        fade_weights = torch.linspace(0, 1, overlap_samples + 2)[1:-1]  # exclude 0 and 1
        
        for i, (start, end, chunk) in enumerate(processed_chunks):
            chunk_length = chunk.size(-1)
            if i == 0:
                # First chunk: no fade at start
                fade_start = 0
                fade_end = min(overlap_samples, chunk_length)
                if fade_end > 0:
                    # Apply fade-out for overlapping part
                    fade_out = 1 - fade_weights[:fade_end]
                    chunk[:, :fade_end] *= fade_out
                result[:, start:end] += chunk
            elif i == len(processed_chunks) - 1:
                # Last chunk: no fade at end
                fade_start = max(0, chunk_length - overlap_samples)
                fade_end = chunk_length
                if fade_start < fade_end:
                    # Apply fade-in for overlapping part
                    fade_in = fade_weights[-(fade_end - fade_start):]
                    chunk[:, fade_start:fade_end] *= fade_in
                result[:, start:end] += chunk
            else:
                # Middle chunk: fade both sides
                # Fade-in at start (overlap with previous chunk)
                fade_start = 0
                fade_end = min(overlap_samples, chunk_length)
                if fade_end > 0:
                    fade_in = fade_weights[:fade_end]
                    chunk[:, fade_start:fade_end] *= fade_in
                
                # Fade-out at end (overlap with next chunk)
                fade_start = max(0, chunk_length - overlap_samples)
                fade_end = chunk_length
                if fade_start < fade_end:
                    fade_out = 1 - fade_weights[-(fade_end - fade_start):]
                    chunk[:, fade_start:fade_end] *= fade_out
                
                result[:, start:end] += chunk
        
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
