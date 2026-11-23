# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.helios.helios_transformer import HeliosTransformer3DModel
from vllm_omni.diffusion.models.helios.pipeline_helios import HeliosPipeline
from vllm_omni.diffusion.models.helios.scheduling_helios import HeliosScheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_helios_cache_dit_counts_steps_when_cfg_changes(tmp_path, monkeypatch):
    device = torch.device("cpu")
    platform = type(current_omni_platform)
    monkeypatch.setattr(
        platform,
        "get_diffusion_attn_backend_cls",
        Mock(return_value="vllm_omni.diffusion.attention.backends.sdpa.SDPABackend"),
    )
    monkeypatch.setattr(platform, "is_out_of_tree", Mock(return_value=True))
    monkeypatch.setattr(platform, "is_available", Mock(return_value=False), raising=False)

    rendezvous = (tmp_path / "gloo").as_uri()
    torch.distributed.init_process_group("gloo", init_method=rendezvous, world_size=1, rank=0)
    try:
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method=rendezvous, backend="gloo"
        )
        initialize_model_parallel()
        config = OmniDiffusionConfig.from_kwargs(
            model=str(tmp_path),
            dtype=torch.float32,
            diffusion_attention_config={"default": "TORCH_SDPA"},
            linear_backend="TORCH",
        )
        vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
        with (
            set_forward_context(omni_diffusion_config=config),
            set_current_diffusion_config(config),
            set_current_vllm_config(vllm_config),
            torch.random.fork_rng(devices=[]),
            torch.no_grad(),
        ):
            torch.manual_seed(92)
            pipeline = HeliosPipeline.__new__(HeliosPipeline)
            torch.nn.Module.__init__(pipeline)
            pipeline.od_config = config
            pipeline.device = device
            pipeline.is_distilled = False
            pipeline.vae_scale_factor_temporal = 4
            pipeline.vae_scale_factor_spatial = 8
            pipeline.vae = Mock(device=device, dtype=torch.float32)
            pipeline.vae.config = Mock(z_dim=4, latents_mean=[0.0] * 4, latents_std=[1.0] * 4)
            pipeline.vae.decode.side_effect = lambda latents, return_dict: (latents,)
            pipeline.scheduler = HeliosScheduler()
            pipeline.set_progress_bar_config(disable=True)
            transformer = HeliosTransformer3DModel(
                patch_size=(1, 2, 2),
                num_attention_heads=4,
                attention_head_dim=32,
                in_channels=4,
                out_channels=4,
                text_dim=64,
                freq_dim=32,
                ffn_dim=128,
                num_layers=2,
                rope_dim=(12, 10, 10),
            ).eval()
            for parameter in transformer.parameters():
                parameter.normal_(mean=0.0, std=0.05)
            for module in transformer.modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
            pipeline.transformer = transformer

            backend = CacheDiTBackend({"max_warmup_steps": 4})
            backend.enable(pipeline)
            prompt_embeds = torch.randn(1, 8, 64)
            negative_embeds = torch.zeros_like(prompt_embeds)
            try:
                for guidance_scale, negative_prompt_embeds, expected_forwards in (
                    (1.0, None, 4),
                    (4.0, negative_embeds, 8),
                    (4.0, None, 4),
                    (4.0, negative_embeds, 8),
                ):
                    backend.refresh(pipeline, 4)
                    request = OmniDiffusionRequest(
                        prompt="a video",
                        request_id="helios-cache-cfg",
                        sampling_params=OmniDiffusionSamplingParams(
                            height=32,
                            width=48,
                            num_frames=5,
                            num_inference_steps=4,
                            guidance_scale=guidance_scale,
                            seed=0,
                        ),
                    )
                    result = pipeline(
                        DiffusionRequestBatch(requests=[request]),
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        num_latent_frames_per_chunk=2,
                        history_sizes=[4, 2, 1],
                        output_type="latent",
                    )
                    assert torch.isfinite(result.output).all()
                    context = transformer._context_manager.get_context()
                    assert context.transformer_executed_steps == expected_forwards
                    assert context.executed_steps == 4
            finally:
                backend.disable(pipeline)
    finally:
        destroy_distributed_env()
