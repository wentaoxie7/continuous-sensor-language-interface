import torch
import torch.nn as nn

from .config import ENCODER_OUTPUT_DIM
from .llm_utils import load_transformers_components
from .models import SensorEncoder
from .utils import resolve_torch_dtype


class SensorProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ContextEmbeddingClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        llm_model_name: str,
        prompt_template: str,
        sensor_placeholder: str,
        projector_hidden_dim: int,
        device: torch.device,
        llm_dtype: str = "auto",
    ) -> None:
        super().__init__()

        if prompt_template.count(sensor_placeholder) != 1:
            raise ValueError("Prompt template must contain the sensor placeholder exactly once.")

        AutoModelForCausalLM, AutoTokenizer = load_transformers_components()

        self.sensor_encoder = SensorEncoder()
        self.device_for_llm = device
        self.prompt_template = prompt_template
        self.sensor_placeholder = sensor_placeholder

        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        model_dtype = resolve_torch_dtype(device, llm_dtype)
        model_kwargs = {"torch_dtype": model_dtype} if model_dtype is not None else {}
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name, **model_kwargs)
        self.llm.to(device)
        self.llm.eval()
        self.llm_dtype = next(self.llm.parameters()).dtype

        for parameter in self.llm.parameters():
            parameter.requires_grad = False

        self.llm_hidden_size = self.llm.config.hidden_size
        self.sensor_projector = SensorProjector(
            input_dim=ENCODER_OUTPUT_DIM,
            hidden_dim=projector_hidden_dim,
            output_dim=self.llm_hidden_size,
        )
        self.classifier = nn.Linear(self.llm_hidden_size, num_classes)

        prompt_prefix, prompt_suffix = prompt_template.split(sensor_placeholder)
        prefix_ids = self._encode_prompt_part(prompt_prefix, add_bos_token=True)
        suffix_ids = self._encode_prompt_part(prompt_suffix, add_bos_token=False)

        self.register_buffer("prompt_prefix_ids", prefix_ids, persistent=False)
        self.register_buffer("prompt_suffix_ids", suffix_ids, persistent=False)

    def _encode_prompt_part(self, text: str, add_bos_token: bool) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_bos_token and self.tokenizer.bos_token_id is not None:
            token_ids = [self.tokenizer.bos_token_id, *token_ids]
        return torch.tensor(token_ids, dtype=torch.long)

    def _expand_prompt_ids(self, prompt_ids: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        return prompt_ids.to(device).unsqueeze(0).expand(batch_size, -1)

    def _embed_prompt_ids(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        embedding_layer = self.llm.get_input_embeddings()
        return embedding_layer(prompt_ids)

    def encode_sensor(self, x: torch.Tensor) -> torch.Tensor:
        sensor_features = self.sensor_encoder(x)
        return self.sensor_projector(sensor_features)

    def build_inputs_embeds(
        self,
        sensor_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = sensor_embeddings.size(0)
        device = sensor_embeddings.device

        prefix_ids = self._expand_prompt_ids(self.prompt_prefix_ids, batch_size, device)
        suffix_ids = self._expand_prompt_ids(self.prompt_suffix_ids, batch_size, device)

        prefix_embeds = self._embed_prompt_ids(prefix_ids)
        suffix_embeds = self._embed_prompt_ids(suffix_ids)

        sensor_token = sensor_embeddings.unsqueeze(1).to(prefix_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embeds, sensor_token, suffix_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        return inputs_embeds, attention_mask

    def forward(
        self,
        x: torch.Tensor | None = None,
        sensor_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sensor_embeddings is None:
            if x is None:
                raise ValueError("Either x or sensor_embeddings must be provided.")
            sensor_embeddings = self.encode_sensor(x)

        inputs_embeds, attention_mask = self.build_inputs_embeds(sensor_embeddings)
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        activity_hidden_state = outputs.hidden_states[-1][:, -1, :]
        activity_hidden_state = activity_hidden_state.to(self.classifier.weight.dtype)
        return self.classifier(activity_hidden_state)

    def train(self, mode: bool = True):
        super().train(mode)
        self.llm.eval()
        return self
