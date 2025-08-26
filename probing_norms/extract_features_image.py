import pdb
import requests

from functools import partial
from pathlib import Path
from tqdm import tqdm

import click
import h5py
import torch
import numpy as np
import timm
import torch.nn as nn

from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    LlavaForConditionalGeneration,
    PaliGemmaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    ViTMAEModel,
)
from transformers.models.siglip import SiglipVisionModel
from qwen_vl_utils import process_vision_info

from torch.utils.data import DataLoader

import torchvision.transforms as T

from open_clip import create_model_from_pretrained

from probing_norms.data import DATASETS
from probing_norms.utils import implies


def count_params(model):
    return sum(p.numel() for p in model.parameters())


class ImageBackboneDINO(nn.Module):
    def __init__(self, type_):
        assert type_ == "resnet50"
        super(ImageBackboneDINO, self).__init__()

        self.model = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_" + type_,
            pretrained=True,
        )
        self.model = nn.Sequential(*list(self.model.children())[:-2])

        IMAGE_SIZE = 224
        transform_image_norm = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.transform = T.Compose(
            [
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                transform_image_norm,
            ]
        )

    def forward(self, x):
        features = self.model(x)
        features = features.mean([2, 3])
        return features


class CLIP(nn.Module):
    def __init__(self, model_id, tokens, layer):
        super(CLIP, self).__init__()

        assert tokens in ["CLS", "patches"]
        assert implies(layer == "post-projection", tokens == "CLS")

        self.model = CLIPModel.from_pretrained(model_id).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

        if layer == "post-projection":
            self.feature_dim = self.model.config.projection_dim
            self.get_image_features = self.get_image_features_post_projection
        else:
            self.feature_dim = self.model.config.vision_config.hidden_size
            self.get_image_features = partial(
                self.get_image_features_pre_projection,
                tokens=tokens,
                layer=layer,
            )

    def transform(self, x):
        output = self.processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        output = output.squeeze(0)
        return output

    def get_image_features_pre_projection(self, x, tokens, layer):
        outputs = self.model.vision_model(x, output_hidden_states=True)
        outputs = outputs.hidden_states[layer]
        outputs = outputs[:, :1] if tokens == "CLS" else outputs[:, 1:]
        outputs = outputs.mean(1)
        return outputs

    def get_image_features_post_projection(self, x):
        return self.model.get_image_features(x)

    def forward(self, x):
        return self.get_image_features(x)


class OpenCLIP(nn.Module):
    def __init__(self, name):
        super(OpenCLIP, self).__init__()
        assert name == "hf-hub:apple/DFN2B-CLIP-ViT-L-14"
        model, processor = create_model_from_pretrained(name)
        self.model = model.eval()
        self.transform = processor

        name_short = name.split(":")[1]
        URL = "https://huggingface.co/{}/resolve/main/config.json".format(name_short)
        config = requests.get(URL)
        config = config.json()
        self.feature_dim = config["projection_dim"]

    def forward(self, x):
        return self.model.encode_image(x)


class SigLIP(nn.Module):
    def __init__(self, use_random_weights=False):
        super(SigLIP, self).__init__()
        model_id = "google/siglip-so400m-patch14-224"
        model = AutoModel.from_pretrained(model_id).eval()
        self.feature_dim = model.config.vision_config.hidden_size

        if use_random_weights:
            model = SiglipVisionModel(model.config.vision_config)
            model = model.eval()

        self.model = model.vision_model
        self.model.head = nn.Identity()
        self.processor = AutoProcessor.from_pretrained(model_id)

    def transform(self, x):
        output = self.processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        output = output.squeeze(0)
        return output

    def forward(self, x):
        features = self.model(x)
        features = features.last_hidden_state.mean(1)
        return features


class PaliGemma(nn.Module):
    def __init__(self):
        super(PaliGemma, self).__init__()
        model_id = "google/paligemma-3b-mix-224"
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.feature_dim = self.model.config.vision_config.hidden_size

    def transform(self, x):
        output = self.processor.image_processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        output = output.squeeze(0)
        return output

    def forward(self, x):
        features = self.model.vision_tower(x)
        features = features.last_hidden_state.mean(1)
        return features


class Qwen25VL(nn.Module):
    def __init__(self):
        super(Qwen25VL, self).__init__()
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.feature_dim = self.model.config.vision_config.out_hidden_size
        pdb.set_trace()

    def transform(self, x):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": x,
                        "resized_height": 224,
                        "resized_width": 224,
                    }
                ],
            }
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        assert video_inputs is None
        inputs = self.processor.image_processor(
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )
        return inputs

    def forward(self, inp):
        pixel_values = inp["pixel_values"]
        image_grid_thw = inp["image_grid_thw"].squeeze(1)
        features = self.model.visual(pixel_values, image_grid_thw)
        features = features.unsqueeze(0)
        features = features.mean(0)
        features = features.float()
        return features


class LLaVa(nn.Module):
    def __init__(self, to_apply_projector=True):
        super(LLaVa, self).__init__()
        model_id = "llava-hf/llava-1.5-7b-hf"
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            # torch_dtype="auto",
            # device_map="auto",
        )
        model.eval()
        self.vision_tower = model.vision_tower
        self.multi_modal_projector = model.multi_modal_projector
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.config = model.config
        self.to_apply_projector = to_apply_projector

        if self.to_apply_projector:
            self.feature_dim = model.config.text_config.hidden_size
        else:
            self.feature_dim = model.config.vision_config.hidden_size

    def transform(self, x):
        inputs = self.processor.image_processor(x, return_tensors="pt")
        inputs = inputs["pixel_values"]
        inputs = inputs.squeeze(0)
        return inputs

    def get_image_features(
        self,
        pixel_values,
        vision_feature_layer,
        vision_feature_select_strategy,
    ):
        """
        Code take from here:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py#L281

        """
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)

        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [
                image_outputs.hidden_states[layer_idx]
                for layer_idx in vision_feature_layer
            ]
            # For default; crop CLS from each hidden state in the hidden state pool
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        if self.to_apply_projector:
            image_features = self.multi_modal_projector(selected_image_feature)
        else:
            image_features = selected_image_feature

        return image_features

    def forward(self, x):
        image_features = self.get_image_features(
            x,
            self.config.vision_feature_layer,
            self.config.vision_feature_select_strategy,
        )
        image_features = image_features.mean(1)
        image_features = image_features.float()
        return image_features


class VITMAE(nn.Module):
    def __init__(self):
        super(VITMAE, self).__init__()
        model_id = "facebook/vit-mae-large"
        self.model = ViTMAEModel.from_pretrained(model_id).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.feature_dim = self.model.config.hidden_size

    def transform(self, x):
        output = self.processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        output = output.squeeze(0)
        return output

    def forward(self, x):
        features = self.model(x)
        features = features.last_hidden_state.mean(1)
        return features


class TimmModel(nn.Module):
    def __init__(self, model_id):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0)
        self.model = self.model.eval()
        self.data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(
            **self.data_config,
            is_training=False,
        )
        self.feature_dim = self.model.num_features

    def forward(self, x):
        return self.model(x)


class DINOV2(nn.Module):
    def __init__(self):
        super(DINOV2, self).__init__()
        model_id = "facebook/dinov2-large"
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.feature_dim = self.model.config.hidden_size

    def transform(self, x):
        output = self.processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        output = output.squeeze(0)
        return output

    def forward(self, x):
        features = self.model(x)
        features = features.last_hidden_state.mean(1)
        return features
        

class DINOV3(nn.Module):
    def __init__(self, variant="vitb16"):
        super(DINOV3, self).__init__()
        model_id = f"facebook/dinov3-{variant}-pretrain-lvd1689m"
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.feature_dim = self.model.config.hidden_size

    def transform(self, x):
        output = self.processor(images=x, return_tensors="pt")
        output = output["pixel_values"]
        output = output.squeeze(0)
        return output

    def forward(self, x):
        features = self.model(x)
        features = features.pooler_output
        return features


FEATURE_EXTRACTORS = {
    # fmt: off
    # Self supervised models
    "dino-resnet50": partial(ImageBackboneDINO, type_="resnet50"),
    "dino-v2": DINOV2,
    "dino-v3-vitl16": partial(DINOV3, variant="vitl16"),
    "vit-mae-large": VITMAE,
    # Image-text models
    "clip": partial(CLIP, model_id="openai/clip-vit-large-patch14", tokens="CLS", layer="post-projection"),
    "clip-dfn2b": partial(OpenCLIP, name="hf-hub:apple/DFN2B-CLIP-ViT-L-14"),
    "siglip-224": SigLIP,
    "pali-gemma-224": PaliGemma,
    "qwen2.5-vl-3b-instruct": Qwen25VL,
    "llava-1.5-7b": LLaVa,
    "llava-1.5-7b-pre-projector": partial(LLaVa, to_apply_projector=False),
    # Supervised models
    "swin-v2-ssl": partial(TimmModel, model_id="swinv2_large_window12_192.ms_in22k"),
    "swin-v2": partial(TimmModel, model_id="swinv2_large_window12to24_192to384.ms_in22k_ft_in1k"),
    "max-vit-large": partial(TimmModel, model_id="maxvit_large_tf_384.in1k"),
    "max-vit-large-in21k": partial(TimmModel, model_id="maxvit_large_tf_224.in21k"),
    # Random models
    "random-siglip": partial(SigLIP, use_random_weights=True),
    # CLIP study to answer the following question: Why do LLaVa embeddings are worse than the CLIP ones?
    "clip-layer--2": partial(CLIP, model_id="openai/clip-vit-large-patch14", tokens="CLS", layer=-2),
    "clip-layer--2-patches": partial(CLIP, model_id="openai/clip-vit-large-patch14", tokens="patches", layer=-2),
    "clip-336px": partial(CLIP, model_id="openai/clip-vit-large-patch14-336", tokens="CLS", layer="post-projection"),
    "clip-llava": partial(CLIP, model_id="openai/clip-vit-large-patch14-336", tokens="patches", layer=-2),
    # fmt: on
}


@click.command()
@click.option("-d", "--dataset", "dataset_name", type=str, required=True)
@click.option("-f", "--feature-type", "feature_type", type=str, required=True)
def main(dataset_name, feature_type):
    DEVICE = "cuda"

    feature_extractor = FEATURE_EXTRACTORS[feature_type]()
    feature_extractor.eval()
    feature_extractor.to(DEVICE)

    # print(count_params(feature_extractor))

    BATCH_SIZES = {
        "qwen2.5-vl-3b-instruct": 1,
        "llava-1.5-7b": 64,
    }
    batch_size = BATCH_SIZES.get(feature_type, 16)

    dataset = DATASETS[dataset_name](transform=feature_extractor.transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=3)

    def extract1(image):
        with torch.no_grad():
            image = image.to(DEVICE)
            feature = feature_extractor(image)
            feature = feature.cpu().numpy()
            return feature

    num_samples = len(dataset)
    feature_dim = feature_extractor.feature_dim

    X = np.zeros((num_samples, feature_dim))
    y = np.zeros(num_samples)

    i = 0
    for batch in tqdm(dataloader):
        features = extract1(batch["image"])
        for j, feature in enumerate(features):
            X[i] = feature
            y[i] = batch["label"][j].item()
            i += 1

    path_np = f"output/features-image/{dataset_name}-{feature_type}.npz"
    np.savez(path_np, X=X, y=y)


if __name__ == "__main__":
    main()
