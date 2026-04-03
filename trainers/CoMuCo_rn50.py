import os.path as osp
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from typing import Dict, Iterable, Callable
import copy
import json
from tqdm import tqdm

_tokenizer = _Tokenizer()

def gpt_clip_classifier(classnames, gpt_prompts, clip_model, template):
    with torch.no_grad():
        clip_model = clip_model.cuda()
        clip_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=0).cuda()
    return clip_weights

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ],
    "ImageNetSketch": [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ],
    "ImageNetV2": [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ],
    "ImageNetA": [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ],
    "ImageNetR": [
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a {} in a video game.",
        "art of the {}.",
        "a photo of the small {}.",
    ],
    "Skin40": "a photo of a {}.",
    "galaxy": "a photo of a {}.",
    "IP102": "a photo of a {}.",
    "NWPU_RESISC45": "a centered satellite photo of {}.",
    "RFMiD": "a fundus image of {}.",
    "RFMiD12": "a fundus image of {}.",
    "TCGA12": "a photo of a {}.",
    "NEU_CLS": "a photo of a hot-rolled steel plate with {}.",
}


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model


class feature_map_extractor(nn.Module):
    def __init__(self, model, layer_name):
        super(feature_map_extractor, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self._features = {layer: torch.empty(0) for layer in layer_name}

        for layer_id in layer_name:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn
    
    def forward(self, x):
        image_feat = self.model(x)
        return self._features, image_feat


class container(nn.Module):
    def __init__(self, finetune_layer4, finetune_attnpool, finetune_attnpool_model):
        super(container, self).__init__()
        self.finetune_layer4 = finetune_layer4
        self.finetune_attnpool = finetune_attnpool
        self.finetune_attnpool_model = finetune_attnpool_model

    def forward(self, feature_map_layer3, feature_map_layer4):
        feature_map_layer4_finbetune = self.finetune_layer4(feature_map_layer3)
        feature_map_attnpool = self.finetune_attnpool(feature_map_layer4_finbetune)

        feature_map_attnpool_model = self.finetune_attnpool_model(feature_map_layer4)

        return feature_map_attnpool, feature_map_attnpool_model
        

class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        with torch.no_grad():
            if "ImageNet" not in cfg.DATASET.NAME:
                temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
                prompts = [temp.format(c.replace('_', ' ')) for c in classnames]
                prompts = torch.cat([clip.tokenize(p) for p in prompts])
                prompts = prompts.to('cuda')
                clip_model = clip_model.cuda()
                text_features = clip_model.encode_text(prompts)
            else:
                temp_list = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
                text_features = []
                for temp in temp_list:
                    prompts = [temp.format(c.replace('_', ' ')) for c in classnames]
                    prompts = torch.cat([clip.tokenize(p) for p in prompts])
                    prompts = prompts.to('cuda')
                    clip_model = clip_model.cuda()
                    text_features_temp = clip_model.encode_text(prompts)
                    text_features.append(text_features_temp)
                text_features = torch.stack(text_features, dim=0).mean(dim=0)

            self.text_feature_test = text_features.cuda()
            self.text_feature = text_features.cuda()

        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.feature_extractor = feature_map_extractor(self.image_encoder, ['layer4', 'layer3'])
        self.finetune_container = container(copy.deepcopy(clip_model.visual.layer4), copy.deepcopy(clip_model.visual.attnpool), copy.deepcopy(clip_model.visual.attnpool))


    def forward(self, image):
        text_features = self.text_feature

        with torch.no_grad():
            feat_dict, image_features = self.feature_extractor(image.type(self.dtype))
        feature_map_layer3 = feat_dict['layer3']
        feature_map_layer4 = feat_dict['layer4']
        feature_map_attnpool, feature_map_attnpool_model = self.finetune_container(feature_map_layer3, feature_map_layer4)
        image_features_ad = feature_map_attnpool
        image_features_ad_little = feature_map_attnpool_model

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features_ad = image_features_ad / image_features_ad.norm(dim=-1, keepdim=True)
        image_features_ad_little = image_features_ad_little / image_features_ad_little.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        logits_ad = logit_scale * image_features_ad @ text_features.t()
        logits_ad_little = logit_scale * image_features_ad_little @ text_features.t()

        raw_logits = logits

        alpha = 0.2
        alpha_little = 0.2
        logits = alpha * logits_ad + alpha_little * logits_ad_little + (1 - alpha - alpha_little) * logits

        return logits, raw_logits, logits_ad, logits_ad_little
    

@TRAINER_REGISTRY.register()
class CoMuCo_save_amp_rn50(TrainerX):
    """ CLIP-Adapter """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            if 'CNN_Adapter' not in name and 'finetune' not in name:
                param.requires_grad_(False)
        
        # print the number of trainable parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {num_params}')

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)
        
        self.model.to(self.device)
        params = list(self.model.finetune_container.parameters())
        self.optim = build_optimizer(self.model.finetune_container, cfg.OPTIM, params)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.scaler = torch.cuda.amp.GradScaler()
        self.register_model('finetune_container', self.model.finetune_container, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        optim = self.optim
        sched = self.sched
        scaler = self.scaler
        image, label = self.parse_batch_train(batch)
        with torch.cuda.amp.autocast(enabled=True):
            output, raw_output, output_fr, output_fi = self.model(image)
            loss = F.cross_entropy(output, label)

            prior_reg_fr = F.l1_loss(raw_output, output_fr)
            loss += prior_reg_fr * 0.1
            prior_reg_fi = F.l1_loss(raw_output, output_fi)
            loss += prior_reg_fi * 0.1

            con_reg_1 = F.kl_div(F.log_softmax(output_fr, dim=1), F.softmax(output_fi, dim=1), reduction="batchmean")
            con_reg_2 = F.kl_div(F.log_softmax(output_fi, dim=1), F.softmax(output_fr, dim=1), reduction="batchmean")
            con_reg = (con_reg_1 + con_reg_2) / 2
            loss += con_reg * 0.1

        # self.model_backward_and_update(loss)
        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            self.model_path_tmp = model_path

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']
            
            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models[name].load_state_dict(state_dict, strict=False)
   
   
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader
        print(f"Evaluate on the *{split}* set")
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output, _, _, _ = self.model(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()
        
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
