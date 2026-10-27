# Adapted from the user's existing importance implementation; see PROVENANCE.json.
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
CIFAR_NUM_CLASSES = 45
CIFAR_IMAGE_SIZE = 224
ALL_CLASSES = tuple(range(45))
CLASS_NAMES = tuple(str(x) for x in ALL_CLASSES)
PAPER_IMPORTANCE_METHOD = 'TFIDF_x_PaperCAMRF'
PAPER_IMPORTANCE_VERSION = 'paper_camrf_relu_peak_norm_before_upsample_v1'


def normalize_classes(class_indices: Sequence[int], argument_name: str) -> Tuple[int, ...]:
    classes = tuple(dict.fromkeys((int(v) for v in class_indices)))
    if not classes:
        raise ValueError(f'{argument_name} cannot be empty.')
    invalid = [value for value in classes if not 0 <= value < CIFAR_NUM_CLASSES]
    if invalid:
        raise ValueError(f'{argument_name} contains invalid class indices: {invalid}')
    return classes

def select_mask_compatible_conv_layers(model: nn.Module) -> Dict[str, nn.Conv2d]:
    """
    Select the exact ResNet-18 convolution set used by the pruning script.

    Included:
      conv1
      layer1.*.conv1 / conv2
      layer2.*.conv1 / conv2
      layer3.*.conv1 / conv2
      layer4.*.conv1 / conv2

    Downsample projection convolutions are excluded so the saved importance
    profiles match the mask-compatible pruning layer set exactly.
    """
    layers: Dict[str, nn.Conv2d] = {}
    if not isinstance(model.conv1, nn.Conv2d):
        raise TypeError('Expected model.conv1 to be Conv2d.')
    layers['conv1'] = model.conv1
    for stage_name in ('layer1', 'layer2', 'layer3', 'layer4'):
        stage = getattr(model, stage_name)
        for block_index, block in enumerate(stage):
            for conv_name in ('conv1', 'conv2'):
                module = getattr(block, conv_name)
                if not isinstance(module, nn.Conv2d):
                    raise TypeError(f'{stage_name}.{block_index}.{conv_name} is not Conv2d.')
                layers[f'{stage_name}.{block_index}.{conv_name}'] = module
    expected = 17
    if len(layers) != expected:
        raise RuntimeError(f'Expected {expected} mask-compatible ResNet-18 Conv2d layers, found {len(layers)}.')
    return layers

@dataclass(frozen=True)
class RFInfo:
    rf: float = 1.0
    jump: float = 1.0
    start: float = 0.5

def _first(value: Any) -> float:
    if isinstance(value, Iterable) and (not isinstance(value, (str, bytes))):
        return float(tuple(value)[0])
    return float(value)

def _apply_spatial_layer(info: RFInfo, layer: nn.Module) -> RFInfo:
    kernel = _first(layer.kernel_size)
    raw_stride = getattr(layer, 'stride', 1)
    stride = kernel if raw_stride is None else _first(raw_stride)
    padding = _first(getattr(layer, 'padding', 0))
    dilation = _first(getattr(layer, 'dilation', 1))
    effective_kernel = dilation * (kernel - 1.0) + 1.0
    return RFInfo(rf=info.rf + (effective_kernel - 1.0) * info.jump, jump=info.jump * stride, start=info.start + ((effective_kernel - 1.0) / 2.0 - padding) * info.jump)

def compute_resnet18_rf(model: nn.Module) -> Dict[str, RFInfo]:
    """
    Receptive-field table for the 17 selected ResNet-18 convolutions.

    The transformed residual branch is followed through conv1 -> conv2
    inside every BasicBlock. For stride-changing blocks, the stride is
    carried by block.conv1. The transformed branch has at least the
    receptive-field extent of the shortcut, so this gives the maximum
    spatial support used by CAM-RF.
    """
    selected = select_mask_compatible_conv_layers(model)
    table: Dict[str, RFInfo] = {}
    state = RFInfo()
    state = _apply_spatial_layer(state, model.conv1)
    table['conv1'] = state
    state = _apply_spatial_layer(state, model.maxpool)
    for stage_name in ('layer1', 'layer2', 'layer3', 'layer4'):
        stage = getattr(model, stage_name)
        for block_index, block in enumerate(stage):
            conv1_name = f'{stage_name}.{block_index}.conv1'
            state = _apply_spatial_layer(state, block.conv1)
            table[conv1_name] = state
            conv2_name = f'{stage_name}.{block_index}.conv2'
            state = _apply_spatial_layer(state, block.conv2)
            table[conv2_name] = state
    if set(table) != set(selected):
        missing = sorted(set(selected) - set(table))
        extra = sorted(set(table) - set(selected))
        raise RuntimeError(f'ResNet-18 RF table mismatch; missing={missing}, extra={extra}.')
    return table

class ActivationCollector:

    def __init__(self, layers: Mapping[str, nn.Module]) -> None:
        self.outputs: Dict[str, torch.Tensor] = {}
        self.handles = [module.register_forward_hook(self._make_hook(name)) for name, module in layers.items()]

    def _make_hook(self, name: str):

        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.outputs[name] = output.detach()
        return hook

    def clear(self) -> None:
        self.outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

class GradCAM:

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.feature: Optional[torch.Tensor] = None
        self.handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.feature = output

    def generate(self, images: torch.Tensor, target_class: int, output_size: Tuple[int, int]) -> torch.Tensor:
        self.feature = None
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(images)
            if self.feature is None:
                raise RuntimeError('Grad-CAM target-layer hook did not run.')
            target_score = logits[:, target_class].sum()
            gradients = torch.autograd.grad(target_score, self.feature, retain_graph=False, create_graph=False)[0]
        feature = self.feature.detach()
        weights = gradients.detach().mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature).sum(dim=1, keepdim=True))
        # Paper: normalize the native Grad-CAM grid BEFORE upsampling.
        maximum = cam.flatten(1).max(dim=1).values[:, None, None, None]
        cam = cam / maximum.clamp_min(1e-08)
        cam = F.interpolate(cam, size=output_size, mode='bilinear', align_corners=False).squeeze(1)
        self.feature = None
        return cam.detach()

    def close(self) -> None:
        self.handle.remove()

def select_spatial_gradcam_layer(model: nn.Module, input_size: int=CIFAR_IMAGE_SIZE, minimum_grid_size: int=4) -> Tuple[str, nn.Module, Tuple[int, int]]:
    """
    Choose the deepest selected ResNet-18 convolution whose output feature
    grid is at least minimum_grid_size x minimum_grid_size.
    """
    selected_layers = select_mask_compatible_conv_layers(model)
    candidates = list(selected_layers.items())[::-1]
    shapes: Dict[str, Tuple[int, int]] = {}
    handles = []
    for name, layer in candidates:

        def record_shape(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor, layer_name: str=name) -> None:
            shapes[layer_name] = tuple(output.shape[-2:])
        handles.append(layer.register_forward_hook(record_shape))
    parameter = next(model.parameters())
    dummy = torch.zeros(1, 3, input_size, input_size, device=parameter.device, dtype=parameter.dtype)
    try:
        with torch.inference_mode():
            model(dummy)
    finally:
        for handle in handles:
            handle.remove()
    for name, layer in candidates:
        shape = shapes.get(name)
        if shape is not None and min(shape) >= minimum_grid_size:
            return (name, layer, shape)
    raise RuntimeError(f'No ResNet-18 Grad-CAM layer has at least {minimum_grid_size}x{minimum_grid_size} grid.')

@torch.no_grad()
def compute_class_channel_activations(model: nn.Module, loader: DataLoader, device: torch.device, available_classes: Sequence[int]) -> Dict[str, torch.Tensor]:
    """
    A*[c,k,l] =
        mean_{x:y=c} GAP(ReLU(O_l,k(x))).
    """
    classes = normalize_classes(available_classes, 'available_classes')
    class_to_row = {cls: row for row, cls in enumerate(classes)}
    conv_layers = select_mask_compatible_conv_layers(model)
    sums = {name: torch.zeros(len(classes), layer.out_channels, dtype=torch.float64) for name, layer in conv_layers.items()}
    counts = torch.zeros(len(classes), dtype=torch.long)
    collector = ActivationCollector(conv_layers)
    processed = 0
    model.eval()
    try:
        for batch_index, (images, labels) in enumerate(loader, 1):
            label_values = [int(value) for value in labels.tolist()]
            row_indices = torch.tensor([class_to_row[value] for value in label_values], dtype=torch.long)
            images = images.to(device, non_blocking=True)
            collector.clear()
            model(images)
            missing_layers = set(conv_layers) - set(collector.outputs)
            if missing_layers:
                raise RuntimeError(f'Activation hooks did not run for: {sorted(missing_layers)}')
            for name, feature in collector.outputs.items():
                image_channel_activation = torch.relu(feature).mean(dim=(2, 3)).cpu().double()
                sums[name].index_add_(0, row_indices, image_channel_activation)
            counts.index_add_(0, row_indices, torch.ones_like(row_indices))
            processed += len(label_values)
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(f'FUDP activations: {processed}/{len(loader.dataset)} images')
    finally:
        collector.close()
    if (counts == 0).any():
        missing_classes = [classes[row] for row, count in enumerate(counts.tolist()) if count == 0]
        raise RuntimeError(f'Cannot calculate class means; no samples for {missing_classes}.')
    denominator = counts.double().view(-1, 1)
    return {name: (layer_sum / denominator).float() for name, layer_sum in sums.items()}

def compute_fudp_tfidf_importance(class_channel_activations: Mapping[str, torch.Tensor], available_classes: Sequence[int], target_class: int, epsilon: float=1e-12) -> Dict[str, torch.Tensor]:
    """
    Same FUDP/TF-IDF definition:

      TF[c,k] =
        A*[c,k] / sum_j A*[c,j]

      mean_c =
        mean_k A*[c,k]

      DF[k] =
        sum_c 1[A*[c,k] >= mean_c]

      IDF[k] =
        log((1 + |C|) / (1 + DF[k]))

      S[c,k] =
        TF[c,k] * IDF[k]
    """
    classes = normalize_classes(available_classes, 'available_classes')
    target = normalize_classes((target_class,), 'target_class')[0]
    if target not in classes:
        raise ValueError('target_class must belong to available_classes.')
    class_to_row = {cls: row for row, cls in enumerate(classes)}
    result: Dict[str, torch.Tensor] = {}
    for layer_name, raw_activation in class_channel_activations.items():
        activation = torch.as_tensor(raw_activation).detach().cpu().float()
        if activation.ndim != 2 or activation.shape[0] != len(classes):
            raise ValueError(f'{layer_name}: expected [{len(classes)}, C_out], found {tuple(activation.shape)}.')
        per_class_mean = activation.mean(dim=1, keepdim=True)
        document_frequency = (activation >= per_class_mean).sum(dim=0).float()
        idf = torch.log(torch.tensor(1.0 + len(classes), dtype=torch.float32) / (1.0 + document_frequency))
        target_activation = activation[class_to_row[target]]
        tf = target_activation / target_activation.sum().clamp_min(epsilon)
        result[layer_name] = (tf * idf).cpu()
    return result

def _rectangle_sum(integral_flat: torch.Tensor, y: torch.Tensor, x: torch.Tensor, integral_width: int) -> torch.Tensor:
    batch_size = integral_flat.shape[0]
    indices = y[:, None] * integral_width + x[None, :]
    indices = indices.flatten()[None, :].expand(batch_size, -1)
    return integral_flat.gather(1, indices)

def mean_cam_in_receptive_fields(cam: torch.Tensor, feature_height: int, feature_width: int, info: RFInfo) -> torch.Tensor:
    batch_size, image_height, image_width = cam.shape
    device = cam.device
    center_y = info.start + torch.arange(feature_height, device=device, dtype=torch.float32) * info.jump
    center_x = info.start + torch.arange(feature_width, device=device, dtype=torch.float32) * info.jump
    half_rf = info.rf / 2.0
    y0 = torch.floor(center_y - half_rf).long().clamp(0, image_height - 1)
    x0 = torch.floor(center_x - half_rf).long().clamp(0, image_width - 1)
    y1 = torch.ceil(center_y + half_rf).long().clamp(1, image_height)
    x1 = torch.ceil(center_x + half_rf).long().clamp(1, image_width)
    y1 = torch.maximum(y1, y0 + 1)
    x1 = torch.maximum(x1, x0 + 1)
    integral = F.pad(cam, (1, 0, 1, 0)).cumsum(dim=1).cumsum(dim=2)
    integral_width = image_width + 1
    integral_flat = integral.flatten(1)
    bottom_right = _rectangle_sum(integral_flat, y1, x1, integral_width)
    top_right = _rectangle_sum(integral_flat, y0, x1, integral_width)
    bottom_left = _rectangle_sum(integral_flat, y1, x0, integral_width)
    top_left = _rectangle_sum(integral_flat, y0, x0, integral_width)
    rectangle_sums = bottom_right - top_right - bottom_left + top_left
    area = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return rectangle_sums.reshape(batch_size, feature_height, feature_width) / area.to(cam.dtype)

def combine_fudp_and_cam_rf(fudp_importance: Mapping[str, torch.Tensor], cam_rf_factors: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if set(fudp_importance) != set(cam_rf_factors):
        raise ValueError('FUDP and CAM-RF layer sets do not match.')
    result: Dict[str, torch.Tensor] = {}
    for layer_name, raw_fudp in fudp_importance.items():
        fudp = torch.as_tensor(raw_fudp).detach().cpu().float()
        cam_rf = torch.as_tensor(cam_rf_factors[layer_name]).detach().cpu().float()
        if fudp.shape != cam_rf.shape:
            raise ValueError(f'{layer_name}: incompatible shapes {tuple(fudp.shape)} and {tuple(cam_rf.shape)}.')
        score = fudp * cam_rf
        if not torch.isfinite(score).all():
            raise RuntimeError(f'{layer_name}: FUDP+CAM contains NaN/Inf.')
        result[layer_name] = score
    return result

@torch.no_grad()
def compute_paper_camrf_importance(model: nn.Module, class_loader: DataLoader,
                                   device: torch.device, target_class: int,
                                   fudp_importance: Mapping[str, torch.Tensor],
                                   image_size: int = CIFAR_IMAGE_SIZE) -> Dict[str, torch.Tensor]:
    """Fifth method: PaperCAMRF, I = TF-IDF * mean_x[CAM-RF at ReLU peak].

    class_loader must use the same target-class images as the TF-IDF computation.
    """
    layers = select_mask_compatible_conv_layers(model)
    rf_table = compute_resnet18_rf(model)
    _, cam_layer, _ = select_spatial_gradcam_layer(model, input_size=image_size)
    collector = ActivationCollector(layers)
    gradcam = GradCAM(model, cam_layer)
    totals = {name: torch.zeros(layer.out_channels, dtype=torch.float64)
              for name, layer in layers.items()}
    count = 0
    model.eval()
    try:
        for images, labels in class_loader:
            if not torch.all(torch.as_tensor(labels) == target_class):
                raise ValueError('class_loader must contain only target_class images.')
            images = images.to(device, non_blocking=True)
            collector.clear()
            cam = gradcam.generate(images, target_class, tuple(images.shape[-2:]))
            if set(collector.outputs) != set(layers):
                raise RuntimeError('Missing activation hooks for one or more layers.')
            for name, feature in collector.outputs.items():
                rf_cam = mean_cam_in_receptive_fields(
                    cam, feature.shape[-2], feature.shape[-1], rf_table[name]
                ).clamp_min(0)
                relu_peak_idx = feature.relu().flatten(2).argmax(dim=2)
                peak_cam = rf_cam.flatten(1).gather(1, relu_peak_idx)
                totals[name] += peak_cam.double().sum(dim=0).cpu()
            count += images.shape[0]
            collector.clear()
    finally:
        gradcam.close()
        collector.close()
    if count == 0:
        raise ValueError('Empty target-class loader.')
    factors = {name: (total / count).float() for name, total in totals.items()}
    return combine_fudp_and_cam_rf(fudp_importance, factors)
