
from models.CKAs_module import conv_cka, bn_cka


def get_cka_params(model):
    params = []
    for name, module in model.encoder.named_modules():
        if isinstance(module, conv_cka):
            if hasattr(module, 'delta') and module.delta.requires_grad:
                params.append(module.delta)
        if isinstance(module, bn_cka):
            if hasattr(module, 'alpha') and module.alpha.requires_grad:
                params.append(module.alpha)
            params.extend(module.domain_transform.parameters())
    return params
