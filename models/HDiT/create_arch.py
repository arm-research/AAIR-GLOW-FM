# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
from models.HDiT.hourglass import image_transformer_v2 as itv2
from models.HDiT.hourglass.image_transformer_v2 import ImageTransformerDenoiserModelV2


def _hdit_config(in_channels, out_channels, widths, depths, mapping_width, mapping_depth=2, d_head=64):
    return {
        'in_channels': in_channels,
        'out_channels': out_channels,
        'widths': widths,
        'depths': depths,
        'patch_size': [2, 2],
        'self_attns': [
            {"type": "neighborhood", "d_head": d_head, "kernel_size": 7},
            {"type": "global", "d_head": d_head}
        ],
        'mapping_depth': mapping_depth,
        'mapping_width': mapping_width,
        'dropout_rate': [0, 0],
        'mapping_dropout_rate': 0.0
    }


def _hdit_family(in_channels, out_channels):
    return {
        "XL2lat": _hdit_config(in_channels, out_channels, [384, 768], [2, 11], 768),
        "M2lat": _hdit_config(in_channels, out_channels, [192, 384], [2, 6], 384),
        "S2lat": _hdit_config(in_channels, out_channels, [128, 256], [2, 4], 256),
    }


def create_arch(arch, condition_channels=0):
    # arch should be, e.g., hdit16_XL2lat
    arch_name, arch_size = arch.split('_')
    arch_config = arch_configs[arch_name][arch_size].copy()
    arch_config['in_channels'] += condition_channels
    return arch_name_to_object[arch_name](**arch_config)


arch_configs = {
    'hdit16': _hdit_family(16, 16),
}

def create_hdit_model(widths,
                      depths,
                      self_attns,
                      dropout_rate,
                      mapping_depth,
                      mapping_width,
                      mapping_dropout_rate,
                      in_channels,
                      out_channels,
                      patch_size
                      ):
    assert len(widths) == len(depths)
    assert len(widths) == len(self_attns)
    assert len(widths) == len(dropout_rate)
    mapping_d_ff = mapping_width * 3
    d_ffs = []
    for width in widths:
        d_ffs.append(width * 3)

    levels = []
    for depth, width, d_ff, self_attn, dropout in zip(depths, widths, d_ffs, self_attns, dropout_rate):
        if self_attn['type'] == 'global':
            self_attn = itv2.GlobalAttentionSpec(self_attn.get('d_head', 64))
        elif self_attn['type'] == 'neighborhood':
            self_attn = itv2.NeighborhoodAttentionSpec(self_attn.get('d_head', 64), self_attn.get('kernel_size', 7))
        elif self_attn['type'] == 'shifted-window':
            self_attn = itv2.ShiftedWindowAttentionSpec(self_attn.get('d_head', 64), self_attn['window_size'])
        elif self_attn['type'] == 'none':
            self_attn = itv2.NoAttentionSpec()
        else:
            raise ValueError(f'unsupported self attention type {self_attn["type"]}')
        levels.append(itv2.LevelSpec(depth, width, d_ff, self_attn, dropout))
    mapping = itv2.MappingSpec(mapping_depth, mapping_width, mapping_d_ff, mapping_dropout_rate)
    model = ImageTransformerDenoiserModelV2(
        levels=levels,
        mapping=mapping,
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=patch_size,
        num_classes=0,
        mapping_cond_dim=0,
    )

    return model


arch_name_to_object = {
    'hdit': create_hdit_model,
    'hdit16': create_hdit_model,
}
