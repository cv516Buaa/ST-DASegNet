# Copyright (c) OpenMMLab. All rights reserved.
from .aspp_head import ASPPHead
from .fcn_head import FCNHead
from .segformer_head import SegformerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead

__all__ = [
    'FCNHead', 'ASPPHead',
    'DepthwiseSeparableASPPHead',
    'SegformerHead'
]
