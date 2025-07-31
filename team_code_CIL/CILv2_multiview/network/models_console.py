"""
    A simple factory module that returns instances of possible modules 

"""

from .vanilla import CILv2_multiview_attention
from .attention import CILv2_multiview_attention_diego


def Models(architecture_name, configuration):
    # Baseline end-to-end behavior cloning model, with TFM in multi-view feature space
    print(f"Selected {str(architecture_name)}")
    
    if architecture_name == 'CILv2_multiview_attention':
        return CILv2_multiview_attention(configuration)
    
    elif architecture_name == 'CILv2_multiview_attention_diego':
        return CILv2_multiview_attention_diego(configuration)
    
    else:
        raise NotImplementedError(" Not found architecture name")
