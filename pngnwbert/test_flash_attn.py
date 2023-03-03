import torch

from pngnwbert.transformers_bert.modeling_bert import BertSelfAttention
from pngnwbert.transformers_bert.configuration_bert import BertConfig

def test_flash_attn():
    config = BertConfig()
    device = 'cuda'

    torch.random.manual_seed(0)
    bert_self_att = BertSelfAttention(config, flash_attn=False).to(device).eval()

    torch.random.manual_seed(0)
    bert_self_att_flash = BertSelfAttention(config, flash_attn=True).to(device).eval()
    bert_self_att_flash.force_flash_path = True
    
    x = torch.randn(1, 10, config.hidden_size).to(device)
    attn_mask = torch.ones(1, 1, 10, 10).to(device)
    
    attn = bert_self_att(x, attn_mask)[0]
    attn_flash = bert_self_att_flash(x, attn_mask)[0]
    
    assert torch.allclose(attn, attn_flash, atol=0.01)
    print('test_flash_attn passed')
    
if __name__ == '__main__':
    test_flash_attn()