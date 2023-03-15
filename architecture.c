
RegTR{
  {preprocessor}: PreprocessorGPU{}
  {kpf_encoder}: KPFEncoder{
    {encoder_blocks}: ModuleList{
      {0}: SimpleBlock{
        {KPConv}: KPConv{radius: 0.08, extent: 0.06, in_feat: 1, out_feat: 256}
        {batch_norm}: BatchNormBlock{in_feat: 256, momentum: 0.020, only_bias: False}
        {leaky_relu}: LeakyReLU{negative_slope=0.1}
      }
      {1}: ResnetBottleneckBlock{
        {unary1}: UnaryBlock{in_feat: 256, out_feat: 128, BN: True, ReLU: True}
        {KPConv}: KPConv{radius: 0.08, extent: 0.06, in_feat: 128, out_feat: 128}
        {batch_norm_conv}: BatchNormBlock{in_feat: 128, momentum: 0.020, only_bias: False}
        {unary2}: UnaryBlock{in_feat: 128, out_feat: 512, BN: True, ReLU: False}
        {unary_shortcut}: UnaryBlock{in_feat: 256, out_feat: 512, BN: True, ReLU: False}
        {leaky_relu}: LeakyReLU{negative_slope=0.1}
      }
      {2}: ResnetBottleneckBlock{
        {unary1}: UnaryBlock{in_feat: 512, out_feat: 128, BN: True, ReLU: True}
        {KPConv}: KPConv{radius: 0.08, extent: 0.06, in_feat: 128, out_feat: 128}
        {batch_norm_conv}: BatchNormBlock{in_feat: 128, momentum: 0.020, only_bias: False}
        {unary2}: UnaryBlock{in_feat: 128, out_feat: 512, BN: True, ReLU: False}
        {unary_shortcut}: Identity{}
        {leaky_relu}: LeakyReLU{negative_slope=0.1}
      }
      {3}: ResnetBottleneckBlock{
        {unary1}: UnaryBlock{in_feat: 512, out_feat: 128, BN: True, ReLU: True}
        {KPConv}: KPConv{radius: 0.08, extent: 0.06, in_feat: 128, out_feat: 128}
        {batch_norm_conv}: BatchNormBlock{in_feat: 128, momentum: 0.020, only_bias: False}
        {unary2}: UnaryBlock{in_feat: 128, out_feat: 512, BN: True, ReLU: False}
        {unary_shortcut}: Identity{}
        {leaky_relu}: LeakyReLU{negative_slope=0.1}
      }
      {4}: ResnetBottleneckBlock{
        {unary1}: UnaryBlock{in_feat: 512, out_feat: 256, BN: True, ReLU: True}
        {KPConv}: KPConv{radius: 0.16, extent: 0.12, in_feat: 256, out_feat: 256}
        {batch_norm_conv}: BatchNormBlock{in_feat: 256, momentum: 0.020, only_bias: False}
        {unary2}: UnaryBlock{in_feat: 256, out_feat: 1024, BN: True, ReLU: False}
        {unary_shortcut}: UnaryBlock{in_feat: 512, out_feat: 1024, BN: True, ReLU: False}
        {leaky_relu}: LeakyReLU{negative_slope=0.1}
      }
      {5}: ResnetBottleneckBlock{
        {unary1}: UnaryBlock{in_feat: 1024, out_feat: 256, BN: True, ReLU: True}
        {KPConv}: KPConv{radius: 0.16, extent: 0.12, in_feat: 256, out_feat: 256}
        {batch_norm_conv}: BatchNormBlock{in_feat: 256, momentum: 0.020, only_bias: False}
        {unary2}: UnaryBlock{in_feat: 256, out_feat: 1024, BN: True, ReLU: False}
        {unary_shortcut}: Identity{}
        {leaky_relu}: LeakyReLU{negative_slope=0.1}
      }
    }
  }
  {feat_proj}: Linear{in_features=1024, out_features=256, bias=True}
  {pos_embed}: PositionEmbeddingCoordsSine{}
  {transformer_encoder}: TransformerCrossEncoder{
    {layers}: ModuleList{
      {0}: TransformerCrossEncoderLayer{
        {self_attn}: MultiheadAttention{
          {out_proj}: NonDynamicallyQuantizableLinear{in_features=256, out_features=256, bias=True}
        }
        {multihead_attn}: MultiheadAttention{
          {out_proj}: NonDynamicallyQuantizableLinear{in_features=256, out_features=256, bias=True}
        }
        {linear1}: Linear{in_features=256, out_features=1024, bias=True}
        {dropout}: Dropout{p=0.0, inplace=False}
        {linear2}: Linear{in_features=1024, out_features=256, bias=True}
        {norm1}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {norm2}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {norm3}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {dropout1}: Dropout{p=0.0, inplace=False}
        {dropout2}: Dropout{p=0.0, inplace=False}
        {dropout3}: Dropout{p=0.0, inplace=False}
      }
      {1}: TransformerCrossEncoderLayer{
        {self_attn}: MultiheadAttention{
          {out_proj}: NonDynamicallyQuantizableLinear{in_features=256, out_features=256, bias=True}
        }
        {multihead_attn}: MultiheadAttention{
          {out_proj}: NonDynamicallyQuantizableLinear{in_features=256, out_features=256, bias=True}
        }
        {linear1}: Linear{in_features=256, out_features=1024, bias=True}
        {dropout}: Dropout{p=0.0, inplace=False}
        {linear2}: Linear{in_features=1024, out_features=256, bias=True}
        {norm1}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {norm2}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {norm3}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {dropout1}: Dropout{p=0.0, inplace=False}
        {dropout2}: Dropout{p=0.0, inplace=False}
        {dropout3}: Dropout{p=0.0, inplace=False}
      }
      {2}: TransformerCrossEncoderLayer{
        {self_attn}: MultiheadAttention{
          {out_proj}: NonDynamicallyQuantizableLinear{in_features=256, out_features=256, bias=True}
        }
        {multihead_attn}: MultiheadAttention{
          {out_proj}: NonDynamicallyQuantizableLinear{in_features=256, out_features=256, bias=True}
        }
        {linear1}: Linear{in_features=256, out_features=1024, bias=True}
        {dropout}: Dropout{p=0.0, inplace=False}
        {linear2}: Linear{in_features=1024, out_features=256, bias=True}
        {norm1}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {norm2}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {norm3}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {dropout1}: Dropout{p=0.0, inplace=False}
        {dropout2}: Dropout{p=0.0, inplace=False}
        {dropout3}: Dropout{p=0.0, inplace=False}
      }
      {3}: TransformerCrossEncoderLayer{
        {self_attn}: MultiheadAttention{
          {out_proj}: NonDynamicallyQuantizableLinear{in_features=256, out_features=256, bias=True}
        }
        {multihead_attn}: MultiheadAttention{
          {out_proj}: NonDynamicallyQuantizableLinear{in_features=256, out_features=256, bias=True}
        }
        {linear1}: Linear{in_features=256, out_features=1024, bias=True}
        {dropout}: Dropout{p=0.0, inplace=False}
        {linear2}: Linear{in_features=1024, out_features=256, bias=True}
        {norm1}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {norm2}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {norm3}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {dropout1}: Dropout{p=0.0, inplace=False}
        {dropout2}: Dropout{p=0.0, inplace=False}
        {dropout3}: Dropout{p=0.0, inplace=False}
      }
      {4}: TransformerCrossEncoderLayer{
        {self_attn}: MultiheadAttention{
          {out_proj}: NonDynamicallyQuantizableLinear{in_features=256, out_features=256, bias=True}
        }
        {multihead_attn}: MultiheadAttention{
          {out_proj}: NonDynamicallyQuantizableLinear{in_features=256, out_features=256, bias=True}
        }
        {linear1}: Linear{in_features=256, out_features=1024, bias=True}
        {dropout}: Dropout{p=0.0, inplace=False}
        {linear2}: Linear{in_features=1024, out_features=256, bias=True}
        {norm1}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {norm2}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {norm3}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {dropout1}: Dropout{p=0.0, inplace=False}
        {dropout2}: Dropout{p=0.0, inplace=False}
        {dropout3}: Dropout{p=0.0, inplace=False}
      }
      {5}: TransformerCrossEncoderLayer{
        {self_attn}: MultiheadAttention{
          {out_proj}: NonDynamicallyQuantizableLinear{in_features=256, out_features=256, bias=True}
        }
        {multihead_attn}: MultiheadAttention{
          {out_proj}: NonDynamicallyQuantizableLinear{in_features=256, out_features=256, bias=True}
        }
        {linear1}: Linear{in_features=256, out_features=1024, bias=True}
        {dropout}: Dropout{p=0.0, inplace=False}
        {linear2}: Linear{in_features=1024, out_features=256, bias=True}
        {norm1}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {norm2}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {norm3}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
        {dropout1}: Dropout{p=0.0, inplace=False}
        {dropout2}: Dropout{p=0.0, inplace=False}
        {dropout3}: Dropout{p=0.0, inplace=False}
      }
    }
    {norm}: LayerNorm{{256,}, eps=1e-05, elementwise_affine=True}
  }
  {correspondence_decoder}: CorrespondenceRegressor{
    {coor_mlp}: Sequential{
      {0}: Linear{in_features=256, out_features=256, bias=True}
      {1}: ReLU{}
      {2}: Linear{in_features=256, out_features=256, bias=True}
      {3}: ReLU{}
      {4}: Linear{in_features=256, out_features=3, bias=True}
    }
    {conf_logits_decoder}: Linear{in_features=256, out_features=1, bias=True}
  }
  {overlap_criterion}: BCEWithLogitsLoss{}
  {feature_criterion}: InfoNCELossFull{}
  {feature_criterion_un}: InfoNCELossFull{}
  {corr_criterion}: CorrCriterion{}
}



modelnet:
Transform:
    SetDeterministic: True
    SplitSourceRef: sample['points_raw']=sample['points_src']=sample['points_ref'], generate sample['corresp']
    RandomCrop: idx_deterministic=seed, (分别对src和ref进行随机2048->1433<球面均匀采样>,(1433->xxx),对两部分的overlap进行记录为gt)
    RandomTransformSE3：idx_deterministic=seed, transform src and save transformed-src, transform-gt
    Resampler: resample (random) points=717, and recollate corresp
    RandomJitter:   points+gauss noisy
    ShufflePoints:   permutation
