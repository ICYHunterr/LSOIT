python train.py --dataset rest15 --cs_dropout 0.2 --linear_dropout 0.2  --dep_dim 30 --use_const_sememe_dep pyramid2_layer_norm_drop --pyramid_hidden_dim 256
python train.py --dataset laptop14 --cs_dropout 0.2 --linear_dropout 0.2  --dep_dim 30 --use_const_sememe_dep pyramid2_layer_norm_drop --pyramid_hidden_dim 256
python train.py --dataset rest16 --cs_dropout 0.2 --linear_dropout 0.2 --dep_dim 30 --use_const_sememe_dep pyramid2_layer_norm_drop --pyramid_hidden_dim 512
python train.py --dataset mams --cs_dropout 0.2 --linear_dropout 0.3 --dep_dim 40 --use_const_sememe_dep pyramid2_layer_norm_drop --pyramid_hidden_dim 512
python train.py --dataset twitter --cs_dropout 0.2 --linear_dropout 0.1 --dep_dim 50 --use_const_sememe_dep pyramid2_layer_norm_drop --pyramid_hidden_dim 512
python train.py --dataset rest14 --cs_dropout 0.2 --linear_dropout 0.3 --dep_dim 40 --use_const_sememe_dep pyramid2_layer_norm_drop --pyramid_hidden_dim 512
python train.py --dataset small --cs_dropout 0.3 --linear_dropout 0.3 --dep_dim 40 --use_const_sememe_dep pyramid2_layer_norm_drop --pyramid_hidden_dim 512