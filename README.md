# AutoCV2019
[![KakaoBrain](https://img.shields.io/badge/kakao-brain-ffcd00.svg)](http://kakaobrain.com/)
[![pytorch](https://img.shields.io/badge/pytorch-1.0.1-%23ee4c2c.svg)](https://pytorch.org/)
[![tensorflow](https://img.shields.io/badge/tensorflow-1.13.1-ed6c20.svg)](https://www.tensorflow.org/)

A special designed **[Fast AutoAugment][]** is implemented to effectively maximize accuracy in **various tasks** while paying attention to **limited resources**.

## Results
### Public

#### V1.XLARGE
* experiment environment: [BrainCloud][] V1.XLARGE Type (V100 1GPU, 14CPU, 122GB)

| metrics | Munster |  Chucky |   Pedro |   Decal |  Hammer |
|:-------:|--------:|--------:|--------:|--------:|--------:|
| ALC     |  0.9818 |  0.8633 |  0.8300 |  0.8935 |  0.8764 |
| 2*AUC-1 |  0.9976 |  0.9292 |  0.9147 |  0.9247 |  0.9106 |
| curves  | ![](./assets/public_final_result_v1_munster.png) | ![](./assets/public_final_result_v1_Chuckey.png) | ![](./assets/public_final_result_v1_pedro.png) | ![](./assets/public_final_result_v1_Decal.png) | ![](./assets/public_final_result_v1_Hammer.png) |

#### P1.XLARGE
* experiment environment: [BrainCloud][] P1.XLARGE Type (P40 1GPU, 6CPU, 61GB)

| metrics | Munster |  Chucky |   Pedro |   Decal |  Hammer |
|:-------:|--------:|--------:|--------:|--------:|--------:|
| ALC     |  0.9440 |  0.7835 |  0.7366 |  0.8353 |  0.8286 |
| 2*AUC-1 |  0.9977 |  0.9353 |  0.9214 |  0.9347 |  0.9142 |
| curves  | ![](./assets/public_final_result_p1_munster.png) | ![](./assets/public_final_result_p1_Chuckey.png) | ![](./assets/public_final_result_p1_pedro.png) | ![](./assets/public_final_result_p1_Decal.png) | ![](./assets/public_final_result_p1_Hammer.png) |

### Private
* experiment environment: [CodaLab](https://autodl.lri.fr/)

| metrics | beatriz | Caucase | Hippoc. |  Saturn | ukulele |
|:-------:|--------:|--------:|--------:|--------:|--------:|
| ALC     |  0.6756 |  0.7359 |  0.7744 |  0.8309 |  0.9075 |
| 2*AUC-1 |  0.8014 |  0.9411 |  0.9534 |  0.9884 |  0.9985 |
| curves  | ![](./assets/private_final_result_beatriz.png) | ![](./assets/private_final_result_Caucase.png) | ![](./assets/private_final_result_Hippocrate.png) | ![](./assets/private_final_result_Saturn.png) | ![](./assets/private_final_result_ukulele.png) |


## Environment Setup & Experiments
* base docker environment: https://hub.docker.com/r/evariste/autodl

* pre requirements
```
apt update
apt install python3-tk
```

* clone and init. the repository
```bash
$ git clone https://github.com/kakaobrain/autocv2019 && cd autocv2019
$ # 3rd parties libarary
$ git submodule init
$ git submodule update
$ # download pretrained models
$ wget https://download.pytorch.org/models/resnet18-5c106cde.pth -O ./models/resnet18-5c106cde.pth
$ # download public datasets
$ cd autodl; python download_public_datasets.py; cd ..
```

* run public datasets
```bash
$ python autodl/run_local_test.py -time_budget=1200 -code_dir='./' -dataset_dir='autodl/AutoDL_public_data/Munster/'; cp autodl/AutoDL_scoring_output/learning-curve-*.png ./results
$ python autodl/run_local_test.py -time_budget=1200 -code_dir='./' -dataset_dir='autodl/AutoDL_public_data/Chucky/'; cp autodl/AutoDL_scoring_output/learning-curve-*.png ./results
$ python autodl/run_local_test.py -time_budget=1200 -code_dir='./' -dataset_dir='autodl/AutoDL_public_data/Pedro/'; cp autodl/AutoDL_scoring_output/learning-curve-*.png ./results
$ python autodl/run_local_test.py -time_budget=1200 -code_dir='./' -dataset_dir='autodl/AutoDL_public_data/Decal/'; cp autodl/AutoDL_scoring_output/learning-curve-*.png ./results
$ python autodl/run_local_test.py -time_budget=1200 -code_dir='./' -dataset_dir='autodl/AutoDL_public_data/Hammer/'; cp autodl/AutoDL_scoring_output/learning-curve-*.png ./results
```

* (optional) display learning curve
```bash
$ # item2 utils to visualize learning curve
$ wget https://www.iterm2.com/utilities/imgcat -O bin/imgcat; chmod 0677 bin/imgcat
$ bin/imgcat ./results/learning-curve-*.png
```

## Authors and Licensing

this project is developed by [Woonhyuk Baek][], [Ildoo Kim][] and [Sungbin Lim][] at
[Kakao Brain][]. It is distributed under [Apache License
2.0](LICENSE).

## References & Opensources
1. [Fast AutoAugment][]
    - paper: https://arxiv.org/abs/1905.00397
    - codes: https://github.com/kakaobrain/fast-autoaugment
2. Pretrained models for Pytorch
    - codes: https://github.com/Cadene/pretrained-models.pytorch
3. TorchVision models
    - pages: https://pytorch.org/docs/stable/torchvision/models.html
3. TQDM: Progress Bar for Python and CLI
    - codes: https://github.com/tqdm/tqdm
4. AutoCV/AutoDL startking kit
    - codes: https://github.com/zhengying-liu/autodl_starting_kit_stable

[Kakao Brain]: https://kakaobrain.com/
[BrainCloud]: https://cloud.kakaobrain.com/
[Sungbin Lim]: https://github.com/sungbinlim
[Ildoo Kim]: https://github.com/ildoonet
[Woonhyuk Baek]: https://github.com/wbaek
[Fast AutoAugment]: https://github.com/kakaobrain/fast-autoaugment
