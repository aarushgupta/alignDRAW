# Diffusion+Transformer Based Text-to-Image Generation

This repo serves our code for the 10-701 class project. It contains a TTI model based on a diffusion generator and a RoBERTa language backbone for text conditioning.  

We also implemented the alignDRAW model from [Generating Images from Captions with Attention](http://arxiv.org/abs/1511.02793) in PyTorch as a part of this project (on branch aligndraw).

<p align="center">
<img src="./images/tti_32_ft_ep_249.gif" title="Samples of MS-COCO dataset generated using the our TTI model (Image sizez 32)" alt="Generated Data Animation">

<!-- <img src="./images/tti_64_ft_ep_246.gif" title="Samples of MS-COCO dataset generated using the our TTI model (Image sizez 64)" alt="Generated Data Animation"> -->
</p>

## Training


To train the model with MS-COCO data, run

```sh
bash run_tti_64_coco_ft_roberta.sh $gpu_ids $run_idx $dataset_path
```

where gpu_ids are the GPU numbers you want to train on (for example, 1,2).

This script trains a model with 64 input image size with 200 diffusion steps and fine-tuns the RoBERTa backbone as well.

Various arguments are provided to train with a different image size, number of diffusion steps, frozen language backbone, etc.

## Generating New Images
To generate new images run 

```sh
bash run_tti_eval_coco.sh $gpu_ids $original_expt_run_idx
```

`original_expt_run_idx` is the original experiment's run idx. 

## Metrics

We evaluate the trained models on FID and CLIP Scores. 

### Metrics Setup
- Download clipscore repository (https://github.com/jmhessel/clipscore (Need to install dependencies first: OpenAI CLIP and PyCOCOEvalCap)

```sh
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/jmhessel/pycocoevalcap.git
git clone https://github.com/jmhessel/clipscore.git

```

- Setup python-fid to compute FID scores

```sh
pip install pytorch-fid
```
### Evaluation

1. Run `gen_images_for_metrics.sh` to run model inference and generate random images for metric calculation

```sh
bash gen_images_for_metrics.sh $gpu_ids $original_expt_run_idx $pretrained_model_path
```

`original_expt_run_idx` is the original experiment's run idx. Ouput data (needed for the next steps) is stored in a folder `./eval_data_run_${run_idx}`. 

2. Compute CLIP Scores in the following way:

```sh
cd clipscore

python clipscore.py ../eval_data_run_${run_idx}/gt/gt_captions.json ../eval_data_run_${run_idx}/gt/images/

python clipscore.py ../eval_data_run_${run_idx}/tti/tti_captions.json ../eval_data_run_${run_idx}/tti/images/

cd -
```


3. Compute FID scores in the following way:

```sh
python -m pytorch_fid ./eval_data_run_${run_idx}/gt/images/ ./eval_data_run_${run_idx}/tti/images/
```


## Results on MS-COCO dataset
<p align="center">
<img src="./images/tti_32_ft_ep_249.gif" title="Samples of MS-COCO dataset generated using the our TTI model (Image sizez 32)" alt="Generated Data Animation">

<img src="./images/tti_64_ft_ep_246.gif" title="Samples of MS-COCO dataset generated using the our TTI model (Image sizez 64)" alt="Generated Data Animation">

| Model      | FID ($\downarrow$) | GT CLIP Score ($\uparrow$)| TTI CLIP Score ($\uparrow$)| Checkpoints ([Drive Link](https://drive.google.com/drive/folders/1DZcWFM1MACbo4KOylFdHItSDZW3v0RkY?usp=share_link))| 
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 32x32 (RoBERTa frozen)   | 76.623 | 0.6249 |0.5286 | `checkpoint/model_0` |
| 64x64 (RoBERTa frozen)   | 69.985 | 0.6172 |0.5467 | `checkpoint/model_2` |
| 32x32 (RoBERTa fine-tuned) | 104.627 | 0.6907 |0.5192 | `checkpoint/model_3` |
| 64x64 (RoBERTa fine-tuned) | 92.941  | 0.7023 |0.5467 | `checkpoint/model_4` |


- Models are identified by their input image size and if their language backbone is fine-tuned or not. 

- GT CLIP Score is the CLIP score obtained from downsampled (to input image size) images and corresponding captions and serves as a benchmark to make sense of the TTI CLIP Scores (the nearer TTI CLIP scores are to the GT CLIP scores, the better)

<!-- ### Some more generated images: -->
<!-- <center><img src="./images/sample_image_tti_1.png"></center> -->
<!-- <center>This is an image</center> -->

## References
1. Generating Images from Captions with Attention [[arxiv]](http://arxiv.org/abs/1511.02793)
2. DRAW: A Recurrent Neural Network For Image Generation. [[arxiv](https://arxiv.org/abs/1502.04623)]
3. ericjang/draw [[repo](https://github.com/ericjang/draw)]
4. What is DRAW (Deep Recurrent Attentive Writer)? [[blog](http://kvfrans.com/what-is-draw-deep-recurrent-attentive-writer/)]
5. HuggingFace Annotated Diffusion Model [[blog](https://huggingface.co/blog/annotated-diffusion)]
6. HuggingFace transformers library [[page](https://huggingface.co/docs/transformers/index)]
