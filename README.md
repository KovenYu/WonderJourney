<p align="center">
    <img src="assets/logo.png" height=150>
</p>

# WonderJourney: Going from Anywhere to Everywhere

<div align="center">

[![a](https://img.shields.io/badge/Website-WonderJourney-blue)](https://kovenyu.com/wonderjourney/)
[![arXiv](https://img.shields.io/badge/arXiv-2312.03884-red)](https://arxiv.org/abs/2312.03884)
[![twitter](https://img.shields.io/twitter/url?label=Koven_Yu&url=https%3A%2F%2Ftwitter.com%2FKoven_Yu)](https://twitter.com/Koven_Yu)
</div>



> #### [WonderJourney: Going from Anywhere to Everywhere](https://arxiv.org/abs/2312.03884)
> ##### [Hong-Xing "Koven" Yu](https://kovenyu.com/), [Haoyi Duan](https://haoyi-duan.github.io/), [Junhwa Hur](https://hurjunhwa.github.io/), [Kyle Sargent](https://kylesargent.github.io/), [Michael Rubinstein](https://people.csail.mit.edu/mrub/), [William T. Freeman](https://billf.mit.edu/), [Forrester Cole](https://people.csail.mit.edu/fcole/), [Deqing Sun](https://deqings.github.io/), [Noah Snavely](https://www.cs.cornell.edu/~snavely/), [Jiajun Wu](https://jiajunwu.com/), [Charles Herrmann](https://scholar.google.com/citations?user=LQvi5XAAAAAJ&hl=en)


## Getting Started

### Installation
For the installation to be done correctly, please proceed only with CUDA-compatible GPU available.
It requires 24GB GPU memory to run.

Clone the repo and create the environment:
```bash
git clone https://github.com/KovenYu/WonderJourney.git
cd WonderJourney
mamba create --name wonderjourney python=3.10
mamba activate wonderjourney
```
We are using  <a href="https://github.com/facebookresearch/pytorch3d" target="_blank">Pytorch3D</a> to perform rendering.
Run the following commands to install it or follow their <a href="https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md" target="_blank">installation guide</a> (it may take some time).
```bash
mamba install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
mamba install -c fvcore -c iopath -c conda-forge fvcore iopath
mamba install -c bottler nvidiacub
mamba install pytorch3d -c pytorch3d
```

Install the rest of the requirements:

```bash
pip install -r requirements.txt
```

Load English language model for spacy:

```bash
python -m spacy download en_core_web_sm
```

Export your OpenAI api_key (since we use GPT-4 to generate scene descriptions):

```bash
export OPENAI_API_KEY='your_api_key_here'
```

Download Midas DPT model and put it to the root directory.
```bash
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
```

### Run examples 

- Example config file

  To run an example, first you need to write a config. An example config `./config/village.yaml` is shown below:

  ```yaml
  runs_dir: output/56_village
  
  example_name: village
  
  seed: -1
  frames: 10
  save_fps: 10
  
  finetune_decoder_gen: True
  finetune_decoder_interp: False  # Turn on this for higher-quality rendered video
  finetune_depth_model: True
  
  num_scenes: 4
  num_keyframes: 2
  use_gpt: True
  kf2_upsample_coef: 4
  skip_interp: False
  skip_gen: False
  enable_regenerate: True
  
  debug: True
  inpainting_resolution_gen: 512
  
  rotation_range: 0.45
  rotation_path: [0, 0, 0, 1, 1, 0, 0, 0]
  camera_speed_multiplier_rotation: 0.2
  ```

  The total frames of the generated example is `num_scenes` $\times$ `num_keyframes`. You can manually adjust `rotation_path` in the config file to control the rotation state of the camera in each frame. A value of $0$ indicates moving straight, $1$ signifies a right turn, and $-1$ indicates a left turn.  

- Run

  ```bash
  python run.py --example_config config/village.yaml
  ```
  You will see results in `output/56_village/{time-string}_merged`.

### How to add more examples?

We highly encourage you to add new images and try new stuff!
You would need to do the image-caption pairing separately (e.g., using DALL-E to generate image and GPT4V to generate description).

- Add a new image in `./examples/images/`.

- Add content of this new image in `./examples/examples.yaml`.

  Here is an example:

  ```yaml
  - name: new_example
    image_filepath: examples/images/new_example.png
    style_prompt: DSLR 35mm landscape
    content_prompt: scene name, object 1, object 2, object 3
    negative_prompt: ''
    background: ''
  ```

  - **content_prompt**: "scene name", "object 1", "object 2", "object 3"

  - **negative_prompt** and **background** are optional

  For controlled journey, you need to add `control_text`. Examples are as follow:

  ```yaml
  - name: poem_jiangxue
    image_filepath: examples/images/60_poem_jiangxue.png
    style_prompt: black and white color ink painting
    content_prompt: Expansive mountainous landscape, old man in traditional attire, calm river, mountains
    negative_prompt: ""
    background: ""
    control_text: ["千山鸟飞绝", "万径人踪灭", "孤舟蓑笠翁", "独钓寒江雪"]
    
  - name: poem_snowy_evening
    image_filepath: examples/images/72_poem_snowy_evening.png
    style_prompt: Monet painting
    content_prompt: Stopping by woods on a snowy evening, woods, snow, village
    negative_prompt: ""
    background: ""
    control_text: ["Snowy Woods and Farmhouse: A secluded farmhouse, a frozen lake, a dense thicket, a quiet meadow, a chilly wind, a pale twilight, a covered bridge, a rustic fence, a snow-laden tree, and a frosty ground", "The Traveler's Horse: A restless horse, a jingling harness, a snowy mane, a curious gaze, a sturdy hoof, a foggy breath, a leather saddle, a woolen blanket, a frost-covered tail, and a patient stance", "Snowfall in the Woods: A gentle snowflake, a whispering wind, a soft flurry, a white blanket, a twinkling icicle, a bare branch, a hushed forest, a crystalline droplet, a serene atmosphere, and a quiet night", "Deep, Dark Woods in the Evening: A mysterious grove, a shadowy tree, a darkened sky, a hidden trail, a silent owl, a moonlit glade, a dense underbrush, a quiet clearing, a looming branch, and an eerie stillness"]
  ```

- Write a config `config/new_example.yaml` like `./config/village.yaml` for the new example

- Run

  ```bash
  python run.py --example_config config/new_example.yaml
  ```

## Citation

```
@article{yu2023wonderjourney,
  title={WonderJourney: Going from Anywhere to Everywhere},
  author={Yu, Hong-Xing and Duan, Haoyi and Hur, Junhwa and Sargent, Kyle and Rubinstein, Michael and Freeman, William T and Cole, Forrester and Sun, Deqing and Snavely, Noah and Wu, Jiajun and Herrmann, Charles},
  journal={arXiv preprint arXiv:2312.03884},
  year={2023}
}
```

## Acknowledgement

We appreciate the authors of [SceneScape](https://github.com/RafailFridman/SceneScape), [MiDaS](https://github.com/isl-org/MiDaS), [SAM](https://github.com/facebookresearch/segment-anything), [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting), and [OneFormer](https://github.com/SHI-Labs/OneFormer) to share their code.