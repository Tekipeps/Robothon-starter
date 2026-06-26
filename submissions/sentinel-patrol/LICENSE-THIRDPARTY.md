# Third-party assets & licenses

## Unitree Go1 (MuJoCo model)

The quadruped model under [`assets/unitree_go1/`](assets/unitree_go1) is the
**Unitree Go1** from Google DeepMind's **MuJoCo Menagerie**.

- **License:** BSD-3-Clause (see [`assets/unitree_go1/LICENSE`](assets/unitree_go1/LICENSE)).
- **Source:** https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go1
- **Vendored** here verbatim (mesh assets, `go1.xml`, `scene.xml`) so the submission
  runs with **no network download** and is fully reproducible.

### Modifications

The vendored `go1.xml` is unchanged except for one addition made by the course
builder at load time (`sentinel/scene.py`, via `MjSpec`): a forward-facing **`head`
camera** attached to the `trunk` body, used for the inspection picture-in-picture.
All terrain, inspection props, lights, and the chase/overview cameras are added
procedurally and are original to this submission.

## Software

- **MuJoCo** (Apache-2.0), **NumPy** (BSD), **imageio** + **imageio-ffmpeg** (BSD /
  bundled FFmpeg under LGPL/GPL), **Pillow** (MIT-CMU). Installed via
  [`requirements.txt`](requirements.txt).

All original code in `sentinel/`, `run_demo.py`, and `validate.py` is authored for
this submission.
